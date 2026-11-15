"""P3 · Challenger: la receta del champion reentrenada con el régimen post-COVID.

Mismo modelo (K-Means sobre las 42 características), mismo k, n_init y semilla que el
champion; la única diferencia es que la matriz de estaciones se calcula solo con los
viajes desde el fin de la emergencia sanitaria (INICIO_POSTCOVID) hasta el 30 de
septiembre de 2026: el régimen actual de la red, sin octubre. Registra la corrida en
MLflow, crea una nueva versión de mibici-equipo52-p3-grupos y le asigna el alias
challenger (la versión que lo tenía sigue en el registry, sin alias).

Uso (con el servidor de MLflow encendido):
    python -m equipo52_module.monitoring.challenger_p3
"""

import json
import os
import tempfile
import time

from loguru import logger

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")  # oculta un aviso de MLflow que no aplica aquí
import mlflow  # noqa: E402
from mlflow import MlflowClient  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
import typer  # noqa: E402

from equipo52_module.config import (  # noqa: E402
    FECHA_FIN,
    INICIO_POSTCOVID,
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
    SEED,
    VERSION_DATOS,
)
from equipo52_module.modeling.registrar_p3 import NOMBRE_MODELO  # noqa: E402
from equipo52_module.modeling.train_p3 import (  # noqa: E402
    MESES_ESTACION_CONSOLIDADA,
    TIPOS_CONFIABLES,
    etiquetas_corrida,
    metricas_internas,
)
from equipo52_module.monitoring.drift_p3 import (  # noqa: E402
    columnas_modelo,
    matriz_postcovid,
)

app = typer.Typer()

NOMBRE_CORRIDA = "p3-kmeans-postcovid-hasta-2026-09"


def receta_del_champion(cliente: MlflowClient):
    """Versión champion y sus hiperparámetros, leídos de su corrida en MLflow."""
    version = cliente.get_model_version_by_alias(NOMBRE_MODELO, "champion")
    parametros = cliente.get_run(version.run_id).data.params
    receta = {
        "k": int(parametros["k_elegido"]),
        "n_init": int(parametros["kmeans_n_init"]),
    }
    return version, receta


@app.command()
def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    cliente = MlflowClient()

    champion, receta = receta_del_champion(cliente)
    logger.info(f"Champion: versión {champion.version} · corrida {champion.run_id} · receta {receta}")

    inicio = time.perf_counter()
    matriz = matriz_postcovid()
    columnas = columnas_modelo(matriz)
    consolidadas = matriz["meses_activa"] > MESES_ESTACION_CONSOLIDADA
    logger.info(
        f"Entrenamiento del challenger: régimen post-COVID ({INICIO_POSTCOVID} a {FECHA_FIN}) · "
        f"{len(matriz):,} estaciones × {len(columnas)} características "
        f"({int(consolidadas.sum())} con más de 6 meses en el periodo)"
    )

    escalador = StandardScaler().fit(matriz[columnas])
    X = pd.DataFrame(escalador.transform(matriz[columnas]), columns=columnas, index=matriz.index)
    modelo = KMeans(n_clusters=receta["k"], n_init=receta["n_init"], random_state=SEED).fit(X)
    segundos = time.perf_counter() - inicio
    metricas = metricas_internas(X, modelo.labels_, prefijo="hist") | metricas_internas(
        X.loc[consolidadas], modelo.labels_[consolidadas.to_numpy()], prefijo="val"
    )
    metricas["segundos_entrenamiento"] = round(segundos, 1)
    logger.info(f"Entrenado en {segundos:.0f} s · silueta hist {metricas['hist_silhouette']:.4f}")

    with mlflow.start_run(run_name=NOMBRE_CORRIDA) as corrida:
        mlflow.set_tags(etiquetas_corrida("kmeans", "challenger")
                        | {"champion_de_referencia": champion.run_id})
        mlflow.log_params({
            "modelo": "kmeans", "k": receta["k"], "k_elegido": receta["k"],
            "k_elegido_por": "receta_del_champion", "kmeans_n_init": receta["n_init"],
            "caracteristicas_modelo": "grupos A-F (42 variables)",
            "periodo": "postcovid", "fecha_inicio_entrenamiento": INICIO_POSTCOVID,
            "fecha_fin_entrenamiento": FECHA_FIN, "version_datos": VERSION_DATOS,
            "n_estaciones": len(matriz), "n_caracteristicas": len(columnas),
            "semilla": SEED, "variables": ", ".join(columnas),
        })
        mlflow.log_metrics(metricas)
        ejemplo = X.head(50)
        with tempfile.TemporaryDirectory() as temporal:
            ruta_escalador = os.path.join(temporal, "escalador.json")
            with open(ruta_escalador, "w") as archivo:
                json.dump({
                    "descripcion": (f"Media y desviación estándar (poblacional) con las que la corrida "
                                    f"{NOMBRE_CORRIDA} estandarizó las {len(columnas)} características; "
                                    "se aplica antes de predecir."),
                    "variables": columnas,
                    "media": {c: float(v) for c, v in zip(columnas, escalador.mean_)},
                    "escala": {c: float(v) for c, v in zip(columnas, escalador.scale_)},
                }, archivo, indent=1, ensure_ascii=False)
            mlflow.log_artifact(ruta_escalador)
        info = mlflow.sklearn.log_model(
            modelo, name="modelo", signature=infer_signature(ejemplo, modelo.predict(ejemplo)),
            input_example=ejemplo.head(5), skops_trusted_types=TIPOS_CONFIABLES,
            registered_model_name=NOMBRE_MODELO,
        )

    version = info.registered_model_version
    for clave, valor in {"rol_al_registrar": "challenger", "corrida_origen": corrida.info.run_id,
                         "nombre_corrida": NOMBRE_CORRIDA,
                         "periodo": f"{INICIO_POSTCOVID} a {FECHA_FIN}",
                         "version_datos": VERSION_DATOS,
                         "champion_de_referencia": champion.run_id}.items():
        cliente.set_model_version_tag(NOMBRE_MODELO, version, clave, str(valor))
    cliente.update_model_version(
        NOMBRE_MODELO, version,
        description=(f"**Challenger de P3 (semana 9):** la receta del champion (versión {champion.version}, "
                     f"K-Means k = {receta['k']} sobre las 42 características) reentrenada solo con el "
                     f"régimen post-COVID ({INICIO_POSTCOVID} a {FECHA_FIN}), sin octubre de 2026. "
                     "Se compara con el champion sobre los perfiles de octubre (A/B testing)."),
    )
    cliente.set_registered_model_alias(NOMBRE_MODELO, "challenger", version)
    logger.success(f"Alias challenger → {NOMBRE_MODELO} versión {version} (corrida {corrida.info.run_id})")


if __name__ == "__main__":
    app()
