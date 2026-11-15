"""P3 · A/B testing offline: champion (A) contra challenger (B) en octubre de 2026.

Las dos versiones se cargan del Model Registry por su alias. Como el K-Means registrado
espera las características estandarizadas, cada una predice con su propio escalador (el
del champion es el escalador de la matriz de referencia; el del challenger, el de la
matriz post-COVID: ambos se reconstruyen con las mismas funciones de features.py). El
conjunto de evaluación es la matriz de estaciones de octubre de 2026, que ninguna vio al
entrenar.

No hay etiqueta real: la métrica es la silueta de la partición que cada versión induce
sobre las estaciones de octubre, medida en un espacio común (las características de
octubre estandarizadas con sus propias medias y desviaciones). Se compara con
comparar_versiones() (bootstrap pareado sobre estaciones) y se separa el resultado entre
las estaciones que el champion conoció y las nuevas. El resultado se registra en MLflow
y en reports/monitoreo/ab_p3.json. No cambia los alias: promover al challenger es una
decisión aparte.

Uso (con el servidor de MLflow encendido):
    python -m equipo52_module.monitoring.ab_p3
"""

import json
import os

from loguru import logger

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")  # oculta un aviso de MLflow que no aplica aquí
import mlflow  # noqa: E402
from mlflow import MlflowClient  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import silhouette_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
import typer  # noqa: E402

from equipo52_module.config import MLFLOW_EXPERIMENT, MLFLOW_TRACKING_URI  # noqa: E402
from equipo52_module.modeling.registrar_p3 import NOMBRE_MODELO  # noqa: E402
from equipo52_module.modeling.train_p3 import etiquetas_corrida  # noqa: E402
from equipo52_module.monitoring.ab_test import comparar_versiones  # noqa: E402
from equipo52_module.monitoring.drift import MONITOREO_DIR  # noqa: E402
from equipo52_module.monitoring.drift_p3 import (  # noqa: E402
    columnas_modelo,
    matriz_actual,
    matriz_postcovid,
    matriz_referencia,
)

app = typer.Typer()


def predecir(alias: str, datos: pd.DataFrame):
    """Carga del Model Registry la versión con ese alias y asigna grupo a cada estación.

    Regresa la versión, las etiquetas y el escalador con el que esa versión fue entrenada.
    """
    version = MlflowClient().get_model_version_by_alias(NOMBRE_MODELO, alias)
    modelo = mlflow.pyfunc.load_model(f"models:/{NOMBRE_MODELO}@{alias}")
    return version, np.asarray(modelo.predict(datos)).astype(int)


@app.command()
def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    actual = matriz_actual()
    columnas = columnas_modelo(actual)

    # Cada versión predice en su propio espacio estandarizado, como en producción.
    escalador_a = StandardScaler().fit(matriz_referencia()[columnas])
    escalador_b = StandardScaler().fit(matriz_postcovid()[columnas])
    version_a, pred_a = predecir("champion", pd.DataFrame(
        escalador_a.transform(actual[columnas]), columns=columnas, index=actual.index))
    version_b, pred_b = predecir("challenger", pd.DataFrame(
        escalador_b.transform(actual[columnas]), columns=columnas, index=actual.index))
    logger.info(f"A = versión {version_a.version} · B = versión {version_b.version} · "
                f"{len(actual):,} estaciones de octubre")

    # La silueta de ambas particiones se mide en el mismo espacio de octubre.
    X_octubre = StandardScaler().fit_transform(actual[columnas])
    resultado = comparar_versiones(X_octubre, pred_a, pred_b, silhouette_score, mayor_es_mejor=True)

    conocidas = actual.index.isin(matriz_referencia().index)
    por_grupo = {}
    for grupo, filtro in {"conocidas": conocidas, "nuevas": ~conocidas}.items():
        if filtro.sum() < 2:
            continue
        por_grupo[grupo] = {
            "estaciones": int(filtro.sum()),
            "silueta_a": float(silhouette_score(X_octubre[filtro], pred_a[filtro])),
            "silueta_b": float(silhouette_score(X_octubre[filtro], pred_b[filtro])),
        }
        g = por_grupo[grupo]
        logger.info(f"Estaciones {grupo}: {g['estaciones']} · "
                    f"silueta A = {g['silueta_a']:.4f} · silueta B = {g['silueta_b']:.4f}")

    salida = {
        "modelo": NOMBRE_MODELO, "periodo_evaluacion": "2026-10",
        "version_a": version_a.version, "version_b": version_b.version,
        "metrica": "silueta", "resultado": resultado, "por_grupo": por_grupo,
        "detalle": {
            "estaciones_octubre": int(len(actual)),
            "estaciones_conocidas": int(conocidas.sum()),
            "estaciones_nuevas": int((~conocidas).sum()),
            "grupos_a": sorted(int(g) for g in np.unique(pred_a)),
            "grupos_b": sorted(int(g) for g in np.unique(pred_b)),
            "estaciones_que_cambian_de_grupo": int((pred_a != pred_b).sum()),
        },
    }
    MONITOREO_DIR.mkdir(parents=True, exist_ok=True)
    ruta = MONITOREO_DIR / "ab_p3.json"
    ruta.write_text(json.dumps(salida, indent=2, ensure_ascii=False))

    with mlflow.start_run(run_name="p3-ab-octubre-2026"):
        mlflow.set_tags(etiquetas_corrida("kmeans", "ab_test") | {
            "version_a": version_a.version, "version_b": version_b.version,
            "gana": resultado["gana"]})
        mlflow.log_metrics({
            "silueta_a": resultado["metrica_a"], "silueta_b": resultado["metrica_b"],
            "diferencia_b_menos_a": resultado["diferencia"],
            "ic95_inferior": resultado["ic_95"][0], "ic95_superior": resultado["ic_95"][1],
        } | {f"{grupo}_silueta_{v}": g[f"silueta_{v}"] for grupo, g in por_grupo.items()
             for v in ["a", "b"]})
        mlflow.log_artifact(str(ruta))

    logger.success(f"A/B guardado en {ruta}. Según la regla de decisión se queda {resultado['gana']} "
                   "(los alias no se cambiaron).")


if __name__ == "__main__":
    app()
