"""P2 · Challenger: la receta del champion reentrenada con datos hasta septiembre de 2026.

Mismo modelo (HGB), mismos hiperparámetros y mismo peso de fallas que el champion; la única
diferencia es que entrena hasta el 30 de septiembre de 2026, incluidas las estaciones abiertas de
mayo a agosto. Registra la corrida en MLflow, crea una nueva versión de mibici-equipo52-p2-riesgo
y le asigna el alias challenger (la versión que lo tenía sigue en el registry, sin alias).

Uso (con el servidor de MLflow encendido):
    python -m equipo52_module.monitoring.challenger_p2
"""

import os

from loguru import logger

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")  # oculta un aviso de MLflow que no aplica aquí
import mlflow  # noqa: E402
from mlflow import MlflowClient  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402
import pandas as pd  # noqa: E402
import typer  # noqa: E402

from equipo52_module.config import (  # noqa: E402
    FECHA_FIN,
    FECHA_INICIO,
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
    SEED,
    VERSION_DATOS,
)
from equipo52_module.modeling.registrar_p2 import NOMBRE_MODELO  # noqa: E402
from equipo52_module.modeling.train_p2 import (  # noqa: E402
    HIPERPARAMETROS,
    TIPOS_CONFIABLES,
    VARIABLES,
    entre,
    entrenar,
    etiquetas,
    pesos_por_clase,
    preparar_datos,
)

app = typer.Typer()

FIN_SIGUIENTE = str((pd.Timestamp(FECHA_FIN) + pd.Timedelta(days=1)).date())  # 2026-10-01
NOMBRE_CORRIDA = "p2-hgb-completo-peso_fallas3-hasta-2026-09"


def receta_del_champion(cliente: MlflowClient):
    """Versión champion y sus hiperparámetros, leídos de su corrida en MLflow."""
    version = cliente.get_model_version_by_alias(NOMBRE_MODELO, "champion")
    parametros = cliente.get_run(version.run_id).data.params
    receta = {}
    for nombre, omision in HIPERPARAMETROS["hgb"].items():
        clave = nombre if nombre == "peso_fallas" else f"hgb_{nombre}"
        valor = parametros.get(f"{clave}_script", parametros[clave])  # registrar_parametros de train_p2
        receta[nombre] = type(omision)(float(valor))
    return version, receta


def datos_hasta_septiembre() -> pd.DataFrame:
    """Horas con clase desde FECHA_INICIO hasta el 30 de septiembre de 2026, todas las estaciones."""
    tabla, _ = preparar_datos("completo")
    return entre(tabla[tabla["clase"].notna()], FECHA_INICIO, FIN_SIGUIENTE)


@app.command()
def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    cliente = MlflowClient()

    champion, receta = receta_del_champion(cliente)
    logger.info(f"Champion: versión {champion.version} · corrida {champion.run_id} · receta {receta}")

    datos = datos_hasta_septiembre()
    logger.info(
        f"Entrenamiento del challenger: {datos['hora'].min():%Y-%m-%d} a {datos['hora'].max():%Y-%m-%d} · "
        f"{len(datos):,} horas · {datos['estacion_id'].nunique()} estaciones"
    )
    pesos = pesos_por_clase(receta["peso_fallas"], False)
    clasificador, segundos = entrenar("hgb", receta, pesos, datos)
    logger.info(f"Entrenado en {segundos:.0f} s")

    with mlflow.start_run(run_name=NOMBRE_CORRIDA) as corrida:
        mlflow.set_tags(etiquetas("hgb", "challenger") | {"champion_de_referencia": champion.run_id})
        mlflow.log_params(
            {f"hgb_{n}": v for n, v in receta.items() if n != "peso_fallas"}
            | {"peso_fallas": receta["peso_fallas"], "periodo": "completo",
               "fecha_inicio_entrenamiento": FECHA_INICIO, "fecha_fin_entrenamiento": FECHA_FIN,
               "version_datos": VERSION_DATOS, "filas_entrenamiento": len(datos),
               "estaciones_entrenamiento": datos["estacion_id"].nunique(), "semilla": SEED,
               "variables": ", ".join(VARIABLES)}
        )
        mlflow.log_metric("segundos_entrenamiento", round(segundos, 1))
        ejemplo = datos[VARIABLES].head(1000)
        info = mlflow.sklearn.log_model(
            clasificador, name="modelo", signature=infer_signature(ejemplo, clasificador.predict(ejemplo)),
            input_example=ejemplo.head(5), skops_trusted_types=TIPOS_CONFIABLES["hgb"],
            registered_model_name=NOMBRE_MODELO,
        )

    version = info.registered_model_version
    for clave, valor in {"rol_al_registrar": "challenger", "corrida_origen": corrida.info.run_id,
                         "nombre_corrida": NOMBRE_CORRIDA, "periodo": f"{FECHA_INICIO} a {FECHA_FIN}",
                         "version_datos": VERSION_DATOS, "champion_de_referencia": champion.run_id}.items():
        cliente.set_model_version_tag(NOMBRE_MODELO, version, clave, str(valor))
    cliente.update_model_version(
        NOMBRE_MODELO, version,
        description=(f"**Challenger de P2 (semana 9):** la receta del champion (versión {champion.version}) "
                     f"reentrenada hasta {FECHA_FIN}, incluidas las estaciones abiertas de mayo a agosto de 2026. "
                     "Se compara con el champion en octubre de 2026 (A/B testing)."),
    )
    cliente.set_registered_model_alias(NOMBRE_MODELO, "challenger", version)
    logger.success(f"Alias challenger → {NOMBRE_MODELO} versión {version} (corrida {corrida.info.run_id})")


if __name__ == "__main__":
    app()