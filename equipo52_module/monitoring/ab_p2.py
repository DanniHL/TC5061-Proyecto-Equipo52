"""P2 · A/B testing offline: champion (A) contra challenger (B) en octubre de 2026.

Las dos versiones se cargan del Model Registry por su alias y predicen las mismas horas de
octubre, que ninguna vio al entrenar. Se compara el F1 de fallas con comparar_versiones()
(bootstrap pareado) y se separa el resultado entre las estaciones que el champion conoció y las
nuevas. El resultado se registra en MLflow y en reports/monitoreo/ab_p2.json. No cambia los
alias: promover al challenger es una decisión aparte.

Uso (con el servidor de MLflow encendido):
    python -m equipo52_module.monitoring.ab_p2
"""

import json
import os

from loguru import logger

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")  # oculta un aviso de MLflow que no aplica aquí
import mlflow  # noqa: E402
from mlflow import MlflowClient  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
import typer  # noqa: E402

from equipo52_module.config import MLFLOW_EXPERIMENT, MLFLOW_TRACKING_URI  # noqa: E402
from equipo52_module.features import SIN_ANCLAJES, SIN_BICICLETAS  # noqa: E402
from equipo52_module.modeling.registrar_p2 import NOMBRE_MODELO  # noqa: E402
from equipo52_module.modeling.train_p2 import (  # noqa: E402
    VARIABLES,
    calcular_metricas,
    etiquetas,
    preparar_datos,
)
from equipo52_module.monitoring.ab_test import comparar_versiones  # noqa: E402
from equipo52_module.monitoring.drift import MONITOREO_DIR  # noqa: E402
from equipo52_module.monitoring.drift_p2 import tabla_actual  # noqa: E402

app = typer.Typer()


def f1_fallas(real, prediccion) -> float:
    """F1 promedio de las dos clases de falla: la métrica principal de P2 desde la semana 5."""
    return f1_score(real, prediccion, labels=[SIN_BICICLETAS, SIN_ANCLAJES], average="macro")


def predecir(alias: str, datos: pd.DataFrame):
    """Carga del Model Registry la versión con ese alias y predice; regresa la versión y las predicciones."""
    version = MlflowClient().get_model_version_by_alias(NOMBRE_MODELO, alias)
    modelo = mlflow.pyfunc.load_model(f"models:/{NOMBRE_MODELO}@{alias}")
    return version, np.asarray(modelo.predict(datos[VARIABLES])).astype(int)


def estaciones_del_champion() -> set:
    """Estaciones con las que se entrenó el champion (las 371 de su entrenamiento final)."""
    _, partes = preparar_datos("completo")
    return set(partes["entrenamiento_final"]["estacion_id"].unique())


@app.command()
def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    actual = tabla_actual()
    real = actual["clase"].astype(int).to_numpy()
    version_a, pred_a = predecir("champion", actual)
    version_b, pred_b = predecir("challenger", actual)
    logger.info(f"A = versión {version_a.version} · B = versión {version_b.version} · "
                f"{len(actual):,} horas de octubre")

    resultado = comparar_versiones(real, pred_a, pred_b, f1_fallas, mayor_es_mejor=True)

    es_conocida = actual["estacion_id"].isin(estaciones_del_champion()).to_numpy()
    por_grupo = {}
    for grupo, filtro in {"conocidas": es_conocida, "nuevas": ~es_conocida}.items():
        por_grupo[grupo] = {
            "estaciones": int(actual.loc[filtro, "estacion_id"].nunique()), "horas": int(filtro.sum()),
            "f1_a": f1_fallas(real[filtro], pred_a[filtro]), "f1_b": f1_fallas(real[filtro], pred_b[filtro]),
        }
        g = por_grupo[grupo]
        logger.info(f"Estaciones {grupo}: {g['estaciones']} · {g['horas']:,} horas · "
                    f"F1 A = {g['f1_a']:.4f} · F1 B = {g['f1_b']:.4f}")

    salida = {
        "modelo": NOMBRE_MODELO, "periodo_evaluacion": "2026-10",
        "version_a": version_a.version, "version_b": version_b.version,
        "metrica": "f1_fallas", "resultado": resultado, "por_grupo": por_grupo,
        "detalle": {"A": calcular_metricas(real, pred_a, "oct"), "B": calcular_metricas(real, pred_b, "oct")},
    }
    MONITOREO_DIR.mkdir(parents=True, exist_ok=True)
    ruta = MONITOREO_DIR / "ab_p2.json"
    ruta.write_text(json.dumps(salida, indent=2, ensure_ascii=False))

    with mlflow.start_run(run_name="p2-ab-octubre-2026"):
        mlflow.set_tags(etiquetas("hgb", "ab_test") | {
            "version_a": version_a.version, "version_b": version_b.version, "gana": resultado["gana"]})
        mlflow.log_metrics({
            "f1_fallas_a": resultado["metrica_a"], "f1_fallas_b": resultado["metrica_b"],
            "diferencia_b_menos_a": resultado["diferencia"],
            "ic95_inferior": resultado["ic_95"][0], "ic95_superior": resultado["ic_95"][1],
        } | {f"{grupo}_f1_{v}": g[f"f1_{v}"] for grupo, g in por_grupo.items() for v in ["a", "b"]})
        mlflow.log_artifact(str(ruta))

    logger.success(f"A/B guardado en {ruta}. Según la regla de decisión se queda {resultado['gana']} "
                   "(los alias no se cambiaron).")


if __name__ == "__main__":
    app()