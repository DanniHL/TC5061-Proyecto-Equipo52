
"""Evaluación A/B del champion y challenger del modelo P1."""

import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from sklearn.metrics import mean_absolute_error, mean_squared_error

from equipo52_module.config import MLFLOW_TRACKING_URI
from equipo52_module.modeling.train_p1 import VARIABLES_CICLICAS
from equipo52_module.monitoring.ab_test import comparar_versiones


NOMBRE_MODELO = "mibici-equipo52-p1-demanda"
RUTA_DATOS = Path("data/processed/monitoring/p1_actual.parquet")
RUTA_REPORTE = Path("reports/monitoreo/ab_p1.json")


def evaluar_modelos_p1():
    """Compara champion y challenger con los datos de octubre de 2026."""

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    cliente = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    version_a = cliente.get_model_version_by_alias(
        NOMBRE_MODELO, "champion"
    ).version

    version_b = cliente.get_model_version_by_alias(
        NOMBRE_MODELO, "challenger"
    ).version

    print("\n=== VERSIONES A EVALUAR ===")
    print(f"Champion (A): versión {version_a}")
    print(f"Challenger (B): versión {version_b}")

    datos = pd.read_parquet(RUTA_DATOS)

    columnas = VARIABLES_CICLICAS + ["Numero_viajes"]

    faltantes = [col for col in columnas if col not in datos.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas en los datos: {faltantes}")

    if datos[columnas].isna().any().any():
        raise ValueError(
            "Hay valores faltantes en las variables de evaluación."
        )

    X = datos[VARIABLES_CICLICAS].copy()
    y_real = datos["Numero_viajes"].to_numpy()

    print("\n=== CONJUNTO DE EVALUACIÓN ===")
    print(f"Registros: {len(datos):,}")
    print(f"Periodo: {datos['Fecha'].min()} a {datos['Fecha'].max()}")

    uri_a = f"models:/{NOMBRE_MODELO}@champion"
    uri_b = f"models:/{NOMBRE_MODELO}@challenger"

    modelo_a = mlflow.pyfunc.load_model(uri_a)
    modelo_b = mlflow.pyfunc.load_model(uri_b)

    pred_a = modelo_a.predict(X)
    pred_b = modelo_b.predict(X)

    print("\n=== COMPARACIÓN MAE ===")
    resultado_mae = comparar_versiones(
        y_real,
        pred_a,
        pred_b,
        metrica=mean_absolute_error,
        mayor_es_mejor=False,
    )

    print("\n=== COMPARACIÓN RMSE ===")
    resultado_rmse = comparar_versiones(
        y_real,
        pred_a,
        pred_b,
        metrica=lambda real, pred: np.sqrt(
            mean_squared_error(real, pred)
        ),
        mayor_es_mejor=False,
    )

    resultado = {
        "modelo": NOMBRE_MODELO,
        "champion_version": version_a,
        "challenger_version": version_b,
        "periodo_evaluacion": {
            "inicio": str(datos["Fecha"].min()),
            "fin": str(datos["Fecha"].max()),
        },
        "registros_evaluacion": len(datos),
        "mae": resultado_mae,
        "rmse": resultado_rmse,
        "criterio": (
            "El challenger gana únicamente si el IC 95% de B-A "
            "queda completamente por debajo de cero."
        ),
    }

    RUTA_REPORTE.parent.mkdir(parents=True, exist_ok=True)

    with RUTA_REPORTE.open("w", encoding="utf-8") as archivo:
        json.dump(resultado, archivo, indent=4, ensure_ascii=False)

    print("\n=== RESULTADO A/B P1 ===")
    print(f"Reporte guardado en: {RUTA_REPORTE}")
    print(f"Decisión MAE: se queda {resultado_mae['gana']}")
    print(f"Decisión RMSE: se queda {resultado_rmse['gana']}")

    return resultado


if __name__ == "__main__":
    evaluar_modelos_p1()