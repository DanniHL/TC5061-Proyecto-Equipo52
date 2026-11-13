"""Monitoreo de drift con Evidently (código común del Equipo 52).

Compara un conjunto de referencia (los datos con los que se entrenó el modelo)
contra un conjunto actual (datos posteriores, por ejemplo octubre de 2026)
y guarda el reporte de Evidently en HTML y JSON en reports/monitoreo/.
Lo usan P1, P2 y P3: cada pregunta arma sus propios DataFrames.
"""

import json
from pathlib import Path
import re

from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset, DataSummaryPreset
from loguru import logger
import pandas as pd

from equipo52_module.config import REPORTS_DIR, SEED

MONITOREO_DIR = REPORTS_DIR / "monitoreo"
MAX_FILAS = 200_000  # arriba de este tamaño se toma una muestra reproducible

# Nombres que Evidently 0.7 da a sus métricas, por ejemplo:
#   ValueDrift(column=x,method=K-S p_value,threshold=0.05)
#   DriftedColumnsCount(drift_share=0.5)
PATRON_COLUMNA = re.compile(r"ValueDrift\(column=(.+?),method=(.+?),threshold=([\d.]+)\)")
PATRON_CONJUNTO = re.compile(r"DriftedColumnsCount\(drift_share=([\d.]+)\)")


def muestrear(datos: pd.DataFrame, max_filas: int = MAX_FILAS, semilla: int = SEED) -> pd.DataFrame:
    """Regresa a lo más max_filas filas, elegidas al azar con la semilla del proyecto."""
    if len(datos) <= max_filas:
        return datos
    logger.info(f"Muestreando {max_filas:,} de {len(datos):,} filas (semilla {semilla})")
    return datos.sample(n=max_filas, random_state=semilla)


def resumir_drift(metricas: list[dict]) -> dict:
    """Extrae del JSON de Evidently el drift por columna y el del conjunto completo."""
    filas = []
    resumen = {}
    for metrica in metricas:
        nombre, valor = metrica["metric_name"], metrica["value"]
        if coincide := PATRON_COLUMNA.fullmatch(nombre):
            columna, metodo, umbral = coincide[1], coincide[2], float(coincide[3])
            # Con un p-value hay drift si es menor al umbral; con una distancia, si lo alcanza.
            drift = valor < umbral if "p_value" in metodo else valor >= umbral
            filas.append(
                {"columna": columna, "metodo": metodo, "umbral": umbral, "valor": valor, "drift": drift}
            )
        elif coincide := PATRON_CONJUNTO.fullmatch(nombre):
            resumen = {
                "columnas_con_drift": int(valor["count"]),
                "proporcion_con_drift": valor["share"],
                "dataset_drift": valor["share"] >= float(coincide[1]),
            }
    por_columna = pd.DataFrame(filas).sort_values(["drift", "columna"], ascending=[False, True])
    return {**resumen, "columnas_analizadas": len(por_columna), "por_columna": por_columna}


def generar_reporte_drift(
    referencia: pd.DataFrame,
    actual: pd.DataFrame,
    columnas_numericas: list[str],
    columnas_categoricas: list[str],
    nombre: str,
    carpeta: Path = MONITOREO_DIR,
    max_filas: int = MAX_FILAS,
) -> dict:
    """Compara referencia contra actual con Evidently y guarda HTML, JSON y CSV por columna.

    Regresa el resumen: columnas con drift, proporción, dataset_drift y la tabla por columna.
    """
    columnas = columnas_numericas + columnas_categoricas
    faltantes = [c for c in columnas if c not in referencia.columns or c not in actual.columns]
    if faltantes:
        raise ValueError(f"Columnas que no están en ambos conjuntos: {faltantes}")

    definicion = DataDefinition(
        numerical_columns=columnas_numericas,
        categorical_columns=columnas_categoricas,
    )
      # Evidently no acepta enteros de numpy (int8, int64) como etiquetas de una categórica:
    # se convierten a texto, que para una categoría es lo mismo.
    tipos = {columna: str for columna in columnas_categoricas}
    ref = Dataset.from_pandas(
        muestrear(referencia[columnas], max_filas).astype(tipos), data_definition=definicion
    )
    act = Dataset.from_pandas(
        muestrear(actual[columnas], max_filas).astype(tipos), data_definition=definicion
    )
    
    reporte = Report([DataDriftPreset(), DataSummaryPreset()])
    resultado = reporte.run(act, ref)  # Evidently recibe primero el actual y luego la referencia

    carpeta.mkdir(parents=True, exist_ok=True)
    ruta_json = carpeta / f"drift_{nombre}.json"
    resultado.save_html(str(carpeta / f"drift_{nombre}.html"))
    resultado.save_json(str(ruta_json))

    resumen = resumir_drift(json.loads(ruta_json.read_text())["metrics"])
    resumen["por_columna"].to_csv(carpeta / f"drift_{nombre}_por_columna.csv", index=False)
    logger.success(
        f"{nombre}: {resumen['columnas_con_drift']} de {resumen['columnas_analizadas']} "
        f"columnas con drift (dataset drift: {resumen['dataset_drift']}). Reporte en {carpeta}"
    )
    return resumen