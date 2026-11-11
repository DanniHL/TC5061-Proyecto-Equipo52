"""Pruebas del monitoreo de drift común (equipo52_module/monitoring/drift.py)."""

import numpy as np
import pandas as pd
import pytest

from equipo52_module.monitoring.drift import generar_reporte_drift, muestrear


def crear_datos(media_x: float, prob_a: float, semilla: int, n: int = 1000) -> pd.DataFrame:
    """Tabla con dos numéricas (x, y) y una categórica (c) de distribución conocida."""
    rng = np.random.default_rng(semilla)
    return pd.DataFrame(
        {
            "x": rng.normal(media_x, 1, n),
            "y": rng.normal(5, 1, n),
            "c": rng.choice(["a", "b"], n, p=[prob_a, 1 - prob_a]),
        }
    )


def test_detecta_drift_en_las_columnas_que_cambian(tmp_path):
    referencia = crear_datos(media_x=0, prob_a=0.5, semilla=42)
    actual = crear_datos(media_x=1, prob_a=0.8, semilla=7)

    resumen = generar_reporte_drift(
        referencia, actual, ["x", "y"], ["c"], "prueba", carpeta=tmp_path
    )

    drift = resumen["por_columna"].set_index("columna")["drift"]
    assert drift["x"] and drift["c"]
    assert not drift["y"]
    assert resumen["dataset_drift"]
    assert (tmp_path / "drift_prueba.html").exists()
    assert (tmp_path / "drift_prueba_por_columna.csv").exists()


def test_no_declara_dataset_drift_con_la_misma_distribucion(tmp_path):
    referencia = crear_datos(media_x=0, prob_a=0.5, semilla=42)
    actual = crear_datos(media_x=0, prob_a=0.5, semilla=7)

    resumen = generar_reporte_drift(
        referencia, actual, ["x", "y"], ["c"], "prueba", carpeta=tmp_path
    )

    assert not resumen["dataset_drift"]


def test_falla_si_falta_una_columna(tmp_path):
    datos = crear_datos(media_x=0, prob_a=0.5, semilla=42)
    with pytest.raises(ValueError, match="no_existe"):
        generar_reporte_drift(datos, datos, ["x", "no_existe"], [], "prueba", carpeta=tmp_path)


def test_muestreo_es_reproducible():
    datos = crear_datos(media_x=0, prob_a=0.5, semilla=42, n=500)
    muestra = muestrear(datos, max_filas=100)
    assert len(muestra) == 100
    assert muestra.equals(muestrear(datos, max_filas=100))