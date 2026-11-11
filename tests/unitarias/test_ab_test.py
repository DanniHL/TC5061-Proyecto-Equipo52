"""Pruebas del A/B testing común (equipo52_module/monitoring/ab_test.py)."""

import numpy as np
import pytest
from sklearn.metrics import accuracy_score, mean_absolute_error

from equipo52_module.monitoring.ab_test import comparar_versiones


def test_se_queda_a_si_las_versiones_predicen_igual():
    rng = np.random.default_rng(42)
    y_real = rng.integers(0, 2, 500)
    pred = rng.integers(0, 2, 500)

    resultado = comparar_versiones(y_real, pred, pred, accuracy_score)

    assert resultado["diferencia"] == 0
    assert resultado["gana"] == "A"


def test_gana_b_si_es_claramente_mejor():
    rng = np.random.default_rng(42)
    y_real = rng.integers(0, 2, 500)
    pred_a = rng.integers(0, 2, 500)

    resultado = comparar_versiones(y_real, pred_a, y_real, accuracy_score)

    assert resultado["gana"] == "B"
    assert resultado["ic_95"][0] > 0


def test_respeta_metricas_donde_menor_es_mejor():
    rng = np.random.default_rng(42)
    y_real = rng.normal(10, 2, 500)
    pred_a = y_real + rng.normal(0, 3, 500)
    pred_b = y_real + rng.normal(0, 0.5, 500)

    resultado = comparar_versiones(
        y_real, pred_a, pred_b, mean_absolute_error, mayor_es_mejor=False
    )

    assert resultado["gana"] == "B"
    assert resultado["diferencia"] < 0


def test_falla_si_los_tamanos_no_coinciden():
    with pytest.raises(ValueError, match="mismo número de filas"):
        comparar_versiones([1, 0, 1], [1, 0], [1, 0, 1], accuracy_score)