"""A/B testing offline entre dos versiones de un modelo (código común del Equipo 52).

Compara la versión A (champion) contra la B (challenger) sobre el mismo conjunto
de evaluación y estima la diferencia B - A con un intervalo de confianza del 95%
por bootstrap pareado. La métrica la elige cada pregunta.
"""

from collections.abc import Callable

from loguru import logger
import numpy as np

from equipo52_module.config import SEED

N_REMUESTRAS = 1_000


def comparar_versiones(
    y_real,
    pred_a,
    pred_b,
    metrica: Callable,
    mayor_es_mejor: bool = True,
    n_remuestras: int = N_REMUESTRAS,
    semilla: int = SEED,
) -> dict:
    """Compara A contra B con bootstrap pareado y aplica la regla de decisión.

    B gana solo si todo el intervalo del 95% de la diferencia B - A le favorece;
    si el intervalo incluye el 0, se queda A.
    """
    y_real, pred_a, pred_b = np.asarray(y_real), np.asarray(pred_a), np.asarray(pred_b)
    if not len(y_real) == len(pred_a) == len(pred_b):
        raise ValueError("y_real, pred_a y pred_b deben tener el mismo número de filas")

    rng = np.random.default_rng(semilla)
    n = len(y_real)
    diferencias = np.empty(n_remuestras)
    for i in range(n_remuestras):
        filas = rng.integers(0, n, n)  # las mismas filas para A y para B
        valor_b = metrica(y_real[filas], pred_b[filas])
        valor_a = metrica(y_real[filas], pred_a[filas])
        diferencias[i] = valor_b - valor_a

    metrica_a = float(metrica(y_real, pred_a))
    metrica_b = float(metrica(y_real, pred_b))
    ic_inferior, ic_superior = (float(v) for v in np.percentile(diferencias, [2.5, 97.5]))
    if mayor_es_mejor:
        gana = "B" if ic_inferior > 0 else "A"
    else:
        gana = "B" if ic_superior < 0 else "A"

    logger.info(
        f"A = {metrica_a:.4f} | B = {metrica_b:.4f} | B - A = {metrica_b - metrica_a:+.4f} "
        f"(IC 95%: {ic_inferior:+.4f} a {ic_superior:+.4f}) -> se queda {gana}"
    )
    return {
        "metrica_a": metrica_a,
        "metrica_b": metrica_b,
        "diferencia": metrica_b - metrica_a,
        "ic_95": (ic_inferior, ic_superior),
        "gana": gana,
    }