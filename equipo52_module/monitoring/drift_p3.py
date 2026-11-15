"""P3 · Drift entre las estaciones con las que se entrenó el champion y octubre de 2026.

Referencia: la matriz estación × característica con la que se entrenó el champion
(p3-kmeans-k4), calculada con toda la historia desde FECHA_INICIO hasta el 30 de
septiembre de 2026. Actual: la misma matriz calculada solo con los viajes de octubre
de 2026 y la nomenclatura de ese mes: el perfil de uso que cada estación presentó en
el periodo en producción.

Las tres características de evolución histórica (meses_activa, share_viajes_ultimos_12m
y crecimiento_vs_esperado) se excluyen del reporte: sobre un solo mes son constantes por
construcción (meses_activa = 1, share = 1 y crecimiento = 0 para todas las estaciones) y
marcarían drift artificial, igual que "mes" en P2. Se reportan las 39 características
restantes más capacidad, latitud y longitud de la nomenclatura.

Uso: python -m equipo52_module.monitoring.drift_p3
"""

from loguru import logger
import pandas as pd
import typer

from equipo52_module.config import (
    DURACION_MINIMA_SEGUNDOS,
    ESTACIONES_EXCLUIDAS,
    FECHA_FIN,
    INICIO_POSTCOVID,
    NOMENCLATURA_FILE,
    RAW_DATA_DIR,
    VIAJES_FILE,
)
from equipo52_module.dataset import leer_viajes_mes
from equipo52_module.features import construir_matriz_p3, quitar_viajes_cortos
from equipo52_module.modeling.train_p3 import GEO, cargar_estaciones
from equipo52_module.monitoring.drift import MAX_FILAS, generar_reporte_drift

app = typer.Typer()

FIN_REFERENCIA = pd.Timestamp(FECHA_FIN)          # 2026-09-30: corte de la historia del champion
FIN_ACTUAL = pd.Timestamp("2026-10-31")
VIAJES_OCTUBRE = RAW_DATA_DIR / "datos_abiertos_2026_10.csv"
NOMENCLATURA_OCTUBRE = RAW_DATA_DIR / "nomenclatura_2026_10.csv"

# Constantes por construcción sobre un solo mes; se excluyen del reporte (como "mes" en P2).
EVOLUCION = ["meses_activa", "share_viajes_ultimos_12m", "crecimiento_vs_esperado"]

# Las 39 características del modelo que sí cambian en un periodo + las tres de la nomenclatura.
def columnas_modelo(matriz: pd.DataFrame) -> list[str]:
    """Las 42 características con las que se ajustó el escalador y el K-Means."""
    return [c for c in matriz.columns if c not in GEO + ["capacidad"]]


def columnas_reporte(matriz: pd.DataFrame) -> list[str]:
    """Columnas del reporte de drift: el perfil de octubre sin las que degeneran en un mes."""
    return [c for c in columnas_modelo(matriz) if c not in EVOLUCION] + GEO + ["capacidad"]


def matriz(viajes: pd.DataFrame, estaciones: pd.DataFrame, fin: pd.Timestamp) -> pd.DataFrame:
    """Matriz estación × característica con capacidad incluida para el reporte."""
    return construir_matriz_p3(viajes, estaciones, fin).join(estaciones["capacidad"])


def matriz_referencia() -> pd.DataFrame:
    """Estaciones con las que se entrenó el champion: la misma construcción de train_p3.py."""
    viajes = pd.read_parquet(VIAJES_FILE)
    viajes = quitar_viajes_cortos(viajes, DURACION_MINIMA_SEGUNDOS)
    estaciones = cargar_estaciones(NOMENCLATURA_FILE, ESTACIONES_EXCLUIDAS)
    return matriz(viajes, estaciones, FIN_REFERENCIA)


def matriz_postcovid() -> pd.DataFrame:
    """Estaciones descritas solo con el régimen reciente: del fin del COVID a septiembre de 2026."""
    viajes = pd.read_parquet(
        VIAJES_FILE, filters=[("Inicio_del_viaje", ">=", pd.Timestamp(INICIO_POSTCOVID))]
    )
    viajes = quitar_viajes_cortos(viajes, DURACION_MINIMA_SEGUNDOS)
    estaciones = cargar_estaciones(NOMENCLATURA_FILE, ESTACIONES_EXCLUIDAS)
    return matriz(viajes, estaciones, FIN_REFERENCIA)


def matriz_actual() -> pd.DataFrame:
    """Estaciones descritas con sus viajes de octubre de 2026: el periodo en producción."""
    viajes = quitar_viajes_cortos(leer_viajes_mes(VIAJES_OCTUBRE), DURACION_MINIMA_SEGUNDOS)
    estaciones = cargar_estaciones(NOMENCLATURA_OCTUBRE, ESTACIONES_EXCLUIDAS)
    return matriz(viajes, estaciones, FIN_ACTUAL)


def describir(nombre: str, viajes: pd.DataFrame, tabla: pd.DataFrame) -> None:
    """Fechas, viajes y estaciones de un conjunto: lo que va en la tabla de la sección 2."""
    logger.info(
        f"{nombre}: {viajes['Inicio_del_viaje'].min():%Y-%m-%d} a {viajes['Inicio_del_viaje'].max():%Y-%m-%d} · "
        f"{len(viajes):,} viajes · {len(tabla):,} estaciones × {len(columnas_modelo(tabla))} características"
    )


@app.command()
def main(max_filas: int = typer.Option(MAX_FILAS, help="filas máximas por conjunto antes de muestrear")):
    viajes_ref = quitar_viajes_cortos(
        pd.read_parquet(VIAJES_FILE), DURACION_MINIMA_SEGUNDOS
    )
    referencia = matriz_referencia()
    describir("Referencia", viajes_ref, referencia)
    viajes_oct = quitar_viajes_cortos(leer_viajes_mes(VIAJES_OCTUBRE), DURACION_MINIMA_SEGUNDOS)
    actual = matriz_actual()
    describir("Actual", viajes_oct, actual)

    columnas = columnas_reporte(referencia)
    resumen = generar_reporte_drift(
        referencia[columnas], actual[columnas],
        columnas_numericas=columnas, columnas_categoricas=[], nombre="p3", max_filas=max_filas,
    )
    logger.info("\n" + resumen["por_columna"].to_string(index=False))


if __name__ == "__main__":
    app()
