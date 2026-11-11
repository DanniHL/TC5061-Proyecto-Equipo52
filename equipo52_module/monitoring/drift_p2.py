"""P2 · Drift entre los datos de entrenamiento del champion y octubre de 2026.

Referencia: las horas estación × hora con las que se entrenó el champion
(p2-hgb-completo-peso_fallas3), desde FECHA_INICIO hasta el 31 de marzo de 2026.
Actual: octubre de 2026, armado con las mismas funciones de features.py. Se usa también
septiembre para que las variables de horas y semanas anteriores tengan historial.

Uso: python -m equipo52_module.monitoring.drift_p2
"""

from loguru import logger
import pandas as pd
import typer

from equipo52_module.config import (
    DURACION_MINIMA_SEGUNDOS,
    ESTACIONES_EXCLUIDAS,
    RAW_DATA_DIR,
    VIAJES_FILE,
)
from equipo52_module.dataset import leer_viajes_mes
from equipo52_module.features import construir_tabla_p2, quitar_viajes_cortos
from equipo52_module.modeling.train_p2 import (
    HORAS_ANTICIPACION,
    SEMANAS_HISTORIAL,
    UMBRAL_FALLA,
    cargar_estaciones,
    entre,
    preparar_datos,
)
from equipo52_module.monitoring.drift import MAX_FILAS, generar_reporte_drift

app = typer.Typer()

INICIO_HISTORIAL = "2026-09-01"  # septiembre da el historial de las primeras semanas de octubre
INICIO_ACTUAL = "2026-10-01"
FIN_ACTUAL_SIGUIENTE = "2026-11-01"
VIAJES_OCTUBRE = RAW_DATA_DIR / "datos_abiertos_2026_10.csv"
NOMENCLATURA_OCTUBRE = RAW_DATA_DIR / "nomenclatura_2026_10.csv"

# "mes" se excluye: octubre solo tiene mes = 10 y marcaría drift por construcción.
COLUMNAS_NUMERICAS = [
    "capacidad", "latitud", "longitud",
    "salidas_hora_anterior", "llegadas_hora_anterior",
    "salidas_3_horas_anteriores", "llegadas_3_horas_anteriores",
    "semanas_sin_bicicletas", "semanas_sin_anclajes",
]
COLUMNAS_CATEGORICAS = ["hora_del_dia", "dia_semana", "fin_de_semana", "clase"]


def tabla_referencia() -> pd.DataFrame:
    """Horas con las que se entrenó el champion: la misma partición de train_p2.py."""
    _, partes = preparar_datos("completo")
    return partes["entrenamiento_final"]


def tabla_actual() -> pd.DataFrame:
    """Horas de octubre de 2026 con sus variables y su clase, como las vería el modelo en producción."""
    septiembre = pd.read_parquet(
        VIAJES_FILE, filters=[("Inicio_del_viaje", ">=", pd.Timestamp(INICIO_HISTORIAL))]
    )
    viajes = pd.concat([septiembre, leer_viajes_mes(VIAJES_OCTUBRE)], ignore_index=True)
    viajes = quitar_viajes_cortos(viajes, DURACION_MINIMA_SEGUNDOS)
    estaciones = cargar_estaciones(NOMENCLATURA_OCTUBRE, ESTACIONES_EXCLUIDAS)
    tabla = construir_tabla_p2(viajes, estaciones, HORAS_ANTICIPACION, UMBRAL_FALLA, SEMANAS_HISTORIAL)
    tabla = tabla[tabla["clase"].notna()]
    return entre(tabla, INICIO_ACTUAL, FIN_ACTUAL_SIGUIENTE)


def preparar_columnas(tabla: pd.DataFrame) -> pd.DataFrame:
    """Columnas del reporte; la clase pasa de entero con nulos (Int8) a entero normal."""
    return tabla[COLUMNAS_NUMERICAS + COLUMNAS_CATEGORICAS].astype({"clase": int})


def describir(nombre: str, tabla: pd.DataFrame) -> None:
    """Fechas, filas y estaciones de un conjunto: lo que va en la tabla de la sección 2."""
    logger.info(
        f"{nombre}: {tabla['hora'].min():%Y-%m-%d} a {tabla['hora'].max():%Y-%m-%d} · "
        f"{len(tabla):,} horas · {tabla['estacion_id'].nunique()} estaciones"
    )


@app.command()
def main(max_filas: int = typer.Option(MAX_FILAS, help="filas máximas por conjunto antes de muestrear")):
    referencia = tabla_referencia()
    describir("Referencia", referencia)
    actual = tabla_actual()
    describir("Actual", actual)

    resumen = generar_reporte_drift(
        preparar_columnas(referencia), preparar_columnas(actual),
        COLUMNAS_NUMERICAS, COLUMNAS_CATEGORICAS, "p2", max_filas=max_filas,
    )
    logger.info("\n" + resumen["por_columna"].to_string(index=False))


if __name__ == "__main__":
    app()