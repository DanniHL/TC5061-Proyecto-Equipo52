"""Preparación de conjuntos de referencia y actuales para el monitoreo de P1."""

import pandas as pd
from pathlib import Path

from equipo52_module.config import (
    ESTACIONES_EXCLUIDAS,
    RAW_DATA_DIR,
)
from equipo52_module.dataset import leer_viajes_mes
from equipo52_module.modeling.train_p1 import (
    VARIABLES_CICLICAS,
    agregar_variables_ciclicas,
    agregar_variables_historicas,
    completar_calendario_por_estacion,
    construir_demanda_diaria,
    preparar_viajes_p1,
)


FECHA_INICIO_PRUEBA = pd.Timestamp("2026-07-01")
FECHA_INICIO_ACTUAL = pd.Timestamp("2026-10-01")
FECHA_FIN_ACTUAL = pd.Timestamp("2026-10-31")

REGISTROS_ENTRENAMIENTO_CAMPEON = 1_070_809


def preparar_conjuntos_p1():
    """Construye los conjuntos de referencia y actuales con la lógica original de P1."""

    # 1. Recuperar el historial hasta septiembre con los filtros originales de P1.
    viajes_historicos = preparar_viajes_p1()

    # 2. Leer octubre y aplicar las mismas exclusiones de estaciones.
    ruta_octubre = RAW_DATA_DIR / "datos_abiertos_2026_10.csv"
    viajes_octubre = leer_viajes_mes(ruta_octubre)

    sin_excluidas = (
        ~viajes_octubre["Origen_Id"].isin(ESTACIONES_EXCLUIDAS)
        & ~viajes_octubre["Destino_Id"].isin(ESTACIONES_EXCLUIDAS)
    )

    viajes_octubre = viajes_octubre.loc[
        sin_excluidas,
        ["Inicio_del_viaje", "Origen_Id"],
    ].copy()

    print(f"Viajes históricos P1: {len(viajes_historicos):,}")
    print(f"Viajes de octubre después de exclusiones: {len(viajes_octubre):,}")

    # 3. Integrar el historial y octubre para calcular correctamente los rezagos.
    viajes = pd.concat(
        [viajes_historicos, viajes_octubre],
        ignore_index=True,
    )

    # 4. Reutilizar las transformaciones originales de P1.
    demanda = construir_demanda_diaria(viajes)
    datos = completar_calendario_por_estacion(demanda)
    datos = agregar_variables_historicas(datos)
    datos = agregar_variables_ciclicas(datos)

    # 5. Conservar únicamente registros con variables y objetivo completos.
    datos_validos = datos.dropna(
        subset=VARIABLES_CICLICAS + ["Numero_viajes"]
    ).copy()

    # 6. Separar la referencia y el periodo actual.
    referencia = datos_validos.loc[
        datos_validos["Fecha"] < FECHA_INICIO_PRUEBA
    ].copy()

    actual = datos_validos.loc[
        datos_validos["Fecha"].between(
            FECHA_INICIO_ACTUAL,
            FECHA_FIN_ACTUAL,
        )
    ].copy()

    if referencia.empty:
        raise ValueError("El conjunto de referencia está vacío.")

    if actual.empty:
        raise ValueError("El conjunto actual de octubre está vacío.")

    # 7. Mostrar información para validar la reconstrucción.
    print("\n=== CONJUNTO DE REFERENCIA ===")
    print(f"Registros: {len(referencia):,}")
    print(f"Periodo: {referencia['Fecha'].min()} a {referencia['Fecha'].max()}")

    print("\n=== CONJUNTO ACTUAL ===")
    print(f"Registros: {len(actual):,}")
    print(f"Periodo: {actual['Fecha'].min()} a {actual['Fecha'].max()}")

    print("\n=== VALIDACIÓN CONTRA EL CAMPEÓN ===")
    print(f"Registros de entrenamiento del campeón: {REGISTROS_ENTRENAMIENTO_CAMPEON:,}")
    print(f"Registros reconstruidos de referencia: {len(referencia):,}")

    if len(referencia) == REGISTROS_ENTRENAMIENTO_CAMPEON:
        print("Resultado: los conteos coinciden.")
    else:
        print("ADVERTENCIA: los conteos no coinciden; debemos investigar la diferencia.")
    # 8. Guardar los conjuntos para el monitoreo y la evaluación A/B.
    carpeta_salida = Path("data/processed/monitoring")
    carpeta_salida.mkdir(parents=True, exist_ok=True)

    referencia.to_parquet(
        carpeta_salida / "p1_referencia.parquet",
        index=False,
    )

    actual.to_parquet(
        carpeta_salida / "p1_actual.parquet",
        index=False,
    )

    print("\n=== ARCHIVOS GUARDADOS ===")
    print(f"Referencia: {carpeta_salida / 'p1_referencia.parquet'}")
    print(f"Actual: {carpeta_salida / 'p1_actual.parquet'}")

    return {
        "datos_validos": datos_validos,
        "referencia": referencia,
        "actual": actual,
    }


from equipo52_module.monitoring.drift import generar_reporte_drift

def generar_reporte_p1():
    """Genera el reporte Evidently de drift y calidad para P1."""
    carpeta_datos = Path("data/processed/monitoring")
    referencia = pd.read_parquet(
        carpeta_datos / "p1_referencia.parquet"
    )

    actual = pd.read_parquet(
        carpeta_datos / "p1_actual.parquet"
    )

    columnas_numericas = [
        "Viajes_dia_anterior",
        "Viajes_hace_7_dias",
        "Promedio_7_dias",
        "Dia_semana_sin",
        "Dia_semana_cos",
        "Mes_sin",
        "Mes_cos",
    ]

    columnas_categoricas = ["Fin_de_semana"]
    resultado = generar_reporte_drift(
        referencia=referencia,
        actual=actual,
        columnas_numericas=columnas_numericas,
        columnas_categoricas=columnas_categoricas,
        nombre="p1",
    )

    print("\n=== REPORTE DE DRIFT P1 ===")
    print(resultado)

    return resultado

if __name__ == "__main__":
    generar_reporte_p1()