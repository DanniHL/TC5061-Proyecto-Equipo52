"""Pruebas del feature engineering de P2 (riesgo de saturación por estación y hora).

Además de revisar cada cálculo, estas pruebas cuidan la regla más importante de features.py:
una variable de la hora t solo usa datos de horas anteriores a t (sin fuga de información).
"""

import numpy as np
import pandas as pd
import pytest

from equipo52_module.features import (
    HORAS_POR_SEMANA,
    NORMAL,
    SIN_ANCLAJES,
    SIN_BICICLETAS,
    agregar_calendario,
    agregar_historial_semanas,
    agregar_movimientos_previos,
    completar_horas_sin_movimiento,
    construir_tabla_p2,
    contar_movimientos_por_hora,
    etiquetar_riesgo,
    quitar_viajes_cortos,
    sumar_proximas_horas,
)


def una_estacion(salidas, llegadas, estacion=1, inicio="2026-10-05 00:00"):
    """Tabla por hora de una estación con las salidas y llegadas indicadas."""
    return pd.DataFrame({
        "estacion_id": estacion,
        "hora": pd.date_range(inicio, periods=len(salidas), freq="h"),
        "salidas": salidas,
        "llegadas": llegadas,
    })


# ------------------------------------------------------------------ limpieza de viajes


def test_quitar_viajes_cortos_borde_de_60_segundos():
    inicio = pd.Timestamp("2026-10-05 08:00")
    viajes = pd.DataFrame({
        "Inicio_del_viaje": [inicio] * 3,
        "Fin_del_viaje": [inicio + pd.Timedelta(seconds=s) for s in (59, 60, 61)],
    })
    resultado = quitar_viajes_cortos(viajes, 60)
    # 59 s se quita; 60 s (exactamente el mínimo) y 61 s se conservan.
    assert list(resultado.index) == [1, 2]


def test_quitar_viajes_cortos_con_datos_del_proyecto(viajes_pequenos):
    resultado = quitar_viajes_cortos(viajes_pequenos, 60)
    assert 4 not in resultado["Viaje_Id"].values
    assert len(resultado) == len(viajes_pequenos) - 1


def test_quitar_viajes_cortos_tabla_vacia():
    vacia = pd.DataFrame({"Inicio_del_viaje": pd.Series(dtype="datetime64[us]"),
                          "Fin_del_viaje": pd.Series(dtype="datetime64[us]")})
    assert quitar_viajes_cortos(vacia, 60).empty


# ------------------------------------------------------------------ movimientos por hora


def test_contar_movimientos_por_hora(viajes_pequenos):
    movimientos = contar_movimientos_por_hora(viajes_pequenos)
    # Estación 1 a las 7:00 del lunes: salen los viajes 1 y 2, no llega ninguno.
    assert movimientos.loc[(1, pd.Timestamp("2026-10-05 07:00"))].tolist() == [2, 0]
    # El total de salidas y de llegadas es el número de viajes (cada viaje sale y llega una vez).
    assert movimientos["salidas"].sum() == movimientos["llegadas"].sum() == len(viajes_pequenos)


def test_completar_horas_sin_movimiento_rellena_con_cero():
    movimientos = pd.DataFrame(
        {"salidas": [2, 1], "llegadas": [0, 3]},
        index=pd.MultiIndex.from_tuples([(1, pd.Timestamp("2026-10-05 07:00")),
                                         (1, pd.Timestamp("2026-10-05 10:00"))],
                                        names=["estacion_id", "hora"]),
    )
    por_hora = completar_horas_sin_movimiento(movimientos)
    assert len(por_hora) == 4                               # 7, 8, 9 y 10 horas
    assert por_hora.loc[1:2, ["salidas", "llegadas"]].sum().sum() == 0
    assert por_hora["hora"].is_monotonic_increasing


# ------------------------------------------------------------------ etiqueta de riesgo


def test_sumar_proximas_horas():
    resultado = sumar_proximas_horas(pd.Series([1.0, 2.0, 3.0, 4.0]), horas=2)
    # Cada hora suma ella misma y la siguiente; la última no tiene ventana completa.
    assert resultado.iloc[:3].tolist() == [3.0, 5.0, 7.0]
    assert np.isnan(resultado.iloc[3])


def test_etiquetar_riesgo_borde_exacto_del_umbral():
    # Capacidad 3 y umbral 1/3: un desbalance de 1 bicicleta ya es exactamente el umbral.
    por_hora = una_estacion(salidas=[1, 0, 0], llegadas=[0, 1, 0])
    clase = etiquetar_riesgo(por_hora, pd.Series({1: 3}), horas=1, umbral=1 / 3)
    assert clase.tolist() == [SIN_BICICLETAS, SIN_ANCLAJES, NORMAL]


def test_etiquetar_riesgo_debajo_del_umbral_es_normal():
    # Capacidad 10: salir 3 bicicletas más de las que llegan (30 %) no alcanza el umbral de 1/3.
    por_hora = una_estacion(salidas=[3, 0], llegadas=[0, 0])
    clase = etiquetar_riesgo(por_hora, pd.Series({1: 10}), horas=1, umbral=1 / 3)
    assert clase.tolist() == [NORMAL, NORMAL]


def test_etiquetar_riesgo_ultimas_horas_sin_ventana_quedan_vacias():
    por_hora = una_estacion(salidas=[0, 0, 0, 0], llegadas=[0, 0, 0, 0])
    clase = etiquetar_riesgo(por_hora, pd.Series({1: 10}), horas=3, umbral=1 / 3)
    assert clase.iloc[:2].tolist() == [NORMAL, NORMAL]
    assert clase.iloc[2:].isna().all()


# ------------------------------------------------------------------ calendario


def test_agregar_calendario():
    por_hora = una_estacion(salidas=[0], llegadas=[0], inicio="2026-10-10 18:00")   # sábado
    fila = agregar_calendario(por_hora).iloc[0]
    assert (fila["hora_del_dia"], fila["dia_semana"], fila["mes"], fila["fin_de_semana"]) == (18, 5, 10, 1)


# ------------------------------------------------------------------ variables del modelo (sin fuga)


def test_movimientos_previos_no_usan_la_hora_actual():
    por_hora = una_estacion(salidas=[5, 1, 2, 3], llegadas=[0, 0, 0, 0])
    resultado = agregar_movimientos_previos(por_hora)
    assert np.isnan(resultado["salidas_hora_anterior"].iloc[0])        # no hay hora anterior
    assert resultado["salidas_hora_anterior"].tolist()[1:] == [5, 1, 2]
    assert resultado["salidas_3_horas_anteriores"].iloc[3] == 5 + 1 + 2

    # Sin fuga: cambiar la hora actual no cambia sus propias variables.
    modificada = por_hora.copy()
    modificada.loc[3, "salidas"] = 99
    assert agregar_movimientos_previos(modificada).iloc[3]["salidas_hora_anterior"] == 2


def test_movimientos_previos_no_mezclan_estaciones():
    dos = pd.concat([una_estacion([4, 4], [0, 0], estacion=1), una_estacion([7, 7], [0, 0], estacion=2)],
                    ignore_index=True)
    resultado = agregar_movimientos_previos(dos)
    # La primera hora de la estación 2 no hereda las salidas de la estación 1.
    assert np.isnan(resultado.loc[2, "salidas_hora_anterior"])


def test_historial_semanas_cuenta_fallas_a_la_misma_hora():
    horas = HORAS_POR_SEMANA * 3
    por_hora = una_estacion(salidas=[0] * horas, llegadas=[0] * horas)
    por_hora["clase"] = pd.array([NORMAL] * horas, dtype="Int8")
    por_hora.loc[[0, HORAS_POR_SEMANA], "clase"] = SIN_BICICLETAS   # misma hora en semanas 1 y 2

    resultado = agregar_historial_semanas(por_hora, semanas=2)
    fila = resultado.iloc[2 * HORAS_POR_SEMANA]                     # misma hora de la semana 3
    assert fila["semanas_sin_bicicletas"] == 2
    assert fila["semanas_sin_anclajes"] == 0
    assert fila["clase_semana_anterior"] == SIN_BICICLETAS
    # Caso límite: antes de tener 2 semanas completas de historia la variable queda vacía.
    assert resultado["semanas_sin_bicicletas"].iloc[: 2 * HORAS_POR_SEMANA].isna().all()


# ------------------------------------------------------------------ tabla completa


def test_construir_tabla_p2(viajes_pequenos, estaciones_pequenas):
    tabla = construir_tabla_p2(viajes_pequenos, estaciones_pequenas, horas=3, umbral=1 / 3, semanas=1)
    columnas_modelo = {"hora_del_dia", "capacidad", "latitud", "salidas_hora_anterior", "semanas_sin_bicicletas"}
    assert columnas_modelo <= set(tabla.columns)
    assert not tabla.duplicated(["estacion_id", "hora"]).any()       # una fila por estación y hora
    assert set(tabla["clase"].dropna().unique()) <= {NORMAL, SIN_BICICLETAS, SIN_ANCLAJES}


def test_construir_tabla_p2_excluye_estaciones_sin_ficha(viajes_pequenos, estaciones_pequenas):
    tabla = construir_tabla_p2(viajes_pequenos, estaciones_pequenas.drop(index=3), horas=3,
                               umbral=1 / 3, semanas=1)
    assert 3 not in tabla["estacion_id"].values


@pytest.mark.parametrize("umbral", [0.1, 0.5, 1.0])
def test_construir_tabla_p2_umbral_mas_alto_no_agrega_fallas(viajes_pequenos, estaciones_pequenas, umbral):
    base = construir_tabla_p2(viajes_pequenos, estaciones_pequenas, horas=3, umbral=0.05, semanas=1)
    otra = construir_tabla_p2(viajes_pequenos, estaciones_pequenas, horas=3, umbral=umbral, semanas=1)
    assert (otra["clase"] != NORMAL).sum() <= (base["clase"] != NORMAL).sum()