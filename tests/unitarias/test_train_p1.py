import numpy as np
import pandas as pd
import pytest

from equipo52_module.modeling import train_p1

def test_construir_demanda_diaria_conserva_total_de_viajes(
    viajes_p1_ejemplo,
):
    viajes = viajes_p1_ejemplo

    demanda = train_p1.construir_demanda_diaria(viajes)

    # La suma de los viajes agrupados debe ser igual a los viajes originales.
    assert demanda["Numero_viajes"].sum() == len(viajes)

    # Debe existir un solo registro por estación y fecha.
    assert not demanda.duplicated(["Origen_Id", "Fecha"]).any()

    # Se esperan tres combinaciones de estación y fecha.
    assert len(demanda) == 3


def test_construir_demanda_diaria_con_dataframe_vacio():
    viajes = pd.DataFrame(
        {
            "Inicio_del_viaje": pd.to_datetime([]),
            "Origen_Id": pd.Series(dtype="int64"),
        }
    )

    demanda = train_p1.construir_demanda_diaria(viajes)

    assert demanda.empty
    assert list(demanda.columns) == [
        "Origen_Id",
        "Fecha",
        "Numero_viajes",
    ]


def test_completar_calendario_conserva_dias_sin_registros_como_nan():
    demanda = pd.DataFrame(
        {
            "Origen_Id": [10, 10, 20],
            "Fecha": pd.to_datetime(
                ["2026-01-01", "2026-01-03", "2026-01-01"]
            ),
            "Numero_viajes": [5, 8, 3],
        }
    )

    completo = train_p1.completar_calendario_por_estacion(demanda)

    # La estación 10 debe tener también el 2 de enero.
    dia_faltante = completo.loc[
        (completo["Origen_Id"] == 10)
        & (completo["Fecha"] == pd.Timestamp("2026-01-02"))
    ]

    assert len(dia_faltante) == 1
    assert pd.isna(dia_faltante.iloc[0]["Numero_viajes"])

    # La estación 20 solamente tiene un día observado.
    assert len(completo.loc[completo["Origen_Id"] == 20]) == 1


def test_completar_calendario_con_dataframe_vacio():
    demanda = pd.DataFrame(
        columns=["Origen_Id", "Fecha", "Numero_viajes"]
    )

    with pytest.raises(ValueError, match="No hay registros"):
        train_p1.completar_calendario_por_estacion(demanda)


def test_agregar_variables_historicas_calcula_desplazamientos():
    datos = pd.DataFrame(
        {
            "Origen_Id": [10] * 9,
            "Fecha": pd.date_range("2026-01-01", periods=9, freq="D"),
            "Numero_viajes": range(1, 10),
        }
    )

    resultado = train_p1.agregar_variables_historicas(datos)

    # Revisamos el 8 de enero.
    dia = resultado.loc[
        resultado["Fecha"] == pd.Timestamp("2026-01-08")
    ].iloc[0]

    assert dia["Viajes_dia_anterior"] == 7
    assert dia["Viajes_hace_7_dias"] == 1
    assert dia["Promedio_7_dias"] == pytest.approx(4.0)


def test_agregar_variables_historicas_no_calcula_promedio_sin_siete_dias():
    datos = pd.DataFrame(
        {
            "Origen_Id": [10] * 6,
            "Fecha": pd.date_range("2026-01-01", periods=6, freq="D"),
            "Numero_viajes": [2, 4, 6, 8, 10, 12],
        }
    )

    resultado = train_p1.agregar_variables_historicas(datos)

    # Durante los primeros seis días no hay siete días previos completos.
    assert resultado["Promedio_7_dias"].isna().all()


def test_separar_datos_cronologicamente_sin_traslape():
    datos = pd.DataFrame(
        {
            "Origen_Id": [10] * 5,
            "Fecha": pd.date_range("2026-01-01", periods=5, freq="D"),
            "Numero_viajes": [10, 12, 14, 16, 18],
            "Viajes_dia_anterior": [8, 10, 12, 14, 16],
            "Viajes_hace_7_dias": [5, 6, 7, 8, 9],
            "Promedio_7_dias": [6, 7, 8, 9, 10],
        }
    )

    datos_validos, entrenamiento, prueba = (
        train_p1.separar_datos_cronologicamente(
            datos,
            fecha_inicio_prueba=pd.Timestamp("2026-01-04"),
        )
    )

    assert len(datos_validos) == 5
    assert len(entrenamiento) == 3
    assert len(prueba) == 2

    assert entrenamiento["Fecha"].max() < prueba["Fecha"].min()

    assert (
        entrenamiento["Fecha"].max()
        < pd.Timestamp("2026-01-04")
    )
    assert (
        prueba["Fecha"].min()
        >= pd.Timestamp("2026-01-04")
    )


def test_separar_datos_cronologicamente_falla_si_no_hay_prueba():
    datos = pd.DataFrame(
        {
            "Origen_Id": [10] * 3,
            "Fecha": pd.date_range("2026-01-01", periods=3, freq="D"),
            "Numero_viajes": [10, 12, 14],
            "Viajes_dia_anterior": [8, 10, 12],
            "Viajes_hace_7_dias": [5, 6, 7],
            "Promedio_7_dias": [6, 7, 8],
        }
    )

    with pytest.raises(ValueError, match="conjunto vacío"):
        train_p1.separar_datos_cronologicamente(
            datos,
            fecha_inicio_prueba=pd.Timestamp("2026-02-01"),
        )


def test_calcular_metricas_devuelve_mae_y_rmse():
    reales = [1, 2, 3]
    predicciones = [1, 3, 2]

    metricas = train_p1.calcular_metricas(reales, predicciones)

    assert metricas["MAE"] == pytest.approx(2 / 3)
    assert metricas["RMSE"] == pytest.approx(np.sqrt(2 / 3))
    assert set(metricas.keys()) == {"MAE", "RMSE"}