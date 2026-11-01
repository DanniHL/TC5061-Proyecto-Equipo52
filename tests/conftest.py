import pandas as pd
import pytest


@pytest.fixture
def viajes_p1_ejemplo():
    """Genera un conjunto pequeño de viajes para las pruebas de P1."""
    return pd.DataFrame(
        {
            "Inicio_del_viaje": pd.to_datetime(
                [
                    "2026-01-01 08:00:00",
                    "2026-01-01 09:00:00",
                    "2026-01-02 10:00:00",
                    "2026-01-01 11:00:00",
                ]
            ),
            "Origen_Id": [10, 10, 10, 20],
        }
    )


# ------------------------------------------------------------------ P2 (Roberto)


def viaje(viaje_id, inicio, minutos, origen, destino):
    """Un viaje con la misma forma que deja la ingesta (data/interim/viajes.parquet)."""
    inicio = pd.Timestamp(inicio)
    return {"Viaje_Id": viaje_id, "Inicio_del_viaje": inicio,
            "Fin_del_viaje": inicio + pd.Timedelta(minutes=minutos),
            "Origen_Id": origen, "Destino_Id": destino}


@pytest.fixture
def viajes_pequenos() -> pd.DataFrame:
    """Ocho viajes entre las estaciones 1, 2 y 3 en dos días (lunes 5 y sábado 10 de octubre de 2026).

    Casos que importan para P2: dos salidas de la estación 1 en la misma hora (conteo por hora),
    un viaje de menos de 1 minuto (lo quita quitar_viajes_cortos) y viajes en sábado (calendario).
    """
    filas = [
        viaje(1, "2026-10-05 07:10", 12, 1, 2),
        viaje(2, "2026-10-05 07:40", 15, 1, 3),
        viaje(3, "2026-10-05 08:05", 9, 2, 1),
        viaje(4, "2026-10-05 08:30", 0.5, 3, 3),   # menos de 1 minuto: no es un viaje real
        viaje(5, "2026-10-05 18:00", 20, 3, 1),
        viaje(6, "2026-10-10 10:00", 40, 2, 2),
        viaje(7, "2026-10-10 10:15", 2, 1, 1),
        viaje(8, "2026-10-10 12:00", 25, 1, 2),
    ]
    viajes = pd.DataFrame(filas)
    for columna in ["Inicio_del_viaje", "Fin_del_viaje"]:
        viajes[columna] = viajes[columna].astype("datetime64[us]")
    return viajes


@pytest.fixture
def estaciones_pequenas() -> pd.DataFrame:
    """Capacidad y ubicación de las tres estaciones, indexadas como en cargar_estaciones() de train_p2.py."""
    return pd.DataFrame(
        {"capacidad": [12, 15, 9], "latitud": [20.67, 20.68, 20.69], "longitud": [-103.35, -103.36, -103.37]},
        index=pd.Index([1, 2, 3], name="estacion_id"),
    )

# ------------------------------------------------------------------ P3 (Joel)


@pytest.fixture
def viajes_p3() -> pd.DataFrame:
    """Doce viajes entre las estaciones 1 a 6, pensados para las características de P3.

    Casos que importan: la estación 1 solo tiene salidas (balance −1) y la 2 solo llegadas
    (las características de salidas quedan vacías y la matriz las rellena); la 3 concentra
    los viajes redondos (uno de menos de 3 min que quitar_viajes_redondos_cortos descarta,
    uno de exactamente 3 min que se conserva y un paseo largo de sábado); la 4 tiene viajes
    pero no ficha en la nomenclatura (la matriz la excluye), la 5 tiene ficha pero ningún
    viaje (también queda fuera) y la 6 es una estación consolidada (activa desde 2020, con
    un viaje reciente para medir su actividad en los últimos 12 meses).
    """
    filas = [
        viaje(1, "2026-01-05 08:00", 20, 1, 2),    # lunes: pico de mañana laboral
        viaje(2, "2026-01-05 08:30", 15, 1, 3),
        viaje(3, "2026-01-06 17:45", 25, 1, 2),    # martes: pico de tarde
        viaje(4, "2026-02-02 07:50", 30, 1, 3),    # segundo mes activo de la estación 1
        viaje(5, "2026-01-10 10:00", 40, 3, 3),    # sábado: paseo redondo recreativo
        viaje(6, "2026-01-10 11:00", 2, 3, 3),     # devolución tardía: la quita el filtro de 3 min
        viaje(7, "2026-01-10 11:30", 3, 3, 3),     # exactamente 3 min: se conserva
        viaje(8, "2026-02-01 12:00", 18, 3, 4),    # domingo, hacia una estación sin ficha
        viaje(9, "2026-02-05 19:00", 10, 3, 4),    # jueves, hacia una estación sin ficha
        viaje(10, "2026-02-09 09:00", 22, 4, 3),   # la estación sin ficha también origina viajes
        viaje(11, "2020-01-10 08:00", 25, 6, 2),   # la estación 6 lleva activa desde 2020
        viaje(12, "2026-01-20 08:30", 25, 6, 2),   # y sigue usándose en 2026
    ]
    viajes = pd.DataFrame(filas)
    for columna in ["Inicio_del_viaje", "Fin_del_viaje"]:
        viajes[columna] = viajes[columna].astype("datetime64[us]")
    return viajes


@pytest.fixture
def estaciones_p3() -> pd.DataFrame:
    """Capacidad y ubicación de las estaciones 1, 2, 3, 5 y 6 (la 4 no tiene ficha), como en cargar_estaciones()."""
    return pd.DataFrame(
        {"capacidad": [10, 14, 8, 12, 16],
         "latitud": [20.67, 20.68, 20.69, 20.70, 20.71],
         "longitud": [-103.35, -103.36, -103.37, -103.38, -103.39]},
        index=pd.Index([1, 2, 3, 5, 6], name="estacion_id"),
    )


@pytest.fixture
def fin_p3() -> pd.Timestamp:
    """Cierre del periodo de las pruebas de P3: febrero de 2026, como FECHA_FIN cierra septiembre."""
    return pd.Timestamp("2026-02-28")
