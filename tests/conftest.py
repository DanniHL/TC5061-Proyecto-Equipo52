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