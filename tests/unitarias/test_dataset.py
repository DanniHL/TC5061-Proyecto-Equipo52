"""Pruebas de la ingesta (equipo52_module/dataset.py).

La ingesta es la puerta de entrada de los datos, si lee mal una fecha o deja pasar una
columna con datos personales, el error llega a los tres modelos.
"""

import hashlib

import pandas as pd
import pytest

from equipo52_module.config import COLUMNAS_VIAJES
from equipo52_module.dataset import (
    PATRON_ARCHIVO,
    convertir_fechas,
    en_periodo,
    huella_sha256,
    leer_viajes_mes,
    validar_liga,
)

# ------------------------------------------------------------------ nombres de archivo


@pytest.mark.parametrize("nombre, esperado", [
    ("datos_abiertos_2024_09.csv", ("datos_abiertos", "2024", "09")),
    ("datos_abiertos_2016_01-1.csv", ("datos_abiertos", "2016", "01")),   # liga con sufijo "-1"
    ("nomenclatura_2026_09.csv", ("nomenclatura", "2026", "09")),
])
def test_patron_archivo_reconoce_nombres_validos(nombre, esperado):
    assert PATRON_ARCHIVO.search(nombre).groups() == esperado


@pytest.mark.parametrize("nombre", ["datos_abiertos_2024_9.csv", "viajes_2024_09.csv", "datos_abiertos_2024_09.xlsx"])
def test_patron_archivo_rechaza_nombres_invalidos(nombre):
    assert PATRON_ARCHIVO.search(nombre) is None


# ------------------------------------------------------------------ periodo de análisis


@pytest.mark.parametrize("nombre, dentro", [
    ("datos_abiertos_2020_06.csv", True),
    ("datos_abiertos_2014_12.csv", True),    # borde: primer mes del periodo
    ("datos_abiertos_2026_09.csv", True),    # borde: último mes del periodo
    ("datos_abiertos_2014_11.csv", False),
    ("datos_abiertos_2026_10.csv", False),   # un mes publicado después no cambia los datos
    ("nomenclatura_2030_01.csv", True),      # la nomenclatura siempre se descarga
])
def test_en_periodo(nombre, dentro):
    assert en_periodo(nombre, "2014-12-01", "2026-09-30") is dentro


# ------------------------------------------------------------------ fechas


def test_convertir_fechas_formato_principal():
    fechas = convertir_fechas(pd.Series(["2026-09-01 07:15:00", "2026-09-30 23:59:59"]))
    assert fechas.dtype == "datetime64[us]"
    assert fechas.iloc[0] == pd.Timestamp("2026-09-01 07:15:00")


def test_convertir_fechas_formato_dia_mes_anio():
    # 2023-11, 2024-09 y 2024-10 usan este formato; el día va primero.
    fechas = convertir_fechas(pd.Series(["05/11/2023 08:30", "30/11/2023 17:05"]))
    assert fechas.iloc[0] == pd.Timestamp("2023-11-05 08:30")


def test_convertir_fechas_formato_desconocido_se_detiene():
    with pytest.raises(ValueError, match="formato desconocido"):
        convertir_fechas(pd.Series(["2026-09-01 07:15:00", "1 de septiembre"]))


def test_convertir_fechas_no_acepta_formatos_mezclados():
    # Caso límite: un mes con dos formatos no se acepta a medias (quedarían fechas vacías).
    with pytest.raises(ValueError):
        convertir_fechas(pd.Series(["2026-09-01 07:15:00", "05/11/2023 08:30"]))


# ------------------------------------------------------------------ lectura de un mes

def escribir_mes(ruta, codificacion):
    """CSV con el formato de MiBici, incluidas las columnas con datos personales."""
    contenido = (
        "Viaje_Id,Usuario_Id,Genero,Año_de_nacimiento,Inicio_del_viaje,Fin_del_viaje,Origen_Id,Destino_Id\n"
        "101,9001,M,1990,2026-09-01 07:00:00,2026-09-01 07:12:00,10,20\n"
        "102,9002,F,1985,2026-09-01 07:05:00,2026-09-01 07:30:00,20,10\n"
    )
    ruta.write_bytes(contenido.encode(codificacion))
    return ruta


@pytest.mark.parametrize("codificacion", ["utf-8", "latin-1"])
def test_leer_viajes_mes_omite_datos_personales(tmp_path, codificacion):
    viajes = leer_viajes_mes(escribir_mes(tmp_path / "datos_abiertos_2026_09.csv", codificacion))
    assert list(viajes.columns) == COLUMNAS_VIAJES
    assert not {"Usuario_Id", "Genero", "Año_de_nacimiento"} & set(viajes.columns)
    assert len(viajes) == 2


def test_leer_viajes_mes_tipos(tmp_path):
    viajes = leer_viajes_mes(escribir_mes(tmp_path / "mes.csv", "utf-8"))
    assert viajes["Inicio_del_viaje"].dtype == "datetime64[us]"
    assert (viajes[["Viaje_Id", "Origen_Id", "Destino_Id"]].dtypes == "int64").all()    



# ------------------------------------------------------------------ huella del manifiesto


def test_huella_sha256_coincide_con_hashlib(tmp_path):
    ruta = tmp_path / "archivo.csv"
    ruta.write_bytes(b"Viaje_Id\n1\n")
    assert huella_sha256(ruta) == hashlib.sha256(b"Viaje_Id\n1\n").hexdigest()


def test_huella_sha256_cambia_con_un_byte(tmp_path):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    a.write_bytes(b"1,2,3")
    b.write_bytes(b"1,2,4")
    assert huella_sha256(a) != huella_sha256(b)


def test_huella_sha256_archivo_vacio(tmp_path):
    ruta = tmp_path / "vacio.csv"
    ruta.write_bytes(b"")
    assert huella_sha256(ruta) == hashlib.sha256(b"").hexdigest()


# ------------------------------------------------------------------ seguridad de las ligas


def test_validar_liga_acepta_mibici():
    url = "https://mibici.net/site/assets/files/1079/datos_abiertos_2014_12.csv"
    assert validar_liga(url) == url


@pytest.mark.parametrize("url", [
    "http://mibici.net/datos.csv",                 # sin cifrar
    "file:///etc/passwd",                          # archivo local
    "https://otro-sitio.com/datos.csv",            # otro dominio
    "https://mibici.net.sitio-falso.com/x.csv",    # dominio que solo empieza igual
    "ftp://mibici.net/datos.csv",                  # otro protocolo
])
def test_validar_liga_rechaza_ligas_no_permitidas(url):
    with pytest.raises(ValueError, match="Liga no permitida"):
        validar_liga(url)    