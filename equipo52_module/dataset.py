"""Ingesta de los datos abiertos de MiBici.

Descarga los archivos mensuales de viajes y la nomenclatura de estaciones, unifica los
formatos que cambian entre meses y guarda un solo archivo Parquet con las columnas
permitidas. También escribe un manifiesto (archivo, registros y huella SHA-256) que se
guarda en Git para comprobar que todo el equipo usa los mismos datos.

Uso: python equipo52_module/dataset.py
"""

import hashlib
import re
import urllib.request
from pathlib import Path
from urllib.parse import urljoin

from loguru import logger
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
import typer

from equipo52_module.config import (
    COLUMNAS_VIAJES,
    FECHA_FIN,
    FECHA_INICIO,
    MANIFIESTO_FILE,
    RAW_DATA_DIR,
    URL_DATOS_ABIERTOS,
    VIAJES_FILE,
)

app = typer.Typer()

# Nombre limpio de cada archivo; algunas ligas traen un sufijo como "-1" (datos_abiertos_2016_01-1.csv).
PATRON_ARCHIVO = re.compile(r"(datos_abiertos|nomenclatura)_(\d{4})_(\d{2})(?:-\d+)?\.csv$")

# Formatos de fecha observados en el notebook 1.0-rr-p2-exploracion-datos:
# la mayoría de los meses usa el primero; 2023-11, 2024-09 y 2024-10 usan el segundo.
FORMATOS_FECHA = ["%Y-%m-%d %H:%M:%S", "%d/%m/%Y %H:%M"]

# El servidor de MiBici rechaza (error 406) las peticiones sin identificación o con la de un navegador genérico.
ENCABEZADOS = {"User-Agent": "equipo52-mibici/1.0"}

# Codificaciones observadas en el mismo notebook: 97 archivos en UTF-8 y 45 en Latin-1.
CODIFICACIONES = ["utf-8-sig", "latin-1"]


def obtener_ligas(url_pagina: str) -> dict[str, str]:
    """Busca en la página de datos abiertos las ligas a los archivos CSV.

    Las direcciones no siguen un patrón (cada archivo está en una carpeta distinta),
    así que se leen de la página. Regresa {nombre limpio: dirección completa}.
    """
    peticion = urllib.request.Request(url_pagina, headers=ENCABEZADOS)
    with urllib.request.urlopen(peticion) as respuesta:
        html = respuesta.read().decode("utf-8")
    ligas = {}
    for ruta in re.findall(r'href="([^"]+\.csv)"', html):
        coincidencia = PATRON_ARCHIVO.search(ruta)
        if coincidencia:
            tipo, anio, mes = coincidencia.groups()
            ligas[f"{tipo}_{anio}_{mes}.csv"] = urljoin(url_pagina, ruta)
    return ligas


def en_periodo(nombre: str, inicio: str, fin: str) -> bool:
    """Indica si un archivo mensual de viajes cae dentro del periodo de análisis.

    Evita que un mes publicado después (por ejemplo, octubre de 2026) cambie los datos
    del proyecto sin que el equipo lo decida.
    """
    tipo, anio, mes = PATRON_ARCHIVO.search(nombre).groups()
    if tipo != "datos_abiertos":
        return True
    return inicio[:7] <= f"{anio}-{mes}" <= fin[:7]


def descargar_faltantes(ligas: dict[str, str], carpeta: Path) -> None:
    """Descarga solo los archivos que todavía no están en la carpeta."""
    carpeta.mkdir(parents=True, exist_ok=True)
    faltantes = {nombre: url for nombre, url in ligas.items() if not (carpeta / nombre).exists()}
    logger.info(f"Archivos ya descargados: {len(ligas) - len(faltantes)}; por descargar: {len(faltantes)}")
    for nombre, url in tqdm(faltantes.items(), desc="Descargando"):
        destino = carpeta / nombre
        temporal = destino.with_suffix(".parcial")
        peticion = urllib.request.Request(url, headers=ENCABEZADOS)
        with urllib.request.urlopen(peticion) as respuesta:
            temporal.write_bytes(respuesta.read())
        temporal.rename(destino)


def convertir_fechas(texto: pd.Series) -> pd.Series:
    """Convierte las fechas de un mes probando los formatos conocidos.

    Si ningún formato entiende todas las fechas, se detiene con un error en lugar de
    dejar fechas vacías: así un formato nuevo de MiBici no pasa desapercibido.
    """
    for formato in FORMATOS_FECHA:
        fechas = pd.to_datetime(texto, format=formato, errors="coerce")
        if fechas.notna().all():
            return fechas.astype("datetime64[us]")
    ejemplo = texto[fechas.isna()].iloc[0]
    raise ValueError(f"Fecha con formato desconocido: {ejemplo!r}")


def leer_viajes_mes(ruta: Path) -> pd.DataFrame:
    """Lee las columnas permitidas de un archivo mensual de viajes.

    Prueba las codificaciones conocidas y convierte las fechas a un solo formato.
    Nunca carga las columnas con datos personales.
    """
    for codificacion in CODIFICACIONES:
        try:
            viajes = pd.read_csv(
                ruta,
                usecols=COLUMNAS_VIAJES,
                dtype={"Inicio_del_viaje": str, "Fin_del_viaje": str},
                encoding=codificacion,
            )
            break
        except UnicodeDecodeError:
            continue
    for columna in ["Inicio_del_viaje", "Fin_del_viaje"]:
        viajes[columna] = convertir_fechas(viajes[columna])
    for columna in ["Viaje_Id", "Origen_Id", "Destino_Id"]:
        viajes[columna] = viajes[columna].astype("int64")
    return viajes[COLUMNAS_VIAJES]


def huella_sha256(ruta: Path) -> str:
    """Calcula la huella SHA-256 de un archivo para comprobar que es idéntico en cada computadora."""
    huella = hashlib.sha256()
    with ruta.open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(1024 * 1024), b""):
            huella.update(bloque)
    return huella.hexdigest()


@app.command()
def main(
    carpeta_raw: Path = RAW_DATA_DIR,
    salida: Path = VIAJES_FILE,
    manifiesto: Path = MANIFIESTO_FILE,
):
    ligas = obtener_ligas(URL_DATOS_ABIERTOS)
    ligas = {nombre: url for nombre, url in ligas.items() if en_periodo(nombre, FECHA_INICIO, FECHA_FIN)}
    logger.info(f"Ligas encontradas en el periodo {FECHA_INICIO} a {FECHA_FIN}: {len(ligas)}")
    descargar_faltantes(ligas, carpeta_raw)

    salida.parent.mkdir(parents=True, exist_ok=True)
    registros = {}
    escritor = None
    meses = sorted(nombre for nombre in ligas if nombre.startswith("datos_abiertos"))
    for nombre in tqdm(meses, desc="Leyendo meses"):
        viajes = leer_viajes_mes(carpeta_raw / nombre)
        registros[nombre] = len(viajes)
        tabla = pa.Table.from_pandas(viajes, preserve_index=False)
        if escritor is None:
            escritor = pq.ParquetWriter(salida, tabla.schema)
        escritor.write_table(tabla)
    escritor.close()

    for nombre in ligas:
        if nombre.startswith("nomenclatura"):
            registros[nombre] = len(pd.read_csv(carpeta_raw / nombre))
    filas = [
        {"archivo": nombre, "registros": registros[nombre], "sha256": huella_sha256(carpeta_raw / nombre),
         "url": ligas[nombre]}
        for nombre in sorted(ligas)
    ]
    manifiesto.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(filas).to_csv(manifiesto, index=False)

    total = sum(registros[nombre] for nombre in meses)
    logger.success(f"{len(meses)} meses, {total:,} viajes guardados en {salida}")
    logger.success(f"Manifiesto con {len(filas)} archivos en {manifiesto}")


if __name__ == "__main__":
    app()
