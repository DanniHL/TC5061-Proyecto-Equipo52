"""Chequeos de seguridad del repositorio: secretos, datos personales y configuración.

Complementan a bandit (código) y pip-audit (dependencias): revisan lo que Git sí versiona.
Se basan en `git ls-files`, así que miran exactamente lo que verá quien clone el repositorio.
"""

import os
from pathlib import Path
import re
import subprocess
import sys

import pytest

PROJ_ROOT = Path(__file__).resolve().parents[2]

# Patrones de secretos comunes (sensibles a mayúsculas para no confundirlos con imágenes en base64).
PATRONES_SECRETOS = {
    "llave de AWS": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "token de GitHub": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36}\b"),
    "llave privada": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "contraseña en el código": re.compile(r"(?i)\b(password|passwd|contrase[nñ]a|api_key|secret)\s*=\s*['\"][^'\"]{6,}['\"]"),
    "URI con usuario y contraseña": re.compile(r"://[^/\s:@'\"]+:[^/\s@'\"]+@"),
}
ARCHIVOS_PROHIBIDOS = re.compile(r"(^|/)(\.env(\.(?!example$).+)?|id_rsa|credentials.*\.json|.*\.pem|.*\.key|mlflow\.db)$")


@pytest.fixture(scope="module")
def versionados() -> list[str]:
    """Archivos que Git sube a GitHub (git ls-files)."""
    salida = subprocess.run(["git", "ls-files", "-z"], cwd=PROJ_ROOT, capture_output=True, text=True, encoding="utf-8", check=True)
    return [ruta for ruta in salida.stdout.split("\0") if ruta]


def texto_sin_imagenes(ruta: str) -> str:
    """Contenido del archivo sin las imágenes en base64 de los notebooks (no son texto y dan falsos positivos)."""
    texto = (PROJ_ROOT / ruta).read_text(encoding="utf-8", errors="ignore")
    return re.sub(r'"image/(png|jpeg|svg\+xml)": "[^"]*"', '""', texto)

# ------------------------------------------------------------------ secretos

def test_no_se_versionan_archivos_de_credenciales(versionados):
    prohibidos = [ruta for ruta in versionados if ARCHIVOS_PROHIBIDOS.search(ruta)]
    assert prohibidos == [], f"Archivos con credenciales en Git: {prohibidos}"


def test_no_hay_secretos_en_archivos_versionados(versionados):
    extensiones = (".py", ".ipynb", ".md", ".toml", ".yaml", ".yml", ".txt", ".cfg", ".json", "MLproject", "Makefile")
    hallazgos = []
    for ruta in versionados:
        if not ruta.endswith(extensiones) or not (PROJ_ROOT / ruta).exists():
            continue
        texto = texto_sin_imagenes(ruta)
        hallazgos += [f"{ruta}: {nombre}" for nombre, patron in PATRONES_SECRETOS.items() if patron.search(texto)]
    assert hallazgos == [], "Posibles secretos:\n" + "\n".join(hallazgos)


@pytest.mark.parametrize("texto, tipo", [
    ("clave = 'AKIA" + "ABCDEFGHIJKLMNOP'", "llave de AWS"),
    ("pass" + "word = 'superSecreta123'", "contraseña en el código"),
    ("uri = 'postgresql://admin:" + "1234@servidor/db'", "URI con usuario y contraseña"),
])
def test_los_patrones_si_detectan_secretos(texto, tipo):
    """Control: si los patrones no detectaran nada, la prueba anterior pasaría sin revisar."""
    assert PATRONES_SECRETOS[tipo].search(texto)


def test_patrones_ignoran_codigo_normal():
    codigo = 'MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")'
    assert not any(patron.search(codigo) for patron in PATRONES_SECRETOS.values())


# ------------------------------------------------------------------ .gitignore y variables de entorno

@pytest.mark.parametrize("ruta", [
    ".env", ".env.local", "data/raw/datos_abiertos_2026_09.csv", "data/interim/viajes.parquet",
    "mlflow.db", "mlruns/0/meta.yaml", "mlartifacts/1/modelo.pkl", ".venv/bin/python",
    "llave.pem", "credentials.json",
])
def test_gitignore_excluye_lo_sensible(ruta):
    resultado = subprocess.run(["git", "check-ignore", "-q", ruta], cwd=PROJ_ROOT)
    assert resultado.returncode == 0, f"{ruta} no está en .gitignore"


def test_env_example_se_versiona_y_no_trae_valores_secretos(versionados):
    assert ".env.example" in versionados
    for linea in (PROJ_ROOT / ".env.example").read_text().splitlines():
        if re.match(r"\s*[A-Z_]*(PASSWORD|TOKEN|SECRET|KEY)[A-Z_]*\s*=", linea):
            assert linea.split("=", 1)[1].strip() == "", f"Valor real en .env.example: {linea}"


def test_mlflow_lee_la_direccion_de_una_variable_de_entorno():
    # Se corre en otro proceso porque config.py lee la variable una sola vez, al importarse.
    entorno = {**os.environ, "MLFLOW_TRACKING_URI": "http://servidor-del-equipo:5000"}
    salida = subprocess.run([sys.executable, "-c", "from equipo52_module.config import MLFLOW_TRACKING_URI as u; print(u)"],
                            cwd=PROJ_ROOT, env=entorno, capture_output=True, text=True, check=True)
    assert salida.stdout.strip() == "http://servidor-del-equipo:5000"

# ------------------------------------------------------------------ datos personales y datos crudos

COLUMNAS_PERSONALES = ["Usuario_Id", "Genero", "Año_de_nacimiento"]


def test_ingesta_no_lee_columnas_personales():
    from equipo52_module.config import COLUMNAS_VIAJES

    assert not set(COLUMNAS_PERSONALES) & set(COLUMNAS_VIAJES)


def test_no_se_versionan_datos(versionados):
    # El manifiesto (references/manifiesto_datos.csv) sí se versiona: trae huellas SHA-256, no viajes.
    datos = [ruta for ruta in versionados
             if ruta.startswith("data/") or (ruta.endswith((".parquet", ".csv")) and not ruta.startswith("references/"))]
    assert datos == [], f"Datos en Git: {datos}"


def test_notebooks_no_muestran_valores_personales(versionados):
    # Los notebooks pueden nombrar las columnas (p. ej. al listar encabezados), pero no mostrar sus valores:
    # una tabla impresa con Usuario_Id traería la columna seguida de números en la misma salida.
    patron = re.compile(r"Usuario_Id\s+Genero|Usuario_Id\s*\n\s*\d+\s+\d{5,}")
    con_valores = [ruta for ruta in versionados if ruta.endswith(".ipynb") and patron.search(texto_sin_imagenes(ruta))]
    assert con_valores == [], f"Notebooks con datos personales: {con_valores}"    