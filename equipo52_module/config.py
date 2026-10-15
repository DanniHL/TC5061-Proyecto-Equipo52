from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Acuerdos del Equipo 52 sobre los datos de MiBici
# Se usan en los notebooks y scripts de las tres preguntas para trabajar con los mismos datos.

# Datos descargados de https://mibici.net/es/datos-abiertos/ (no se suben a GitHub).
VERSION_DATOS = "2026-09"
NOMENCLATURA_FILE = RAW_DATA_DIR / "nomenclatura_2026_09.csv"

# Periodo de análisis: historia completa publicada por MiBici.
FECHA_INICIO = "2014-12-01"
FECHA_FIN = "2026-09-30"

# Columnas de viajes que se leen. Se excluyen Usuario_Id, Genero y Año_de_nacimiento
# porque son datos personales y no se necesitan para las preguntas del proyecto.
COLUMNAS_VIAJES = ["Viaje_Id", "Inicio_del_viaje", "Fin_del_viaje", "Origen_Id", "Destino_Id"]

# Estaciones dadas de baja: nombre con prefijo "ex-" en la nomenclatura (4 de ellas con capacidad 0).
# Las estaciones 124, 405 y 407 no aparecen en la nomenclatura y quedan fuera al unir con ella.
ESTACIONES_EXCLUIDAS = [22, 97, 99, 109, 113, 114, 116, 138, 147, 149, 223, 240, 250, 310]

# MLflow
MLFLOW_EXPERIMENT = "mibici-equipo52"
SEED = 42

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
