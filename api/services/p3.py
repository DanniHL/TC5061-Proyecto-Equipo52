"""P3 · Carga el champion empaquetado en models/p3 y asigna la estación a su grupo operativo.

El K-Means registrado se entrenó sobre las 42 características estandarizadas, así que el
servicio aplica la misma transformación (media y desviación guardadas en escalador.json)
antes de predecir. La respuesta incluye la distancia al centroide de cada grupo.
"""

import json
from functools import lru_cache
from pathlib import Path

import mlflow
import pandas as pd

from api.schemas.p3 import EntradaP3, SalidaP3

MODELO_DIR = Path(__file__).resolve().parents[2] / "models" / "p3"
ESCALADOR_FILE = MODELO_DIR / "escalador.json"
GRUPOS = {0: "nucleo_receptor", 1: "empleo_vespertino",
          2: "residencial_matutino", 3: "periferico_recreativo"}
VARIABLES = list(EntradaP3.model_fields)  # mismo orden que la firma del modelo


@lru_cache(maxsize=1)
def cargar_modelo():
    """Carga el modelo y su escalador una sola vez y comprueba que coincidan con el esquema de la API."""
    firma = mlflow.models.Model.load(str(MODELO_DIR)).signature.inputs.input_names()
    if firma != VARIABLES:
        raise RuntimeError(f"El esquema de la API no coincide con la firma del modelo: {firma}")
    escalador = json.loads(ESCALADOR_FILE.read_text())
    if escalador["variables"] != VARIABLES:
        raise RuntimeError("escalador.json no corresponde a la firma del modelo")
    modelo = mlflow.sklearn.load_model(str(MODELO_DIR))
    media = pd.Series({variable: escalador["media"][variable] for variable in VARIABLES})
    escala = pd.Series({variable: escalador["escala"][variable] for variable in VARIABLES})
    return modelo, media, escala


def predecir(entrada: EntradaP3) -> SalidaP3:
    modelo, media, escala = cargar_modelo()
    tabla = pd.DataFrame([entrada.model_dump()], columns=VARIABLES).astype("float64")
    estandarizada = (tabla - media) / escala
    grupo = int(modelo.predict(estandarizada)[0])
    distancias = modelo.transform(estandarizada)[0]
    return SalidaP3(grupo=grupo, etiqueta=GRUPOS[grupo],
                    distancias={GRUPOS[g]: round(float(d), 4)
                                for g, d in enumerate(distancias)})
