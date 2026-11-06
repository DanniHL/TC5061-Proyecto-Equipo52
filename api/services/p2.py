"""P2 · Carga el champion empaquetado en models/p2 y predice el riesgo de una estación."""

from functools import lru_cache
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from api.schemas.p2 import EntradaP2, SalidaP2

MODELO_DIR = Path(__file__).resolve().parents[2] / "models" / "p2"
CLASES = {0: "normal", 1: "sin_bicicletas", 2: "sin_anclajes"}
VARIABLES = list(EntradaP2.model_fields)  # mismo orden que la firma del modelo
# Tipos de la firma: integer → int32, long → int64, double → float64 y float → float32.
TIPOS = {"hora_del_dia": "int32", "dia_semana": "int32", "mes": "int32", "fin_de_semana": "int32",
         "capacidad": "int64", "latitud": "float64", "longitud": "float64"}


@lru_cache(maxsize=1)
def cargar_modelo():
    """Carga el modelo una sola vez y comprueba que la API y el modelo esperan las mismas variables."""
    firma = mlflow.models.Model.load(str(MODELO_DIR)).signature.inputs.input_names()
    if firma != VARIABLES:
        raise RuntimeError(f"El esquema de la API no coincide con la firma del modelo: {firma}")
    return mlflow.sklearn.load_model(str(MODELO_DIR))


def predecir(entrada: EntradaP2) -> SalidaP2:
    modelo = cargar_modelo()
    tabla = pd.DataFrame([entrada.model_dump()], columns=VARIABLES)
    tabla = tabla.astype({columna: TIPOS.get(columna, "float32") for columna in VARIABLES})
    probabilidades = modelo.predict_proba(tabla)[0]
    clase = int(modelo.classes_[np.argmax(probabilidades)])
    return SalidaP2(clase=clase, etiqueta=CLASES[clase],
                    probabilidades={CLASES[int(c)]: round(float(p), 4)
                                    for c, p in zip(modelo.classes_, probabilidades)})