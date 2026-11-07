
"""P1 · Carga el modelo empaquetado y pronostica la demanda de una estación."""
from datetime import timedelta
from functools import lru_cache
from pathlib import Path
import mlflow
import pandas as pd
from api.schemas.p1 import EntradaP1, PronosticoDiarioP1, SalidaP1
import math

MODELO_DIR = Path(__file__).resolve().parents[2] / "models" / "p1"

VARIABLES = [
    "Viajes_dia_anterior",
    "Viajes_hace_7_dias",
    "Promedio_7_dias",
    "Fin_de_semana",
    "Dia_semana_sin",
    "Dia_semana_cos",
    "Mes_sin",
    "Mes_cos",
]

TIPOS = {
    "Viajes_dia_anterior": "float64",
    "Viajes_hace_7_dias": "float64",
    "Promedio_7_dias": "float64",
    "Fin_de_semana": "int64",
    "Dia_semana_sin": "float64",
    "Dia_semana_cos": "float64",
    "Mes_sin": "float64",
    "Mes_cos": "float64",
}

@lru_cache(maxsize=1)
def cargar_modelo():
    """Carga el modelo P1 y verifica que coincida con las variables esperadas."""
    firma = mlflow.models.Model.load(str(MODELO_DIR)).signature.inputs.input_names()
    if firma != VARIABLES:
        raise RuntimeError(
            f"El esquema de la API no coincide con la firma del modelo: {firma}"
        )
    return mlflow.sklearn.load_model(str(MODELO_DIR))

def construir_variables(historial: list[float], fecha) -> dict:
    """Construye las ocho variables que necesita el modelo P1."""
    dia_semana = fecha.weekday()
    mes = fecha.month
    return {
        "Viajes_dia_anterior": historial[-1],
        "Viajes_hace_7_dias": historial[0],
        "Promedio_7_dias": sum(historial) / 7,
        "Fin_de_semana": int(dia_semana >= 5),
        "Dia_semana_sin": math.sin(2 * math.pi * dia_semana / 7),
        "Dia_semana_cos": math.cos(2 * math.pi * dia_semana / 7),
        "Mes_sin": math.sin(2 * math.pi * (mes - 1) / 12),
        "Mes_cos": math.cos(2 * math.pi * (mes - 1) / 12),
    }

def predecir(entrada: EntradaP1) -> SalidaP1:
    """Genera siete pronósticos diarios de forma recursiva para una estación."""
    modelo = cargar_modelo()
    historial = entrada.viajes_ultimos_7_dias.copy()
    fecha_actual = entrada.fecha_ultimo_dato
    pronosticos = []

    for _ in range(7):
        fecha_objetivo = fecha_actual + timedelta(days=1)
        variables = construir_variables(historial, fecha_objetivo)
        tabla = pd.DataFrame([variables], columns=VARIABLES)
        tabla = tabla.astype(TIPOS)
        viajes_predichos = float(modelo.predict(tabla)[0])
        pronosticos.append(
            PronosticoDiarioP1(
                fecha=fecha_objetivo,
                viajes_predichos=viajes_predichos,
            )
        )

        # El pronóstico se incorpora al historial para calcular el día siguiente.
        historial.append(viajes_predichos)
        historial = historial[-7:]
        fecha_actual = fecha_objetivo

    return SalidaP1(
        estacion_id=entrada.estacion_id,
        pronosticos=pronosticos,
    )