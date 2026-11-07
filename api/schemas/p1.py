"""P1 · Esquemas de entrada y salida de la API de pronóstico de demanda."""
from datetime import date
from pydantic import BaseModel, ConfigDict, Field, field_validator

class EntradaP1(BaseModel):
    """Datos históricos de una estación para generar el pronóstico de demanda."""

    model_config = ConfigDict(extra="forbid")
    estacion_id: int = Field(gt=0, description="Identificador de la estación MiBici")
    fecha_ultimo_dato: date = Field(
        description="Fecha del último día disponible en el historial"
    )

    viajes_ultimos_7_dias: list[float] = Field(
        min_length=7,
        max_length=7,
        description=(
            "Viajes diarios de los siete días consecutivos anteriores, "
            "ordenados del más antiguo al más reciente"
        ),
    )

    @field_validator("viajes_ultimos_7_dias")
    @classmethod
    def validar_viajes(cls, valores: list[float]) -> list[float]:
        if any(valor < 0 for valor in valores):
            raise ValueError("Los valores de viajes no pueden ser negativos")
        return valores

class PronosticoDiarioP1(BaseModel):
    """Pronóstico de demanda para un día."""
    fecha: date
    viajes_predichos: float

class SalidaP1(BaseModel):
    """Resultado del pronóstico de demanda de siete días."""
    estacion_id: int
    pronosticos: list[PronosticoDiarioP1] = Field(
        min_length=7,
        max_length=7,
        description="Pronósticos diarios para los siguientes siete días",
    )