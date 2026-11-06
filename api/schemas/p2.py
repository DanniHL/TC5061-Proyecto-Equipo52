"""P2 · Esquemas de entrada y salida de la API de riesgo de estaciones.

Las 13 variables y sus tipos son los de la firma del modelo registrado
mibici-equipo52-p2-riesgo@champion.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SEMANAS_HISTORIAL = 4  # semanas que revisa agregar_historial_semanas en features.py


class EntradaP2(BaseModel):
    """Lo que se sabe de una estación al empezar una hora."""

    model_config = ConfigDict(extra="forbid")  # rechaza campos que el modelo no conoce

    hora_del_dia: int = Field(ge=0, le=23)
    dia_semana: int = Field(ge=0, le=6, description="0 = lunes … 6 = domingo")
    mes: int = Field(ge=1, le=12)
    fin_de_semana: Literal[0, 1]
    capacidad: int = Field(gt=0, le=100, description="Anclajes de la estación")
    # Área Metropolitana de Guadalajara, con margen sobre las estaciones del catálogo 2026-09.
    latitud: float = Field(ge=20.55, le=20.85)
    longitud: float = Field(ge=-103.50, le=-103.20)
    # Pueden faltar en estaciones nuevas: el modelo HGB acepta valores vacíos.
    salidas_hora_anterior: float | None = Field(default=None, ge=0)
    llegadas_hora_anterior: float | None = Field(default=None, ge=0)
    salidas_3_horas_anteriores: float | None = Field(default=None, ge=0)
    llegadas_3_horas_anteriores: float | None = Field(default=None, ge=0)
    semanas_sin_bicicletas: float | None = Field(default=None, ge=0, le=SEMANAS_HISTORIAL)
    semanas_sin_anclajes: float | None = Field(default=None, ge=0, le=SEMANAS_HISTORIAL)

    @model_validator(mode="after")
    def revisar_consistencia(self):
        if self.fin_de_semana != int(self.dia_semana >= 5):
            raise ValueError("fin_de_semana debe ser 1 solo si dia_semana es 5 (sábado) o 6 (domingo)")
        if (self.semanas_sin_bicicletas or 0) + (self.semanas_sin_anclajes or 0) > SEMANAS_HISTORIAL:
            raise ValueError(f"las semanas con falla no pueden sumar más de {SEMANAS_HISTORIAL}")
        return self


class SalidaP2(BaseModel):
    clase: Literal[0, 1, 2]
    etiqueta: Literal["normal", "sin_bicicletas", "sin_anclajes"]
    probabilidades: dict[str, float]