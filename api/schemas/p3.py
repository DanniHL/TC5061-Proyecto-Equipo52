"""P3 · Esquemas de entrada y salida de la API de grupos de estaciones.

Las 42 variables y sus tipos son los de la firma del modelo registrado
mibici-equipo52-p3-grupos@champion. Se reciben en su escala original (viajes por
mes, proporciones, minutos); el servicio aplica la estandarización con la que se
entrenó el K-Means antes de predecir.
"""

from pydantic import BaseModel, ConfigDict, Field, model_validator

TOLERANCIA_FIRMA = 0.02  # las proporciones pueden traer redondeo del cliente


class EntradaP3(BaseModel):
    """Perfil de uso de una estación, calculado con toda su historia de viajes."""

    model_config = ConfigDict(extra="forbid")  # rechaza campos que el modelo no conoce

    # Nivel de actividad (viajes por mes activo)
    salidas_por_mes: float = Field(ge=0)
    llegadas_por_mes: float = Field(ge=0)
    viajes_por_anclaje_mes: float = Field(ge=0)
    # Firma horaria: proporción de las salidas en cada hora del día (h00 a h23)
    h00: float = Field(ge=0, le=1)
    h01: float = Field(ge=0, le=1)
    h02: float = Field(ge=0, le=1)
    h03: float = Field(ge=0, le=1)
    h04: float = Field(ge=0, le=1)
    h05: float = Field(ge=0, le=1)
    h06: float = Field(ge=0, le=1)
    h07: float = Field(ge=0, le=1)
    h08: float = Field(ge=0, le=1)
    h09: float = Field(ge=0, le=1)
    h10: float = Field(ge=0, le=1)
    h11: float = Field(ge=0, le=1)
    h12: float = Field(ge=0, le=1)
    h13: float = Field(ge=0, le=1)
    h14: float = Field(ge=0, le=1)
    h15: float = Field(ge=0, le=1)
    h16: float = Field(ge=0, le=1)
    h17: float = Field(ge=0, le=1)
    h18: float = Field(ge=0, le=1)
    h19: float = Field(ge=0, le=1)
    h20: float = Field(ge=0, le=1)
    h21: float = Field(ge=0, le=1)
    h22: float = Field(ge=0, le=1)
    h23: float = Field(ge=0, le=1)
    # Firma semanal: proporción de las salidas en cada día (d0 = lunes … d6 = domingo)
    d0: float = Field(ge=0, le=1)
    d1: float = Field(ge=0, le=1)
    d2: float = Field(ge=0, le=1)
    d3: float = Field(ge=0, le=1)
    d4: float = Field(ge=0, le=1)
    d5: float = Field(ge=0, le=1)
    d6: float = Field(ge=0, le=1)
    # Rol en la red
    flujo_neto_por_mes: float = Field(description="Llegadas menos salidas por mes activo")
    balance: float = Field(ge=-1, le=1, description="Dirección del flujo, acotado en [-1, 1]")
    num_destinos: float = Field(ge=0, description="Estaciones distintas a las que salen viajes")
    # Tipo de viaje
    prop_viajes_redondos: float = Field(ge=0, le=1)
    duracion_mediana_min: float = Field(ge=0)
    # Evolución histórica
    meses_activa: float = Field(gt=0, description="Meses desde el primer viaje de la estación")
    share_viajes_ultimos_12m: float = Field(ge=0, le=1)
    crecimiento_vs_esperado: float = Field(ge=-1, le=1)

    @model_validator(mode="after")
    def revisar_firmas(self):
        """Las proporciones de cada firma suman 1; solo pueden ser 0 si la estación no tuvo salidas."""
        for nombre, columnas in (("horaria", [f"h{hora:02d}" for hora in range(24)]),
                                 ("semanal", [f"d{dia}" for dia in range(7)])):
            suma = sum(getattr(self, columna) for columna in columnas)
            if suma > TOLERANCIA_FIRMA and abs(suma - 1) > TOLERANCIA_FIRMA:
                raise ValueError(
                    f"la firma {nombre} debe sumar 1 (o 0 si la estación no registra salidas), "
                    f"no {suma:.3f}"
                )
        return self


class SalidaP3(BaseModel):
    grupo: int = Field(ge=0, le=3)
    etiqueta: str
    distancias: dict[str, float] = Field(
        description="Distancia al centroide de cada grupo; la menor corresponde al grupo asignado"
    )
