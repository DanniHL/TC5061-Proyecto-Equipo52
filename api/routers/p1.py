"""P1 · Endpoint de pronóstico de demanda por estación."""

from fastapi import APIRouter
from api.schemas.p1 import EntradaP1, SalidaP1
from api.services.p1 import predecir

router = APIRouter(
    prefix="/p1",
    tags=["P1 · Pronóstico de demanda"],
)

@router.post("/predict", response_model=SalidaP1)
def predecir_demanda(entrada: EntradaP1) -> SalidaP1:
    """Genera el pronóstico de demanda para los siguientes siete días."""
    return predecir(entrada)