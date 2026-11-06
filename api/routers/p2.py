"""P2 · Endpoint de riesgo de estaciones."""

from fastapi import APIRouter

from api.schemas.p2 import EntradaP2, SalidaP2
from api.services.p2 import predecir

router = APIRouter(prefix="/p2", tags=["P2 · Riesgo de estaciones"])


@router.post("/predict", response_model=SalidaP2)
def predecir_riesgo(entrada: EntradaP2) -> SalidaP2:
    """Predice si la estación se quedará sin bicicletas o sin anclajes en las próximas 3 horas."""
    return predecir(entrada)