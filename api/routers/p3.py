"""P3 · Endpoint de grupos de estaciones."""

from fastapi import APIRouter

from api.schemas.p3 import EntradaP3, SalidaP3
from api.services.p3 import predecir

router = APIRouter(prefix="/p3", tags=["P3 · Grupos de estaciones"])


@router.post("/predict", response_model=SalidaP3)
def asignar_grupo(entrada: EntradaP3) -> SalidaP3:
    """Asigna una estación a su grupo operativo a partir de su perfil de uso (42 características)."""
    return predecir(entrada)
