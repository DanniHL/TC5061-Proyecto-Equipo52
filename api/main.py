"""API REST de los modelos del proyecto MiBici (Equipo 52)."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.routers import p2
from api.services.p2 import cargar_modelo as cargar_modelo_p2


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Los modelos se cargan al arrancar: si alguno falla, el servicio no levanta.
    cargar_modelo_p2()
    yield


app = FastAPI(title="MiBici · Equipo 52", version="1.0.0", lifespan=lifespan)
app.include_router(p2.router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"estado": "ok"}