"""API REST de los modelos del proyecto MiBici (Equipo 52)."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.routers import p1, p2, p3
from api.services.p1 import cargar_modelo as cargar_modelo_p1
from api.services.p2 import cargar_modelo as cargar_modelo_p2
from api.services.p3 import cargar_modelo as cargar_modelo_p3


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Los modelos se cargan al arrancar: si alguno falla, el servicio no levanta.
    cargar_modelo_p1()
    cargar_modelo_p2()
    cargar_modelo_p3()
    yield


app = FastAPI(title="MiBici · Equipo 52", version="1.0.0", lifespan=lifespan)

app.include_router(p1.router)
app.include_router(p2.router)
app.include_router(p3.router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"estado": "ok"}
