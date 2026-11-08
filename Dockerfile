# Imagen de la API REST de los modelos MiBici (Equipo 52).
# Empaqueta FastAPI y los tres modelos champion exportados del Model Registry
# (models/p1, models/p2 y models/p3), para servir P1, P2 y P3 sin necesidad de un
# servidor de MLflow encendido.

FROM python:3.12-slim

# El servicio corre con un usuario sin privilegios dentro del contenedor.
RUN useradd --create-home --shell /usr/sbin/nologin mibici
WORKDIR /app

# Primero solo las dependencias: mientras no cambien, Docker reutiliza esta capa.
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY api/ ./api/
COPY models/ ./models/

USER mibici
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
