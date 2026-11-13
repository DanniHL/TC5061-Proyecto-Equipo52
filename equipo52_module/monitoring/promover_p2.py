"""P2 · Promueve el challenger a champion si ganó el A/B testing.

Lee reports/monitoreo/ab_p2.json y, solo si la regla de decisión dio "B", mueve los alias:
champion a la versión B, anterior a la versión A (para regresar a ella si hace falta) y quita el
alias challenger. No borra ninguna versión; el motivo queda como etiqueta en MLflow.

Uso (con el servidor de MLflow encendido):
    python -m equipo52_module.monitoring.promover_p2
"""

import json
import os

from loguru import logger

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")  # oculta un aviso de MLflow que no aplica aquí
import mlflow  # noqa: E402
from mlflow import MlflowClient  # noqa: E402
import typer  # noqa: E402

from equipo52_module.config import MLFLOW_TRACKING_URI  # noqa: E402
from equipo52_module.modeling.registrar_p2 import NOMBRE_MODELO  # noqa: E402
from equipo52_module.monitoring.drift import MONITOREO_DIR  # noqa: E402

app = typer.Typer()

AB_FILE = MONITOREO_DIR / "ab_p2.json"


@app.command()
def main():
    ab = json.loads(AB_FILE.read_text())
    resultado = ab["resultado"]
    if resultado["gana"] != "B":
        logger.warning("El A/B no dio como ganador a B: el champion se queda igual")
        return

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    cliente = MlflowClient()
    version_a, version_b = ab["version_a"], ab["version_b"]
    champion_actual = cliente.get_model_version_by_alias(NOMBRE_MODELO, "champion").version
    if champion_actual != version_a:
        raise RuntimeError(f"El champion actual es la versión {champion_actual}, "
                           f"no la {version_a} que se comparó en el A/B")

    ic = resultado["ic_95"]
    motivo = (f"A/B en {ab['periodo_evaluacion']}: F1 de fallas {resultado['metrica_b']:.4f} contra "
              f"{resultado['metrica_a']:.4f} (B - A = {resultado['diferencia']:+.4f}, "
              f"IC 95% {ic[0]:+.4f} a {ic[1]:+.4f})")
    cliente.set_registered_model_alias(NOMBRE_MODELO, "anterior", version_a)
    cliente.set_registered_model_alias(NOMBRE_MODELO, "champion", version_b)
    cliente.delete_registered_model_alias(NOMBRE_MODELO, "challenger")
    cliente.set_model_version_tag(NOMBRE_MODELO, version_b, "promovido_a_champion", motivo)
    cliente.set_model_version_tag(NOMBRE_MODELO, version_a, "reemplazado_por", f"versión {version_b}")
    logger.success(f"champion → versión {version_b} · anterior → versión {version_a} · {motivo}")


if __name__ == "__main__":
    app()