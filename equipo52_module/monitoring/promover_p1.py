
"""P1 · Promueve el challenger a champion si ganó ambas métricas del A/B."""

import json
import os
from pathlib import Path

from loguru import logger

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow  # noqa: E402
from mlflow import MlflowClient  # noqa: E402

from equipo52_module.config import MLFLOW_TRACKING_URI  # noqa: E402


NOMBRE_MODELO = "mibici-equipo52-p1-demanda"
AB_FILE = Path("reports/monitoreo/ab_p1.json")


def main():
    """Promueve el challenger si cumple la regla de decisión."""

    if not AB_FILE.exists():
        raise FileNotFoundError(f"No se encontró el reporte: {AB_FILE}")

    ab = json.loads(AB_FILE.read_text(encoding="utf-8"))

    resultado_mae = ab["mae"]
    resultado_rmse = ab["rmse"]

    if resultado_mae["gana"] != "B" or resultado_rmse["gana"] != "B":
        logger.warning(
            "El challenger no ganó ambas métricas. "
            "El champion se conserva sin cambios."
        )
        return

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    cliente = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    version_a = str(ab["champion_version"])
    version_b = str(ab["challenger_version"])

    champion_actual = str(
        cliente.get_model_version_by_alias(
            NOMBRE_MODELO, "champion"
        ).version
    )

    challenger_actual = str(
        cliente.get_model_version_by_alias(
            NOMBRE_MODELO, "challenger"
        ).version
    )

    if champion_actual != version_a:
        raise RuntimeError(
            f"El champion actual es la versión {champion_actual}, "
            f"no la {version_a} evaluada en el A/B."
        )

    if challenger_actual != version_b:
        raise RuntimeError(
            f"El challenger actual es la versión {challenger_actual}, "
            f"no la {version_b} evaluada en el A/B."
        )

    mae_ic = resultado_mae["ic_95"]
    rmse_ic = resultado_rmse["ic_95"]

    motivo = (
        f"A/B P1 en {ab['periodo_evaluacion']['inicio']} a "
        f"{ab['periodo_evaluacion']['fin']}. "
        f"MAE: {resultado_mae['metrica_b']:.4f} contra "
        f"{resultado_mae['metrica_a']:.4f} "
        f"(B-A={resultado_mae['diferencia']:+.4f}, "
        f"IC 95% {mae_ic[0]:+.4f} a {mae_ic[1]:+.4f}). "
        f"RMSE: {resultado_rmse['metrica_b']:.4f} contra "
        f"{resultado_rmse['metrica_a']:.4f} "
        f"(B-A={resultado_rmse['diferencia']:+.4f}, "
        f"IC 95% {rmse_ic[0]:+.4f} a {rmse_ic[1]:+.4f})."
    )

    # Conservar la versión anterior y promover la nueva.
    cliente.set_registered_model_alias(
        NOMBRE_MODELO, "anterior", version_a
    )
    cliente.set_registered_model_alias(
        NOMBRE_MODELO, "champion", version_b
    )
    cliente.delete_registered_model_alias(
        NOMBRE_MODELO, "challenger"
    )

    cliente.set_model_version_tag(
        NOMBRE_MODELO, version_b, "promovido_a_champion", motivo
    )
    cliente.set_model_version_tag(
        NOMBRE_MODELO, version_a, "reemplazado_por", f"versión {version_b}"
    )

    logger.success(
        f"champion → versión {version_b} · "
        f"anterior → versión {version_a}"
    )
    logger.info(motivo)


if __name__ == "__main__":
    main()