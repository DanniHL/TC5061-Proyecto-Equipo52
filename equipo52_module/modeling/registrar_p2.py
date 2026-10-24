"""P2 · Registro del modelo elegido en el Model Registry de MLflow y su paso a producción.

La decisión se toma en el notebook 2.3-rr-p2-seleccion-modelo.ipynb, que compara las candidatas y
guarda los run_id elegidos en reports/p2_seleccion_modelo.json. Este script solo ejecuta esa
decisión: registra cada modelo como una versión de mibici-equipo52-p2-riesgo con su linaje, pone
los alias champion (producción) y challenger (siguiente mejor) y prueba que el champion se carga por
su nombre y alias.

Uso (con el servidor de MLflow encendido):
    python -m equipo52_module.modeling.registrar_p2
    python -m equipo52_module.modeling.registrar_p2 --verificar    # solo prueba la carga del champion
"""

import json
import os
from pathlib import Path

from loguru import logger

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")  # oculta un aviso de MLflow que no aplica aquí
import mlflow  # noqa: E402
from mlflow import MlflowClient  # noqa: E402
from mlflow.exceptions import MlflowException  # noqa: E402
import typer

from equipo52_module.config import MLFLOW_EXPERIMENT, MLFLOW_TRACKING_URI, PROJ_ROOT

app = typer.Typer()

NOMBRE_MODELO = "mibici-equipo52-p2-riesgo"
SELECCION_FILE = PROJ_ROOT / "reports" / "p2_seleccion_modelo.json"

DESCRIPCION_MODELO = """**P2 · Riesgo de que una estación de MiBici se quede sin bicicletas o sin anclajes libres en las próximas 3 horas.**

Clasificación en tres clases (normal, sin bicicletas, sin anclajes) por estación y hora. Métrica principal: F1 de las dos
clases de falla. El modelo se entrena con `equipo52_module/modeling/train_p2.py` desde el MLflow Project del repositorio
TC5061-Proyecto-Equipo52 (Equipo 52).

Ciclo de vida con aliases: `champion` es la versión en producción y `challenger` la siguiente mejor. Se carga con
`models:/mibici-equipo52-p2-riesgo@champion`."""


def validar_corrida(cliente: MlflowClient, run_id: str, rol: str):
    """Comprueba que la corrida elegida sea una candidata terminada de P2 y devuelve la corrida y su modelo."""
    corrida = cliente.get_run(run_id)
    etiquetas = corrida.data.tags
    problemas = []
    if corrida.info.status != "FINISHED":
        problemas.append(f"su estado es {corrida.info.status}")
    if etiquetas.get("question") != "p2":
        problemas.append("no es de P2")
    if etiquetas.get("stage_of_work") != "candidate":
        problemas.append(f"su etapa es {etiquetas.get('stage_of_work')}, no candidate")
    modelos = mlflow.search_logged_models(experiment_ids=[corrida.info.experiment_id],
                                          filter_string=f"source_run_id = '{run_id}'", output_format="list")
    if not modelos:
        problemas.append("no tiene modelo registrado con log_model")
    if problemas:
        raise typer.BadParameter(f"La corrida {rol} {run_id} no se puede registrar: {'; '.join(problemas)}")
    return corrida, modelos[0]


def etiquetas_de_version(corrida, rol: str, eleccion: dict, decision: dict, experimento: str) -> dict:
    """Linaje y desempeño de la versión: de dónde salió el modelo y cómo le fue contra el baseline."""
    p, m, t = corrida.data.params, corrida.data.metrics, corrida.data.tags
    return {
        "rol_al_registrar": rol, "corrida_origen": corrida.info.run_id,
        "nombre_corrida": t.get("mlflow.runName", ""), "experimento": experimento,
        "commit": t.get("commit") or t.get("mlflow.source.git.commit", ""),
        "fuente": t.get("mlflow.source.name", ""), "version_datos": p.get("version_datos", ""),
        "huella_manifiesto": p.get("huella_manifiesto", ""),
        "periodo": p.get("periodo_script") or p.get("periodo", ""), "model_family": t.get("model_family", ""),
        "val_f1_fallas": f"{m.get('val_f1_fallas', float('nan')):.4f}",
        "test_f1_fallas": f"{m.get('test_f1_fallas', float('nan')):.4f}",
        "mejora_test_contra_baseline": f"{eleccion['mejora_test_puntos']:+.4f}",
        "segundos_entrenamiento": f"{m.get('segundos_entrenamiento', 0):.0f}",
        "seleccion": decision["fecha"], "validation_status": "approved",
    }


def descripcion_de_version(corrida, rol: str, eleccion: dict, decision: dict) -> str:
    """Descripción en Markdown de la versión: qué modelo es, sus métricas y por qué se eligió."""
    base = decision["baseline"]
    return (f"**{rol.capitalize()} de P2:** `{corrida.data.tags.get('mlflow.runName')}` "
            f"({corrida.data.tags.get('model_family')}).\n\n"
            f"F1 de fallas: validación {eleccion['val_f1_fallas']:.3f} y prueba {eleccion['test_f1_fallas']:.3f} "
            f"(baseline: {base['val_f1_fallas']:.3f} y {base['test_f1_fallas']:.3f}); "
            f"mejora en prueba {eleccion['mejora_test_puntos']:+.3f}.\n\n"
            f"Regla de selección: {decision['regla']}.\n\nJustificación: {decision['justificacion']}")


def version_de_la_corrida(cliente: MlflowClient, run_id: str):
    """Versión ya registrada para esa corrida, si existe (para no duplicar al volver a ejecutar el script)."""
    versiones = cliente.search_model_versions(f"name = '{NOMBRE_MODELO}' and run_id = '{run_id}'")
    return max(versiones, key=lambda v: int(v.version)) if versiones else None


def verificar_champion(cliente: MlflowClient, run_id_esperado: str | None = None) -> None:
    """Carga el champion por nombre y alias y predice con su ejemplo de entrada: la prueba de que está en producción."""
    uri = f"models:/{NOMBRE_MODELO}@champion"
    version = cliente.get_model_version_by_alias(NOMBRE_MODELO, "champion")
    if run_id_esperado and version.run_id != run_id_esperado:
        raise RuntimeError(f"champion apunta a la corrida {version.run_id}, no a la elegida {run_id_esperado}")
    modelo = mlflow.pyfunc.load_model(uri)
    ejemplo = modelo.input_example
    prediccion = modelo.predict(ejemplo)
    logger.success(f"Carga correcta de {uri} → versión {version.version} · corrida de origen {version.run_id}")
    logger.info(f"Entrada esperada (firma): {modelo.metadata.signature.inputs.input_names()}")
    logger.info(f"Predicción para el ejemplo de entrada ({len(ejemplo)} filas): {[int(p) for p in prediccion]} "
                "(0 = normal, 1 = sin bicicletas, 2 = sin anclajes)")


@app.command()
def main(seleccion: Path = typer.Option(SELECCION_FILE, help="decisión guardada por el notebook 2.3"),
         verificar: bool = typer.Option(False, help="solo probar la carga del champion, sin registrar")):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    cliente = MlflowClient()
    if verificar:
        verificar_champion(cliente)
        return

    decision = json.loads(Path(seleccion).read_text())
    if decision.get("pregunta") != "p2" or decision.get("modelo_registrado") != NOMBRE_MODELO:
        raise typer.BadParameter(f"{seleccion} no es una decisión de P2 para {NOMBRE_MODELO}")
    roles = {rol: decision[rol] for rol in ["champion", "challenger"] if decision.get(rol)}
    # Se validan todas las corridas antes de registrar nada.
    validadas = {rol: validar_corrida(cliente, eleccion["run_id"], rol) for rol, eleccion in roles.items()}

    try:
        cliente.create_registered_model(NOMBRE_MODELO)
        logger.info(f"Modelo registrado {NOMBRE_MODELO} creado")
    except MlflowException:
        logger.info(f"Modelo registrado {NOMBRE_MODELO} ya existía: se agregan o reutilizan versiones")
    cliente.update_registered_model(NOMBRE_MODELO, description=DESCRIPCION_MODELO)
    for clave, valor in {"equipo": "52", "pregunta": "P2", "question": "p2", "tarea": "clasificacion",
                         "metrica_principal": "test_f1_fallas"}.items():
        cliente.set_registered_model_tag(NOMBRE_MODELO, clave, valor)

    for rol, (corrida, modelo) in validadas.items():
        eleccion = roles[rol]
        version = version_de_la_corrida(cliente, corrida.info.run_id)
        if version is None:
            version = mlflow.register_model(f"models:/{modelo.model_id}", NOMBRE_MODELO)
            logger.info(f"{rol}: versión {version.version} creada desde la corrida {corrida.info.run_id}")
        else:
            logger.info(f"{rol}: la corrida ya estaba registrada como versión {version.version}; se reutiliza")
        for clave, valor in etiquetas_de_version(corrida, rol, eleccion, decision, MLFLOW_EXPERIMENT).items():
            cliente.set_model_version_tag(NOMBRE_MODELO, version.version, clave, str(valor))
        cliente.update_model_version(NOMBRE_MODELO, version.version,
                                     description=descripcion_de_version(corrida, rol, eleccion, decision))
        cliente.set_registered_model_alias(NOMBRE_MODELO, rol, version.version)
        logger.success(f"Alias {rol} → {NOMBRE_MODELO} versión {version.version} "
                       f"({corrida.data.tags.get('mlflow.runName')})")

    verificar_champion(cliente, run_id_esperado=decision["champion"]["run_id"])


if __name__ == "__main__":
    app()
