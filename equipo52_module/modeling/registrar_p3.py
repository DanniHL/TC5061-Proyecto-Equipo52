"""P3 · Registro del modelo elegido en el Model Registry de MLflow y su paso a producción.

Compara las corridas oficiales de P3 registradas en mibici-equipo52, guarda o lee la decisión en
reports/p3_seleccion_modelo.json, registra el modelo elegido y su alternativa como versiones de
mibici-equipo52-p3-grupos con su linaje completo, asigna los alias champion (producción) y
challenger (siguiente mejor) y comprueba que el champion se carga por su nombre y alias.

Uso (con el servidor de MLflow encendido):
    python -m equipo52_module.modeling.registrar_p3
    python -m equipo52_module.modeling.registrar_p3 --verificar    # solo prueba la carga del champion
"""

from datetime import datetime
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

NOMBRE_MODELO = "mibici-equipo52-p3-grupos"
SELECCION_FILE = PROJ_ROOT / "reports" / "p3_seleccion_modelo.json"

REGLA_SELECCION = (
    "champion: K-Means k=4 sobre las 42 características de uso (segundo punto de inflexión de silueta "
    "que iguala el índice de Davies-Bouldin del óptimo global y desagrega la dirección del flujo y la "
    "franja horaria pico en 4 perfiles accionables para planeación y rebalanceo); "
    "challenger: K-Means k=2 sobre las 42 características (máximo matemático global de silueta y "
    "estabilidad para segmentación macro de la red)"
)

JUSTIFICACION_SELECCION = (
    "K-Means con k=4 (p3-kmeans-k4) multiplica por 5.6 la silueta en el espacio de uso respecto al "
    "baseline geográfico (0.105 contra 0.019, +457.7 %; 0.131 contra 0.039 en las 371 estaciones consolidadas), "
    "alcanza el menor índice de Davies-Bouldin en validación (2.075 contra 2.982 del baseline) e iguala al "
    "óptimo global en la red completa (2.309 contra 3.267 del baseline), mantiene alta estabilidad temporal "
    "entre eras (ARI = 0.571) y por remuestreo bootstrap (ARI = 0.841), y es el único corte compacto que "
    "separa los 4 regímenes operativos que necesita MiBici (núcleo central receptor, polos de empleo con "
    "vaciado vespertino 17-19 h, zonas residenciales emisoras matutinas 7-9 h y expansión periférica "
    "recreativa de fin de semana). K-Means con k=2 (p3-kmeans-k2) queda como challenger por tener la mayor "
    "silueta global (0.151) y estabilidad bootstrap (0.952) para diagnóstico macro."
)

DESCRIPCION_MODELO = """**P3 · Agrupamiento de estaciones de MiBici por patrones de uso y rol en la red.**

Segmentación de las 469 estaciones activas a partir de 42 características estandarizadas (nivel de actividad mensual,
firma horaria de 24 h, firma semanal de 7 días, rol origen/destino, viajes redondos y evolución). Métricas principales:
coeficiente de silueta en el espacio de uso (`val_silhouette` en las 371 estaciones consolidadas y `hist_silhouette` en
las 469 estaciones) y estabilidad entre eras (`estabilidad_eras_ari`) y por remuestreo (`estabilidad_bootstrap_ari`).
El modelo se entrena con `equipo52_module/modeling/train_p3.py` desde el MLflow Project del repositorio
TC5061-Proyecto-Equipo52 (Equipo 52).

Ciclo de vida con aliases: `champion` es la versión en producción (`p3-kmeans-k4`, 4 perfiles operativos de planeación)
y `challenger` la siguiente mejor (`p3-kmeans-k2`, partición macro de máxima silueta). Se carga con
`models:/mibici-equipo52-p3-grupos@champion`."""


def construir_seleccion_desde_mlflow(cliente: MlflowClient) -> dict:
    """Consulta las corridas recientes de P3 en mibici-equipo52 y arma reports/p3_seleccion_modelo.json."""
    experimento = cliente.get_experiment_by_name(MLFLOW_EXPERIMENT)
    if experimento is None:
        raise RuntimeError(f"No existe el experimento {MLFLOW_EXPERIMENT} en {MLFLOW_TRACKING_URI}")

    corridas = mlflow.search_runs(
        [experimento.experiment_id],
        filter_string="tags.question = 'p3' and attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
    )
    if corridas.empty:
        raise RuntimeError("No se encontraron corridas terminadas de P3 en MLflow.")

    def buscar_corrida(nombre: str):
        sub = corridas[corridas["tags.mlflow.runName"] == nombre]
        if sub.empty:
            raise RuntimeError(f"No se encontró la corrida '{nombre}' en {MLFLOW_EXPERIMENT}.")
        return sub.iloc[0]

    base = buscar_corrida("p3-geo-k5")
    champ = buscar_corrida("p3-kmeans-k4")
    chall = buscar_corrida("p3-kmeans-k2")

    sil_base_val = float(base["metrics.val_silhouette"])
    sil_base_hist = float(base["metrics.hist_silhouette"])

    decision = {
        "pregunta": "p3",
        "modelo_registrado": NOMBRE_MODELO,
        "fecha": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "experimento": MLFLOW_EXPERIMENT,
        "regla": REGLA_SELECCION,
        "baseline": {
            "run_id": str(base["run_id"]),
            "corrida": "p3-geo-k5",
            "val_silhouette": round(sil_base_val, 4),
            "hist_silhouette": round(sil_base_hist, 4),
            "estabilidad_eras_ari": round(float(base["metrics.estabilidad_eras_ari"]), 4),
            "estabilidad_bootstrap_ari": round(float(base["metrics.estabilidad_bootstrap_ari"]), 4),
        },
        "champion": {
            "run_id": str(champ["run_id"]),
            "corrida": "p3-kmeans-k4",
            "familia": "kmeans",
            "k": 4,
            "val_silhouette": round(float(champ["metrics.val_silhouette"]), 4),
            "hist_silhouette": round(float(champ["metrics.hist_silhouette"]), 4),
            "estabilidad_eras_ari": round(float(champ["metrics.estabilidad_eras_ari"]), 4),
            "estabilidad_bootstrap_ari": round(float(champ["metrics.estabilidad_bootstrap_ari"]), 4),
            "mejora_test_puntos": round(float(champ["metrics.hist_silhouette"]) - sil_base_hist, 4),
            "mejora_test_pct": round((float(champ["metrics.hist_silhouette"]) / sil_base_hist - 1) * 100, 1),
        },
        "challenger": {
            "run_id": str(chall["run_id"]),
            "corrida": "p3-kmeans-k2",
            "familia": "kmeans",
            "k": 2,
            "val_silhouette": round(float(chall["metrics.val_silhouette"]), 4),
            "hist_silhouette": round(float(chall["metrics.hist_silhouette"]), 4),
            "estabilidad_eras_ari": round(float(chall["metrics.estabilidad_eras_ari"]), 4),
            "estabilidad_bootstrap_ari": round(float(chall["metrics.estabilidad_bootstrap_ari"]), 4),
            "mejora_test_puntos": round(float(chall["metrics.hist_silhouette"]) - sil_base_hist, 4),
            "mejora_test_pct": round((float(chall["metrics.hist_silhouette"]) / sil_base_hist - 1) * 100, 1),
        },
        "justificacion": JUSTIFICACION_SELECCION,
    }
    SELECCION_FILE.parent.mkdir(parents=True, exist_ok=True)
    SELECCION_FILE.write_text(json.dumps(decision, indent=2, ensure_ascii=False))
    logger.info(f"Archivo de selección actualizado en {SELECCION_FILE}")
    return decision


def validar_corrida(cliente: MlflowClient, run_id: str, rol: str):
    """Comprueba que la corrida elegida sea una candidata terminada de P3 y devuelve la corrida y su modelo."""
    corrida = cliente.get_run(run_id)
    etiquetas = corrida.data.tags
    problemas = []
    if corrida.info.status != "FINISHED":
        problemas.append(f"su estado es {corrida.info.status}")
    if etiquetas.get("question") != "p3":
        problemas.append("no es de P3")
    if etiquetas.get("stage_of_work") != "candidate":
        problemas.append(f"su etapa es {etiquetas.get('stage_of_work')}, no candidate")
    modelos = mlflow.search_logged_models(
        experiment_ids=[corrida.info.experiment_id],
        filter_string=f"source_run_id = '{run_id}'",
        output_format="list",
    )
    if not modelos:
        problemas.append("no tiene modelo registrado con log_model")
    if problemas:
        raise typer.BadParameter(f"La corrida {rol} {run_id} no se puede registrar: {'; '.join(problemas)}")
    return corrida, modelos[0]


def etiquetas_de_version(corrida, rol: str, eleccion: dict, decision: dict, experimento: str) -> dict:
    """Linaje y desempeño de la versión: de dónde salió el modelo y cómo le fue contra el baseline."""
    p, m, t = corrida.data.params, corrida.data.metrics, corrida.data.tags
    return {
        "rol_al_registrar": rol,
        "corrida_origen": corrida.info.run_id,
        "nombre_corrida": t.get("mlflow.runName", ""),
        "experimento": experimento,
        "commit": t.get("commit") or t.get("mlflow.source.git.commit", ""),
        "fuente": t.get("mlflow.source.name", ""),
        "version_datos": p.get("version_datos", ""),
        "huella_manifiesto": p.get("huella_manifiesto", ""),
        "model_family": t.get("model_family", ""),
        "k": str(p.get("k_elegido") or p.get("k", "")),
        "val_silhouette": f"{m.get('val_silhouette', float('nan')):.4f}",
        "hist_silhouette": f"{m.get('hist_silhouette', float('nan')):.4f}",
        "estabilidad_eras_ari": f"{m.get('estabilidad_eras_ari', float('nan')):.4f}",
        "estabilidad_bootstrap_ari": f"{m.get('estabilidad_bootstrap_ari', float('nan')):.4f}",
        "mejora_test_contra_baseline": f"{eleccion['mejora_test_puntos']:+.4f}",
        "segundos_entrenamiento": f"{m.get('segundos_entrenamiento', 0):.1f}",
        "seleccion": decision["fecha"],
        "validation_status": "approved",
    }


def descripcion_de_version(corrida, rol: str, eleccion: dict, decision: dict) -> str:
    """Descripción en Markdown de la versión: qué modelo es, sus métricas y por qué se eligió."""
    base = decision["baseline"]
    return (
        f"**{rol.capitalize()} de P3:** `{corrida.data.tags.get('mlflow.runName')}` "
        f"({corrida.data.tags.get('model_family')}, k = {eleccion['k']}).\n\n"
        f"Silueta en el espacio de uso: validación (consolidadas) {eleccion['val_silhouette']:.3f} y "
        f"red completa {eleccion['hist_silhouette']:.3f} "
        f"(baseline geográfico: {base['val_silhouette']:.3f} y {base['hist_silhouette']:.3f}); "
        f"mejora en red completa {eleccion['mejora_test_puntos']:+.3f} ({eleccion['mejora_test_pct']:+.1f} %). "
        f"Estabilidad entre eras ARI = {eleccion['estabilidad_eras_ari']:.3f} y bootstrap ARI = "
        f"{eleccion['estabilidad_bootstrap_ari']:.3f}.\n\n"
        f"Regla de selección: {decision['regla']}.\n\nJustificación: {decision['justificacion']}"
    )


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
    logger.info(
        f"Entrada esperada (firma): {len(modelo.metadata.signature.inputs.input_names())} características "
        f"({', '.join(modelo.metadata.signature.inputs.input_names()[:6])}, ...)"
    )
    logger.info(
        f"Predicción para el ejemplo de entrada ({len(ejemplo)} estaciones): {[int(p) for p in prediccion]} "
        "(0 = núcleo receptor, 1 = empleo vespertino, 2 = residencial matutino, 3 = periférico/recreativo)"
    )


@app.command()
def main(seleccion: Path = typer.Option(SELECCION_FILE, help="decisión guardada en reports/p3_seleccion_modelo.json"),
         verificar: bool = typer.Option(False, help="solo probar la carga del champion, sin registrar")):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    cliente = MlflowClient()
    if verificar:
        verificar_champion(cliente)
        return

    decision = None
    if Path(seleccion).exists():
        try:
            decision_leida = json.loads(Path(seleccion).read_text())
            cliente.get_run(decision_leida["champion"]["run_id"])
            decision = decision_leida
        except Exception:
            decision = None
    if decision is None:
        decision = construir_seleccion_desde_mlflow(cliente)

    if decision.get("pregunta") != "p3" or decision.get("modelo_registrado") != NOMBRE_MODELO:
        raise typer.BadParameter(f"{seleccion} no es una decisión de P3 para {NOMBRE_MODELO}")

    roles = {rol: decision[rol] for rol in ["champion", "challenger"] if decision.get(rol)}
    validadas = {rol: validar_corrida(cliente, eleccion["run_id"], rol) for rol, eleccion in roles.items()}

    try:
        cliente.create_registered_model(NOMBRE_MODELO)
        logger.info(f"Modelo registrado {NOMBRE_MODELO} creado")
    except MlflowException:
        logger.info(f"Modelo registrado {NOMBRE_MODELO} ya existía: se agregan o reutilizan versiones")
    cliente.update_registered_model(NOMBRE_MODELO, description=DESCRIPCION_MODELO)
    for clave, valor in {
        "equipo": "52",
        "pregunta": "P3",
        "question": "p3",
        "tarea": "agrupamiento",
        "metrica_principal": "hist_silhouette",
    }.items():
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
        cliente.update_model_version(
            NOMBRE_MODELO,
            version.version,
            description=descripcion_de_version(corrida, rol, eleccion, decision),
        )
        cliente.set_registered_model_alias(NOMBRE_MODELO, rol, version.version)
        logger.success(
            f"Alias {rol} → {NOMBRE_MODELO} versión {version.version} "
            f"({corrida.data.tags.get('mlflow.runName')})"
        )

    verificar_champion(cliente, run_id_esperado=decision["champion"]["run_id"])


if __name__ == "__main__":
    app()
