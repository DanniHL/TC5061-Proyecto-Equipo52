"""P2 · Entrenamiento y registro en MLflow del modelo de riesgo por estación.

Pregunta: ¿Una estación estará en riesgo de quedarse sin bicicletas o sin anclajes libres en
las próximas 3 horas? Cada corrida entrena con un periodo (historia completa o Post-Covid),
se valida con octubre de 2025 a marzo de 2026, se prueba con abril a septiembre de 2026 y
registra parámetros, métricas y artefactos en MLflow.

Uso:
    python equipo52_module/modeling/train_p2.py --modelo base --periodo completo
    python equipo52_module/modeling/train_p2.py --modelo hgb --periodo completo
    python equipo52_module/modeling/train_p2.py --modelo hgb --periodo postcovid
    python equipo52_module/modeling/train_p2.py --modelo hgb --periodo completo --balanceado
    python equipo52_module/modeling/train_p2.py --modelo hgb --params "max_iter=300,learning_rate=0.05"
"""

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from loguru import logger
import matplotlib

matplotlib.use("Agg")  # gráficas a archivo, sin abrir ventanas
import matplotlib.pyplot as plt  # noqa: E402

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")  # oculta un aviso de MLflow que no aplica aquí
import mlflow  # noqa: E402
from mlflow.data.dataset_source_registry import resolve_dataset_source  # noqa: E402
from mlflow.data.pandas_dataset import PandasDataset  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402
from mlflow.types.utils import _infer_schema  # noqa: E402
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
import typer

from equipo52_module.config import (
    DURACION_MINIMA_SEGUNDOS,
    ESTACIONES_EXCLUIDAS,
    FECHA_FIN,
    FECHA_INICIO,
    INICIO_POSTCOVID,
    MANIFIESTO_FILE,
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
    NOMENCLATURA_FILE,
    PROJ_ROOT,
    SEED,
    VERSION_DATOS,
    VIAJES_FILE,
)
from equipo52_module.features import (
    NOMBRES_CLASES,
    NORMAL,
    SIN_ANCLAJES,
    SIN_BICICLETAS,
    construir_tabla_p2,
    quitar_viajes_cortos,
)

app = typer.Typer()

# Regla de riesgo (notebook 1.2): cada hora se revisan las próximas 3 horas; hay riesgo si el
# desbalance llega a un tercio de la capacidad.
HORAS_ANTICIPACION = 3
UMBRAL_FALLA = 1 / 3
SEMANAS_HISTORIAL = 4

# Partición en el tiempo: se entrena con lo anterior y se prueba con meses que el modelo no vio.
# La prueba incluye las 100 estaciones abiertas de mayo a agosto de 2026 (acuerdo del equipo).
# Validación: los 6 meses previos a la prueba (mismo largo). Sirve para elegir modelo e
# hiperparámetros sin mirar la prueba; después se reentrena con entrenamiento + validación,
# que es el mismo periodo de entrenamiento de la entrega anterior.
INICIO_VALIDACION = "2025-10-01"
INICIO_PRUEBA = "2026-04-01"
INICIOS_ENTRENAMIENTO = {
    "completo": FECHA_INICIO,
    "postcovid": INICIO_POSTCOVID,
}

VARIABLES = [
    "hora_del_dia", "dia_semana", "mes", "fin_de_semana",
    "capacidad", "latitud", "longitud",
    "salidas_hora_anterior", "llegadas_hora_anterior",
    "salidas_3_horas_anteriores", "llegadas_3_horas_anteriores",
    "semanas_sin_bicicletas", "semanas_sin_anclajes",
]

TIPOS_CONFIABLES_HGB = ["sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"]

# Valores por omisión; con --params se cambian los que se indiquen (los demás se quedan así).
# min_samples_leaf y l2_regularization tienen los mismos valores por omisión de scikit-learn.
HIPERPARAMETROS_HGB = {"max_iter": 200, "learning_rate": 0.1, "max_leaf_nodes": 31,
                       "min_samples_leaf": 20, "l2_regularization": 0.0}
# Con --balanceado, las fallas pesan lo mismo que las horas normales: el modelo detecta más fallas
# pero da muchas más alertas falsas. Sin él, da menos alertas y más certeras.


def cargar_estaciones(ruta: Path, excluidas: list[int]) -> pd.DataFrame:
    """Lee capacidad y ubicación de la nomenclatura, sin las estaciones excluidas ni las de capacidad 0."""
    nomenclatura = pd.read_csv(ruta).rename(columns={"id": "estacion_id", "dpcapacity": "capacidad",
                                                     "latitude": "latitud", "longitude": "longitud"})
    nomenclatura = nomenclatura[~nomenclatura["estacion_id"].isin(excluidas) & (nomenclatura["capacidad"] > 0)]
    return nomenclatura.set_index("estacion_id")[["capacidad", "latitud", "longitud"]]


def entre(tabla: pd.DataFrame, inicio: str, fin_siguiente: str) -> pd.DataFrame:
    """Horas desde inicio hasta antes de fin_siguiente, sin las últimas 3.

    Esas últimas horas se recortan para que su ventana de 3 horas no entre al periodo que sigue
    (si no, el modelo vería el inicio de la validación o de la prueba al entrenar).
    """
    ultima_hora = pd.Timestamp(fin_siguiente) - pd.Timedelta(hours=HORAS_ANTICIPACION)
    return tabla[(tabla["hora"] >= pd.Timestamp(inicio)) & (tabla["hora"] <= ultima_hora)]


def partir_en_el_tiempo(tabla: pd.DataFrame, inicio_entrenamiento: str) -> dict[str, pd.DataFrame]:
    """Separa por fecha: entrenamiento, validación, entrenamiento final (entrenamiento + validación) y prueba."""
    tabla = tabla[tabla["clase"].notna()]
    fin_prueba = pd.Timestamp(FECHA_FIN) + pd.Timedelta(days=1)
    return {
        "entrenamiento": entre(tabla, inicio_entrenamiento, INICIO_VALIDACION),
        "validacion": entre(tabla, INICIO_VALIDACION, INICIO_PRUEBA),
        "entrenamiento_final": entre(tabla, inicio_entrenamiento, INICIO_PRUEBA),
        "prueba": tabla[(tabla["hora"] >= pd.Timestamp(INICIO_PRUEBA)) & (tabla["hora"] < fin_prueba)],
    }


def leer_hiperparametros(texto: str) -> dict:
    """Convierte "max_iter=300,learning_rate=0.05" en los hiperparámetros de HGB.

    Parte de HIPERPARAMETROS_HGB y solo cambia los indicados; cada valor se convierte al tipo de
    su valor por omisión (entero o decimal). Un nombre desconocido detiene la corrida, para no
    entrenar en silencio con un hiperparámetro mal escrito.
    """
    hiperparametros = dict(HIPERPARAMETROS_HGB)
    for par in filter(None, texto.replace(" ", "").split(",")):
        nombre, valor = par.split("=")
        if nombre not in HIPERPARAMETROS_HGB:
            raise typer.BadParameter(f"hiperparámetro desconocido: {nombre}. Válidos: {', '.join(HIPERPARAMETROS_HGB)}")
        hiperparametros[nombre] = type(HIPERPARAMETROS_HGB[nombre])(valor)
    return hiperparametros


def predecir_con_base(datos: pd.DataFrame) -> np.ndarray:
    """Modelo base: repite lo que pasó en las últimas 4 semanas a la misma hora y día.

    Predice una falla si ocurrió en 2 o más de esas 4 semanas; si los dos tipos empatan, gana el
    de la semana más reciente. Representa lo que haría un operador con experiencia, sin modelo.
    """
    sin_bicicletas = datos["semanas_sin_bicicletas"].fillna(0).to_numpy()
    sin_anclajes = datos["semanas_sin_anclajes"].fillna(0).to_numpy()
    semana_anterior = datos["clase_semana_anterior"].fillna(NORMAL).to_numpy()
    prediccion = np.full(len(datos), NORMAL)
    prediccion[(sin_bicicletas >= 2) & (sin_bicicletas > sin_anclajes)] = SIN_BICICLETAS
    prediccion[(sin_anclajes >= 2) & (sin_anclajes > sin_bicicletas)] = SIN_ANCLAJES
    empate = (sin_bicicletas >= 2) & (sin_bicicletas == sin_anclajes)
    prediccion[empate] = semana_anterior[empate]
    return prediccion


def calcular_metricas(real: np.ndarray, prediccion: np.ndarray, prefijo: str) -> dict[str, float]:
    """Métricas de un conjunto: prefijo val_ (validación) o test_ (prueba).

    Lo importante son las dos clases de falla: recall = de las fallas reales, cuántas se
    anticiparon; precision = de las alertas, cuántas fueron fallas reales. La exactitud se
    registra para mostrar que engaña: casi todas las horas son normales.
    """
    metricas = {f"{prefijo}_exactitud": accuracy_score(real, prediccion),
                f"{prefijo}_f1_fallas": f1_score(real, prediccion, labels=[SIN_BICICLETAS, SIN_ANCLAJES], average="macro")}
    for clase in [SIN_BICICLETAS, SIN_ANCLAJES]:
        nombre = NOMBRES_CLASES[clase]
        metricas[f"{prefijo}_recall_{nombre}"] = recall_score(real, prediccion, labels=[clase], average="macro")
        metricas[f"{prefijo}_precision_{nombre}"] = precision_score(real, prediccion, labels=[clase],
                                                                    average="macro", zero_division=0)
        metricas[f"{prefijo}_f1_{nombre}"] = f1_score(real, prediccion, labels=[clase], average="macro")
    return metricas


def tabla_confusion(real: np.ndarray, prediccion: np.ndarray) -> pd.DataFrame:
    """Tabla de aciertos y errores: filas = lo que pasó, columnas = lo que predijo el modelo."""
    clases = list(NOMBRES_CLASES)
    nombres = [NOMBRES_CLASES[clase] for clase in clases]
    return pd.DataFrame(confusion_matrix(real, prediccion, labels=clases),
                        index=pd.Index(nombres, name="real"), columns=pd.Index(nombres, name="predicho"))


# Colores de las gráficas: un tono por clase de falla (mismo color en todas las gráficas) y tinta
# gris para textos y ejes, para que el color quede solo en los datos.
COLOR_FALLA = {SIN_BICICLETAS: "#2a78d6", SIN_ANCLAJES: "#eb6834"}
TINTA = "#52514e"


def nombre_legible(clase: int) -> str:
    return NOMBRES_CLASES[clase].replace("_", " ")


def grafica_confusion(real: np.ndarray, prediccion: np.ndarray, ruta: Path) -> None:
    """Matriz de confusión en porcentaje por fila (de lo que pasó, qué predijo el modelo).

    En porcentaje porque casi todas las horas son normales: en conteos, las fallas no se verían.
    Cada celda muestra también el conteo de horas.
    """
    conteo = tabla_confusion(real, prediccion)
    porcentaje = conteo.div(conteo.sum(axis=1), axis=0) * 100
    nombres = [nombre_legible(clase) for clase in NOMBRES_CLASES]
    fig, ax = plt.subplots(figsize=(6, 4.8))
    ax.imshow(porcentaje.to_numpy(), cmap="Blues", vmin=0, vmax=100)
    for i in range(len(nombres)):
        for j in range(len(nombres)):
            valor = porcentaje.iloc[i, j]
            ax.text(j, i, f"{valor:.1f} %\n{conteo.iloc[i, j]:,} h", ha="center", va="center", fontsize=9,
                    color="white" if valor > 55 else "#0b0b0b")
    ax.set_xticks(range(len(nombres)), nombres, color=TINTA)
    ax.set_yticks(range(len(nombres)), nombres, color=TINTA)
    ax.set_xlabel("Predicho", color=TINTA)
    ax.set_ylabel("Real", color=TINTA)
    ax.set_title("P2 · Matriz de confusión en prueba (% por fila)", color="#0b0b0b", fontsize=11)
    ax.tick_params(length=0)
    for borde in ax.spines.values():
        borde.set_visible(False)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def grafica_precision_recall(real: np.ndarray, probabilidades: np.ndarray, clases_modelo: np.ndarray,
                             prediccion: np.ndarray, prediccion_base: np.ndarray, ruta: Path) -> None:
    """Curva precisión-recall de cada clase de falla, con el punto donde opera el modelo y el del baseline.

    Si el punto del modelo (círculo) queda arriba y a la derecha del baseline (rombo), el modelo
    anticipa más fallas con alertas más certeras.
    """
    fig, ax = plt.subplots(figsize=(6.5, 5.6))
    for clase in [SIN_BICICLETAS, SIN_ANCLAJES]:
        color = COLOR_FALLA[clase]
        columna = list(clases_modelo).index(clase)
        precision, recall, _ = precision_recall_curve(real == clase, probabilidades[:, columna])
        ax.plot(recall, precision, color=color, linewidth=2, label=f"{nombre_legible(clase)} · curva del modelo")
        for pred, marca, relleno, etiqueta in [(prediccion, "o", color, "modelo"),
                                               (prediccion_base, "D", "white", "baseline")]:
            r = recall_score(real, pred, labels=[clase], average="macro")
            p = precision_score(real, pred, labels=[clase], average="macro", zero_division=0)
            ax.plot(r, p, marca, markersize=9, markerfacecolor=relleno, markeredgecolor=color, markeredgewidth=2,
                    label=f"{nombre_legible(clase)} · {etiqueta}")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.04)  # un margen para que no se corten los puntos en 1
    ax.set_xlabel("Recall: de las fallas reales, cuántas se anticipan", color=TINTA)
    ax.set_ylabel("Precisión: de las alertas, cuántas fueron fallas", color=TINTA)
    ax.set_title("P2 · Curva precisión-recall en prueba", color="#0b0b0b", fontsize=11)
    ax.grid(color="#e4e3df", linewidth=0.8)
    ax.tick_params(colors=TINTA, length=0)
    for lado in ["top", "right"]:
        ax.spines[lado].set_visible(False)
    # Leyenda debajo del área de la gráfica, para que no tape ninguna curva ni punto.
    ax.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def porcentaje_por_anio(tabla: pd.DataFrame) -> pd.DataFrame:
    """Porcentaje de horas de cada clase por año, para ver si la regla cambia con el tiempo."""
    conteo = pd.crosstab(tabla["hora"].dt.year.rename("anio"), tabla["clase"].map(NOMBRES_CLASES))
    return (conteo.div(conteo.sum(axis=1), axis=0) * 100).round(2)


def etiquetas(modelo: str, etapa: str) -> dict[str, str]:
    """Etiquetas de la corrida para filtrar y comparar en la interfaz de MLflow.

    equipo, pregunta y commit son las del equipo desde la semana 5; question, model_family y
    stage_of_work (baseline, search o candidate) son las que pide la semana 6.
    """
    return {"equipo": "52", "pregunta": "P2", "commit": commit_actual(),
            "question": "p2", "model_family": "baseline" if modelo == "base" else "hgb",
            "stage_of_work": etapa}


class ConjuntoP2(PandasDataset):
    """Conjunto de datos para mlflow.log_input que infiere el esquema con las primeras 1,000 filas.

    PandasDataset revisa todas las filas para saber el tipo de cada columna: con los 26 millones
    de filas de entrenamiento tarda unos 90 s cada vez. Los tipos son los mismos en las primeras filas.
    """

    @property
    def schema(self):
        return _infer_schema(self._df.head(1000))


def conjunto_para_mlflow(datos: pd.DataFrame, nombre: str, huella_datos: str) -> ConjuntoP2:
    """Describe un conjunto (filas, columnas, huella y origen) para registrarlo sin subir los datos.

    La huella combina la del manifiesto con el nombre y el tamaño del conjunto: identifica la versión
    de los datos y la partición, y se calcula al instante (la de MLflow revisa todas las filas).
    """
    huella = hashlib.sha256(f"{huella_datos}|{nombre}|{datos.shape}".encode()).hexdigest()[:8]
    return ConjuntoP2(df=datos, source=resolve_dataset_source(str(VIAJES_FILE.relative_to(PROJ_ROOT))),
                      targets="clase", name=nombre, digest=huella)


def huella_manifiesto(ruta: Path) -> str:
    """Huella SHA-256 del manifiesto de datos: identifica exactamente qué archivos se usaron."""
    return hashlib.sha256(ruta.read_bytes()).hexdigest()


def commit_actual() -> str:
    """Commit de Git desde el que se ejecuta la corrida."""
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJ_ROOT, capture_output=True, text=True).stdout.strip()


@app.command()
def main(modelo: str = typer.Option("hgb", help="base o hgb"),
         periodo: str = typer.Option("completo", help="completo o postcovid"),
         balanceado: bool = typer.Option(False, help="dar a las fallas el mismo peso que a las horas normales"),
         params: str = typer.Option("", help='hiperparámetros de hgb, por ejemplo "max_iter=300,learning_rate=0.05"')):
    if params and modelo != "hgb":
        raise typer.BadParameter("--params solo aplica al modelo hgb")
    hiperparametros = leer_hiperparametros(params)
    inicio_entrenamiento = INICIOS_ENTRENAMIENTO[periodo]
    logger.info(f"P2 · modelo {modelo} · entrenamiento desde {inicio_entrenamiento} · prueba desde {INICIO_PRUEBA}")

    viajes = pd.read_parquet(VIAJES_FILE)
    viajes = quitar_viajes_cortos(viajes, DURACION_MINIMA_SEGUNDOS)
    estaciones = cargar_estaciones(NOMENCLATURA_FILE, ESTACIONES_EXCLUIDAS)
    tabla = construir_tabla_p2(viajes, estaciones, HORAS_ANTICIPACION, UMBRAL_FALLA, SEMANAS_HISTORIAL)
    del viajes
    partes = partir_en_el_tiempo(tabla, inicio_entrenamiento)
    entrenamiento, validacion = partes["entrenamiento"], partes["validacion"]
    entrenamiento_final, prueba = partes["entrenamiento_final"], partes["prueba"]
    logger.info(f"Filas estación × hora · entrenamiento: {len(entrenamiento):,} · validación: {len(validacion):,}"
                f" · entrenamiento final: {len(entrenamiento_final):,} · prueba: {len(prueba):,}")

    parametros = {
        "pregunta": "p2", "modelo": modelo, "periodo": periodo,
        "fecha_inicio_entrenamiento": inicio_entrenamiento, "fecha_inicio_validacion": INICIO_VALIDACION,
        "fecha_inicio_prueba": INICIO_PRUEBA, "fecha_fin_prueba": FECHA_FIN, "version_datos": VERSION_DATOS,
        "huella_manifiesto": huella_manifiesto(MANIFIESTO_FILE), "semilla": SEED,
        "horas_anticipacion": HORAS_ANTICIPACION, "umbral_falla": round(UMBRAL_FALLA, 4),
        "semanas_historial": SEMANAS_HISTORIAL, "estaciones_excluidas": len(ESTACIONES_EXCLUIDAS),
        "duracion_minima_segundos": DURACION_MINIMA_SEGUNDOS,
        "filas_entrenamiento": len(entrenamiento), "filas_validacion": len(validacion),
        "filas_entrenamiento_final": len(entrenamiento_final), "filas_prueba": len(prueba),
    }

    real_validacion = validacion["clase"].astype(int).to_numpy()
    real = prueba["clase"].astype(int).to_numpy()
    if modelo == "base":
        prediccion_validacion = predecir_con_base(validacion)
        prediccion = predecir_con_base(prueba)
        clasificador = None
    elif modelo == "hgb":
        pesos = "balanced" if balanceado else None

        def nuevo_clasificador() -> HistGradientBoostingClassifier:
            return HistGradientBoostingClassifier(**hiperparametros, class_weight=pesos, random_state=SEED)

        # Validación: se entrena sin los 6 meses de validación y se evalúa con ellos.
        clasificador = nuevo_clasificador().fit(entrenamiento[VARIABLES], entrenamiento["clase"].astype(int))
        prediccion_validacion = clasificador.predict(validacion[VARIABLES])
        # Prueba: se reentrena con entrenamiento + validación; este es el modelo que se registra.
        clasificador = nuevo_clasificador().fit(entrenamiento_final[VARIABLES],
                                                entrenamiento_final["clase"].astype(int))
        prediccion = clasificador.predict(prueba[VARIABLES])
        probabilidades = clasificador.predict_proba(prueba[VARIABLES])  # para la curva precisión-recall
        parametros.update({f"hgb_{nombre}": valor for nombre, valor in hiperparametros.items()})
        parametros["hgb_class_weight"] = pesos
        parametros["variables"] = ", ".join(VARIABLES)
    else:
        raise typer.BadParameter("modelo debe ser base o hgb")

    metricas = (calcular_metricas(real_validacion, prediccion_validacion, "val")
                | calcular_metricas(real, prediccion, "test"))

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    nombre_corrida = f"p2-{modelo}{'-balanceado' if balanceado else ''}-{periodo}"
    if params:  # por ejemplo p2-hgb-completo-max_iter300-learning_rate0.05
        nombre_corrida += "-" + params.replace(" ", "").replace("=", "").replace(",", "-")
    with mlflow.start_run(run_name=nombre_corrida):
        etapa = "baseline" if modelo == "base" else "candidate"
        mlflow.set_tags(etiquetas(modelo, etapa))
        mlflow.log_params(parametros)
        mlflow.log_metrics(metricas)

        # Versión de los datos: MLflow guarda la descripción de cada conjunto (filas, columnas, huella y
        # origen), no los datos. Son conteos por estación y hora, sin datos personales de las personas usuarias.
        columnas = VARIABLES + ["clase"]
        if clasificador is not None:
            mlflow.log_input(conjunto_para_mlflow(entrenamiento_final[columnas], f"p2-entrenamiento-{periodo}",
                                                  parametros["huella_manifiesto"]), context="training")
        mlflow.log_input(conjunto_para_mlflow(prueba[columnas], "p2-prueba", parametros["huella_manifiesto"]),
                         context="testing")

        # Artefactos en subcarpetas: plots/ (gráficas), data/ (versión de datos y tablas) y config/.
        with tempfile.TemporaryDirectory() as carpeta:
            carpeta = Path(carpeta)
            for sub in ["plots", "data", "config"]:
                (carpeta / sub).mkdir()
            grafica_confusion(real, prediccion, carpeta / "plots" / "p2_matriz_confusion_prueba.png")
            if clasificador is not None:
                grafica_precision_recall(real, probabilidades, clasificador.classes_, prediccion,
                                         predecir_con_base(prueba), carpeta / "plots" / "p2_precision_recall_prueba.png")
            shutil.copy(MANIFIESTO_FILE, carpeta / "data" / MANIFIESTO_FILE.name)
            tabla_confusion(real, prediccion).to_csv(carpeta / "data" / "p2_tabla_confusion_prueba.csv")
            porcentaje_por_anio(tabla[tabla["clase"].notna()]).to_csv(
                carpeta / "data" / "p2_porcentaje_clases_por_anio.csv")
            configuracion = {"parametros": parametros, "etiquetas": etiquetas(modelo, etapa),
                             "hiperparametros": hiperparametros if clasificador is not None else None,
                             "variables": VARIABLES}
            (carpeta / "config" / "corrida_p2.json").write_text(json.dumps(configuracion, indent=2, ensure_ascii=False))
            for archivo in ["equipo52_module/config.py", "MLproject", "python_env.yaml"]:
                if (PROJ_ROOT / archivo).exists():
                    shutil.copy(PROJ_ROOT / archivo, carpeta / "config" / Path(archivo).name)
            mlflow.log_artifacts(carpeta)

        if clasificador is not None:
            # Firma explícita: qué columnas y tipos recibe el modelo y qué devuelve.
            ejemplo = prueba[VARIABLES].head(1000)
            firma = infer_signature(ejemplo, clasificador.predict(ejemplo))
            # MLflow guarda el modelo con skops, que pide declarar los tipos internos del modelo como confiables;
            # el modelo lo acaba de entrenar este mismo script.
            mlflow.sklearn.log_model(clasificador, name="modelo", signature=firma, input_example=ejemplo.head(5),
                                     skops_trusted_types=TIPOS_CONFIABLES_HGB)

    for nombre, valor in metricas.items():
        logger.info(f"{nombre}: {valor:.3f}")
    logger.success(f"Corrida {nombre_corrida} registrada en el experimento {MLFLOW_EXPERIMENT}")


if __name__ == "__main__":
    app()