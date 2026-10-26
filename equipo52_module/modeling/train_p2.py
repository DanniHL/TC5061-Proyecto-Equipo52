"""P2 · Entrenamiento y registro en MLflow del modelo de riesgo por estación.

Pregunta: ¿Una estación estará en riesgo de quedarse sin bicicletas o sin anclajes libres en
las próximas 3 horas? Cada corrida entrena con un periodo (historia completa o Post-Covid),
se valida con octubre de 2025 a marzo de 2026, se prueba con abril a septiembre de 2026 y
registra parámetros, métricas y artefactos en MLflow.

Uso con MLflow Project (como se registran las corridas de la entrega):
    mlflow run . -e train --env-manager local --experiment-name mibici-equipo52 \
        -P pregunta=p2 -P modelo=hgb -P periodo=completo -P learning_rate=0.05 -P peso_fallas=5

Uso directo para pruebas (mismas opciones que pasa el despachador project_train.py):
    python -m equipo52_module.modeling.train_p2 --modelo base --periodo completo
    python -m equipo52_module.modeling.train_p2 --modelo hgb --periodo completo --learning-rate 0.05 --peso-fallas 5
    python -m equipo52_module.modeling.train_p2 --modelo logistica --peso-fallas 3 --params "C=0.5"
    python -m equipo52_module.modeling.train_p2 --modelo busqueda --periodo completo
"""

import hashlib
import itertools
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from loguru import logger
import matplotlib.pyplot as plt  # las gráficas se guardan con savefig; nunca se muestran en pantalla

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")  # oculta un aviso de MLflow que no aplica aquí
import mlflow  # noqa: E402
from mlflow.data.dataset_source_registry import resolve_dataset_source  # noqa: E402
from mlflow.data.pandas_dataset import PandasDataset  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402
from mlflow.types.utils import _infer_schema  # noqa: E402
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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

# Tipos internos de cada modelo que skops (el formato con que MLflow guarda los modelos de
# scikit-learn) pide declarar como confiables: el árbol de HGB y el tipo de dato que recuerda el imputador.
TIPOS_CONFIABLES = {
    "hgb": ["sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"],
    "logistica": ["numpy.dtype"],
}

# Hiperparámetros de cada modelo con sus valores por omisión; con --params se cambian los que se
# indiquen. En hgb, min_samples_leaf y l2_regularization tienen los valores por omisión de
# scikit-learn, así que con estos valores el modelo es el mismo de la semana 5.
# peso_fallas: cuánto pesa cada hora de falla frente a una hora normal al entrenar. Con 1 todas
# pesan igual: el modelo da pocas alertas, muy certeras, y se le escapan muchas fallas. Con
# --balanceado (18 a 19 veces en el entrenamiento completo; el notebook 2.1 lo calcula) da demasiadas
# alertas falsas. Un valor intermedio busca el equilibrio.
HIPERPARAMETROS = {
    "hgb": {"max_iter": 200, "learning_rate": 0.1, "max_leaf_nodes": 31,
            "min_samples_leaf": 20, "l2_regularization": 0.0, "peso_fallas": 1.0},
    "logistica": {"C": 1.0, "max_iter": 200, "peso_fallas": 1.0},
}
FAMILIAS = {"base": "baseline", "hgb": "hgb", "logistica": "logistica"}

# Rejilla de la búsqueda con nested runs (--modelo busqueda): una corrida hija por combinación.
# Está en el código para que la búsqueda se reproduzca igual desde cualquier clon del repositorio.
REJILLA_BUSQUEDA = {"peso_fallas": [2.0, 3.0, 5.0, 8.0], "learning_rate": [0.05, 0.1]}


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


def leer_hiperparametros(modelo: str, texto: str) -> dict:
    """Convierte "max_iter=300,peso_fallas=3" en los hiperparámetros del modelo.

    Parte de los valores por omisión del modelo y solo cambia los indicados; cada valor se
    convierte al tipo de su valor por omisión (entero o decimal). Un nombre desconocido detiene la
    corrida, para no entrenar en silencio con un hiperparámetro mal escrito.
    """
    omision = HIPERPARAMETROS[modelo]
    hiperparametros = dict(omision)
    for par in filter(None, texto.replace(" ", "").split(",")):
        nombre, valor = par.split("=")
        if nombre not in omision:
            raise typer.BadParameter(f"hiperparámetro desconocido para {modelo}: {nombre}. "
                                     f"Válidos: {', '.join(omision)}")
        hiperparametros[nombre] = type(omision[nombre])(valor)
    return hiperparametros


def pesos_por_clase(peso_fallas: float, balanceado: bool) -> dict | str | None:
    """Pesos de las clases al entrenar: normal = 1 y cada clase de falla = peso_fallas.

    Con peso 1 devuelve None (todas pesan igual, como en la semana 5); con --balanceado,
    scikit-learn calcula los pesos para que cada clase pese lo mismo en total.
    """
    if balanceado:
        if peso_fallas != 1:
            raise typer.BadParameter("usa --balanceado o peso_fallas, no los dos")
        return "balanced"
    if peso_fallas == 1:
        return None
    return {NORMAL: 1.0, SIN_BICICLETAS: peso_fallas, SIN_ANCLAJES: peso_fallas}


def nuevo_clasificador(modelo: str, hiperparametros: dict, pesos: dict | str | None):
    """Crea el modelo sin entrenar.

    La regresión logística no acepta valores vacíos (las primeras horas y semanas de cada estación
    no tienen historial): se rellenan con la mediana y se agrega una columna que marca dónde
    faltaban. Después se estandarizan las variables para que sus coeficientes sean comparables.
    """
    propios = {nombre: valor for nombre, valor in hiperparametros.items() if nombre != "peso_fallas"}
    if modelo == "hgb":
        return HistGradientBoostingClassifier(**propios, class_weight=pesos, random_state=SEED)
    return Pipeline([
        ("imputar", SimpleImputer(strategy="median", add_indicator=True)),
        ("escalar", StandardScaler()),
        ("logistica", LogisticRegression(**propios, class_weight=pesos, random_state=SEED)),
    ])


def variables(datos: pd.DataFrame, modelo: str) -> pd.DataFrame:
    """Columnas de entrada del modelo; la logística las recibe en float32 para usar la mitad de memoria."""
    return datos[VARIABLES].astype("float32") if modelo == "logistica" else datos[VARIABLES]


def coeficientes_logistica(clasificador: Pipeline) -> pd.DataFrame:
    """Coeficientes de la regresión logística por variable y clase.

    Como las variables están estandarizadas, el tamaño del coeficiente indica cuánto empuja cada
    variable hacia esa clase: positivo la hace más probable, negativo menos.
    """
    nombres = clasificador.named_steps["imputar"].get_feature_names_out(VARIABLES)
    logistica = clasificador.named_steps["logistica"]
    return pd.DataFrame(logistica.coef_.T, index=pd.Index(nombres, name="variable"),
                        columns=[NOMBRES_CLASES[c] for c in logistica.classes_]).round(4)


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
            "question": "p2", "model_family": FAMILIAS[modelo],
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


def preparar_datos(periodo: str) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Construye la tabla estación × hora y la parte en entrenamiento, validación, entrenamiento final y prueba."""
    viajes = quitar_viajes_cortos(pd.read_parquet(VIAJES_FILE), DURACION_MINIMA_SEGUNDOS)
    estaciones = cargar_estaciones(NOMENCLATURA_FILE, ESTACIONES_EXCLUIDAS)
    tabla = construir_tabla_p2(viajes, estaciones, HORAS_ANTICIPACION, UMBRAL_FALLA, SEMANAS_HISTORIAL)
    del viajes
    partes = partir_en_el_tiempo(tabla, INICIOS_ENTRENAMIENTO[periodo])
    logger.info("Filas estación × hora · " + " · ".join(f"{nombre}: {len(datos):,}" for nombre, datos in partes.items()))
    return tabla, partes


def parametros_comunes(modelo: str, periodo: str, partes: dict[str, pd.DataFrame]) -> dict:
    """Parámetros que registra toda corrida de P2: datos, partición, regla de riesgo y semilla."""
    return {
        "pregunta": "p2", "modelo": modelo, "periodo": periodo,
        "fecha_inicio_entrenamiento": INICIOS_ENTRENAMIENTO[periodo], "fecha_inicio_validacion": INICIO_VALIDACION,
        "fecha_inicio_prueba": INICIO_PRUEBA, "fecha_fin_prueba": FECHA_FIN, "version_datos": VERSION_DATOS,
        "huella_manifiesto": huella_manifiesto(MANIFIESTO_FILE), "semilla": SEED,
        "horas_anticipacion": HORAS_ANTICIPACION, "umbral_falla": round(UMBRAL_FALLA, 4),
        "semanas_historial": SEMANAS_HISTORIAL, "estaciones_excluidas": len(ESTACIONES_EXCLUIDAS),
        "duracion_minima_segundos": DURACION_MINIMA_SEGUNDOS,
        "filas_entrenamiento": len(partes["entrenamiento"]), "filas_validacion": len(partes["validacion"]),
        "filas_entrenamiento_final": len(partes["entrenamiento_final"]), "filas_prueba": len(partes["prueba"]),
    }


def resolver_hiperparametros(modelo: str, params: str, opciones: dict) -> dict:
    """Hiperparámetros finales: valores por omisión, luego --params y luego las opciones explícitas.

    Las opciones explícitas (--max-iter, --learning-rate, --max-leaf-nodes, --peso-fallas) son las que
    pasa el MLproject. El despachador las manda siempre, así que se ignoran las que no aplican al
    modelo (por ejemplo learning_rate en la logística) y, en la búsqueda, las que fija la rejilla.
    """
    if modelo == "base":
        return {}
    familia = "hgb" if modelo == "busqueda" else modelo
    hiperparametros = leer_hiperparametros(familia, params)
    for nombre, valor in opciones.items():
        aplica = nombre in HIPERPARAMETROS[familia] and not (modelo == "busqueda" and nombre in REJILLA_BUSQUEDA)
        if valor is not None and aplica:
            hiperparametros[nombre] = type(HIPERPARAMETROS[familia][nombre])(valor)
    return hiperparametros


def mismo_valor(a, b) -> bool:
    """Compara dos valores de parámetro como los guarda MLflow (texto): 5, 5.0 y "5" son el mismo valor."""
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return str(a) == str(b)


def registrar_parametros(parametros: dict) -> None:
    """Registra los parámetros de la corrida sin chocar con los que ya registró mlflow run.

    Con mlflow run, la corrida ya trae los parámetros del MLproject (pregunta, modelo, periodo,
    max_iter…) y MLflow no deja cambiar un parámetro registrado. Si el script trae el mismo nombre con
    el mismo valor, se omite; si el valor es distinto, se registra como <nombre>_script, para que la
    corrida no falle y la diferencia quede a la vista.
    """
    existentes = mlflow.get_run(mlflow.active_run().info.run_id).data.params
    nuevos = {}
    for nombre, valor in parametros.items():
        if nombre not in existentes:
            nuevos[nombre] = valor
        elif not mismo_valor(existentes[nombre], valor):
            nuevos[f"{nombre}_script"] = valor
    mlflow.log_params(nuevos)


def sufijo_hiperparametros(familia: str, hiperparametros: dict) -> str:
    """Los hiperparámetros distintos de los valores por omisión, para el nombre de la corrida."""
    distintos = [f"{nombre}{valor:g}" if isinstance(valor, float) else f"{nombre}{valor}"
                 for nombre, valor in hiperparametros.items() if valor != HIPERPARAMETROS[familia][nombre]]
    return "-" + "-".join(distintos) if distintos else ""


def parametros_del_modelo(modelo: str, hiperparametros: dict, pesos: dict | str | None) -> dict:
    """Hiperparámetros con prefijo del modelo; peso_fallas sin prefijo para comparar familias en la misma columna."""
    parametros = {f"{modelo}_{nombre}": valor for nombre, valor in hiperparametros.items() if nombre != "peso_fallas"}
    parametros["peso_fallas"] = hiperparametros["peso_fallas"]
    parametros[f"{modelo}_class_weight"] = pesos
    parametros["variables"] = ", ".join(VARIABLES)
    return parametros


def entrenar(modelo: str, hiperparametros: dict, pesos, datos: pd.DataFrame):
    """Entrena un modelo y mide cuánto tarda (dato de costo para elegir el modelo)."""
    inicio = time.perf_counter()
    clasificador = nuevo_clasificador(modelo, hiperparametros, pesos).fit(variables(datos, modelo),
                                                                          datos["clase"].astype(int))
    return clasificador, time.perf_counter() - inicio


def guardar_configuracion(carpeta: Path, configuracion: dict) -> None:
    """config/: la configuración de la corrida en JSON y los archivos de configuración del proyecto."""
    (carpeta / "config" / "corrida_p2.json").write_text(json.dumps(configuracion, indent=2, ensure_ascii=False))
    for archivo in ["equipo52_module/config.py", "MLproject", "python_env.yaml"]:
        if (PROJ_ROOT / archivo).exists():
            shutil.copy(PROJ_ROOT / archivo, carpeta / "config" / Path(archivo).name)


def carpeta_de_artefactos(carpeta: str) -> Path:
    """Crea las subcarpetas plots/, data/ y config/ dentro de una carpeta temporal."""
    carpeta = Path(carpeta)
    for sub in ["plots", "data", "config"]:
        (carpeta / sub).mkdir()
    return carpeta


def correr_candidata(modelo: str, periodo: str, balanceado: bool, hiperparametros: dict,
                     tabla: pd.DataFrame, partes: dict[str, pd.DataFrame]) -> None:
    """Una corrida baseline o candidata: métricas de validación y prueba, artefactos y, si entrena, el modelo."""
    pesos = pesos_por_clase(hiperparametros.get("peso_fallas", 1.0), balanceado)
    entrenamiento, validacion = partes["entrenamiento"], partes["validacion"]
    entrenamiento_final, prueba = partes["entrenamiento_final"], partes["prueba"]
    parametros = parametros_comunes(modelo, periodo, partes)

    real_validacion = validacion["clase"].astype(int).to_numpy()
    real = prueba["clase"].astype(int).to_numpy()
    if modelo == "base":
        prediccion_validacion = predecir_con_base(validacion)
        prediccion = predecir_con_base(prueba)
        clasificador, segundos = None, 0.0
    else:
        # Validación: se entrena sin los 6 meses de validación y se evalúa con ellos.
        clasificador, _ = entrenar(modelo, hiperparametros, pesos, entrenamiento)
        prediccion_validacion = clasificador.predict(variables(validacion, modelo))
        # Prueba: se reentrena con entrenamiento + validación; este es el modelo que se registra.
        clasificador, segundos = entrenar(modelo, hiperparametros, pesos, entrenamiento_final)
        prediccion = clasificador.predict(variables(prueba, modelo))
        probabilidades = clasificador.predict_proba(variables(prueba, modelo))  # para la curva precisión-recall
        parametros.update(parametros_del_modelo(modelo, hiperparametros, pesos))

    metricas = (calcular_metricas(real_validacion, prediccion_validacion, "val")
                | calcular_metricas(real, prediccion, "test")
                | {"segundos_entrenamiento": round(segundos, 1)})

    # Por ejemplo p2-hgb-completo-learning_rate0.05-peso_fallas5: el nombre dice qué cambió respecto a los valores por omisión.
    nombre_corrida = (f"p2-{modelo}{'-balanceado' if balanceado else ''}-{periodo}"
                      + (sufijo_hiperparametros(modelo, hiperparametros) if modelo != "base" else ""))
    with mlflow.start_run(run_name=nombre_corrida):
        # Con mlflow run la corrida ya existe con un nombre al azar: se le pone el de la corrida.
        mlflow.set_tag("mlflow.runName", nombre_corrida)
        etapa = "baseline" if modelo == "base" else "candidate"
        mlflow.set_tags(etiquetas(modelo, etapa))
        registrar_parametros(parametros)
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
        with tempfile.TemporaryDirectory() as temporal:
            carpeta = carpeta_de_artefactos(temporal)
            grafica_confusion(real, prediccion, carpeta / "plots" / "p2_matriz_confusion_prueba.png")
            if clasificador is not None:
                grafica_precision_recall(real, probabilidades, clasificador.classes_, prediccion,
                                         predecir_con_base(prueba), carpeta / "plots" / "p2_precision_recall_prueba.png")
            shutil.copy(MANIFIESTO_FILE, carpeta / "data" / MANIFIESTO_FILE.name)
            tabla_confusion(real, prediccion).to_csv(carpeta / "data" / "p2_tabla_confusion_prueba.csv")
            porcentaje_por_anio(tabla[tabla["clase"].notna()]).to_csv(
                carpeta / "data" / "p2_porcentaje_clases_por_anio.csv")
            if modelo == "logistica":
                coeficientes_logistica(clasificador).to_csv(carpeta / "data" / "p2_coeficientes_logistica.csv")
            guardar_configuracion(carpeta, {"parametros": parametros, "etiquetas": etiquetas(modelo, etapa),
                                            "hiperparametros": hiperparametros or None, "variables": VARIABLES})
            mlflow.log_artifacts(carpeta)

        if clasificador is not None:
            # Firma explícita: qué columnas y tipos recibe el modelo y qué devuelve.
            ejemplo = prueba[VARIABLES].head(1000)
            firma = infer_signature(ejemplo, clasificador.predict(ejemplo))
            # MLflow guarda el modelo con skops, que pide declarar los tipos internos del modelo como confiables;
            # el modelo lo acaba de entrenar este mismo script.
            mlflow.sklearn.log_model(clasificador, name="modelo", signature=firma, input_example=ejemplo.head(5),
                                     skops_trusted_types=TIPOS_CONFIABLES[modelo])

    for nombre, valor in metricas.items():
        logger.info(f"{nombre}: {valor:.3f}")
    logger.success(f"Corrida {nombre_corrida} registrada en el experimento {MLFLOW_EXPERIMENT}")


def grafica_busqueda(resultados: pd.DataFrame, f1_base: float, ruta: Path) -> None:
    """F1 de fallas en validación según peso_fallas, una línea por learning_rate, contra el baseline."""
    colores = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for color, (tasa, grupo) in zip(colores, resultados.groupby("learning_rate")):
        grupo = grupo.sort_values("peso_fallas")
        ax.plot(grupo["peso_fallas"], grupo["val_f1_fallas"], marker="o", markersize=8, linewidth=2, color=color,
                markeredgecolor="white", label=f"HGB · learning_rate {tasa}")
    ax.axhline(f1_base, color=TINTA, linestyle="--", linewidth=1)
    ax.text(resultados["peso_fallas"].max(), f1_base, f"baseline {f1_base:.3f}", va="bottom", ha="right",
            fontsize=9, color=TINTA)
    mejor = resultados.loc[resultados["val_f1_fallas"].idxmax()]
    ax.annotate(f"mejor: {mejor['val_f1_fallas']:.3f}", (mejor["peso_fallas"], mejor["val_f1_fallas"]),
                textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9, color="#0b0b0b")
    ax.set_xticks(sorted(resultados["peso_fallas"].unique()))
    ax.set_xlabel("peso_fallas: cuánto pesa una hora de falla frente a una normal", color=TINTA)
    ax.set_ylabel("F1 de fallas en validación", color=TINTA)
    ax.set_title("P2 · Búsqueda de hiperparámetros de HGB", color="#0b0b0b", fontsize=11)
    ax.grid(color="#e4e3df", linewidth=0.8)
    ax.tick_params(colors=TINTA, length=0)
    for lado in ["top", "right"]:
        ax.spines[lado].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def correr_busqueda(periodo: str, base: dict, partes: dict[str, pd.DataFrame]) -> None:
    """Búsqueda de hiperparámetros de HGB con nested runs: una corrida padre y una hija por combinación.

    Cada hija entrena con entrenamiento y mide solo en validación: la prueba no se usa para elegir.
    El padre guarda la tabla de resultados, la gráfica y cuál fue la mejor hija; la mejor
    combinación se corre después como candidata completa (con prueba y modelo). base trae los
    hiperparámetros que no están en la rejilla (por ejemplo max_iter).
    """
    entrenamiento, validacion = partes["entrenamiento"], partes["validacion"]
    real_validacion = validacion["clase"].astype(int).to_numpy()
    f1_base = calcular_metricas(real_validacion, predecir_con_base(validacion), "val")["val_f1_fallas"]
    combinaciones = [dict(zip(REJILLA_BUSQUEDA, valores)) for valores in itertools.product(*REJILLA_BUSQUEDA.values())]
    nombre_padre = f"p2-busqueda-hgb-{periodo}"

    resultados = []
    with mlflow.start_run(run_name=nombre_padre) as padre:
        mlflow.set_tag("mlflow.runName", nombre_padre)  # con mlflow run, la corrida padre es la del proyecto
        parametros = parametros_comunes("busqueda", periodo, partes) | {
            "rejilla": json.dumps(REJILLA_BUSQUEDA), "combinaciones": len(combinaciones),
            "metrica_seleccion": "val_f1_fallas"}
        parametros.update({f"hgb_{nombre}": valor for nombre, valor in base.items() if nombre not in REJILLA_BUSQUEDA})
        mlflow.set_tags(etiquetas("hgb", "search"))
        registrar_parametros(parametros)
        columnas = VARIABLES + ["clase"]
        mlflow.log_input(conjunto_para_mlflow(entrenamiento[columnas], f"p2-entrenamiento-sin-validacion-{periodo}",
                                              parametros["huella_manifiesto"]), context="training")
        mlflow.log_input(conjunto_para_mlflow(validacion[columnas], "p2-validacion", parametros["huella_manifiesto"]),
                         context="validation")

        for numero, combinacion in enumerate(combinaciones, start=1):
            hiperparametros = base | combinacion
            pesos = pesos_por_clase(hiperparametros["peso_fallas"], False)
            sufijo = "-".join(f"{nombre}{valor}" for nombre, valor in combinacion.items())
            logger.info(f"Hija {numero}/{len(combinaciones)} · {combinacion}")
            with mlflow.start_run(run_name=f"{nombre_padre}-{sufijo}", nested=True):
                clasificador, segundos = entrenar("hgb", hiperparametros, pesos, entrenamiento)
                metricas = (calcular_metricas(real_validacion, clasificador.predict(validacion[VARIABLES]), "val")
                            | {"segundos_entrenamiento": round(segundos, 1)})
                mlflow.set_tags(etiquetas("hgb", "search"))
                mlflow.log_params(parametros_del_modelo("hgb", hiperparametros, pesos) | {"periodo": periodo})
                mlflow.log_metrics(metricas)
            resultados.append(combinacion | metricas)
            logger.info(f"   val_f1_fallas {metricas['val_f1_fallas']:.3f} · {segundos:.0f} s")

        resultados = pd.DataFrame(resultados).sort_values("val_f1_fallas", ascending=False)
        mejor = resultados.iloc[0]
        mejor_combinacion = {nombre: mejor[nombre] for nombre in REJILLA_BUSQUEDA}
        mlflow.log_metrics({"mejor_val_f1_fallas": mejor["val_f1_fallas"], "baseline_val_f1_fallas": f1_base,
                            "segundos_busqueda": round(resultados["segundos_entrenamiento"].sum(), 1)})
        mlflow.set_tag("mejor_combinacion", json.dumps(mejor_combinacion))
        with tempfile.TemporaryDirectory() as temporal:
            carpeta = carpeta_de_artefactos(temporal)
            grafica_busqueda(resultados, f1_base, carpeta / "plots" / "p2_busqueda_val_f1.png")
            resultados.round(4).to_csv(carpeta / "data" / "p2_busqueda_resultados.csv", index=False)
            shutil.copy(MANIFIESTO_FILE, carpeta / "data" / MANIFIESTO_FILE.name)
            guardar_configuracion(carpeta, {"parametros": parametros, "etiquetas": etiquetas("hgb", "search"),
                                            "rejilla": REJILLA_BUSQUEDA, "hiperparametros_base": base,
                                            "variables": VARIABLES})
            mlflow.log_artifacts(carpeta)

    logger.info("\n" + resultados.round(3).to_string(index=False))
    opciones = " ".join(f"-P {nombre}={valor:g}" for nombre, valor in mejor_combinacion.items())
    logger.success(f"Búsqueda {nombre_padre} ({padre.info.run_id}) registrada. Mejor combinación: {mejor_combinacion} · "
                   f"val_f1_fallas {mejor['val_f1_fallas']:.3f} (baseline {f1_base:.3f}). Siguiente paso, la candidata: "
                   f"mlflow run . -e train -P pregunta=p2 -P modelo=hgb -P periodo={periodo} {opciones}")


@app.command()
def main(modelo: str = typer.Option("hgb", help="base, hgb, logistica o busqueda (búsqueda de HGB con nested runs)"),
         periodo: str = typer.Option("completo", help="completo o postcovid"),
         balanceado: bool = typer.Option(False, help="dar a las fallas el mismo peso que a las horas normales"),
         max_iter: int = typer.Option(None, help="iteraciones de HGB o de la logística"),
         learning_rate: float = typer.Option(None, help="tasa de aprendizaje de HGB"),
         max_leaf_nodes: int = typer.Option(None, help="hojas por árbol de HGB"),
         peso_fallas: float = typer.Option(None, help="peso de cada hora de falla frente a una normal"),
         params: str = typer.Option("", help='otros hiperparámetros, por ejemplo "C=0.5" o "min_samples_leaf=50"')):
    # Las opciones --max-iter, --learning-rate, --max-leaf-nodes y --peso-fallas son las del MLproject; el
    # despachador project_train.py las manda siempre, también al baseline, que no las usa.
    if modelo not in FAMILIAS and modelo != "busqueda":
        raise typer.BadParameter("modelo debe ser base, hgb, logistica o busqueda")
    if modelo == "base" and (params or balanceado):
        raise typer.BadParameter("el baseline no tiene hiperparámetros ni pesos")
    if modelo == "busqueda" and (balanceado or set(REJILLA_BUSQUEDA) & {p.split("=")[0].strip() for p in params.split(",") if p}):
        raise typer.BadParameter("la búsqueda ya explora peso_fallas y learning_rate con su rejilla")
    opciones = {"max_iter": max_iter, "learning_rate": learning_rate, "max_leaf_nodes": max_leaf_nodes,
                "peso_fallas": peso_fallas}
    hiperparametros = resolver_hiperparametros(modelo, params, opciones)  # valida antes de cargar datos
    logger.info(f"P2 · modelo {modelo} · {hiperparametros or 'sin hiperparámetros'} · "
                f"entrenamiento desde {INICIOS_ENTRENAMIENTO[periodo]} · prueba desde {INICIO_PRUEBA}")

    tabla, partes = preparar_datos(periodo)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    if modelo == "busqueda":
        correr_busqueda(periodo, hiperparametros, partes)
    else:
        correr_candidata(modelo, periodo, balanceado, hiperparametros, tabla, partes)


if __name__ == "__main__":
    app()
