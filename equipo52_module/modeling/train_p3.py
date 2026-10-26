"""P3 · Agrupamiento de estaciones por patrones de uso y registro en MLflow.

Pregunta: ¿Qué grupos de estaciones comparten patrones de uso y dónde están? Se describe
cada estación con las 42 características de features.py (firma horaria y semanal, nivel de
actividad, rol en la red, tipo de viaje y evolución) calculadas con toda la historia, se
agrupan y se registran parámetros, métricas, conjuntos de datos, modelo y artefactos en MLflow.

Uso con MLflow Project (como se registran las corridas oficiales de la entrega):
    mlflow run . -e train --env-manager local --experiment-name mibici-equipo52 -P pregunta=p3 -P modelo=geo
    mlflow run . -e train --env-manager local --experiment-name mibici-equipo52 -P pregunta=p3 -P modelo=busqueda
    mlflow run . -e train --env-manager local --experiment-name mibici-equipo52 -P pregunta=p3 -P modelo=kmeans -P k=4
    mlflow run . -e train --env-manager local --experiment-name mibici-equipo52 -P pregunta=p3 -P modelo=kmeans -P k=2
    mlflow run . -e train --env-manager local --experiment-name mibici-equipo52 -P pregunta=p3 -P modelo=kmeans -P k=4 -P solo_consolidadas=true
    mlflow run . -e train --env-manager local --experiment-name mibici-equipo52 -P pregunta=p3 -P modelo=ward -P k=4
    mlflow run . -e train --env-manager local --experiment-name mibici-equipo52 -P pregunta=p3 -P modelo=hdbscan -P min_cluster_size=15

Uso directo para pruebas (mismas opciones que pasa el despachador project_train.py):
    python -m equipo52_module.modeling.train_p3 --modelo geo
    python -m equipo52_module.modeling.train_p3 --modelo busqueda
    python -m equipo52_module.modeling.train_p3 --modelo kmeans --k 4
    python -m equipo52_module.modeling.train_p3 --modelo kmeans --k 2
    python -m equipo52_module.modeling.train_p3 --modelo kmeans --k 4 --solo-consolidadas
    python -m equipo52_module.modeling.train_p3 --modelo ward --k 4
    python -m equipo52_module.modeling.train_p3 --modelo hdbscan --min-cluster-size 15
"""

import hashlib
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

from loguru import logger

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")  # oculta un aviso de MLflow que no aplica aquí
import matplotlib
import mlflow  # noqa: E402
from mlflow.data.dataset_source_registry import resolve_dataset_source  # noqa: E402
from mlflow.data.pandas_dataset import PandasDataset  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402
from mlflow.types.utils import _infer_schema  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import HDBSCAN, AgglomerativeClustering, KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import StandardScaler
import typer

from equipo52_module.config import (
    DURACION_MINIMA_SEGUNDOS,
    ESTACIONES_EXCLUIDAS,
    FECHA_FIN,
    FECHA_INICIO,
    FIGURES_DIR,
    MANIFIESTO_FILE,
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
    NOMENCLATURA_FILE,
    PROCESSED_DATA_DIR,
    PROJ_ROOT,
    SEED,
    VERSION_DATOS,
    VIAJES_FILE,
)
from equipo52_module.features import (
    SEGUNDOS_REDONDO_CORTO,
    construir_matriz_p3,
    quitar_viajes_cortos,
    quitar_viajes_redondos_cortos,
)

app = typer.Typer()

# Grupos de características de la matriz (ver features.py); se registran como parámetro.
GRUPOS_CARACTERISTICAS = "actividad,horaria,semanal,rol,tipo_viaje,evolucion"
GEO = ["latitud", "longitud"]

# Barrido de k para K-Means y Ward: el k con mayor silueta se reporta en el nombre del run.
K_MIN, K_MAX = 2, 12
K_GEO = 5          # línea base geográfica: pocas zonas, como las trazaría un planeador a mano
N_INIT = 20        # repeticiones de K-Means con semillas distintas para evitar mínimos locales
REPETICIONES_BOOTSTRAP = 30   # remuestreos de estaciones para la estabilidad bootstrap
PROPORCION_BOOTSTRAP = 0.8
MIN_CLUSTER_SIZE_HDBSCAN = 15  # estaciones mínimas para que HDBSCAN considere un grupo
CORTE_ERAS = "2020-09-01"      # era A: 2014-12 a 2020-08 · era B: 2020-09 a 2026-09
MESES_ESTACION_CONSOLIDADA = 6  # separa las 98 estaciones abiertas en may-ago 2026 del resto

FAMILIAS = {
    "geo": "baseline",
    "kmeans": "kmeans",
    "ward": "ward",
    "hdbscan": "hdbscan",
}

# Rejilla de la búsqueda con nested runs (--modelo busqueda): una corrida hija por combinación.
# Explora el número de grupos k (2 a 6) y si el escalador/centroides se ajustan con toda la red o
# solo con las 371 estaciones consolidadas (> 6 meses activas).
REJILLA_BUSQUEDA = {
    "k": [2, 3, 4, 5, 6],
    "solo_consolidadas": [False, True],
}

# Tipos internos del modelo que MLflow declara confiables al guardarlo con skops.
TIPOS_CONFIABLES = [
    "sklearn.cluster._kmeans.KMeans",
    "sklearn.neighbors._nearest_centroid.NearestCentroid",
    "sklearn.cluster._hdbscan.hdbscan.HDBSCAN",
    "sklearn.cluster._hdbscan._tree.CondensedTree",
    "numpy.dtype",
]

TINTA = "#52514e"


def cargar_estaciones(ruta: Path, excluidas: list[int]) -> pd.DataFrame:
    """Lee capacidad, ubicación y municipio de la nomenclatura, sin excluidas ni capacidad 0."""
    nomenclatura = pd.read_csv(ruta).rename(columns={"id": "estacion_id", "dpcapacity": "capacidad",
                                                     "latitude": "latitud", "longitude": "longitud"})
    nomenclatura = nomenclatura[~nomenclatura["estacion_id"].isin(excluidas) & (nomenclatura["capacidad"] > 0)]
    nomenclatura["municipio"] = nomenclatura["name"].str.extract(r"\(([A-Z]+)-\d+\)")[0].fillna("OTRO")
    return nomenclatura.set_index("estacion_id")[["name", "municipio", "capacidad", "latitud", "longitud"]]


def barrer_k(X: pd.DataFrame, algoritmo: str) -> pd.DataFrame:
    """Silueta de cada k entre K_MIN y K_MAX; el máximo define el número de grupos del run."""
    filas = []
    for k in range(K_MIN, K_MAX + 1):
        if algoritmo == "kmeans":
            etiquetas = KMeans(n_clusters=k, n_init=N_INIT, random_state=SEED).fit_predict(X)
        else:
            etiquetas = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)
        filas.append({
            "k": k,
            "silueta": silhouette_score(X, etiquetas),
            "davies_bouldin": davies_bouldin_score(X, etiquetas),
            "calinski_harabasz": calinski_harabasz_score(X, etiquetas),
        })
    return pd.DataFrame(filas)


def metricas_internas(X: pd.DataFrame, etiquetas: np.ndarray, prefijo: str = "hist") -> dict[str, float]:
    """Cohesión y separación en el espacio de características, con prefijo del conjunto.

    No hay etiqueta real, así que la calidad se mide por dentro: silueta (cohesión vs.
    separación), Davies-Bouldin (traslape entre grupos, menor es mejor) y
    Calinski-Harabasz (dispersión entre grupos contra dentro, mayor es mejor).
    """
    return {
        f"{prefijo}_silhouette": float(silhouette_score(X, etiquetas)),
        f"{prefijo}_davies_bouldin": float(davies_bouldin_score(X, etiquetas)),
        f"{prefijo}_calinski_harabasz": float(calinski_harabasz_score(X, etiquetas)),
    }


def estabilidad_entre_eras(viajes: pd.DataFrame, estaciones: pd.DataFrame, escalador: StandardScaler,
                           modelo: KMeans, columnas: list[str]) -> float:
    """ARI entre los grupos que cada estación obtiene en la era A y en la era B.

    Se recalcula la matriz con los viajes de cada era (mismo escalador y modelo ajustados con
    la historia completa) y se comparan las asignaciones de las estaciones activas en ambas.
    Un ARI alto indica que los grupos no son un artefacto del periodo: las estaciones
    conservan su rol a través del tiempo, incluyendo la pandemia.
    """
    corte = pd.Timestamp(CORTE_ERAS)
    fin_a = corte - pd.Timedelta(days=1)
    matriz_a = construir_matriz_p3(viajes[viajes["Inicio_del_viaje"] < corte], estaciones, fin_a)
    matriz_b = construir_matriz_p3(viajes[viajes["Inicio_del_viaje"] >= corte], estaciones,
                                   pd.Timestamp(FECHA_FIN))
    comunes = matriz_a.index.intersection(matriz_b.index)
    etiquetas_a = modelo.predict(pd.DataFrame(escalador.transform(matriz_a.loc[comunes, columnas]),
                                              columns=columnas, index=comunes))
    etiquetas_b = modelo.predict(pd.DataFrame(escalador.transform(matriz_b.loc[comunes, columnas]),
                                              columns=columnas, index=comunes))
    return float(adjusted_rand_score(etiquetas_a, etiquetas_b))


def estabilidad_bootstrap(X: pd.DataFrame, etiquetas_base: np.ndarray, k: int,
                          algoritmo: str = "kmeans") -> float:
    """ARI promedio al re-entrenar con remuestreos del 80 % de las estaciones.

    Verifica que ningún grupo depende de unas pocas estaciones: si los grupos son reales,
    quitar una quinta parte al azar no debe cambiar la asignación del resto.
    """
    generador = np.random.default_rng(SEED)
    aris = []
    for _ in range(REPETICIONES_BOOTSTRAP):
        muestra = generador.choice(len(X), size=int(len(X) * PROPORCION_BOOTSTRAP), replace=False)
        if algoritmo == "ward":
            etiquetas_m = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X.iloc[muestra])
        else:
            etiquetas_m = KMeans(n_clusters=k, n_init=N_INIT, random_state=SEED).fit(X.iloc[muestra]).labels_
        aris.append(adjusted_rand_score(etiquetas_base[muestra], etiquetas_m))
    return float(np.mean(aris))


def describir_grupos(matriz: pd.DataFrame, etiquetas: np.ndarray,
                     estaciones: pd.DataFrame | None = None) -> pd.DataFrame:
    """Resumen cuantitativo y geográfico de cada grupo: uso, horarios y ubicación municipal."""
    columnas = ["salidas_por_mes", "viajes_por_anclaje_mes", "balance", "flujo_neto_por_mes",
                "prop_viajes_redondos", "duracion_mediana_min", "meses_activa",
                "share_viajes_ultimos_12m", "crecimiento_vs_esperado"]
    con_grupo = matriz.assign(grupo=etiquetas)
    descripcion = con_grupo.groupby("grupo")[columnas].mean().round(3)
    descripcion["fin_de_semana"] = con_grupo.groupby("grupo")[["d5", "d6"]].mean().sum(axis=1).round(3)
    descripcion["pico_manana_07_09"] = con_grupo.groupby("grupo")[["h07", "h08", "h09"]].mean().sum(axis=1).round(3)
    descripcion["pico_tarde_17_19"] = con_grupo.groupby("grupo")[["h17", "h18", "h19"]].mean().sum(axis=1).round(3)
    descripcion["estaciones"] = np.bincount(etiquetas[etiquetas >= 0])
    descripcion["nuevas_2026"] = con_grupo.groupby("grupo")["meses_activa"].apply(
        lambda s: int((s <= MESES_ESTACION_CONSOLIDADA).sum())
    )
    horas = con_grupo.groupby("grupo")[[f"h{h:02d}" for h in range(24)]].mean()
    descripcion["hora_pico"] = horas.idxmax(axis=1).str[1:].astype(int)
    if estaciones is not None and "municipio" in estaciones.columns:
        mun = con_grupo.join(estaciones[["municipio"]])
        for m_cod in ["GDL", "ZPN", "TLQ"]:
            descripcion[f"est_{m_cod}"] = mun.groupby("grupo")["municipio"].apply(
                lambda s, c=m_cod: int((s == c).sum())
            )
    return descripcion.sort_index()


def figura_mapa(matriz: pd.DataFrame, etiquetas: np.ndarray, titulo: str, ruta: Path) -> None:
    """Mapa de Guadalajara con cada estación coloreada por su grupo."""
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    puntos = ax.scatter(matriz["longitud"], matriz["latitud"], c=etiquetas, cmap="tab10",
                        s=30, alpha=0.85, edgecolors="white", linewidths=0.3)
    ax.set_title(titulo, color="#0b0b0b", fontsize=11)
    ax.set_xlabel("Longitud", color=TINTA)
    ax.set_ylabel("Latitud", color=TINTA)
    ax.legend(*puntos.legend_elements(), title="Grupo", loc="best", fontsize=8)
    ax.grid(color="#e4e3df", linewidth=0.6, alpha=0.6)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def figura_perfiles(matriz: pd.DataFrame, etiquetas: np.ndarray, ruta: Path) -> None:
    """Perfil horario promedio de cada grupo: la firma temporal que lo define."""
    horas = matriz.assign(grupo=etiquetas).groupby("grupo")[[f"h{h:02d}" for h in range(24)]].mean()
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    for grupo, fila in horas.iterrows():
        ax.plot(range(24), fila.to_numpy(), marker="o", ms=3.5, linewidth=1.8, label=f"Grupo {grupo}")
    ax.axvspan(7, 9, color="#2ca02c", alpha=0.08, label="Pico mañana (7-9 h)")
    ax.axvspan(17, 19, color="#eb6834", alpha=0.08, label="Pico tarde (17-19 h)")
    ax.set_title("P3 · Proporción de salidas por hora del día, promedio por grupo", color="#0b0b0b", fontsize=11)
    ax.set_xlabel("Hora del día", color=TINTA)
    ax.set_ylabel("Proporción de salidas", color=TINTA)
    ax.set_xticks(range(0, 24, 2))
    ax.grid(color="#e4e3df", linewidth=0.8)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def figura_silueta(barrido: pd.DataFrame, ruta: Path, sil_base: float | None = None) -> None:
    """Silueta por k: evidencia de la elección del número de grupos."""
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.plot(barrido["k"], barrido["silueta"], marker="o", markersize=7, linewidth=2, color="#2a78d6",
            label="Silueta en el espacio de uso")
    if sil_base is not None:
        ax.axhline(sil_base, color=TINTA, ls="--", lw=1, label=f"Baseline geográfico ({sil_base:.3f})")
    mejor = barrido.loc[barrido["silueta"].idxmax()]
    ax.axvline(mejor["k"], color="gray", ls=":", lw=1)
    if 4 in barrido["k"].values:
        ax.axvline(4, color="#1baf7a", ls=":", lw=1, label="Subdivisión operativa (k = 4)")
    ax.set_title(f"P3 · Silueta por número de grupos (máx. global en k = {int(mejor['k'])})",
                 color="#0b0b0b", fontsize=11)
    ax.set_xlabel("Número de grupos (k)", color=TINTA)
    ax.set_ylabel("Coeficiente de silueta", color=TINTA)
    ax.set_xticks(barrido["k"])
    ax.grid(color="#e4e3df", linewidth=0.8)
    ax.legend(fontsize=8, frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def figura_dendrograma(X: pd.DataFrame, ruta: Path) -> None:
    """Dendrograma del enlace Ward: estructura anidada de los grupos para discutir k."""
    fig, ax = plt.subplots(figsize=(11, 4.8))
    dendrogram(linkage(X.to_numpy(), method="ward"), truncate_mode="lastp", p=30, ax=ax)
    ax.set_title("P3 · Dendrograma Ward de estaciones (truncado a 30 ramas)", color="#0b0b0b", fontsize=11)
    ax.set_ylabel("Distancia euclidiana (Ward)", color=TINTA)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def grafica_busqueda(resultados: pd.DataFrame, sil_base_val: float, sil_base_hist: float, ruta: Path) -> None:
    """Silueta en validación (consolidadas) y red completa según k y solo_consolidadas, contra el baseline."""
    colores = {False: "#2a78d6", True: "#eb6834"}
    nombres = {
        False: "Ajuste con las 469 estaciones (solo_consolidadas=False)",
        True: "Ajuste con las 371 consolidadas > 6 m (solo_consolidadas=True)",
    }
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for cons, grupo in resultados.groupby("solo_consolidadas"):
        grupo = grupo.sort_values("k")
        ax.plot(grupo["k"], grupo["val_silhouette"], marker="o", markersize=8, linewidth=2,
                color=colores[cons], markeredgecolor="white", label=nombres[cons])
    ax.axhline(sil_base_val, color=TINTA, linestyle="--", linewidth=1)
    ax.text(resultados["k"].max(), sil_base_val + 0.003,
            f"baseline geográfico (val {sil_base_val:.3f} · hist {sil_base_hist:.3f})",
            va="bottom", ha="right", fontsize=8.5, color=TINTA)
    mejor_k2 = resultados.loc[resultados["val_silhouette"].idxmax()]
    ax.annotate(f"máx. global k=2: {mejor_k2['val_silhouette']:.3f}",
                (mejor_k2["k"], mejor_k2["val_silhouette"]),
                textcoords="offset points", xytext=(38, -4), ha="center", fontsize=8.5, color="#0b0b0b")
    sub_k4 = resultados[resultados["k"] == 4].sort_values("val_silhouette", ascending=False).iloc[0]
    ax.annotate(f"óptimo operativo k=4: {sub_k4['val_silhouette']:.3f}",
                (sub_k4["k"], sub_k4["val_silhouette"]),
                textcoords="offset points", xytext=(0, 10), ha="center", fontsize=8.5, color="#0b0b0b")
    ax.set_xticks(sorted(resultados["k"].unique()))
    ax.set_ylim(0, max(resultados["val_silhouette"].max() + 0.03, 0.18))
    ax.set_xlabel("k: número de grupos de estaciones", color=TINTA)
    ax.set_ylabel("Silueta en validación (371 estaciones consolidadas)", color=TINTA)
    ax.set_title("P3 · Búsqueda de hiperparámetros de K-Means (nested runs)", color="#0b0b0b", fontsize=11)
    ax.grid(color="#e4e3df", linewidth=0.8)
    ax.tick_params(colors=TINTA, length=0)
    for lado in ["top", "right"]:
        ax.spines[lado].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def huella_manifiesto(ruta: Path) -> str:
    """Huella SHA-256 del manifiesto de datos: identifica exactamente qué archivos se usaron."""
    return hashlib.sha256(ruta.read_bytes()).hexdigest()


def commit_actual() -> str:
    """Commit de Git desde el que se ejecuta la corrida."""
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJ_ROOT, capture_output=True, text=True).stdout.strip()


def etiquetas_corrida(modelo: str, etapa: str) -> dict[str, str]:
    """Etiquetas de la corrida para filtrar y comparar en la interfaz de MLflow.

    equipo, pregunta y commit son las del equipo desde la semana 5; question, model_family y
    stage_of_work (baseline, search o candidate) son las que pide la semana 6.
    """
    return {
        "equipo": "52",
        "pregunta": "P3",
        "commit": commit_actual(),
        "question": "p3",
        "model_family": FAMILIAS[modelo],
        "stage_of_work": etapa,
    }


class ConjuntoP3(PandasDataset):
    """Conjunto de datos para mlflow.log_input que infiere el esquema de forma rápida."""

    @property
    def schema(self):
        return _infer_schema(self._df.head(500))


def conjunto_para_mlflow(datos: pd.DataFrame, nombre: str, huella_datos: str) -> ConjuntoP3:
    """Describe la matriz de estaciones (filas, columnas, huella y origen) para mlflow.log_input."""
    huella = hashlib.sha256(f"{huella_datos}|{nombre}|{datos.shape}".encode()).hexdigest()[:8]
    return ConjuntoP3(
        df=datos,
        source=resolve_dataset_source(str(VIAJES_FILE.relative_to(PROJ_ROOT))),
        name=nombre,
        digest=huella,
    )


def mismo_valor(a, b) -> bool:
    """Compara dos valores de parámetro como los guarda MLflow (texto)."""
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return str(a).lower() == str(b).lower()


def registrar_parametros(parametros: dict) -> None:
    """Registra los parámetros de la corrida sin chocar con los que ya registró mlflow run.

    Con mlflow run, la corrida ya trae los parámetros del MLproject (pregunta, modelo, k,
    min_cluster_size, solo_consolidadas, sin_redondos_cortos) y MLflow no permite sobrescribir
    un parámetro con valor distinto. Si cambia (por ejemplo k=0 elegido por silueta), se guarda
    también con sufijo _script.
    """
    existentes = mlflow.get_run(mlflow.active_run().info.run_id).data.params
    nuevos = {}
    for nombre, valor in parametros.items():
        if nombre not in existentes:
            nuevos[nombre] = valor
        elif not mismo_valor(existentes[nombre], valor):
            nuevos[f"{nombre}_script"] = valor
    mlflow.log_params(nuevos)


def carpeta_de_artefactos(carpeta: str) -> Path:
    """Crea las subcarpetas plots/, data/ y config/ dentro de una carpeta temporal."""
    carpeta = Path(carpeta)
    for sub in ["plots", "data", "config"]:
        (carpeta / sub).mkdir()
    return carpeta


def guardar_configuracion(carpeta: Path, configuracion: dict) -> None:
    """config/: la configuración de la corrida en JSON y los archivos de configuración del proyecto."""
    (carpeta / "config" / "corrida_p3.json").write_text(
        json.dumps(configuracion, indent=2, ensure_ascii=False, default=str)
    )
    for archivo in ["equipo52_module/config.py", "MLproject", "python_env.yaml"]:
        if (PROJ_ROOT / archivo).exists():
            shutil.copy(PROJ_ROOT / archivo, carpeta / "config" / Path(archivo).name)


def preparar_entorno_mlflow() -> None:
    """Configura el servidor y experimento de MLflow asegurando compatibilidad con mlflow run."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    experimento = mlflow.set_experiment(MLFLOW_EXPERIMENT)
    run_id_entorno = os.environ.get("MLFLOW_RUN_ID")
    if run_id_entorno:
        try:
            corrida_activa = mlflow.MlflowClient().get_run(run_id_entorno)
            if corrida_activa.info.experiment_id != experimento.experiment_id:
                os.environ.pop("MLFLOW_RUN_ID", None)
        except Exception:
            os.environ.pop("MLFLOW_RUN_ID", None)


def parametros_comunes(modelo: str, matriz: pd.DataFrame, mascara_consolidadas: pd.Series,
                       columnas: list[str], solo_consolidadas: bool, sin_redondos_cortos: bool) -> dict:
    """Parámetros comunes de reproducción que registra toda corrida de P3."""
    return {
        "pregunta": "p3",
        "modelo": modelo,
        "fecha_inicio": FECHA_INICIO,
        "fecha_fin": FECHA_FIN,
        "version_datos": VERSION_DATOS,
        "huella_manifiesto": huella_manifiesto(MANIFIESTO_FILE),
        "semilla": SEED,
        "estaciones_excluidas": len(ESTACIONES_EXCLUIDAS),
        "duracion_minima_segundos": DURACION_MINIMA_SEGUNDOS,
        "sin_redondos_cortos": sin_redondos_cortos,
        "solo_consolidadas": solo_consolidadas,
        "n_estaciones": int(matriz.shape[0]),
        "n_estaciones_consolidadas": int(mascara_consolidadas.sum()),
        "n_estaciones_nuevas_2026": int((~mascara_consolidadas).sum()),
        "n_caracteristicas": len(columnas),
        "grupos_caracteristicas": GRUPOS_CARACTERISTICAS,
        "corte_eras": CORTE_ERAS,
    }


def correr_busqueda(viajes: pd.DataFrame, estaciones: pd.DataFrame, matriz: pd.DataFrame,
                    columnas: list[str], mascara_consolidadas: pd.Series,
                    sin_redondos_cortos: bool) -> None:
    """Búsqueda de hiperparámetros de K-Means con nested runs: una corrida padre y una hija por combinación.

    Evalúa cada combinación de la rejilla (k de 2 a 6 × solo_consolidadas False/True) midiendo
    la silueta de validación en las 371 estaciones consolidadas (> 6 meses), la silueta en toda
    la red (469 estaciones) y la estabilidad tanto entre eras como por remuestreo bootstrap.
    """
    X_full = pd.DataFrame(
        StandardScaler().fit_transform(matriz[columnas]), columns=columnas, index=matriz.index
    )
    X_cons_std = X_full.loc[mascara_consolidadas]

    # Referencia del baseline geográfico (k = 5)
    X_geo = pd.DataFrame(StandardScaler().fit_transform(matriz[GEO]), columns=GEO, index=matriz.index)
    et_geo = KMeans(n_clusters=K_GEO, n_init=N_INIT, random_state=SEED).fit_predict(X_geo)
    sil_base_val = float(silhouette_score(X_cons_std, et_geo[mascara_consolidadas.to_numpy()]))
    sil_base_hist = float(silhouette_score(X_full, et_geo))

    combinaciones = [
        dict(zip(REJILLA_BUSQUEDA, valores))
        for valores in itertools.product(*REJILLA_BUSQUEDA.values())
    ]
    nombre_padre = "p3-busqueda-kmeans"
    parametros = parametros_comunes(
        "busqueda", matriz, mascara_consolidadas, columnas, False, sin_redondos_cortos
    ) | {
        "rejilla": json.dumps(REJILLA_BUSQUEDA),
        "combinaciones": len(combinaciones),
        "metrica_seleccion": "val_silhouette",
        "kmeans_n_init": N_INIT,
        "repeticiones_bootstrap": REPETICIONES_BOOTSTRAP,
    }

    resultados = []
    with mlflow.start_run(run_name=nombre_padre) as padre:
        mlflow.set_tag("mlflow.runName", nombre_padre)
        mlflow.set_tags(etiquetas_corrida("kmeans", "search"))
        registrar_parametros(parametros)
        mlflow.log_input(
            conjunto_para_mlflow(
                matriz.loc[mascara_consolidadas, columnas],
                "p3-estaciones-consolidadas",
                parametros["huella_manifiesto"],
            ),
            context="validation",
        )
        mlflow.log_input(
            conjunto_para_mlflow(
                matriz[columnas],
                "p3-estaciones-red-completa",
                parametros["huella_manifiesto"],
            ),
            context="training",
        )

        for numero, combinacion in enumerate(combinaciones, start=1):
            k_iter = int(combinacion["k"])
            cons_iter = bool(combinacion["solo_consolidadas"])
            sufijo = f"k{k_iter}-consolidadas{cons_iter}"
            logger.info(f"Hija {numero}/{len(combinaciones)} · {combinacion}")
            inicio_hija = time.perf_counter()

            base_fit = matriz.loc[mascara_consolidadas, columnas] if cons_iter else matriz[columnas]
            esc_iter = StandardScaler().fit(base_fit)
            X_iter_all = pd.DataFrame(esc_iter.transform(matriz[columnas]), columns=columnas, index=matriz.index)
            X_iter_fit = X_iter_all.loc[mascara_consolidadas] if cons_iter else X_iter_all

            modelo_iter = KMeans(n_clusters=k_iter, n_init=N_INIT, random_state=SEED).fit(X_iter_fit)
            etiquetas_all = modelo_iter.predict(X_iter_all) if cons_iter else modelo_iter.labels_
            etiquetas_val = etiquetas_all[mascara_consolidadas.to_numpy()]

            metricas_hija = (
                metricas_internas(X_iter_all.loc[mascara_consolidadas], etiquetas_val, prefijo="val")
                | metricas_internas(X_iter_all, etiquetas_all, prefijo="hist")
                | {
                    "estabilidad_eras_ari": estabilidad_entre_eras(
                        viajes, estaciones, esc_iter, modelo_iter, columnas
                    ),
                    "estabilidad_bootstrap_ari": estabilidad_bootstrap(
                        X_iter_fit, modelo_iter.labels_, k_iter
                    ),
                    "segundos_entrenamiento": round(time.perf_counter() - inicio_hija, 2),
                }
            )

            with mlflow.start_run(run_name=f"{nombre_padre}-{sufijo}", nested=True):
                mlflow.set_tags(etiquetas_corrida("kmeans", "search"))
                mlflow.log_params({
                    "k": k_iter,
                    "solo_consolidadas": cons_iter,
                    "kmeans_n_init": N_INIT,
                    "n_estaciones_ajuste": int(len(X_iter_fit)),
                })
                mlflow.log_metrics(metricas_hija)

            resultados.append(combinacion | metricas_hija)
            logger.info(
                f"   val_silhouette {metricas_hija['val_silhouette']:.3f} · "
                f"hist_silhouette {metricas_hija['hist_silhouette']:.3f} · "
                f"eras_ari {metricas_hija['estabilidad_eras_ari']:.3f}"
            )

        resultados_df = pd.DataFrame(resultados).sort_values("val_silhouette", ascending=False)
        mejor = resultados_df.iloc[0]
        mejor_k4 = resultados_df[resultados_df["k"] == 4].iloc[0]
        mejor_combinacion = {"k": int(mejor["k"]), "solo_consolidadas": bool(mejor["solo_consolidadas"])}
        mejor_combinacion_operativa = {"k": 4, "solo_consolidadas": False}

        mlflow.log_metrics({
            "mejor_val_silhouette": float(mejor["val_silhouette"]),
            "mejor_hist_silhouette": float(mejor["hist_silhouette"]),
            "k4_val_silhouette": float(mejor_k4["val_silhouette"]),
            "k4_hist_silhouette": float(
                resultados_df[(resultados_df["k"] == 4) & (~resultados_df["solo_consolidadas"])]["hist_silhouette"].iloc[0]
            ),
            "baseline_val_silhouette": sil_base_val,
            "baseline_hist_silhouette": sil_base_hist,
            "segundos_busqueda": round(float(resultados_df["segundos_entrenamiento"].sum()), 2),
        })
        mlflow.set_tag("mejor_combinacion", json.dumps(mejor_combinacion))
        mlflow.set_tag("mejor_combinacion_operativa", json.dumps(mejor_combinacion_operativa))

        with tempfile.TemporaryDirectory() as temporal:
            carpeta = carpeta_de_artefactos(temporal)
            ruta_plot = carpeta / "plots" / "p3_busqueda_silueta.png"
            grafica_busqueda(resultados_df, sil_base_val, sil_base_hist, ruta_plot)
            resultados_df.round(4).to_csv(carpeta / "data" / "p3_busqueda_resultados.csv", index=False)
            shutil.copy(MANIFIESTO_FILE, carpeta / "data" / MANIFIESTO_FILE.name)
            guardar_configuracion(carpeta, {
                "parametros": parametros,
                "etiquetas": etiquetas_corrida("kmeans", "search"),
                "rejilla": REJILLA_BUSQUEDA,
                "mejor_combinacion": mejor_combinacion,
                "mejor_combinacion_operativa": mejor_combinacion_operativa,
                "columnas": columnas,
            })
            mlflow.log_artifacts(carpeta)

    logger.info("\n" + resultados_df.round(3).to_string(index=False))
    logger.success(
        f"Búsqueda {nombre_padre} ({padre.info.run_id}) registrada con {len(combinaciones)} corridas hijas. "
        f"Máximo global: {mejor_combinacion} (val_silhouette {mejor['val_silhouette']:.3f}) · "
        f"Óptimo operativo: k=4 (val_silhouette {mejor_k4['val_silhouette']:.3f} vs baseline {sil_base_val:.3f})."
    )


@app.command()
def main(modelo: str = typer.Option("kmeans", help="geo, kmeans, ward, hdbscan o busqueda (nested runs)"),
         k: int = typer.Option(0, help="número de grupos; 0 = elegir por silueta"),
         min_cluster_size: int = typer.Option(MIN_CLUSTER_SIZE_HDBSCAN,
                                              help="tamaño mínimo de grupo para hdbscan"),
         solo_consolidadas: bool = typer.Option(
             False,
             help="ajustar el escalador y los centroides solo con estaciones > 6 meses y proyectar las de 2026"),
         sin_redondos_cortos: bool = typer.Option(
             False,
             help="descartar viajes origen=destino de menos de 3 min (devoluciones tardías)")):
    if modelo not in FAMILIAS and modelo != "busqueda":
        raise typer.BadParameter("modelo debe ser geo, kmeans, ward, hdbscan o busqueda")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"P3 · modelo {modelo} · periodo {FECHA_INICIO} a {FECHA_FIN}")

    viajes = pd.read_parquet(VIAJES_FILE)
    viajes = quitar_viajes_cortos(viajes, DURACION_MINIMA_SEGUNDOS)
    if sin_redondos_cortos:
        viajes = quitar_viajes_redondos_cortos(viajes, SEGUNDOS_REDONDO_CORTO)
    estaciones = cargar_estaciones(NOMENCLATURA_FILE, ESTACIONES_EXCLUIDAS)
    matriz = construir_matriz_p3(viajes, estaciones, pd.Timestamp(FECHA_FIN))
    columnas = [c for c in matriz.columns if c not in GEO]
    mascara_consolidadas = matriz["meses_activa"] > MESES_ESTACION_CONSOLIDADA
    logger.info(
        f"Matriz estación × característica: {matriz.shape[0]} estaciones "
        f"({int(mascara_consolidadas.sum())} consolidadas + {int((~mascara_consolidadas).sum())} nuevas 2026) × "
        f"{len(columnas)} características"
    )

    preparar_entorno_mlflow()
    if modelo == "busqueda":
        correr_busqueda(viajes, estaciones, matriz, columnas, mascara_consolidadas, sin_redondos_cortos)
        return

    inicio_entrenamiento = time.perf_counter()
    base_ajuste = matriz.loc[mascara_consolidadas, columnas] if solo_consolidadas else matriz[columnas]
    escalador = StandardScaler().fit(base_ajuste)
    X = pd.DataFrame(escalador.transform(matriz[columnas]), columns=columnas, index=matriz.index)
    X_ajuste = X.loc[mascara_consolidadas] if solo_consolidadas else X
    X_consolidadas = X.loc[mascara_consolidadas]

    # Silueta del baseline geográfico para referencia en las gráficas de silueta vs. k
    X_geo_ref = pd.DataFrame(StandardScaler().fit_transform(matriz[GEO]), columns=GEO, index=matriz.index)
    sil_geo_ref = float(
        silhouette_score(X, KMeans(n_clusters=K_GEO, n_init=N_INIT, random_state=SEED).fit_predict(X_geo_ref))
    )

    parametros = parametros_comunes(
        modelo, matriz, mascara_consolidadas, columnas, solo_consolidadas, sin_redondos_cortos
    ) | {"n_estaciones_ajuste": int(len(X_ajuste))}

    metricas, artefactos_plot = {}, {}
    etiquetas = None
    modelo_entrenado = None
    ejemplo_entrada = X.head(50)

    if modelo == "geo":
        # Línea base: agrupar solo por cercanía geográfica. Si un modelo sobre características
        # de uso no la supera en las métricas internas, no aporta nada sobre un mapa dibujado a mano.
        escalador_geo = StandardScaler().fit(matriz[GEO])
        X_geo = pd.DataFrame(escalador_geo.transform(matriz[GEO]), columns=GEO, index=matriz.index)
        modelo_entrenado = KMeans(n_clusters=K_GEO, n_init=N_INIT, random_state=SEED).fit(X_geo)
        ejemplo_entrada = X_geo.head(50)
        etiquetas = modelo_entrenado.labels_
        parametros.update({
            "k": K_GEO,
            "k_elegido": K_GEO,
            "kmeans_n_init": N_INIT,
            "caracteristicas_modelo": "latitud,longitud",
        })
        metricas.update(metricas_internas(X_consolidadas, etiquetas[mascara_consolidadas.to_numpy()], prefijo="val"))
        metricas.update(metricas_internas(X, etiquetas, prefijo="hist"))
        metricas["estabilidad_eras_ari"] = estabilidad_entre_eras(
            viajes, estaciones, escalador_geo, modelo_entrenado, GEO
        )
        metricas["estabilidad_bootstrap_ari"] = estabilidad_bootstrap(X_geo, etiquetas, K_GEO)
        artefactos_plot["p3_mapa_clusters_geo.png"] = (
            lambda ruta: figura_mapa(matriz, etiquetas, f"P3 · Línea base geográfica (k = {K_GEO})", ruta)
        )
        artefactos_plot["p3_perfiles_horarios_geo.png"] = (
            lambda ruta: figura_perfiles(matriz, etiquetas, ruta)
        )

    elif modelo == "kmeans":
        barrido = barrer_k(X_ajuste, "kmeans")
        k_elegido = k or int(barrido.loc[barrido["silueta"].idxmax(), "k"])
        modelo_entrenado = KMeans(n_clusters=k_elegido, n_init=N_INIT, random_state=SEED).fit(X_ajuste)
        etiquetas = modelo_entrenado.predict(X) if solo_consolidadas else modelo_entrenado.labels_
        parametros.update({
            "k": k_elegido,
            "k_elegido": k_elegido,
            "k_elegido_por": "manual" if k else "silueta",
            "kmeans_n_init": N_INIT,
            "caracteristicas_modelo": "grupos A-F (42 variables)",
            "repeticiones_bootstrap": REPETICIONES_BOOTSTRAP,
        })
        metricas.update(metricas_internas(X_consolidadas, etiquetas[mascara_consolidadas.to_numpy()], prefijo="val"))
        metricas.update(metricas_internas(X, etiquetas, prefijo="hist"))
        if solo_consolidadas:
            metricas.update(metricas_internas(X_ajuste, modelo_entrenado.labels_, prefijo="consolidadas"))
        logger.info("Calculando estabilidad entre eras y por remuestreo...")
        metricas["estabilidad_eras_ari"] = estabilidad_entre_eras(
            viajes, estaciones, escalador, modelo_entrenado, columnas
        )
        metricas["estabilidad_bootstrap_ari"] = estabilidad_bootstrap(
            X_ajuste, modelo_entrenado.labels_, k_elegido
        )
        sufijo = f"_k{k_elegido}" + ("_consolidadas" if solo_consolidadas else "")
        artefactos_plot[f"p3_silueta_vs_k{'_consolidadas' if solo_consolidadas else ''}.png"] = (
            lambda ruta: figura_silueta(barrido, ruta, sil_base=sil_geo_ref)
        )
        artefactos_plot[f"p3_mapa_clusters_kmeans{sufijo}.png"] = (
            lambda ruta: figura_mapa(matriz, etiquetas, f"P3 · Grupos K-Means (k = {k_elegido})", ruta)
        )
        artefactos_plot[f"p3_perfiles_horarios{sufijo}.png"] = (
            lambda ruta: figura_perfiles(matriz, etiquetas, ruta)
        )

    elif modelo == "ward":
        barrido = barrer_k(X_ajuste, "ward")
        k_elegido = k or int(barrido.loc[barrido["silueta"].idxmax(), "k"])
        etiquetas_fit = AgglomerativeClustering(n_clusters=k_elegido, linkage="ward").fit_predict(X_ajuste)
        # NearestCentroid representa los centroides de los grupos de Ward para poder predecir y registrar el modelo
        modelo_entrenado = NearestCentroid().fit(X_ajuste, etiquetas_fit)
        etiquetas = modelo_entrenado.predict(X) if solo_consolidadas else etiquetas_fit
        parametros.update({
            "k": k_elegido,
            "k_elegido": k_elegido,
            "k_elegido_por": "manual" if k else "silueta",
            "linkage": "ward",
            "caracteristicas_modelo": "grupos A-F (42 variables)",
            "repeticiones_bootstrap": REPETICIONES_BOOTSTRAP,
        })
        metricas.update(metricas_internas(X_consolidadas, etiquetas[mascara_consolidadas.to_numpy()], prefijo="val"))
        metricas.update(metricas_internas(X, etiquetas, prefijo="hist"))
        etiquetas_km = KMeans(n_clusters=k_elegido, n_init=N_INIT, random_state=SEED).fit_predict(X)
        metricas["concordancia_ward_kmeans_ari"] = float(adjusted_rand_score(etiquetas, etiquetas_km))
        metricas["estabilidad_eras_ari"] = estabilidad_entre_eras(
            viajes, estaciones, escalador, modelo_entrenado, columnas
        )
        metricas["estabilidad_bootstrap_ari"] = estabilidad_bootstrap(
            X_ajuste, etiquetas_fit, k_elegido, algoritmo="ward"
        )
        sufijo = f"_k{k_elegido}"
        artefactos_plot["p3_dendrograma.png"] = lambda ruta: figura_dendrograma(X_ajuste, ruta)
        artefactos_plot["p3_silueta_vs_k_ward.png"] = (
            lambda ruta: figura_silueta(barrido, ruta, sil_base=sil_geo_ref)
        )
        artefactos_plot[f"p3_mapa_clusters_ward{sufijo}.png"] = (
            lambda ruta: figura_mapa(matriz, etiquetas, f"P3 · Grupos Ward (k = {k_elegido})", ruta)
        )
        artefactos_plot[f"p3_perfiles_horarios_ward{sufijo}.png"] = (
            lambda ruta: figura_perfiles(matriz, etiquetas, ruta)
        )

    elif modelo == "hdbscan":
        # Diagnóstico: estaciones marcadas como ruido (-1) no comparten patrón con ningún grupo;
        # son candidatas a revisión individual por el operador, no forzadas a un grupo.
        etiquetas = HDBSCAN(min_cluster_size=min_cluster_size, copy=True).fit_predict(X)
        nucleo = etiquetas >= 0
        grupos = int(etiquetas[nucleo].max() + 1) if nucleo.any() else 0
        # Se registra un clasificador de centroide (o de grupo único/ruido) para conservar firma e input_example
        if len(np.unique(etiquetas)) >= 2:
            modelo_entrenado = NearestCentroid().fit(X, etiquetas)
        else:
            modelo_entrenado = KMeans(n_clusters=1, n_init=1, random_state=SEED).fit(X)
        parametros.update({
            "min_cluster_size": min_cluster_size,
            "caracteristicas_modelo": "grupos A-F (42 variables)",
            "grupos_encontrados": grupos,
        })
        if grupos >= 2:
            metricas.update(metricas_internas(X[nucleo], etiquetas[nucleo], prefijo="hist"))
            nucleo_val = nucleo & mascara_consolidadas.to_numpy()
            if len(np.unique(etiquetas[nucleo_val])) >= 2:
                metricas.update(metricas_internas(X[nucleo_val], etiquetas[nucleo_val], prefijo="val"))
        else:
            logger.warning(
                "HDBSCAN no encontró dos o más grupos densos: "
                "las estaciones forman un continuo urbano, no islas separadas"
            )
            metricas["val_silhouette"] = 0.0
            metricas["hist_silhouette"] = 0.0
        metricas["proporcion_ruido"] = float((~nucleo).mean())
        artefactos_plot[f"p3_mapa_clusters_hdbscan_mcs{min_cluster_size}.png"] = (
            lambda ruta: figura_mapa(matriz, etiquetas, f"P3 · Grupos HDBSCAN (mcs = {min_cluster_size}, −1 = atípica)", ruta)
        )

    metricas["segundos_entrenamiento"] = round(time.perf_counter() - inicio_entrenamiento, 2)

    nombre_corrida = f"p3-{modelo}" + (f"-k{parametros.get('k_elegido')}" if "k_elegido" in parametros else "")
    if modelo == "hdbscan":
        nombre_corrida += f"-mcs{min_cluster_size}"
    if solo_consolidadas:
        nombre_corrida += "-consolidadas"
    if sin_redondos_cortos:
        nombre_corrida += "-sin-redondos-cortos"

    etapa = "baseline" if modelo == "geo" else "candidate"
    etiquetas_mlflow = etiquetas_corrida(modelo, etapa)

    with mlflow.start_run(run_name=nombre_corrida):
        mlflow.set_tag("mlflow.runName", nombre_corrida)
        mlflow.set_tags(etiquetas_mlflow)
        registrar_parametros(parametros)
        mlflow.log_metrics(metricas)

        # Registro de los conjuntos con mlflow.log_input (descripción y esquema, sin datos personales)
        mlflow.log_input(
            conjunto_para_mlflow(
                base_ajuste,
                f"p3-estaciones-ajuste-{'consolidadas' if solo_consolidadas else 'completo'}",
                parametros["huella_manifiesto"],
            ),
            context="training",
        )
        mlflow.log_input(
            conjunto_para_mlflow(
                matriz.loc[mascara_consolidadas, columnas],
                "p3-estaciones-consolidadas-validacion",
                parametros["huella_manifiesto"],
            ),
            context="validation",
        )
        mlflow.log_input(
            conjunto_para_mlflow(
                matriz[columnas],
                "p3-estaciones-red-completa",
                parametros["huella_manifiesto"],
            ),
            context="testing",
        )

        # Artefactos organizados en plots/, data/ y config/
        with tempfile.TemporaryDirectory() as temporal:
            carpeta = carpeta_de_artefactos(temporal)
            for nombre_fig, hacer in artefactos_plot.items():
                hacer(carpeta / "plots" / nombre_fig)

            shutil.copy(MANIFIESTO_FILE, carpeta / "data" / MANIFIESTO_FILE.name)
            if etiquetas is not None:
                salida = matriz.assign(grupo=etiquetas)
                nombre_matriz = f"p3_matriz_{nombre_corrida.replace('p3-', '')}.csv"
                salida.to_csv(carpeta / "data" / nombre_matriz)
                salida.to_csv(PROCESSED_DATA_DIR / f"p3_matriz_{modelo}.csv")
                if (etiquetas >= 0).any():
                    descripcion = describir_grupos(matriz, etiquetas, estaciones)
                    nombre_desc = f"p3_descripcion_grupos_{nombre_corrida.replace('p3-', '')}.csv"
                    descripcion.to_csv(carpeta / "data" / nombre_desc)
                    descripcion.to_csv(PROCESSED_DATA_DIR / f"p3_descripcion_grupos_{modelo}.csv")
                if modelo == "hdbscan":
                    anomalas = salida[salida["grupo"] == -1]
                    anomalas.to_csv(carpeta / "data" / "p3_estaciones_anomalas.csv")
                    anomalas.to_csv(PROCESSED_DATA_DIR / "p3_estaciones_anomalas.csv")

            guardar_configuracion(carpeta, {
                "parametros": parametros,
                "etiquetas": etiquetas_mlflow,
                "metricas": metricas,
                "columnas": columnas if modelo != "geo" else GEO,
            })
            mlflow.log_artifacts(carpeta)

        if modelo_entrenado is not None:
            firma = infer_signature(ejemplo_entrada, modelo_entrenado.predict(ejemplo_entrada))
            mlflow.sklearn.log_model(
                modelo_entrenado,
                name="modelo",
                signature=firma,
                input_example=ejemplo_entrada.head(5),
                skops_trusted_types=TIPOS_CONFIABLES,
            )

    for nombre, valor in metricas.items():
        logger.info(f"{nombre}: {valor:.3f}")
    logger.success(f"Corrida {nombre_corrida} registrada en el experimento {MLFLOW_EXPERIMENT}")


if __name__ == "__main__":
    app()
