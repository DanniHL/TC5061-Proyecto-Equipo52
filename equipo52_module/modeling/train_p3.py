"""P3 · Agrupamiento de estaciones por patrones de uso y registro en MLflow.

Pregunta: ¿Qué grupos de estaciones comparten patrones de uso y dónde están? Se describe
cada estación con las características de features.py (firma horaria y semanal, nivel de
actividad, rol en la red, tipo de viaje y evolución) calculadas con toda la historia, se
agrupan y se registran parámetros, métricas y artefactos en MLflow.

Uso:
    python equipo52_module/modeling/train_p3.py --modelo geo
    python equipo52_module/modeling/train_p3.py --modelo kmeans
    python equipo52_module/modeling/train_p3.py --modelo kmeans --k 6
    python equipo52_module/modeling/train_p3.py --modelo ward
    python equipo52_module/modeling/train_p3.py --modelo hdbscan
"""

import hashlib
import os
from pathlib import Path
import subprocess

from loguru import logger

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")  # oculta un aviso de MLflow que no aplica aquí
import matplotlib
import mlflow  # noqa: E402

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
from equipo52_module.features import construir_matriz_p3, quitar_viajes_cortos

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

# Tipos internos del modelo que MLflow declara confiables al guardarlo con skops.
TIPOS_CONFIABLES = ["sklearn.cluster._kmeans.KMeans"]


def cargar_estaciones(ruta: Path, excluidas: list[int]) -> pd.DataFrame:
    """Lee capacidad y ubicación de la nomenclatura, sin las estaciones excluidas ni las de capacidad 0."""
    nomenclatura = pd.read_csv(ruta).rename(columns={"id": "estacion_id", "dpcapacity": "capacidad",
                                                     "latitude": "latitud", "longitude": "longitud"})
    nomenclatura = nomenclatura[~nomenclatura["estacion_id"].isin(excluidas) & (nomenclatura["capacidad"] > 0)]
    return nomenclatura.set_index("estacion_id")[["capacidad", "latitud", "longitud"]]


def barrer_k(X: pd.DataFrame, algoritmo: str) -> pd.DataFrame:
    """Silueta de cada k entre K_MIN y K_MAX; el máximo define el número de grupos del run."""
    filas = []
    for k in range(K_MIN, K_MAX + 1):
        if algoritmo == "kmeans":
            etiquetas = KMeans(n_clusters=k, n_init=N_INIT, random_state=SEED).fit_predict(X)
        else:
            etiquetas = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)
        filas.append({"k": k, "silueta": silhouette_score(X, etiquetas)})
    return pd.DataFrame(filas)


def metricas_internas(X: pd.DataFrame, etiquetas: np.ndarray, prefijo: str = "hist") -> dict[str, float]:
    """Cohesión y separación en el espacio de características, con prefijo del conjunto.

    No hay etiqueta real, así que la calidad se mide por dentro: silueta (cohesión vs.
    separación), Davies-Bouldin (traslape entre grupos, menor es mejor) y
    Calinski-Harabasz (dispersión entre grupos contra dentro, mayor es mejor).
    """
    return {
        f"{prefijo}_silhouette": silhouette_score(X, etiquetas),
        f"{prefijo}_davies_bouldin": davies_bouldin_score(X, etiquetas),
        f"{prefijo}_calinski_harabasz": calinski_harabasz_score(X, etiquetas),
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
    return adjusted_rand_score(etiquetas_a, etiquetas_b)


def estabilidad_bootstrap(X: pd.DataFrame, etiquetas_base: np.ndarray, k: int) -> float:
    """ARI promedio al re-entrenar K-Means con remuestreos del 80 % de las estaciones.

    Verifica que ningún grupo depende de unas pocas estaciones: si los grupos son reales,
    quitar una quinta parte al azar no debe cambiar la asignación del resto.
    """
    generador = np.random.default_rng(SEED)
    aris = []
    for _ in range(REPETICIONES_BOOTSTRAP):
        muestra = generador.choice(len(X), size=int(len(X) * PROPORCION_BOOTSTRAP), replace=False)
        modelo = KMeans(n_clusters=k, n_init=N_INIT, random_state=SEED).fit(X.iloc[muestra])
        aris.append(adjusted_rand_score(etiquetas_base[muestra], modelo.labels_))
    return float(np.mean(aris))


def describir_grupos(matriz: pd.DataFrame, etiquetas: np.ndarray) -> pd.DataFrame:
    """Resumen en lenguaje de negocio de cada grupo: tamaño y promedios interpretables."""
    columnas = ["salidas_por_mes", "viajes_por_anclaje_mes", "balance", "flujo_neto_por_mes",
                "prop_viajes_redondos", "duracion_mediana_min", "meses_activa",
                "share_viajes_ultimos_12m", "crecimiento_vs_esperado"]
    descripcion = matriz.assign(grupo=etiquetas).groupby("grupo")[columnas].mean().round(3)
    descripcion["fin_de_semana"] = matriz.assign(grupo=etiquetas).groupby("grupo")[["d5", "d6"]].mean().sum(axis=1)
    descripcion["estaciones"] = np.bincount(etiquetas[etiquetas >= 0])
    horas = matriz.assign(grupo=etiquetas).groupby("grupo")[[f"h{h:02d}" for h in range(24)]].mean()
    descripcion["hora_pico"] = horas.idxmax(axis=1).str[1:].astype(int)
    return descripcion.sort_index()


def figura_mapa(matriz: pd.DataFrame, etiquetas: np.ndarray, titulo: str, ruta: Path) -> None:
    """Mapa de Guadalajara con cada estación coloreada por su grupo."""
    fig, ax = plt.subplots(figsize=(9, 9))
    puntos = ax.scatter(matriz["longitud"], matriz["latitud"], c=etiquetas, cmap="tab10",
                        s=28, alpha=0.85, edgecolors="white", linewidths=0.3)
    ax.set_title(titulo)
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    ax.legend(*puntos.legend_elements(), title="Grupo", loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def figura_perfiles(matriz: pd.DataFrame, etiquetas: np.ndarray, ruta: Path) -> None:
    """Perfil horario promedio de cada grupo: la firma temporal que lo define."""
    horas = matriz.assign(grupo=etiquetas).groupby("grupo")[[f"h{h:02d}" for h in range(24)]].mean()
    fig, ax = plt.subplots(figsize=(10, 5))
    for grupo, fila in horas.iterrows():
        ax.plot(range(24), fila.to_numpy(), marker="o", ms=3, label=f"Grupo {grupo}")
    ax.set_title("Proporción de salidas por hora del día, promedio por grupo")
    ax.set_xlabel("Hora del día")
    ax.set_ylabel("Proporción de salidas")
    ax.set_xticks(range(0, 24, 2))
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def figura_silueta(barrido: pd.DataFrame, ruta: Path) -> None:
    """Silueta por k: evidencia de la elección del número de grupos."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(barrido["k"], barrido["silueta"], marker="o")
    mejor = barrido.loc[barrido["silueta"].idxmax()]
    ax.axvline(mejor["k"], color="gray", ls="--", lw=1)
    ax.set_title(f"Silueta por número de grupos (máx. en k = {int(mejor['k'])})")
    ax.set_xlabel("k")
    ax.set_ylabel("Silueta")
    ax.set_xticks(barrido["k"])
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def figura_dendrograma(X: pd.DataFrame, ruta: Path) -> None:
    """Dendrograma del enlace Ward: estructura anidada de los grupos para discutir k."""
    fig, ax = plt.subplots(figsize=(12, 5))
    dendrogram(linkage(X.to_numpy(), method="ward"), no_labels=True, color_threshold=None, ax=ax)
    ax.set_title("Dendrograma Ward de estaciones")
    ax.set_ylabel("Distancia")
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def huella_manifiesto(ruta: Path) -> str:
    """Huella SHA-256 del manifiesto de datos: identifica exactamente qué archivos se usaron."""
    return hashlib.sha256(ruta.read_bytes()).hexdigest()


def commit_actual() -> str:
    """Commit de Git desde el que se ejecuta la corrida."""
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJ_ROOT, capture_output=True, text=True).stdout.strip()


@app.command()
def main(modelo: str = typer.Option("kmeans", help="geo, kmeans, ward o hdbscan"),
         k: int = typer.Option(0, help="número de grupos; 0 = elegir por silueta"),
         min_cluster_size: int = typer.Option(MIN_CLUSTER_SIZE_HDBSCAN,
                                              help="tamaño mínimo de grupo para hdbscan")):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"P3 · modelo {modelo} · periodo {FECHA_INICIO} a {FECHA_FIN}")

    viajes = pd.read_parquet(VIAJES_FILE)
    viajes = quitar_viajes_cortos(viajes, DURACION_MINIMA_SEGUNDOS)
    estaciones = cargar_estaciones(NOMENCLATURA_FILE, ESTACIONES_EXCLUIDAS)
    matriz = construir_matriz_p3(viajes, estaciones, pd.Timestamp(FECHA_FIN))
    logger.info(f"Matriz estación × característica: {matriz.shape[0]} estaciones × "
                f"{matriz.shape[1] - len(GEO)} características")

    columnas = [c for c in matriz.columns if c not in GEO]
    escalador = StandardScaler().fit(matriz[columnas])
    # X es DataFrame para que el modelo guarde los nombres de las características y MLflow
    # pueda registrar la firma con un ejemplo de entrada real.
    X = pd.DataFrame(escalador.transform(matriz[columnas]), columns=columnas, index=matriz.index)

    parametros = {
        "pregunta": "P3", "modelo": modelo,
        "fecha_inicio": FECHA_INICIO, "fecha_fin": FECHA_FIN,
        "version_datos": VERSION_DATOS, "huella_manifiesto": huella_manifiesto(MANIFIESTO_FILE),
        "semilla": SEED, "estaciones_excluidas": len(ESTACIONES_EXCLUIDAS),
        "duracion_minima_segundos": DURACION_MINIMA_SEGUNDOS,
        "n_estaciones": int(matriz.shape[0]), "n_caracteristicas": len(columnas),
        "grupos_caracteristicas": GRUPOS_CARACTERISTICAS, "corte_eras": CORTE_ERAS,
    }
    metricas, artefactos = {}, {}
    etiquetas = None
    modelo_entrenado = None
    ejemplo_entrada = X.head(5)

    if modelo == "geo":
        # Línea base: agrupar solo por cercanía geográfica. Si un modelo sobre características
        # de uso no la supera en las métricas internas, no aporta nada sobre un mapa dibujado a mano.
        X_geo = pd.DataFrame(StandardScaler().fit_transform(matriz[GEO]),
                             columns=GEO, index=matriz.index)
        modelo_entrenado = KMeans(n_clusters=K_GEO, n_init=N_INIT, random_state=SEED)
        modelo_entrenado.fit(X_geo)
        ejemplo_entrada = X_geo.head(5)
        etiquetas = modelo_entrenado.labels_
        parametros.update({"k": K_GEO, "kmeans_n_init": N_INIT,
                           "caracteristicas_modelo": "latitud,longitud"})
        # La silueta se mide en el espacio de uso: dice si las zonas geográficas también son
        # coherentes en su comportamiento, que es lo que la pregunta pide.
        metricas.update(metricas_internas(X, etiquetas))
        artefactos["p3_mapa_clusters_geo.png"] = (
            lambda ruta: figura_mapa(matriz, etiquetas, "Línea base: grupos por ubicación", ruta))

    elif modelo == "kmeans":
        barrido = barrer_k(X, "kmeans")
        k_elegido = k or int(barrido.loc[barrido["silueta"].idxmax(), "k"])
        modelo_entrenado = KMeans(n_clusters=k_elegido, n_init=N_INIT, random_state=SEED).fit(X)
        etiquetas = modelo_entrenado.labels_
        parametros.update({"k": k_elegido, "k_elegido_por": "manual" if k else "silueta",
                           "kmeans_n_init": N_INIT, "caracteristicas_modelo": "grupos A-F",
                           "repeticiones_bootstrap": REPETICIONES_BOOTSTRAP})
        metricas.update(metricas_internas(X, etiquetas))
        logger.info("Calculando estabilidad entre eras y por remuestreo (tarda unos minutos)...")
        metricas["estabilidad_eras_ari"] = estabilidad_entre_eras(viajes, estaciones, escalador,
                                                                modelo_entrenado, columnas)
        metricas["estabilidad_bootstrap_ari"] = estabilidad_bootstrap(X, etiquetas, k_elegido)
        artefactos["p3_silueta_vs_k.png"] = lambda ruta: figura_silueta(barrido, ruta)
        artefactos["p3_mapa_clusters_kmeans.png"] = (
            lambda ruta: figura_mapa(matriz, etiquetas, f"Grupos K-Means (k = {k_elegido})", ruta))
        artefactos["p3_perfiles_horarios.png"] = (
            lambda ruta: figura_perfiles(matriz, etiquetas, ruta))

    elif modelo == "ward":
        barrido = barrer_k(X, "ward")
        k_elegido = k or int(barrido.loc[barrido["silueta"].idxmax(), "k"])
        etiquetas = AgglomerativeClustering(n_clusters=k_elegido, linkage="ward").fit_predict(X)
        parametros.update({"k": k_elegido, "k_elegido_por": "manual" if k else "silueta",
                           "linkage": "ward", "caracteristicas_modelo": "grupos A-F"})
        metricas.update(metricas_internas(X, etiquetas))
        # Concordancia con K-Means al mismo k: si coinciden, los grupos son robustos al algoritmo.
        etiquetas_km = KMeans(n_clusters=k_elegido, n_init=N_INIT, random_state=SEED).fit_predict(X)
        metricas["concordancia_ward_kmeans_ari"] = adjusted_rand_score(etiquetas, etiquetas_km)
        artefactos["p3_dendrograma.png"] = lambda ruta: figura_dendrograma(X, ruta)
        artefactos["p3_mapa_clusters_ward.png"] = (
            lambda ruta: figura_mapa(matriz, etiquetas, f"Grupos Ward (k = {k_elegido})", ruta))

    elif modelo == "hdbscan":
        # Diagnóstico: estaciones marcadas como ruido (-1) no comparten patrón con ningún grupo;
        # son candidatas a revisión individual por el operador, no forzadas a un grupo.
        etiquetas = HDBSCAN(min_cluster_size=min_cluster_size).fit_predict(X)
        nucleo = etiquetas >= 0
        grupos = int(etiquetas[nucleo].max() + 1) if nucleo.any() else 0
        parametros.update({"min_cluster_size": min_cluster_size, "caracteristicas_modelo": "grupos A-F",
                           "grupos_encontrados": grupos})
        if grupos >= 2:
            metricas.update(metricas_internas(X[nucleo], etiquetas[nucleo]))
        else:
            logger.warning("HDBSCAN no encontró dos o más grupos densos: "
                           "las estaciones forman un continuo, no islas separadas")
        metricas["proporcion_ruido"] = float((~nucleo).mean())
        artefactos["p3_mapa_clusters_hdbscan.png"] = (
            lambda ruta: figura_mapa(matriz, etiquetas, "Grupos HDBSCAN (−1 = atípica)", ruta))

    else:
        raise typer.BadParameter("modelo debe ser geo, kmeans, ward o hdbscan")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    nombre_corrida = f"p3-{modelo}" + (f"-k{parametros.get('k')}" if "k" in parametros else "")
    if modelo == "hdbscan":
        nombre_corrida += f"-mcs{min_cluster_size}"
    with mlflow.start_run(run_name=nombre_corrida):
        mlflow.set_tags({"equipo": "52", "pregunta": "P3", "commit": commit_actual()})
        mlflow.log_params(parametros)
        mlflow.log_metrics(metricas)
        for nombre, hacer in artefactos.items():
            ruta = FIGURES_DIR / nombre
            hacer(ruta)
            mlflow.log_artifact(str(ruta))
        if etiquetas is not None:
            salida = matriz.assign(grupo=etiquetas)
            salida.to_csv(PROCESSED_DATA_DIR / f"p3_matriz_{modelo}.csv")
            mlflow.log_artifact(str(PROCESSED_DATA_DIR / f"p3_matriz_{modelo}.csv"))
            if modelo in ("kmeans", "ward"):
                descripcion = describir_grupos(matriz, etiquetas)
                descripcion.to_csv(PROCESSED_DATA_DIR / f"p3_descripcion_grupos_{modelo}.csv")
                mlflow.log_artifact(str(PROCESSED_DATA_DIR / f"p3_descripcion_grupos_{modelo}.csv"))
            if modelo == "hdbscan":
                anomalas = salida[salida["grupo"] == -1]
                anomalas.to_csv(PROCESSED_DATA_DIR / "p3_estaciones_anomalas.csv")
                mlflow.log_artifact(str(PROCESSED_DATA_DIR / "p3_estaciones_anomalas.csv"))
        if modelo_entrenado is not None:
            mlflow.sklearn.log_model(modelo_entrenado, name="modelo",
                                     input_example=ejemplo_entrada,
                                     skops_trusted_types=TIPOS_CONFIABLES)

    for nombre, valor in metricas.items():
        logger.info(f"{nombre}: {valor:.3f}")
    logger.success(f"Corrida {nombre_corrida} registrada en el experimento {MLFLOW_EXPERIMENT}")


if __name__ == "__main__":
    app()
