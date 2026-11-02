"""Reproducibilidad del entrenamiento de P2: misma semilla y mismos datos → mismo modelo.

"Reproducibility is the ability to obtain the same result twice" (MLOps and Model Governance).
Cada prueba entrena dos veces desde cero con la configuración del champion de la semana 6
(HGB = HistGradientBoostingClassifier de scikit-learn, con peso_fallas=3) y compara
predicciones, probabilidades y métricas.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans
from sklearn.datasets import make_classification
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

from equipo52_module.config import SEED
from equipo52_module.features import construir_matriz_p3, construir_tabla_p2
from equipo52_module.modeling.train_p2 import (
    HIPERPARAMETROS,
    VARIABLES,
    calcular_metricas,
    nuevo_clasificador,
    pesos_por_clase,
)
from equipo52_module.modeling.train_p3 import GEO, N_INIT, barrer_k, estabilidad_bootstrap


@pytest.fixture(scope="module")
def datos_p2():
    """14,000 filas (12,000 de entrenamiento) con las 13 variables de P2 y 3 clases desbalanceadas.

    Con más de 10,000 filas de entrenamiento HGB activa early stopping y separa una validación
    al azar, así que aquí la semilla sí cambia el resultado.
    """
    X, y = make_classification(n_samples=14_000, n_features=len(VARIABLES), n_informative=6,
                               n_classes=3, weights=[0.9, 0.05, 0.05], random_state=0)
    X = pd.DataFrame(X, columns=VARIABLES)
    X.iloc[::50, -2:] = np.nan                      # historial vacío, como las primeras semanas
    return X.iloc[:12_000], y[:12_000], X.iloc[12_000:], y[12_000:]


def entrenar_y_predecir(modelo, datos, hiperparametros):
    X_train, y_train, X_test, y_test = datos
    pesos = pesos_por_clase(hiperparametros["peso_fallas"], balanceado=False)
    clasificador = nuevo_clasificador(modelo, hiperparametros, pesos).fit(X_train, y_train)
    prediccion = clasificador.predict(X_test)
    return prediccion, clasificador.predict_proba(X_test), calcular_metricas(y_test, prediccion, "test")


@pytest.mark.parametrize("modelo", ["hgb", "logistica"])
def test_entrenamiento_p2_reproducible(modelo, datos_p2):
    hiperparametros = {**HIPERPARAMETROS[modelo], "peso_fallas": 3.0}
    pred_1, proba_1, metricas_1 = entrenar_y_predecir(modelo, datos_p2, hiperparametros)
    pred_2, proba_2, metricas_2 = entrenar_y_predecir(modelo, datos_p2, hiperparametros)

    np.testing.assert_array_equal(pred_1, pred_2)
    np.testing.assert_array_equal(proba_1, proba_2)       # idénticas, no solo parecidas
    assert metricas_1 == metricas_2


def test_hgb_la_semilla_controla_el_early_stopping(datos_p2):
    """Control negativo: con otra semilla la validación interna cambia y el modelo también.

    Demuestra que la prueba anterior no pasa por casualidad: el resultado sí depende de la
    semilla, y fijarla (SEED en config.py) es lo que lo vuelve reproducible.
    """
    X_train, y_train, X_test, _ = datos_p2
    propios = {nombre: valor for nombre, valor in HIPERPARAMETROS["hgb"].items() if nombre != "peso_fallas"}
    proba = {semilla: HistGradientBoostingClassifier(**propios, random_state=semilla)
             .fit(X_train, y_train).predict_proba(X_test) for semilla in (SEED, SEED + 1)}
    assert not np.array_equal(proba[SEED], proba[SEED + 1])


def test_variables_p2_deterministas(viajes_pequenos, estaciones_pequenas):
    """El feature engineering también debe ser reproducible: mismos viajes → misma tabla."""
    argumentos = dict(horas=3, umbral=1 / 3, semanas=1)
    primera = construir_tabla_p2(viajes_pequenos, estaciones_pequenas, **argumentos)
    segunda = construir_tabla_p2(viajes_pequenos.sample(frac=1, random_state=7), estaciones_pequenas,
                                 **argumentos)
    pd.testing.assert_frame_equal(primera, segunda)        # el orden de los viajes no importa


def test_semilla_unica_del_proyecto():
    assert SEED == 42

# ------------------------------------------------------------------ P3 (Joel)


@pytest.fixture(scope="module")
def matriz_p3():
    """Estaciones × características de P3 con forma realista, generada con semilla fija.

    120 estaciones en 3 regímenes (parecido a lo que encuentra el champion: núcleo,
    residencial y periférico) × 44 columnas: 42 de uso + latitud y longitud.
    """
    generador = np.random.default_rng(SEED)
    nubes = [generador.normal(loc=centro, scale=0.3, size=(40, 44)) for centro in (0.0, 3.0, 6.0)]
    matriz = pd.DataFrame(np.vstack(nubes), index=pd.Index(range(1, 121), name="estacion_id"))
    matriz.columns = [f"c{i:02d}" for i in range(42)] + GEO
    return matriz


def ajustar_kmeans_champion(matriz):
    """K-Means con la configuración del champion de la semana 6 (p3-kmeans-k4)."""
    columnas = [c for c in matriz.columns if c not in GEO]
    escalador = StandardScaler().fit(matriz[columnas])
    X = escalador.transform(matriz[columnas])
    return KMeans(n_clusters=4, n_init=N_INIT, random_state=SEED).fit(X)


def test_entrenamiento_p3_reproducible(matriz_p3):
    """Dos K-Means con la misma semilla asignan las mismas etiquetas y centroides."""
    primero = ajustar_kmeans_champion(matriz_p3)
    segundo = ajustar_kmeans_champion(matriz_p3)

    np.testing.assert_array_equal(primero.labels_, segundo.labels_)
    np.testing.assert_array_equal(primero.cluster_centers_, segundo.cluster_centers_)
    assert primero.inertia_ == segundo.inertia_


def test_kmeans_la_semilla_cambia_los_centroides(matriz_p3):
    """Control negativo: con otra semilla los centroides de arranque cambian y el resultado también.

    Sin fijar SEED, dos corridas de la semana 6 podrían haber caído en mínimos locales
    distintos: la prueba anterior no pasa por casualidad.
    """
    columnas = [c for c in matriz_p3.columns if c not in GEO]
    X = StandardScaler().fit_transform(matriz_p3[columnas])
    inercias = {semilla: KMeans(n_clusters=4, n_init=N_INIT, random_state=semilla).fit(X).inertia_
                for semilla in (SEED, SEED + 1)}
    assert inercias[SEED] != inercias[SEED + 1]


def test_barrido_k_determinista(matriz_p3):
    """El barrido de silueta que elige k también es reproducible."""
    X = StandardScaler().fit_transform(matriz_p3.drop(columns=GEO))
    pd.testing.assert_frame_equal(barrer_k(X, "kmeans"), barrer_k(X, "kmeans"))


def test_bootstrap_determinista(matriz_p3):
    """La estabilidad bootstrap (remuestreo de estaciones) usa una semilla fija."""
    modelo = ajustar_kmeans_champion(matriz_p3)
    X = pd.DataFrame(StandardScaler().fit_transform(matriz_p3.drop(columns=GEO)),
                     index=matriz_p3.index)
    assert estabilidad_bootstrap(X, modelo.labels_, k=4) == \
        estabilidad_bootstrap(X, modelo.labels_, k=4)


def test_matriz_p3_determinista(viajes_p3, estaciones_p3, fin_p3):
    """El feature engineering también es reproducible: mismos viajes → misma matriz."""
    primera = construir_matriz_p3(viajes_p3, estaciones_p3, fin_p3)
    segunda = construir_matriz_p3(viajes_p3.sample(frac=1, random_state=7), estaciones_p3, fin_p3)
    pd.testing.assert_frame_equal(primera, segunda)        # el orden de los viajes no importa
