"""Pruebas de las funciones de P2 en modeling/train_p2.py: hiperparámetros, pesos, partición y métricas."""

import itertools
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
import typer

from equipo52_module.config import SEED
from equipo52_module.features import NORMAL, SIN_ANCLAJES, SIN_BICICLETAS
from equipo52_module.modeling.train_p2 import (
    HIPERPARAMETROS,
    HORAS_ANTICIPACION,
    INICIOS_ENTRENAMIENTO,
    INICIO_PRUEBA,
    INICIO_VALIDACION,
    REJILLA_BUSQUEDA,
    calcular_metricas,
    entre,
    leer_hiperparametros,
    nuevo_clasificador,
    partir_en_el_tiempo,
    pesos_por_clase,
    predecir_con_base,
)

# ------------------------------------------------------------------ hiperparámetros


def test_leer_hiperparametros_sin_cambios_regresa_omision():
    assert leer_hiperparametros("hgb", "") == HIPERPARAMETROS["hgb"]


def test_leer_hiperparametros_convierte_tipos():
    resultado = leer_hiperparametros("hgb", "max_iter=300, peso_fallas=3")
    assert resultado["max_iter"] == 300 and isinstance(resultado["max_iter"], int)
    assert resultado["peso_fallas"] == 3.0 and isinstance(resultado["peso_fallas"], float)
    assert resultado["learning_rate"] == HIPERPARAMETROS["hgb"]["learning_rate"]   # sin cambio


def test_leer_hiperparametros_no_modifica_la_omision():
    leer_hiperparametros("logistica", "C=5")
    assert HIPERPARAMETROS["logistica"]["C"] == 1.0


def test_leer_hiperparametros_nombre_desconocido_se_detiene():
    with pytest.raises(typer.BadParameter, match="desconocido"):
        leer_hiperparametros("hgb", "max_itr=300")         # error de dedo


# ------------------------------------------------------------------ pesos por clase


@pytest.mark.parametrize("peso, balanceado, esperado", [
    (1.0, False, None),
    (3.0, False, {NORMAL: 1.0, SIN_BICICLETAS: 3.0, SIN_ANCLAJES: 3.0}),
    (1.0, True, "balanced"),
])
def test_pesos_por_clase(peso, balanceado, esperado):
    assert pesos_por_clase(peso, balanceado) == esperado


def test_pesos_por_clase_no_acepta_las_dos_opciones():
    with pytest.raises(typer.BadParameter):
        pesos_por_clase(3.0, balanceado=True)


# ------------------------------------------------------------------ modelos y baseline


def test_nuevo_clasificador_hgb_usa_la_semilla_del_proyecto():
    modelo = nuevo_clasificador("hgb", HIPERPARAMETROS["hgb"], None)
    assert isinstance(modelo, HistGradientBoostingClassifier)
    assert modelo.random_state == SEED
    assert "peso_fallas" not in modelo.get_params()


def test_nuevo_clasificador_logistica_imputa_y_estandariza():
    modelo = nuevo_clasificador("logistica", HIPERPARAMETROS["logistica"], "balanced")
    assert isinstance(modelo, Pipeline)
    assert list(modelo.named_steps) == ["imputar", "escalar", "logistica"]
    assert modelo.named_steps["logistica"].random_state == SEED


def test_predecir_con_base():
    datos = pd.DataFrame({
        "semanas_sin_bicicletas": [0, 3, 1, 2, np.nan],
        "semanas_sin_anclajes":   [0, 0, 2, 2, np.nan],
        "clase_semana_anterior":  [NORMAL, NORMAL, NORMAL, SIN_ANCLAJES, np.nan],
    })
    # normal · sin bicicletas · sin anclajes · empate → gana la semana anterior · sin historia → normal
    assert predecir_con_base(datos).tolist() == [NORMAL, SIN_BICICLETAS, SIN_ANCLAJES, SIN_ANCLAJES, NORMAL]



# ------------------------------------------------------------------ partición en el tiempo


@pytest.fixture
def tabla_por_hora():
    horas = pd.date_range("2025-09-20", "2026-04-05", freq="h")
    clase = pd.array([NORMAL] * len(horas), dtype="Int8")
    clase[-5:] = pd.NA                       # últimas horas sin ventana completa
    return pd.DataFrame({"estacion_id": 1, "hora": horas, "clase": clase})


def test_entre_recorta_las_ultimas_horas(tabla_por_hora):
    parte = entre(tabla_por_hora, "2025-09-20", INICIO_VALIDACION)
    assert parte["hora"].max() == pd.Timestamp(INICIO_VALIDACION) - pd.Timedelta(hours=HORAS_ANTICIPACION)


def test_partir_en_el_tiempo_sin_fuga_entre_conjuntos(tabla_por_hora):
    partes = partir_en_el_tiempo(tabla_por_hora, "2025-09-20")
    assert set(partes) == {"entrenamiento", "validacion", "entrenamiento_final", "prueba"}
    assert partes["entrenamiento"]["hora"].max() < partes["validacion"]["hora"].min()
    assert partes["validacion"]["hora"].max() < partes["prueba"]["hora"].min()
    assert partes["prueba"]["hora"].min() == pd.Timestamp(INICIO_PRUEBA)
    # Ninguna hora de prueba aparece en el entrenamiento final.
    assert not set(partes["prueba"]["hora"]) & set(partes["entrenamiento_final"]["hora"])
    # Las horas sin clase no entran a ningún conjunto.
    assert all(parte["clase"].notna().all() for parte in partes.values())


# ------------------------------------------------------------------ métricas


def test_calcular_metricas_prediccion_perfecta():
    real = np.array([NORMAL, SIN_BICICLETAS, SIN_ANCLAJES, NORMAL])
    metricas = calcular_metricas(real, real, "val")
    assert all(nombre.startswith("val_") for nombre in metricas)
    assert metricas["val_f1_fallas"] == 1.0
    assert metricas["val_recall_sin_bicicletas"] == 1.0


def test_calcular_metricas_modelo_que_nunca_alerta():
    # Caso límite: predecir siempre "normal" da exactitud alta pero F1 de fallas en 0.
    real = np.array([NORMAL] * 8 + [SIN_BICICLETAS, SIN_ANCLAJES])
    metricas = calcular_metricas(real, np.full(10, NORMAL), "test")
    assert metricas["test_exactitud"] == 0.8
    assert metricas["test_f1_fallas"] == 0.0
    assert metricas["test_precision_sin_bicicletas"] == 0.0     # zero_division=0, sin error


# ------------------------------------------------------------------ corridas oficiales de la semana 6
# Comandos mlflow run de P2 en el README de main: base, busqueda y estas tres candidatas.
# No se entrenan aquí (cada una lee 39 millones de viajes); se comprueba que su configuración es válida.

CORRIDAS_OFICIALES = {
    "champion": ("hgb", "completo", "learning_rate=0.1,peso_fallas=3"),
    "logistica": ("logistica", "completo", "peso_fallas=3"),
    "challenger": ("hgb", "postcovid", "learning_rate=0.1,peso_fallas=3"),
}


@pytest.mark.parametrize("modelo, periodo, params", CORRIDAS_OFICIALES.values(), ids=CORRIDAS_OFICIALES.keys())
def test_corridas_oficiales_de_p2_son_validas(modelo, periodo, params):
    assert periodo in INICIOS_ENTRENAMIENTO
    hiperparametros = leer_hiperparametros(modelo, params)
    clasificador = nuevo_clasificador(modelo, hiperparametros, pesos_por_clase(hiperparametros["peso_fallas"], False))
    pesos = [valor for nombre, valor in clasificador.get_params().items() if nombre.endswith("class_weight")]
    assert pesos == [{NORMAL: 1.0, SIN_BICICLETAS: 3.0, SIN_ANCLAJES: 3.0}]


def test_busqueda_oficial_genera_8_combinaciones_validas():
    combinaciones = [dict(zip(REJILLA_BUSQUEDA, valores)) for valores in itertools.product(*REJILLA_BUSQUEDA.values())]
    assert len(combinaciones) == 8
    for combinacion in combinaciones:
        texto = ",".join(f"{nombre}={valor}" for nombre, valor in combinacion.items())
        hiperparametros = leer_hiperparametros("hgb", texto)
        nuevo_clasificador("hgb", hiperparametros, pesos_por_clase(hiperparametros["peso_fallas"], False))