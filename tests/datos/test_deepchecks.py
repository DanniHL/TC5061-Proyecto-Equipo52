"""Validación de los datos reales del proyecto con DeepChecks.

DeepChecks 0.19.1 no es compatible con el NumPy 2 ni con el scikit-learn 1.9 del entorno
principal, así que corre en un entorno aparte (.venv-calidad, creado con
`make entorno-calidad` a partir de requirements-calidad.txt). Este módulo prepara las
muestras con el entorno del proyecto —los viajes de septiembre de 2026 y muestras de
100,000 filas de la tabla estación × hora de P2, partidas con partir_en_el_tiempo igual
que en las corridas oficiales—, ejecuta tests/datos/reporte_deepchecks.py con el
intérprete de .venv-calidad y verifica los valores que quedan medidos en
resultados_deepchecks.json. Los reportes HTML se guardan en reports/calidad/.
"""

import json
import subprocess

import pandas as pd
import pytest

from equipo52_module.config import (
    DATA_DIR,
    DURACION_MINIMA_SEGUNDOS,
    ESTACIONES_EXCLUIDAS,
    NOMENCLATURA_FILE,
    PROJ_ROOT,
    REPORTS_DIR,
    SEED,
    VIAJES_FILE,
)
from equipo52_module.features import construir_tabla_p2, quitar_viajes_cortos
from equipo52_module.modeling.train_p2 import (
    HORAS_ANTICIPACION,
    SEMANAS_HISTORIAL,
    UMBRAL_FALLA,
    cargar_estaciones,
    partir_en_el_tiempo,
)

PYTHON_CALIDAD = PROJ_ROOT / ".venv-calidad" / "bin" / "python"
CARPETA_MUESTRAS = DATA_DIR / "calidad"
CARPETA_REPORTES = REPORTS_DIR / "calidad"
RESULTADOS = CARPETA_MUESTRAS / "resultados_deepchecks.json"
N_MUESTRA = 100_000

# Con viajes desde noviembre de 2024 alcanza para las 4 semanas de historial de las filas
# de 2025 y para la prueba de abril a septiembre de 2026.
INICIO_VIAJES_MUESTRA = "2024-11-01"
INICIO_ENTRENAMIENTO_MUESTRA = "2025-01-01"

# Variables que miran hacia atrás: son vacías en las primeras horas o semanas de cada
# estación por diseño (no hay historia que mostrar), no por un problema de los datos.
COLUMNAS_HISTORIAL = {
    "salidas_hora_anterior", "llegadas_hora_anterior",
    "salidas_3_horas_anteriores", "llegadas_3_horas_anteriores",
    "semanas_sin_bicicletas", "semanas_sin_anclajes", "clase_semana_anterior",
}

necesita_datos = pytest.mark.skipif(
    not VIAJES_FILE.exists(),
    reason="data/interim/viajes.parquet no existe: corre primero la ingesta",
)
necesita_calidad = pytest.mark.skipif(
    not PYTHON_CALIDAD.exists(),
    reason="falta .venv-calidad: instálalo con `make entorno-calidad`",
)
pytestmark = [necesita_datos, necesita_calidad]


def muestra_viajes_septiembre() -> pd.DataFrame:
    """Los 490,056 viajes del mes más reciente publicado, con las variables derivadas de la exploración."""
    viajes = pd.read_parquet(VIAJES_FILE)
    septiembre = viajes[(viajes["Inicio_del_viaje"] >= "2026-09-01")
                        & (viajes["Inicio_del_viaje"] < "2026-10-01")].copy()
    return septiembre.assign(
        duracion_min=(septiembre["Fin_del_viaje"] - septiembre["Inicio_del_viaje"])
        .dt.total_seconds() / 60,
        hora=septiembre["Inicio_del_viaje"].dt.hour,
        dia_semana=septiembre["Inicio_del_viaje"].dt.dayofweek,
        es_redondo=septiembre["Origen_Id"] == septiembre["Destino_Id"],
    )


def muestras_p2() -> tuple[pd.DataFrame, pd.DataFrame]:
    """100,000 filas de entrenamiento (2025) y de prueba (abril a septiembre de 2026)."""
    viajes = pd.read_parquet(VIAJES_FILE)
    viajes = viajes[viajes["Inicio_del_viaje"] >= INICIO_VIAJES_MUESTRA]
    viajes = quitar_viajes_cortos(viajes, DURACION_MINIMA_SEGUNDOS)
    estaciones = cargar_estaciones(NOMENCLATURA_FILE, ESTACIONES_EXCLUIDAS)
    tabla = construir_tabla_p2(viajes, estaciones, HORAS_ANTICIPACION, UMBRAL_FALLA,
                               SEMANAS_HISTORIAL)
    partes = partir_en_el_tiempo(tabla, INICIO_ENTRENAMIENTO_MUESTRA)
    return tuple(
        parte.sample(n=N_MUESTRA, random_state=SEED).astype(
            {"clase": "int8", "clase_semana_anterior": "Int8"})
        for parte in (partes["entrenamiento"], partes["prueba"])
    )


@pytest.fixture(scope="module")
def resultados(tmp_path_factory):
    """Genera las muestras, corre DeepChecks en .venv-calidad y devuelve los valores medidos."""
    del tmp_path_factory  # las muestras van a data/calidad/ para poder reutilizarlas
    CARPETA_MUESTRAS.mkdir(parents=True, exist_ok=True)
    entrenamiento, prueba = muestras_p2()
    muestra_viajes_septiembre().to_parquet(CARPETA_MUESTRAS / "muestra_viajes_2026_09.parquet")
    entrenamiento.to_parquet(CARPETA_MUESTRAS / "muestra_p2_entrenamiento.parquet")
    prueba.to_parquet(CARPETA_MUESTRAS / "muestra_p2_prueba.parquet")

    proceso = subprocess.run(
        [str(PYTHON_CALIDAD), str(PROJ_ROOT / "tests" / "datos" / "reporte_deepchecks.py"),
         str(CARPETA_MUESTRAS)],
        cwd=PROJ_ROOT, capture_output=True, text=True, timeout=900,
    )
    assert proceso.returncode == 0, (
        f"reporte_deepchecks.py falló en .venv-calidad:\n{proceso.stdout[-2000:]}\n"
        f"{proceso.stderr[-4000:]}"
    )
    return json.loads(RESULTADOS.read_text())


# ------------------------------------------------------------------ muestras


def test_muestras_tienen_el_tamano_y_periodo_acordados(resultados):
    assert resultados["viajes"]["filas"] == 490_056
    assert resultados["p2"]["filas_entrenamiento"] == N_MUESTRA
    assert resultados["p2"]["filas_prueba"] == N_MUESTRA


# ------------------------------------------------------------------ integridad


def test_viajes_sin_nulos_ni_tipos_mezclados(resultados):
    viajes = resultados["viajes"]
    assert viajes["nulos"] == {}
    assert viajes["tipos_mezclados"] == {}
    assert viajes["caracteres_especiales"] == {}


def test_duplicados_de_viajes_documentados(resultados):
    """Hallazgo: el archivo de septiembre de 2026 publicado por MiBici trae filas repetidas.

    Es un porcentaje pequeño pero real y queda medido: conviene quitarlas en la ingesta.
    """
    viajes = resultados["viajes"]
    assert 0 < viajes["duplicados"] < 0.01          # menos del 1 %, pero no es cero
    assert viajes["duplicados_filas"] > 0


def test_p2_sin_duplicados_y_nulos_solo_en_historial(resultados):
    p2 = resultados["p2"]
    assert p2["duplicados_entrenamiento"] == pytest.approx(0, abs=0.001)
    assert p2["duplicados_prueba"] == pytest.approx(0, abs=0.001)
    assert p2["tipos_mezclados"] == {}
    # los únicos nulos permitidos son los de las variables de historial, que nacen vacías
    # en las primeras horas o semanas de cada estación (por diseño, no por un problema)
    for conjunto in ("nulos_entrenamiento", "nulos_prueba"):
        assert set(p2[conjunto]) <= COLUMNAS_HISTORIAL


# ------------------------------------------------------------------ drift entrenamiento vs. prueba


def test_drift_de_etiqueta_aceptable(resultados):
    drift = resultados["p2"]["drift_etiqueta"]
    assert drift["metodo"] == "Cramer's V"
    assert drift["score"] < 0.15          # el desbalance de clases cambia poco entre periodos


def test_drift_de_variables_esperado_en_mes(resultados):
    drift = resultados["p2"]["drift_variables"]
    mas_drift = max(drift, key=lambda variable: drift[variable]["score"])
    # la prueba solo tiene abril a septiembre: el mes es lo único que debe destacar
    assert mas_drift == "mes"
    assert drift["mes"]["score"] > 0.2
    demas = {v: d["score"] for v, d in drift.items() if v != "mes"}
    assert max(demas.values()) < 0.15


def test_drift_multivariado_explicado_por_el_mes(resultados):
    multi = resultados["p2"]["drift_multivariado"]
    # un clasificador apenas distingue los periodos, y casi toda la señal es el mes
    assert multi["auc_clasificador_dominio"] < 0.8
    mas_importante = max(multi["importancia_variables"],
                         key=multi["importancia_variables"].get)
    assert mas_importante == "mes"


def test_estaciones_nuevas_solo_en_prueba(resultados):
    """Acuerdo del equipo: la prueba incluye las estaciones abiertas de mayo a agosto de 2026."""
    nuevas = resultados["p2"]["categorias_nuevas"]
    assert "estacion_id" in nuevas
    assert nuevas["estacion_id"]["nuevas"] > 50


def test_sin_fuga_de_fechas_entre_conjuntos(resultados):
    assert resultados["p2"]["fuga_fechas_traslape"] == 0
    assert resultados["p2"]["fuga_fechas_duplicados"] == 0


def test_reportes_html_quedan_en_reports(resultados):
    assert (CARPETA_REPORTES / "integridad_viajes.html").stat().st_size > 0
    assert (CARPETA_REPORTES / "drift_entrenamiento_prueba.html").stat().st_size > 0
