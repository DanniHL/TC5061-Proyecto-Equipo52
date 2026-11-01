"""Reporte de calidad de datos con DeepChecks (corre con .venv-calidad, no con .venv).

DeepChecks 0.19.1 no es compatible con el NumPy 2 ni con el scikit-learn 1.9 del entorno
del proyecto, así que vive en un entorno aparte que se instala con `make entorno-calidad`
(requirements-calidad.txt).

Uso:
    .venv-calidad/bin/python tests/datos/reporte_deepchecks.py [carpeta_muestras]

Lee los parquet que prepara tests/datos/test_deepchecks.py en data/calidad/ (o en la
carpeta que se indique), corre los chequeos de integridad de DeepChecks sobre los viajes
de septiembre de 2026 y los de drift entre las muestras de entrenamiento y prueba de la
tabla estación × hora de P2, y escribe:

- resultados_deepchecks.json  con los valores medidos (en la carpeta de las muestras)
- reports/calidad/*.html      el reporte Data Integrity y Train Test Validation
"""

import json
import sys
from pathlib import Path

import pandas as pd
from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import (
    DataDuplicates,
    DateTrainTestLeakageDuplicates,
    DateTrainTestLeakageOverlap,
    FeatureDrift,
    LabelDrift,
    MixedDataTypes,
    MixedNulls,
    MultivariateDrift,
    NewCategoryTrainTest,
    SpecialCharacters,
)
from deepchecks.tabular.suites import data_integrity, train_test_validation

PROJ_ROOT = Path(__file__).resolve().parents[2]
CARPETA_MUESTRAS = Path(sys.argv[1]) if len(sys.argv) > 1 else PROJ_ROOT / "data" / "calidad"
CARPETA_REPORTES = PROJ_ROOT / "reports" / "calidad"

# Identificadores y variables de calendario son categorías, no cantidades: DeepChecks les
# aplica la V de Cramér en lugar de Kolmogorov-Smirnov.
CATEGORICAS_P2 = ["estacion_id", "hora_del_dia", "dia_semana", "mes", "fin_de_semana",
                  "clase_semana_anterior"]
CATEGORICAS_VIAJES = ["Origen_Id", "Destino_Id", "hora", "dia_semana"]


def columnas_con_problemas(resultado, seccion="columns"):
    """Nombres de columnas en las que un chequeo de integridad encontró algo."""
    if seccion in resultado:
        return {columna: detalle for columna, detalle in resultado[seccion].items() if detalle}
    return {columna: detalle for columna, detalle in resultado.items() if detalle}


def main() -> None:
    viajes = pd.read_parquet(CARPETA_MUESTRAS / "muestra_viajes_2026_09.parquet")
    entrenamiento = pd.read_parquet(CARPETA_MUESTRAS / "muestra_p2_entrenamiento.parquet")
    prueba = pd.read_parquet(CARPETA_MUESTRAS / "muestra_p2_prueba.parquet")

    ds_viajes = Dataset(viajes, cat_features=CATEGORICAS_VIAJES)
    ds_entrenamiento = Dataset(entrenamiento.assign(estacion_id=entrenamiento["estacion_id"].astype(str)),
                               label="clase", datetime_name="hora", cat_features=CATEGORICAS_P2)
    ds_prueba = Dataset(prueba.assign(estacion_id=prueba["estacion_id"].astype(str)),
                        label="clase", datetime_name="hora", cat_features=CATEGORICAS_P2)

    resultados = {
        "viajes": {"filas": int(len(viajes))},
        "p2": {"filas_entrenamiento": int(len(entrenamiento)), "filas_prueba": int(len(prueba))},
    }

    # ------------------------------------------------- integridad: viajes de septiembre 2026
    resultados["viajes"]["duplicados"] = DataDuplicates().run(ds_viajes).value
    resultados["viajes"]["duplicados_filas"] = int(round(resultados["viajes"]["duplicados"] * len(viajes)))
    resultados["viajes"]["nulos"] = columnas_con_problemas(MixedNulls().run(ds_viajes).value)
    resultados["viajes"]["tipos_mezclados"] = columnas_con_problemas(MixedDataTypes().run(ds_viajes).value)
    resultados["viajes"]["caracteres_especiales"] = columnas_con_problemas(
        SpecialCharacters().run(ds_viajes).value)

    # ------------------------------------------------- integridad: tabla estación × hora de P2
    resultados["p2"]["duplicados_entrenamiento"] = DataDuplicates().run(ds_entrenamiento).value
    resultados["p2"]["duplicados_prueba"] = DataDuplicates().run(ds_prueba).value
    resultados["p2"]["nulos_entrenamiento"] = columnas_con_problemas(
        MixedNulls().run(ds_entrenamiento).value)
    resultados["p2"]["nulos_prueba"] = columnas_con_problemas(MixedNulls().run(ds_prueba).value)
    resultados["p2"]["tipos_mezclados"] = columnas_con_problemas(
        MixedDataTypes().run(ds_entrenamiento).value)

    # ------------------------------------------------- drift entrenamiento vs. prueba
    drift_variables = FeatureDrift().run(ds_entrenamiento, ds_prueba).value
    resultados["p2"]["drift_variables"] = {
        variable: {"score": detalle["Drift score"], "metodo": detalle["Method"]}
        for variable, detalle in drift_variables.items()
    }
    drift_etiqueta = LabelDrift().run(ds_entrenamiento, ds_prueba).value
    resultados["p2"]["drift_etiqueta"] = {"score": drift_etiqueta["Drift score"],
                                          "metodo": drift_etiqueta["Method"]}

    multivariado = MultivariateDrift().run(ds_entrenamiento, ds_prueba).value
    resultados["p2"]["drift_multivariado"] = {
        "auc_clasificador_dominio": multivariado["domain_classifier_auc"],
        "importancia_variables": multivariado["domain_classifier_feature_importance"],
    }

    nuevas = NewCategoryTrainTest().run(ds_entrenamiento, ds_prueba).value
    resultados["p2"]["categorias_nuevas"] = (
        {str(feature): {"nuevas": int(fila["# New Categories"]),
                        "proporcion": float(fila["Ratio of New Categories"]),
                        "categorias": [str(c) for c in fila["New categories"]][:20]}
         for feature, fila in nuevas.iterrows()}
        if isinstance(nuevas, pd.DataFrame) and not nuevas.empty else {}
    )

    resultados["p2"]["fuga_fechas_traslape"] = DateTrainTestLeakageOverlap().run(
        ds_entrenamiento, ds_prueba).value
    resultados["p2"]["fuga_fechas_duplicados"] = DateTrainTestLeakageDuplicates().run(
        ds_entrenamiento, ds_prueba).value

    salida = CARPETA_MUESTRAS / "resultados_deepchecks.json"
    salida.write_text(json.dumps(resultados, indent=2, ensure_ascii=False, default=str))
    print(f"Resultados en {salida}")
    print(json.dumps(resultados, indent=2, ensure_ascii=False, default=str)[:3000])

    # ------------------------------------------------- reportes HTML para la documentación
    # deepchecks crea copias "nombre (N).html" si el archivo ya existe: se borran antes
    CARPETA_REPORTES.mkdir(parents=True, exist_ok=True)
    for reporte in ["integridad_viajes.html", "drift_entrenamiento_prueba.html"]:
        (CARPETA_REPORTES / reporte).unlink(missing_ok=True)
    data_integrity().run(ds_viajes).save_as_html(
        str(CARPETA_REPORTES / "integridad_viajes.html"), as_widget=False
    )
    train_test_validation().run(ds_entrenamiento, ds_prueba).save_as_html(
        str(CARPETA_REPORTES / "drift_entrenamiento_prueba.html"), as_widget=False
    )
    print(f"Reportes HTML en {CARPETA_REPORTES}")


if __name__ == "__main__":
    main()
