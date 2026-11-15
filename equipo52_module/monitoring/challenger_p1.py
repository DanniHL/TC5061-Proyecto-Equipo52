
"""Entrenamiento y registro del challenger para el modelo P1."""

import mlflow
import mlflow.sklearn
import pandas as pd

from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from equipo52_module.config import (
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
)
from equipo52_module.modeling.train_p1 import (
    VARIABLES_CICLICAS,
    agregar_variables_ciclicas,
    agregar_variables_historicas,
    completar_calendario_por_estacion,
    construir_demanda_diaria,
    preparar_viajes_p1,
)


NOMBRE_MODELO = "mibici-equipo52-p1-demanda"
FECHA_FIN_ENTRENAMIENTO = pd.Timestamp("2026-09-30")


def preparar_datos_challenger():
    """Prepara los datos históricos hasta septiembre de 2026."""

    viajes = preparar_viajes_p1()

    demanda = construir_demanda_diaria(viajes)
    datos = completar_calendario_por_estacion(demanda)
    datos = agregar_variables_historicas(datos)
    datos = agregar_variables_ciclicas(datos)

    datos = datos.loc[
        datos["Fecha"] <= FECHA_FIN_ENTRENAMIENTO
    ].copy()

    datos = datos.dropna(
        subset=VARIABLES_CICLICAS + ["Numero_viajes"]
    ).copy()

    X = datos[VARIABLES_CICLICAS].copy()
    y = datos["Numero_viajes"].copy()

    if X.empty:
        raise ValueError("No hay registros disponibles para entrenar el challenger.")

    print("\n=== DATOS DEL CHALLENGER ===")
    print(f"Registros: {len(X):,}")
    print(f"Periodo: {datos['Fecha'].min()} a {datos['Fecha'].max()}")
    print(f"Variables: {len(VARIABLES_CICLICAS)}")

    return X, y


def entrenar_y_registrar_challenger():
    """Entrena el challenger y lo registra como nueva versión del modelo P1."""

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    X, y = preparar_datos_challenger()

    modelo = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("regresion", LinearRegression()),
        ]
    )

    with mlflow.start_run(
        run_name="Challenger_Regresion_Ciclica_P1"
    ) as run:

        modelo.fit(X, y)
        predicciones = modelo.predict(X)

        firma = infer_signature(X, predicciones)

        mlflow.log_params(
            {
                "modelo": "LinearRegression",
                "preprocesamiento": "StandardScaler",
                "fecha_inicio_entrenamiento": "2014-12-01",
                "fecha_fin_entrenamiento": "2026-09-30",
                "registros_entrenamiento": len(X),
                "semilla": 42,
                "version_datos": "2026-10",
                "tipo_experimento": "challenger_regresion_ciclica",
            }
        )

        mlflow.set_tags(
            {
                "equipo": "52",
                "pregunta": "P1",
                "model_family": "LinearRegression",
                "stage_of_work": "challenger",
                "tipo_modelo": "challenger",
            }
        )

        mlflow.sklearn.log_model(
            sk_model=modelo,
            artifact_path="model",
            registered_model_name=NOMBRE_MODELO,
            signature=firma,
            input_example=X.head(5),
        )

        run_id = run.info.run_id

    # Identificar la versión registrada a partir del run_id.
    cliente = MlflowClient()

    versiones = cliente.search_model_versions(
        f"name='{NOMBRE_MODELO}'"
    )

    version_challenger = next(
        (
            version
            for version in versiones
            if version.run_id == run_id
        ),
        None,
    )

    if version_challenger is None:
        raise RuntimeError(
            "El modelo se entrenó, pero no se encontró su versión registrada."
        )

    numero_version = version_challenger.version

    cliente.set_model_version_tag(
        NOMBRE_MODELO,
        numero_version,
        "tipo_modelo",
        "challenger",
    )

    cliente.set_registered_model_alias(
        NOMBRE_MODELO,
        "challenger",
        numero_version,
    )

    print("\n=== CHALLENGER REGISTRADO ===")
    print(f"Modelo: {NOMBRE_MODELO}")
    print(f"Versión: {numero_version}")
    print(f"Run ID: {run_id}")
    print("Alias: challenger")
    print("Alias champion: sin modificaciones")

    return {
        "run_id": run_id,
        "version": numero_version,
        "modelo": NOMBRE_MODELO,
    }


if __name__ == "__main__":
    entrenar_y_registrar_challenger()