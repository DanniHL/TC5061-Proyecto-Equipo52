"""

P1 - Pronóstico de demanda de MiBici por estación.

Flujo principal:

1. Carga y validación de los viajes.

2. Agregación diaria y análisis de cobertura.

3. Construcción de variables temporales e históricas.

4. División cronológica de entrenamiento y prueba.

5. Entrenamiento y comparación de modelos.

6. Evaluación recursiva a siete días.

7. Registro de métricas, modelos y gráficas en MLflow.

Los días sin registros se conservan como NaN; no se asume que equivalen

automáticamente a días con cero viajes.

"""

import hashlib

import subprocess

from pathlib import Path

import matplotlib.pyplot as plt

import mlflow

import mlflow.sklearn

import numpy as np

import pandas as pd

import json
import re

from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_absolute_error, mean_squared_error

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder, StandardScaler

import argparse

from equipo52_module.config import (

    ESTACIONES_EXCLUIDAS,

    FECHA_FIN,

    FECHA_INICIO,

    FIGURES_DIR,

    MANIFIESTO_FILE,

    MLFLOW_EXPERIMENT,

    MLFLOW_TRACKING_URI,

    PROJ_ROOT,

    SEED,

    VERSION_DATOS,

    VIAJES_FILE,

)

# ---------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------

# El experimento, el registro de MLflow y la semilla vienen de config.py para que P1 y P2

# queden en un solo experimento del proyecto (punto 6 de la actividad).

SEMILLA_P1 = SEED  # Se conserva como referencia del proyecto; este flujo no usa aleatoriedad.

URI_MLFLOW = MLFLOW_TRACKING_URI

EXPERIMENTO_MLFLOW = MLFLOW_EXPERIMENT

FECHA_INICIO_PRUEBA = pd.Timestamp("2026-07-01")

FECHA_FIN_PRUEBA = pd.Timestamp("2026-09-30")

CARPETA_GRAFICAS = FIGURES_DIR  # reports/figures/, estructura de Cookiecutter; archivos con prefijo p1_

VARIABLES_HISTORICAS = [

    "Viajes_dia_anterior",

    "Viajes_hace_7_dias",

    "Promedio_7_dias",

]

VARIABLES_CICLICAS = [

    *VARIABLES_HISTORICAS,

    "Fin_de_semana",

    "Dia_semana_sin",

    "Dia_semana_cos",

    "Mes_sin",

    "Mes_cos",

]

# ---------------------------------------------------------------------
# 1. Carga y preparación de los datos
# ---------------------------------------------------------------------

def preparar_viajes_p1(ruta_viajes=VIAJES_FILE):

    """Arma los viajes de P1 desde el Parquet de la ingesta, con los filtros del notebook de P1.

    Conserva los viajes del periodo del proyecto (FECHA_INICIO a FECHA_FIN) y quita los que

    salen o llegan a una de las 14 estaciones excluidas. Igual que el notebook 1.0-dh-p1, conserva

    los viajes de menos de 1 minuto y los de mismo origen y destino, porque para P1 son inicios

    de viaje registrados. Así el modelo corre con un solo comando, sin depender del notebook.

    """

    viajes = pd.read_parquet(ruta_viajes, columns=["Inicio_del_viaje", "Origen_Id", "Destino_Id"])

    fin = pd.Timestamp(FECHA_FIN) + pd.Timedelta(days=1)

    en_periodo = (viajes["Inicio_del_viaje"] >= pd.Timestamp(FECHA_INICIO)) & (viajes["Inicio_del_viaje"] < fin)

    sin_excluidas = ~viajes["Origen_Id"].isin(ESTACIONES_EXCLUIDAS) & ~viajes["Destino_Id"].isin(ESTACIONES_EXCLUIDAS)

    return viajes.loc[en_periodo & sin_excluidas, ["Inicio_del_viaje", "Origen_Id"]].reset_index(drop=True)

def cargar_viajes_p1(ruta_archivo=None):

    """Carga los viajes de P1 y valida el tipo de la fecha.

    Sin ruta, los prepara desde el Parquet de la ingesta con preparar_viajes_p1().

    """

    viajes = preparar_viajes_p1() if ruta_archivo is None else pd.read_parquet(ruta_archivo)

    if "Inicio_del_viaje" not in viajes.columns or "Origen_Id" not in viajes.columns:

        raise ValueError(

            "El archivo debe contener las columnas 'Inicio_del_viaje' y 'Origen_Id'."

        )

    viajes["Inicio_del_viaje"] = pd.to_datetime(

        viajes["Inicio_del_viaje"], errors="raise"

    )

    print(f"Registros originales: {len(viajes):,}")

    print(f"Columnas: {viajes.shape[1]}")

    print(f"Fecha inicial: {viajes['Inicio_del_viaje'].min()}")

    print(f"Fecha final: {viajes['Inicio_del_viaje'].max()}")

    return viajes

def construir_demanda_diaria(viajes):

    """Cuenta los viajes iniciados por estación y fecha."""

    demanda = (

        viajes.assign(Fecha=viajes["Inicio_del_viaje"].dt.normalize())

        .groupby(["Origen_Id", "Fecha"])

        .size()

        .reset_index(name="Numero_viajes")

        .sort_values(["Origen_Id", "Fecha"])

        .reset_index(drop=True)

    )

    total_original = len(viajes)

    total_agregado = int(demanda["Numero_viajes"].sum())

    duplicados = int(demanda.duplicated(["Origen_Id", "Fecha"]).sum())

    print(f"Registros diarios estación-fecha: {len(demanda):,}")

    print(f"Estaciones: {demanda['Origen_Id'].nunique():,}")

    print(f"Fechas: {demanda['Fecha'].min()} a {demanda['Fecha'].max()}")

    print(f"Viajes originales: {total_original:,}")

    print(f"Viajes agregados: {total_agregado:,}")

    print(f"Diferencia de viajes: {total_original - total_agregado:,}")

    print(f"Duplicados estación-fecha: {duplicados:,}")

    assert total_original == total_agregado, "La agregación no conserva todos los viajes."

    assert duplicados == 0, "Hay más de un registro por estación y fecha."

    return demanda

def resumir_historial_estaciones(demanda):

    """Resume fechas observadas, viajes y cobertura por estación."""

    historial = (

        demanda.groupby("Origen_Id")

        .agg(

            Primera_fecha=("Fecha", "min"),

            Ultima_fecha=("Fecha", "max"),

            Dias_con_registros=("Fecha", "nunique"),

            Total_viajes=("Numero_viajes", "sum"),

        )

        .reset_index()

    )

    historial["Dias_calendario"] = (

        historial["Ultima_fecha"] - historial["Primera_fecha"]

    ).dt.days + 1

    historial["Dias_sin_registro"] = (

        historial["Dias_calendario"] - historial["Dias_con_registros"]

    )

    historial["Cobertura_pct"] = (

        historial["Dias_con_registros"] / historial["Dias_calendario"] * 100

    )

    print(f"Estaciones con historial: {historial['Origen_Id'].nunique():,}")

    print(

        "Cobertura promedio: "

        f"{historial['Cobertura_pct'].mean():.2f}%"

    )

    print(

        "Estaciones con cobertura menor al 90%: "

        f"{(historial['Cobertura_pct'] < 90).sum():,}"

    )

    return historial

def analizar_huecos_por_estacion(demanda, historial, umbral_cobertura=90):

    """Identifica el mayor intervalo sin registros para estaciones con baja cobertura."""

    estaciones_baja = historial.loc[

        historial["Cobertura_pct"] < umbral_cobertura

    ].copy()

    resultados = []

    fechas_por_estacion = {

        estacion: pd.DatetimeIndex(grupo["Fecha"].sort_values().unique())

        for estacion, grupo in demanda.groupby("Origen_Id")

    }

    for estacion in estaciones_baja["Origen_Id"]:

        fechas_observadas = fechas_por_estacion.get(estacion, pd.DatetimeIndex([]))

        if len(fechas_observadas) == 0:

            continue

        calendario = pd.date_range(

            fechas_observadas.min(), fechas_observadas.max(), freq="D"

        )

        faltantes = calendario.difference(fechas_observadas)

        if len(faltantes) == 0:

            continue

        grupos = pd.Series(faltantes).diff().dt.days.ne(1).cumsum()

        huecos = (

            pd.DataFrame({"Fecha": faltantes, "Grupo": grupos.to_numpy()})

            .groupby("Grupo")["Fecha"]

            .agg(Inicio="min", Fin="max", Dias="size")

        )

        mayor = huecos.loc[huecos["Dias"].idxmax()]

        resultados.append({

            "Origen_Id": estacion,

            "Dias_faltantes": len(faltantes),

            "Mayor_hueco_dias": int(mayor["Dias"]),

            "Inicio_mayor_hueco": mayor["Inicio"],

            "Fin_mayor_hueco": mayor["Fin"],

        })

    huecos_df = pd.DataFrame(resultados)

    if not huecos_df.empty:

        huecos_df = huecos_df.sort_values(

            "Mayor_hueco_dias", ascending=False

        ).reset_index(drop=True)

    return estaciones_baja, huecos_df

# ---------------------------------------------------------------------
# 2. Calendario y variables históricas
# ---------------------------------------------------------------------

def agregar_variables_calendario(datos):

    """Añade variables de calendario derivadas de Fecha."""

    resultado = datos.copy()

    resultado["Dia_semana"] = resultado["Fecha"].dt.dayofweek

    resultado["Mes"] = resultado["Fecha"].dt.month

    resultado["Anio"] = resultado["Fecha"].dt.year

    resultado["Dia_mes"] = resultado["Fecha"].dt.day

    resultado["Fin_de_semana"] = (resultado["Dia_semana"] >= 5).astype(int)

    return resultado

def completar_calendario_por_estacion(demanda):

    """

    Completa las fechas entre la primera y última observación de cada estación.

    Los días no observados quedan como NaN, nunca se convierten a cero aquí.

    """

    series = []

    for estacion, grupo in demanda.groupby("Origen_Id", sort=False):

        grupo = grupo.sort_values("Fecha").set_index("Fecha")

        calendario = pd.date_range(grupo.index.min(), grupo.index.max(), freq="D")

        grupo_diario = grupo.reindex(calendario)

        grupo_diario.index.name = "Fecha"

        grupo_diario["Origen_Id"] = estacion

        series.append(grupo_diario)

    if not series:

        raise ValueError("No hay registros para construir el calendario.")

    completo = (

        pd.concat(series)

        .reset_index()

        .sort_values(["Origen_Id", "Fecha"])

        .reset_index(drop=True)

    )

    completo = agregar_variables_calendario(completo)

    print(f"Registros tras completar calendarios: {len(completo):,}")

    print(

        "Días sin registro conservados como faltantes: "

        f"{completo['Numero_viajes'].isna().sum():,}"

    )

    return completo

def agregar_variables_historicas(datos):

    """

    Añade demanda del día anterior, de hace siete días y promedio de los

    siete días anteriores. Los faltantes se propagan sin imputarlos.

    """

    resultado = datos.sort_values(["Origen_Id", "Fecha"]).copy()

    grupo = resultado.groupby("Origen_Id")["Numero_viajes"]

    resultado["Viajes_dia_anterior"] = grupo.shift(1)

    resultado["Viajes_hace_7_dias"] = grupo.shift(7)

    resultado["Promedio_7_dias"] = grupo.transform(

        lambda serie: serie.shift(1).rolling(window=7, min_periods=7).mean()

    )

    return resultado.reset_index(drop=True)

def agregar_variables_ciclicas(datos):

    """Representa día de semana y mes mediante seno y coseno."""

    resultado = datos.copy()

    resultado["Dia_semana_sin"] = np.sin(2 * np.pi * resultado["Dia_semana"] / 7)

    resultado["Dia_semana_cos"] = np.cos(2 * np.pi * resultado["Dia_semana"] / 7)

    resultado["Mes_sin"] = np.sin(2 * np.pi * (resultado["Mes"] - 1) / 12)

    resultado["Mes_cos"] = np.cos(2 * np.pi * (resultado["Mes"] - 1) / 12)

    return resultado

def separar_datos_cronologicamente(datos, fecha_inicio_prueba=FECHA_INICIO_PRUEBA):

    """Filtra registros completos y separa entrenamiento/prueba por fecha."""

    columnas_requeridas = ["Numero_viajes", *VARIABLES_HISTORICAS]

    mascara_valida = datos[columnas_requeridas].notna().all(axis=1)

    datos_validos = datos.loc[mascara_valida].copy()

    entrenamiento = datos_validos.loc[

        datos_validos["Fecha"] < fecha_inicio_prueba

    ].copy()

    prueba = datos_validos.loc[

        datos_validos["Fecha"] >= fecha_inicio_prueba

    ].copy()

    if entrenamiento.empty or prueba.empty:

        raise ValueError(

            "La separación temporal produjo un conjunto vacío. "

            "Revisa las fechas y la cobertura del archivo."

        )

    print(f"Registros válidos: {len(datos_validos):,}")

    print(f"Entrenamiento: {len(entrenamiento):,}")

    print(f"Prueba: {len(prueba):,}")

    print(f"Última fecha de entrenamiento: {entrenamiento['Fecha'].max()}")

    print(f"Primera fecha de prueba: {prueba['Fecha'].min()}")

    print(f"Última fecha de prueba: {prueba['Fecha'].max()}")

    return datos_validos, entrenamiento, prueba

# ---------------------------------------------------------------------
# 3. Métricas, gráficas y registro en MLflow
# ---------------------------------------------------------------------

def calcular_metricas(y_real, y_predicho):

    """Devuelve MAE y RMSE como números de punto flotante."""

    return {

        "MAE": float(mean_absolute_error(y_real, y_predicho)),

        "RMSE": float(np.sqrt(mean_squared_error(y_real, y_predicho))),

    }

def guardar_grafica_comparacion(y_real, y_predicho, titulo, nombre_archivo,

                                etiqueta_real="Demanda real",

                                etiqueta_predicha="Demanda predicha"):

    """Guarda una gráfica compacta de valores reales y predichos."""

    CARPETA_GRAFICAS.mkdir(parents=True, exist_ok=True)

    reales = np.asarray(y_real).reshape(-1)

    predichos = np.asarray(y_predicho).reshape(-1)

    if len(reales) == 0:

        raise ValueError("No hay observaciones para graficar.")

    indices = np.unique(

        np.linspace(0, len(reales) - 1, min(300, len(reales)), dtype=int)

    )

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(indices, reales[indices], label=etiqueta_real, linewidth=1.5)

    ax.plot(indices, predichos[indices], label=etiqueta_predicha, linewidth=1.5)

    ax.set_title(titulo)

    ax.set_xlabel("Observaciones del conjunto de prueba")

    ax.set_ylabel("Número de viajes")

    ax.legend()

    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    ruta = CARPETA_GRAFICAS / nombre_archivo

    fig.savefig(ruta, dpi=150, bbox_inches="tight")

    plt.close(fig)

    return ruta

def configurar_mlflow():

    """Selecciona el servidor y experimento de MLflow."""

    mlflow.set_tracking_uri(URI_MLFLOW)

    mlflow.set_experiment(EXPERIMENTO_MLFLOW)

    print("Tracking URI:", mlflow.get_tracking_uri())

    print("Experimento:", EXPERIMENTO_MLFLOW)

def commit_actual():

    """Commit de Git desde el que se ejecuta la corrida."""

    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJ_ROOT,

                          capture_output=True, text=True).stdout.strip()

def parametros_comunes():

    """Parámetros de reproducción que pide la actividad: fechas, versión de datos y semilla."""

    return {

        "fecha_inicio_entrenamiento": FECHA_INICIO,

        "fecha_fin_prueba": str(FECHA_FIN_PRUEBA.date()),

        "version_datos": VERSION_DATOS,

        "huella_manifiesto": hashlib.sha256(MANIFIESTO_FILE.read_bytes()).hexdigest(),

        "semilla": SEMILLA_P1,

    }

def registrar_resultado_mlflow(
    nombre_run,
    parametros,
    metricas,
    etiquetas=None,
    modelo=None,
    ruta_grafica=None,
    input_example=None,
    metric_prefix="test",
):
    """Registra métricas, modelo y artefactos de una corrida P1 en MLflow."""
    with mlflow.start_run(run_name=nombre_run, nested=True):
        parametros_registro = {**parametros_comunes(), **parametros}
        mlflow.log_params(parametros_registro)
        mlflow.log_metrics({
            f"{metric_prefix}_{nombre.lower()}": float(valor)
            for nombre, valor in metricas.items()
        })

        modelo_nombre = str(parametros.get("modelo", "desconocido"))
        etapa = "baseline" if modelo_nombre.lower() == "baseline" else "candidate"
        etiquetas_finales = {
            "equipo": "52",
            "question": "p1",
            "model_family": modelo_nombre,
            "stage_of_work": etapa,
            "commit": commit_actual(),
        }
        etiquetas_finales.update(etiquetas or {})
        mlflow.set_tags(etiquetas_finales)

        if MANIFIESTO_FILE.exists():
            mlflow.log_artifact(str(MANIFIESTO_FILE), artifact_path="data")

        # Cada corrida obtiene su propio archivo de configuración.
        carpeta_config = CARPETA_GRAFICAS / "config_mlflow"
        carpeta_config.mkdir(parents=True, exist_ok=True)
        nombre_seguro = re.sub(r"[^A-Za-z0-9_.-]+", "_", nombre_run)
        archivo_config = carpeta_config / f"{nombre_seguro}.json"
        with archivo_config.open("w", encoding="utf-8") as archivo:
            json.dump(
                {
                    "parametros": parametros_registro,
                    "metricas": {k: float(v) for k, v in metricas.items()},
                    "etiquetas": etiquetas_finales,
                },
                archivo,
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        mlflow.log_artifact(str(archivo_config), artifact_path="config")

        if modelo is not None:
            if input_example is None or input_example.empty:
                raise ValueError(
                    "Para registrar el modelo se requiere un input_example no vacío."
                )
            ejemplo = input_example.head(3).copy()
            prediccion_ejemplo = modelo.predict(ejemplo)
            firma = mlflow.models.infer_signature(ejemplo, prediccion_ejemplo)
            mlflow.sklearn.log_model(
                modelo,
                name="modelo",
                signature=firma,
                input_example=ejemplo,
            )

        if ruta_grafica is not None and Path(ruta_grafica).exists():
            mlflow.log_artifact(str(ruta_grafica), artifact_path="plots")

# ---------------------------------------------------------------------
# 4. Modelos de un día adelante
# ---------------------------------------------------------------------

def evaluar_baseline(datos_entrenamiento, datos_prueba):

    """Evalúa el baseline que predice la demanda del día anterior."""

    evaluacion = datos_prueba.dropna(

        subset=["Numero_viajes", "Viajes_dia_anterior"]

    ).copy()

    evaluacion["Prediccion_baseline"] = evaluacion["Viajes_dia_anterior"]

    metricas = calcular_metricas(

        evaluacion["Numero_viajes"], evaluacion["Prediccion_baseline"]

    )

    registrar_resultado_mlflow(

        nombre_run="Baseline_P1",

        parametros={

            "modelo": "Baseline",

            "horizonte_dias": 1,

            "fecha_inicio_prueba": str(FECHA_INICIO_PRUEBA.date()),

            "registros_entrenamiento": len(datos_entrenamiento),

            "registros_prueba": len(evaluacion),

        },

        metricas=metricas,

        etiquetas={

            "tipo_experimento": "baseline",

            "pregunta_proyecto": "P1",

            "limitacion": "Evaluacion de un dia adelante; no es el pronostico a siete dias",

        },

    )

    print("Baseline:", metricas)

    return evaluacion, metricas

def entrenar_regresion_lineal(datos_entrenamiento, datos_prueba):

    """Entrena regresión lineal con variables históricas y de calendario."""

    variables = [

        *VARIABLES_HISTORICAS,

        "Dia_semana",

        "Mes",

        "Fin_de_semana",

    ]

    train = datos_entrenamiento.dropna(subset=variables + ["Numero_viajes"]).copy()

    test = datos_prueba.dropna(subset=variables + ["Numero_viajes"]).copy()

    X_train, y_train = train[variables], train["Numero_viajes"]

    X_test, y_test = test[variables], test["Numero_viajes"]

    modelo = Pipeline([

        ("escalador", StandardScaler()),

        ("regresion", LinearRegression()),

    ])

    modelo.fit(X_train, y_train)

    predicciones = modelo.predict(X_test)

    metricas = calcular_metricas(y_test, predicciones)

    ruta = guardar_grafica_comparacion(

        y_test,

        predicciones,

        "Regresión lineal: demanda real vs. predicha en test",

        "p1_regresion_lineal_real_vs_predicho.png",

    )

    registrar_resultado_mlflow(

        "Pipeline_Regresion_Lineal_P1",

        {

            "modelo": "LinearRegression",

            "preprocesamiento": "StandardScaler",

            "registros_entrenamiento": len(X_train),

            "registros_prueba": len(X_test),

            "fecha_inicio_prueba": str(FECHA_INICIO_PRUEBA.date()),

            "horizonte_dias": 1,

        },

        metricas,

        etiquetas={"tipo_experimento": "pipeline"},

        modelo=modelo,

        ruta_grafica=ruta,

        input_example=X_train,
    )

    coeficientes = pd.DataFrame({

        "Variable": variables,

        "Coeficiente": modelo.named_steps["regresion"].coef_,

    })

    coeficientes["Magnitud_absoluta"] = coeficientes["Coeficiente"].abs()

    coeficientes = coeficientes.sort_values(

        "Magnitud_absoluta", ascending=False

    ).reset_index(drop=True)

    print("Regresión lineal:", metricas)

    print("Gráfica guardada en:", ruta)

    return modelo, metricas, coeficientes

def entrenar_regresion_con_estacion(datos_entrenamiento, datos_prueba):

    """Entrena una regresión que codifica Origen_Id como variable categórica."""

    variables_numericas = [

        *VARIABLES_HISTORICAS,

        "Dia_semana",

        "Mes",

        "Fin_de_semana",

    ]

    columnas = variables_numericas + ["Origen_Id"]

    train = datos_entrenamiento.dropna(

        subset=columnas + ["Numero_viajes"]

    ).copy()

    test = datos_prueba.dropna(subset=columnas + ["Numero_viajes"]).copy()

    X_train, y_train = train[columnas].copy(), train["Numero_viajes"]

    X_test, y_test = test[columnas].copy(), test["Numero_viajes"]

    X_train["Origen_Id"] = X_train["Origen_Id"].astype(str)

    X_test["Origen_Id"] = X_test["Origen_Id"].astype(str)

    preprocesador = ColumnTransformer([

        ("numericas", StandardScaler(), variables_numericas),

        ("estacion", OneHotEncoder(handle_unknown="ignore"), ["Origen_Id"]),

    ])

    modelo = Pipeline([

        ("preprocesamiento", preprocesador),

        ("regresion", LinearRegression()),

    ])

    modelo.fit(X_train, y_train)

    predicciones = modelo.predict(X_test)

    metricas = calcular_metricas(y_test, predicciones)

    ruta = guardar_grafica_comparacion(

        y_test,

        predicciones,

        "Regresión lineal con estación: demanda real vs. predicha",

        "p1_regresion_estacion_real_vs_predicho.png",

    )

    registrar_resultado_mlflow(

        "Pipeline_Regresion_Estacion_P1",

        {

            "modelo": "LinearRegression",

            "preprocesamiento": "StandardScaler + OneHotEncoder",

            "registros_entrenamiento": len(X_train),

            "registros_prueba": len(X_test),

            "fecha_inicio_prueba": str(FECHA_INICIO_PRUEBA.date()),

            "horizonte_dias": 1,

        },

        metricas,

        etiquetas={"tipo_experimento": "pipeline_con_estacion"},

        modelo=modelo,

        ruta_grafica=ruta,

        input_example=X_train,
    )

    print("Regresión con estación:", metricas)

    return modelo, metricas

def entrenar_regresion_ciclica(datos_entrenamiento, datos_prueba):

    """Entrena regresión lineal usando codificación cíclica del calendario."""

    train = agregar_variables_ciclicas(datos_entrenamiento)

    test = agregar_variables_ciclicas(datos_prueba)

    train = train.dropna(subset=VARIABLES_CICLICAS + ["Numero_viajes"])

    test = test.dropna(subset=VARIABLES_CICLICAS + ["Numero_viajes"])

    X_train, y_train = train[VARIABLES_CICLICAS], train["Numero_viajes"]

    X_test, y_test = test[VARIABLES_CICLICAS], test["Numero_viajes"]

    modelo = Pipeline([

        ("escalador", StandardScaler()),

        ("regresion", LinearRegression()),

    ])

    modelo.fit(X_train, y_train)

    predicciones = modelo.predict(X_test)

    metricas = calcular_metricas(y_test, predicciones)

    ruta = guardar_grafica_comparacion(

        y_test,

        predicciones,

        "Regresión cíclica: demanda real vs. predicha",

        "p1_regresion_ciclica_real_vs_predicho.png",

    )

    registrar_resultado_mlflow(

        "Pipeline_Regresion_Ciclica_P1",

        {

            "modelo": "LinearRegression",

            "preprocesamiento": "StandardScaler",

            "horizonte_dias": 1,

            "fecha_inicio_prueba": str(FECHA_INICIO_PRUEBA.date()),

            "registros_entrenamiento": len(X_train),

            "registros_prueba": len(X_test),

        },

        metricas,

        etiquetas={"tipo_experimento": "pipeline_regresion_ciclica"},

        modelo=modelo,

        ruta_grafica=ruta,

        input_example=X_train,
    )

    print("Regresión cíclica:", metricas)

    return modelo, metricas

# ---------------------------------------------------------------------
# 5. Pronóstico recursivo a siete días
# ---------------------------------------------------------------------

def construir_matriz_demanda(demanda_completa):

    """Crea una matriz estación x fecha para consultar demanda histórica."""

    return (

        demanda_completa.pivot(

            index="Origen_Id", columns="Fecha", values="Numero_viajes"

        )

        .sort_index(axis=1)

    )

def pronosticar_siete_dias(matriz_demanda, modelo, fecha_inicio="2026-06-30",

                           fecha_fin="2026-09-23"):

    """

    Evalúa orígenes móviles. Cada estación necesita siete días consecutivos

    de historial conocido al inicio; dentro del horizonte se usan predicciones.

    """

    origenes = pd.date_range(fecha_inicio, fecha_fin, freq="D")

    resultados = []

    for fecha_origen in origenes:

        fechas_historial = pd.date_range(end=fecha_origen, periods=7, freq="D")

        historial_origen = (

            matriz_demanda.reindex(columns=fechas_historial)

            .dropna(axis=0, how="any")

        )

        historiales = {

            estacion: historial_origen.loc[estacion].to_numpy(dtype=float).tolist()

            for estacion in historial_origen.index

        }

        for horizonte in range(1, 8):

            fecha_objetivo = fecha_origen + pd.Timedelta(days=horizonte)

            estaciones = list(historiales)

            if not estaciones:

                break

            filas = []

            for estacion in estaciones:

                historial = historiales[estacion][-7:]

                filas.append({

                    "Viajes_dia_anterior": historial[-1],

                    "Viajes_hace_7_dias": historial[0],

                    "Promedio_7_dias": float(np.mean(historial)),

                    "Fin_de_semana": int(fecha_objetivo.dayofweek >= 5),

                    "Dia_semana_sin": np.sin(

                        2 * np.pi * fecha_objetivo.dayofweek / 7

                    ),

                    "Dia_semana_cos": np.cos(

                        2 * np.pi * fecha_objetivo.dayofweek / 7

                    ),

                    "Mes_sin": np.sin(

                        2 * np.pi * (fecha_objetivo.month - 1) / 12

                    ),

                    "Mes_cos": np.cos(

                        2 * np.pi * (fecha_objetivo.month - 1) / 12

                    ),

                })

            X_futuro = pd.DataFrame(filas, columns=VARIABLES_CICLICAS)

            predicciones = modelo.predict(X_futuro)

            reales = matriz_demanda.reindex(

                index=estaciones, columns=[fecha_objetivo]

            )[fecha_objetivo]

            for i, estacion in enumerate(estaciones):

                resultados.append({

                    "Fecha_origen": fecha_origen,

                    "Fecha_pronostico": fecha_objetivo,

                    "Horizonte_dias": horizonte,

                    "Origen_Id": estacion,

                    "Numero_viajes_real": reales.loc[estacion],

                    "Numero_viajes_predicho": float(predicciones[i]),

                })

                historiales[estacion].append(float(predicciones[i]))

                historiales[estacion] = historiales[estacion][-7:]

    pronosticos = pd.DataFrame(resultados)

    if pronosticos.empty:

        raise ValueError("No se generaron pronósticos; revisa las fechas y el historial.")

    print(f"Pronósticos generados: {len(pronosticos):,}")

    print(f"Estaciones evaluadas: {pronosticos['Origen_Id'].nunique():,}")

    print(

        "Fechas pronosticadas:",

        pronosticos["Fecha_pronostico"].min(),

        "a",

        pronosticos["Fecha_pronostico"].max(),

    )

    return pronosticos

def evaluar_pronostico_siete_dias(pronosticos, matriz_demanda):

    """Calcula métricas por horizonte y globales, y compara contra baseline."""

    evaluables = pronosticos.dropna(subset=["Numero_viajes_real"]).copy()

    if evaluables.empty:

        raise ValueError("No hay pronósticos con demanda real para evaluar.")

    filas = []

    for horizonte, grupo in evaluables.groupby("Horizonte_dias"):

        metricas = calcular_metricas(

            grupo["Numero_viajes_real"], grupo["Numero_viajes_predicho"]

        )

        filas.append({

            "Horizonte_dias": int(horizonte),

            **metricas,

            "Registros_evaluados": len(grupo),

        })

    metricas_horizonte = pd.DataFrame(filas)

    metricas_globales = calcular_metricas(

        evaluables["Numero_viajes_real"],

        evaluables["Numero_viajes_predicho"],

    )

    comparacion = evaluables.copy()

    comparacion["Prediccion_baseline"] = [

        matriz_demanda.loc[estacion, fecha]

        for estacion, fecha in zip(

            comparacion["Origen_Id"], comparacion["Fecha_origen"]

        )

    ]

    filas_comparacion = []

    for horizonte, grupo in comparacion.groupby("Horizonte_dias"):

        reales = grupo["Numero_viajes_real"]

        metricas_modelo = calcular_metricas(

            reales, grupo["Numero_viajes_predicho"]

        )

        metricas_baseline = calcular_metricas(reales, grupo["Prediccion_baseline"])

        filas_comparacion.append({

            "Horizonte_dias": int(horizonte),

            "MAE_modelo": metricas_modelo["MAE"],

            "RMSE_modelo": metricas_modelo["RMSE"],

            "MAE_baseline": metricas_baseline["MAE"],

            "RMSE_baseline": metricas_baseline["RMSE"],

            "Registros": len(grupo),

        })

    comparacion_metricas = pd.DataFrame(filas_comparacion)

    resumen_grafica = (

        evaluables.groupby("Horizonte_dias", as_index=False)

        .agg(

            Numero_viajes_real=("Numero_viajes_real", "mean"),

            Numero_viajes_predicho=("Numero_viajes_predicho", "mean"),

        )

        .sort_values("Horizonte_dias")

    )

    CARPETA_GRAFICAS.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(

        resumen_grafica["Horizonte_dias"],

        resumen_grafica["Numero_viajes_real"],

        marker="o",

        label="Demanda real promedio",

    )

    ax.plot(

        resumen_grafica["Horizonte_dias"],

        resumen_grafica["Numero_viajes_predicho"],

        marker="o",

        label="Demanda predicha promedio",

    )

    ax.set_title("Pronóstico recursivo de demanda: horizonte de 7 días")

    ax.set_xlabel("Horizonte de pronóstico (días)")

    ax.set_ylabel("Promedio de viajes")

    ax.set_xticks(range(1, 8))

    ax.legend()

    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    ruta = CARPETA_GRAFICAS / "p1_regresion_ciclica_evaluacion_7_dias.png"

    fig.savefig(ruta, dpi=150, bbox_inches="tight")

    plt.close(fig)

    registrar_resultado_mlflow(

        "Pipeline_Regresion_Ciclica_7Dias_P1",

        {

            "modelo": "Regresion_lineal_con_variables_ciclicas",

            "horizonte_dias": 7,

            "fecha_inicio_prueba": str(FECHA_INICIO_PRUEBA.date()),

            "fecha_fin_prueba": str(FECHA_FIN_PRUEBA.date()),

            "tipo_evaluacion": "recursiva",

        },

        metricas_globales,

        etiquetas={"tipo_experimento": "evaluacion_recursiva_7_dias"},

        ruta_grafica=ruta,


    )

    print("Métricas por horizonte:")

    print(metricas_horizonte.round(3).to_string(index=False))

    print("Métricas globales:", metricas_globales)

    print("Comparación con baseline:")

    print(comparacion_metricas.round(3).to_string(index=False))

    print("Gráfica guardada en:", ruta)

    return evaluables, metricas_horizonte, metricas_globales, comparacion_metricas

# ---------------------------------------------------------------------
# 6. Orquestación del flujo
# ---------------------------------------------------------------------

def buscar_hiperparametros_ciclicos(datos_entrenamiento, datos_prueba):
    """Busca parámetros con validación temporal y registra padre e hijas en MLflow."""
    datos_train = agregar_variables_ciclicas(datos_entrenamiento)
    datos_test = agregar_variables_ciclicas(datos_prueba)
    columnas = VARIABLES_CICLICAS

    fecha_maxima = datos_train["Fecha"].max()
    fecha_corte_validacion = fecha_maxima - pd.Timedelta(days=30)
    train_busqueda = datos_train.loc[datos_train["Fecha"] < fecha_corte_validacion]
    validacion = datos_train.loc[datos_train["Fecha"] >= fecha_corte_validacion]

    train_busqueda = train_busqueda.dropna(subset=columnas + ["Numero_viajes"])
    validacion = validacion.dropna(subset=columnas + ["Numero_viajes"])
    if train_busqueda.empty or validacion.empty:
        raise ValueError("La partición temporal interna para validación quedó vacía.")

    combinaciones = [
        {"fit_intercept": True, "positive": False},
        {"fit_intercept": False, "positive": False},
        {"fit_intercept": True, "positive": True},
        {"fit_intercept": False, "positive": True},
    ]
    resultados_busqueda = []

    with mlflow.start_run(run_name="P1_Busqueda_Hiperparametros_Ciclica", nested=True):
        mlflow.set_tags({
            "equipo": "52",
            "question": "p1",
            "model_family": "LinearRegression",
            "stage_of_work": "search",
            "commit": commit_actual(),
        })
        mlflow.log_params({
            "algoritmo_busqueda": "busqueda_exhaustiva",
            "combinaciones": len(combinaciones),
            "fecha_corte_validacion": str(fecha_corte_validacion.date()),
            "criterio_seleccion": "menor_val_mae",
        })

        for numero, parametros in enumerate(combinaciones, start=1):
            modelo = Pipeline([
                ("escalador", StandardScaler()),
                ("regresion", LinearRegression(**parametros)),
            ])
            X_train = train_busqueda[columnas]
            y_train = train_busqueda["Numero_viajes"]
            X_val = validacion[columnas]
            y_val = validacion["Numero_viajes"]
            modelo.fit(X_train, y_train)
            predicciones = modelo.predict(X_val)
            metricas_val = calcular_metricas(y_val, predicciones)
            ruta = guardar_grafica_comparacion(
                y_val,
                predicciones,
                f"Validación temporal: combinación {numero}",
                f"p1_busqueda_ciclica_validacion_{numero}.png",
            )
            registrar_resultado_mlflow(
                nombre_run=f"Busqueda_Ciclica_Combinacion_{numero}",
                parametros={
                    "modelo": "LinearRegression",
                    "fit_intercept": parametros["fit_intercept"],
                    "positive": parametros["positive"],
                    "fecha_corte_validacion": str(fecha_corte_validacion.date()),
                    "registros_entrenamiento": len(X_train),
                    "registros_validacion": len(X_val),
                },
                metricas=metricas_val,
                etiquetas={"stage_of_work": "search", "tipo_experimento": "busqueda_hiperparametros"},
                modelo=modelo,
                ruta_grafica=ruta,
                input_example=X_train,
                metric_prefix="val",
            )
            resultados_busqueda.append({
                **parametros,
                "val_MAE": metricas_val["MAE"],
                "val_RMSE": metricas_val["RMSE"],
            })

        mejor = min(resultados_busqueda, key=lambda resultado: resultado["val_MAE"])
        parametros_mejores = {
            "fit_intercept": mejor["fit_intercept"],
            "positive": mejor["positive"],
        }

        # Reentrenar con todo el entrenamiento original y evaluar una sola vez en test.
        train_final = datos_train.dropna(subset=columnas + ["Numero_viajes"])
        test_final = datos_test.dropna(subset=columnas + ["Numero_viajes"])
        X_train_final, y_train_final = train_final[columnas], train_final["Numero_viajes"]
        X_test_final, y_test_final = test_final[columnas], test_final["Numero_viajes"]
        modelo_final = Pipeline([
            ("escalador", StandardScaler()),
            ("regresion", LinearRegression(**parametros_mejores)),
        ])
        modelo_final.fit(X_train_final, y_train_final)
        predicciones_test = modelo_final.predict(X_test_final)
        metricas_test = calcular_metricas(y_test_final, predicciones_test)
        ruta_test = guardar_grafica_comparacion(
            y_test_final,
            predicciones_test,
            "Mejor regresión cíclica: demanda real vs. predicha en test",
            "p1_busqueda_ciclica_mejor_modelo_test.png",
        )
        registrar_resultado_mlflow(
            nombre_run="Busqueda_Ciclica_Mejor_Candidato_Test",
            parametros={
                "modelo": "LinearRegression",
                "fit_intercept": parametros_mejores["fit_intercept"],
                "positive": parametros_mejores["positive"],
                "criterio_seleccion": "menor_val_mae",
                "mejor_val_MAE": mejor["val_MAE"],
                "registros_entrenamiento": len(X_train_final),
                "registros_prueba": len(X_test_final),
                "horizonte_dias": 1,
            },
            metricas=metricas_test,
            etiquetas={"stage_of_work": "candidate", "tipo_experimento": "mejor_modelo_busqueda"},
            modelo=modelo_final,
            ruta_grafica=ruta_test,
            input_example=X_train_final,
        )

    print("Mejores hiperparámetros P1:", parametros_mejores)
    print("Métricas de test del candidato seleccionado:", metricas_test)
    return modelo_final, metricas_test, pd.DataFrame(resultados_busqueda)

def main(modelo_seleccionado="todos"):

    """Ejecuta P1 completo o el modelo seleccionado."""

    modelos_validos = {

        "baseline", "lineal", "estacion", "ciclico", "busqueda", "7_dias", "todos"

    }

    if modelo_seleccionado not in modelos_validos:

        raise ValueError(

            f"Modelo no válido: {modelo_seleccionado}. "

            f"Opciones: {sorted(modelos_validos)}"

        )

    configurar_mlflow()

    resultados = {}

    # Mantiene una corrida principal y registra cada resultado como hija.

    with mlflow.start_run(run_name="P1"):

        mlflow.log_param("modelo_seleccionado", modelo_seleccionado)

        viajes = cargar_viajes_p1()

        demanda = construir_demanda_diaria(viajes)

        historial_estaciones = resumir_historial_estaciones(demanda)

        estaciones_baja, resumen_huecos = analizar_huecos_por_estacion(

            demanda, historial_estaciones

        )

        print("\nEstaciones con baja cobertura:")

        print(

            estaciones_baja.sort_values("Cobertura_pct")

            .head(10)

            .to_string(index=False)

        )

        print("\nMayores intervalos sin registros:")

        if not resumen_huecos.empty:

            print(resumen_huecos.head(10).to_string(index=False))

        demanda_completa = completar_calendario_por_estacion(demanda)

        demanda_completa = agregar_variables_historicas(demanda_completa)

        columnas_modelo = ["Numero_viajes", *VARIABLES_HISTORICAS]

        mascara_valida = demanda_completa[columnas_modelo].notna().all(axis=1)

        print(

            f"\nRegistros tras completar calendario: "

            f"{len(demanda_completa):,}"

        )

        print(f"Registros con variables completas: {mascara_valida.sum():,}")

        print(f"Registros no utilizables: {(~mascara_valida).sum():,}")

        print(f"Porcentaje utilizable: {mascara_valida.mean() * 100:.2f}%")

        datos_validos, entrenamiento, prueba = separar_datos_cronologicamente(

            demanda_completa

        )

        resultados.update({

            "demanda_diaria": demanda,

            "historial_estaciones": historial_estaciones,

            "demanda_completa": demanda_completa,

            "datos_validos": datos_validos,

            "entrenamiento": entrenamiento,

            "prueba": prueba,

        })

        ejecutar_todos = modelo_seleccionado == "todos"

        if ejecutar_todos or modelo_seleccionado == "baseline":

            _, metricas = evaluar_baseline(entrenamiento, prueba)

            resultados["metricas_baseline"] = metricas

        if ejecutar_todos or modelo_seleccionado == "lineal":

            _, metricas, coeficientes = entrenar_regresion_lineal(

                entrenamiento, prueba

            )

            resultados["metricas_lineal"] = metricas

            resultados["coeficientes_lineal"] = coeficientes

            print("\nCoeficientes de regresión lineal:")

            print(coeficientes.to_string(index=False))

        if ejecutar_todos or modelo_seleccionado == "estacion":

            _, metricas = entrenar_regresion_con_estacion(

                entrenamiento, prueba

            )

            resultados["metricas_estacion"] = metricas

        necesita_ciclico = (

            ejecutar_todos

            or modelo_seleccionado in {"ciclico", "7_dias"}

        )

        if necesita_ciclico:

            modelo_ciclico, metricas = entrenar_regresion_ciclica(

                entrenamiento, prueba

            )

            resultados["modelo_ciclico"] = modelo_ciclico

            resultados["metricas_ciclicas"] = metricas

        if ejecutar_todos or modelo_seleccionado == "busqueda":
            modelo_busqueda, metricas_busqueda, resultados_busqueda = (
                buscar_hiperparametros_ciclicos(entrenamiento, prueba)
            )
            resultados["modelo_busqueda"] = modelo_busqueda
            resultados["metricas_busqueda"] = metricas_busqueda
            resultados["resultados_busqueda"] = resultados_busqueda

        if ejecutar_todos:

            comparacion = [

                {"Modelo": "Baseline", **resultados["metricas_baseline"]},

                {"Modelo": "Regresión lineal", **resultados["metricas_lineal"]},

                {

                    "Modelo": "Regresión con estación",

                    **resultados["metricas_estacion"],

                },

                {

                    "Modelo": "Regresión cíclica",

                    **resultados["metricas_ciclicas"],

                },

            ]

            print("\nComparación de modelos de un día:")

            print(pd.DataFrame(comparacion).round(3).to_string(index=False))

        if ejecutar_todos or modelo_seleccionado == "7_dias":

            matriz_demanda = construir_matriz_demanda(demanda_completa)

            pronosticos = pronosticar_siete_dias(

                matriz_demanda, resultados["modelo_ciclico"]

            )

            evaluables, metricas_horizonte, metricas_globales, comparacion_7d = (

                evaluar_pronostico_siete_dias(pronosticos, matriz_demanda)

            )

            print(

                "\nPronósticos negativos:",

                int((pronosticos["Numero_viajes_predicho"] < 0).sum()),

            )

            resultados.update({

                "pronosticos_7d": pronosticos,

                "pronosticos_evaluables_7d": evaluables,

                "metricas_horizonte_7d": metricas_horizonte,

                "metricas_globales_7d": metricas_globales,

                "comparacion_metricas_7d": comparacion_7d,

            })

    return resultados

if __name__ == "__main__":

    parser = argparse.ArgumentParser(

        description="Entrenamiento y evaluación de modelos P1 de MiBici."

    )

    parser.add_argument(

        "--modelo",

        choices=["baseline", "lineal", "estacion", "ciclico", "busqueda", "7_dias", "todos"],

        default="todos",

        help="Modelo que se ejecutará. Por defecto, ejecuta el flujo completo.",

    )

    args = parser.parse_args()

    RESULTADOS_P1 = main(modelo_seleccionado=args.modelo)