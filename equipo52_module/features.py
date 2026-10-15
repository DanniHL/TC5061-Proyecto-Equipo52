"""Variables para los modelos del proyecto MiBici.

Cada pregunta agrega aquí sus funciones en su propia sección. Todas trabajan sobre los
viajes que deja la ingesta (data/interim/viajes.parquet) y no usan información del futuro:
una variable de la hora t solo usa datos de horas anteriores a t.
"""

import pandas as pd

# =====================================================================================
# P2 · Riesgo de que una estación se quede sin bicicletas o sin anclajes libres
# Regla definida en notebooks/1.2-rr-p2-exploracion-flujo-neto.ipynb
# =====================================================================================

NORMAL, SIN_BICICLETAS, SIN_ANCLAJES = 0, 1, 2
NOMBRES_CLASES = {NORMAL: "normal", SIN_BICICLETAS: "sin_bicicletas", SIN_ANCLAJES: "sin_anclajes"}
HORAS_POR_SEMANA = 24 * 7


def contar_movimientos_por_hora(viajes: pd.DataFrame) -> pd.DataFrame:
    """Cuenta las bicicletas que salen y llegan a cada estación en cada hora.

    P2 mide el desbalance de cada estación: si salen muchas más de las que llegan se vacía,
    y si llegan muchas más de las que salen se llena.
    """
    salidas = viajes.groupby([viajes["Origen_Id"].rename("estacion_id"),
                              viajes["Inicio_del_viaje"].dt.floor("h").rename("hora")]).size()
    llegadas = viajes.groupby([viajes["Destino_Id"].rename("estacion_id"),
                               viajes["Fin_del_viaje"].dt.floor("h").rename("hora")]).size()
    return pd.concat({"salidas": salidas, "llegadas": llegadas}, axis=1).fillna(0).astype("int32")


def completar_horas_sin_movimiento(movimientos: pd.DataFrame) -> pd.DataFrame:
    """Agrega con 0 las horas sin viajes entre la primera y la última hora con viajes de cada estación.

    Una hora sin viajes también es información (la estación no cambió), y las variables que
    miran horas o semanas hacia atrás necesitan que no falte ninguna hora.
    """
    partes = []
    for estacion_id, grupo in movimientos.groupby(level="estacion_id"):
        horas = grupo.index.get_level_values("hora")
        rango = pd.date_range(horas.min(), horas.max(), freq="h", name="hora")
        partes.append(grupo.droplevel("estacion_id").reindex(rango, fill_value=0).assign(estacion_id=estacion_id))
    return pd.concat(partes).reset_index()[["estacion_id", "hora", "salidas", "llegadas"]]


def sumar_proximas_horas(serie: pd.Series, horas: int) -> pd.Series:
    """Suma, para cada hora, el valor de esa hora y de las siguientes (ventana que avanza hacia adelante)."""
    return serie[::-1].rolling(horas, min_periods=horas).sum()[::-1]


def etiquetar_riesgo(por_hora: pd.DataFrame, capacidad: pd.Series, horas: int, umbral: float) -> pd.Series:
    """Asigna la clase de riesgo de cada estación y hora (lo que el modelo debe anticipar).

    Se revisan la hora actual y las siguientes (horas en total). Si salen al menos 'umbral' de la
    capacidad más bicicletas de las que llegan: sin bicicletas. Si llegan al menos 'umbral' más
    de las que salen: sin anclajes libres. Si no: normal. Las últimas horas de cada estación, sin
    ventana completa, quedan vacías.
    """
    agrupado = por_hora.groupby("estacion_id")
    salen = agrupado["salidas"].transform(sumar_proximas_horas, horas=horas)
    llegan = agrupado["llegadas"].transform(sumar_proximas_horas, horas=horas)
    desbalance = (llegan - salen) / por_hora["estacion_id"].map(capacidad)
    clase = pd.Series(NORMAL, index=por_hora.index, dtype="Int8")
    clase[desbalance <= -umbral] = SIN_BICICLETAS
    clase[desbalance >= umbral] = SIN_ANCLAJES
    clase[desbalance.isna()] = pd.NA
    return clase


def agregar_calendario(por_hora: pd.DataFrame) -> pd.DataFrame:
    """Agrega hora del día, día de la semana, mes y fin de semana.

    El uso de MiBici cambia por hora y día: en días hábiles hay picos de 7 a 9 y de 15 a 19 horas
    (notebook 1.2), y el fin de semana el patrón es otro.
    """
    return por_hora.assign(
        hora_del_dia=por_hora["hora"].dt.hour.astype("int8"),
        dia_semana=por_hora["hora"].dt.dayofweek.astype("int8"),
        mes=por_hora["hora"].dt.month.astype("int8"),
        fin_de_semana=(por_hora["hora"].dt.dayofweek >= 5).astype("int8"),
    )


def agregar_movimientos_previos(por_hora: pd.DataFrame) -> pd.DataFrame:
    """Agrega salidas y llegadas de la hora anterior y de las 3 horas anteriores.

    Es lo que el operador ya vio al empezar la hora: si la estación se viene vaciando o llenando.
    Usa solo horas anteriores (desplazamiento de 1 hora), así no hay información del futuro.
    """
    agrupado = por_hora.groupby("estacion_id")
    previas = {}
    for columna in ["salidas", "llegadas"]:
        anterior = agrupado[columna].shift(1)
        previas[f"{columna}_hora_anterior"] = anterior
        previas[f"{columna}_3_horas_anteriores"] = anterior.groupby(por_hora["estacion_id"]).transform(
            lambda serie: serie.rolling(3, min_periods=3).sum())
    return por_hora.assign(**{nombre: valores.astype("float32") for nombre, valores in previas.items()})


def agregar_historial_semanas(por_hora: pd.DataFrame, semanas: int) -> pd.DataFrame:
    """Agrega en cuántas de las últimas 'semanas' hubo cada tipo de falla, a la misma hora y día de la semana.

    El uso se repite por día de la semana; mirar varias semanas evita que un asueto o un día de
    lluvia defina solo la predicción. La clase de hace una semana usa horas que ya pasaron.
    """
    agrupado = por_hora.groupby("estacion_id")["clase"]
    anteriores = pd.concat([agrupado.shift(HORAS_POR_SEMANA * k) for k in range(1, semanas + 1)], axis=1)
    disponibles = anteriores.notna().sum(axis=1)
    return por_hora.assign(
        semanas_sin_bicicletas=(anteriores == SIN_BICICLETAS).sum(axis=1).astype("float32").where(disponibles == semanas),
        semanas_sin_anclajes=(anteriores == SIN_ANCLAJES).sum(axis=1).astype("float32").where(disponibles == semanas),
        clase_semana_anterior=anteriores.iloc[:, 0],
    )


def construir_tabla_p2(viajes: pd.DataFrame, estaciones: pd.DataFrame, horas: int, umbral: float,
                       semanas: int) -> pd.DataFrame:
    """Arma la tabla estación × hora de P2: clase de riesgo y variables para el modelo.

    'estaciones' trae capacidad, latitud y longitud indexadas por número de estación; las
    estaciones que no están ahí (excluidas o sin capacidad) quedan fuera.
    """
    movimientos = contar_movimientos_por_hora(viajes)
    movimientos = movimientos[movimientos.index.get_level_values("estacion_id").isin(estaciones.index)]
    por_hora = completar_horas_sin_movimiento(movimientos)
    por_hora["clase"] = etiquetar_riesgo(por_hora, estaciones["capacidad"], horas, umbral)
    por_hora = agregar_calendario(por_hora)
    por_hora = agregar_movimientos_previos(por_hora)
    por_hora = agregar_historial_semanas(por_hora, semanas)
    return por_hora.join(estaciones, on="estacion_id")
