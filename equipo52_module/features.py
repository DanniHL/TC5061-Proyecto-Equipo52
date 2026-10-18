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


def quitar_viajes_cortos(viajes: pd.DataFrame, segundos_minimos: int) -> pd.DataFrame:
    """Quita los viajes que duran menos de 'segundos_minimos' (acuerdo del equipo: 1 minuto).

    No son viajes reales: la mayoría sale y regresa a la misma estación porque la bicicleta se
    devuelve de inmediato. En P2 casi no cambian el desbalance, pero se quitan para que las tres
    preguntas usen los mismos viajes.
    """
    duracion = (viajes["Fin_del_viaje"] - viajes["Inicio_del_viaje"]).dt.total_seconds()
    return viajes[duracion >= segundos_minimos]


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


# =====================================================================================
# P3 · Grupos de estaciones con patrones de uso similares
# =====================================================================================
#
# Cada estación se describe con características calculadas solo con los viajes que recibe
# 'construir_matriz_p3'. Como las estaciones no llevan el mismo tiempo en el sistema (14 se
# dieron de baja y 100 abrieron en 2026), los conteos se dividen entre sus meses activos y
# las firmas temporales son proporciones: así una estación nueva compite parejo con una de 2014.

INICIO_POSTCOVID = pd.Timestamp("2023-05-09")  # fin de la emergencia sanitaria (misma fecha que P2)


def meses_entre(inicio: pd.Series, fin: pd.Timestamp) -> pd.Series:
    """Meses calendario desde cada fecha de 'inicio' hasta 'fin', inclusive."""
    return (fin.year - inicio.dt.year) * 12 + (fin.month - inicio.dt.month) + 1


def primer_viaje_por_estacion(viajes: pd.DataFrame) -> pd.Series:
    """Primer viaje observado de cada estación, como origen o como destino.

    Una estación 'nace' cuando registra su primer viaje; mirar ambos lados evita que una
    estación parezca más vieja o más nueva por el orden en que se usó.
    """
    primera_salida = viajes.groupby("Origen_Id")["Inicio_del_viaje"].min()
    primera_llegada = viajes.groupby("Destino_Id")["Fin_del_viaje"].min()
    return pd.concat([primera_salida, primera_llegada], axis=1).min(axis=1)


def nivel_de_actividad(viajes: pd.DataFrame, meses_activa: pd.Series,
                        capacidad: pd.Series) -> pd.DataFrame:
    """Salidas y llegadas por mes activo, y viajes por anclaje por mes activo.

    Separa estaciones centrales de las de bajo uso. Dividir entre meses activos y entre la
    capacidad distingue saturación real de simple tamaño: una estación nueva con pocos viajes
    totales puede ser muy intensa por anclaje.
    """
    salidas = viajes.groupby("Origen_Id").size()
    llegadas = viajes.groupby("Destino_Id").size()
    tabla = pd.DataFrame({
        "salidas_por_mes": salidas / meses_activa,
        "llegadas_por_mes": llegadas / meses_activa,
    })
    tabla["viajes_por_anclaje_mes"] = (salidas.add(llegadas, fill_value=0)
                                     / meses_activa / capacidad)
    return tabla


def firma_horaria(viajes: pd.DataFrame) -> pd.DataFrame:
    """Proporción de las salidas de cada estación en cada hora del día (columnas h00 a h23).

    Es la huella digital de uso de la estación: picos de 7-9 h y 17-19 h describen uso
    laboral o escolar; una distribución plana o nocturna describe otros perfiles. Al ser
    proporciones no depende del volumen ni de la antigüedad de la estación.
    """
    conteo = pd.crosstab(viajes["Origen_Id"].to_numpy(),
                         viajes["Inicio_del_viaje"].dt.hour.to_numpy())
    conteo = conteo.reindex(columns=range(24), fill_value=0)
    proporciones = conteo.div(conteo.sum(axis=1), axis=0)
    proporciones.columns = [f"h{hora:02d}" for hora in range(24)]
    return proporciones


def firma_semanal(viajes: pd.DataFrame) -> pd.DataFrame:
    """Proporción de las salidas de cada estación en cada día de la semana (columnas d0 a d6).

    El uso recreativo se concentra en sábado (d5) y domingo (d6); el laboral y escolar, de
    lunes a viernes. Complementa la firma horaria, que no distingue entre días.
    """
    conteo = pd.crosstab(viajes["Origen_Id"].to_numpy(),
                         viajes["Inicio_del_viaje"].dt.dayofweek.to_numpy())
    conteo = conteo.reindex(columns=range(7), fill_value=0)
    proporciones = conteo.div(conteo.sum(axis=1), axis=0)
    proporciones.columns = [f"d{dia}" for dia in range(7)]
    return proporciones


def rol_en_la_red(viajes: pd.DataFrame, meses_activa: pd.Series) -> pd.DataFrame:
    """Flujo neto por mes, balance y diversidad de destinos de cada estación.

    flujo_neto_por_mes = llegadas − salidas por mes activo: magnitud del papel de la estación
    como sumidero (+) o fuente (−) de bicicletas. balance = (llegadas − salidas) / total,
    acotado en [-1, 1]: la dirección del papel sin importar el tamaño. num_destinos = estaciones
    distintas a las que salen viajes: distingue estaciones conectoras de las de un solo corredor.
    """
    salidas = viajes.groupby("Origen_Id").size()
    llegadas = viajes.groupby("Destino_Id").size()
    total = salidas.add(llegadas, fill_value=0)
    destinos = viajes.groupby("Origen_Id")["Destino_Id"].nunique()
    return pd.DataFrame({
        "flujo_neto_por_mes": (llegadas.reindex(total.index, fill_value=0)
                             .sub(salidas.reindex(total.index, fill_value=0))) / meses_activa,
        "balance": (llegadas.reindex(total.index, fill_value=0)
                    .sub(salidas.reindex(total.index, fill_value=0))) / total,
        "num_destinos": destinos,
    })


def tipo_de_viaje(viajes: pd.DataFrame) -> pd.DataFrame:
    """Proporción de viajes redondos y duración mediana de los viajes que inician en la estación.

    Los viajes que regresan a la misma estación y los viajes largos sugieren uso recreativo
    (parques, ciclovía dominical); los viajes cortos de ida, uso utilitario.
    """
    redondos = viajes["Origen_Id"] == viajes["Destino_Id"]
    duracion_min = (viajes["Fin_del_viaje"] - viajes["Inicio_del_viaje"]).dt.total_seconds() / 60
    return pd.DataFrame({
        "prop_viajes_redondos": redondos.groupby(viajes["Origen_Id"]).mean(),
        "duracion_mediana_min": duracion_min.groupby(viajes["Origen_Id"]).median(),
    })


def evolucion_historica(viajes: pd.DataFrame, primer_viaje: pd.Series, meses_activa: pd.Series,
                        fin: pd.Timestamp) -> pd.DataFrame:
    """Antigüedad, proporción de actividad reciente y crecimiento respecto a lo esperado.

    Solo existen gracias a la historia completa: distinguen estaciones nuevas, en crecimiento,
    maduras o en declive aunque hoy tengan el mismo volumen. 'crecimiento_vs_esperado' compara
    la proporción de viajes post-COVID con la que le tocaría a la estación por el tiempo que
    lleva activa: positivo = creció, negativo = declinó, cero = estable (incluidas las nuevas).
    """
    inicio_12m = fin - pd.DateOffset(months=12)
    salidas = viajes.groupby("Origen_Id").size()
    salidas_12m = viajes[viajes["Inicio_del_viaje"] >= inicio_12m].groupby("Origen_Id").size()
    salidas_post = viajes[viajes["Inicio_del_viaje"] >= INICIO_POSTCOVID].groupby("Origen_Id").size()

    share_12m = (salidas_12m / salidas).reindex(meses_activa.index, fill_value=0)
    share_post = (salidas_post / salidas).reindex(meses_activa.index, fill_value=0)
    inicio_post = primer_viaje.clip(lower=INICIO_POSTCOVID)
    meses_post = meses_entre(inicio_post, fin).clip(upper=meses_activa)
    share_esperado = meses_post / meses_activa
    return pd.DataFrame({
        "meses_activa": meses_activa,
        "share_viajes_ultimos_12m": share_12m.fillna(0.0),
        "crecimiento_vs_esperado": (share_post - share_esperado).fillna(0.0),
    })


def construir_matriz_p3(viajes: pd.DataFrame, estaciones: pd.DataFrame,
                        fin: pd.Timestamp) -> pd.DataFrame:
    """Arma la matriz estación × característica de P3.

    'estaciones' trae capacidad, latitud y longitud indexadas por número de estación; las
    estaciones que no están ahí (excluidas o sin ficha en la nomenclatura) quedan fuera, y
    las que no tienen ningún viaje en el periodo también (no hay patrón que describir).
    'fin' es el cierre del periodo sobre el que se calculan los meses activos: la matriz de
    una era del pasado se calcula con el fin de esa era, no con el de hoy.
    """
    viajes = viajes[viajes["Origen_Id"].isin(estaciones.index)
                    & viajes["Destino_Id"].isin(estaciones.index)].reset_index(drop=True)
    primer_viaje = primer_viaje_por_estacion(viajes)
    meses_activa = meses_entre(primer_viaje, fin)
    matriz = pd.concat([
        nivel_de_actividad(viajes, meses_activa, estaciones["capacidad"]),
        firma_horaria(viajes),
        firma_semanal(viajes),
        rol_en_la_red(viajes, meses_activa),
        tipo_de_viaje(viajes),
        evolucion_historica(viajes, primer_viaje, meses_activa, fin),
    ], axis=1)
    matriz = matriz.reindex(primer_viaje.index).dropna(how="all")
    # Una estación solo con llegadas deja NaN en las características de salidas: se rellenan con 0
    # (no hay viajes de ese tipo) y la duración con la mediana global (0 minutos sería una señal falsa).
    mediana_duracion = matriz["duracion_mediana_min"].median()
    matriz["duracion_mediana_min"] = matriz["duracion_mediana_min"].fillna(mediana_duracion)
    matriz = matriz.fillna(0)
    matriz.index.name = "estacion_id"
    return matriz.join(estaciones[["latitud", "longitud"]])
