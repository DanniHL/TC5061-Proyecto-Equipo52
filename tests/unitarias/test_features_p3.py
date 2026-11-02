"""Pruebas del feature engineering y del agrupamiento de P3 (grupos de estaciones por uso).

Las pruebas cuidan la regla que hace comparables a las estaciones nuevas con las de 2014:
los conteos se dividen entre los meses activos de cada estación y las firmas horaria y
semanal son proporciones, no volúmenes. Usan las fixtures viajes_p3, estaciones_p3 y
fin_p3 de tests/conftest.py (12 viajes entre las estaciones 1 a 6).
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from equipo52_module.config import INICIO_POSTCOVID, SEED
from equipo52_module.features import (
    construir_matriz_p3,
    evolucion_historica,
    firma_horaria,
    firma_semanal,
    meses_entre,
    nivel_de_actividad,
    primer_viaje_por_estacion,
    quitar_viajes_redondos_cortos,
    rol_en_la_red,
    tipo_de_viaje,
)
from equipo52_module.modeling.train_p3 import (
    GEO,
    barrer_k,
    cargar_estaciones,
    describir_grupos,
    estabilidad_bootstrap,
    estabilidad_entre_eras,
    metricas_internas,
    mismo_valor,
    parametros_comunes,
)


def meses_activos(viajes, fin):
    """Meses activos de cada estación, como los calcula construir_matriz_p3."""
    return meses_entre(primer_viaje_por_estacion(viajes), fin)


# ------------------------------------------------------------------ limpieza de viajes


def test_quitar_redondos_cortos_quita_solo_el_de_2_min(viajes_p3):
    resultado = quitar_viajes_redondos_cortos(viajes_p3)
    assert 6 not in resultado["Viaje_Id"].values
    assert len(resultado) == len(viajes_p3) - 1


def test_quitar_redondos_cortos_borde_de_180_segundos(viajes_p3):
    resultado = quitar_viajes_redondos_cortos(viajes_p3)
    # el viaje 7 dura exactamente 3 min y se conserva (el corte es <, no <=)
    assert 7 in resultado["Viaje_Id"].values
    # el paseo redondo de 40 min de la estación 3 también
    assert 5 in resultado["Viaje_Id"].values


def test_quitar_redondos_cortos_no_afecta_viajes_de_ida():
    """Un viaje de 1 minuto a otra estación no es redondo: se conserva."""
    inicio = pd.Timestamp("2026-01-05 08:00")
    viajes = pd.DataFrame({
        "Inicio_del_viaje": [inicio],
        "Fin_del_viaje": [inicio + pd.Timedelta(minutes=1)],
        "Origen_Id": [1],
        "Destino_Id": [2],
    })
    assert len(quitar_viajes_redondos_cortos(viajes)) == 1


def test_quitar_redondos_cortos_con_umbral_mayor(viajes_p3):
    # con 50 min se van también el paseo de 40 min: ningún redondo sobrevive
    resultado = quitar_viajes_redondos_cortos(viajes_p3, segundos_maximos=50 * 60)
    redondos = resultado[resultado["Origen_Id"] == resultado["Destino_Id"]]
    assert redondos.empty


# ------------------------------------------------------------------ antigüedad de estaciones


def test_meses_entre_cuenta_meses_calendario():
    inicio = pd.Series(pd.to_datetime(["2026-02-10", "2020-01-10", "2026-02-28"]))
    resultado = meses_entre(inicio, pd.Timestamp("2026-02-28"))
    # el mismo mes cuenta 1; de enero de 2020 a febrero de 2026 son 74 meses calendario
    assert resultado.tolist() == [1, 74, 1]


def test_primer_viaje_mira_salidas_y_llegadas(viajes_p3):
    primer_viaje = primer_viaje_por_estacion(viajes_p3)
    # la estación 2 nunca origina viajes: su primer viaje es una llegada (la de 2020)
    assert primer_viaje[2] == pd.Timestamp("2020-01-10 08:25")
    # la 3 recibe el viaje 2 antes de su primera salida del 10 de enero
    assert primer_viaje[3] == pd.Timestamp("2026-01-05 08:45")
    assert primer_viaje[1] == pd.Timestamp("2026-01-05 08:00")


# ------------------------------------------------------------------ nivel de actividad


def test_nivel_de_actividad_normaliza_por_meses(viajes_p3, estaciones_p3, fin_p3):
    actividad = nivel_de_actividad(viajes_p3, meses_activos(viajes_p3, fin_p3),
                                 estaciones_p3["capacidad"])
    # estación 3: 5 salidas y 6 llegadas en 2 meses activos
    assert actividad.loc[3, "salidas_por_mes"] == 2.5
    assert actividad.loc[3, "llegadas_por_mes"] == 3.0
    # 11 movimientos / 2 meses / 8 anclajes: distingue intensidad real de tamaño
    assert actividad.loc[3, "viajes_por_anclaje_mes"] == pytest.approx(11 / 2 / 8)


def test_nivel_de_actividad_sin_llegadas_deja_vacio(viajes_p3, estaciones_p3, fin_p3):
    actividad = nivel_de_actividad(viajes_p3, meses_activos(viajes_p3, fin_p3),
                                 estaciones_p3["capacidad"])
    # la estación 1 nunca recibe viajes: queda vacío (la matriz lo rellena con 0 después)
    assert np.isnan(actividad.loc[1, "llegadas_por_mes"])
    # pero viajes_por_anclaje_mes sí cuenta sus 4 salidas
    assert actividad.loc[1, "viajes_por_anclaje_mes"] == pytest.approx(4 / 2 / 10)


# ------------------------------------------------------------------ firmas temporales


def test_firma_horaria_proporciones_suman_uno(viajes_p3):
    firma = firma_horaria(viajes_p3)
    assert list(firma.columns) == [f"h{h:02d}" for h in range(24)]
    np.testing.assert_allclose(firma.sum(axis=1).to_numpy(), 1.0)


def test_firma_horaria_picos_laborales(viajes_p3):
    firma = firma_horaria(viajes_p3)
    # la estación 1 solo se usa en horario laboral: salidas a las 7, 8, 8 y 17 h
    assert firma.loc[1, "h07"] == 0.25
    assert firma.loc[1, "h08"] == 0.5
    assert firma.loc[1, "h17"] == 0.25


def test_firma_horaria_estacion_con_un_solo_viaje(viajes_p3):
    firma = firma_horaria(viajes_p3)
    # la estación 4 solo origina el viaje 10 a las 9 h: toda su firma está ahí
    assert firma.loc[4, "h09"] == 1.0
    assert (firma.loc[4].drop("h09") == 0).all()


def test_firma_semanal_fin_de_semana_recreativo(viajes_p3):
    firma = firma_semanal(viajes_p3)
    assert list(firma.columns) == [f"d{d}" for d in range(7)]
    np.testing.assert_allclose(firma.sum(axis=1).to_numpy(), 1.0)
    # la estación 3: 3 de 5 salidas en sábado y 1 en domingo (los paseos redondos)
    assert firma.loc[3, "d5"] == 0.6
    assert firma.loc[3, ["d5", "d6"]].sum() == 0.8


def test_firma_semanal_laboral(viajes_p3):
    firma = firma_semanal(viajes_p3)
    # la estación 1 solo se usa entre semana
    assert firma.loc[1, [f"d{d}" for d in range(5)]].sum() == 1.0


# ------------------------------------------------------------------ rol en la red


def test_rol_fuente_y_sumidero(viajes_p3, fin_p3):
    rol = rol_en_la_red(viajes_p3, meses_activos(viajes_p3, fin_p3))
    # estación 1: 4 salidas y 0 llegadas → fuente pura de bicicletas
    assert rol.loc[1, "balance"] == -1.0
    assert rol.loc[1, "flujo_neto_por_mes"] == -2.0       # −4 viajes / 2 meses
    # estación 2: 4 llegadas y 0 salidas → sumidero puro
    assert rol.loc[2, "balance"] == 1.0
    # el balance de toda la red queda acotado en [-1, 1]
    assert rol["balance"].between(-1, 1).all()


def test_rol_sin_salidas_deja_num_destinos_vacio(viajes_p3, fin_p3):
    rol = rol_en_la_red(viajes_p3, meses_activos(viajes_p3, fin_p3))
    # la 2 no tiene salidas: no hay destinos que contar (la matriz lo rellena con 0)
    assert np.isnan(rol.loc[2, "num_destinos"])
    # la 1 conecta con las estaciones 2 y 3
    assert rol.loc[1, "num_destinos"] == 2


# ------------------------------------------------------------------ tipo de viaje


def test_tipo_de_viaje_recreo(viajes_p3):
    tipo = tipo_de_viaje(viajes_p3)
    # 3 de los 5 viajes que salen de la 3 regresan a la 3
    assert tipo.loc[3, "prop_viajes_redondos"] == 0.6
    # duraciones {40, 2, 3, 18, 10} → mediana 10
    assert tipo.loc[3, "duracion_mediana_min"] == 10.0


def test_tipo_de_viaje_sin_redondos(viajes_p3):
    tipo = tipo_de_viaje(viajes_p3)
    assert tipo.loc[1, "prop_viajes_redondos"] == 0.0
    # duraciones {20, 15, 25, 30} → mediana de número par de viajes
    assert tipo.loc[1, "duracion_mediana_min"] == 22.5


# ------------------------------------------------------------------ evolución histórica


def test_evolucion_estacion_consolidada(viajes_p3, fin_p3):
    primer_viaje = primer_viaje_por_estacion(viajes_p3)
    evolucion = evolucion_historica(viajes_p3, primer_viaje,
                                  meses_entre(primer_viaje, fin_p3), fin_p3)
    # la estación 6 nació en enero de 2020: 74 meses activa y la mitad de sus salidas
    # (1 de 2) cayó en los últimos 12 meses
    assert evolucion.loc[6, "meses_activa"] == 74
    assert evolucion.loc[6, "share_viajes_ultimos_12m"] == 0.5
    # el periodo post-COVID le toca en ≈ 46 % por antigüedad; tuvo 50 %: creció un poco
    esperado = meses_entre(pd.Series([pd.Timestamp(INICIO_POSTCOVID)]), fin_p3).iloc[0] / 74
    assert evolucion.loc[6, "crecimiento_vs_esperado"] == pytest.approx(0.5 - esperado)


def test_evolucion_estacion_nueva_no_declina(viajes_p3, fin_p3):
    primer_viaje = primer_viaje_por_estacion(viajes_p3)
    evolucion = evolucion_historica(viajes_p3, primer_viaje,
                                  meses_entre(primer_viaje, fin_p3), fin_p3)
    # la estación 4 lleva un mes y toda su actividad es reciente: ni creció ni declinó
    assert evolucion.loc[4, "meses_activa"] == 1
    assert evolucion.loc[4, "share_viajes_ultimos_12m"] == 1.0
    assert evolucion.loc[4, "crecimiento_vs_esperado"] == 0.0


# ------------------------------------------------------------------ matriz completa


def test_matriz_una_fila_por_estacion_con_viajes(viajes_p3, estaciones_p3, fin_p3):
    matriz = construir_matriz_p3(viajes_p3, estaciones_p3, fin_p3)
    # la 4 no tiene ficha y la 5 no tiene viajes: quedan 1, 2, 3 y 6
    assert sorted(matriz.index) == [1, 2, 3, 6]
    # 42 características de uso + latitud y longitud
    assert matriz.shape[1] == 44
    assert matriz.index.name == "estacion_id"
    assert not matriz.isna().any().any()


def test_matriz_estacion_solo_con_llegadas(viajes_p3, estaciones_p3, fin_p3):
    matriz = construir_matriz_p3(viajes_p3, estaciones_p3, fin_p3)
    # la 2 nunca origina viajes: firmas y conteos de salidas en 0, duración = mediana de la red
    assert matriz.loc[2, [f"h{h:02d}" for h in range(24)]].sum() == 0.0
    assert matriz.loc[2, "prop_viajes_redondos"] == 0.0
    assert matriz.loc[2, "salidas_por_mes"] == 0.0
    assert matriz.loc[2, "duracion_mediana_min"] == 22.5   # mediana de {22.5, 10, 25}
    # pero sí describe su papel de sumidero de la red
    assert matriz.loc[2, "balance"] == 1.0
    assert matriz.loc[2, "llegadas_por_mes"] == pytest.approx(4 / 74)


def test_matriz_une_capacidad_y_coordenadas(viajes_p3, estaciones_p3, fin_p3):
    matriz = construir_matriz_p3(viajes_p3, estaciones_p3, fin_p3)
    assert matriz.loc[3, "latitud"] == estaciones_p3.loc[3, "latitud"]
    assert matriz.loc[3, "longitud"] == estaciones_p3.loc[3, "longitud"]


# ------------------------------------------------------------------ train_p3.py (agrupamiento)


@pytest.fixture
def nubes_separadas() -> pd.DataFrame:
    """Tres nubes de estaciones bien separadas: las pruebas no dependen del azar de K-Means."""
    generador = np.random.default_rng(0)
    nubes = [generador.normal(loc=centro, scale=0.05, size=(20, 4)) for centro in (-10.0, 0.0, 10.0)]
    return pd.DataFrame(np.vstack(nubes), columns=["a", "b", "c", "d"])


def test_metricas_internas_con_prefijo(nubes_separadas):
    etiquetas = KMeans(n_clusters=3, n_init=10, random_state=SEED).fit_predict(nubes_separadas)
    metricas = metricas_internas(nubes_separadas, etiquetas, prefijo="val")
    assert set(metricas) == {"val_silhouette", "val_davies_bouldin", "val_calinski_harabasz"}
    # nubes compactas y separadas: silueta alta y solape entre grupos bajo
    assert metricas["val_silhouette"] > 0.9
    assert metricas["val_davies_bouldin"] < 0.2


def test_barrer_k_recorre_todo_el_rango(nubes_separadas):
    barrido = barrer_k(nubes_separadas, "kmeans")
    assert barrido["k"].tolist() == list(range(2, 13))
    assert barrido["silueta"].between(-1, 1).all()
    # hay tres nubes reales: k = 3 agrupa mejor que cortar una nube en dos (k = 2)
    mejor = barrido.loc[barrido["silueta"].idxmax(), "k"]
    assert mejor == 3


def test_estabilidad_bootstrap_con_grupos_claros(nubes_separadas):
    base = KMeans(n_clusters=3, n_init=20, random_state=SEED).fit(nubes_separadas).labels_
    # con grupos claramente separados, quitar el 20 % de las estaciones no cambia la asignación
    assert estabilidad_bootstrap(nubes_separadas, base, k=3) == pytest.approx(1.0)
    # y el remuestreo usa semilla fija: dos llamadas devuelven exactamente el mismo ARI
    assert estabilidad_bootstrap(nubes_separadas, base, k=3) == \
        estabilidad_bootstrap(nubes_separadas, base, k=3)


def test_describir_grupos(viajes_p3, estaciones_p3, fin_p3):
    matriz = construir_matriz_p3(viajes_p3, estaciones_p3, fin_p3)
    etiquetas = (matriz.index == 3).astype(int)   # grupo 1: la estación 3; grupo 0: el resto
    descripcion = describir_grupos(matriz, etiquetas, estaciones_p3)
    assert list(descripcion["estaciones"]) == [3, 1]
    # la estación 3 (la única de su grupo) tiene su pico a las 11 h, por los paseos de sábado,
    # y dentro de la matriz sus 3 salidas válidas (las de la 4 no cuentan) son todas redondas
    assert descripcion.loc[1, "hora_pico"] == 11
    assert descripcion.loc[1, "prop_viajes_redondos"] == 1.0


def test_estabilidad_entre_eras_corre_sobre_las_das_eras(viajes_p3, estaciones_p3, fin_p3):
    matriz = construir_matriz_p3(viajes_p3, estaciones_p3, fin_p3)
    columnas = [c for c in matriz.columns if c not in GEO]
    escalador = StandardScaler().fit(matriz[columnas])
    X = pd.DataFrame(escalador.transform(matriz[columnas]), columns=columnas, index=matriz.index)
    modelo = KMeans(n_clusters=2, n_init=20, random_state=SEED).fit(X)
    # era A solo tiene el viaje de 2020 de la estación 6; aun así el cálculo devuelve un ARI válido
    ari = estabilidad_entre_eras(viajes_p3, estaciones_p3, escalador, modelo, columnas)
    assert isinstance(ari, float)
    assert -1.0 <= ari <= 1.0


def test_cargar_estaciones_filtra_y_extrae_municipio(tmp_path):
    csv = tmp_path / "nomenclatura.csv"
    csv.write_text(
        "id,name,latitude,longitude,dpcapacity\n"
        '2,"(GDL-001) C. Epigmenio",20.66,-103.34,15\n'
        '22,"ex-(GDL-023) Cerrada",20.67,-103.35,15\n'
        '30,"(ZPN-010) Sin anclajes",20.70,-103.38,0\n'
        '31,"Sin prefijo",20.71,-103.39,12\n'
    )
    estaciones = cargar_estaciones(csv, excluidas=[22])
    # queda fuera la excluida (22) y la de capacidad 0 (30)
    assert sorted(estaciones.index) == [2, 31]
    assert estaciones.index.name == "estacion_id"
    assert estaciones.loc[2, "municipio"] == "GDL"
    assert estaciones.loc[31, "municipio"] == "OTRO"


def test_mismo_valor_compara_como_mlflow():
    assert mismo_valor("0.05", 0.05)
    assert mismo_valor("42", SEED)
    assert not mismo_valor("3", "4")
    # cuando no son números, compara texto sin distinguir mayúsculas
    assert mismo_valor("KMeans", "kmeans")
    assert not mismo_valor("hgb", "logistica")


def test_parametros_comunes_registra_la_corrida(viajes_p3, estaciones_p3, fin_p3):
    matriz = construir_matriz_p3(viajes_p3, estaciones_p3, fin_p3)
    columnas = [c for c in matriz.columns if c not in GEO]
    mascara = matriz["meses_activa"] > 6
    parametros = parametros_comunes("kmeans", matriz, mascara, columnas,
                                  solo_consolidadas=False, sin_redondos_cortos=False)
    assert parametros["semilla"] == SEED
    assert parametros["n_estaciones"] == 4
    assert parametros["n_estaciones_consolidadas"] == 2     # las estaciones 2 y 6, de 2020
    assert parametros["n_estaciones_nuevas_2026"] == 2      # las 1 y 3
    assert parametros["n_caracteristicas"] == 42
