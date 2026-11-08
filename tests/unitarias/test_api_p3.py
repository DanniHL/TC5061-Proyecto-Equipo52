"""Pruebas de la API de P3: esquema de entrada, predicción con el champion empaquetado y peticiones HTTP."""

from api.main import app
from api.schemas.p3 import EntradaP3, SalidaP3
from api.services.p3 import VARIABLES, cargar_modelo, predecir
from fastapi.testclient import TestClient
from pydantic import ValidationError
import pytest


@pytest.fixture
def entrada_valida() -> dict:
    """Perfil de la estación 2 (López Cotilla / Penitenciaría): núcleo central de máxima rotación."""
    return {
        "salidas_por_mes": 1316.584507,
        "llegadas_por_mes": 1397.739437,
        "viajes_por_anclaje_mes": 180.95493,
        "h00": 0.002915,
        "h01": 1.1e-05,
        "h02": 5e-06,
        "h03": 5e-06,
        "h04": 0.0,
        "h05": 0.004402,
        "h06": 0.027916,
        "h07": 0.066273,
        "h08": 0.121864,
        "h09": 0.073413,
        "h10": 0.049948,
        "h11": 0.045,
        "h12": 0.044182,
        "h13": 0.060865,
        "h14": 0.063523,
        "h15": 0.066342,
        "h16": 0.063555,
        "h17": 0.055521,
        "h18": 0.082405,
        "h19": 0.069573,
        "h20": 0.050124,
        "h21": 0.032018,
        "h22": 0.012062,
        "h23": 0.008077,
        "d0": 0.158717,
        "d1": 0.169522,
        "d2": 0.171303,
        "d3": 0.167463,
        "d4": 0.166596,
        "d5": 0.104004,
        "d6": 0.062395,
        "flujo_neto_por_mes": 81.15493,
        "balance": 0.029899,
        "num_destinos": 338.0,
        "prop_viajes_redondos": 0.0309,
        "duracion_mediana_min": 11.066667,
        "meses_activa": 142.0,
        "share_viajes_ultimos_12m": 0.095103,
        "crecimiento_vs_esperado": 0.092012,
    }


@pytest.fixture
def entrada_periferica() -> dict:
    """Perfil de la estación 46: expansión periférica con uso recreativo dominical."""
    return {
        "salidas_por_mes": 1053.697183,
        "llegadas_por_mes": 955.260563,
        "viajes_por_anclaje_mes": 105.734618,
        "h00": 0.003977,
        "h01": 4e-05,
        "h02": 1.3e-05,
        "h03": 0.0,
        "h04": 0.0,
        "h05": 0.000475,
        "h06": 0.006302,
        "h07": 0.013213,
        "h08": 0.032956,
        "h09": 0.055332,
        "h10": 0.059776,
        "h11": 0.071084,
        "h12": 0.08135,
        "h13": 0.084264,
        "h14": 0.061955,
        "h15": 0.061079,
        "h16": 0.066901,
        "h17": 0.081129,
        "h18": 0.106312,
        "h19": 0.073544,
        "h20": 0.058373,
        "h21": 0.043021,
        "h22": 0.024434,
        "h23": 0.01447,
        "d0": 0.135933,
        "d1": 0.141701,
        "d2": 0.140424,
        "d3": 0.137978,
        "d4": 0.132565,
        "d5": 0.091602,
        "d6": 0.219796,
        "flujo_neto_por_mes": -98.43662,
        "balance": -0.048999,
        "num_destinos": 370.0,
        "prop_viajes_redondos": 0.029808,
        "duracion_mediana_min": 10.433333,
        "meses_activa": 142.0,
        "share_viajes_ultimos_12m": 0.105397,
        "crecimiento_vs_esperado": 0.072892,
    }


# ------------------------------------------------------------------ esquema de entrada


def test_acepta_una_entrada_valida(entrada_valida):
    assert EntradaP3(**entrada_valida).meses_activa == 142.0


@pytest.mark.parametrize(
    "campo, valor",
    [
        ("salidas_por_mes", 0),
        ("balance", -1),
        ("balance", 1),
        ("flujo_neto_por_mes", -250.5),
        ("meses_activa", 1),
    ],
)
def test_acepta_los_limites_de_los_rangos(entrada_valida, campo, valor):
    entrada_valida[campo] = valor
    assert getattr(EntradaP3(**entrada_valida), campo) == valor


def test_acepta_una_firma_con_toda_la_proporcion_en_una_posicion(entrada_valida):
    """Una estación usada solo a una hora y un día tendría 1 en esa posición y 0 en las demás."""
    for hora in range(24):
        entrada_valida[f"h{hora:02d}"] = 0.0
    for dia in range(7):
        entrada_valida[f"d{dia}"] = 0.0
    entrada_valida["h12"] = 1.0
    entrada_valida["d6"] = 1.0
    entrada = EntradaP3(**entrada_valida)
    assert entrada.h12 == 1.0 and entrada.d6 == 1.0


@pytest.mark.parametrize(
    "campo, valor",
    [
        ("salidas_por_mes", -1),
        ("viajes_por_anclaje_mes", -0.5),
        ("h08", -0.1),
        ("h08", 1.01),
        ("d3", 1.5),
        ("balance", -1.5),
        ("balance", 1.5),
        ("num_destinos", -1),
        ("prop_viajes_redondos", 1.2),
        ("duracion_mediana_min", -3),
        ("meses_activa", 0),
        ("share_viajes_ultimos_12m", 1.1),
        ("crecimiento_vs_esperado", -2),
    ],
)
def test_rechaza_valores_fuera_de_rango(entrada_valida, campo, valor):
    entrada_valida[campo] = valor
    with pytest.raises(ValidationError):
        EntradaP3(**entrada_valida)


def test_rechaza_un_campo_desconocido(entrada_valida):
    with pytest.raises(ValidationError, match="Extra inputs"):
        EntradaP3(**entrada_valida, estacion=2)


def test_rechaza_si_falta_un_campo_obligatorio(entrada_valida):
    del entrada_valida["h08"]
    with pytest.raises(ValidationError, match="h08"):
        EntradaP3(**entrada_valida)


def test_rechaza_una_firma_horaria_que_no_suma_uno(entrada_valida):
    entrada_valida["h08"] = 0.9  # la firma pasa a sumar ≈ 1.78
    with pytest.raises(ValidationError, match="firma horaria"):
        EntradaP3(**entrada_valida)


def test_rechaza_una_firma_semanal_que_no_suma_uno(entrada_valida):
    for dia in range(7):
        entrada_valida[f"d{dia}"] = 0.5  # la firma pasa a sumar 3.5
    with pytest.raises(ValidationError, match="firma semanal"):
        EntradaP3(**entrada_valida)


def test_acepta_firmas_en_cero_para_estacion_sin_salidas(entrada_valida):
    """Una estación que solo recibe llegadas tiene sus firmas en 0 (features.py las rellena así)."""
    for hora in range(24):
        entrada_valida[f"h{hora:02d}"] = 0.0
    for dia in range(7):
        entrada_valida[f"d{dia}"] = 0.0
    entrada_valida["salidas_por_mes"] = 0.0
    assert EntradaP3(**entrada_valida).h08 == 0.0


# ------------------------------------------------------------------ modelo y predicción


def test_el_esquema_coincide_con_la_firma_del_modelo():
    """cargar_modelo falla si las variables de la API y las del modelo empaquetado no son las mismas."""
    assert cargar_modelo() is not None
    assert len(VARIABLES) == 42


def test_predice_nucleo_receptor_para_estacion_central(entrada_valida):
    salida = predecir(EntradaP3(**entrada_valida))
    assert isinstance(salida, SalidaP3)
    assert (salida.grupo, salida.etiqueta) == (0, "nucleo_receptor")


def test_predice_periferico_recreativo_para_estacion_de_expansion(entrada_periferica):
    salida = predecir(EntradaP3(**entrada_periferica))
    assert (salida.grupo, salida.etiqueta) == (3, "periferico_recreativo")


def test_el_grupo_asignado_es_el_centroide_mas_cercano(entrada_valida):
    salida = predecir(EntradaP3(**entrada_valida))
    assert set(salida.distancias) == {
        "nucleo_receptor",
        "empleo_vespertino",
        "residencial_matutino",
        "periferico_recreativo",
    }
    assert min(salida.distancias, key=salida.distancias.get) == salida.etiqueta


# ------------------------------------------------------------------ rutas


def test_la_aplicacion_expone_las_rutas_de_p3():
    rutas = app.openapi()["paths"]
    assert "post" in rutas["/p3/predict"]
    assert "get" in rutas["/health"]


def test_el_endpoint_usa_los_esquemas_de_p3():
    """La documentación de OpenAPI confirma que FastAPI valida con EntradaP3 y responde con SalidaP3."""
    operacion = app.openapi()["paths"]["/p3/predict"]["post"]
    entrada = operacion["requestBody"]["content"]["application/json"]["schema"]["$ref"]
    salida = operacion["responses"]["200"]["content"]["application/json"]["schema"]["$ref"]
    assert entrada.endswith("/EntradaP3")
    assert salida.endswith("/SalidaP3")
    assert "422" in operacion["responses"]


# ------------------------------------------------------------------ peticiones HTTP


@pytest.fixture(scope="module")
def cliente():
    """Cliente HTTP de la API; con el bloque with corre el arranque real, que carga el modelo."""
    with TestClient(app) as cliente_http:
        yield cliente_http


def test_health_responde_ok(cliente):
    respuesta = cliente.get("/health")
    assert respuesta.status_code == 200
    assert respuesta.json() == {"estado": "ok"}


def test_predict_responde_200_con_la_prediccion(cliente, entrada_valida):
    respuesta = cliente.post("/p3/predict", json=entrada_valida)
    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert set(cuerpo) == {"grupo", "etiqueta", "distancias"}
    assert (cuerpo["grupo"], cuerpo["etiqueta"]) == (0, "nucleo_receptor")


@pytest.mark.parametrize(
    "campo_en_error, valor",
    [
        ("balance", 99),
        ("meses_activa", 0),
        ("h08", 1.5),
        ("salidas_por_mes", -10),
        ("h18", "ocho"),
    ],
)
def test_predict_responde_422_con_valores_fuera_de_rango(
    cliente, entrada_valida, campo_en_error, valor
):
    respuesta = cliente.post("/p3/predict", json={**entrada_valida, campo_en_error: valor})
    assert respuesta.status_code == 422
    assert campo_en_error in respuesta.json()["detail"][0]["loc"]


def test_predict_responde_422_con_un_campo_desconocido(cliente, entrada_valida):
    respuesta = cliente.post("/p3/predict", json={**entrada_valida, "estacion": 2})
    assert respuesta.status_code == 422
    assert "estacion" in respuesta.json()["detail"][0]["loc"]


def test_predict_responde_422_si_la_firma_no_suma_uno(cliente, entrada_valida):
    respuesta = cliente.post("/p3/predict", json={**entrada_valida, "h08": 0.9})
    assert respuesta.status_code == 422
    assert "firma horaria" in respuesta.json()["detail"][0]["msg"]


def test_predict_responde_422_si_falta_un_campo_o_el_cuerpo(cliente, entrada_valida):
    del entrada_valida["duracion_mediana_min"]
    assert cliente.post("/p3/predict", json=entrada_valida).status_code == 422
    assert cliente.post("/p3/predict").status_code == 422


def test_predict_no_acepta_get(cliente):
    assert cliente.get("/p3/predict").status_code == 405
