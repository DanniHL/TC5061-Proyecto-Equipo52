"""Pruebas de la API de P2: validación de entradas (esquema) y predicción con el champion empaquetado."""

import pytest
from pydantic import ValidationError

from api.main import app
from api.schemas.p2 import EntradaP2, SalidaP2
from api.services.p2 import VARIABLES, cargar_modelo, predecir


@pytest.fixture
def entrada_valida() -> dict:
    """Estación que se viene vaciando en el pico de la mañana de un miércoles."""
    return {"hora_del_dia": 7, "dia_semana": 2, "mes": 10, "fin_de_semana": 0, "capacidad": 15,
            "latitud": 20.666378, "longitud": -103.34882,
            "salidas_hora_anterior": 10, "llegadas_hora_anterior": 0,
            "salidas_3_horas_anteriores": 25, "llegadas_3_horas_anteriores": 1,
            "semanas_sin_bicicletas": 4, "semanas_sin_anclajes": 0}


# ------------------------------------------------------------------ esquema de entrada

def test_acepta_una_entrada_valida(entrada_valida):
    assert EntradaP2(**entrada_valida).hora_del_dia == 7


@pytest.mark.parametrize("campo", ["salidas_hora_anterior", "llegadas_hora_anterior",
                                   "salidas_3_horas_anteriores", "llegadas_3_horas_anteriores",
                                   "semanas_sin_bicicletas", "semanas_sin_anclajes"])
def test_el_historial_puede_faltar(entrada_valida, campo):
    entrada_valida[campo] = None
    assert getattr(EntradaP2(**entrada_valida), campo) is None


@pytest.mark.parametrize("campo, valor", [("hora_del_dia", 0), ("hora_del_dia", 23), ("mes", 1), ("mes", 12),
                                          ("semanas_sin_bicicletas", 4)])
def test_acepta_los_limites_de_los_rangos(entrada_valida, campo, valor):
    entrada_valida[campo] = valor
    assert getattr(EntradaP2(**entrada_valida), campo) == valor


@pytest.mark.parametrize("campo, valor", [("hora_del_dia", -1), ("hora_del_dia", 24), ("hora_del_dia", 99),
                                          ("dia_semana", 7), ("mes", 0), ("mes", 13), ("fin_de_semana", 2),
                                          ("capacidad", 0), ("capacidad", 101),
                                          ("latitud", 19.4), ("longitud", -99.1),
                                          ("salidas_hora_anterior", -1), ("semanas_sin_bicicletas", 5)])
def test_rechaza_valores_fuera_de_rango(entrada_valida, campo, valor):
    entrada_valida[campo] = valor
    with pytest.raises(ValidationError):
        EntradaP2(**entrada_valida)


def test_rechaza_un_campo_desconocido(entrada_valida):
    with pytest.raises(ValidationError, match="Extra inputs"):
        EntradaP2(**entrada_valida, estacion=5)


def test_rechaza_si_falta_un_campo_obligatorio(entrada_valida):
    del entrada_valida["capacidad"]
    with pytest.raises(ValidationError, match="capacidad"):
        EntradaP2(**entrada_valida)


@pytest.mark.parametrize("dia, fin_de_semana", [(6, 0), (5, 0), (2, 1)])
def test_rechaza_un_calendario_inconsistente(entrada_valida, dia, fin_de_semana):
    entrada_valida.update(dia_semana=dia, fin_de_semana=fin_de_semana)
    with pytest.raises(ValidationError, match="fin_de_semana"):
        EntradaP2(**entrada_valida)


def test_rechaza_mas_semanas_con_falla_que_las_revisadas(entrada_valida):
    entrada_valida.update(semanas_sin_bicicletas=3, semanas_sin_anclajes=2)
    with pytest.raises(ValidationError, match="no pueden sumar"):
        EntradaP2(**entrada_valida)


# ------------------------------------------------------------------ modelo y predicción

def test_el_esquema_coincide_con_la_firma_del_modelo():
    """cargar_modelo falla si las variables de la API y las del modelo empaquetado no son las mismas."""
    assert cargar_modelo() is not None
    assert len(VARIABLES) == 13


def test_predice_sin_bicicletas_cuando_la_estacion_se_vacia(entrada_valida):
    salida = predecir(EntradaP2(**entrada_valida))
    assert isinstance(salida, SalidaP2)
    assert (salida.clase, salida.etiqueta) == (1, "sin_bicicletas")


def test_predice_sin_anclajes_cuando_la_estacion_se_llena(entrada_valida):
    entrada_valida.update(hora_del_dia=8, salidas_hora_anterior=0, llegadas_hora_anterior=12,
                          salidas_3_horas_anteriores=1, llegadas_3_horas_anteriores=30,
                          semanas_sin_bicicletas=0, semanas_sin_anclajes=4)
    salida = predecir(EntradaP2(**entrada_valida))
    assert (salida.clase, salida.etiqueta) == (2, "sin_anclajes")


def test_las_probabilidades_suman_uno(entrada_valida):
    probabilidades = predecir(EntradaP2(**entrada_valida)).probabilidades
    assert set(probabilidades) == {"normal", "sin_bicicletas", "sin_anclajes"}
    assert sum(probabilidades.values()) == pytest.approx(1, abs=0.001)


def test_predice_aunque_falte_el_historial(entrada_valida):
    for campo in ["salidas_hora_anterior", "llegadas_hora_anterior", "salidas_3_horas_anteriores",
                  "llegadas_3_horas_anteriores", "semanas_sin_bicicletas", "semanas_sin_anclajes"]:
        entrada_valida[campo] = None
    assert predecir(EntradaP2(**entrada_valida)).clase in {0, 1, 2}


# ------------------------------------------------------------------ rutas

def test_la_aplicacion_expone_las_rutas_de_p2():
    rutas = app.openapi()["paths"]
    assert "post" in rutas["/p2/predict"]
    assert "get" in rutas["/health"]


def test_el_endpoint_usa_los_esquemas_de_p2():
    """La documentación de OpenAPI confirma que FastAPI valida con EntradaP2 y responde con SalidaP2."""
    operacion = app.openapi()["paths"]["/p2/predict"]["post"]
    entrada = operacion["requestBody"]["content"]["application/json"]["schema"]["$ref"]
    salida = operacion["responses"]["200"]["content"]["application/json"]["schema"]["$ref"]
    assert entrada.endswith("/EntradaP2")
    assert salida.endswith("/SalidaP2")
    assert "422" in operacion["responses"]