
from locust import HttpUser, task, between


# P1: Pronóstico de demanda por estación
P1_PAYLOAD = {
    "estacion_id": 123,
    "fecha_ultimo_dato": "2026-11-20",
    "viajes_ultimos_7_dias": [18, 22, 20, 25, 19, 24, 21],
}


# P2: Clasificación de riesgo de estaciones
P2_PAYLOAD = {
    "hora_del_dia": 10,
    "dia_semana": 4,
    "mes": 11,
    "fin_de_semana": 0,
    "capacidad": 30,
    "latitud": 20.6736,
    "longitud": -103.344,
    "salidas_hora_anterior": 5,
    "llegadas_hora_anterior": 4,
    "salidas_3_horas_anteriores": 15,
    "llegadas_3_horas_anteriores": 12,
    "semanas_sin_bicicletas": 0,
    "semanas_sin_anclajes": 0,
}


# P3: Clasificación de grupos de estaciones
# Las proporciones horarias y semanales suman 1.
horas = [0.0] * 24
horas[7] = 0.20
horas[8] = 0.30
horas[13] = 0.10
horas[17] = 0.20
horas[18] = 0.20

dias = [0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.10]

P3_PAYLOAD = {
    "salidas_por_mes": 450,
    "llegadas_por_mes": 430,
    "viajes_por_anclaje_mes": 15,
    **{f"h{i:02d}": valor for i, valor in enumerate(horas)},
    **{f"d{i}": valor for i, valor in enumerate(dias)},
    "flujo_neto_por_mes": -20,
    "balance": -0.02,
    "num_destinos": 35,
    "prop_viajes_redondos": 0.25,
    "duracion_mediana_min": 12,
    "meses_activa": 60,
    "share_viajes_ultimos_12m": 0.35,
    "crecimiento_vs_esperado": 0.05,
}


class MiBiciAPIUser(HttpUser):
    wait_time = between(1, 2)

    @task(1)
    def pronostico_p1(self):
        self.client.post(
            "/p1/predict",
            json=P1_PAYLOAD,
            name="P1 - Pronostico de demanda",
        )

    @task(1)
    def riesgo_p2(self):
        self.client.post(
            "/p2/predict",
            json=P2_PAYLOAD,
            name="P2 - Riesgo de estaciones",
        )

    @task(1)
    def grupos_p3(self):
        self.client.post(
            "/p3/predict",
            json=P3_PAYLOAD,
            name="P3 - Grupos de estaciones",
        )