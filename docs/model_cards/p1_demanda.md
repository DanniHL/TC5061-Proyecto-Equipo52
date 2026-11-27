# Model Card – P1: Pronóstico de demanda MiBici
Responsable: Daniela Hernández López - A01840424
· Equipo 52 · TC5061.

Última actualización: 27 de noviembre de 2026.

## 1. Identificación del modelo

| Campo | Descripción |
|---|---|
| Nombre | P1 – Pronóstico de demanda por estación |
| Proyecto | MiBici – Equipo 52 |
| Versión registrada | 1 |
| Nombre en MLflow Model Registry | `mibici-equipo52-p1-demanda` |
| Tipo de modelo | Regresión supervisada |
| Algoritmo | Regresión lineal (Linear Regression) |
| Artefacto | `models/p1/` |
| Formato del modelo | Pipeline de scikit-learn |
| Semilla aleatoria | 42 |

## 2. Descripción del modelo

El modelo P1 tiene como objetivo pronosticar la demanda diaria de viajes por estación del sistema público de bicicletas MiBici, ubicado en el Área Metropolitana de Guadalajara.

A partir de información histórica de viajes y variables temporales, genera pronósticos para los siguientes siete días. Su propósito es proporcionar información que facilite la identificación de patrones de demanda y apoye la planeación operativa del sistema.

## 3. Datos utilizados

La información utilizada proviene de los conjuntos de datos abiertos del sistema MiBici, integrados a partir de archivos históricos mensuales.

Los datos se procesan mediante pandas y se preparan para obtener agregaciones diarias de viajes por estación.

**Criterios de preparación:**

- Se excluyen los viajes con duración inferior a 60 segundos durante la etapa de preparación y modelado.
- Se conservan los registros de viajes cuyo origen y destino corresponden a la misma estación.
- Se construyen series temporales de demanda diaria por estación.
- Se generan variables históricas y temporales para representar los patrones de demanda.
- Se mantienen los valores faltantes como NaN cuando corresponde, sin sustituirlos automáticamente por cero.

## 4. Variables del modelo

El modelo utiliza ocho variables de entrada que representan información histórica, temporal y de comportamiento de la demanda.

Entre las variables utilizadas se encuentran:

- `Viajes_dia_anterior`: demanda registrada el día anterior.
- `Viajes_hace_7_dias`: demanda registrada siete días antes.
- `Promedio_7_dias`: promedio histórico de los últimos siete días.
- `Dia_semana_sin`: componente cíclico del día de la semana.
- `Mes_sin`: componente cíclico del mes.

También se consideran variables asociadas a la estación y al calendario.

**Variable objetivo:** cantidad de viajes diarios registrados por estación.

## 5. Metodología de entrenamiento

Se utiliza un modelo de regresión lineal entrenado con variables históricas y temporales para estimar la demanda diaria.

La división de los datos se realiza de manera cronológica, respetando el orden temporal de los registros para evitar utilizar información futura durante el entrenamiento.

El periodo de prueba documentado corresponde al:

- Inicio: 1 de julio de 2026.
- Fin: 30 de septiembre de 2026.

Para generar los pronósticos de siete días se utiliza un esquema recursivo, en el que las predicciones anteriores alimentan las variables históricas de los días siguientes.

## 6. Evaluación del modelo

El desempeño se evalúa mediante las métricas MAE (Mean Absolute Error) y RMSE (Root Mean Squared Error).

Como referencia se utiliza un modelo baseline de persistencia, que estima la demanda del día siguiente utilizando la demanda del día anterior.

**Resultados del baseline:**

| Métrica | Resultado |
|---|---:|
| MAE | 9.229 |
| RMSE | 16.012 |

**Resultados del modelo P1:**

| Métrica | Resultado |
|---|---:|
| MAE | Pendiente de recuperar de MLflow |
| RMSE | Pendiente de recuperar de MLflow |

Un menor valor de MAE y RMSE representa un mejor desempeño predictivo.

## 7. Limitaciones y consideraciones

El modelo presenta las siguientes limitaciones:

- Depende de los patrones históricos de demanda registrados en el sistema.
- Puede presentar desviaciones ante cambios importantes en los hábitos de movilidad.
- Los eventos extraordinarios, condiciones climáticas y modificaciones operativas pueden afectar la precisión de los pronósticos.
- El desempeño puede variar entre estaciones debido a sus diferentes patrones de utilización.
- Los pronósticos recursivos pueden acumular errores conforme aumenta el horizonte de predicción.

Los resultados deben interpretarse como estimaciones y no como valores garantizados de demanda.

## 8. Uso previsto

El modelo está diseñado para apoyar el análisis y la planeación de la demanda diaria de viajes por estación de MiBici.

Su integración se realiza mediante una API desarrollada con FastAPI, que permite enviar las variables de entrada y obtener siete pronósticos consecutivos.

El modelo se carga desde el artefacto versionado en `models/p1/`, sin necesidad de consultar directamente los archivos históricos durante cada solicitud.

## 9. Registro y versionamiento

El seguimiento del experimento y la gestión del modelo se realizan mediante MLflow.

| Elemento | Ubicación |
|---|---|
| Experiment tracking | MLflow |
| Modelo registrado | `mibici-equipo52-p1-demanda` |
| Versión | 1 |
| Artefacto local | `models/p1/` |
| Integración API | `api/routers/p1.py` |
| Esquemas de entrada y salida | `api/schemas/p1.py` |
| Servicio de predicción | `api/services/p1.py` |