# Model Card · P2 Riesgo operativo de estaciones

Responsable: Roberto Robles Arroyo · Equipo 52 · TC5061.

Última actualización: 24 de noviembre de 2026.

Lo común a los tres modelos (fuente de datos, ética y método de monitoreo) está en el [índice de Model Cards](README.md).

## 1. Detalles del modelo

| Campo | Valor |
|---|---|
| Nombre en el Registry | `mibici-equipo52-p2-riesgo` |
| Versión en producción | **v3**, alias `champion`; la v1 queda como `anterior` |
| Algoritmo | HistGradientBoostingClassifier (scikit-learn 1.9.1), con tres clases |
| Hiperparámetros | `learning_rate = 0.1` y `peso_fallas = 3` |
| Corrida de origen | `d444d7cc1e624e439c735525bbb86fd6` (`challenger_p2.py`) |
| Formato | `skops`, 4.2 MB; la API lo sirve en `/p2/predict` |

## 2. Uso previsto

- **Para qué sirve:** anticipar, para cada estación y hora, si quedará sin bicicletas, sin anclajes o en estado normal, para decidir a qué estaciones mandar primero el rebalanceo.
- **Quién lo usaría:** el área de operación de MiBici.
- **Para qué no sirve:** no mide cuántas bicicletas hay en realidad, y no debe usarse para tomar decisiones automáticas sin que alguien las revise.

## 3. Datos

| | Periodo | Volumen |
|---|---|---|
| Entrenamiento de la v3 | 2014-12-01 a 2026-09-30 | 28,028,515 horas-estación de 469 estaciones |
| Evaluación (A/B) | octubre de 2026 | 346,943 horas-estación: 370 estaciones conocidas y 99 nuevas |

Versión de datos `2026-09`; cada corrida guarda la huella SHA-256 de los datos que usó.

## 4. Variables

13 variables, calculadas en `equipo52_module/features.py`:

- **Calendario:** `hora_del_dia`, `dia_semana`, `mes` y `fin_de_semana`.
- **Estación:** `capacidad`, `latitud` y `longitud`.
- **Movimientos recientes:** salidas y llegadas de la hora anterior y de las 3 horas anteriores.
- **Historial de fallas:** `semanas_sin_bicicletas` y `semanas_sin_anclajes` (cuántas de las últimas 4 semanas tuvieron falla a la misma hora).

**Etiqueta (lo que el modelo predice):** «sin bicicletas» si las salidas superan a las llegadas en al menos un tercio de la capacidad; «sin anclajes» en el caso contrario, y «normal» en el resto.

## 5. Métricas

Métrica principal: **F1 de fallas**, el promedio del F1 de «sin bicicletas» y de «sin anclajes». Se eligió porque las fallas son poco frecuentes: la exactitud sería de 0.90 aunque el modelo casi no detectara fallas.

| Modelo | F1 de fallas | Dónde se midió |
|---|---|---|
| Baseline (repite las últimas 4 semanas) | 0.405 | prueba, abr-sep 2026 |
| v1 | 0.449 | prueba, abr-sep 2026 |
| v1 | 0.4768 | octubre de 2026 |
| **v3** | **0.4787** | octubre de 2026 |

En octubre, la v3 acierta 4 de cada 10 alertas (precisión de 0.41 en «sin bicicletas» y 0.44 en «sin anclajes») y detecta un poco más de la mitad de las fallas reales (recall de 0.54 y 0.55).

## 6. Análisis por grupo

| Grupo (octubre de 2026) | Estaciones | F1 v1 | F1 v3 |
|---|---|---|---|
| Estaciones conocidas | 370 | 0.4764 | 0.4774 |
| Estaciones nuevas de 2026 | 99 | 0.4836 | 0.4951 |

El modelo no empeora en las 99 estaciones abiertas en 2026, aunque tienen poca historia, y la v3 mejora más en ellas.

## 7. Limitaciones y riesgos

- **El riesgo es una estimación.** MiBici no publica cuántas bicicletas había en cada estación hora por hora, así que la etiqueta se calcula con las salidas y llegadas de los viajes. Si el personal de MiBici mueve bicicletas en camión (rebalanceo), ese movimiento no aparece en los datos.
- **Es una herramienta para priorizar, no una certeza.** Acierta unas 4 de cada 10 alertas y detecta poco más de la mitad de las fallas reales.
- **Es el endpoint más lento de la API.** Con 100 usuarios simultáneos, su tiempo de respuesta mediano fue de 2.5 segundos (pruebas con Locust, semana 10).

## 8. Consideraciones éticas

- El modelo trabaja solo con conteos de viajes agregados por estación y hora; no usa ningún dato de las personas.
- Las zonas con más estaciones y más viajes aportan más datos al entrenamiento. Por eso, las alertas en zonas con poca actividad deben revisarse con criterio operativo, para que no queden desatendidas.

## 9. Monitoreo y reentrenamiento

- **Drift (Evidently, entrenamiento contra octubre de 2026):** 6 de 13 variables cambiaron (46 %), por debajo del 50 % que marca drift del conjunto. Cambiaron sobre todo la longitud, la capacidad y los movimientos recientes, por las 99 estaciones nuevas; la proporción de fallas no cambió (*covariate shift*).
- **A/B (v1 contra v3, octubre de 2026):** el F1 de fallas mejoró +0.0018, con un intervalo de confianza del 95 % de +0.0005 a +0.0033. Como todo el intervalo es mayor que cero, se promovió la v3 (`promover_p2.py`).
- **Cómo regresar a la versión anterior:** `models:/mibici-equipo52-p2-riesgo@anterior` apunta a la v1; basta exportarla a `models/p2` y reconstruir la imagen.
- **Recomendación:** reentrenar cada mes con la misma receta (unos 4.5 minutos) y promover solo si gana el A/B.
- **Reportes:** `reports/monitoreo/drift_p2.html` y `reports/monitoreo/ab_p2.json`.

## 10. Historial de versiones

| Versión | Datos de entrenamiento | F1 de fallas | Alias | Motivo |
|---|---|---|---|---|
| v1 | 2014-12 a 2026-03 | 0.449 (prueba) | `anterior` | Champion inicial (24-oct-2026): el mejor en validación, 10.9 % mejor que el baseline |
| v2 | 2023-05 a 2026-03 (post-COVID) | 0.444 (prueba) | — | Challenger inicial, 6.4 veces más barato de entrenar; no se promovió |
| v3 | 2014-12 a 2026-09 | 0.4787 (octubre) | `champion` | Ganó el A/B de octubre (11-nov-2026) |
