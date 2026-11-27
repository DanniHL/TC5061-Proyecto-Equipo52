# Model Card · P3 Grupos de estaciones

Responsable: Joel Sánchez Olvera · Equipo 52 · TC5061.

Última actualización: 27 de noviembre de 2026.

Lo común a los tres modelos (fuente de datos, ética y método de monitoreo) está en el [índice de Model Cards](README.md).

## 1. Detalles del modelo

| Campo | Valor |
|---|---|
| Nombre en el Registry | `mibici-equipo52-p3-grupos` |
| Versión en producción | **v1**, alias `champion`; la v3 queda como `challenger` (evaluada en octubre, no promovida) |
| Algoritmo | K-Means (scikit-learn 1.9.1) con $k = 4$, `n_init = 20`, semilla 42, sobre 42 características estandarizadas |
| Corrida de origen | `f1721278a8f44272ac05bae27b51fe1b` (`p3-kmeans-k4`) |
| Commit | `d31f904e4c7ba6edd7059af7cc25bf4e74c47e45` |
| Formato | `skops`, 22 KB; la API lo sirve en `/p3/predict` |

## 2. Uso previsto

- **Para qué sirve:** segmentar las 469 estaciones de la red en cuatro perfiles operativos (núcleo central receptor, polos de empleo con vaciado vespertino, zonas residenciales emisoras matutinas y expansión periférica recreativa) para apoyar la planeación y el rebalanceo.
- **Quién lo usaría:** el área de operación y planeación de MiBici.
- **Para qué no sirve:** no predice demanda futura ni disponibilidad de bicicletas o anclajes, y los grupos no son una verdad de terreno sino una partición interpretada; no debe alimentar decisiones automáticas sin revisión humana.

## 3. Datos

| | Periodo | Volumen |
|---|---|---|
| Entrenamiento de la v1 | 2014-12-01 a 2026-09-30 | 469 estaciones activas: 371 consolidadas ($> 6$ meses) y 98 abiertas en 2026; 14 estaciones excluidas |
| Evaluación (A/B) | octubre de 2026 | 469 estaciones: 468 conocidas y 1 nueva |

Versión de datos `2026-09`; la corrida guarda la huella SHA-256 del manifiesto (`a9c1150d…`).

## 4. Variables

42 características por estación, calculadas en `equipo52_module/features.py` (`construir_matriz_p3`):

- **Nivel de actividad:** `salidas_por_mes`, `llegadas_por_mes` y `viajes_por_anclaje_mes`.
- **Firma horaria:** proporción de salidas en cada hora (`h00`–`h23`).
- **Firma semanal:** proporción de salidas por día (`d0`–`d6`).
- **Rol en la red:** `flujo_neto_por_mes`, `balance` y `num_destinos`.
- **Tipo de viaje:** `prop_viajes_redondos` y `duracion_mediana_min`.
- **Evolución:** `meses_activa`, `share_viajes_ultimos_12m` y `crecimiento_vs_esperado`.

El modelo espera las variables estandarizadas; la API aplica la media y la escala de entrenamiento guardadas en `models/p3/escalador.json`.

## 5. Métricas

Métrica principal: **coeficiente de silueta en el espacio de uso**, porque mide qué tan cohesivos y separados quedan los grupos sobre las variables de comportamiento (no sobre la geografía). De apoyo: Davies-Bouldin, Calinski-Harabasz y la estabilidad de la partición (ARI entre eras y por bootstrap).

| Modelo | Silueta validación (371 consolidadas) | Silueta red completa (469) |
|---|---|---|
| Baseline geográfico (K-Means $k = 5$ solo con lat/lon) | 0.039 | 0.019 |
| **v1** (K-Means $k = 4$) | **0.131** | 0.105 |
| v2 (K-Means $k = 2$) | 0.145 | **0.151** |

En octubre de 2026, la silueta de la partición inducida fue de 0.0701 para la v1 y 0.0657 para la v3 (A/B, sección 9).

## 6. Análisis por grupo

Los cuatro perfiles de la v1 sobre las 469 estaciones:

| Grupo | Perfil | Estaciones | Nuevas 2026 | Rasgo distintivo |
|---|---|---|---|---|
| 0 | Núcleo receptor | 134 | 0 | Mayor rotación (1,461 salidas/mes), flujo neto positivo, pico 18 h |
| 1 | Empleo vespertino | 132 | 28 | Vaciado 17–19 h, flujo neto negativo, concentrado en GDL/ZPN |
| 2 | Residencial matutino | 149 | 33 | Emisión 7–9 h, mayor número de estaciones |
| 3 | Periférico recreativo | 54 | 37 | Baja rotación (244 salidas/mes), uso dominical, estaciones más jóvenes |

Las estaciones abiertas en 2026 caen sobre todo en los grupos 1 a 3 (ninguna en el núcleo histórico), consistente con su rol de expansión de la red.

## 7. Limitaciones y riesgos

- **No hay etiqueta real.** Los grupos son una partición interpretada a partir de viajes; no existe una clasificación oficial de MiBici con la que validarla.
- **La silueta es baja en valor absoluto** (0.10–0.15): los perfiles forman un continuo urbano, no grupos aislados — el diagnóstico con HDBSCAN lo confirma. Las estaciones frontera pueden cambiar de grupo al reentrenar (en el A/B, 349 de 468 estaciones conocidas cambiaron de grupo entre v1 y v3).
- **Las estaciones nuevas tienen poca historia:** se proyectan con el escalador de toda la red y sus perfiles pueden reasignarse conforme acumulan meses.
- **El modelo no ve contexto externo:** clima, eventos, obras ni el rebalanceo en camión que hace la operación.

## 8. Consideraciones éticas

- El modelo trabaja solo con conteos de viajes agregados por estación; no usa ningún dato de las personas.
- Las zonas con más estaciones y más viajes dominan el ajuste de los centroides: el núcleo (134 estaciones) pesa más que la periferia (54). Las lecturas sobre grupos minoritarios deben revisarse con criterio operativo.
- La segmentación apoya la planeación; no debe usarse para retirar servicio de zonas sin análisis de equidad.

## 9. Monitoreo y reentrenamiento

- **Drift (Evidently, historia contra octubre de 2026):** 29 de 42 características cambiaron (69 %), por encima del 50 % que declara drift del conjunto: las firmas semanal y buena parte de la horaria, los volúmenes mensuales, `num_destinos` y `duracion_mediana_min`. El perfil de uso de octubre difiere de la historia completa con la que se formaron los grupos.
- **A/B (v1 contra v3, octubre de 2026):** silueta 0.0701 contra 0.0657, con intervalo de confianza del 95 % para la diferencia de −0.0124 a +0.0030. Como el intervalo incluye cero, se conservó la v1 como `champion` y la v3 quedó con el alias `challenger`.
- **Cómo regresar a la versión anterior:** la v1 sigue siendo `champion`, así que no hay nada que revertir; si una versión futura la sustituyera, la v1 quedaría con el alias `anterior` y bastaría exportarla a `models/p3` y reconstruir la imagen.
- **Recomendación:** reentrenar la misma receta con datos recientes (unos 11 segundos) y promover solo si el challenger gana el A/B.
- **Reportes:** `reports/monitoreo/drift_p3.html` y `reports/monitoreo/ab_p3.json`.

## 10. Historial de versiones

| Versión | Datos de entrenamiento | Silueta | Alias | Motivo |
|---|---|---|---|---|
| v1 | 2014-12 a 2026-09 | 0.105 (red) / 0.131 (consolidadas) | `champion` | Champion inicial (25-oct-2026): 5.6 veces la silueta del baseline geográfico y separa los 4 perfiles operativos |
| v2 | 2014-12 a 2026-09 | 0.151 (red) / 0.145 (consolidadas) | — | Challenger inicial ($k = 2$): máxima silueta global, pero solo segmenta a nivel macro; perdió el alias al registrarse la v3 |
| v3 | 2023-05 a 2026-09 (post-COVID) | 0.066 (octubre de 2026) | `challenger` | Receta de la v1 reentrenada con datos recientes (15-nov-2026); no ganó el A/B de octubre |
