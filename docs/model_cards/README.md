# Model Cards · Proyecto MiBici · Equipo 52

Cada modelo del proyecto tiene su propio Model Card, con las mismas diez secciones (Mitchell et al., 2019). Lo que es común a los tres se documenta una sola vez en esta página.

## Fichas por modelo

| Pregunta | Modelo en el Registry | Versión en producción | Responsable | Ficha |
|---|---|---|---|---|
| P1 · Demanda diaria | `mibici-equipo52-p1-demanda` | v2 (`champion`) | Danni | [p1_demanda.md](p1_demanda.md) |
| P2 · Riesgo operativo | `mibici-equipo52-p2-riesgo` | v3 (`champion`) | Roberto | [p2_riesgo.md](p2_riesgo.md) |
| P3 · Grupos de estaciones | `mibici-equipo52-p3-grupos` | v1 (`champion`) | Joel | [p3_grupos.md](p3_grupos.md) |

## Lo común a los tres modelos

### Fuente de datos

- Datos abiertos de MiBici: 142 archivos mensuales con 39,078,984 viajes, de diciembre de 2014 a septiembre de 2026 (versión de datos `2026-09`). Octubre de 2026 se usa solo para monitorear.
- Cada corrida registra en MLflow la huella SHA-256 del manifiesto de datos y el commit, para que el resultado se pueda reproducir.
- Los datos no se guardan en Git por su tamaño; se generan con la ingesta (`mlflow run . -e ingest`).

### Privacidad y uso responsable

- Los archivos de MiBici incluyen `Usuario_Id`, `Genero` y `Año_de_nacimiento`. **La ingesta no lee esas columnas**: los tres modelos trabajan solo con viajes agregados por estación, día u hora.
- Los modelos apoyan decisiones de operación; no deben usarse para decisiones automáticas sin revisión humana.

### Monitoreo y promoción

1. **Drift:** Evidently compara los datos de entrenamiento contra octubre de 2026.
2. **Challenger:** se reentrena la receta del champion con datos más recientes.
3. **A/B:** champion contra challenger sobre los mismos datos, con bootstrap pareado de 1,000 remuestras. El challenger gana solo si todo el intervalo de confianza del 95 % lo favorece; la regla se fija antes de ver los resultados.
4. **Promoción:** los alias del Registry cambian (`champion` es producción, `anterior` es el respaldo y `challenger` la candidata) y quedan etiquetas con el motivo. Para revertir, se exporta la versión `@anterior`.

## Referencia

Mitchell, M., Wu, S., Zaldivar, A., Barnes, P., Vasserman, L., Hutchinson, B., Spitzer, E., Raji, I. D., & Gebru, T. (2019). Model cards for model reporting. En *Proceedings of the Conference on Fairness, Accountability, and Transparency* (pp. 220–229). Association for Computing Machinery. https://doi.org/10.1145/3287560.3287596