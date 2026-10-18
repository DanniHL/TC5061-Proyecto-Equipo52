# TC5061 · Proyecto MiBici · Equipo 52

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Proyecto de la materia Operaciones de Aprendizaje Automático (TC5061, Maestría en Inteligencia Artificial Aplicada, Tecnológico de Monterrey). Usa los datos abiertos de viajes de MiBici, el sistema de bicicleta pública del Área Metropolitana de Guadalajara, para responder tres preguntas de operación y planeación con una base reproducible: mismos datos, mismo entorno y experimentos registrados en MLflow.

## Preguntas e integrantes

| # | Pregunta | Tipo de modelo | Responsable |
|---|---|---|---|
| P1 | ¿Cuántos viajes se iniciarán en una estación o zona en un periodo futuro? | Regresión | Daniela Hernández (DanniHL) |
| P2 | ¿Una estación estará en riesgo de quedarse sin bicicletas o sin anclajes libres en un periodo futuro? | Clasificación | Roberto Robles (robrobles-droid) |
| P3 | ¿Qué grupos de estaciones comparten patrones de uso y dónde están? | Agrupamiento | Joel Sánchez (PittJR1607) |

Además de su pregunta: Daniela Hernández administra el repositorio y Roberto Robles prepara la ingesta de datos (`dataset.py`) y mantiene `config.py` y este README.

## Datos

**Fuente:** sección de datos abiertos de MiBici, https://mibici.net/es/datos-abiertos/. MiBici es operado por BKT Bici Pública S.A. de C.V. y administrado por la Agencia Metropolitana de Servicios de Infraestructura para la Movilidad del Área Metropolitana de Guadalajara.

**Archivos usados:**
- 142 archivos mensuales de viajes, de diciembre de 2014 a septiembre de 2026 (`datos_abiertos_AAAA_MM.csv`).
- Nomenclatura de estaciones `nomenclatura_2026_09.csv`: número, nombre, latitud, longitud y capacidad de anclajes.

**Uso de los datos:** la página invita a explorar, experimentar y compartir los datos de uso de MiBici, pero no publica una licencia explícita y su pie de página indica «Todos los derechos reservados Mi Bici © 2026». Por eso el equipo los usa solo con fines académicos, cita la fuente y no los vuelve a publicar: los archivos no se suben a GitHub y cada integrante los descarga directamente de MiBici.

**Datos personales:** los archivos de viajes incluyen `Usuario_Id`, `Genero` y `Año_de_nacimiento`. No se necesitan para las preguntas del proyecto, así que no se leen: `config.py` define en `COLUMNAS_VIAJES` las únicas cinco columnas que se cargan.

**Cómo se obtienen:** nadie los descarga a mano. Cada integrante ejecuta `dataset.py` en su computadora (ver «Ingesta y experimento base»). Toda la carpeta `data/` está excluida de Git.

## Acuerdos de análisis

Están escritos en `equipo52_module/config.py` para que las tres preguntas usen los mismos valores.

- **Periodo:** historia completa, del 1 de diciembre de 2014 al 30 de septiembre de 2026 (`FECHA_INICIO`, `FECHA_FIN`). Versión de datos `2026-09`.
- **Estaciones excluidas:** las 14 dadas de baja, con prefijo "ex-" en la nomenclatura, 4 de ellas con capacidad 0 (`ESTACIONES_EXCLUIDAS`). La ingesta no las quita, para que el Parquet sea copia fiel de lo publicado; cada pregunta las quita al preparar sus datos. Las estaciones 124, 405 y 407 tienen viajes pero no aparecen en la nomenclatura, así que quedan fuera al unir con ella.
- **Estaciones nuevas:** las 100 estaciones abiertas entre mayo y agosto de 2026 se incluyen en la evaluación.
- **MLflow:** experimento `mibici-equipo52`, semilla `42`.

### Regla de riesgo de P2

MiBici no publica cuántas bicicletas hay en cada estación, así que el riesgo se aproxima con los viajes y la capacidad. Cada hora se revisan las 3 horas siguientes. Hay falla probable:
- **sin bicicletas**, si las salidas superan a las llegadas en un tercio de la capacidad de la estación o más;
- **sin anclajes libres**, si las llegadas superan a las salidas en un tercio de la capacidad o más.

La justificación está en los notebooks `1.0`, `1.1` y `1.2` de P2.

**Corrida adicional de P2 (Post-Covid):** además del modelo con toda la historia, P2 entrena un segundo modelo solo con datos desde el 09/05/2023, fin de la emergencia sanitaria. Así se comprueba si entrenar con el comportamiento reciente de los usuarios permite anticipar mejor las estaciones en riesgo. Las dos corridas se registran en MLflow y se comparan con las mismas métricas. La evaluación de ambas incluye las 100 estaciones abiertas en 2026.

### Criterios de P3 (viajes redondos y estaciones nuevas)

- **Viajes menores a 1 minuto y viajes redondos (`Origen_Id == Destino_Id`):** se descartan los 1,269,425 viajes `< 60 s` (`DURACION_MINIMA_SEGUNDOS = 60`), de los cuales el 95.0 % salen y regresan al mismo puerto por falla inmediata al desanclar. En cambio, los 1,516,002 viajes con `Origen_Id == Destino_Id` que duran $\ge 60$ s (4.03 % de los viajes útiles) se conservan: el 75.9 % dura entre 5 y 30 min (mediana 12.1 min) y se concentra en fin de semana y domingo de Vía RecreActiva, por lo que su frecuencia por estación (`prop_viajes_redondos`) es una señal real de uso recreativo. En `features.py` se incluye además `quitar_viajes_redondos_cortos` (1 a 3 min) para comprobar que las devoluciones tardías no alteran los grupos (ARI = 1.000 en $k = 2$ y 0.924 en $k = 4$).
- **Estaciones incorporadas recientemente (expansión 2026):** para que las 98 estaciones con viajes abiertas entre mayo y agosto de 2026 ($\le 6$ meses activas) compitan en igualdad con las 233 fundadoras ($> 96$ meses), ningún conteo crudo entra a la matriz de P3: todos los volúmenes se dividen entre los meses activos de cada estación (`salidas_por_mes`, `viajes_por_anclaje_mes`, `flujo_neto_por_mes`) y las firmas horaria y semanal se expresan como proporciones. Adicionalmente, `train_p3.py` incluye `--solo-consolidadas` para ajustar el escalador y los centroides solo con las 371 estaciones de $> 6$ meses y proyectar después las 98 nuevas.
- La evidencia completa está en los notebooks `3.0-js-p3-exploracion-estaciones.ipynb` y `3.1-js-p3-agrupamiento.ipynb`.

## Instalación

Requiere Python 3.12 y [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/DanniHL/TC5061-Proyecto-Equipo52.git
cd TC5061-Proyecto-Equipo52
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

En Windows, la activación es `.venv\Scripts\activate`.

## Ingesta y experimento base

Con el entorno activado, desde la carpeta del proyecto:

```bash
python equipo52_module/dataset.py
```

El comando:
1. Busca en la página de datos abiertos las ligas de los archivos del periodo de análisis (142 meses y la nomenclatura).
2. Descarga a `data/raw/` solo los que falten. La primera vez son unos 3.1 GB.
3. Lee de cada mes solo las columnas de `COLUMNAS_VIAJES`, resuelve las dos codificaciones y los dos formatos de fecha que usa MiBici, y guarda todos los viajes en `data/interim/viajes.parquet`. Si encuentra una fecha con un formato nuevo, se detiene y avisa.
4. Escribe `references/manifiesto_datos.csv` con el número de registros, la huella SHA-256 y la liga de cada archivo. El manifiesto sí se guarda en Git: si otra persona corre la ingesta, sus huellas deben coincidir.

Resultado esperado: 142 meses y 39,078,984 viajes.

MiBici puede volver a publicar archivos ya existentes: en octubre de 2026 corrigió abril y mayo de 2023 (la hora de los viajes y 144 viajes más). Si tu manifiesto no coincide con el del repositorio, borra de `data/raw/` los archivos con huella distinta y vuelve a correr la ingesta.

### Experimento base de P1

Requiere haber corrido la ingesta: `train_p1.py` prepara sus datos desde `data/interim/viajes.parquet` (periodo del proyecto, sin las 14 estaciones excluidas). Desde la raíz del proyecto se ejecuta:
```bash
python equipo52_module/modeling/train_p1.py
```

- `Baseline`: utiliza la demanda observada en el periodo anterior como referencia para el pronóstico.
- `Regresión lineal`: estima la demanda futura a partir de las variables predictoras disponibles.
- `Evaluación`: compara los modelos mediante MAE y RMSE. MAE mide el error absoluto promedio, mientras que RMSE penaliza con mayor peso los errores grandes.
- `Registro en MLflow`: permite comparar las corridas y conservar los parámetros, las métricas, la información de los datos y los artefactos generados por cada modelo.

Las corridas deben mantener el mismo criterio de evaluación para que la comparación sea consistente. Los resultados permitirán determinar si los modelos de regresión aportan una mejora frente a la persistencia y si su precisión resulta útil para la planeación operativa de MiBici.


### Experimento base de P2

Requiere haber corrido la ingesta. Desde la carpeta del proyecto:

```bash
python equipo52_module/modeling/train_p2.py --modelo base --periodo completo
python equipo52_module/modeling/train_p2.py --modelo hgb --periodo completo
python equipo52_module/modeling/train_p2.py --modelo hgb --periodo postcovid
python equipo52_module/modeling/train_p2.py --modelo hgb --periodo completo --balanceado
```

- `base`: repite lo que pasó en las últimas 4 semanas a la misma hora y día (falla si ocurrió en 2 o más).
- `hgb`: HistGradientBoosting de scikit-learn con calendario, capacidad, ubicación, movimientos de las horas anteriores y fallas de las últimas 4 semanas.
- `completo` entrena desde 2014-12-01 y `postcovid` desde 2023-05-09; las dos se prueban con abril a septiembre de 2026.
- `--balanceado` da a las fallas el mismo peso que a las horas normales (más fallas detectadas, más alertas falsas).

Cada corrida se registra en el experimento `mibici-equipo52` con fechas, versión y huella de datos, semilla, hiperparámetros, métricas `test_*`, etiquetas (equipo, pregunta, commit) y artefactos (tabla de aciertos y errores, porcentaje de clases por año y el modelo).

### Experimento base de P3

Requiere haber corrido la ingesta. Desde la carpeta del proyecto:

```bash
python equipo52_module/modeling/train_p3.py --modelo geo
python equipo52_module/modeling/train_p3.py --modelo kmeans
python equipo52_module/modeling/train_p3.py --modelo kmeans --k 4
python equipo52_module/modeling/train_p3.py --modelo kmeans --k 4 --solo-consolidadas
python equipo52_module/modeling/train_p3.py --modelo ward
python equipo52_module/modeling/train_p3.py --modelo hdbscan
```

- `geo`: línea base. K-Means con k = 5 solo sobre latitud y longitud; su silueta se mide en el espacio de las 42 características de uso (`hist_silhouette = 0.019`), para comprobar si las zonas geográficas son coherentes en comportamiento.
- `kmeans`: candidato principal sobre las 469 estaciones activas y 42 características de uso calculadas con toda la historia; elige k por silueta en el barrido 2-12 (`k = 2`, `hist_silhouette = 0.151`, 8× la línea base) o fijo con `--k 4` para obtener los 4 perfiles operativos de planeación (`hist_silhouette = 0.105`).
- `--solo-consolidadas`: ajusta el escalador y los centroides únicamente con las 371 estaciones consolidadas ($> 6$ meses activas, `consolidadas_silhouette = 0.127` en $k = 4$) y asigna las 98 estaciones abiertas en 2026 al centroide más cercano; `--sin-redondos-cortos` descarta viajes `Origen_Id == Destino_Id` de 1 a 3 min.
- `ward`: agrupamiento jerárquico con enlace Ward al mismo procedimiento de k (`hist_silhouette = 0.129`); reporta el ARI de concordancia con K-Means (`0.565`) y el dendrograma.
- `hdbscan`: diagnóstico de densidad (`--min-cluster-size`, 15 por omisión). Muestra que las estaciones forman un continuo urbano y no islas aisladas, e identifica las estaciones más atípicas.
- Para los candidatos se reporta la estabilidad: `estabilidad_eras_ari` compara la asignación de las 274 estaciones activas en ambas mitades de la historia (corte `2020-09-01`, ARI = `0.618` en $k = 2$ y `0.571` en $k = 4$) y `estabilidad_bootstrap_ari` re-entrena 30 veces con el 80 % de las estaciones (ARI = `0.952` en $k = 2$ y `0.841` en $k = 4$).

Cada corrida se registra en `mibici-equipo52` con fechas, versión y huella de datos, semilla, hiperparámetros, métricas `hist_*` y de estabilidad, etiquetas (equipo, pregunta, commit) y artefactos (mapa de grupos, perfiles horarios, silueta por k, dendrograma, matriz estación × característica y descripción de grupos con desglose por municipio `GDL`/`ZPN`/`TLQ`). La matriz y las descripciones quedan en `data/processed/` y las figuras en `reports/figures/`.

### Ver los experimentos

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Abrir http://127.0.0.1:5000.

## Forma de trabajo en Git

- `main` está protegida: los cambios entran solo por Pull Request con al menos una aprobación de otra persona del equipo.
- Una rama por tarea, con prefijo según el tipo de cambio: `feature/` (funcionalidad nueva), `fix/` (corrección), `docs/` (documentación).
- Mensajes de commit con el formato Conventional Commits, por ejemplo `feat(p2): …`, `fix: …`, `docs: …`, `build: …`.
- Los PR se unen con *merge commit* (sin *squash*) para conservar quién hizo cada commit.
- Los notebooks se nombran `número-iniciales-descripción.ipynb`, por ejemplo `1.0-rr-p2-exploracion-datos.ipynb`.

## Estructura del proyecto

```
├── LICENSE            <- Licencia del código (MIT); no aplica a los datos de MiBici
├── Makefile           <- Comandos de apoyo
├── README.md          <- Este documento
├── data               <- Datos del proyecto; toda la carpeta está excluida de Git
│   ├── external       <- Datos de otras fuentes
│   ├── interim        <- Datos intermedios ya transformados
│   ├── processed      <- Datos finales para entrenar
│   └── raw            <- Datos originales de MiBici, sin modificar
│
├── docs               <- Documentación del proyecto (mkdocs)
├── models             <- Modelos entrenados
├── notebooks          <- Notebooks de exploración: número, iniciales y descripción
    └── 1.0-dh-p1-exploración-datos.ipynb
    └── 2.0-dh-p1-preparación-modelado.ipynb
    └── 1.0-rr-p2-exploracion-datos.ipynb
    └── 1.1-rr-p2-exploracion-estaciones.ipynb
    └── 1.2-rr-p2-exploracion-flujo-neto.ipynb
    └── 3.0-js-p3-exploracion-estaciones.ipynb
    └── 3.1-js-p3-agrupamiento.ipynb
├── pyproject.toml     <- Datos del paquete equipo52_module y configuración de herramientas
├── references         <- Diccionarios de datos, manuales y material de consulta
│   ├── manifiesto_datos.csv
│   └── Propuesta_P3_Equipo52.docx
├── reports
│   └── figures        <- Gráficas generadas para los reportes (prefijo p1_, p2_, p3_ según la pregunta)
├── requirements.txt   <- Paquetes con versión fija para reproducir el entorno
│
└── equipo52_module    <- Código del proyecto
    ├── __init__.py
    ├── config.py      <- Rutas y acuerdos del equipo (periodo, exclusiones, MLflow)
    ├── dataset.py     <- Ingesta de los datos de MiBici
    ├── features.py    <- Variables para los modelos
    ├── modeling
    │   ├── __init__.py
    │   ├── predict.py <- Predicciones con modelos entrenados
    │   ├── train_p1.py
    │   ├── train_p2.py
    │   ├── train_p3.py
    │   └── train.py   <- Entrenamiento de modelos
    └── plots.py       <- Gráficas
```
