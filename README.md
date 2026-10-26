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


### P2 · Experimentos y modelo en producción (semana 6)

Requiere haber corrido la ingesta y tener encendido el servidor de MLflow (ver «Ver los experimentos en MLflow»). En la terminal donde se lanzan las corridas:

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

Todas las corridas de la entrega se lanzan con `mlflow run`. `--env-manager local` usa el entorno virtual activo (Python 3.12 con las versiones de `requirements.txt`) y `--experiment-name` hace que la corrida del proyecto nazca en `mibici-equipo52`, el mismo experimento de la entrega anterior.

**Partición en el tiempo.** Entrenamiento desde 2014-12-01 (o desde 2023-05-09 en `postcovid`) hasta septiembre de 2025, validación de octubre de 2025 a marzo de 2026 y prueba de abril a septiembre de 2026. Se elige con validación (`val_*`) y se confirma con prueba (`test_*`). El modelo final se reentrena con entrenamiento + validación: es el mismo periodo de la entrega anterior, así que las métricas de prueba se comparan directo con el baseline.

**1. Baseline** (repite lo que pasó en las últimas 4 semanas a la misma hora y día):

```bash
mlflow run . -e train --env-manager local --experiment-name mibici-equipo52 -P pregunta=p2 -P modelo=base -P periodo=completo
```

**2. Búsqueda de hiperparámetros con nested runs** (HistGradientBoosting; una corrida padre y 8 hijas: `peso_fallas` 2, 3, 5 y 8 × `learning_rate` 0.05 y 0.1; cada hija se mide solo en validación):

```bash
mlflow run . -e train --env-manager local --experiment-name mibici-equipo52 -P pregunta=p2 -P modelo=busqueda -P periodo=completo
```

**3. Candidatas** con la mejor combinación de la búsqueda (`peso_fallas = 3`, `learning_rate = 0.1`, F1 de fallas en validación 0.483 contra 0.442 del baseline):

```bash
mlflow run . -e train --env-manager local --experiment-name mibici-equipo52 -P pregunta=p2 -P modelo=hgb -P periodo=completo -P learning_rate=0.1 -P peso_fallas=3
mlflow run . -e train --env-manager local --experiment-name mibici-equipo52 -P pregunta=p2 -P modelo=logistica -P periodo=completo -P peso_fallas=3
mlflow run . -e train --env-manager local --experiment-name mibici-equipo52 -P pregunta=p2 -P modelo=hgb -P periodo=postcovid -P learning_rate=0.1 -P peso_fallas=3
```

- `hgb`: HistGradientBoosting con calendario, capacidad, ubicación, movimientos de las horas anteriores y fallas de las últimas 4 semanas.
- `logistica`: regresión logística (imputación por mediana, estandarización); más simple y con coeficientes interpretables.
- `peso_fallas`: cuánto pesa cada hora de falla frente a una normal al entrenar. Con 1 el modelo da pocas alertas y se le escapan muchas fallas; con `balanceado=true` (18 a 19 veces) da demasiadas alertas falsas.

**4. Selección.** El notebook `notebooks/2.3-rr-p2-seleccion-modelo.ipynb` compara las candidatas contra el baseline (mejora, F1 de fallas, estabilidad y costo) y guarda la decisión en `reports/p2_seleccion_modelo.json`.

**5. Registro y paso a producción.** Registra el modelo elegido como `mibici-equipo52-p2-riesgo` con su linaje (corrida, commit, versión de datos), pone los alias `champion` (producción) y `challenger` (siguiente mejor) y prueba la carga:

```bash
python -m equipo52_module.modeling.registrar_p2
```

El modelo en producción se carga siempre por su alias, así que cambiar de versión no requiere cambiar el código que lo usa:

```python
import mlflow
modelo = mlflow.pyfunc.load_model("models:/mibici-equipo52-p2-riesgo@champion")
```

Se usan aliases y no etapas porque MLflow declaró obsoletas las etapas (`Staging`, `Production`) desde la versión 2.9.

**Qué registra cada corrida en `mibici-equipo52`:** fechas, versión y huella de datos, semilla e hiperparámetros; métricas `val_*`, `test_*` y `segundos_entrenamiento`; etiquetas `equipo`, `pregunta`, `commit`, `question`, `model_family` y `stage_of_work` (`baseline`, `search` o `candidate`); conjuntos de entrenamiento y prueba con `mlflow.log_input`; artefactos en `plots/` (matriz de confusión, curva precisión-recall), `data/` (manifiesto de datos, tablas) y `config/` (configuración de la corrida), y el modelo con firma y ejemplo de entrada.

**Pruebas rápidas sin MLflow Project** (solo para depurar; no cuentan para la entrega): `python -m equipo52_module.modeling.train_p2 --modelo hgb --peso-fallas 3`. Los notebooks `2.1` y `2.2` explican la partición y cómo leer las corridas desde Python.

### P3 · Experimentos y modelo en producción (semana 6)

Requiere haber corrido la ingesta (`mlflow run . -e ingest --env-manager local --experiment-name mibici-equipo52`) y tener encendido el servidor de MLflow (ver «Ver los experimentos en MLflow»). En la terminal donde se lanzan las corridas:

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

Todas las corridas de P3 se lanzan con `mlflow run` en el experimento `mibici-equipo52`:

**Validación y evaluación en toda la red.** Se mide la cohesión y separación en las 371 estaciones consolidadas ($> 6$ meses activas, `val_silhouette`) y en las 469 estaciones activas de toda la red incluyendo las 98 abiertas en mayo–agosto de 2026 (`hist_silhouette`, `hist_davies_bouldin`, `hist_calinski_harabasz`), junto con la estabilidad temporal entre eras (`estabilidad_eras_ari`, corte `2020-09-01` sobre las 274 estaciones comunes) y por remuestreo (`estabilidad_bootstrap_ari`, 30 remuestreos al 80 %).

**1. Baseline** (K-Means con $k = 5$ solo sobre latitud y longitud, evaluado en el espacio de las 42 características de uso):

```bash
mlflow run . -e train --env-manager local --experiment-name mibici-equipo52 -P pregunta=p3 -P modelo=geo
```

**2. Búsqueda de hiperparámetros con nested runs** (K-Means; una corrida padre `p3-busqueda-kmeans` y 10 hijas: `k` de 2 a 6 × `solo_consolidadas` `False` y `True`):

```bash
mlflow run . -e train --env-manager local --experiment-name mibici-equipo52 -P pregunta=p3 -P modelo=busqueda
```

**3. Candidatas** (óptimo operativo de 4 perfiles `k = 4`, óptimo global macro `k = 2`, variante de ajuste en estaciones consolidadas, agrupamiento jerárquico `ward` y densidad `hdbscan`):

```bash
mlflow run . -e train --env-manager local --experiment-name mibici-equipo52 -P pregunta=p3 -P modelo=kmeans -P k=4
mlflow run . -e train --env-manager local --experiment-name mibici-equipo52 -P pregunta=p3 -P modelo=kmeans -P k=2
mlflow run . -e train --env-manager local --experiment-name mibici-equipo52 -P pregunta=p3 -P modelo=kmeans -P k=4 -P solo_consolidadas=true
mlflow run . -e train --env-manager local --experiment-name mibici-equipo52 -P pregunta=p3 -P modelo=ward
mlflow run . -e train --env-manager local --experiment-name mibici-equipo52 -P pregunta=p3 -P modelo=hdbscan -P min_cluster_size=15
```

- `kmeans` ($k = 4$, `p3-kmeans-k4`): candidato elegido (`champion`); cuadruplica la resolución del corte macro y multiplica por 5.6 la silueta del baseline (`hist_silhouette = 0.105` vs. `0.019`, `val_silhouette = 0.131` vs. `0.039`, `Davies-Bouldin = 2.075` en validación y `2.309` en la red completa), separando los 4 perfiles operativos de planeación (núcleo central receptor, polos de empleo con vaciado vespertino 17–19 h, zonas residenciales emisoras matutinas 7–9 h y expansión periférica recreativa dominical).
- `kmeans` ($k = 2$, `p3-kmeans-k2`): `challenger`; alcanza el máximo matemático global de silueta (`hist_silhouette = 0.151`, `val_silhouette = 0.145`, `estabilidad_bootstrap_ari = 0.952`, `estabilidad_eras_ari = 0.618`) para segmentación macro de la red.
- `solo_consolidadas=true` (`p3-kmeans-k4-consolidadas`): ajusta el escalador y los centroides con las 371 estaciones consolidadas ($> 6$ meses, `val_silhouette = 0.127`) y proyecta las 98 estaciones abiertas en 2026.
- `ward` (`p3-ward-k2`): agrupamiento jerárquico con enlace Ward (`hist_silhouette = 0.129`, `estabilidad_eras_ari = 0.648`, dendrograma y concordancia con K-Means `ARI = 0.565`).
- `hdbscan` (`p3-hdbscan-mcs15`): diagnóstico de densidad que confirma el continuo urbano de la red e identifica estaciones atípicas.

**4. Selección.** El notebook `notebooks/3.2-js-p3-mlflow-seleccion-modelo.ipynb` comprueba que las corridas oficiales de `mlflow run` coincidan con la referencia del notebook `3.1`, revisa los artefactos y la búsqueda con nested runs, compara las candidatas con los cuatro criterios de la actividad y respalda la decisión en `reports/p3_seleccion_modelo.json`.

**5. Registro y paso a producción.** Registra el modelo elegido y el challenger como versiones de `mibici-equipo52-p3-grupos` con su linaje (experimento, corrida de origen, commit, versión y huella de datos), asigna los alias `champion` (producción) y `challenger` (siguiente mejor) y verifica la carga desde producción:

```bash
python -m equipo52_module.modeling.registrar_p3
```

El modelo en producción se carga siempre por su nombre y alias, lo que desacopla el código de inferencia de la versión activa y permite revertir o actualizar el modelo reasignando el alias (esquema vigente de MLflow desde que las etapas clásicas fueron declaradas obsoletas en MLflow 2.9):

```python
import mlflow
modelo = mlflow.pyfunc.load_model("models:/mibici-equipo52-p3-grupos@champion")
```

**Qué registra cada corrida de P3 en `mibici-equipo52`:** fechas, versión y huella de datos, semilla e hiperparámetros; métricas `val_*`, `hist_*`, `estabilidad_eras_ari`, `estabilidad_bootstrap_ari` y `segundos_entrenamiento`; etiquetas `equipo`, `pregunta`, `commit`, `question`, `model_family` y `stage_of_work` (`baseline`, `search` o `candidate`); conjuntos de estaciones con `mlflow.log_input`; artefactos en `plots/` (mapa de grupos, perfiles horarios, silueta por $k$, dendrograma), `data/` (manifiesto de datos, matriz estación × característica y descripción de grupos con desglose por municipio `GDL`/`ZPN`/`TLQ`) y `config/` (configuración de la corrida y del proyecto), y el modelo con firma y ejemplo de entrada.


### Ver los experimentos en MLflow

Para registrar y consultar los experimentos, inicia el servidor local desde la raíz del proyecto y con el entorno virtual activado:

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Mantén la terminal abierta mientras ejecutas los entrenamientos. El servidor utiliza `mlflow.db` como base de datos de seguimiento y expone la interfaz web en:

http://127.0.0.1:5000

Los scripts de entrenamiento utilizan la variable `MLFLOW_TRACKING_URI`, configurada por defecto como `http://127.0.0.1:5000`. Así, P1, P2 y P3 registran sus experimentos en el mismo servidor local.

## Forma de trabajo en Git

- `main` está protegida: los cambios entran solo por Pull Request con al menos una aprobación de otra persona del equipo.
- Una rama por tarea, con prefijo según el tipo de cambio: `feature/` (funcionalidad nueva), `fix/` (corrección), `docs/` (documentación).
- Mensajes de commit con el formato Conventional Commits, por ejemplo `feat(p2): …`, `fix: …`, `docs: …`, `build: …`.
- Los PR se unen con *merge commit* (sin *squash*) para conservar quién hizo cada commit.
- Los notebooks se nombran `número-iniciales-descripción.ipynb`, por ejemplo `1.0-rr-p2-exploracion-datos.ipynb`.

## Registro y selección de modelo en MLflow
El uso de MLflow permite mantener la relación entre el modelo registrado, su corrida de origen y las métricas utilizadas para evaluarlo. El alias champion representa la versión seleccionada por el equipo y puede actualizarse cuando una nueva versión demuestre un mejor desempeño conforme a los criterios de evaluación definidos.

### P1
Para la pregunta P1, se seleccionó una regresión lineal con variables cíclicas del calendario y preprocesamiento mediante StandardScaler, con el objetivo de predecir la demanda diaria de viajes de MiBici.

El modelo se registró en MLflow Model Registry con el nombre 
```bash
mibici-equipo52-p1-demanda. 
```
La versión 1 se identificó con el alias champion, que permite recuperar la versión seleccionada sin depender de un número de versión fijo en el código.

La selección se basó en la evaluación con datos de prueba. El modelo obtuvo un MAE de 6.6464 y un RMSE de 11.3658, mejorando los resultados del modelo de persistencia utilizado como línea base. La corrida de origen fue Pipeline_Regresion_Ciclica_P1, con identificador db35a9caee2b4d25bc6ea3faee674842.

Para cargar el modelo registrado mediante su alias:
```bash
import mlflow.sklearn

modelo = mlflow.sklearn.load_model(
    "models:/mibici-equipo52-p1-demanda@champion"
)
```

### P2
Para la pregunta P2
El modelo se registró en MLflow Model Registry con el nombre 
```bash
mibici-equipo52-p2-riesgo. 
```
La versión 1 se identificó con el alias champion, que permite recuperar la versión seleccionada sin depender de un número de versión fijo en el código.
L
a versión seleccionada se identifica mediante el alias champion, de modo que pueda cargarse sin depender de un número de versión específico:
```bash
import mlflow.sklearn

modelo = mlflow.sklearn.load_model(
    "models:/mibici-equipo52-p2-riesgo@champion"
)
```

### P3

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
    └── 2.1-rr-p2-particion-validacion.ipynb
    └── 2.2-rr-p2-mlflow-corridas-artefactos.ipynb
    └── 2.3-rr-p2-seleccion-modelo.ipynb
    └── 3.0-js-p3-exploracion-estaciones.ipynb
    └── 3.1-js-p3-agrupamiento.ipynb
    └── 3.2-js-p3-mlflow-seleccion-modelo.ipynb
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
    │   ├── registrar_p2.py
    │   ├── registrar_p3.py
    │   ├── train_p1.py
    │   ├── train_p2.py
    │   ├── train_p3.py
    │   └── train.py   <- Entrenamiento de modelos
    └── plots.py       <- Gráficas
```
