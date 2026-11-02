# TC5061 · Proyecto MiBici · Equipo 52

Proyecto de Machine Learning Operations (MLOps) para analizar los viajes del sistema público de bicicletas **MiBici**, del Área Metropolitana de Guadalajara, y desarrollar modelos de aprendizaje automático para responder tres preguntas de negocio.

El proyecto integra un flujo de trabajo que contempla la obtención y preparación de datos, el desarrollo de modelos, la ejecución de experimentos, la evaluación de resultados y el registro de modelos mediante MLflow. También incorpora pruebas automatizadas para verificar el funcionamiento de los componentes del proyecto.

---

## 1. Descripción del proyecto

MiBici es el sistema de bicicletas públicas del Área Metropolitana de Guadalajara. A través de sus datos abiertos es posible analizar los viajes realizados por los usuarios, identificar patrones de movilidad y estudiar el comportamiento de las estaciones.

El objetivo del proyecto es utilizar estos datos para desarrollar soluciones de Machine Learning que permitan analizar la demanda, anticipar situaciones de riesgo operativo y agrupar estaciones según sus patrones de uso.

El equipo trabaja con tres preguntas:

| Pregunta | Objetivo | Responsable |
|---|---|---|
| P1 | Analizar y pronosticar la demanda diaria de viajes por estación. | Danni |
| P2 | Identificar y anticipar situaciones de riesgo relacionadas con la disponibilidad de bicicletas y anclajes. | Roberto |
| P3 | Identificar grupos de estaciones con patrones de comportamiento similares. | Joel |

Los tres modelos utilizan una fuente de datos común y comparten la infraestructura de experimentación y seguimiento de MLflow.

## 2. Fuente de datos

Los datos utilizados provienen del portal de datos abiertos de MiBici:

- Datos históricos de viajes: archivos mensuales `datos_abiertos_AAAA_MM.csv`.
- Catálogo de estaciones: archivo `nomenclatura_2026_09.csv`.

Los archivos de viajes contienen información sobre los recorridos realizados, incluyendo identificador del viaje, fechas y horas de inicio y finalización, así como las estaciones de origen y destino.

El catálogo de estaciones contiene información de identificación, nombre, ubicación geográfica y capacidad de las estaciones.

### 2.1 Periodo de análisis

El periodo considerado comprende del **1 de diciembre de 2014 al 30 de septiembre de 2026**.

| Elemento | Descripción |
|---|---|
| Inicio del periodo | 2014-12-01 |
| Fin del periodo | 2026-09-30 |
| Versión de datos | 2026-09 |
| Archivos mensuales | 142 |
| Registros de viajes | 39,078,984 |
| Tamaño aproximado de los datos | 3.1 GB |
| Catálogo de estaciones | `nomenclatura_2026_09.csv` |

### 2.2 Consideraciones sobre los datos

Durante la preparación se identificaron diferencias entre los archivos históricos, principalmente en los nombres de columnas y formatos de fechas. Por esta razón, el proceso de ingesta contempla la normalización de los datos antes de utilizarlos en los modelos.

También se identificaron las siguientes condiciones:

- Existen 14 estaciones excluidas de los análisis por los criterios definidos para el proyecto.
- Las estaciones 124, 405 y 407 no cuentan con correspondencia en el catálogo de nomenclatura utilizado.
- Durante 2026 se incorporaron 100 estaciones nuevas, por lo que su disponibilidad de información histórica debe considerarse al construir las variables de los modelos.

Los datos originales y procesados no se incluyen en el repositorio debido a su tamaño. Se generan localmente mediante el proceso de ingesta.

## 3. Estructura del proyecto

El repositorio se organiza de la siguiente manera:

```text
TC5061-Proyecto-Equipo52/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── docs/
│
├── equipo52_module/
│   ├── __init__.py
│   ├── dataset.py
│   ├── features.py
│   └── modeling/
│       ├── train_p1.py
│       ├── train_p2.py
│       └── train_p3.py
│
├── models/
│
├── notebooks/
│
├── references/
│   └── manifiesto_datos.json
│
├── reports/
│   └── figures/
│
├── tests/
│   ├── conftest.py
│   ├── unitarias/
│   │   ├── test_dataset.py
│   │   ├── test_features_p2.py
│   │   ├── test_train_p2.py
│   │   ├── test_train_p1.py
│   │   └── test_features_p3.py
│   │
│   ├── reproducibilidad/
│   │   └── test_reproducibilidad.py
│   │
│   ├── seguridad/
│   │   └── test_seguridad.py
│   │
│   └── datos/
│       └── test_deepchecks.py
│
├── .gitignore
├── mkdocs.yml
├── pyproject.toml
├── README.md
└── requirements.txt
```

### 3.1 Organización de carpetas

| Carpeta | Propósito |
|---|---|
| `data/raw/` | Datos originales obtenidos de la fuente. |
| `data/processed/` | Datos transformados y preparados para el modelado. |
| `docs/` | Documentación del proyecto. |
| `equipo52_module/` | Código fuente del proyecto. |
| `equipo52_module/modeling/` | Scripts de entrenamiento de P1, P2 y P3. |
| `models/` | Espacio destinado a los modelos generados localmente. |
| `notebooks/` | Análisis exploratorio y experimentación. |
| `references/` | Manifiesto y referencias de los datos utilizados. |
| `reports/figures/` | Gráficas y resultados visuales. |
| `tests/` | Pruebas automatizadas y validaciones del proyecto. |

Los datos, modelos y artefactos generados localmente no se incorporan al control de versiones cuando están excluidos mediante `.gitignore`.

## 4. Tecnologías utilizadas

| Tecnología | Uso dentro del proyecto |
|---|---|
| Python 3.12 | Lenguaje de programación. |
| Pandas | Lectura, transformación y análisis de datos. |
| NumPy | Operaciones numéricas. |
| Scikit-learn | Desarrollo y evaluación de modelos. |
| PyArrow | Lectura y escritura de archivos Parquet. |
| Matplotlib | Generación de gráficas. |
| MLflow 3.17.2 | Seguimiento de experimentos y registro de modelos. |
| Pytest | Ejecución de pruebas automatizadas. |
| Ruff | Revisión y organización del código. |
| Python-dotenv | Manejo de variables de entorno. |
| Typer | Interfaz de comandos. |
| Loguru | Registro de eventos y mensajes. |

## 5. Instalación y configuración del entorno

### 5.1 Requisitos previos

- Python 3.12.
- Git.
- Acceso al repositorio de GitHub.
- Un entorno virtual de Python.

### 5.2 Clonar el repositorio

```bash
git clone https://github.com/DanniHL/TC5061-Proyecto-Equipo52.git
```

Ingresar a la carpeta:

```bash
cd TC5061-Proyecto-Equipo52
```

### 5.3 Crear y activar el entorno virtual

En Windows PowerShell:

```powershell
python -m venv .venv312
```

Activar el entorno:

```powershell
.\.venv312\Scripts\Activate.ps1
```

En macOS o Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 5.4 Instalar las dependencias

Actualizar pip:

```bash
python -m pip install --upgrade pip
```

Instalar las dependencias del proyecto:

```bash
python -m pip install -r requirements.txt
```

El archivo `requirements.txt` contiene las versiones de las bibliotecas utilizadas, incluyendo MLflow 3.17.2 y Pytest.

### 5.5 Verificar la instalación

```bash
python --version
python -m pip check
```

Consultar las versiones principales:

```bash
python -m pip show mlflow pytest pandas scikit-learn
```

## 6. Ingesta y preparación de datos

La ingesta se realiza mediante el módulo `dataset.py`, encargado de obtener los archivos mensuales, normalizar su estructura y preparar los datos para su utilización en las tres preguntas del proyecto.

### 6.1 Ejecutar la ingesta

Desde la raíz del repositorio:

```bash
python equipo52_module/dataset.py
```

El proceso obtiene los archivos históricos, realiza las transformaciones correspondientes y genera los datos preparados para los modelos.

### 6.2 Resultados de la ingesta

En la ejecución documentada se procesaron:

- 142 archivos mensuales.
- 39,078,984 registros de viajes.
- Periodo de diciembre de 2014 a septiembre de 2026.

Los datos se almacenan localmente y se genera un manifiesto para identificar la versión utilizada.

### 6.3 Manifiesto de datos

El archivo `references/manifiesto_datos.json` permite documentar la información correspondiente al conjunto de datos utilizado, incluyendo su versión, fuente, cantidad de registros y huella digital SHA256.

La huella permite comprobar si los archivos utilizados corresponden a la misma versión de datos entre diferentes ejecuciones.

El manifiesto también se utiliza como referencia para registrar la procedencia de los datos en los experimentos de MLflow.

## 7. Pregunta P1: Pronóstico de demanda diaria

### 7.1 Objetivo

Analizar el comportamiento de la demanda de viajes por estación y desarrollar un modelo que permita estimar la cantidad de viajes diarios.

La variable objetivo corresponde al número de viajes registrados por estación y día, considerando la estación como punto de origen.

### 7.2 Preparación de los datos

El proceso de preparación contempla:

1. Lectura de los datos históricos de viajes.
2. Selección de las columnas necesarias para el análisis.
3. Aplicación del periodo de estudio y exclusión de las estaciones definidas.
4. Conversión de las fechas al formato correspondiente.
5. Agregación de viajes por estación y día.
6. Construcción de calendarios diarios por estación.
7. Generación de variables históricas y de calendario.
8. Incorporación de variables cíclicas para representar el comportamiento temporal.
9. Separación cronológica de los conjuntos de entrenamiento y prueba.

La preparación de P1 conserva los datos originales de ingesta y realiza sus transformaciones únicamente para el análisis y modelado de esta pregunta.

### 7.3 Variables del modelo

Se utilizan variables relacionadas con:

- El calendario: día de la semana y otras características temporales.
- El comportamiento histórico de la demanda.
- La demanda del día anterior.
- La demanda de la misma estación en periodos anteriores.
- Promedios históricos calculados con información previa.
- Variables cíclicas para representar patrones temporales.

Las variables históricas se construyen utilizando desplazamientos temporales para evitar incorporar información futura al entrenamiento.

### 7.4 Modelos y evaluación

Se utiliza un modelo de persistencia como referencia inicial y se compara contra un modelo de regresión lineal.

La evaluación se realiza mediante las siguientes métricas:

| Métrica | Descripción |
|---|---|
| MAE | Error absoluto medio entre los valores reales y las predicciones. |
| RMSE | Raíz del error cuadrático medio, que penaliza en mayor medida los errores grandes. |

Los resultados registrados para la comparación inicial fueron:

| Modelo | MAE | RMSE |
|---|---:|---:|
| Persistencia | 9.2290 | 16.0120 |
| Regresión lineal | Se registra en MLflow. | Se registra en MLflow. |

La selección del modelo considera la reducción del error y su comportamiento en los datos de prueba.

### 7.5 Ejecutar el entrenamiento

```bash
python equipo52_module/modeling/train_p1.py
```

El script realiza la preparación de los datos, ejecuta el entrenamiento y registra los resultados del experimento en MLflow.

Las gráficas generadas durante el proceso se almacenan en `reports/figures/`, utilizando el prefijo `p1_` para identificar los resultados de esta pregunta.

### 7.6 Registro del modelo P1 en MLflow

El modelo seleccionado para P1 corresponde a una regresión lineal con variables cíclicas y escalamiento mediante `StandardScaler`.

| Elemento | Valor |
|---|---|
| Nombre del modelo registrado | `mibici-equipo52-p1-demanda` |
| Versión registrada | 1 |
| Alias | `champion` |
| Corrida de origen | `Pipeline_Regresion_Ciclica_P1` |
| Run ID | `db35a9caee2b4d25bc6ea3faee674842` |
| MAE | 6.6464 |
| RMSE | 11.3658 |

El registro permite identificar el modelo seleccionado, consultar la corrida que lo originó y recuperar los resultados asociados a su entrenamiento.

## 8. Pregunta P2: Identificación de riesgo operativo

### 8.1 Objetivo

Identificar situaciones de riesgo relacionadas con la disponibilidad de bicicletas y anclajes en las estaciones de MiBici.

Debido a que el sistema no publica directamente la cantidad de bicicletas disponibles en cada estación para el periodo histórico, el análisis utiliza los movimientos de entrada y salida para estimar posibles situaciones de riesgo.

### 8.2 Definición del riesgo

Se consideran dos escenarios principales:

- **Sin bicicletas:** cuando las salidas superan a las llegadas por un margen igual o mayor a un tercio de la capacidad de la estación.
- **Sin anclajes libres:** cuando las llegadas superan a las salidas por un margen igual o mayor a un tercio de la capacidad de la estación.

El resto de los casos se considera como comportamiento normal.

La definición representa una estimación del riesgo a partir de los movimientos registrados, no una medición directa de la disponibilidad real de bicicletas o espacios.

### 8.3 Enfoques de modelado

Se consideran dos enfoques para analizar el comportamiento:

1. Modelo con información histórica completa.
2. Modelo con información del periodo Post-Covid, tomando como fecha de inicio el 9 de mayo de 2023.

El análisis considera el comportamiento de las estaciones y la información temporal para anticipar posibles situaciones de riesgo.

### 8.4 Registro de experimentos

Los experimentos de P2 se registran en el experimento compartido `mibici-equipo52`.

El seguimiento permite comparar las ejecuciones, sus parámetros y métricas, así como conservar los artefactos generados durante el entrenamiento.

### 8.5 Registro del modelo P2 en MLflow

El modelo de P2 se administra mediante el registro de modelos de MLflow, utilizando el nombre:

```text
mibici-equipo52-p2-riesgo
```

La selección del modelo considera la comparación entre los experimentos realizados y el desempeño obtenido en las métricas definidas para esta pregunta.

Los detalles de la versión seleccionada, su corrida de origen y sus métricas deben consultarse directamente en el registro de MLflow.

## 9. Pregunta P3: Agrupación de estaciones

### 9.1 Objetivo

Identificar grupos de estaciones que presenten patrones de comportamiento similares a partir de sus movimientos y características de uso.

El propósito es reconocer diferentes roles dentro de la red de estaciones de MiBici y analizar cómo se relacionan con la movilidad de los usuarios.

### 9.2 Preparación de los datos

Para el análisis se construyen características relacionadas con la actividad de las estaciones, incluyendo:

- Proporciones de actividad por hora.
- Patrones de uso por día de la semana.
- Comportamiento de llegadas y salidas.
- Balance de movimientos.
- Características relacionadas con la capacidad de las estaciones.

Se considera el tiempo de actividad de cada estación para evitar que aquellas incorporadas recientemente sean comparadas directamente con estaciones que cuentan con un historial mucho más amplio.

### 9.3 Criterios de preparación

Se consideran los siguientes criterios:

- Se analiza el periodo Post-Covid a partir del 9 de mayo de 2023.
- Se identifican y tratan los viajes redondos de duración corta, utilizando un umbral de 180 segundos.
- Se calculan características de actividad por estación.
- Se normalizan las variables de actividad considerando los meses activos y la capacidad.
- Se construyen perfiles horarios y semanales para representar los patrones de uso.

### 9.4 Modelado y experimentación

Los experimentos de P3 permiten comparar diferentes configuraciones de agrupación y evaluar la formación de grupos de estaciones con patrones similares.

Los resultados se registran en el experimento compartido `mibici-equipo52`, conservando los parámetros, métricas y artefactos correspondientes a cada ejecución.

### 9.5 Registro del modelo P3 en MLflow

El modelo de P3 se administra mediante el registro:

```text
mibici-equipo52-p3-grupos
```

La selección del modelo considera los resultados de los experimentos realizados y la interpretación de los grupos obtenidos.

La versión definitiva, la corrida de origen y las métricas deben consultarse en el registro de MLflow.

## 10. Seguimiento de experimentos con MLflow

### 10.1 Experimento compartido

Las tres preguntas utilizan el mismo experimento de MLflow:

```text
mibici-equipo52
```

Esto permite centralizar el seguimiento de las ejecuciones y comparar los resultados de los modelos desarrollados por el equipo.

Cada corrida registra información relacionada con:

- Parámetros utilizados durante el entrenamiento.
- Métricas de evaluación.
- Artefactos generados.
- Versión de los datos.
- Huella SHA256 del manifiesto de datos.
- Semilla utilizada cuando corresponde.
- Identificador del commit asociado a la ejecución.
- Etiquetas para identificar la pregunta y el equipo.

La identificación de la pregunta mediante etiquetas permite distinguir los experimentos de P1, P2 y P3 dentro del mismo proyecto.

### 10.2 Iniciar el servidor de MLflow

Desde la raíz del repositorio, ejecutar:

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

El servidor estará disponible en:

```text
http://127.0.0.1:5000
```

Para utilizar esta dirección como URI de seguimiento:

```bash
MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

En Windows PowerShell:

```powershell
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
```

Después de iniciar el servidor, se puede consultar la interfaz desde el navegador.

### 10.3 Registro y selección de modelos

El Model Registry permite conservar las versiones de los modelos seleccionados y consultar su procedencia.

Los modelos registrados para el proyecto son:

| Pregunta | Nombre del modelo |
|---|---|
| P1 | `mibici-equipo52-p1-demanda` |
| P2 | `mibici-equipo52-p2-riesgo` |
| P3 | `mibici-equipo52-p3-grupos` |

Para P1 se registró la versión 1 con el alias `champion`.

En P2 y P3, la versión seleccionada, el alias y los resultados específicos pueden consultarse en el Model Registry de MLflow.

## 11. Pruebas automatizadas y calidad del código

El proyecto incorpora pruebas automatizadas mediante Pytest para verificar el comportamiento de las funciones principales y detectar errores antes de integrar cambios o ejecutar los procesos completos de entrenamiento.

Las pruebas se organizan por objetivo para facilitar su mantenimiento y permitir que los integrantes del equipo validen los componentes que tienen asignados.

### 11.1 Organización de las pruebas

| Carpeta | Propósito |
|---|---|
| `tests/unitarias/` | Verificar funciones individuales y sus casos normales y límite. |
| `tests/reproducibilidad/` | Comprobar la consistencia de los resultados de entrenamiento. |
| `tests/seguridad/` | Incorporar verificaciones relacionadas con la seguridad del proyecto. |
| `tests/datos/` | Validar características y calidad de los conjuntos de datos. |
| `tests/conftest.py` | Centralizar fixtures compartidos entre las pruebas. |

### 11.2 Ejecutar las pruebas

Para ejecutar todas las pruebas:

```bash
python -m pytest -v
```

Para ejecutar únicamente las pruebas unitarias de P1:

```bash
python -m pytest -v tests/unitarias/test_train_p1.py
```

Para ejecutar una carpeta específica:

```bash
python -m pytest -v tests/unitarias/
```

La configuración de Pytest se encuentra en `pyproject.toml` y permite identificar automáticamente las pruebas ubicadas dentro de `tests/`.

### 11.3 Pruebas unitarias de P1

Las pruebas de P1 verifican funciones relacionadas con la agregación de viajes, la construcción de calendarios, la generación de variables históricas, la separación cronológica de los datos y el cálculo de métricas.

Entre los casos evaluados se encuentran:

- Conservación del total de viajes durante la agregación.
- Manejo de conjuntos de datos vacíos.
- Conservación de fechas sin registros.
- Cálculo correcto de variables históricas.
- Evitar el uso de información futura.
- Separación cronológica entre entrenamiento y prueba.
- Manejo de casos sin suficientes datos para generar el conjunto de prueba.
- Cálculo correcto de MAE y RMSE.

En la ejecución documentada de `test_train_p1.py` se obtuvieron los siguientes resultados:

```text
9 passed
```

Las nueve pruebas ejecutadas finalizaron correctamente.

### 11.4 Reproducibilidad

La reproducibilidad busca comprobar que las ejecuciones realizadas bajo las mismas condiciones produzcan resultados consistentes.

Para ello se considera:

- Utilizar la misma versión de los datos.
- Registrar la huella SHA256 del manifiesto.
- Conservar los parámetros de entrenamiento.
- Registrar el commit asociado a cada corrida.
- Utilizar la semilla 42 cuando se ejecutan operaciones estocásticas que admiten `random_state`.

La semilla se utiliza únicamente en los procesos donde tiene efecto, evitando incorporarla como parámetro en operaciones deterministas.

Las pruebas específicas de reproducibilidad se ejecutan desde:

```bash
python -m pytest -v tests/reproducibilidad/
```

### 11.5 Calidad de los datos con DeepChecks

Se contempla la utilización de DeepChecks para complementar las pruebas unitarias mediante validaciones de calidad de datos.

Las verificaciones consideradas incluyen:

- Tipos de datos.
- Valores faltantes.
- Registros duplicados.
- Diferencias o cambios en la distribución de los datos entre entrenamiento y prueba.

Estas validaciones permiten identificar problemas que podrían afectar el entrenamiento o el desempeño de los modelos.

La ejecución y los resultados de estas pruebas deben documentarse una vez que estén implementados y validados en el entorno del proyecto.

### 11.6 Seguridad

Se aplicaron dos herramientas de análisis y prácticas para evitar exponer información sensible. Cada hallazgo se corrigió en un commit propio y se verificó volviendo a ejecutar la herramienta.

#### Ejecución

Las herramientas se instalan con `requirements.txt` (`bandit==1.9.4` y `pip-audit==2.10.1`). Para correr ambas:

```bash
make seguridad
```

La regla ejecuta Bandit sobre el código del módulo (solo hallazgos de severidad media o alta) y pip-audit sobre las versiones de `requirements.txt`. pip-audit requiere versiones exactas y no admite la línea `-e .`, por lo que la regla audita una copia del archivo sin esa línea. Los comandos equivalentes son:

```bash
python -m bandit -r equipo52_module -ll
grep -v '^-e' requirements.txt > /tmp/req-audit.txt
python -m pip_audit -r /tmp/req-audit.txt --no-deps --disable-pip
```

Las pruebas de seguridad del repositorio se ejecutan con:

```bash
python -m pytest -v tests/seguridad/
```

#### Resultados

| Herramienta | Antes | Corrección | Después |
|---|---|---|---|
| Bandit | 2 hallazgos de severidad media (B310) en `dataset.py`: las ligas de descarga se leen de la página de MiBici y se abrían sin validar su esquema ni su dominio, y sin tiempo límite. 13 hallazgos de severidad baja. | Función `validar_liga()`: solo acepta ligas `https` de `mibici.net` (las 143 ligas del manifiesto lo son) y las descargas tienen un tiempo límite de 60 s. | 0 hallazgos de severidad media o alta. Los 13 de severidad baja (llamadas fijas a `git rev-parse` y `assert` en `train_p1.py`) se documentan como riesgo aceptado. |
| pip-audit | `mlflow 3.17.0` con la vulnerabilidad CVE-2026-53995. | Actualización a `mlflow==3.17.2`. Además, `pytest` se fijó en `9.2.0`, ya que pip-audit requiere versiones exactas. | Sin vulnerabilidades conocidas. |

![Bandit antes](docs/evidencias/10b_bandit_antes_resumen.png)
![Bandit después](docs/evidencias/11_bandit_despues_sin_medium.png)
![pip-audit antes](docs/evidencias/12_pip_audit_antes_mlflow_cve.png)
![pip-audit después](docs/evidencias/13_pip_audit_despues_sin_vulnerabilidades.png)

#### Datos sensibles y variables de entorno

- `.gitignore` excluye los datos, `mlflow.db`, `mlruns/`, `mlartifacts/`, los entornos virtuales, `.env` y sus variantes (`.env.*`), llaves (`*.pem`, `*.key`) y archivos de credenciales (`credentials*.json`).
- `.env.example` documenta la variable `MLFLOW_TRACKING_URI` sin valores secretos. Para usarlo: `cp .env.example .env`.
- La ingesta no lee las columnas con datos personales (`Usuario_Id`, `Genero` y `Año_de_nacimiento`).

![Archivos ignorados por Git](docs/evidencias/09_gitignore_archivos_sensibles.png)

`tests/seguridad/test_seguridad.py` convierte estas revisiones en pruebas automáticas sobre los archivos que versiona Git: ausencia de archivos de credenciales y de patrones de secretos, reglas del `.gitignore`, `.env.example` sin valores, lectura de `MLFLOW_TRACKING_URI` desde el entorno, ausencia de columnas personales y de datos en el repositorio.

Resultado de la ejecución documentada:

```text
21 passed
```

![Pruebas de seguridad](docs/evidencias/14_pytest_seguridad.png)

## 12. Control de versiones y colaboración

El desarrollo se realiza mediante Git y GitHub, utilizando ramas independientes para trabajar en los componentes del proyecto.

La integración de cambios se realiza mediante Pull Requests para facilitar la revisión por parte de los demás integrantes.

### 12.1 Flujo de trabajo

1. Actualizar la referencia local de `main`.
2. Crear o cambiar a la rama de trabajo correspondiente.
3. Realizar los cambios en el código o documentación.
4. Ejecutar las pruebas correspondientes.
5. Revisar los archivos modificados.
6. Crear un commit con un mensaje descriptivo.
7. Enviar los cambios a GitHub.
8. Crear un Pull Request para su revisión e integración.

### 12.2 Comandos de Git

Actualizar la información del repositorio remoto:

```bash
git fetch origin
```

Cambiar a `main`:

```bash
git checkout main
```

Actualizar la rama:

```bash
git pull origin main
```

Crear una rama de trabajo:

```bash
git checkout -b feature/nombre-de-la-funcionalidad
```

Revisar los cambios:

```bash
git status
```

Agregar archivos:

```bash
git add .
```

Crear un commit:

```bash
git commit -m "test: agregar pruebas automatizadas"
```

Enviar la rama al repositorio remoto:

```bash
git push -u origin feature/nombre-de-la-funcionalidad
```

Una vez revisado y aprobado el Pull Request, los cambios pueden integrarse a `main`.

### 12.3 Convención de commits

Se recomienda utilizar mensajes descriptivos siguiendo una estructura similar a Conventional Commits:

| Prefijo | Uso |
|---|---|
| `feat:` | Nueva funcionalidad. |
| `fix:` | Corrección de errores. |
| `test:` | Incorporación o modificación de pruebas. |
| `docs:` | Cambios en la documentación. |
| `refactor:` | Reorganización del código sin modificar su funcionalidad. |
| `chore:` | Cambios de mantenimiento y configuración. |

## 13. Consideraciones para la ejecución del proyecto

- Ejecutar los comandos desde la raíz del repositorio.
- Activar el entorno virtual antes de instalar dependencias o ejecutar los scripts.
- Generar los datos mediante el proceso de ingesta antes de entrenar los modelos, cuando no se encuentren disponibles localmente.
- Mantener actualizado el manifiesto de datos cuando cambie la versión utilizada.
- Iniciar el servidor de MLflow antes de ejecutar los scripts de entrenamiento que registran sus resultados en dicho servidor.
- No incorporar al repositorio los archivos de datos locales, credenciales ni artefactos que estén excluidos mediante `.gitignore`.
- Ejecutar las pruebas correspondientes antes de integrar cambios mediante un Pull Request.

## 14. Resultados y entregables

El proyecto integra los siguientes componentes:

1. Proceso de ingesta y preparación de los datos históricos de MiBici.
2. Desarrollo de modelos para las preguntas P1, P2 y P3.
3. Seguimiento de experimentos mediante MLflow.
4. Comparación y evaluación de modelos.
5. Registro de modelos seleccionados mediante MLflow Model Registry.
6. Organización de pruebas automatizadas mediante Pytest.
7. Documentación del proceso de desarrollo y colaboración mediante GitHub.

El objetivo de esta estructura es mantener un flujo de trabajo reproducible, organizado y trazable, en el que sea posible identificar qué datos, código y parámetros se utilizaron para generar cada resultado.

---

**Equipo 52 · TC5061 · Maestría en Inteligencia Artificial**