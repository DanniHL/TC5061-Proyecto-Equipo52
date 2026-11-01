#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = equipo52_proyecto_mibici
PYTHON_VERSION = 3.12
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	uv pip install -r requirements.txt
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format





## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	uv venv --python $(PYTHON_VERSION)
	@echo ">>> New uv virtual environment created. Activate with:"
	@echo ">>> Windows: .\\\\.venv\\\\Scripts\\\\activate"
	@echo ">>> Unix/macOS: source ./.venv/bin/activate"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Make dataset
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) equipo52_module/dataset.py


## Corre todas las pruebas: unitarias, reproducibilidad, seguridad y datos reales
.PHONY: test
test:
	$(PYTHON_INTERPRETER) -m pytest tests


## Genera las muestras reales y corre la validación de datos con DeepChecks
.PHONY: test-datos
test-datos: .venv-calidad/bin/python
	$(PYTHON_INTERPRETER) -m pytest tests/datos -v


## Crea .venv-calidad: DeepChecks 0.19 no funciona con el NumPy 2 ni con el
## scikit-learn 1.9 del entorno principal, así que vive en un entorno aparte
#  (requirements-calidad.txt). Solo se crea si no existe; para reinstalarlo se
#  borra la carpeta y se corre de nuevo.
.venv-calidad/bin/python:
	python3.12 -m venv .venv-calidad
	.venv-calidad/bin/pip install -r requirements-calidad.txt

.PHONY: entorno-calidad
entorno-calidad: .venv-calidad/bin/python


## Corre bandit sobre el código del módulo y pip-audit sobre requirements.txt
.PHONY: seguridad
seguridad:
	bandit -r equipo52_module -ll
	pip-audit -r requirements.txt --no-deps


## Corre todo lo anterior con un solo comando
.PHONY: verificar
verificar: entorno-calidad test seguridad


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
