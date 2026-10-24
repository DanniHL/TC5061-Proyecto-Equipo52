
import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[1]


def convertir_booleano(valor: str) -> bool:
    """Convierte una cadena de texto en un valor booleano."""
    valor = valor.lower()
    if valor in ("true", "1", "si"):
        return True
    if valor in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError("El valor debe ser true o false.")


def construir_comando(args: argparse.Namespace) -> list[str]:
    """Construye el comando correspondiente a P1, P2 o P3."""
    scripts = {
        "p1": "equipo52_module.modeling.train_p1",
        "p2": "equipo52_module.modeling.train_p2",
        "p3": "equipo52_module.modeling.train_p3",
    }

    comando = [sys.executable, "-m", scripts[args.pregunta]]

    if args.modelo:
        comando.extend(["--modelo", args.modelo])

    if args.pregunta == "p2":
        if args.periodo:
            comando.extend(["--periodo", args.periodo])
        if args.balanceado:
            comando.append("--balanceado")
        comando.extend(["--max-iter", str(args.max_iter)])
        comando.extend(["--learning-rate", str(args.learning_rate)])
        comando.extend(["--max-leaf-nodes", str(args.max_leaf_nodes)])

    elif args.pregunta == "p3":
        comando.extend(["--k", str(args.k)])
        comando.extend(["--min-cluster-size", str(args.min_cluster_size)])

        if args.solo_consolidadas:
            comando.append("--solo-consolidadas")
        if args.sin_redondos_cortos:
            comando.append("--sin-redondos-cortos")

    return comando


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ejecutor de los modelos del proyecto MiBici."
    )
    parser.add_argument("--pregunta", required=True, choices=["p1", "p2", "p3"])
    parser.add_argument("--modelo", default="")
    parser.add_argument("--periodo", default="")
    parser.add_argument("--balanceado", type=convertir_booleano, default=False)
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--min-cluster-size", type=int, default=15)
    parser.add_argument(
        "--solo-consolidadas", type=convertir_booleano, default=False
    )
    parser.add_argument(
        "--sin-redondos-cortos", type=convertir_booleano, default=False
    )
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--max-leaf-nodes", type=int, default=31)

    args = parser.parse_args()
    comando = construir_comando(args)

    # Conservar MLFLOW_RUN_ID para mantener la relación con MLflow Projects.
    subprocess.run(comando, cwd=PROJ_ROOT, check=True, env=os.environ.copy())


if __name__ == "__main__":
    main()
