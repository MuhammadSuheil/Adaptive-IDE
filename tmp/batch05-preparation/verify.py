"""Execute Stage 1 cells against local CSVs, without any classifier fitting."""

import ast
import json
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "models/batch-05/exp01/EyeHeartCognitiveLoad.ipynb"


def main():
    notebook = nbformat.read(PATH, as_version=4)
    nbformat.validate(notebook)
    namespace = {"__name__": "batch05_verification"}
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        ast.parse(cell.source)
        if cell.id == "configuration":
            continue
        if cell.id == "execution":
            namespace.update(
                DATASET_ROOT=ROOT
                / "analysis_sessions/dataset_fusion/output/v2_session_phase",
                OUTPUT_ROOT=ROOT / "models/batch-05/outputs/exp01",
            )
        # This notebook is a trusted local artifact, not an external attachment.
        exec(compile(cell.source, str(PATH), "exec"), namespace)  # noqa: S102  # nosec B102
    run_dir = namespace["RUN_DIR"]
    for path in run_dir.glob("*.json"):
        json.loads(path.read_text(encoding="utf-8"))
    print("VALIDATED_RUN=" + str(run_dir), flush=True)


if __name__ == "__main__":
    main()
