"""Execute the standalone Stage 3 notebook locally; keep its textual outputs."""

import ast
import base64
import contextlib
import io
import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "models/batch-05/exp01/EyeHeartCognitiveLoad.ipynb"


class Capture(io.StringIO):
    def write(self, text):
        sys.__stdout__.write(text)
        sys.__stdout__.flush()
        return super().write(text)


def attach_figures(notebook, run_dir):
    target = next(c for c in notebook.cells if c.id == "stage3-figures")
    target.outputs = [
        nbformat.v4.new_output(
            "display_data",
            data={
                "image/png": base64.b64encode(path.read_bytes()).decode("ascii"),
                "text/plain": path.name,
            },
            metadata={},
        )
        for path in sorted((Path(run_dir) / "figures").glob("*.png"))
    ]


def main():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    if len(sys.argv) > 1 and sys.argv[1] == "--embed-only":
        attach_figures(notebook, Path(sys.argv[2]))
        nbformat.validate(notebook)
        nbformat.write(notebook, NOTEBOOK)
        return
    namespace = {"__name__": "__main__"}
    count = 0
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        ast.parse(cell.source)
        if cell.id == "configuration":
            namespace.update(
                PROJECT_ROOT=ROOT,
                DATASET_ROOT=ROOT
                / "analysis_sessions/dataset_fusion/output/v2_session_phase",
                OUTPUT_ROOT=ROOT / "models/batch-05/outputs/exp01",
                RESUME_DIR=Path(sys.argv[1]) if len(sys.argv) > 1 else None,
            )
            continue
        count += 1
        capture = Capture()
        with contextlib.redirect_stdout(capture):
            # The notebook is a trusted repository artifact.
            exec(compile(cell.source, str(NOTEBOOK), "exec"), namespace)  # noqa: S102  # nosec B102
        cell.execution_count = count
        cell.outputs = []
        if capture.getvalue():
            cell.outputs.append(
                nbformat.v4.new_output("stream", name="stdout", text=capture.getvalue())
            )
    attach_figures(notebook, namespace["RUN_DIR"])
    nbformat.validate(notebook)
    nbformat.write(notebook, NOTEBOOK)
    print("COMPLETED_RUN=" + str(namespace["RUN_DIR"]), flush=True)


if __name__ == "__main__":
    main()
