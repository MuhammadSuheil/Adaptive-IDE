"""Run the standalone notebook locally and retain its textual cell outputs."""

import ast
import contextlib
import io
import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "models/batch-05/exp01/EyeHeartCognitiveLoad.ipynb"


class Capture(io.StringIO):
    def write(self, value):
        sys.__stdout__.write(value)
        sys.__stdout__.flush()
        return super().write(value)


def main():
    notebook = nbformat.read(PATH, as_version=4)
    nbformat.validate(notebook)
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
                RESUME_DIR=None,
            )
            continue
        count += 1
        capture = Capture()
        with contextlib.redirect_stdout(capture):
            # Execute this trusted repository notebook, not an external attachment.
            exec(compile(cell.source, str(PATH), "exec"), namespace)  # noqa: S102  # nosec B102
        cell.execution_count = count
        cell.outputs = []
        if capture.getvalue():
            cell.outputs.append(
                nbformat.v4.new_output("stream", name="stdout", text=capture.getvalue())
            )
    nbformat.validate(notebook)
    nbformat.write(notebook, PATH)
    print("COMPLETED_RUN=" + str(namespace["RUN_DIR"]), flush=True)


if __name__ == "__main__":
    main()
