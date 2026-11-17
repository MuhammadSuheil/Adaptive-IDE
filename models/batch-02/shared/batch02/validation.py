"""Technical checks only: no model fit, backward, or checkpoint download."""

import ast
from pathlib import Path

from .artifacts import code_hash, environment
from .estimators import make_estimator
from .registry import Recipe, model_bank, recipes_for
from .runtime import ensure_environment


def validate_notebook(path):
    import nbformat

    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    cells = executed = 0
    for cell in notebook.cells:
        if cell.cell_type == "code":
            ast.parse(cell.source, filename=str(path))
            if any(output.output_type == "error" for output in cell.outputs):
                raise ValueError(f"Notebook contains an error traceback: {path}")
            cells += 1
            executed += cell.execution_count is not None
    return {"code_cells": cells, "executed_code_cells": executed}


def validate_package(batch):
    batch = Path(batch)
    notebooks = sorted(batch.glob("exp*/EyeHeartCognitiveLoad.ipynb"))
    if len(notebooks) != 15:
        raise ValueError("Paket memerlukan notebook exp00 sampai exp14.")
    cells = executed = 0
    for path in notebooks:
        checked = validate_notebook(path)
        cells += checked["code_cells"]
        executed += checked["executed_code_cells"]
    constructors = []
    for recipe in [Recipe("dummy"), *model_bank()]:
        model = make_estimator(recipe)
        constructors.append(
            {
                "recipe": recipe.key,
                "class": type(model).__name__,
                "parameters": model.get_params(),
            }
        )
    counts = {f"exp{i:02d}": len(recipes_for(f"exp{i:02d}")) for i in range(1, 14)}
    return {
        "status": "technical_validation_only",
        "training_performed": False,
        "notebooks": len(notebooks),
        "parsed_code_cells": cells,
        "executed_code_cells": executed,
        "recipe_counts": counts,
        "constructors": constructors,
        "environment": environment(),
        "code_hash": code_hash(),
        "dependency_check": ensure_environment(batch),
        "not_verified": [
            "real_classifier_fitting",
            "neural_training",
            "GPU_execution",
            "TabPFN_checkpoint_inference",
            "complete_nested_evaluation",
            "Colab_runtime",
        ],
    }
