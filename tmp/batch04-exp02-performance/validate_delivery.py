"""Audit the exp02 delivery without training research models or changing CSVs."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import nbformat
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tests.test_batch04_current import definitions

for experiment in ("exp01", "exp03", "exp04"):
    relative = f"models/batch-04/{experiment}/EyeHeartCognitiveLoad.ipynb"
    original = subprocess.run(
        ["git", "show", f"HEAD:{relative}"], cwd=ROOT, capture_output=True, check=True
    ).stdout
    assert (
        hashlib.sha256(original.replace(b"\r\n", b"\n")).digest()
        == hashlib.sha256(
            (ROOT / relative).read_bytes().replace(b"\r\n", b"\n")
        ).digest()
    )
    print(experiment, "unchanged")

path = ROOT / "models/batch-04/exp02/EyeHeartCognitiveLoad.ipynb"
notebook = nbformat.read(path, as_version=4)
nbformat.validate(notebook)
for cell in notebook.cells:
    if cell.cell_type == "code":
        compile(cell.source, str(path), "exec")
        assert cell.execution_count is None and not cell.outputs
ns, old = definitions("exp02"), definitions("exp01")
assert ns["BUDGET"] == old["BUDGET"]
assert ns["FINAL_SEEDS"] == old["FINAL_SEEDS"]
views, manifest, _, hashes = ns["read_datasets"](
    ROOT / "analysis_sessions/dataset_fusion/output/v2_session_phase"
)
assert (
    ns["digest"](manifest)
    == "da5549e9d7fd62ce0f6cb0375f2dd486accace366f84b125a3f77b1c0b5709c7"
)
for window, frame in views.items():
    for group in ("core", "core_head"):
        for history in (15, 30, 60):
            for representation in ("current", "history"):
                recipe = ns["base_recipe"]("logistic")
                recipe.update(
                    group=group,
                    history=history,
                    representation=representation,
                    normalization="causal",
                    derived=True,
                )
                expected = old["SensorTransform"](recipe)._matrix(frame.copy())
                actual = ns["SensorTransform"](recipe)._matrix(frame)
                np.testing.assert_array_equal(expected, actual)
    recipe = ns["base_recipe"]("logistic")
    recipe.update(normalization="causal", history=60, representation="history")
    old_pre, new_pre = old["SensorTransform"](recipe), ns["SensorTransform"](recipe)
    results = []
    for title, pre in (("old", old_pre), ("optimized", new_pre)):
        pre._matrix(frame)
        start = perf_counter()
        for _ in range(5):
            pre._matrix(frame)
        results.append((title, round((perf_counter() - start) / 5, 5)))
    print(
        window,
        "all causal representations equivalent; repeated matrix seconds:",
        results,
        "cache bytes:",
        frame.attrs["_cl_features"].bytes_used,
        flush=True,
    )
    source = (
        ROOT
        / f"analysis_sessions/dataset_fusion/output/v2_session_phase/window_{window}s_step_5s/good_acceptable.csv"
    )
    assert hashlib.sha256(source.read_bytes()).hexdigest() == hashes[str(window)]
print(
    "Notebook, split, actual features, and dataset hashes verified; no research model trained."
)
