"""Build a v3-only Colab notebook with the project's canonical label and window logic."""

import ast
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def definitions(path, names):
    source = path.read_text(encoding="utf-8")
    selected = {node.name: ast.get_source_segment(source, node) for node in ast.parse(source).body
                if getattr(node, "name", None) in names}
    return "\n\n".join(selected[name] for name in names)


def build():
    batch = HERE.parent
    fusion = batch.parents[1] / "analysis_sessions/dataset_fusion"
    base = json.loads((batch / "exp05/EyeHeartCognitiveLoad.ipynb").read_text(encoding="utf-8"))
    canonical = """import itertools
from bisect import bisect_right
from collections import defaultdict
"""
    behavior = fusion / "v3_behavior.py"
    source = behavior.read_text(encoding="utf-8")
    canonical += "\n".join(ast.get_source_segment(source, node) for node in ast.parse(source).body
                            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                            and node.targets[0].id in {"MIN_REFERENCE", "POLICY", "LEGACY_POLICY"})
    canonical += "\n\n" + definitions(behavior, ["make_reference", "reference_for", "label_episode", "WindowIndex"])
    canonical += "\n\n" + definitions(fusion / "build_dataset.py", ["epoch_ms"])
    history = batch / "exp06/history.py"
    text = history.read_text(encoding="utf-8")
    canonical += "\n\n" + "\n".join(ast.get_source_segment(text, node) for node in ast.parse(text).body
                                     if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                                     and node.targets[0].id in {"HEAD", "SENSORS"})
    canonical += "\n\n" + definitions(history, ["history_streams", "prior_reference"])
    cells = base["cells"]
    cells[0]["source"] = ["# Exp07 · deteksi High dan evaluasi kelas Medium\n",
                          "Run all pada CPU. Simpan paket v3 dan kedua CSV v2 15 detik (all_data.csv dan good_acceptable.csv) versi repo di bawah Dataset. "
                          "Eksperimen mengevaluasi LOPO dalam Train dan validation resmi."]
    cells[1]["source"] = "".join(cells[1]["source"]).replace("exp05/outputs", "exp07/outputs").splitlines(True)
    cells[4]["source"] = ["## Rekonstruksi referensi per fold dan model pembanding"]
    cells[5]["source"] = (canonical + "\n\n" + (HERE / "experiment.py").read_text(encoding="utf-8")).splitlines(True)
    cells[6]["source"] = ["## Jalankan v3 dan simpan hasil"]
    cells[7]["source"] = ["archive = run(DATASET_ROOT, OUTPUT_ROOT)"]
    (HERE / "EyeHeartCognitiveLoad.ipynb").write_text(json.dumps(base, ensure_ascii=False, indent=1) + "\n",
                                                      encoding="utf-8")


if __name__ == "__main__":
    build()
