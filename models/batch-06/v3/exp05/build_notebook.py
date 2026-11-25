"""Package the paired experiment, including the same input checks as exp03/04."""

import ast
import json
from pathlib import Path


def build():
    here = Path(__file__).resolve().parent
    shared = (here.parent / "relabeling/experiment.py").read_text(encoding="utf-8")
    inputs = shared.split("# %% Input CSV\n")[1].split("# %% Model")[0].strip()
    helpers = "\n\n".join(ast.get_source_segment(shared, node) for node in ast.parse(shared).body
                           if getattr(node, "name", "") in {"scores", "save_json"})
    source = (here / "episodes.py").read_text(encoding="utf-8")
    cells = []
    for kind, content in [
        ("markdown", "# Exp05 · dua target pada tingkat episode\n\nJalankan semua sel pada runtime CPU. "
         "V3 memakai Train dan validation resmi. V4 menahan seluruh episode satu peserta bergantian. "
         "Kedua hasil masuk ke satu ZIP. Kedua subfolder paket harus berada langsung di Dataset."),
        ("code", """from pathlib import Path
from google.colab import drive

drive.mount('/content/drive')
PROJECT_ROOT = Path('/content/drive/MyDrive/Penelitian-Adaptive-IDE')
DATASET_ROOT = PROJECT_ROOT / 'Dataset'
OUTPUT_ROOT = PROJECT_ROOT / 'models/batch-06/exp05/outputs'"""),
        ("markdown", "## Validasi paket, fitur sensor dan metrik"), ("code", inputs + "\n\n" + helpers),
        ("markdown", "## Prediksi dari window pertama dan median seluruh episode"), ("code", source),
        ("markdown", "## Jalankan kedua kasus dan simpan ZIP"), ("code", "archive = run_episodes(DATASET_ROOT, OUTPUT_ROOT)")
    ]:
        cell = {"cell_type": kind, "metadata": {}, "source": content.strip().splitlines(keepends=True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 4,
          "metadata": {"kernelspec": {"name": "python3", "language": "python", "display_name": "Python 3"}}}
    (here / "EyeHeartCognitiveLoad.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n",
                                                      encoding="utf-8")


if __name__ == "__main__":
    build()
