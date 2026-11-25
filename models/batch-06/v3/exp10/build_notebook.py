"""Embed the v4 preprocessing/weighting and v3 fold definitions in one Colab notebook."""

import ast
import json
from pathlib import Path


def build():
    here = Path(__file__).resolve().parent
    v3 = here.parent
    name = "EyeHeartCognitiveLoad.ipynb"
    nb = json.loads((v3 / "exp09" / name).read_text(encoding="utf-8"))
    cells = nb["cells"]
    cells[0]["source"] = ["# Exp10 · transfer resep biner v4 ke proxy v3\n",
                          "Gunakan paket v3 dan kedua CSV v2 15 detik seperti exp09. Jalankan semua sel pada CPU."]
    cells[1]["source"] = "".join(cells[1]["source"]).replace("exp09/outputs", "v3/exp10/outputs").splitlines(True)
    common = "".join(cells[5]["source"])
    stop = next(n.end_lineno for n in ast.parse(common).body if getattr(n, "name", "") == "high_score")
    team = v3.parent / "exp03_v4_binary_overload/EyeHeartCognitiveLoad_v4_binary.ipynb"
    source = "".join(json.loads(team.read_text(encoding="utf-8"))["cells"][3]["source"])
    weight_fn = next(n for n in ast.parse(source).body if getattr(n, "name", "") == "episode_balanced_weights")
    cells[4]["source"] = ["## Fold v3 dan bobot window dari eksperimen biner v4"]
    cells[5]["source"] = ("".join(common.splitlines(True)[:stop]) + "\n\n" +
                          ast.get_source_segment(source, weight_fn) + "\n\n" +
                          (here / "experiment.py").read_text(encoding="utf-8")).splitlines(True)
    cells[6]["source"] = ["## Jalankan perbandingan berpasangan dan ekspor ZIP"]
    (here / name).write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    build()
