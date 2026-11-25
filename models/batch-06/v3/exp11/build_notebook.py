"""Embed the previous cohort, window mapping and evaluation functions for Colab."""

import ast
import json
from pathlib import Path


def function(source, name):
    node = next(n for n in ast.parse(source).body if getattr(n, "name", "") == name)
    return ast.get_source_segment(source, node)


def build():
    here = Path(__file__).resolve().parent
    v3 = here.parent
    name = "EyeHeartCognitiveLoad.ipynb"
    nb = json.loads((v3 / "exp09" / name).read_text(encoding="utf-8"))
    cells = nb["cells"]
    common = "".join(cells[5]["source"])
    stop = next(n.end_lineno for n in ast.parse(common).body if getattr(n, "name", "") == "load_cohort")
    cells[0]["source"] = ["# Exp11 · median fitur versus rata-rata skor SVM per window\n",
                          "Jalankan semua sel pada CPU. Gunakan paket v3, kedua CSV v2 seperti exp09 dan "
                          "v4_self_reported_rating/episode_ratings.csv untuk pemeriksaan setelah fitting."]
    cells[1]["source"] = "".join(cells[1]["source"]).replace("exp09/outputs", "v3/exp11/outputs").splitlines(True)
    cells[4]["source"] = ["## Dua unit fitting dengan transformasi dan kernel yang sama"]
    parts = ["".join(common.splitlines(True)[:stop]), function(common, "high_score"),
             function(common, "hinge"), function(common, "evaluate"),
             'RATIO = "rmssd_baseline_ratio"',
             function((v3 / "exp10/experiment.py").read_text(encoding="utf-8"), "window_pool"),
             function((v3 / "exp10/rating_check.py").read_text(encoding="utf-8"), "compare"),
             (here / "experiment.py").read_text(encoding="utf-8")]
    cells[5]["source"] = "\n\n".join(parts).splitlines(True)
    cells[6]["source"] = ["## Evaluasi v3, periksa rating, lalu simpan ZIP"]
    (here / name).write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    build()
