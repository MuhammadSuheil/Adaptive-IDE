"""Build a standalone notebook using the verified exp07 data and reference routines."""

import ast
import json
from pathlib import Path


def build():
    here = Path(__file__).resolve().parent
    name = "EyeHeartCognitiveLoad.ipynb"
    nb = json.loads((here.parent / "exp07" / name).read_text(encoding="utf-8"))
    cells = nb["cells"]
    cells[0]["source"] = ["# Exp08 · audit alarm High pada v3\n",
                          "Run all pada CPU dengan paket v3 dan kedua CSV v2 15 detik versi repo seperti exp07. "
                          "Gunakan prediksi per episode, audit referensi, dan hasil ambang di satu ZIP."]
    cells[1]["source"] = "".join(cells[1]["source"]).replace("exp07/outputs", "exp08/outputs").splitlines(True)
    shared = "".join(cells[5]["source"])
    boundary = next(n.lineno for n in ast.parse(shared).body if getattr(n, "name", "") == "classify")
    definitions = "".join(shared.splitlines(True)[:boundary - 1])
    cells[4]["source"] = ["## Referensi sensor, fitting, dan pemilihan ambang"]
    cells[5]["source"] = (definitions + (here / "experiment.py").read_text(encoding="utf-8")).splitlines(True)
    cells[6]["source"] = ["## Jalankan dan ekspor ZIP"]
    cells[7]["source"] = ["archive = run(DATASET_ROOT, OUTPUT_ROOT)"]
    (here / name).write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    build()
