"""Package the loss diagnostic with the same input checks and fold rules as exp08."""

import ast
import json
from pathlib import Path


def build():
    here = Path(__file__).resolve().parent
    name = "EyeHeartCognitiveLoad.ipynb"
    nb = json.loads((here.parent / "exp08" / name).read_text(encoding="utf-8"))
    cells = nb["cells"]
    cells[0]["source"] = ["# Exp09 · loss dan validitas proxy perilaku v3\n",
                          "Jalankan semua sel pada CPU dengan paket input exp08. Grafik tampil di notebook "
                          "dan tersimpan bersama nilai tiap titik di dalam ZIP."]
    cells[1]["source"] = "".join(cells[1]["source"]).replace("exp08/outputs", "exp09/outputs").splitlines(True)
    embedded = "".join(cells[5]["source"])
    end = next(n.end_lineno for n in ast.parse(embedded).body if getattr(n, "name", "") == "load_cohort")
    cells[4]["source"] = ["## Kurva solver dan pemisahan High dalam peserta yang sama"]
    cells[5]["source"] = ("".join(embedded.splitlines(True)[:end]) + "\n\n" +
                          (here / "experiment.py").read_text(encoding="utf-8")).splitlines(True)
    cells[6]["source"] = ["## Tampilkan kurva dan simpan ZIP"]
    (here / name).write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    build()
