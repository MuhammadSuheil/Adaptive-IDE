"""Package the exp11 cohort and the paired normalization experiment for Colab."""

import json
from pathlib import Path


def build():
    here = Path(__file__).resolve().parent
    name = "EyeHeartCognitiveLoad.ipynb"
    nb = json.loads((here.parent / "exp11" / name).read_text(encoding="utf-8"))
    cells = nb["cells"]
    shared = "".join(cells[5]["source"])
    cohort = shared[:shared.index('RATIO = "rmssd_baseline_ratio"')]
    ratings = shared[shared.index("def compare(predictions, ratings)"):shared.index("def compare_fold(")]
    cells[0]["source"] = ["# Exp13 · variasi sensor sebelum soal\n",
                          "Bandingkan median episode exp11 dengan normalisasi variasi peserta. "
                          "Gunakan paket v3, dua CSV v2 15 detik dan rating v4 seperti exp11. Jalankan semua sel pada CPU."]
    cells[1]["source"] = "".join(cells[1]["source"]).replace("exp11/outputs", "exp13/outputs").splitlines(True)
    cells[4]["source"] = ["## Perkirakan variasi dari window sebelum soal dan uji kedua model per peserta"]
    cells[5]["source"] = (cohort + ratings + (here / "experiment.py").read_text(encoding="utf-8")).splitlines(True)
    cells[6]["source"] = ["## Evaluasi, audit ketersediaan riwayat, dan simpan ZIP"]
    (here / name).write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    build()
