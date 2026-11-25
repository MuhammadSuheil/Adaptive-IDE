"""Build a standalone Colab notebook using the exp11 cohort and exp08 episode SVM."""

import ast
import json
from pathlib import Path


def build():
    here = Path(__file__).resolve().parent
    name = "EyeHeartCognitiveLoad.ipynb"
    nb = json.loads((here.parent / "exp11" / name).read_text(encoding="utf-8"))
    cells = nb["cells"]
    common = "".join(cells[5]["source"]).split("def compare_fold(")[0]
    exp08 = (here.parent / "exp08/experiment.py").read_text(encoding="utf-8")
    score = next(n for n in ast.parse(exp08).body if getattr(n, "name", "") == "score_episodes")
    cells[0]["source"] = ["# Exp12 · durasi HRV pada Eye + HRV + posisi kepala\n",
                          "Gunakan input exp11 dan v2_session_phase/window_45s_step_5s/all_data.csv. "
                          "Notebook membandingkan RMSSD dan SDNN pada waktu akhir yang sama."]
    cells[1]["source"] = "".join(cells[1]["source"]).replace("exp11/outputs", "exp12/outputs").splitlines(True)
    cells[4]["source"] = ["## Pasangkan window dan evaluasi pada peserta yang disisihkan"]
    cells[5]["source"] = (common + ast.get_source_segment(exp08, score) + "\n\n"
                          + (here / "experiment.py").read_text(encoding="utf-8")).splitlines(True)
    cells[6]["source"] = ["## Evaluasi per episode dan ekspor ZIP"]
    (here / name).write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    build()
