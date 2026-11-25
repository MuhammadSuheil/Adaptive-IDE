"""Reuse the standalone exp05 Colab setup and checked dataset reader."""

import json
from pathlib import Path


def build():
    here = Path(__file__).resolve().parent
    template = here.parent / "exp05/EyeHeartCognitiveLoad.ipynb"
    notebook = json.loads(template.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    cells[0]["source"] = ["# Exp06 · riwayat sensor sebelum episode\n",
                          "Run all pada CPU untuk kedua target. Dataset harus berisi paket v3, v4 dan "
                          "folder v2_session_phase/window_15s_step_5s/ dengan good_acceptable.csv dan all_data.csv."]
    cells[1]["source"] = "".join(cells[1]["source"]).replace("exp05/outputs", "exp06/outputs").splitlines(True)
    cells[4]["source"] = ["## Fitur relatif terhadap riwayat sesi dan pembandingan berpasangan"]
    sources = [(here / name).read_text(encoding="utf-8") for name in ["recovery.py", "history.py"]]
    cells[5]["source"] = "\n\n".join(sources).splitlines(True)
    cells[7]["source"] = ["archive = run_history(DATASET_ROOT, OUTPUT_ROOT)"]
    (here / template.name).write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    build()
