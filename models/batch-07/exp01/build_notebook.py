"""Reuse Batch 06 cohort validation and ship a standalone Colab notebook."""

import ast
import json
from pathlib import Path


def definitions(source, names):
    parsed = ast.parse(source)
    functions = {node.name: ast.get_source_segment(source, node) for node in parsed.body
                 if isinstance(node, ast.FunctionDef)}
    return "\n\n".join(functions[name] for name in names)


def build():
    here = Path(__file__).resolve().parent
    v3 = here.parents[1] / "batch-06/v3"
    name = "EyeHeartCognitiveLoad.ipynb"
    template = json.loads((v3 / "exp13" / name).read_text(encoding="utf-8"))
    cells = template["cells"]
    common = "".join(cells[5]["source"])
    label_and_window_code = common[:common.index("def history_streams(")]
    pool_code = definitions(common, ["prior_reference", "binary", "snapshot_hash", "build_pool", "fold_labels"])
    fold_code = (v3 / "exp09/experiment.py").read_text(encoding="utf-8")
    source = (here / "experiment.py").read_text(encoding="utf-8")
    cells[0]["source"] = ["# Batch 07 · Exp01 · Perbandingan model v3\n",
                          "Jalankan semua sel pada CPU (lokal maupun Colab). Secara lokal notebook otomatis "
                          "membaca `analysis_sessions/dataset_fusion/output`."]
    cells[1]["source"] = [
        "import sys\n",
        "from pathlib import Path\n",
        "\n",
        "if 'google.colab' in sys.modules:\n",
        "    from google.colab import drive\n",
        "    drive.mount('/content/drive')\n",
        "    PROJECT_ROOT = Path('/content/drive/MyDrive/Penelitian-Adaptive-IDE')\n",
        "    DATASET_ROOT = PROJECT_ROOT / 'Dataset'\n",
        "else:\n",
        "    cwd = Path.cwd().resolve()\n",
        "    PROJECT_ROOT = next((p for p in [cwd, *cwd.parents] if (p / 'analysis_sessions/dataset_fusion/output').exists()), cwd)\n",
        "    DATASET_ROOT = PROJECT_ROOT / 'analysis_sessions/dataset_fusion/output'\n",
        "OUTPUT_ROOT = PROJECT_ROOT / 'models/batch-07/exp01/outputs'\n",
        "print('PROJECT_ROOT:', PROJECT_ROOT)\n",
        "print('DATASET_ROOT:', DATASET_ROOT)\n",
    ]
    cells[2]["source"] = ["## Reader dataset resmi dan fitur sensor"]
    cells[4]["source"] = ["## Riwayat sebelum soal dan fitting per peserta"]
    cells[5]["source"] = "\n\n".join([label_and_window_code, pool_code,
                                       definitions(fold_code, ["fold_data", "high_score"]), source]).splitlines(True)
    cells[6]["source"] = ["## Evaluasi LOPO dan validasi, lalu ekspor ZIP"]
    (here / name).write_text(json.dumps(template, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    build()
