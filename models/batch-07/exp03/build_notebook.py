"""Build standalone local/Colab notebook for Batch 07 Exp03."""

import json
from pathlib import Path


def build():
    here = Path(__file__).resolve().parent
    exp01_nb = json.loads((here.parent / "exp01/EyeHeartCognitiveLoad.ipynb").read_text(encoding="utf-8"))
    source = (here / "experiment.py").read_text(encoding="utf-8")

    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Batch 07 · Exp03 · Kalibrasi Ambang Keputusan & Perbandingan Model pada V4 (`1–6` vs `7–10`)\n",
                    "Notebook mandiri yang berjalan otomatis di lokal (`analysis_sessions/dataset_fusion/output`) maupun Google Colab.",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
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
                    "OUTPUT_ROOT = PROJECT_ROOT / 'models/batch-07/exp03/outputs'\n",
                    "print('PROJECT_ROOT:', PROJECT_ROOT)\n",
                    "print('DATASET_ROOT:', DATASET_ROOT)\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## Pipeline V4, pembanding episode-median riwayat bersih, dan kalibrasi ambang nested LOPO"],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": source.splitlines(True),
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## Jalankan evaluasi Dev LOPO (13p), Full LOPO (17p), dan Historical Test (4p)"],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["archive = run(DATASET_ROOT, OUTPUT_ROOT)\n"],
            },
        ],
        "metadata": exp01_nb["metadata"],
        "nbformat": 4,
        "nbformat_minor": 4,
    }
    (here / "EyeHeartCognitiveLoad.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    build()
