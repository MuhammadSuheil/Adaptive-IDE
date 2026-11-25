"""Build standalone local/Colab notebook for Batch 07 Exp02."""

import json
from pathlib import Path


def build():
    here = Path(__file__).resolve().parent
    exp01_nb = json.loads((here.parent / "exp01/EyeHeartCognitiveLoad.ipynb").read_text(encoding="utf-8"))
    cells_01 = exp01_nb["cells"]

    # Reuse validated reader and V3 concordance labeler from Exp01
    reader_code = "".join(cells_01[3]["source"])
    common_01 = "".join(cells_01[5]["source"])
    prefix_end = common_01.index('"""Compare three estimators on identical held-out v3 episodes."""')
    cohort_prefix = common_01[:prefix_end].rstrip()
    exp02_source = (here / "experiment.py").read_text(encoding="utf-8")

    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Batch 07 · Exp02 · Normalisasi IQR Personal Riwayat Bersih, Ensemble Peringkat, & Kalibrasi Ambang pada V3\n",
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
                    "OUTPUT_ROOT = PROJECT_ROOT / 'models/batch-07/exp02/outputs'\n",
                    "print('PROJECT_ROOT:', PROJECT_ROOT)\n",
                    "print('DATASET_ROOT:', DATASET_ROOT)\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## Reader dataset resmi dan fitur sensor"],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": reader_code.splitlines(True),
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## Riwayat bersih vs terkurasi, normalisasi IQR personal, ensemble SVM+ET, dan kalibrasi ambang dalam-fold"],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": (cohort_prefix + "\n\n\n" + exp02_source).splitlines(True),
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## Jalankan evaluasi dan simpan artefak"],
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
