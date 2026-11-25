"""Build standalone Colab notebooks from the reviewed source cells."""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CASES = {
    "v3": ("exp03", "Proxy perilaku Low, Medium, High",
           "Bandingkan Logistic Regression dan Extra Trees berbobot pada split resmi. "
           "Ablasi HRV dan eye memakai Extra Trees yang sama. Pilih model menurut Macro-F1 episode "
           "validation, lalu nilai kandidat terkunci dan Dummy pada Test."),
    "v4": ("exp04", "Klasifikasi rating kesulitan 1–10",
           "Bandingkan classifier nominal dan ordinal pada 17 fold peserta (LOPO pada paket saat ini). "
           "Tiap episode mempunyai bobot fitting yang sama. Baca MAE episode bersama accuracy tepat, "
           "akurasi dalam satu tingkat, Macro-F1 sepuluh kelas dan dukungan tiap rating."),
}


def cell(kind, source):
    result = {"cell_type": kind, "metadata": {}, "source": source.strip().splitlines(keepends=True)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def main():
    sections = {}
    for section in (HERE / "experiment.py").read_text(encoding="utf-8").split("# %% ")[1:]:
        title, source = section.split("\n", 1)
        sections[title] = source
    for case, (exp, title, description) in CASES.items():
        setup = f"""from pathlib import Path
from google.colab import drive

drive.mount('/content/drive')
PROJECT_ROOT = Path('/content/drive/MyDrive/Penelitian-Adaptive-IDE')
DATASET_ROOT = PROJECT_ROOT / 'models/batch-06/dataset'
OUTPUT_ROOT = PROJECT_ROOT / 'models/batch-06/outputs/{exp}'
DOWNLOAD_ZIP = False  # True untuk langsung mengunduh arsip melalui browser."""
        cells = [cell("markdown", f"# Batch 06 {exp} · {title}\n\n{description}"),
                 cell("markdown", "## Lokasi data dan hasil\n\nGunakan CPU dan pilih **Run all**. "
                      "Sesuaikan DATASET_ROOT ke folder induk paket bila lokasi Drive berbeda. "
                      "ZIP otomatis tersimpan di OUTPUT_ROOT."),
                 cell("code", setup)]
        for name in ("Input CSV", "Model", "Metrik dan ekspor", case.upper()):
            cells.extend([cell("markdown", f"## {name}"), cell("code", sections[name])])
        cells.extend([cell("markdown", "## Jalankan eksperimen\n\nSalin ZIP hasil ke repository untuk analisis."),
                      cell("code", f"""archive = run_{case}(DATASET_ROOT, OUTPUT_ROOT)
if DOWNLOAD_ZIP:
    from google.colab import files
    files.download(str(archive))""")])
        notebook = {"cells": cells, "nbformat": 4, "nbformat_minor": 4, "metadata": {
            "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
            "language_info": {"name": "python"}}}
        directory = HERE.parent / exp
        directory.mkdir(exist_ok=True)
        (directory / "EyeHeartCognitiveLoad.ipynb").write_text(
            json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
