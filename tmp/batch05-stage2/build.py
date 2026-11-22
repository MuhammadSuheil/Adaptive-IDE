"""Package the Stage 2 notebook; no training runs in this builder."""

import base64
import hashlib
import json
import textwrap
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "models/batch-05/exp01/EyeHeartCognitiveLoad.ipynb"
PREPARATION = ROOT / "models/batch-05/outputs/exp01/20261122T074814995365Z_fe70ac0e"


def cell(kind, text, name, definition=False):
    result = {
        "cell_type": kind,
        "id": "stage2-" + name,
        "metadata": {"tags": ["definitions"]} if definition else {},
        "source": text.strip().splitlines(keepends=True),
    }
    if result["source"]:
        result["source"][-1] = result["source"][-1].rstrip("\n")
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def main():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = [
        c
        for c in notebook["cells"]
        if not c["id"].startswith("stage2-")
        and c["id"] not in ("execution", "execution-description")
    ]
    changes = {
        "introduction": """# Evaluasi Kontrol Cognitive Load

Window sensor 15, 25, dan 45 detik menyediakan target Low, Medium, dan High. Pembagian waktu mengikuti manifest yang dibekukan setelah pemeriksaan data dan dukungan kelas.

Training kontrol memakai window 25 detik dan 10 fitur core. Tiga classifier dibandingkan pada normalisasi global dan per sesi, dengan Dummy sebagai pembanding. Evaluasi mencakup lima fold, validation, kontrol pembagian, dan learning curve. Test tidak dievaluasi.
""",
        "paths-description": """## Lokasi data

Ketiga folder `window_*s_step_5s` menyediakan `good_acceptable.csv` dari satu proyek Drive. Hasil training tersimpan pada run baru. Untuk melanjutkan run terputus, isi `RESUME_DIR` dengan path lengkap folder run yang sama; untuk run baru biarkan None.
""",
    }
    for item in cells:
        if item["id"] in changes:
            item["source"] = changes[item["id"]].strip().splitlines(keepends=True)
        if item["id"] == "configuration":
            text = "".join(item["source"])
            if "RESUME_DIR" not in text:
                item["source"] = (text.rstrip() + "\nRESUME_DIR = None\n").splitlines(
                    keepends=True
                )
    frozen = {
        "development": json.loads(
            (PREPARATION / "development_manifest.json").read_text()
        ),
        "readiness": json.loads((PREPARATION / "readiness.json").read_text()),
    }
    encoded = base64.b85encode(
        zlib.compress(json.dumps(frozen, separators=(",", ":")).encode(), 9)
    ).decode()
    payload = (
        "STAGE2_PAYLOAD = (\n"
        + "\n".join("    " + repr(v) for v in textwrap.wrap(encoded, 100))
        + ")\n"
    )
    payload += "FROZEN_STAGE1 = json.loads(zlib.decompress(base64.b85decode(STAGE2_PAYLOAD)))\n"
    payload += "DEVELOPMENT = FROZEN_STAGE1['development']\nMANIFEST_HASH = FROZEN_STAGE1['readiness']['manifest_sha256']\n"
    stage_source = (Path(__file__).parent / "training.py").read_text(encoding="utf-8")
    before = "\n".join(
        "".join(c["source"])
        for c in cells
        if "definitions" in c["metadata"].get("tags", [])
    )
    fingerprint = hashlib.sha256((before + payload + stage_source).encode()).hexdigest()
    cells += [
        cell(
            "markdown",
            """## Manifest development

Snapshot persiapan mempertahankan seed development 125 dan seed CV 50. Validasi ulang memeriksa hash, cakupan endpoint, guard, perlindungan Test, dan cakupan out-of-fold. Tidak ada pemilihan seed berdasarkan skor classifier.
""",
            "snapshot-description",
        ),
        cell(
            "code", payload + f"STAGE2_CODE_HASH = {fingerprint!r}\n", "snapshot", True
        ),
    ]
    boundaries = [
        "class Stage2Store",
        "def restore_transform",
        "def learning_subsets",
        "def aggregate_candidate",
        "def export_graphs",
        "def run_stage2",
    ]
    pieces = []
    remainder = stage_source
    for boundary in boundaries:
        before, remainder = remainder.split(boundary, 1)
        pieces.append(before)
        remainder = boundary + remainder
    pieces.append(remainder)
    descriptions = [
        """## Model dan metrik

Logistic Regression, Extra Trees, dan HistGradientBoosting memakai konfigurasi tetap tanpa search atau balancing. Normalisasi global, session-centered, dan session-robust memakai statistik fitting saja. Seluruh model mendapat fold yang sama. Metrik mencakup accuracy, balanced accuracy, Macro-F1, dan precision/recall/F1 setiap kelas.
""",
        """## Checkpoint pekerjaan

Setiap pekerjaan selesai menyimpan prediksi, metrik, durasi, dan warning dengan checksum. Run yang dilanjutkan harus cocok pada data, manifest, kode, konfigurasi, dan lingkungan. Satu runtime menulis satu folder run. Pekerjaan terputus yang belum memiliki penanda selesai diulang.
""",
        """## Fitting dan penyimpanan estimator

Preprocessing hanya membaca baris fitting. Prediksi harus sejajar dengan ID dan target window. Warning konvergensi tercatat dan kandidat yang belum konvergen tidak masuk pemilihan akhir. Estimator fitting akhir disimpan bersama statistik preprocessing; pemuatan ulang harus menghasilkan prediksi yang sama.
""",
        """## Dukungan fitting dan pembagian referensi

Learning curve memakai subset bertingkat 25%, 50%, 75%, dan 100% blok fitting. Validation fold tetap. Kontrol pembagian lama dan baru memakai 217 window validation bersama yang mempunyai ketiga kelas. Perbedaan komposisi fitting dan guard dilaporkan bersama hasilnya.
""",
        """## Seleksi berdasarkan out-of-fold

Pemilihan kandidat memakai prediksi out-of-fold Train sebelum membaca validation akhir. Kandidat wajib konvergen dan mengenali ketiga kelas. Peringkat mengutamakan Macro-F1, accuracy, recall minimum, kestabilan fold, lalu kompleksitas. Dummy menjadi pembanding, bukan kandidat akhir.
""",
        """## Grafik dan ringkasan

Confusion matrix memperlihatkan kelas yang tertukar. Learning curve membandingkan kemampuan pada fitting dan validation ketika jumlah blok fitting bertambah. Rincian fase dan dukungan sumber membantu menelusuri kesalahan tanpa memasukkan metadata tersebut sebagai fitur.
""",
        """## Training kontrol

Training dibatasi maksimal 71 pekerjaan unik. Model dipilih melalui CV, kemudian dievaluasi pada validation. Kontrol pembagian dan learning curve menggunakan kembali pekerjaan yang identik. Gate tahap berikutnya memerlukan pengenalan tiga kelas pada OOF/validation dan mengungguli Dummy pada minimal tiga fold; tahap neural tidak berjalan otomatis.
""",
    ]
    for number, piece in enumerate(pieces):
        cells += [
            cell("markdown", descriptions[number], f"method-{number}"),
            cell("code", piece, f"definitions-{number}", True),
        ]
    cells += [
        cell(
            "code",
            """
assert FROZEN_STAGE1['readiness']['stage2_ready']
training = run_stage2(
    DATASET_ROOT, OUTPUT_ROOT, DEVELOPMENT, REFERENCE, EXPECTED_HASHES,
    MANIFEST_HASH, STAGE2_CODE_HASH, RESUME_DIR,
)
RUN_DIR = training['run_dir']
""",
            "execution",
        ),
        cell(
            "markdown",
            """## Hasil development

Tabel pertama memuat seluruh kandidat CV; tabel berikutnya menampilkan validation untuk pemenang, kontrol global, dan Dummy. Pemilihan selesai sebelum validation dibaca. Baca `RESULTS.md`, grafik pada folder `figures`, dan CSV untuk rincian fold serta komposisi data. Hasil ini belum merupakan evaluasi Test.
""",
            "results-description",
        ),
        cell(
            "code",
            """
try:
    from IPython.display import display
except ImportError:
    def display(table):
        print(table.to_string(index=False))

display(training['leaderboard'].sort_values('macro_f1', ascending=False))
display(pd.read_csv(RUN_DIR / 'validation_metrics.csv'))
display(training['controls'])
print('Gate perencanaan tahap 3:', training['gate']['ready_for_stage3_planning'])
print('Laporan:', RUN_DIR / 'RESULTS.md')
""",
            "results",
        ),
        cell(
            "code",
            """
try:
    from IPython.display import Image as NotebookImage
except ImportError:
    NotebookImage = None

for figure_path in sorted((RUN_DIR / 'figures').glob('*.png')):
    if NotebookImage is not None:
        display(NotebookImage(filename=str(figure_path)))
    else:
        print('Grafik:', figure_path)
""",
            "figures",
        ),
    ]
    notebook["cells"] = cells
    NOTEBOOK.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    # Standalone lint/type-check view of the same definitions; not a runtime dependency.
    combined = "\n\n".join(
        "".join(c["source"])
        for c in cells
        if "definitions" in c["metadata"].get("tags", [])
    )
    (Path(__file__).parent / "combined.py").write_text(combined, encoding="utf-8")
    print("Notebook:", len(cells), "cells; implementation", fingerprint)


if __name__ == "__main__":
    main()
