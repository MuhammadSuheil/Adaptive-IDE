"""Mechanical notebook packaging, including immutable historical split data."""

import base64
import hashlib
import json
import textwrap
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "models/batch-04/outputs/exp01/20261121T025639248779Z_9443342741"
DESTINATION = ROOT / "models/batch-05/exp01/EyeHeartCognitiveLoad.ipynb"


def cell(kind, source, name, definitions=False):
    item = {
        "cell_type": kind,
        "id": name,
        "metadata": {},
        "source": source.strip().splitlines(keepends=True),
    }
    if item["source"]:
        item["source"][-1] = item["source"][-1].rstrip("\n")
    if kind == "code":
        item.update(execution_count=None, outputs=[])
    if definitions:
        item["metadata"]["tags"] = ["definitions"]
    return item


def main():
    original = (OLD / "split.json").read_bytes()
    reference = json.loads(original)
    run = json.loads((OLD / "run.json").read_text())
    reference.update(
        source="batch-04/exp01",
        source_sha256=hashlib.sha256(original).hexdigest(),
        dataset_sha256=run["dataset_sha256"],
        role="previously_seen_exploratory_reference",
    )
    encoded = base64.b85encode(
        zlib.compress(json.dumps(reference, separators=(",", ":")).encode(), 9)
    ).decode()
    reference_source = (
        "import base64\nimport zlib\n\nREFERENCE_PAYLOAD = (\n"
        + "\n".join("    " + repr(part) for part in textwrap.wrap(encoded, 100))
        + "\n)\nREFERENCE = json.loads(zlib.decompress(base64.b85decode(REFERENCE_PAYLOAD)))\n"
        + "EXPECTED_HASHES = REFERENCE['dataset_sha256']\n"
    )
    source = (Path(__file__).parent / "definitions.py").read_text(encoding="utf-8")
    boundaries = [
        "def validate_frame",
        "def purge(",
        "def development_split",
        "class SensorPreprocessor",
        "def partition_counts",
        "def preflight",
        "def manifest_records",
        "def run_preparation",
    ]
    pieces = []
    for boundary in boundaries:
        before, source = source.split(boundary, 1)
        pieces.append(before)
        source = boundary + source
    pieces.append(source)
    implementation_sha256 = hashlib.sha256(
        (
            (Path(__file__).parent / "definitions.py").read_text(encoding="utf-8")
            + json.dumps(reference, sort_keys=True)
        ).encode()
    ).hexdigest()
    cells = [
        cell(
            "markdown",
            """
# Persiapan Evaluasi Cognitive Load

Tiga dataset sensor merangkum eye tracking dan HRV dalam window 15, 25, dan 45 detik dengan step 5 detik. Target memakai `cognitive_load_label` (Low, Medium, High). Fase, identitas, dan konteks tugas mendukung pemeriksaan sumber serta pembagian waktu.

Tahap persiapan membekukan data, memeriksa dukungan kelas dan sumber, lalu menyusun Train, validation, fold internal, dan monitor early stopping. Hasil akhir tahap ini berupa laporan kesiapan, belum berupa skor classifier.
""",
            "introduction",
        ),
        cell(
            "markdown",
            """
## Lokasi data

Setiap folder `window_*s_step_5s` menyediakan `good_acceptable.csv`. Semua window dibaca dari satu folder proyek di Drive. Setiap eksekusi menyimpan run persiapan baru.
""",
            "paths-description",
        ),
        cell(
            "code",
            """
from pathlib import Path

try:
    from google.colab import drive
except ImportError:
    drive = None

if drive is not None:
    drive.mount('/content/drive')

PROJECT_ROOT = Path('/content/drive/MyDrive/Penelitian-Adaptive-IDE')
DATASET_ROOT = PROJECT_ROOT / 'Dataset/v2_session_phase'
OUTPUT_ROOT = PROJECT_ROOT / 'models/batch-05/outputs/exp01'
""",
            "configuration",
        ),
    ]
    descriptions = [
        """## Fitur sensor

Profil utama memakai 10 fitur core: RMSSD, SDNN, laju kedipan, EAR, dan ukuran geometri iris. Profil 14 fitur menambahkan head pose jika nilainya lengkap. Ukuran geometri iris bukan diameter pupil. Metadata peserta, soal, dan kualitas tidak masuk input classifier.
""",
        """## Pemeriksaan dataset

Pemeriksaan mencakup kolom, label, timestamp berzona waktu, durasi window, ID duplikat, serta nilai core finite dan rentang fisiknya. Endpoint yang sama harus mempunyai label dan identitas sumber yang konsisten antarwindow. Nilai sumber tidak dikoreksi atau diimputasi pada tahap ini.
""",
        """## Blok waktu dan pemisahan interval

Blok 240 detik menjaga window berdekatan tetap dalam kelompok. Guard 45 detik memisahkan endpoint pada sesi yang sama, mencakup panjang window terpanjang tanpa history tambahan. Pemilihan seed memakai target, fase, waktu, dan identitas sumber; nilai sensor dan skor model tidak digunakan.
""",
        """## Development dan monitor

Test mempertahankan ID batch-04 sebagai referensi eksploratif yang sudah pernah dianalisis. Data non-Test yang memenuhi guard dibagi dengan target Train 80% dan validation 20% sebelum purge. Lima fold berasal dari Train.

Setiap fitting pool menyediakan monitor tersendiri. Fitting dan monitor harus memiliki minimal 10 window setiap kelas, dengan selisih proporsi kelas paling besar 10 poin persentase. Kandidat yang lolos dipilih berdasarkan kemiripan fase Low dan sumber, lalu retensi data. Jika tidak ada kandidat yang memenuhi batas, kesiapan neural bernilai False.
""",
        """## Normalisasi dan aturan seleksi

Tiga kontrol preprocessing memakai median dan IQR: global, pemusatan per sesi, dan penskalaan robust per sesi. Statistik hanya berasal dari fitting. Sesi yang belum tersedia pada fitting memakai statistik global. Nilai negatif hasil pemusatan dipertahankan.

Pada tahap pemodelan, kandidat harus menghasilkan prediksi finite dan recall positif pada tiga kelas development. Peringkat mengutamakan Macro-F1, diikuti accuracy, recall minimum, kestabilan fold, dan kompleksitas.
""",
        """## Dukungan kelas dan sumber

Jumlah baris, blok, dan cakupan sesi-kelas memperlihatkan dukungan tiap bagian. Sel sumber yang pendek tidak dipaksa muncul di semua bagian. Tabel monitor juga memisahkan komposisi kelas, fase, dan sesi; window overlap tidak dihitung sebagai observasi independen.
""",
        """## Pemeriksaan kesiapan

Kesiapan tabular memerlukan kelas lengkap, fold yang sah, pemisahan interval, serta preprocessing dari fitting saja. Kesiapan neural juga memerlukan monitor yang memenuhi kriteria pada setiap fold dan fitting akhir. Gate tahap berikutnya memakai window 25 detik.
""",
        """## Manifest dan laporan

Manifest menyimpan penetapan setiap endpoint dan ID window. Referensi lama dan development baru disimpan terpisah, sehingga perubahan split dapat diuji sebelum mengganti normalisasi atau model.
""",
        """## Penyimpanan persiapan

SHA-256 mengunci versi tiga CSV. Perubahan data memerlukan peninjauan referensi dan pembagian waktu sebelum fitting. Setiap run menyimpan konfigurasi, versi lingkungan, komposisi monitor, dukungan sumber, serta hasil pemeriksaan.
""",
    ]
    for i, piece in enumerate(pieces):
        cells.append(cell("markdown", descriptions[i], f"description-{i}"))
        cells.append(cell("code", piece, f"definitions-{i}", True))
        if i == 0:
            cells.append(
                cell(
                    "code",
                    f"IMPLEMENTATION_SHA256 = {implementation_sha256!r}",
                    "implementation-fingerprint",
                    True,
                )
            )
            cells.append(
                cell(
                    "markdown",
                    """
### Referensi pembagian sebelumnya

Manifest referensi mempertahankan penetapan batch-04 beserta hash dataset. Test ini sudah pernah digunakan dalam analisis; evaluasi padanya tidak dianggap pengujian baru yang independen.
""",
                    "reference-description",
                )
            )
            cells.append(cell("code", reference_source, "frozen-reference", True))
    cells += [
        cell(
            "markdown",
            """
## Uji aturan sebelum membaca data

Kasus sintetis memeriksa batas guard, normalisasi train-only, fallback sesi baru, serta penolakan kandidat yang kehilangan satu kelas. Pemeriksaan ini tidak melakukan training classifier.
""",
            "selfcheck-description",
        ),
        cell(
            "code",
            """
probe = pd.DataFrame({'session_id': ['a', 'a', 'a', 'b'],
                      '_end': np.array([0, 40, 45, 0], dtype=np.int64) * 10**9})
assert purge(probe, [1, 2, 3], [0]).tolist() == [2, 3]
probe = pd.DataFrame({'session_id': ['a', 'a', 'b', 'new'], 'rmssd_ms': [10., 30., 50., 999.]})
assert preprocessing_check(probe, [0, 1, 2], [3], ['rmssd_ms'])
for mode in ('global', 'session_centered', 'session_robust'):
    transform = SensorPreprocessor(['rmssd_ms'], mode).fit(probe, [0, 1, 2])
    assert transform.transform(probe.iloc[[0]])[0, 0] < 0
    assert np.isfinite(transform.transform(probe.iloc[[3]])).all()
assert select_candidate([{'name': 'collapsed', 'macro_f1': .8, 'accuracy': .9,
                          'recall': [.9, .8, 0], 'fold_std': .1, 'complexity': 1,
                          'finite_predictions': True}]) is None
print('Pemeriksaan aturan selesai.')
""",
            "selfchecks",
        ),
        cell(
            "markdown",
            """
## Hasil persiapan

Periksa tabel kesiapan dan `PREPARATION.md` pada folder run. Status False harus diselesaikan sebelum tahap terkait dimulai. Run all pada tahap ini berakhir setelah penyimpanan laporan.
""",
            "execution-description",
        ),
        cell(
            "code",
            """
preparation, RUN_DIR = run_preparation(
    DATASET_ROOT, OUTPUT_ROOT, REFERENCE, EXPECTED_HASHES, IMPLEMENTATION_SHA256
)
try:
    from IPython.display import display
except ImportError:
    def display(table):
        print(table.to_string(index=False))

display(pd.DataFrame(preparation['readiness']))
display(preparation['counts'].query("context == 'outer'").reset_index(drop=True))
print('Laporan:', RUN_DIR / 'PREPARATION.md')
""",
            "execution",
        ),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    DESTINATION.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print(DESTINATION.relative_to(ROOT), len(cells), "cells")


if __name__ == "__main__":
    main()
