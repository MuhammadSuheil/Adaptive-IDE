"""Package Stage 3 into the existing standalone notebook, without training."""

import base64
import hashlib
import json
import textwrap
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "models/batch-05/exp01/EyeHeartCognitiveLoad.ipynb"
MONITORS = (
    ROOT / "models/batch-05/outputs/exp01/20261122T074814995365Z_fe70ac0e/monitors.json"
)


def cell(kind, source, name, definition=False):
    result = {
        "cell_type": kind,
        "id": "stage3-" + name,
        "metadata": {},
        "source": source.strip().splitlines(keepends=True),
    }
    if result["source"]:
        result["source"][-1] = result["source"][-1].rstrip("\n")
    if definition:
        result["metadata"]["tags"] = ["definitions"]
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def main():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    archive = Path(__file__).parent / "stage2_notebook.ipynb"
    if not archive.exists():
        archive.write_bytes(NOTEBOOK.read_bytes())
    cells = [
        c
        for c in notebook["cells"]
        if not c["id"].startswith("stage3-")
        and c["id"]
        not in (
            "stage2-execution",
            "stage2-results-description",
            "stage2-results",
            "stage2-figures",
        )
    ]
    for item in cells:
        if item["cell_type"] == "code":
            item.update(execution_count=None, outputs=[])
        if item["id"] == "introduction":
            item["source"] = [
                "# Evaluasi Neural Cognitive Load\n",
                "\n",
                "MLP dan kontrol Extra Trees memakai window 25 detik dengan 10 fitur core. Pembagian waktu dan monitor mengikuti snapshot persiapan.\n",
                "\n",
                "Tiga aturan checkpoint berasal dari lintasan training yang sama. Kandidat dipilih melalui OOF, diperiksa pada validation, lalu dikunci sebelum evaluasi Test referensi.\n",
            ]
    monitors = json.loads(MONITORS.read_text())["25"]
    serialized = json.dumps(monitors, sort_keys=True, allow_nan=False)
    monitor_hash = hashlib.sha256(serialized.encode()).hexdigest()
    encoded = base64.b85encode(
        zlib.compress(json.dumps(monitors, separators=(",", ":")).encode(), 9)
    ).decode()
    payload = (
        "STAGE3_MONITOR_PAYLOAD = (\n"
        + "\n".join("    " + repr(v) for v in textwrap.wrap(encoded, 100))
        + ")\n"
    )
    payload += "STAGE3_MONITORS = json.loads(zlib.decompress(base64.b85decode(STAGE3_MONITOR_PAYLOAD)))\n"
    payload += f"MONITOR_HASH = {monitor_hash!r}\n"
    source = (Path(__file__).parent / "neural.py").read_text(encoding="utf-8")
    workflow = (Path(__file__).parent / "workflow.py").read_text(encoding="utf-8")
    earlier = "\n".join(
        "".join(c["source"])
        for c in cells
        if "definitions" in c["metadata"].get("tags", [])
    )
    fingerprint = hashlib.sha256(
        (earlier + payload + source + workflow).encode()
    ).hexdigest()
    cells += [
        cell(
            "markdown",
            """## Monitor optimizer-fitting

Monitor yang dibekukan menjaga pemisahan fase pemilihan checkpoint dari validation. Statistik preprocessing hanya berasal dari optimizer-fitting. Neural dan kontrol matched-data menggunakan baris fitting yang sama; baseline penuh mempertahankan seluruh fitting pool.
""",
            "monitor-method",
        ),
        cell(
            "code",
            payload + f"STAGE3_CODE_HASH = {fingerprint!r}",
            "monitor-snapshot",
            True,
        ),
    ]
    pieces, remainder = [], source
    for boundary in (
        "def build_stage3_mlp",
        "def validate_neural_roles",
        "def neural_job",
        "def stage3_tree_job",
        "def predict_stage3_descriptor",
    ):
        before, remainder = remainder.split(boundary, 1)
        pieces.append(before)
        remainder = boundary + remainder
    pieces += [remainder, workflow]
    descriptions = [
        """## Checkpoint dan lingkungan

Training menyimpan optimizer, state RNG, dan bobot terpilih setiap lima epoch. Dua slot menjaga checkpoint sebelumnya tetap tersedia selama penulisan berikutnya. Resume memerlukan konfigurasi, data, kode, versi pustaka, serta device yang kompatibel.
""",
        """## MLP dan tiga aturan checkpoint

Arsitektur memakai hidden layer 64 dan 32, LayerNorm, GELU, serta dropout 0,2. Training memakai AdamW dengan learning rate tetap 0,001 selama 120 epoch. Checkpoint dipilih berdasarkan minimum loss monitor, maksimum Macro-F1 monitor, atau epoch 120. Tiga seed digabung melalui rata-rata probabilitas.
""",
        """## Pemeriksaan peran data

Optimizer-fitting, monitor, dan held-out harus terpisah dengan guard 45 detik. Fitting dan monitor memerlukan minimal 10 window tiap kelas serta gap proporsi maksimal 10 poin persentase. Test dilarang masuk training dan pemilihan checkpoint.
""",
        """## Lintasan training neural

Kurva fitting dan monitor dicatat setiap epoch. Checkpoint terbaik memakai salinan bobot sehingga epoch berikutnya tidak mengubahnya. Prediksi held-out dihitung setelah seluruh aturan checkpoint selesai diterapkan pada monitor.
""",
        """## Kontrol tabular

Extra Trees matched-data memakai baris optimizer-fitting yang sama dengan neural. Baseline penuh memakai fitting pool CV seperti tahap sebelumnya. Model dan normalisasi tetap, tanpa search atau balancing tambahan.
""",
        """## Penguncian model dan Test referensi

Identitas model, preprocessing, dan bobot dikunci sebelum Test dibuka. Hanya kandidat terpilih yang dievaluasi. Hasil Test yang sudah tersimpan dipakai ulang saat resume, dan checksum mencegah penggantian artefak setelah penguncian. Test ini sudah pernah dianalisis, sehingga hasilnya merupakan referensi eksploratif.
""",
        """## Pemilihan OOF dan laporan

Bandingkan sembilan kandidat neural, tiga kontrol matched-data, dan baseline penuh pada baris OOF yang sama. Peringkat mengutamakan Macro-F1, accuracy, recall minimum, dan kestabilan fold. Kandidat terpilih tidak diganti setelah membaca validation atau Test. Training akhir mempertahankan checkpoint monitor; tidak memakai median epoch CV untuk refit.
""",
    ]
    for i, piece in enumerate(pieces):
        cells += [
            cell("markdown", descriptions[i], f"method-{i}"),
            cell("code", piece, f"definitions-{i}", True),
        ]
    cells += [
        cell(
            "code",
            """
training = run_stage3(
    DATASET_ROOT, OUTPUT_ROOT, DEVELOPMENT, REFERENCE, EXPECTED_HASHES,
    MANIFEST_HASH, STAGE3_MONITORS, MONITOR_HASH, STAGE3_CODE_HASH, RESUME_DIR,
)
RUN_DIR = training['run_dir']
""",
            "execution",
        ),
        cell(
            "markdown",
            """## Hasil evaluasi

Tabel OOF menentukan kandidat sebelum evaluasi akhir. Validation dan Test referensi menilai model terkunci yang sama. Periksa recall tiap kelas bersama accuracy dan Macro-F1; satu skor gabungan tidak menunjukkan apakah semua kelas telah dikenali.
""",
            "results-method",
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
if training['winner'] is not None:
    display(pd.read_csv(RUN_DIR / 'validation_metrics.csv'))
    print('Model terpilih:', training['winner']['name'])
    print('Validation:', training['validation'])
    print('Test referensi:', training['test']['metrics'])
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
    combined = "\n\n".join(
        "".join(c["source"])
        for c in cells
        if "definitions" in c["metadata"].get("tags", [])
    )
    (Path(__file__).parent / "combined.py").write_text(combined, encoding="utf-8")
    print("Stage 3 notebook:", len(cells), "cells; implementation", fingerprint)


if __name__ == "__main__":
    main()
