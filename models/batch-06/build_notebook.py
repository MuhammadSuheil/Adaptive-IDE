"""Build the self-contained Colab notebook; runtime needs only three CSVs.

The timing audit is generated from this checkout's canonical sources. Updating
it requires source evidence, not inferred task boundaries from predictions.
"""

import ast
import base64
import hashlib
import json
import textwrap
import zlib
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def file_hash(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def extract_payload(notebook, name):
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        for node in ast.parse("".join(cell["source"])).body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
                return json.loads(zlib.decompress(base64.b85decode(ast.literal_eval(node.value))))
    raise ValueError(f"Payload {name} tidak ditemukan.")


def timing_evidence():
    dataset = ROOT / "analysis_sessions/dataset_fusion/output/v2_session_phase"
    frames = [pd.read_csv(dataset / f"window_{w}s_step_5s/good_acceptable.csv", usecols=["session_id", "session_folder", "participant_id"]) for w in (15, 25, 45)]
    sources = pd.concat(frames).drop_duplicates("session_id").sort_values("session_id")
    evidence = {}
    for row in sources.itertuples(index=False):
        folder = ROOT / "sessions/saved_session" / row.session_folder
        hrv = pd.read_csv(folder / "hrv.csv", float_precision="round_trip")
        events = pd.read_csv(folder / "task_events.csv")
        baseline = hrv[hrv.phase.eq("baseline")]
        if len(baseline) != 1:
            raise ValueError(f"{row.session_folder}: expected one baseline row")
        baseline = baseline.iloc[0]
        if str(baseline.session_id) != str(row.session_id) or not events.session_id.eq(row.session_id).all():
            raise ValueError("Sumber/session_id berbeda.")
        first = pd.to_numeric(events.loc[events.event_type.isin(("set_started", "puzzle_started")), "event_time_ms"], errors="raise").min()
        if pd.isna(first):
            raise ValueError("Awal tugas tidak tersedia.")
        evidence[str(row.session_id)] = {
            "session_folder": str(row.session_folder), "participant_id": str(row.participant_id),
            "baseline_end_ns": int(pd.Timestamp(baseline.window_end).value),
            "first_task_ns": int(first) * 1_000_000,
            "baseline_rmssd_ms": float(baseline.rmssd_ms),
            "baseline_coverage": float(baseline.coverage),
            "baseline_duration_seconds": float(baseline.window_seconds),
            "baseline_status": str(baseline.status),
            "source_sha256": {name: file_hash(folder / name) for name in ("hrv.csv", "task_events.csv")},
        }
    return evidence


def markdown(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.strip().splitlines(keepends=True)}


def code(source, tags=None):
    return {"cell_type": "code", "metadata": {"tags": tags or []}, "execution_count": None, "outputs": [], "source": source.strip().splitlines(keepends=True)}


def compressed_assignment(name, value):
    payload = base64.b85encode(zlib.compress(json.dumps(value, sort_keys=True).encode(), level=9)).decode()
    chunks = "\n".join("    " + repr(payload[i:i + 100]) for i in range(0, len(payload), 100))
    return f"{name}_PAYLOAD = (\n{chunks}\n)\n{name} = json.loads(zlib.decompress(base64.b85decode({name}_PAYLOAD)))\n"


def main():
    historical = json.loads((ROOT / "models/batch-05/exp01/EyeHeartCognitiveLoad.ipynb").read_text(encoding="utf-8"))
    first = extract_payload(historical, "REFERENCE_PAYLOAD")
    frozen = extract_payload(historical, "STAGE2_PAYLOAD")
    reference = {"dataset_sha256": first["dataset_sha256"], "development": frozen["development"]}
    evidence = timing_evidence()
    (HERE / "source_timing_audit.json").write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    implementation = (HERE / "baseline_experiment.py").read_text(encoding="utf-8")
    implementation_hash = hashlib.sha256(implementation.encode()).hexdigest()
    cells = [
        markdown("""
        # Model umum dengan baseline awal sensor

        Batch 06 menguji apakah representasi relatif terhadap baseline meningkatkan
        klasifikasi Low, Medium, dan High. Training memakai window 25 detik dan
        sepuluh fitur core. Ketiga CSV dibaca dari Drive untuk pemeriksaan data.

        Raw dan baseline-relative dibandingkan pada baris serta fold yang identik.
        Evaluasi utama menahan seluruh data seorang peserta dari fitting. Ini bukan
        training classifier personal. Test tidak dievaluasi oleh notebook ini.
        """),
        markdown("""
        ## Mount dan lokasi data

        Root sama dengan batch sebelumnya. Salin notebook ini ke folder exp01 di
        Drive, buka di Google Colab, lalu Run all. Tidak perlu file Python pendamping,
        requirements, rekaman mentah, atau output batch sebelumnya. Ubah PROJECT_ROOT
        hanya jika nama/lokasi folder proyek berbeda. Setiap Run all membuat run baru.
        """),
        code("""
from pathlib import Path
from google.colab import drive

drive.mount('/content/drive')
PROJECT_ROOT = Path('/content/drive/MyDrive/Penelitian-Adaptive-IDE')
DATASET_ROOT = PROJECT_ROOT / 'Dataset/v2_session_phase'
OUTPUT_ROOT = PROJECT_ROOT / 'models/batch-06/outputs/exp01'
        """),
        markdown("""
        ## Lingkungan

        Model tabular memakai CPU, tidak memerlukan GPU. Paket yang belum tersedia
        dipasang otomatis; versi aktual disimpan di plan.json bersama hash tiga CSV.
        """),
        code("""
import importlib.util
import subprocess
import sys

packages = {'numpy': 'numpy', 'pandas': 'pandas>=2.0', 'sklearn': 'scikit-learn>=1.4',
            'joblib': 'joblib', 'matplotlib': 'matplotlib', 'threadpoolctl': 'threadpoolctl'}
missing = [package for module, package in packages.items() if importlib.util.find_spec(module) is None]
if missing:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *missing])
        """),
        markdown("""
        ## Kontrak fitur dan eksperimen

        Dua model tetap: Logistic Regression C=1 dan Extra Trees 200 trees dengan
        min_samples_leaf=5. Dummy memilih kelas mayoritas fitting. Tidak ada search,
        balancing, relabel, atau pemilihan seed dari skor.

        Pada hrv_rmssd, hanya RMSSD diubah menjadi RMSSD / baseline_rmssd_ms - 1.
        SDNN tetap mentah karena CSV tidak menyimpan baseline SDNN independen.
        Pada hrv_eye, fitur eye juga dinormalkan terhadap satu window pre_task awal
        yang lengkap. Acuan ini belum membuktikan kondisi istirahat fisiologis.
        Statistik scaling setelah transformasi hanya dihitung dari fitting fold.
        """),
        code(implementation + f"\nIMPLEMENTATION_SHA256 = {implementation_hash!r}\n", ["definitions"]),
        markdown("""
        ## Bukti baseline dan referensi split

        Batas waktu akhir baseline dan awal tugas di bawah berasal dari ledger
        kanonis yang sudah ditelusuri. Nilai baseline dan coverage wajib cocok dengan
        CSV Drive. Peserta baru yang belum mempunyai bukti waktu dikeluarkan dari
        eksperimen ini dan dicatat pada baseline_audit.csv. Tidak ada baseline yang
        dibuat dari rata-rata data tugas atau dipilih berdasarkan prediksi.

        Notebook menggunakan CSV Drive yang benar-benar dibaca. Bila ketiga hash
        sama dengan batch-05, penetapan Train/validation/Test lama dipertahankan.
        Bila dataset berbeda, split baru menggunakan grup orang dengan seed tetap
        42/43. Kedua kasus tidak boleh dibandingkan sebagai kondisi yang identik.

        Referensi eye dapat berasal dari window kalibrasi pre_task tanpa memakai
        targetnya. Window tersebut dan seluruh window yang mendukung intervalnya
        tidak ikut fitting atau penilaian. Test tidak menjadi endpoint evaluasi.
        """),
        code("import base64\nimport zlib\n\n" + compressed_assignment("REFERENCE", reference) + "\n" + compressed_assignment("SOURCE_EVIDENCE", evidence), ["reference"]),
        markdown("""
        ## Periksa data sebelum fitting

        Pemeriksaan mencakup fitur finite, ID, target, waktu, kecocokan antarpanjang
        window, baseline awal yang layak, serta ukuran subset. Dua rekaman orang
        yang sama tetap satu grup saat evaluasi. Acuan eye harus seluruhnya berada
        sebelum set/practice pertama menurut ledger dan selesai sebelum interval
        fitur yang diprediksi. Subset yang terlalu kecil dilewati secara eksplisit.
        """),
        code("""
views, dataset_hashes, dataset_paths = load_views(DATASET_ROOT)
development, split_protocol = split_frame(views[25], dataset_hashes, REFERENCE)
baseline_audit, baseline_references = audit_baselines(development, SOURCE_EVIDENCE)
print('Protokol:', split_protocol)
print('Hash dataset:', dataset_hashes)
display(development.groupby(['_split', 'cognitive_load_label']).size().unstack(fill_value=0))
display(baseline_audit)
for cohort in ('hrv_rmssd', 'hrv_eye'):
    selected = make_cohort(development, baseline_audit, baseline_references, cohort)
    print(cohort, 'baris per bagian:', selected._split.value_counts().to_dict())
    print(cohort, 'orang Train:', selected.loc[selected._split.eq('train'), 'participant_id'].nunique())
        """),
        markdown("""
        ## Jalankan pembanding terkontrol

        Plan dan daftar fold dibekukan sebelum fitting. OOF utama menahan satu orang
        pada setiap fold dan memakai Train saja. Pada dataset historis, kontrol CV
        blok memakai lima fold batch-05 dengan guard 45 detik. Semua metrik pasangan
        berasal dari endpoint yang sama; skor kedua subset tidak dibandingkan langsung.

        Kandidat tiap representasi dikunci menurut Macro-F1 OOF utama dengan recall
        positif pada ketiga kelas, lalu diperiksa pada validation. Tidak ada refit
        Train + validation, evaluasi Test, atau promosi model dari skor validation.
        """),
        code("""
training = run_experiment(DATASET_ROOT, OUTPUT_ROOT, REFERENCE, SOURCE_EVIDENCE,
                          IMPLEMENTATION_SHA256)
        """),
        markdown("""
        ## Hasil dan interpretasi

        Baca delta raw-relative pada subset dan protokol yang sama. OOF peserta yang
        ditahan mengukur generalisasi ke orang yang tidak ikut fitting; validation
        historis masih dapat berisi sesi yang dikenal. Laporkan Macro-F1, recall,
        jumlah orang/window, dan jumlah fold yang membaik bersama accuracy.

        Seluruh hasil adalah eksplorasi development. Window saling overlap dan bukan
        observasi independen. Label tetap mengikuti protokol difficulty/fase; manfaat
        baseline tidak dengan sendirinya memvalidasi beban mental aktual peserta.
        """),
        code("""
from IPython.display import Image, Markdown, display

display(training['leaderboard'])
display(training['paired'])
display(training['validation'])
display(Markdown((training['run_dir'] / 'RESULTS.md').read_text(encoding='utf-8')))
for name in ('participant_oof.png', 'paired_change.png'):
    display(Image(filename=str(training['run_dir'] / 'figures' / name)))
print('Folder hasil:', training['run_dir'])
        """),
    ]
    # Dedent prose cells without disturbing Python source indentation.
    for cell in cells:
        if cell["cell_type"] == "markdown":
            source = "".join(cell["source"])
            lines = source.splitlines()
            if len(lines) > 1:
                source = lines[0] + "\n" + textwrap.dedent("\n".join(lines[1:]))
            cell["source"] = (source.strip() + "\n").splitlines(keepends=True)
    notebook = {
        "cells": cells,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python", "version": "3.12"},
                     "colab": {"provenance": []}, "batch": "06", "implementation_sha256": implementation_hash},
        "nbformat": 4, "nbformat_minor": 4,
    }
    output = HERE / "exp01/EyeHeartCognitiveLoad.ipynb"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Notebook: {output}")
    print(f"Sumber audit waktu: {len(evidence)} sesi")


if __name__ == "__main__":
    main()
