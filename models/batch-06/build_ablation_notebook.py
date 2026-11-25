"""Package Exp02 as one Colab notebook with its fixed Exp01 cohort embedded."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
RUN_NAME = "baseline_20261123T192338Z_8d8cee5e"
RUN = HERE / "outputs/exp01" / RUN_NAME


def main():
    spec = importlib.util.spec_from_file_location("batch06_notebook_helpers", HERE / "build_notebook.py")
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    base_source = (HERE / "baseline_experiment.py").read_text(encoding="utf-8")
    source = (HERE / "ablation_experiment.py").read_text(encoding="utf-8")
    hashes = {
        "shared_exp01": hashlib.sha256(base_source.encode()).hexdigest(),
        "exp02": hashlib.sha256(source.encode()).hexdigest(),
    }
    old_plan = json.loads((RUN / "plan.json").read_text())
    if hashes["shared_exp01"] != old_plan["implementation_sha256"]:
        raise ValueError("Shared Exp01 source changed; review before packaging identical controls.")
    folds = json.loads((RUN / "hrv_eye_fold_manifest.json").read_text())
    all_refs = json.loads((RUN / "baseline_references.json").read_text())
    refs = {sid: ref for sid, ref in all_refs.items() if "eye_values" in ref}
    source_evidence = json.loads((HERE / "source_timing_audit.json").read_text())
    split = pd.read_csv(RUN / "split_assignments.csv")
    train_set = set().union(*(set(case["held_window_ids"]) for case in folds["participant_held_out"]))
    val_predictions = pd.read_csv(RUN / "validation_predictions.csv")
    val_set = set(val_predictions.loc[val_predictions.cohort.eq("hrv_eye"), "window_id"])
    controls = []
    for path, protocol in ((RUN / "cv_leaderboard.csv", None), (RUN / "validation_metrics.csv", "validation")):
        records = pd.read_csv(path)
        for row in records[records.cohort.eq("hrv_eye") & records.family.eq("extra_trees")].itertuples():
            controls.append({
                "protocol": protocol or row.protocol,
                "representation": "raw" if row.representation == "raw" else "combined",
                "accuracy": float(row.accuracy), "macro_f1": float(row.macro_f1),
            })
    snapshot = {
        "source_run": RUN_NAME,
        "dataset_sha256": old_plan["dataset_sha256"],
        "participants": sorted(split.loc[split.window_id.isin(train_set), "participant_id"].unique().tolist()),
        "train_window_ids": split.loc[split.window_id.isin(train_set), "window_id"].tolist(),
        "validation_window_ids": split.loc[split.window_id.isin(val_set), "window_id"].tolist(),
        "test_window_ids": split.loc[split._split.eq("test"), "window_id"].tolist(),
        "folds": folds, "baseline_references": refs,
        "source_evidence": {sid: source_evidence[sid] for sid in refs},
        "exp01_controls": controls, "exp01_versions": old_plan["versions"],
        "source_artifact_sha256": {name: helpers.file_hash(RUN / name) for name in (
            "plan.json", "hrv_eye_fold_manifest.json", "baseline_references.json",
            "split_assignments.csv", "validation_predictions.csv", "cv_leaderboard.csv", "validation_metrics.csv",
        )},
    }
    destination = HERE / "exp02"
    destination.mkdir(exist_ok=True)
    (destination / "cohort_snapshot.json").write_text(json.dumps(snapshot, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    md, code = helpers.markdown, helpers.code
    cells = [
        md("""# Ablasi normalisasi baseline sensor

Batch 06 Exp02 memisahkan pengaruh transformasi RMSSD dan eye menggunakan
empat peserta, 671 window Train, 99 window validation, serta fold subset eye
Exp01 yang sama. Semua varian memakai sepuluh fitur core dan Extra Trees
200 trees dengan parameter tetap. Tidak ada tuning atau evaluasi Test.

Empat varian: raw, RMSSD-only, eye-only, dan gabungan. Kata only mengacu pada
normalisasi yang diaktifkan; fitur sensor lain tetap masuk classifier.
"""),
        md("""## Mount dan input

Cukup notebook ini serta tiga CSV di root Drive yang sama dengan batch sebelumnya.
Snapshot cohort, baseline, fold, dan bukti sumber sudah tertanam; tidak perlu
output Exp01, file Python pendamping, atau sesi mentah di Drive. Runtime CPU cukup.
Hash CSV harus sesuai run Exp01 agar pembanding benar-benar identik. Jika data
berubah, notebook berhenti dan tidak mencari split pengganti secara otomatis.
"""),
        code("""from pathlib import Path
from google.colab import drive

drive.mount('/content/drive')
PROJECT_ROOT = Path('/content/drive/MyDrive/Penelitian-Adaptive-IDE')
DATASET_ROOT = PROJECT_ROOT / 'Dataset/v2_session_phase'
OUTPUT_ROOT = PROJECT_ROOT / 'models/batch-06/outputs/exp02'
"""),
        md("""## Lingkungan

Model memakai CPU. Paket yang belum ada dipasang otomatis. Versi aktual dicatat;
kontrol raw dan gabungan dihitung ulang dalam run ini. Perbedaan terhadap Exp01
dilaporkan, tanpa mengganti seed atau parameter untuk memaksakan kecocokan.
"""),
        code("""import importlib.util
import subprocess
import sys

packages = {'numpy': 'numpy', 'pandas': 'pandas>=2.0', 'sklearn': 'scikit-learn>=1.4',
            'joblib': 'joblib', 'matplotlib': 'matplotlib', 'threadpoolctl': 'threadpoolctl'}
missing = [package for module, package in packages.items() if importlib.util.find_spec(module) is None]
if missing:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *missing])
"""),
        md("""## Definisi sensor dan fitting

Definisi dasar sama dengan Exp01; cell ini hanya mendefinisikan fungsi, tanpa
menjalankan eksperimen Exp01. Seluruh scaler dihitung dari fitting fold.
"""),
        code(base_source, ["definitions", "shared_exp01"]),
        md("""## Empat representasi tetap

Raw mempertahankan semua fitur. RMSSD-only mengganti RMSSD dengan
RMSSD/baseline − 1. Eye-only menormalkan statistik eye terhadap window pre_task
awal, sementara RMSSD dan SDNN tetap mentah. Gabungan mengaktifkan keduanya.
Tidak ada fitur yang dihapus, baseline nol blink tidak menjadi pembagi, dan
iris tetap geometri iris, bukan pengukuran pupil.

Metrik menyertakan support kelas dan bobot sama per orang. Recall kelas tanpa
contoh ditampilkan N/A; Macro-F1 tetap memakai tiga label untuk konsistensi.
"""),
        code(source + "\nABLATION_IMPLEMENTATION_HASHES = " + repr(hashes) + "\n", ["definitions", "exp02"]),
        md("""## Snapshot pembanding

Window, urutan fitting, baseline, serta empat fold peserta dan lima fold waktu
dibekukan dari run Exp01. Seluruh varian memakai daftar yang sama. Validation
lama sudah pernah diperiksa, hanya berisi 43 Low dan 56 High tanpa Medium,
dan bukan hold-out baru. Split ini dipertahankan untuk kontrol yang sama.
"""),
        code("import base64\nimport zlib\n\n" + helpers.compressed_assignment("ABLATION_SNAPSHOT", snapshot), ["reference"]),
        md("""## Pemeriksaan sebelum fitting

Periksa hash CSV, waktu/angka baseline, dukungan interval, grup orang, larangan
ID Test, serta cakupan fold. Tidak ada training penelitian pada tahap ini.
"""),
        code("""from IPython.display import display

prepared = prepare_ablation(DATASET_ROOT, ABLATION_SNAPSHOT)
print('Train:', len(prepared['train']), 'window /', prepared['train'].participant_id.nunique(), 'orang')
print('Validation:', len(prepared['validation']), 'window')
for name in ('train', 'validation'):
    print(name, 'support kelas:')
    display(prepared[name].cognitive_load_label.value_counts().reindex(LABELS, fill_value=0))
print('Fold:', {key: len(value) for key, value in prepared['folds'].items()})
print('PERHATIAN: validation tanpa Medium tidak menilai tiga kelas secara lengkap.')
"""),
        md("""## Jalankan ablasi

Budget tetap 45 fitting CV: empat varian Extra Trees dan satu Dummy pada
sembilan fold. Empat fitting Train penuh mengevaluasi semua varian yang telah
dibakukan pada validation historis. Tidak ada pemenang yang dipilih berdasarkan
validation, refit Train + validation, atau evaluasi Test. Setiap Run all membuat
folder run baru dan mempertahankan hasil lama.
"""),
        code("""training = run_ablation(DATASET_ROOT, OUTPUT_ROOT, ABLATION_SNAPSHOT,
                        ABLATION_IMPLEMENTATION_HASHES)
"""),
        md("""## Hasil

Utamakan OOF tiga kelas pada orang yang ditahan dari fitting. Baca kontras
eye-only vs raw dan gabungan vs RMSSD-only untuk memisahkan manfaat transformasi.
Laporkan juga metrik berbobot sama per orang dan jumlah sumber. Semua perbandingan
dihitung dari hasil Exp02; skor Exp01 hanya menjadi pemeriksaan kontrol.

Empat peserta membatasi kesimpulan. Label masih difficulty/fase, acuan eye masih
pre_task, dan window overlap bukan observasi independen. Angka validation tidak
boleh dipromosikan menjadi keberhasilan tiga kelas pada pengguna baru.
"""),
        code("""from IPython.display import Image, Markdown, display

display(training['leaderboard'])
display(training['contrasts'])
display(training['validation'])
display(training['controls'])
display(Markdown((training['run_dir'] / 'RESULTS.md').read_text(encoding='utf-8')))
for name in ('ablation_oof.png', 'ablation_contrasts.png', 'validation_support.png'):
    display(Image(filename=str(training['run_dir'] / 'figures' / name)))
print('Hasil:', training['run_dir'])
"""),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
            "colab": {"provenance": []}, "batch": "06", "experiment": "exp02",
            "implementation_sha256": hashes,
        },
        "nbformat": 4, "nbformat_minor": 4,
    }
    target = destination / "EyeHeartCognitiveLoad.ipynb"
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(target)
    print("Frozen cohort:", len(snapshot["train_window_ids"]), "Train /", len(snapshot["validation_window_ids"]), "validation")


if __name__ == "__main__":
    main()
