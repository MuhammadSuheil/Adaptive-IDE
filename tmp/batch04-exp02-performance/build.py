"""Mechanically assemble self-contained notebooks from the reviewed source cells."""

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def cell(kind, name, source):
    result = {
        "cell_type": kind,
        "id": name,
        "metadata": {},
        "source": source.strip().splitlines(keepends=True),
    }
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def build(experiment):
    title = {
        "exp01": "Model Tabular dan Representasi Sensor",
        "exp02": "Acuan dan Riwayat Statistik Sensor",
        "exp03": "Deep Learning Tabular",
        "exp04": "Pemodelan Urutan Sensor",
    }[experiment]
    cells = [
        cell(
            "markdown",
            "title",
            f"# {title}\n\nKlasifikasi cognitive load Low, Medium, dan High dari fitur eye tracking dan HRV pada window 15, 25, dan 45 detik.",
        )
    ]
    cells += [
        cell(
            "markdown",
            "configuration_note",
            "## Konfigurasi\n\nDataset dan hasil eksperimen disimpan di Google Drive. Tentukan folder proyek dan path run bila melanjutkan eksperimen yang terputus.",
        ),
        cell(
            "code",
            "config",
            f"""from pathlib import Path

PROJECT_ROOT = Path('/content/drive/MyDrive/Penelitian-Adaptive-IDE')
DATASET_ROOT = PROJECT_ROOT/'Dataset/v2_session_phase'
RESUME_DIR = None
EXPERIMENT = '{experiment}'
""",
        ),
    ]
    setup = """import importlib.metadata as _metadata
import subprocess
import sys

if sys.version_info < (3, 12):
    raise RuntimeError('Gunakan Python 3.12 atau lebih baru.')
_required = {'numpy': '2.5.1', 'pandas': '3.0.3', 'scipy': '1.18.0', 'scikit-learn': '1.9.1',
             'cloudpickle': '3.1.3', 'matplotlib': '3.11.0', 'optuna': '5.1.0'}
if EXPERIMENT=='exp01':
    _required.update({'xgboost': '3.4.1', 'catboost': '1.2.10', 'imbalanced-learn': '0.14.2'})
if EXPERIMENT=='exp03':
    _required.update(tabm='0.0.3', rtdl_num_embeddings='0.0.12')
_install = []
_loaded_change = False
for _name, _version in _required.items():
    try:
        _current = _metadata.version(_name)
    except _metadata.PackageNotFoundError:
        _current = None
    if _current!=_version:
        _install.append(f'{_name}=={_version}')
        _module = {'scikit-learn': 'sklearn', 'imbalanced-learn': 'imblearn'}.get(_name, _name.replace('-', '_'))
        _loaded_change |= _module in sys.modules
if _install:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *_install], check=True)
    if _loaded_change:
        raise RuntimeError('Dependency diperbarui. Restart runtime, lalu jalankan dari awal.')
if EXPERIMENT in ('exp03', 'exp04'):
    import torch
    print('Perangkat neural:', 'GPU' if torch.cuda.is_available() else 'CPU')
try:
    from google.colab import drive
except ImportError:
    drive = None
if drive is not None:
    drive.mount('/content/drive')
if not DATASET_ROOT.is_dir():
    raise FileNotFoundError(f'Folder dataset tidak ditemukan: {DATASET_ROOT}')
print('Folder dataset:', DATASET_ROOT)
"""
    cells.append(cell("code", "setup", setup))
    notes = {
        "imports": None,
        "data": (
            "## Dataset dan Pembagian Data\n\nBlok waktu 240 detik dipisahkan menjadi Train, validation akhir, dan test. Penempatan dipilih menggunakan proporsi kelas, fase Low, serta sumber rekaman, bukan skor model. Purge 105 detik memisahkan rentang sensor beserta riwayatnya. Data orang yang sama boleh berada pada bagian berbeda. Lima fold internal dan satu pengulangan dipakai untuk menilai kandidat di dalam Train."
        ),
        "features": (
            "## Representasi dan Prapemrosesan\n\nCore memakai 10 fitur HRV, kedipan, EAR, dan estimasi iris. Core-head menambahkan empat statistik orientasi kepala. Gaze, identitas, soal, difficulty, dan fase tidak menjadi predictor. Scaler, seleksi fitur, serta acuan per sesi dihitung hanya pada fitting data. Riwayat bersifat kausal dan berhenti pada celah waktu; nilai negatif setelah pemusatan tetap dipertahankan."
        ),
        "storage": None,
        "models": (
            "## Model dan Strategi Klasifikasi\n\nAlgoritma tabular dibandingkan dengan preprocessing dan split yang sama. Resampling dilakukan hanya pada fitting data. Ordinal dan hierarchy memakai classifier probabilistik. SVM memakai keputusan kelas langsung, bukan argmax probabilitas hasil kalibrasi tambahan."
        ),
        "neural": (
            "## Arsitektur dan Regularisasi\n\nMLP, residual MLP, TabM, dan Transformer tabular mengolah fitur satu-window. GRU dan TCN mengolah riwayat window yang tersedia. Urutan pendek tetap dipertahankan; padding dimasking. Monitor epoch berasal dari training dan terpisah dari validation penilaian. Cross-entropy monitor menentukan checkpoint dan durasi training; kandidat akhir memakai median epoch fold."
        ),
        "search": (
            "## Pencarian Konfigurasi\n\nScreening memberi kesempatan pada keluarga model yang berbeda. Tuning TPE memperdalam kandidat sesuai anggaran eksperimen. Trial yang tidak menjanjikan dapat dipruning setelah pemanasan. Kandidat shortlist harus menyelesaikan lima fold dan pengulangan pembagian sebelum dinilai pada validation akhir."
        ),
        "evaluation": (
            "## Pemilihan Model dan Evaluasi\n\nModel individual dibandingkan dengan ensemble yang sesuai jenis skornya. Stacking dilatih menggunakan prediksi out-of-fold Train. Pemenang per-window dan model utama dikunci dari validation akhir sebelum test dibuka. Model test identik dengan model pilihan, tanpa refit setelah seleksi."
        ),
    }
    for name in (
        "imports",
        "data",
        "features",
        "storage",
        "models",
        "neural",
        "search",
        "evaluation",
    ):
        if name == "search" and experiment == "exp02":
            notes[name] += (
                " Durasi preprocessing, fitting, prediksi, dan checkpoint dicatat terpisah. Setiap fold mencetak identitas kandidat dan waktu berjalan; pesan elapsed muncul setiap 30 detik selama fitting. Status konvergensi SVM ikut disimpan tanpa membatasi iterasi solver."
            )
        if name == "models" and experiment in ("exp03", "exp04"):
            notes[name] = (
                "## Klasifikasi dan Model Pembanding\n\nOutput model memiliki tiga kelas dalam urutan Low, Medium, dan High. "
                "Dummy memakai kelas mayoritas Train sebagai kontrol. Probabilitas, skor keputusan, dan proporsi suara "
                "dibedakan saat pemilihan ensemble."
            )
        if notes[name] and not (name == "neural" and experiment in ("exp01", "exp02")):
            cells.append(cell("markdown", name + "_note", notes[name]))
        source = (HERE / f"{name}.py").read_text(encoding="utf-8")
        cells.append(cell("code", name, source))
    cells += [
        cell(
            "markdown",
            "distribution_note",
            "## Pemeriksaan Input\n\nDistribusi label, jumlah data sesudah purge, dan kelengkapan fitur diperiksa sebelum training. Ketiga CSV harus berasal dari release dataset yang sama.",
        ),
        cell(
            "code",
            "prepare",
            """views, split_manifest, split_counts, dataset_hashes = read_datasets(DATASET_ROOT)
display(pd.DataFrame({f'{w}s': d.cognitive_load_label.value_counts().reindex(LABELS, fill_value=0) for w, d in views.items()}))
display(split_counts[split_counts.partition.eq('_split')])
print('Hash split:', digest(split_manifest))
for _frame in views.values():
    if EXPERIMENT in ('exp03', 'exp04'):
        for _r in range(2):
            for _f in range(5):
                monitor_indices(_frame, np.flatnonzero(_frame[f'_cv_{_r}_{_f}'].eq('train')))
store = start_run(PROJECT_ROOT, EXPERIMENT, dataset_hashes, split_manifest, RESUME_DIR)
store.publish('split_counts.csv', csv_bytes(split_counts))
""",
        ),
        cell(
            "markdown",
            "training_note",
            "## Training dan Validation Internal\n\nTabel kandidat menggunakan prediksi validation internal, bukan skor test. Checkpoint disimpan setelah fold selesai. Kolom training accuracy membantu memeriksa gap generalisasi.",
        ),
        cell("code", "training", "shortlists = run_search(store, views, EXPERIMENT)"),
        cell(
            "markdown",
            "selection_note",
            "## Validation Akhir\n\nShortlist dilatih pada seluruh Train. Model individual, ensemble seed neural, voting, dan stacking dibandingkan pada validation akhir. Pilihan dikunci sebelum evaluasi test.",
        ),
        cell(
            "code",
            "selection_run",
            'selection = select_models(store, views, shortlists, EXPERIMENT)\nprint("Model utama pilihan validation:", selection["primary"])',
        ),
        cell(
            "markdown",
            "test_note",
            "## Hasil Test\n\nAccuracy, Macro-F1, balanced accuracy, precision, recall, dan confusion matrix menilai pemenang yang telah dikunci. Pada eksperimen riwayat, kontrol ditetapkan dari validation dan dibandingkan pada endpoint test yang sama.",
        ),
        cell(
            "code", "test_run", "test_summary = evaluate_test(store, views, selection)"
        ),
        cell(
            "markdown",
            "inference_note",
            "## Prediksi dengan Model Tersimpan\n\nMuat bundle dengan definisi pipeline dan environment yang sama. Model riwayat memerlukan urutan sensor sebelumnya; model satu-window hanya memerlukan fitur saat ini. Gunakan hanya model terpercaya.",
        ),
        cell(
            "code",
            "inference",
            """_chosen = selection['primary']
_record = store.read(_chosen)
_model = unpack_model(store.artifact(_chosen, 'model.pkl'))
_frame = views[_record['window']]
_ids = np.flatnonzero(_frame._split.eq('test'))[:3]
_output = predict_model(_model, _frame, _ids)
display(pd.DataFrame({'window_id': _frame.iloc[_ids].window_id.to_numpy(), 'prediction': np.asarray(LABELS)[_output['labels']]}))
""",
        ),
    ]
    fingerprint = hashlib.sha256(
        "".join(
            "".join(c["source"]) for c in cells if c["cell_type"] == "code"
        ).encode()
    ).hexdigest()
    imports = next(c for c in cells if c["id"] == "imports")
    imports["source"].append(f'\nIMPLEMENTATION_HASH = "{fingerprint}"\n')
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
            "colab": {"provenance": []},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


for number in (2,):
    experiment = f"exp0{number}"
    destination = ROOT / f"models/batch-04/{experiment}/EyeHeartCognitiveLoad.ipynb"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(build(experiment), ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    print(destination.relative_to(ROOT))
