# Batch 08 · Model Final Deteksi Overload (V4)

Batch terakhir pemodelan. Target: **overload = rating kesulitan peserta ≥ 7** (V4, 126 episode, 17 peserta). Semua pilihan model (fitur, normalisasi, target, prior auxiliary, regularisasi, agregasi, ensemble/stacking, kalibrasi, ambang) dibuat **di dalam inner-LOPO** setiap outer fold. Hanya prediksi outer yang dilaporkan.

| Dokumen | Isi |
|---|---|
| [SUMMARY.md](SUMMARY.md) | Laporan lengkap: dataset, pipeline, setiap eksperimen, hasil, grafik, dan rekomendasi |
| [DIAGNOSIS.md](DIAGNOSIS.md) | Keterbatasan yang tersisa dan saran setelah Batch 08 |
| [ROUNDS.md](ROUNDS.md) | Riwayat tiga ronde iterasi dan alasan setiap perubahan |
| [Laporan docx untuk ketua peneliti](../reports/Laporan_Eksperimen_Batch_08.docx) | Versi formal dari SUMMARY |

## Dataset yang dipakai di setiap eksperimen

**Semua hasil utama Batch 08 (A-ens, B3, model final) memakai V4.** V3 hanya dipakai di exp04 sebagai pembanding.

| Eksperimen | Dataset target (label) | Data pendukung (bukan label) | Label yang dibentuk di kode |
|---|---|---|---|
| exp01_reference | **V4** `v4_self_reported_rating` (126 episode, 17 peserta) | — | overload = rating ≥ 7 |
| exp02_nested_main | **V4** | v2 `all_data.csv` 15 s: (a) median riwayat sensor sebelum tugas untuk normalisasi *hybrid*; (b) ±4.300 window aktif untuk prior "kesulitan" auxiliary | overload = rating ≥ 7 (sensitivitas ≥ 6 dan ≥ 8) |
| exp03_early_detection | **V4** (memakai prediksi exp02) | — | overload = rating ≥ 7 |
| exp04_v3_crosscheck | **V3** `v3_behavioral_proxy` (bagian 1) dan irisan **V3∩V4** 40 episode (bagian 2) | v2 `all_data.csv` (riwayat dan prior, seperti exp02) | High vs Non-High (Low+Medium), dihitung ulang per fold |

V3 tidak dipakai sebagai target utama karena model sensor tidak dapat mempelajarinya tanpa kebocoran durasi (exp04: AUC LOPO 0,43).

## Isi folder

| Folder | Isi |
|---|---|
| `exp01_reference/` | Notebook (sudah dieksekusi) dan output: reproduksi model terbaik Batch 06 sebagai pembanding, serta pembanding tanpa sensor |
| `exp02_nested_main/` | Notebook dan output tiga ronde. Ronde 3 (`nested_round3_20261127T031955Z_8703732e`) adalah hasil final, termasuk uji permutasi (`permutation_result.json`) |
| `exp03_early_detection/` | Notebook dan output deteksi dini serta simulasi alarm online |
| `exp04_v3_crosscheck/` | Notebook dan output validasi silang V3 dan 40 episode irisan |
| `figures/` | Grafik yang dipakai di SUMMARY dan laporan docx |
| `final_model/` | Model final untuk IDE: `b8_overload_model.json` (parameter), `b8_inference.py` (inferensi numpy), dan `verification.json` |

Setiap folder output berisi prediksi per episode, metrik, `labels_used.csv`, dan `manifest.json`. Manifest mencatat hash SHA-256 input, konfigurasi, dan environment.

## Catatan eksekusi

- **Lokasi eksekusi:** seluruh notebook dijalankan lokal pada 27 November 2026 dengan Python 3.14.6, scikit-learn 1.7.2, pandas 3.1.1, numpy 2.5.5, dan scipy 1.18.1.
- **Notebook mandiri:** setiap notebook berisi seluruh kode yang dibutuhkan (kode bersama diinline). Venv, skrip pembangun notebook, dan skrip bantu eksekusi sudah dihapus setelah run selesai.
- **Reproduksi:** buat environment dengan versi paket di atas, lalu jalankan notebook berurutan exp01 → exp02 → exp03 → exp04, karena exp02 membaca output exp01 dan exp03/exp04 membaca output exp02.
- **Notebook exp02:** hanya menjalankan ronde final (ronde 3). Ronde 1 dan 2 tersimpan sebagai output historis.
- **Kompatibilitas:** notebook juga dapat dijalankan di Colab (mount Drive dan folder `Dataset/` seperti batch sebelumnya).

## Memakai model final

```python
import sys
sys.path.insert(0, "models/batch-08/final_model")
from b8_inference import OverloadModel

model = OverloadModel.load("models/batch-08/final_model/b8_overload_model.json")
p = model.predict_episode(windows, history_reference)   # windows: list dict per window 15 s
alarm = p >= model.threshold                             # ambang 0,525
```

Format input dijelaskan di docstring `b8_inference.py` dan di [SUMMARY.md §8](SUMMARY.md#8-model-final-untuk-ide).
