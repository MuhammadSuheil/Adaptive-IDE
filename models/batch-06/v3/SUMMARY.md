# Hasil Batch 06

Run Colab `baseline_20261123T192338Z_8d8cee5e` selesai pada 24 November 2026 WIB dan telah diperiksa dari notebook serta ZIP output. Seluruh 165 fitting CV dan empat fitting validation selesai. Baseline memberi perbaikan terbatas pada beberapa model; belum ada bukti pencapaian accuracy 70%. Test tidak dievaluasi.

[Diagnosis](DIAGNOSIS.md) menjelaskan keterbatasan validation eye, variasi manfaat antarorang, serta opsi eksperimen berikutnya. [Laporan run asli](outputs/exp01/baseline_20261123T192338Z_8d8cee5e/RESULTS.md) dipertahankan tanpa perubahan.

## Data dan evaluasi

Tiga hash CSV Drive cocok dengan dataset lokal dan snapshot batch-05. Split historis tetap dipertahankan. Training memakai sepuluh fitur core pada window 25 detik, step 5 detik. OOF utama menahan seluruh data satu orang dari fitting di dalam Train; kontrol sekunder memakai lima fold blok waktu dengan guard 45 detik.

| Subset | Orang pada Train | Baris Train | Baris validation | Kelas validation |
|---|---:|---:|---:|---|
| Baseline RMSSD | 19 | 2.454 | 721 | 198 Low, 269 Medium, 254 High |
| RMSSD dan acuan eye | 4 | 671 | 99 | 43 Low, **0 Medium**, 56 High |

Raw dan baseline-relative memakai baris/fold identik dalam masing-masing subset. RMSSD dinyatakan relatif terhadap baseline recorder; cabang eye juga memakai satu window pre_task awal. SDNN tetap mentah karena CSV tidak menyimpan baseline SDNN independen. Acuan pre_task tidak dianggap baseline istirahat fisiologis tervalidasi.

## Hasil OOF peserta yang ditahan

| Subset | Model | Accuracy raw | Accuracy relatif | Macro-F1 raw | Macro-F1 relatif |
|---|---|---:|---:|---:|---:|
| RMSSD | Logistic Regression | 37,73% | **39,41%** | 35,59% | **38,15%** |
| RMSSD | Extra Trees | 35,66% | 37,33% | 34,03% | 35,10% |
| RMSSD dan eye | Logistic Regression | 34,43% | 31,59% | 32,56% | 26,78% |
| RMSSD dan eye | Extra Trees | 35,77% | **41,43%** | 33,96% | **39,85%** |

Dummy mempunyai accuracy OOF 26,12% pada subset RMSSD dan 39,34% pada subset eye. Extra Trees relatif pada eye menambah 14 prediksi benar dibanding Dummy, atau 2,09 poin accuracy; Macro-F1-nya lebih tinggi karena prediksi tidak hanya mengenali Low.

Pada RMSSD Logistic Regression, baseline menambah 41 prediksi benar pada 2.454 window, atau 1,67 poin accuracy. Namun, rata-rata accuracy dengan bobot sama untuk setiap orang turun dari 40,51% menjadi 38,61%. Sebelas orang membaik, enam turun, dua tetap; beberapa penurunan besar terjadi pada orang dengan sedikit window. Manfaat agregat belum menjadi perbaikan yang merata.

Pada eye Extra Trees, accuracy naik untuk dua orang, tetap untuk satu, dan turun untuk satu. Macro-F1 naik pada keempat orang. Sinyal manfaat tersebut terbatas pada empat sumber dan pada keluarga model tertentu; Logistic Regression pada subset yang sama justru memburuk.

## Validation model yang dipilih melalui OOF

| Subset | Model | Accuracy raw | Accuracy relatif | Macro-F1 raw | Macro-F1 relatif |
|---|---|---:|---:|---:|---:|
| RMSSD, 721 window | Logistic Regression | 36,34% | 39,39% | 36,11% | 39,24% |
| RMSSD dan eye, 99 window | Extra Trees | 52,53% | 65,66% | 33,46% | 44,58% |

**Validation eye tidak mempunyai contoh Medium.** Recall Medium nol pada tabel otomatis berasal dari support nol dan aturan metrik, bukan bukti bahwa seluruh contoh Medium salah. Model relatif tetap memprediksi tujuh window sebagai Medium. Angka 65,66% merupakan 65 benar dari 99 window Low/High, bukan keberhasilan tiga kelas pada pengguna baru.

Validation eye hanya mencakup tiga orang yang juga ada di Train: Baithar seluruhnya High, Mon seluruhnya Low, serta Fabian campuran Low/High. Perbaikan berasal dari Mon, 9 menjadi 23 benar; Baithar turun 39 menjadi 38 benar dan Fabian tetap 4 benar. Karena itu, kenaikan 13,13 poin validation tidak cukup untuk menyatakan manfaat umum.

## Kontrol dan keputusan

Pada CV blok waktu, RMSSD relatif menaikkan Macro-F1 Logistic Regression dari 34,48% menjadi 35,86%, tetapi Extra Trees turun 34,55% menjadi 33,56%. Eye Extra Trees naik dari 34,26% menjadi 38,38%; accuracy-nya membaik pada lima fold waktu. Fold tersebut tetap berasal dari empat orang yang sama, bukan lima sampel populasi baru.

Baseline layak dipertahankan sebagai kandidat representasi. Hasil sekarang tidak mendukung klaim bahwa normalisasi baseline telah menyelesaikan akurasi rendah, bahwa semua model memperoleh manfaat, atau bahwa 70% sudah dekat. Kontrol berikut yang lebih terarah adalah memisahkan RMSSD-only, eye-only, dan gabungannya pada subset/fold yang persis sama, sambil melaporkan bobot per orang dan dukungan kelas setiap bagian. Eksperimen baru belum dijalankan saat rekap ini.

## Pemeriksaan

Metrik dari 20 tabel OOF, empat tabel validation, dan 165 catatan fold dihitung ulang dan cocok. Tiga hash CSV, empat checksum bundle model, 33 manifest fold, delapan baris perbandingan pasangan, serta 19 referensi baseline diperiksa. Label/identitas prediksi cocok dengan CSV sumber; ID Test tidak muncul pada tabel prediksi. Source cell notebook yang dieksekusi sama dengan notebook yang disiapkan dan tidak mempunyai output exception.

Pemeriksaan membaca prediksi, JSON, checksum, dan notebook. Model joblib tidak dimuat untuk inference dan tidak ada fitting ulang. Nilai fitting accuracy berasal dari catatan run, bukan reproduksi inference baru. [Audit numerik](outputs/exp01/baseline_20261123T192338Z_8d8cee5e/analysis/audit.json), [metrik dengan support kelas](outputs/exp01/baseline_20261123T192338Z_8d8cee5e/analysis/metrics_with_class_support.csv), dan [metrik per orang](outputs/exp01/baseline_20261123T192338Z_8d8cee5e/analysis/participant_metrics.csv) menyimpan bukti rekap.

[Notebook hasil run](outputs/exp01/baseline_20261123T192338Z_8d8cee5e/executed_notebook.ipynb) diarsipkan terpisah; notebook sumber di exp01 tidak ditimpa. Panduan mount dan Run all tetap ada di [README.md](README.md).

## Hasil Exp02

[Run ablasi Exp02](exp02/SUMMARY.md) selesai pada 24 November 2026 WIB. Pada OOF peserta, raw memperoleh accuracy 35,77%, RMSSD-only 38,15%, eye-only 31,74%, dan gabungan 41,43%. Gabungan terbaik untuk OOF orang, sementara RMSSD-only sedikit lebih tinggi pada CV waktu dan validation historis. Validation 66,67% RMSSD-only tetap tidak mempunyai contoh Medium. Hasil raw/gabungan cocok dengan Exp01 pada enam pemeriksaan metrik. Analisis berdasarkan output notebook; ZIP pekerjaan/prediksi Exp02 belum tersedia untuk audit per window.
