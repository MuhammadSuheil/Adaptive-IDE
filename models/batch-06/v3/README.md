# Batch 06

## Laporan v3 untuk tim · 25 November 2026

[Unduh laporan v3 dan perbandingan dengan v4 (PDF, 6 halaman)](report/Laporan_v3.pdf). Model gabungan exp11 mencapai 26/37 benar (70,3%) pada LOPO Train, 6/11 (54,5%) pada validasi, dan **11/18 (61,1%) pada Test**. Normalisasi personal exp13 mencapai 25/37 (67,6%), 9/11 (81,8%), dan **9/18 (50,0%)**. Pada tiga peserta Test, kedua resep mendeteksi satu dari dua High. Exp11 memberi 6 alarm palsu dari 16 Non-High dan exp13 memberi 8. Tebakan selalu Non-High benar 16/18 (88,9%) dan melewatkan kedua High.

Evaluasi Test memakai pembagian peserta resmi dan kedua resep yang sudah ditentukan, dengan fitting pada 37 episode Train dan ambang skor nol. Skor validasi kedua model cocok dengan arsip exp13 hingga toleransi 10⁻¹². Untuk mereproduksi, jalankan `python models/batch-06/v3/report/evaluate_test.py` lalu `python models/batch-06/v3/report/build_report.py` dari root repo. [Metrik Test](report/test_metrics.csv) · [Prediksi Test per episode](report/test_episode_predictions.csv)

[Kurva training dan validasi](report/training_curve.png) · [Nilai loss per iterasi](report/training_curve.csv) · [Sebaran episode, window, dan peserta](report/sebaran_data.png) · [Confusion matrix](report/confusion_matrix.png)

## Eksperimen dengan label terbaru

[Exp13 · normalisasi menurut variasi sensor sebelum soal](exp13/README.md) membandingkan akurasi, alarm palsu, dan deteksi High dengan exp11 pada peserta dan episode yang sama. [Notebook Colab](exp13/EyeHeartCognitiveLoad.ipynb)

[Exp12 · uji HRV 15 dan 45 detik pada waktu akhir yang sama](exp12/README.md) melaporkan akurasi, alarm palsu, kesesuaian rating, dan cakupan per kelas. [Notebook Colab](exp12/EyeHeartCognitiveLoad.ipynb)

[Exp11 · median fitur versus rata-rata skor SVM per window](exp11/README.md) membandingkan dua cara memanfaatkan rekaman sensor dengan bobot episode, kernel, dan peserta yang sama. [Notebook Colab](exp11/EyeHeartCognitiveLoad.ipynb)

[Exp10 · transfer metode v4 exp02/exp03 ke v3](exp10/README.md) membandingkan resep Logistic Regression per window dengan kontrol exp09 dan memeriksa kesesuaian skor v3 dengan rating yang diberikan peserta. [Notebook Colab](exp10/EyeHeartCognitiveLoad.ipynb)

[Exp09 · kurva training loss dan evaluasi proxy cognitive load v3](exp09/README.md) menampilkan konvergensi model exp08, pembanding posisi kepala, dan pengurutan High/Non-High di dalam peserta. [Notebook Colab](exp09/EyeHeartCognitiveLoad.ipynb)

[Exp08 · audit alarm palsu High pada model gabungan v3](exp08/README.md) menguji ambang yang dipilih melalui LOPO di dalam Train dan menelusuri tiga alarm palsu yang terkonsentrasi pada satu peserta. [Notebook Colab v3](exp08/EyeHeartCognitiveLoad.ipynb)

[Exp07 · deteksi High pada v3](exp07/README.md) membandingkan tiga kelas, pemetaan Low+Medium menjadi Non-High, dan training biner langsung melalui LOPO di dalam Train serta validation resmi. [Notebook Colab v3](exp07/EyeHeartCognitiveLoad.ipynb)

[Exp06 · audit run terbaru dan hasil pengembangan tanpa merekam ulang](exp06/README.md) menguji referensi sensor sebelum tugas, head pose, dan penambahan episode yang masih memiliki satu modalitas layak. [Notebook Colab](exp06/EyeHeartCognitiveLoad.ipynb) menjalankan kedua kasus dalam satu run.

[Hasil Colab exp03/04 dan percobaan lanjutan exp05](exp05/README.md) memuat evaluasi kedua target pada episode yang sama dan panduan untuk [satu notebook gabungan](exp05/EyeHeartCognitiveLoad.ipynb). Notebook exp05 membaca kedua paket langsung dari `PROJECT_ROOT / 'Dataset'` dan menghasilkan satu ZIP.

[Exp03 untuk proxy perilaku tiga kelas](exp03/EyeHeartCognitiveLoad.ipynb) dan [exp04 untuk rating kesulitan 1–10](exp04/EyeHeartCognitiveLoad.ipynb) memakai paket 15 detik di folder dataset. [Panduan relabeling](relabeling/README.md) memuat alasan pemilihan model, evaluasi per peserta dan episode, serta cara mengembalikan ZIP Colab.

## Eksperimen awal dengan label protokol

Batch ini menguji satu classifier umum dengan fitur relatif terhadap baseline awal sensor. Run Colab pertama selesai dan telah diperiksa: [ringkasan hasil](SUMMARY.md) dan [diagnosis](DIAGNOSIS.md). Baseline memberi manfaat terbatas pada OOF; accuracy validation eye 65,66% berasal dari subset tanpa contoh Medium dan tidak mewakili evaluasi tiga kelas yang lengkap. Test tidak dievaluasi.

## Exp02: pisahkan kontribusi normalisasi

[Notebook exp02](exp02/EyeHeartCognitiveLoad.ipynb) membandingkan raw, RMSSD-only, eye-only, dan gabungan pada empat peserta, baris, serta fold subset eye Exp01 yang sama. [Hasil pertama](exp02/SUMMARY.md) menunjukkan gabungan tertinggi pada OOF peserta, sedangkan eye-only menurunkan accuracy dan Macro-F1 kedua CV. [Panduan exp02](exp02/README.md) menjelaskan mount Drive, budget 45 fitting CV + empat fitting validation, dan pelaporan support kelas. Notebook mandiri; cukup salin notebook dan gunakan tiga CSV yang sama, tanpa menyalin output Exp01 ke Drive.

## Jalankan exp01 di Google Colab

1. Salin [exp01/EyeHeartCognitiveLoad.ipynb](exp01/EyeHeartCognitiveLoad.ipynb) ke `MyDrive/Penelitian-Adaptive-IDE/models/batch-06/exp01/`.
2. Pastikan tiga CSV terbaru berada pada struktur Drive berikut.
3. Buka notebook di Colab dan pilih **Run all**. Mount Drive saat diminta. Runtime CPU cukup; GPU tidak diperlukan.
4. Baca tabel akhir dan `RESULTS.md` pada folder run baru di `models/batch-06/outputs/exp01/`.

```text
MyDrive/Penelitian-Adaptive-IDE/
  Dataset/v2_session_phase/
    window_15s_step_5s/good_acceptable.csv
    window_25s_step_5s/good_acceptable.csv
    window_45s_step_5s/good_acceptable.csv
  models/batch-06/
    exp01/EyeHeartCognitiveLoad.ipynb
    outputs/exp01/                  # dibuat otomatis
```

Root sama dengan batch sebelumnya. Ubah `PROJECT_ROOT` pada cell awal hanya bila nama/lokasi proyek atau shortcut berbeda. Notebook tidak memerlukan file Python pendamping, requirements, sesi mentah, output batch-05, atau file audit terpisah.

## Yang dibandingkan

Training utama memakai window 25 detik dan sepuluh fitur core. CSV 15/45 detik dipakai untuk memeriksa identitas, target endpoint, serta versi data; ketiga panjang window tidak ditumpuk menjadi peserta tambahan.

| Subset | Raw | Baseline-relative |
|---|---|---|
| `hrv_rmssd` | Sepuluh fitur core asli | RMSSD diganti dengan `RMSSD / baseline_rmssd_ms - 1`; sembilan fitur lain tetap |
| `hrv_eye` | Sepuluh fitur core asli pada subset dengan acuan eye awal | RMSSD relatif ditambah transformasi eye terhadap satu window `pre_task` awal 25 detik |

**SDNN tetap mentah.** CSV tidak mempunyai baseline SDNN independen. Mean EAR dan iris dibagi mean referensinya lalu dikurangi satu; standar deviasinya dibagi mean referensi yang sama. Blink rate dikurangi blink rate referensi sehingga baseline nol tetap sah. `iris_size_delta_mean` dipusatkan terhadap nilai referensinya tanpa mengubah satuan. Fitur iris tetap geometri iris, bukan diameter pupil.

Logistic Regression dan Extra Trees mempunyai konfigurasi tetap. Dummy memilih kelas mayoritas fitting. Tidak ada search, balancing, penambahan predictor identitas, penggantian label, atau refit Train + validation. Satu fold evaluasi dapat melatih satu classifier, tetapi tujuan penggunaannya tetap satu model umum; classifier tidak menerima identitas sebagai predictor.

## Baseline dan cakupan

Notebook menyertakan audit batas waktu untuk 23 sesi kanonis yang tersedia saat persiapan. Bukti ini berasal dari `hrv.csv` dan ledger `task_events.csv`; hash sumber tersimpan pada [source_timing_audit.json](source_timing_audit.json) dan payload notebook. Nilai baseline serta coverage tetap dibaca dari CSV Drive dan wajib cocok dengan audit.

Sesi dengan baseline yang bertabrakan dengan awal set/practice pertama dikeluarkan. Baseline RMSSD harus ready, positif, dan coverage minimal 80%. Untuk acuan eye, notebook memilih window `pre_task` lengkap paling awal yang selesai sebelum tugas pertama. Semua interval supervised yang mendukung acuan tersebut dikeluarkan dengan margin satu detik. Tidak ada acuan yang diambil dari tugas pertama sebagai pengganti.

Pada salinan lokal yang diperiksa, dataset 25 detik mencakup 22 sesi dari 21 orang. Baseline RMSSD lolos untuk 19 sesi/orang; acuan gabungan hanya lolos untuk empat: kaka, Fabian, Baithar, dan Mon. Jumlah runtime mengikuti isi CSV Drive. Acuan `pre_task` merupakan referensi sensor sebelum tugas, bukan bukti kondisi istirahat fisiologis.

Jika Drive mempunyai sesi baru atau nilai baseline berubah, notebook melaporkannya dan mengecualikannya dari pembanding yang membutuhkan bukti waktu tersebut. Audit sumber perlu diperbarui sebelum sesi itu diikutkan; notebook tidak menebak waktu baseline dari difficulty atau prediksi. Ini membatasi cakupan bukti yang disertakan, bukan permintaan agar dataset terbaru diganti dengan dataset lama.

## Evaluasi

Jika hash ketiga CSV sama dengan batch-05, split historis dipertahankan. OOF utama menggunakan **Train saja** dan menahan seluruh rekaman satu orang pada tiap fold. Dua rekaman Ramadhan tetap satu grup. Kontrol tambahan memakai lima fold blok batch-05 dengan guard 45 detik.

Jika hash berbeda, notebook membuat split berdasarkan orang menggunakan seed tetap 42/43. Train, validation, dan Test mempunyai orang terpisah; split tidak dicari ulang berdasarkan skor. Sesi yang bukti baselinenya belum tersedia tetap tercatat sebagai pengecualian.

Kandidat raw dan relative masing-masing dipilih melalui Macro-F1 OOF peserta, dengan recall positif pada ketiga kelas, lalu dikunci sebelum validation. Seluruh skor candidate OOF tetap dilaporkan. Test tidak diprediksi. Kalibrasi awal yang tidak berlabel dipisahkan dari endpoint fitting/penilaian, termasuk jika window referensinya dahulu mempunyai assignment historis berbeda.

Membandingkan skor batch-06 dengan skor seluruh dataset batch-05 tidak mengisolasi manfaat baseline, karena subset dan jenis evaluasinya berbeda. Gunakan delta raw-relative dalam subset dan protokol yang sama. Empat orang pada cabang eye hanya cukup untuk eksplorasi awal; kenaikan skor belum membuktikan generalisasi luas atau validitas fisiologis label.

## Hasil dan pengulangan

Setiap **Run all** membuat folder `baseline_<waktu>_<id>` baru. Tidak ada overwrite run lama atau resume otomatis. Jika runtime terputus, hasil parsial tetap tersimpan; Run all berikutnya memulai run baru dengan budget model yang sama.

- `plan.json`: hash CSV, kode, referensi, versi paket, model, serta aturan seleksi.
- `baseline_audit.csv` dan `baseline_references.json`: penerimaan/penolakan baseline serta nilai acuan.
- `split_assignments.csv` dan `*_fold_manifest.json`: identitas endpoint setiap bagian/fold.
- `cv_leaderboard.csv`, `fold_metrics.csv`, dan `oof_predictions.csv`: metrik serta prediksi CV.
- `paired_comparison.csv`: perubahan raw-relative pada baris dan fold identik.
- `model_lock.json`, `validation_metrics.csv`, dan `validation_predictions.csv`: pilihan OOF dan hasil validation.
- `models/*.joblib`: bundle classifier dan scaler terpilih. Inferensi sesi baru tetap memerlukan referensi awal sesuai kontrak representasi; bundle tidak berisi lookup statistik tugas peserta.
- `RESULTS.md`, `figures/`, dan `status.json`: laporan, grafik, serta status akhir.

Bobot `.joblib` mengikuti aturan ignore model terlatih. Metadata, prediksi, laporan, dan grafik mengikuti kebijakan output repositori.

## Pemeliharaan lokal

[baseline_experiment.py](baseline_experiment.py) adalah sumber cell definisi; [build_notebook.py](build_notebook.py) membangun ulang notebook mandiri dan audit waktu dari arsip kanonis. Keduanya bukan dependensi Colab. Setelah mengubah implementasi, jalankan dari root repositori:

```powershell
python models/batch-06/build_notebook.py
python -m unittest discover -s tests -p test_batch06_baseline.py -v
```

[PLAN.md](PLAN.md) menetapkan pertanyaan dan batas eksperimen. [SUMMARY.md](SUMMARY.md) memisahkan kesiapan lokal dari hasil training Drive.
