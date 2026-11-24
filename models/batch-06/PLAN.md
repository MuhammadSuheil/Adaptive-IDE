# Rencana Eksperimen Batch 06 (`v3_behavioral_proxy` & `v4_self_reported_rating`)

Tanggal: 24 November 2026.

## 1. Latar Belakang dan Tujuan

Batch 05 menyimpulkan bahwa model terkunci pada label protokol `v2` hanya mencapai Test accuracy **37,92%** (Macro-F1 **37,59%**), jauh di bawah sasaran 70%, dengan temuan utama dominasi variasi antarsesi ($\eta^2$ sesi 71%–92%) dan ketiadaan validasi independen terhadap beban mental.

Batch 06 menguji dua paket dataset baru pada [dataset/DATASET_README.md](dataset/DATASET_README.md) sebagai dua eksperimen terpisah (window 15 detik, step 5 detik):

1. **`exp01_v3_behavioral_proxy`** — Target `cognitive_load_label` (`Low`, `Medium`, `High`) dari rubrik kesesuaian perilaku (`behavioral_concordance_v2`).
2. **`exp02_v4_self_reported_rating`** — Target `difficulty_rating` (skala 1–10 yang benar-benar dipilih peserta), dijalankan pada tiga formulasi terstruktur: regresi/ordinal 1–10, klasifikasi 3-kelas (`1–3` / `4–6` / `7–10`), serta biner ekstrem (`1–3` vs `7–10`) dan diagnostik kalibrasi dalam-peserta (*within-participant episode-disjoint*).

Kedua eksperimen dirancang sebagai notebook mandiri yang dapat dijalankan **Run All** baik secara lokal maupun di Google Colab tanpa ketergantungan modul `.py` eksternal.

---

## 2. Kontrak Data dan Aturan Anti-Kebocoran

### Prinsip Umum (V3 dan V4)
- **Sumber read-only:** Membaca langsung dari `models/batch-06/dataset/` (lokal) atau path Drive yang dikonfigurasi di `DATASET_ROOT`, dengan verifikasi SHA-256 seluruh CSV pada awal run.
- **Penggabungan wajib lewat kunci:** Mulai dari `good_acceptable.csv`, gabungkan dengan `window_assignments.csv` lewat `window_id`, metadata episode lewat `episode_id`, dan split peserta lewat `participant_id`. Tidak pernah menggabungkan berdasarkan urutan baris.
- **Fitur sensor murni:** Identitas, timestamp, durasi pengerjaan, jumlah kesalahan, batas referensi, `difficulty_level` protokol, `quality_tier`, `split`, dan `sample_weight` **dilarang keras** masuk ke `X`.
- **Evaluasi dua tingkat:** Karena window 15 detik dengan step 5 detik bertumpang tindih, setiap model dilaporkan pada **tingkat window** dan **tingkat episode** (agregasi rata-rata probabilitas/prediksi per `episode_id`).
- **Kunci model sebelum Test:** Pemilihan model dilakukan sebelum menyentuh Test. Hanya **satu konfigurasi pemenang terkunci** (ditambah kontrol `Dummy`) yang dievaluasi pada Test; hasil Test tidak pernah dipakai untuk mengganti pemenang.

### Aturan Khusus V3 (`exp01_v3_behavioral_proxy`)
- **Pertahankan pembagian peserta beku** dari `participant_splits.csv` (16 Train, 3 validation, 3 Test; 21 peserta aktif di `good_acceptable.csv`).
- **Inner-CV bebas kebocoran di dalam Train:** Saat mengevaluasi kandidat lewat 5-fold `GroupKFold` per `participant_id` di dalam Train, referensi waktu (`behavioral_concordance_v2`, minimal 5 peserta pembanding, bebas paparan ulang, tanpa self-reference) dan label episode **dihitung ulang dari `episode_labels.csv` menggunakan inner-Train saja** pada setiap fold. Bobot sampel inner-Train juga dihitung ulang pada setiap fold.
- **Ablasi bobot Train:** Karena `train_sample_weights.csv` memiliki rasio bobot maks/min 214,3× (Kish ESS kelas Low = 13,9), eksperimen membandingkan secara terkontrol:
  1. `v3_full_weights`: bobot resmi `train_sample_weights.csv` (atau rekomputasi identik di inner-fold),
  2. `v3_clipped_weights`: bobot hierarkis dipangkas pada batas atas `5.0` lalu dinormalisasi ulang agar tiap kelas tetap berjumlah $N/3$,
  3. `class_balanced`: bobot keseimbangan kelas murni ($N / (3 N_c)$).
  Bobot hanya diterapkan saat fitting Train, tidak pernah pada validation atau Test.
- **Seleksi:** Pemenang dipilih dari **Validation** (dengan pemeriksaan kewajaran pada inner-CV OOF), memprioritaskan kandidat yang konvergen, prediksi finite, dan recall positif pada ketiga kelas; peringkat mengikuti Macro-F1 episode, Macro-F1 window, balanced accuracy, dan kompleksitas model lebih rendah.

### Aturan Khusus V4 (`exp02_v4_self_reported_rating`)
- **Independen dari V3:** Tidak memakai split atau bobot V3.
- **Pembagian peserta deterministik bebas bocor:** 17 peserta dibagi menjadi **Development (13 peserta)** dan **Held-Out Test (4 peserta)** berdasarkan cakupan metadata kelas rating saja (tanpa membaca nilai sensor atau skor model), memastikan ketiga kelas rating (`1–3`, `4–6`, `7–10`) terwakili oleh minimal 3 peserta pada Held-Out Test.
- **Seleksi via CV peserta di dalam Development:** Seluruh perbandingan fitur, normalisasi, dan keluarga model dijalankan dengan 5-fold `GroupKFold` berbasis `participant_id` di dalam 13 peserta Development. Pemenang tiap track dikunci dari OOF Development sebelum dievaluasi sekali pada 4 peserta Held-Out Test.
- **Diagnostik kalibrasi dalam-peserta (terpisah dari benchmark utama):** Pada peserta yang memiliki minimal 4 episode dengan variasi rating (atau minimal 2 episode Low dan 2 episode High), dijalankan CV **episode-disjoint** (seluruh window satu episode tetap satu fold) untuk menjawab pertanyaan penelitian penting: apakah sinyal fisiologis muncul secara konsisten di dalam individu yang sama meski lemah lintas-peserta?

---

## 3. Profil Fitur dan Normalisasi

| Profil Fitur | Kolom | Alasan Uji |
|---|---|---|
| `core10` | 10 fitur inti sesuai README | Baseline resmi paket Batch 06 |
| `core9` | `core10` tanpa `iris_size_delta_mean` | Menghapus fitur yang terbukti 100% redundan affine ($R^2 = 1{,}0$) di setiap sesi |
| `core9_head4` | `core9` + `head_pitch_mean/std`, `head_yaw_mean/std` | Menguji tambahan sinyal postur kepala (0% missing) |
| `core9_head4_base` (V4) | `core9_head4` + `rmssd_baseline_ratio` | Menguji rasio HRV terhadap baseline kalibrasi (imputasi median Train untuk 5,3% missing) |

| Mode Normalisasi | Cara Kerja | Status Operasional |
|---|---|---|
| `global_robust` | Median & IQR dihitung dari fitting fold/Train saja | Siap pakai lintas-peserta tanpa kalibrasi sesi |
| `causal_session_robust` | Expanding median & IQR hanya dari window sebelumnya pada sesi yang sama (fallback global saat $<3$ window) | 100% kausal, dapat dipakai *online* di Adaptive-IDE tanpa melihat masa depan |
| `unsupervised_session_robust` | Median & IQR per sesi dari fitur sensor `X` saja (tanpa label) | Referensi *upper-bound* kalibrasi penuh satu sesi |

Selain fitur per-window, kedua eksperimen juga menguji cabang **ringkasan episode** (*episode-level summary*: median, IQR, dan tren/slope fitur sensor sepanjang episode) untuk memeriksa apakah agregasi temporal satu soal lebih informatif daripada snapshot 15 detik tunggal.

---

## 4. Keluarga Model

- **Kontrol:** `DummyClassifier` (most frequent / stratified) atau `DummyRegressor` (mean / median).
- **Linear reguler:** `LogisticRegression` (L2, `C=0.5`, `max_iter=5000`) / `Ridge` ($\alpha = 10$).
- **Ensemble pohon:** `ExtraTrees` (200 trees, `min_samples_leaf=5`, `max_features=0.8`) dan `HistGradientBoosting` (`max_leaf_nodes=15`, `min_samples_leaf=10`, `l2_regularization=1.0`).
- **Neural terarah (PyTorch):** MLP 2-layer kecil (`32 -> 16`, LayerNorm, SiLU, Dropout 0.3, AdamW weight decay `1e-2`, CosineAnnealingLR, early stopping pada monitor peserta-terpisah di dalam fitting pool), dirata-ratakan di 3 seed (`42, 43, 44`).

---

## 5. Artefak Output

Setiap eksperimen menyimpan artefak terstruktur ke `models/batch-06/outputs/<exp_name>/`:
- `run.json` — Metadata lingkungan, hash SHA-256 dataset, konfigurasi, dan ringkasan kunci pemenang.
- `cv_leaderboard.csv` — Seluruh kandidat pada OOF/CV (beserta metrik window dan episode).
- `validation_leaderboard.csv` (V3) — Seluruh kandidat pada Validation V3 sebelum penguncian Test.
- `test_evaluation.json` & `test_predictions.csv` — Evaluasi akhir satu model terkunci (+ kontrol Dummy) pada Test.
- `episode_diagnostics.csv` — Analisis kesalahan dan performa per episode / peserta.
- `figures/*.png` — Grafik perbandingan kandidat, confusion matrix / scatter prediksi, dan diagnostik per peserta.
- `RESULTS.md` — Laporan ringkas hasil eksperimen.
