# Dataset v4 untuk Batch 06

Gunakan `window_15s_step_5s/good_acceptable.csv` sebagai dataset utama. CSV ini berisi 1693 window layak dari 126 episode dan 17 peserta. Target `difficulty_rating` adalah angka 1–10 yang dipilih peserta untuk kesulitan soal.

| File | Fungsi |
|---|---|
| `window_15s_step_5s/good_acceptable.csv` | Fitur eye tracking dan HRV serta target |
| `window_15s_step_5s/window_assignments.csv` | Hubungan window berlabel ke episode; mencakup window poor untuk audit |
| `episode_ratings.csv` | Bukti rating per episode, termasuk episode tanpa label |

Mulai penggabungan dari `good_acceptable.csv`. Gunakan `window_id` untuk mencari episode pada `window_assignments.csv`, lalu `episode_id` untuk mencari bukti pada `episode_ratings.csv`. Gunakan `participant_id` sebagai kunci pengelompokan. Jangan gabungkan file menurut posisi baris.

Fitur awal classifier:

```text
rmssd_ms
sdnn_ms
blink_rate_per_minute
ear_left_mean
ear_left_std
ear_right_mean
ear_right_std
iris_size_estimate_mean
iris_size_estimate_std
iris_size_delta_mean
```

Skala iris menyatakan geometri landmark, bukan diameter pupil. Identitas, waktu, soal, difficulty protokol, outcome, dan data audit tidak masuk daftar fitur awal. Paket ini tidak menetapkan split atau bobot; tentukan keduanya dalam pipeline v4. Jangan memakai split atau bobot v3 untuk v4.

| Rating | Window layak |
|---:|---:|
| 1 | 186 |
| 2 | 104 |
| 3 | 222 |
| 4 | 163 |
| 5 | 110 |
| 6 | 201 |
| 7 | 251 |
| 8 | 52 |
| 9 | 202 |
| 10 | 202 |

Satu episode menyumbang satu rating, meski muncul pada banyak window. Periksa jumlah peserta dan episode per rating sebelum membagi data. Soal tanpa angka rating tidak masuk dataset utama; timeout dengan angka rating tetap memakai pilihan peserta.

Laporan sumber di repository: [QUALITY_REPORT.md](../../../../analysis_sessions/dataset_fusion/output/v4_self_reported_rating/QUALITY_REPORT.md). Salin ketiga CSV bersama-sama ketika memperbarui paket dari bundle sumber yang sama.

| File | Baris data | SHA-256 |
|---|---:|---|
| `episode_ratings.csv` | 358 | `9f7ab8997fe84549e222c82f4ae0c968b07e13d53464da48444d383138aef686` |
| `window_15s_step_5s/good_acceptable.csv` | 1693 | `6c2febb856e5986e2d9f76b64c0757b3476547bdcf42ccfcff1f02a3b69c2c79` |
| `window_15s_step_5s/window_assignments.csv` | 2082 | `aeb41a5a730548039fd40caf1bd040ecbd11099cffe209bb7d27e01159ad96be` |
