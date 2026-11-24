# Dataset Batch 06

Batch 06 memiliki dua paket, masing-masing dengan window 15 detik dan step 5 detik. Pilih paket sesuai target eksperimen.

| Paket | Target | Window layak | Pembagian dan bobot |
|---|---|---:|---|
| [v3_behavioral_proxy](v3_behavioral_proxy/README.md) | `cognitive_load_label`: Low, Medium, High berdasarkan waktu aktif dan kesalahan | 1.193 | Mengikuti pembagian peserta dan bobot Train v3 |
| [v4_self_reported_rating](v4_self_reported_rating/README.md) | `difficulty_rating`: angka 1–10 yang dipilih peserta | 1.693 | Ditentukan dalam pipeline v4 |

Kedua paket memakai sepuluh fitur inti yang sama: RMSSD, SDNN, laju kedipan, rata-rata dan simpangan baku EAR kiri/kanan, rata-rata dan simpangan baku estimasi ukuran iris, serta rata-rata perubahan ukuran iris. Daftar nama kolom tersedia di README masing-masing paket. Jalankan v3 dan v4 sebagai eksperimen terpisah. V4 tidak memakai pembagian atau bobot v3.

## Sebaran label

Persentase memakai jumlah baris `good_acceptable.csv` pada masing-masing paket sebagai penyebut. Kolom episode dan peserta menghitung ID unik yang terwakili pada setiap label. Seorang peserta dapat muncul pada lebih dari satu label, sehingga jumlah peserta antarlabel tidak dijumlahkan. Persentase bisa sedikit berbeda dari 100% setelah pembulatan.

### V3: proxy cognitive load

| Label | Window | % window | Episode | Peserta |
|---|---:|---:|---:|---:|
| Low | 130 | 10,90% | 28 | 13 |
| Medium | 237 | 19,87% | 16 | 10 |
| High | 826 | 69,24% | 22 | 14 |

High mencakup 69,24% dari 1.193 window, meski jumlah episodenya lebih sedikit daripada Low. Bobot Train v3 menyetarakan kontribusi kelas, peserta, dan episode saat training tanpa mengubah jumlah baris CSV utama.

### V4: rating kesulitan peserta

| Rating | Window | % window | Episode | Peserta |
|---|---:|---:|---:|---:|
| 1 | 186 | 10,99% | 31 | 9 |
| 2 | 104 | 6,14% | 14 | 8 |
| 3 | 222 | 13,11% | 21 | 10 |
| 4 | 163 | 9,63% | 14 | 8 |
| 5 | 110 | 6,50% | 12 | 10 |
| 6 | 201 | 11,87% | 11 | 9 |
| 7 | 251 | 14,83% | 7 | 5 |
| 8 | 52 | 3,07% | 3 | 3 |
| 9 | 202 | 11,93% | 4 | 4 |
| 10 | 202 | 11,93% | 9 | 6 |

Semua rating 1–10 muncul pada 1.693 window. Rating 8 hanya berasal dari tiga episode dan rating 9 dari empat episode, meskipun rating 9 punya 202 window. Window yang bertumpang tindih dari satu episode berbagi satu rating; periksa juga dukungan peserta dan episode saat merancang evaluasi.

## Fungsi file CSV

Path file pada tabel relatif terhadap folder paket masing-masing.

| Paket | File | Baris | Fungsi |
|---|---|---:|---|
| V3 | `window_15s_step_5s/good_acceptable.csv` | 1.193 | Dataset utama: fitur sensor, metadata, dan target `cognitive_load_label`. |
| V3 | `participant_splits.csv` | 22 | Menentukan Train, validation, dan Test berdasarkan `participant_id`. Pertahankan pembagian ini untuk penggunaan label v3 yang sudah dibangun. |
| V3 | `episode_labels.csv` | 358 | Menyimpan bukti label per `episode_id`: waktu aktif, kesalahan manual, dan referensi, termasuk episode tanpa label. Gunakan untuk audit atau menghitung ulang referensi pada cross-validation di dalam Train. |
| V3 | `window_15s_step_5s/window_assignments.csv` | 1.433 | Menghubungkan `window_id` ke `episode_id` dan bagian data. Tabel ini juga mencakup window berlabel berkualitas `poor` yang tidak masuk dataset utama. |
| V3 | `window_15s_step_5s/train_sample_weights.csv` | 744 | Menyediakan bobot per `window_id` untuk baris Train pada dataset utama. Bobot tidak digunakan sebagai fitur atau diterapkan pada validation/Test. |
| V4 | `window_15s_step_5s/good_acceptable.csv` | 1.693 | Dataset utama: fitur sensor, metadata, dan target `difficulty_rating` bernilai 1–10. |
| V4 | `episode_ratings.csv` | 358 | Menyimpan rating asli, submission sumber, dan status pencocokan per `episode_id`. Episode tanpa rating tetap tercatat untuk audit. |
| V4 | `window_15s_step_5s/window_assignments.csv` | 2.082 | Menghubungkan `window_id` ke `episode_id` dan ratingnya. Tabel juga mencakup window `poor` dan tidak menetapkan split. |

Mulai penggabungan dari `good_acceptable.csv` di paket yang dipilih agar window di luar dataset utama tidak ikut masuk. Gunakan `window_id` untuk pemetaan atau bobot, `episode_id` untuk bukti label, dan `participant_id` untuk pengelompokan atau pembagian v3. Jangan menggabungkan berdasarkan urutan baris. Untuk v4, tentukan pembagian di pipeline pemodelan dan hitung bobot dari data Train v4 jika diperlukan.

## Lokasi dataset

Arahkan `DATASET_ROOT` ke folder yang berisi `v3_behavioral_proxy` dan `v4_self_reported_rating`. Lokasi ditentukan oleh lingkungan yang dipakai:

- **Lokal:** jika proses berjalan dari root repository, gunakan `models/batch-06/dataset`. Jika direktori kerja berbeda, gunakan path relatif yang sesuai atau path absolut.
- **Colab:** gunakan lokasi folder dataset setelah tersedia di runtime, misalnya dari repository yang disalin atau Drive yang sudah dipasang. Tidak ada path Drive wajib.

Di kedua lingkungan, lokasi dataset utama relatif terhadap `DATASET_ROOT` tetap sama:

```text
v3_behavioral_proxy/window_15s_step_5s/good_acceptable.csv
v4_self_reported_rating/window_15s_step_5s/good_acceptable.csv
```

Pertahankan susunan folder saat memindahkan paket. Semua CSV berasal dari bundle masing-masing di `analysis_sessions/dataset_fusion/output/`. Setelah rebuild, perbarui paket yang bersangkutan secara utuh: lima CSV untuk v3 atau tiga CSV untuk v4. Pipeline dapat membaca CSV langsung tanpa file `.py` atau `.json` builder.
