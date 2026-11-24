# Dataset v3 untuk Batch 06

Gunakan `window_15s_step_5s/good_acceptable.csv` sebagai dataset utama. Tim menyiapkan empat CSV pendamping untuk pembagian data, pelacakan episode, dan pembobotan Train. Notebook dapat membaca kelimanya melalui path Drive tanpa modul `.py` atau berkas `.json` tambahan.

## Isi paket

| File | Baris | Fungsi | Kunci penghubung |
|---|---:|---|---|
| `window_15s_step_5s/good_acceptable.csv` | 1.193 | Fitur sensor dan target yang akan dimodelkan | `window_id`, `participant_id` |
| `participant_splits.csv` | 22 | Pembagian peserta ke Train, validation, dan Test | `participant_id` |
| `episode_labels.csv` | 358 | Bukti waktu aktif, kesalahan manual, kelayakan referensi, label, dan alasan keputusan setiap episode | `episode_id`, `participant_id`, `item_key` |
| `window_15s_step_5s/window_assignments.csv` | 1.433 | Pemetaan window berlabel ke episode dan bagian data; mencakup window kualitas poor yang tidak masuk dataset utama | `window_id`, `episode_id` |
| `window_15s_step_5s/train_sample_weights.csv` | 744 | Bobot untuk setiap baris Train pada dataset utama | `window_id` |

Jumlah baris antarfile memang berbeda. `participant_splits.csv` mencakup seluruh peserta pada cohort; dataset utama memuat 21 peserta yang mempunyai window layak berlabel. `episode_labels.csv` mencakup episode tanpa label agar bukti keputusan dan calon referensi tetap tersedia. Gunakan dataset utama sebagai dasar penggabungan, lalu cari metadata pendamping melalui kunci yang sesuai. Jangan menggabungkan berdasarkan nomor atau posisi baris.

## Komposisi dataset utama

Window mempunyai durasi 15 detik dan bergeser setiap 5 detik. Window yang bertumpang tindih berbagi sebagian rekaman sensor.

| Bagian | Low | Medium | High | Total |
|---|---:|---:|---:|---:|
| Train | 54 | 134 | 556 | 744 |
| validation | 14 | 53 | 151 | 218 |
| Test | 62 | 50 | 119 | 231 |
| Total | 130 | 237 | 826 | 1.193 |

Tim menetapkan 16 peserta sebagai Train, 3 sebagai validation, dan 3 sebagai Test. Seluruh rekaman orang yang sama mengikuti bagian yang sama. Pada dataset utama, setiap kelas mempunyai minimal 3 peserta Train dan 2 peserta pada validation/Test. Batas ini memeriksa cakupan teknis evaluasi; jumlah baris overlap tidak menggantikan jumlah peserta atau episode independen.

## Target dan fitur model

Target: `cognitive_load_label`, dengan nilai `Low`, `Medium`, dan `High`.

Label v3 mengikuti kesesuaian waktu aktif terhadap referensi soal yang sama dan pola kesalahan manual. Tim tidak memakai difficulty atau nilai sensor untuk menentukan label ini. V3 merupakan proxy perilaku, belum pengukuran CL yang tervalidasi secara independen.

Gunakan sepuluh fitur core berikut sebagai masukan awal classifier:

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

Kolom iris menyatakan geometri landmark iris, bukan diameter pupil. Dataset juga memuat metadata dan fitur tambahan untuk audit. Pilih fitur melalui daftar di atas, bukan dengan mengambil seluruh kolom numerik. Keluarkan identitas, timestamp, difficulty, konteks soal, kualitas, bagian data, bobot, durasi pengerjaan, jumlah kesalahan, dan batas referensi dari input classifier. Metadata tersebut tetap berguna untuk penggabungan, rekonstruksi label, dan evaluasi.

## Penggunaan pada pemodelan

1. Baca dataset utama. Hubungkan `participant_splits.csv` melalui `participant_id` dan `window_assignments.csv` melalui `window_id`. Pastikan keduanya menunjuk bagian data yang sama dan setiap window mempunyai satu episode.
2. Pertahankan pembagian Train, validation, dan Test tersebut. Jangan membagi ulang baris secara acak. Hubungkan `episode_labels.csv` melalui `episode_id` bila memerlukan bukti keputusan atau rekonstruksi referensi.
3. Pelajari preprocessing dari Train, lalu terapkan hasilnya pada validation/Test. Latih classifier dengan sepuluh fitur core dan target v3.
4. Hubungkan `train_sample_weights.csv` melalui `window_id` untuk baris Train. Periksa bahwa setiap baris Train memperoleh tepat satu bobot. Jangan memberi bobot tersebut pada validation/Test atau memasukkannya sebagai fitur.
5. Gunakan validation untuk pemilihan konfigurasi dan Test untuk penilaian akhir. Pertahankan distribusi validation/Test tanpa resampling atau pemangkasan kelas.

Bobot Train menyetarakan kontribusi kelas, peserta dalam kelas, lalu episode dalam peserta. Episode panjang tidak mendapat pengaruh besar hanya karena menghasilkan banyak window. Rata-rata bobot adalah 1; total bobot tiap kelas adalah 248 untuk 744 baris Train. Paket ini menggunakan full Train dengan bobot, sehingga tidak menyertakan `train_balanced.csv`.

### Jika menggunakan cross-validation di dalam Train

Referensi label hanya boleh berasal dari inner-Train, dengan mengecualikan peserta yang sedang dilabeli, paparan ulang yang diketahui, serta episode yang tidak memenuhi syarat referensi. Gunakan bukti episode dalam `episode_labels.csv` untuk menghitung ulang referensi dan label pada setiap fold. Hasil kosong atau referensi kurang tetap tanpa label. Sesuaikan window yang dipakai melalui `episode_id`, lalu hitung ulang bobot dari inner-Train.

`train_sample_weights.csv` dan label pada CSV utama mengikuti pembagian Train akhir yang dibekukan. Jangan memakainya sebagai hasil perhitungan inner-fold. Keperluan ini memerlukan implementasi pada notebook, meskipun paket input tetap berupa CSV.

## Sumber dan pembaruan

Tim menyalin kelima CSV dari `analysis_sessions/dataset_fusion/output/v3_behavioral_proxy` pada 24 November 2026. Tim mempertahankan nilai sensor, label, pembagian, dan bobot sesuai sumber dengan kebijakan `behavioral_concordance_v2`.

Laporan sumber: [QUALITY_REPORT.md](../../../../analysis_sessions/dataset_fusion/output/v3_behavioral_proxy/QUALITY_REPORT.md). Tautan ini mengikuti susunan repository lokal; salinan laporan tidak termasuk paket Drive ini.

Setelah sumber berubah, salin ulang kelima CSV bersama-sama dan perbarui jumlah pada panduan ini. Jangan mencampur dataset utama dari rebuild baru dengan pembagian, episode, atau bobot dari rebuild lama. Tim belum menjalankan training Batch 06 pada tahap penyiapan paket ini.
