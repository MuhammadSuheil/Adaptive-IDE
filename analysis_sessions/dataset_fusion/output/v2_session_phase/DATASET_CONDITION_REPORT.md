# Kondisi dataset berdasarkan kelayakan fitur

Dataset utama sudah disaring untuk 10 fitur core. Kualitas posisi gaze diperiksa terpisah; jumlah baris yang besar tidak memastikan pemisahan Low/Medium/High akan kuat.

| Window | Low | Medium | High | Rasio terbesar/terkecil | Gaze tidak layak pada curated |
|---|---:|---:|---:|---:|---:|
| 15s | 1687 | 1763 | 1803 | 1.069 | 4205 |
| 25s | 1549 | 1487 | 1572 | 1.057 | 3699 |
| 45s | 1394 | 966 | 1149 | 1.443 | 2797 |

## Kelayakan kelompok fitur

Kegagalan gaze masih ada pada rekaman asal. Nilainya tidak dianggap telah diperbaiki; penggunaan spasialnya dibatasi dan nilai curated dimasking. Core tidak memakai gaze, head pose, atau eye-state. Seluruh predictor core pada curated tersedia tanpa imputasi. Perbedaan level sensor antarsesi tetap ada dan harus ditangani lewat preprocessing Train atau baseline saat pemodelan.

## Validasi sumber gaze

| Rekaman | Lolos asli | Terverifikasi untuk profil spasial | Alasan |
|---|---|---|---|
| Alif | True | True | saved_validation_verified |
| Andrean | False | False | failed_or_unknown_validation |
| Andreas | False | False | failed_or_unknown_validation |
| Athaya | False | False | failed_or_unknown_validation |
| Baithar | True | True | saved_validation_verified |
| bangsur_easy_medium | False | False | failed_or_unknown_validation |
| Fabian | True | True | saved_validation_verified |
| faqih | False | False | failed_or_unknown_validation |
| ihsan | False | False | failed_or_unknown_validation |
| ikhsan | False | False | failed_or_unknown_validation |
| irfan | False | False | failed_or_unknown_validation |
| kaka | False | False | failed_or_unknown_validation |
| Mon | True | True | saved_validation_verified |
| Rafi | False | False | failed_or_unknown_validation |
| raina | False | False | failed_or_unknown_validation |
| Ramadhan | False | False | failed_or_unknown_validation |
| Ramadhan-2 | False | False | failed_or_unknown_validation |
| sam | False | False | failed_or_unknown_validation |
| Savitri | False | False | failed_or_unknown_validation |
| Shakilla | False | False | failed_or_unknown_validation |
| Siti | False | False | failed_or_unknown_validation |
| Sullivan | False | False | failed_or_unknown_validation |
| Tio | False | False | failed_or_unknown_validation |

## Distribusi sumber dan fase

class_distribution.csv memuat jumlah dan durasi unik per kelas. counts_by_session_label_phase_tier.csv memuat distribusi setiap rekaman/fase. Rating, feedback, dan rest yang diberi Low tetap mengikuti protokol v2; fase tidak dimasukkan sebagai predictor. Window overlap tidak dihitung sebagai pengamatan independen.

## Pembaruan berikutnya

Tambahkan rekaman lengkap ke sessions/saved_session, pastikan ledger dan kalibrasi tersedia, lalu jalankan builder dengan --replace. Kebijakan feature-aware diwarisi dari manifest aktif. Jangan menggabungkan CSV dengan concat manual. Timestamp baru yang ambigu harus diaudit secara eksplisit, bukan disesuaikan agar unik. Periksa jumlah kelas, kontribusi sumber, dan profil spasial setelah build sebelum memulai eksperimen.

## 7. Panduan operator sebelum dan saat pengambilan

### 7.1 Persiapan perangkat dan pilot

1. Catat identitas orang, ID sesi, perangkat, konfigurasi kamera/layar, dan urutan set. Rekaman ulang orang yang sama tetap memakai grup orang yang sama. Ihsan berbeda dari Ikhsan.
2. Pastikan kamera melihat wajah dan kedua mata dengan jelas. Atur posisi, pencahayaan, dan jarak sebelum kalibrasi, lalu pertahankan selama tugas. Periksa hasilnya lewat preview; jangan menetapkan penyebab error hanya dari dugaan posisi atau pencahayaan.
3. Cocokkan koordinat/resolusi layar kalibrasi dengan layar fisik tempat peserta melihat target dan mengerjakan tugas. Jika memakai dua device, operator boleh menjalankan server dan sensor pada satu device, tetapi peserta harus melihat target kalibrasi dan tugas pada bidang pandang yang sesuai pemetaan tracker. Koneksi port tidak memindahkan pemetaan koordinat gaze secara otomatis.
4. Pastikan HW9 terhubung dan RR masuk. Pada preview multimodal, tunggu teks `HRV siap; baseline belum dimulai`, lalu tekan Q. Q memulai kalibrasi mata dan baseline HRV bersama-sama. Menunggu di preview belum menghabiskan durasi baseline.
5. Uji persiapan, validasi, serta beberapa window tugas pada pilot sebelum pengambilan penuh. Gunakan hasil pilot untuk memastikan setup menghasilkan data yang layak; jangan melonggarkan ambang hanya agar tampilan berubah menjadi lolos.

### 7.2 Pemeriksaan validasi mata

Setelah peserta mengikuti target kalibrasi, sistem menampilkan validasi terpisah. Periksa terminal dan jendela `Post-Calibration Validation`.

| Yang terlihat | Arti | Tindakan operator |
|---|---|---|
| `Validation Results: Median=...px, P95=...px, CornerMax=...px` | Hasil pengukuran error validasi | Bandingkan ketiganya dengan konfigurasi sesi |
| `VALIDATION PASSED` | Ketiga syarat operasional terpenuhi | Lanjutkan pemeriksaan sensor tugas |
| `VALIDATION ACCURACY NOTICE` | Sedikitnya satu syarat gagal | Periksa kondisi dan ulangi validasi/kalibrasi |
| Tombol V pada layar gagal | Mengulangi validasi memakai kalibrasi yang sama | Gunakan bila peserta belum mengikuti target dengan benar atau perlu mengulang pemeriksaan |
| Tombol R pada layar gagal | Mengulang kalibrasi | Gunakan setelah memperbaiki setup atau ketika mapping perlu diulang |
| Space/Enter pada layar gagal | Melanjutkan dengan keputusan `accepted_manual` | Jangan menganggapnya sebagai hasil validasi lolos |
| Q pada layar gagal | Keluar dari proses | Gunakan jika pengambilan tidak dapat dilanjutkan |

Q pada preview persiapan memulai kalibrasi; Q pada layar gagal validasi keluar dari proses. Fungsi tombol bergantung layar yang sedang aktif.

Baseline HRV tetap berjalan saat operator mengulang kalibrasi mata. Kelulusan validasi awal juga belum menjamin tidak ada drift selama tugas. Jika posisi atau perangkat berubah, tandai waktu perubahan dan lakukan pemeriksaan ulang yang terdokumentasi. Implementasi saat ini menyediakan validasi pada alur awal kalibrasi; laporan ini tidak menambahkan tombol validasi berkala saat tugas.

Untuk pilot, simpan hasil melihat target yang posisinya diketahui setelah peserta berada pada posisi pengerjaan. Tim dapat memakainya untuk memeriksa kesesuaian gaze dengan posisi yang diketahui, tanpa mengarang event CodeArrange atau mengubah label tugas.

### 7.3 Pemeriksaan baseline HRV

1. Amati terminal `Kalibrasi: ... | RR diterima: ...`. RR diterima perlu bertambah dengan aliran yang stabil.
2. Tunggu `Baseline siap: RMSSD ... ms, SDNN ... ms.` dan penyelesaian kalibrasi mata. Jika terminal menyatakan baseline unavailable/timeout, periksa sensor dan lakukan sesi baru sesuai alur aplikasi.
3. Periksa row baseline di `hrv.csv`: `status=ready`, RMSSD positif, dan `coverage >= 0.80`. Gunakan nilai coverage tersimpan; pembagian angka JSON yang sudah dibulatkan hanya perkiraan.
4. Jika baseline berstatus ready tetapi coverage kurang dari 80%, operator perlu memperbaiki kestabilan RR lalu mengambil baseline baru bila menginginkan rasio baseline yang layak. Simpan sesi lama dan catat alasan pengulangan.

Contoh pemeriksaan read-only di PowerShell, ganti folder contoh dengan folder sesi yang sedang berjalan:

```powershell
Get-Content "sessions\saved_session\ihsan\hrv_calib_status.json"

Import-Csv "sessions\saved_session\ihsan\hrv.csv" |
    Where-Object { $_.phase -eq "baseline" } |
    Select-Object status, reason, rmssd_ms, accepted_rr_seconds, window_seconds, coverage
```

Sesi aktif biasanya berada pada folder yang ditampilkan `run_multimodal.py`, bukan harus di `saved_session`. Gunakan lokasi aktual dari terminal. Recorder menulis row baseline ke CSV setelah baseline selesai.

### 7.4 Pemeriksaan HRV selama tugas

Recorder default menerbitkan HRV pertama setelah 25 detik fase sensor task, lalu setiap 5 detik. Waktu sensor task tersebut tidak sama dengan klik Mulai CodeArrange. Event website menentukan bagian pengerjaan dan target saat fusi.

Pada terminal, cari baris `[HRV]`:

| Status CSV | Teks terminal | Tindakan |
|---|---|---|
| `ready` | `status: siap` | Periksa RMSSD/SDNN tersedia dan pantau kontinuitas |
| `insufficient_data` | `data belum cukup` | Dapat terjadi pada awal/gap; baca alasan jika berulang |
| `gap` | `jeda RR` | Periksa sensor dan koneksi; catat gangguan |
| `recovering` | `pemulihan` | Tunggu pemulihan; jangan menganggap interval ini siap |

Baris terminal juga menampilkan jumlah RR, pasangan, durasi RR, dan alasan. Syarat window saat ini: coverage ≥80%, sedikitnya 10 pasangan RR, dan usia RR terakhir ≤3 detik. Pada window live 25 detik, coverage 80% setara sekitar 20 detik durasi RR diterima. Coverage saja belum cukup jika syarat lain gagal.

Operator dapat membaca bagian akhir CSV tanpa mengubahnya:

```powershell
Import-Csv "sessions\saved_session\ihsan\hrv.csv" |
    Where-Object { $_.phase -eq "task" } |
    Select-Object -Last 6 window_end, status, reason, coverage, rr_pairs, data_age_seconds
```

Satu status gagal di awal tidak membatalkan seluruh sesi. Rangkaian gap/recovery yang berulang memerlukan pemeriksaan sebelum operator melanjutkan pengambilan panjang. Jangan menetapkan rentang RMSSD tertentu sebagai bukti sinyal bagus; kesiapan data dan validitas akuisisi yang perlu diperiksa.

Pantauan live tersebut memakai window default 25 detik. Setelah sesi selesai, builder tetap perlu menghitung ulang kelayakan 15/25/45 detik. Live 25 detik yang baik belum menjamin seluruh window 15 atau 45 detik akan lolos.

### 7.5 Pemeriksaan koneksi CodeArrange dan cakupan kelas

1. Salin ID sesi dari terminal `run_multimodal.py` ke kolom `ID Sensor`, tekan `Hubungkan`, dan pastikan status terhubung ke sesi yang benar sebelum peserta mulai.
2. Pastikan folder sesi berisi `task_events.csv` dan event bertambah ketika set/soal dimulai serta jawaban disubmit. Jangan mengisi event pengganti berdasarkan perkiraan waktu.
3. Jika tujuan pengambilan memperkuat kualitas ketiga kelas, usahakan setiap peserta menjalani easy, medium, dan hard dengan setup sensor yang konsisten. Catat urutan, jeda, dan pengulangan tugas. Pengambilan easy/medium saja masih dapat menambah data layak, tetapi tidak menambah keragaman High.
4. Pantau sensor sepanjang set, bukan hanya saat kalibrasi. Catat gangguan/perubahan posisi agar tim dapat menelusuri intervalnya.
5. Gunakan `Akhiri set` untuk menutup set; gunakan `Akhiri sesi penelitian` setelah rangkaian penelitian selesai. Hentikan sensor setelah event akhir tersimpan sesuai panduan operasional.

Sesi `bangsur_easy_only` tidak mempunyai ledger karena operator belum menghubungkan CodeArrange. Tim mempertahankan sumbernya tetapi mengecualikannya dari dataset supervised; tidak ada label Low pengganti.


## Pemeriksaan lonjakan RR sebelum pemodelan

Kebijakan exclude_local_rr_jump_v1 mengeluarkan window curated yang memuat RR diterima dengan deviasi lebih dari 250 ms terhadap median referensi RRPreprocessor. Riwayat normal memakai maksimum 11 RR dan minimum 5; setelah gap, referensi mengikuti kelompok recovery pada waktu rilis. Hanya beat dalam (start, end] yang sudah tersedia pada endpoint yang diperhitungkan.

Aturan ini menandai ketidakpastian lokal, bukan diagnosis bahwa setiap kandidat pasti artefak. Seluruh rekaman diperiksa dengan aturan yang sama tanpa membaca label atau skor model. Nilai RR, RMSSD, SDNN, fitur eye, dan label tidak diganti. Window terdampak tetap di all_data, dengan quality_tier poor dan alasan rr_local_median_jump pada metadata yang sudah ada. Tidak ada kolom flag baru, interpolasi, winsorization, atau aktivasi filter live.

| Window | Window terdampak di all_data | Core layak yang dikeluarkan | Low | Medium | High |
|---|---:|---:|---:|---:|---:|
| window_15s_step_5s | 52 | 27 | 13 | 5 | 9 |
| window_25s_step_5s | 65 | 40 | 22 | 9 | 9 |
| window_45s_step_5s | 79 | 64 | 41 | 14 | 9 |

| Rekaman | RR task | RR diterima pemeriksaan dasar | Kandidat lonjakan |
|---|---:|---:|---:|
| Alif | 3474 | 3460 | 28 |
| Andrean | 2905 | 2905 | 1 |
| Andreas | 2675 | 2675 | 0 |
| Athaya | 2623 | 2623 | 0 |
| Baithar | 2800 | 2800 | 0 |
| bangsur_easy_medium | 1472 | 1472 | 1 |
| Fabian | 2475 | 2475 | 0 |
| faqih | 2587 | 2587 | 2 |
| ihsan | 2016 | 1956 | 0 |
| ikhsan | 2561 | 2554 | 1 |
| irfan | 2990 | 2990 | 0 |
| kaka | 3163 | 3147 | 1 |
| Mon | 4347 | 4347 | 0 |
| Rafi | 2958 | 2958 | 1 |
| raina | 3069 | 3069 | 0 |
| Ramadhan | 556 | 556 | 0 |
| Ramadhan-2 | 2774 | 2774 | 2 |
| sam | 1578 | 1576 | 1 |
| Savitri | 4440 | 4440 | 0 |
| Shakilla | 1056 | 1052 | 0 |
| Siti | 2594 | 2578 | 0 |
| Sullivan | 741 | 705 | 0 |
| Tio | 3807 | 3803 | 2 |

manifest.json menyimpan konfigurasi, raw_id, waktu beat/ketersediaan, median referensi, serta hash rr_raw.csv untuk audit. Pemeriksaan independen saat publikasi mencocokkan penanda window, alasan kualitas, jumlah yang dikeluarkan, dan proyeksi CSV tersaring. Rasio baseline historis tetap informasi audit dan bukan input core. Normalisasi antarsesi serta pembagian Train/validation/test tidak dilakukan di CSV.


## Pemeriksaan dataset sebelum pemodelan

| Window | Missing core | Infinity core | ID window duplikat | Vektor core duplikat | Fitur core konstan | Rasio baseline tidak tersedia |
|---|---:|---:|---:|---:|---|---:|
| window_15s_step_5s | 0 | 0 | 0 | 0 | tidak ada | 468 |
| window_25s_step_5s | 0 | 0 | 0 | 0 | tidak ada | 410 |
| window_45s_step_5s | 0 | 0 | 0 | 0 | tidak ada | 304 |

Fitur inti tidak diimputasi. Ketidaktersediaan rasio baseline tidak diganti dengan nol dan tidak menggugurkan core; rasio itu bukan predictor default. Vektor sama atau korelasi tinggi tidak otomatis membuktikan duplikasi akuisisi. Duplikasi sumber dan ID interval diperiksa terpisah.

| Window | Low active_task | Seluruh Low | Persentase Low active_task |
|---|---:|---:|---:|
| window_15s_step_5s | 707 | 1687 | 41.91% |
| window_25s_step_5s | 470 | 1549 | 30.34% |
| window_45s_step_5s | 229 | 1394 | 16.43% |

Label fase tetap mengikuti protokol v2. Komposisi ini perlu dipertimbangkan saat menyusun development dan evaluasi; dataset tidak diseimbangkan dengan mengganti label atau membuang fase yang sah.

| Window | Fitur | Variasi antar-rerata sesi / total | Variasi antar-rerata kelas / total |
|---|---|---:|---:|
| window_15s_step_5s | rmssd_ms | 67.89% | 0.07% |
| window_15s_step_5s | ear_left_mean | 81.91% | 2.30% |
| window_15s_step_5s | iris_size_estimate_mean | 86.66% | 0.38% |
| window_25s_step_5s | rmssd_ms | 73.94% | 0.01% |
| window_25s_step_5s | ear_left_mean | 85.71% | 2.41% |
| window_25s_step_5s | iris_size_estimate_mean | 89.00% | 0.55% |
| window_45s_step_5s | rmssd_ms | 78.99% | 0.12% |
| window_45s_step_5s | ear_left_mean | 88.31% | 2.56% |
| window_45s_step_5s | iris_size_estimate_mean | 91.56% | 1.27% |

Rasio variasi adalah eta-squared deskriptif satu faktor. Kedua kolom tidak dijumlahkan dan tidak membuktikan penyebab biologis, kerusakan sensor, atau ketiadaan informasi nonlinear. Level dasar antarsesi tidak dikoreksi menggunakan seluruh CSV. Statistik preprocessing harus berasal dari Train atau baseline yang tersedia sebelum prediksi. Identitas tetap metadata, bukan predictor.
