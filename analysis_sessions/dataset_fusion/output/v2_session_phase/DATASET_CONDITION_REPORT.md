# Kondisi dataset berdasarkan kelayakan fitur

Dataset utama sudah disaring untuk 10 fitur core. Kualitas posisi gaze diperiksa terpisah; jumlah baris yang besar tidak memastikan pemisahan Low/Medium/High akan kuat.

| Window | Low | Medium | High | Rasio terbesar/terkecil | Gaze tidak layak pada curated |
|---|---:|---:|---:|---:|---:|
| 15s | 1433 | 1433 | 1433 | 1.000 | 3411 |
| 25s | 1386 | 1386 | 1386 | 1.000 | 3300 |
| 45s | 1149 | 1149 | 1149 | 1.000 | 2823 |

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

class_distribution.csv memuat jumlah dan durasi unik per kelas. counts_by_session_label_phase_tier.csv memuat distribusi setiap rekaman/fase. Feedback, rating, rest, dan post-task mengikuti difficulty tugas yang ditelusuri dari event; fase tidak dimasukkan sebagai predictor. Window overlap tidak dihitung sebagai pengamatan independen.

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
| window_15s_step_5s | 69 | 43 | 11 | 9 | 23 |
| window_25s_step_5s | 113 | 75 | 16 | 23 | 36 |
| window_45s_step_5s | 184 | 147 | 30 | 43 | 74 |

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
| window_15s_step_5s | 0 | 0 | 0 | 0 | tidak ada | 412 |
| window_25s_step_5s | 0 | 0 | 0 | 0 | tidak ada | 405 |
| window_45s_step_5s | 0 | 0 | 0 | 0 | tidak ada | 253 |

Fitur inti tidak diimputasi. Ketidaktersediaan rasio baseline tidak diganti dengan nol dan tidak menggugurkan core; rasio itu bukan predictor default. Vektor sama atau korelasi tinggi tidak otomatis membuktikan duplikasi akuisisi. Duplikasi sumber dan ID interval diperiksa terpisah.

| Window | Low active_task | Seluruh Low | Persentase Low active_task |
|---|---:|---:|---:|
| window_15s_step_5s | 707 | 1433 | 49.34% |
| window_25s_step_5s | 470 | 1386 | 33.91% |
| window_45s_step_5s | 229 | 1149 | 19.93% |

Label fase tetap mengikuti protokol v2. Komposisi ini perlu dipertimbangkan saat menyusun development dan evaluasi; penyeimbangan memilih baris layak tanpa mengganti labelnya; kandidat tidak terpilih tetap ada di all_data.

| Window | Fitur | Variasi antar-rerata sesi / total | Variasi antar-rerata kelas / total |
|---|---|---:|---:|
| window_15s_step_5s | rmssd_ms | 69.75% | 0.09% |
| window_15s_step_5s | ear_left_mean | 85.39% | 2.71% |
| window_15s_step_5s | iris_size_estimate_mean | 87.06% | 0.17% |
| window_25s_step_5s | rmssd_ms | 76.05% | 0.05% |
| window_25s_step_5s | ear_left_mean | 87.39% | 2.57% |
| window_25s_step_5s | iris_size_estimate_mean | 89.24% | 0.53% |
| window_45s_step_5s | rmssd_ms | 79.11% | 0.31% |
| window_45s_step_5s | ear_left_mean | 89.32% | 1.68% |
| window_45s_step_5s | iris_size_estimate_mean | 91.97% | 1.43% |

Rasio variasi adalah eta-squared deskriptif satu faktor. Kedua kolom tidak dijumlahkan dan tidak membuktikan penyebab biologis, kerusakan sensor, atau ketiadaan informasi nonlinear. Level dasar antarsesi tidak dikoreksi menggunakan seluruh CSV. Statistik preprocessing harus berasal dari Train atau baseline yang tersedia sebelum prediksi. Identitas tetap metadata, bukan predictor.


## Batas fase nonpengerjaan

Rest/post-task hanya menjadi kandidat dalam 45 detik sejak active-task terakhir berhenti. Batas tidak dimulai ulang ketika feedback berubah menjadi rating, rest, atau post-task. Feedback/rating mengikuti durasi fase yang tercatat dan tidak dikenai batas khusus ini. Nilai 45 detik adalah batas operasional, bukan hasil pengukuran waktu pemulihan fisiologis. Window yang melewati batas dikeluarkan seluruhnya.

window_phase_mix menyimpan konteks sumber, waktu awal set/soal, akhir pengerjaan, dan batas grace dalam label_context. manifest.json menyimpan ledger dan readiness yang dipakai untuk merekonstruksi ulang segmen. Tidak ada timestamp, event, atau nilai sensor mentah yang diubah.


## Seleksi dataset utama

Basis adalah window yang hanya memuat active-task dan/atau pre-task. Setiap window yang memuat feedback/rating/rest/post-task dihitung sebagai opsional, termasuk campurannya dengan pengerjaan. T = min(jumlah basis terbesar antarkelas, jumlah total kandidat layak terkecil antarkelas). Setiap kelas memilih basis dahulu, kemudian opsional hanya untuk mencapai T. Basis mayoritas dapat diseleksi sebagian. Jika suatu kelas tidak tersedia, build dihentikan, bukan melonggarkan kualitas.

Seleksi mendahulukan tier good kemudian acceptable. Dalam setiap tier, kuota dibagi merata sesuai kapasitas antarpeserta lalu antarsesi, dengan sisa kuota berurutan leksikografis. Tiap rekaman diambil pada indeks tengah dari interval timestamp yang dibagi merata. Tidak ada pemilihan menurut skor model atau perubahan label.

| Window | Kelas | Basis layak | Opsional layak | Basis dipilih | Opsional dipilih | Layak tidak dipilih |
|---|---|---:|---:|---:|---:|---:|
| 15s | Low | 747 | 686 | 747 | 686 | 0 |
| 15s | Medium | 1763 | 774 | 1433 | 0 | 1104 |
| 15s | High | 1803 | 744 | 1433 | 0 | 1114 |
| 25s | Low | 498 | 888 | 498 | 888 | 0 |
| 25s | Medium | 1487 | 1080 | 1386 | 0 | 1181 |
| 25s | High | 1572 | 988 | 1386 | 0 | 1174 |
| 45s | Low | 241 | 1047 | 241 | 908 | 139 |
| 45s | Medium | 966 | 1528 | 966 | 183 | 1345 |
| 45s | High | 1149 | 1354 | 1149 | 0 | 1354 |

## Durasi unik dan fase terpilih

| Window | Kelas | Baris | Durasi unik (menit) |
|---|---|---:|---:|
| 15s | Low | 1433 | 136.58 |
| 15s | Medium | 1433 | 166.00 |
| 15s | High | 1433 | 167.92 |
| 25s | Low | 1386 | 138.92 |
| 25s | Medium | 1386 | 179.33 |
| 25s | High | 1386 | 175.25 |
| 45s | Low | 1149 | 135.42 |
| 45s | Medium | 1149 | 203.58 |
| 45s | High | 1149 | 159.83 |

Jumlah row yang sama bukan jumlah pengamatan independen atau durasi unik yang harus sama. Window tetap overlap. Rincian jumlah/durasi unik setiap versi ada di class_distribution.csv.

| Window | Kelas | Fase window | Baris terpilih |
|---|---|---|---:|
| 15s | High | active_task | 1433 |
| 15s | Low | active_task | 707 |
| 15s | Low | feedback | 12 |
| 15s | Low | mixed | 671 |
| 15s | Low | pre_task | 40 |
| 15s | Low | rest | 3 |
| 15s | Medium | active_task | 1433 |
| 25s | High | active_task | 1386 |
| 25s | Low | active_task | 470 |
| 25s | Low | feedback | 8 |
| 25s | Low | mixed | 880 |
| 25s | Low | pre_task | 28 |
| 25s | Medium | active_task | 1386 |
| 45s | High | active_task | 1149 |
| 45s | Low | active_task | 229 |
| 45s | Low | feedback | 2 |
| 45s | Low | mixed | 906 |
| 45s | Low | pre_task | 12 |
| 45s | Medium | active_task | 966 |
| 45s | Medium | mixed | 183 |

## Window yang dikeluarkan

| Window | Alasan utama | Baris |
|---|---|---:|
| 15s | ambiguous_eye_timestamp | 7 |
| 15s | interrupted | 73 |
| 15s | missing_task_context | 158 |
| 15s | outside_optional_grace | 104 |
| 25s | ambiguous_eye_timestamp | 11 |
| 25s | interrupted | 73 |
| 25s | missing_task_context | 247 |
| 25s | outside_optional_grace | 118 |
| 45s | ambiguous_eye_timestamp | 19 |
| 45s | interrupted | 73 |
| 45s | missing_task_context | 405 |
| 45s | outside_optional_grace | 146 |

excluded_windows.csv mencatat penolakan label/waktu dan timestamp ambigu. all_data menyimpan baris berlabel yang gagal kualitas atau tidak dipilih; curation_reason membedakan keduanya. curation_summary.json dan manifest menyimpan ringkasan seleksi. Riwayat label/seleksi sebelum migrasi disimpan ringkas di history. Hasil eksperimen lama tetap merujuk dataset lama dan tidak berubah.

## Rebuild

`python -m analysis_sessions.dataset_fusion.build_dataset_v2 --replace`

Manifest aktif mewariskan kebijakan label dan seleksi ini. Rebuild menghitung ulang dari rekaman mentah, bukan concat CSV atau hanya mengganti label lama. Hash sumber dan bundle diperiksa sebelum publikasi; penggantian memakai staging dan rollback.
