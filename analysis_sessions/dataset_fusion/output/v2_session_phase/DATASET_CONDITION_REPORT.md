# Kondisi Dataset Setelah Penggantian Selektif

Waktu pemeriksaan UTC: 2026-11-17T10:02:09.193871+00:00.

## Kesimpulan

Proporsi ketiga kelas dipertahankan dari dataset sebelumnya. Sebagian window acceptable lama diganti window sesi tambahan yang lolos validasi gaze dan tidak lebih buruk pada tujuh indikator kualitas. Ini pembaruan dataset, bukan hasil evaluasi model. Data lama tidak dinyatakan seluruhnya tidak layak.

| Window | Low | Medium | High | Total | Orang | Rasio terbesar/terkecil |
|---|---:|---:|---:|---:|---:|---:|
| 15s | 1062 | 1266 | 1251 | 3579 | 21 | 1.192 |
| 25s | 984 | 1093 | 1137 | 3214 | 21 | 1.155 |
| 45s | 908 | 728 | 891 | 2527 | 20 | 1.247 |

## Penggantian bertahap: Mon kemudian Alif

Kebijakan: `conservative_quality_replacement_v2`. Referensi awal dan setiap tahap tersimpan pada replacement_selection.json. Tahap berikutnya memakai hasil curated tahap sebelumnya, bukan memilih ulang dari nol. Semua pasangan tahap lama dan semua window tambahan yang sudah dipilih tetap dipertahankan.

Penggantian hanya terhadap acceptable lama yang gagal validasi gaze. Kandidat baru harus lolos validasi gaze dan good/acceptable. Coverage temporal eye, face/EAR, gaze, iris, serta HRV tidak menurun. Clipping dan usia RR tidak meningkat; baseline layak tidak diganti baseline tidak layak. Toleransi numerik 1e-9.

Pencocokan memaksimalkan pasangan yang memenuhi seluruh syarat, secara deterministik, satu-banding-satu pada kelas dan panjang window sama. Dukungan setiap peserta–kelas lama tetap ada. Tidak ada pemilihan berdasarkan skor model, data sintetis, atau perubahan label.

| Tahap | Sumber | Window | Ganti Low | Ganti Medium | Ganti High | Total |
|---|---|---|---:|---:|---:|---:|
| 1 | Mon | 15s | 146 | 66 | 95 | 307 |
| 1 | Mon | 25s | 140 | 64 | 77 | 281 |
| 1 | Mon | 45s | 120 | 32 | 39 | 191 |
| 2 | Alif | 15s | 98 | 29 | 25 | 152 |
| 2 | Alif | 25s | 80 | 15 | 8 | 103 |
| 2 | Alif | 45s | 65 | 8 | 0 | 73 |

| Window | Low / Medium / High akhir | Orang awal / akhir | Durasi unik awal / akhir (menit) |
|---|---|---|---|
| 15s | 1062 / 1266 / 1251 | 19 / 21 | 373.58 / 390.08 |
| 25s | 984 / 1093 / 1137 | 19 / 21 | 369.42 / 387.75 |
| 45s | 908 / 728 / 891 | 18 / 20 | 330.58 / 350.67 |

Awal berarti referensi sebelum tahap Mon. Jumlah tiap kelas tetap pada semua tahap. replacement_pairs.csv mencatat selection_stage beserta indikator tiap pasangan. curation_selected dan curation_reason adalah metadata, bukan predictor. Row lama yang diganti tetap di all_data; window timestamp ambigu hanya di audit/excluded_windows. good_only berisi good terpilih dan good_acceptable berisi good/acceptable terpilih.

## Kualitas yang masih perlu diperhatikan

| Window | Gaze gagal (row / persen) | Rasio baseline kosong | Blink kosong | HRV tidak usable di curated |
|---|---:|---:|---:|---:|
| 15s | 2654 / 74.2% | 247 | 6 | 0 |
| 25s | 2417 / 75.2% | 222 | 5 | 0 |
| 45s | 1940 / 76.8% | 162 | 10 | 0 |

Pada 23 rekaman sumber, 4 lolos dan 19 gagal validasi gaze. Tabel curated menghitung status rekaman asal, bukan pemeriksaan akurasi baru per window.

- Kegagalan gaze tetap relevan untuk koordinat gaze dan fitur turunannya. Penggantian ini mengurangi kontribusinya, tidak memperbaiki kalibrasi sumber lama.
- HRV absolut yang lolos aturan window tidak dibatalkan hanya karena baseline rasio tidak layak. Nilai rasio kosong tidak diganti nol.
- EAR, blink, estimasi iris, dan HRV tidak otomatis tidak layak ketika validasi spasial gaze gagal.
- Validasi awal yang lolos belum membuktikan tidak ada drift selama pengerjaan. Ambang berasal dari konfigurasi prototype, bukan standar universal.
- Jumlah window overlap bukan pengamatan independen. Dukungan peserta–kelas dipertahankan, tetapi durasi unik dapat berubah.
- Label tetap mengikuti protokol v2: fase sesi Low yang terlacak dan pengerjaan easy/practice menjadi Low; pengerjaan medium/hard menjadi Medium/High.

## Kontribusi sumber tambahan

| Sumber | Window | Berlabel, timestamp layak | Good/acceptable sebelum seleksi | Terpilih | Tidak terpilih |
|---|---|---:|---:|---:|---:|
| Mon | 15s | 392 | 371 | 307 | 85 |
| Alif | 15s | 388 | 230 | 152 | 236 |
| Mon | 25s | 333 | 330 | 281 | 52 |
| Alif | 25s | 308 | 189 | 103 | 205 |
| Mon | 45s | 252 | 252 | 191 | 61 |
| Alif | 45s | 235 | 176 | 73 | 162 |

Kandidat tidak dipilih jika sensornya poor, tidak mempunyai pasangan dominasi kualitas, atau penggantiannya menghilangkan dukungan peserta–kelas lama. Window target campuran tetap berada pada excluded_windows.csv, bukan dianggap sensor buruk. all_data tidak identik dengan dataset utama.

## Kontinuitas HRV sumber sebelum seleksi

| Sumber | Window cache 25s | Ready | Tidak ready | Ready (%) |
|---|---:|---:|---:|---:|
| Alif | 551 | 420 | 131 | 76.2 |
| Andrean | 399 | 393 | 6 | 98.5 |
| Andreas | 404 | 402 | 2 | 99.5 |
| Athaya | 399 | 392 | 7 | 98.2 |
| Baithar | 426 | 407 | 19 | 95.5 |
| bangsur_easy_medium | 252 | 242 | 10 | 96.0 |
| Fabian | 405 | 404 | 1 | 99.8 |
| faqih | 409 | 395 | 14 | 96.6 |
| ihsan | 519 | 170 | 349 | 32.8 |
| ikhsan | 335 | 311 | 24 | 92.8 |
| irfan | 407 | 395 | 12 | 97.1 |
| kaka | 452 | 331 | 121 | 73.2 |
| Mon | 517 | 514 | 3 | 99.4 |
| Rafi | 411 | 390 | 21 | 94.9 |
| raina | 397 | 392 | 5 | 98.7 |
| Ramadhan | 63 | 59 | 4 | 93.7 |
| Ramadhan-2 | 348 | 335 | 13 | 96.3 |
| sam | 210 | 200 | 10 | 95.2 |
| Savitri | 408 | 405 | 3 | 99.3 |
| Shakilla | 155 | 148 | 7 | 95.5 |
| Siti | 397 | 330 | 67 | 83.1 |
| Sullivan | 233 | 0 | 233 | 0.0 |
| Tio | 431 | 382 | 49 | 88.6 |

## Timestamp ambigu: Alif

Semua frame dan timestamp mentah dipertahankan. Tiga kelompok pada sesi Alif merupakan frame berbeda dengan timestamp milidetik sama, bukan baris duplikat. Tidak ada pergeseran waktu atau interpolasi. Setiap window dengan start < waktu ambigu <= end dikeluarkan, termasuk dari all_data. Increment kedipan tetap mengikuti urutan frame asli. Hash sumber serta detail kelompok ada di replacement_selection.json dan replay_validation.json.

| Waktu WIB | Baris CSV | Frame | Capture ID |
|---|---|---|---|
| 2026-11-17T16:09:25.788000+07:00 | [98177, 98178] | [98176, 98177] | [111378, 111379] |
| 2026-11-17T16:09:32.445000+07:00 | [98574, 98575] | [98573, 98574] | [111777, 111778] |
| 2026-11-17T16:21:53.035000+07:00 | [141659, 141660] | [141658, 141659] | [156121, 156122] |

| Window | Semua window menyentuh waktu ambigu | Tambahan exclusion di luar target campuran |
|---|---:|---:|
| 15s | 7 | 2 |
| 25s | 11 | 1 |
| 45s | 19 | 0 |

## Pemeriksaan dan pemakaian berikutnya

Replay memeriksa 8528 window cache HRV; 0 perbedaan tidak terjelaskan. Validator mengulang pencocokan, memeriksa ledger pasangan, tier, label, subset, serta checksum sebelum dan setelah publikasi.

Gunakan tiga CSV good_acceptable secara terpisah. Ikuti 26 predictor pada feature_columns.json; curation_selected, curation_reason, identitas, waktu, dan konteks tugas adalah metadata. Snapshot model lama tetap menggunakan CSV/hash lama. Evaluasi bundle ini memerlukan release/run baru, bukan mencampur hasil lama dan baru.

Periksa [QUALITY_REPORT.md](QUALITY_REPORT.md), [manifest](manifest.json), [pasangan penggantian](replacement_pairs.csv), dan [referensi seleksi](replacement_selection.json).

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
