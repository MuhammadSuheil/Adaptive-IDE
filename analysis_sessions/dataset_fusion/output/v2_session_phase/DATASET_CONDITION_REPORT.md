# Kondisi Dataset dan Pemeriksaan Kualitas Pengambilan Data

Tanggal pemeriksaan: 16 November 2026.

Acuan: bundle v2 dengan waktu build `2026-11-16T15:24:24.254744+00:00`, setelah penggabungan Ihsan dan Bangsur. Laporan ini merupakan snapshot pemeriksaan, bukan hasil training batch-02. Pembaruan sumber berikutnya memerlukan pemeriksaan dan pembaruan angka laporan ini. Builder mengganti folder bundle saat rebuild; laporan manual ini perlu dibuat ulang atau disimpan sebelum rebuild jika tim memerlukan riwayatnya.

## 1. Kesimpulan

Ketiga dataset `good_acceptable` sudah mempunyai proporsi kelas yang berdekatan. Pengambilan tambahan untuk mengejar jumlah persis 1:1:1 tidak mendesak.

Tim masih perlu memperhatikan akurasi posisi gaze dan kestabilan perekaman RR pada sebagian sesi. Pengolahan dataset sudah lolos pemeriksaan integritas, tetapi pemeriksaan tersebut tidak membuktikan akurasi pengukuran fisiologis maupun kemampuan model membedakan cognitive load.

Penilaian per bagian:

| Bagian | Kondisi saat pemeriksaan | Keputusan |
|---|---|---|
| Proporsi Low/Medium/High | Rasio kelas terbesar/terkecil 1,16–1,25 kali | Sudah layak untuk eksperimen; tidak perlu memaksa jumlah sama |
| Integritas CSV dan penggabungan sensor | Lolos pemeriksaan bundle, label, subset, dan checksum | Pertahankan pipeline rekonstruksi yang sudah diperiksa |
| Akurasi posisi gaze | 19 dari 21 rekaman tidak melewati ambang validasi | Prioritas perbaikan pengambilan dan pemeriksaan fitur gaze |
| HRV pada row curated | Seluruh row memenuhi syarat HRV usable builder | Dapat digunakan sesuai definisi fitur; bukan validasi HW9 terhadap alat referensi |
| Kontinuitas HRV pada seluruh rekaman | Beberapa sesi kehilangan banyak window sebelum seleksi | Periksa aliran RR sejak persiapan dan selama tugas |
| Baseline HRV | 18 rekaman layak; 3 tidak memenuhi syarat baseline builder | Rasio baseline memerlukan penanganan nilai hilang atau eksperimen tanpa rasio |
| Keragaman sumber | 18–19 orang mempunyai row curated, bergantung window | Jumlah row overlap tidak sama dengan jumlah pengamatan independen |

Pernyataan bahwa seluruh data eye tracking dan HRV “jelek” terlalu luas. Bukti terkuat saat ini adalah kegagalan validasi posisi gaze pada sebagian besar rekaman dan kehilangan data HRV pada beberapa sesi. Kegagalan gaze tidak membuktikan bahwa HRV, EAR, atau kedipan pada rekaman yang sama ikut tidak layak.

## 2. Proporsi tiga dataset utama

| Window / step | Low | Medium | High | Total | Orang | Rasio terbesar/terkecil |
|---|---:|---:|---:|---:|---:|---:|
| 15 / 5 detik | 1.062 (29,7%) | 1.266 (35,4%) | 1.251 (35,0%) | 3.579 | 19 | 1,192 |
| 25 / 5 detik | 984 (30,6%) | 1.093 (34,0%) | 1.137 (35,4%) | 3.214 | 19 | 1,155 |
| 45 / 5 detik | 908 (35,9%) | 728 (28,8%) | 891 (35,3%) | 2.527 | 18 | 1,247 |

Persentase memakai pembulatan satu desimal. Medium masih paling sedikit pada window 45 detik. Perbedaan tersebut belum menjadi alasan untuk mengambil data ulang hanya demi balancing.

Ketiga CSV mewakili alternatif panjang window dari rekaman yang sama. Tim tidak boleh menumpuk ketiganya menjadi satu dataset untuk memperbanyak sampel.

## 3. Masalah akurasi gaze

### 3.1 Bukti dari rekaman

Validasi gaze terakhir pada sumber yang diproses mencatat 2 rekaman lolos dan 19 gagal. Rekaman yang lolos adalah Baithar dan Fabian. Dua rekaman tambahan, Ihsan dan `bangsur_easy_medium`, mempunyai keputusan `accepted_manual`, tetapi hasil `passed_thresholds` tetap `false`.

| Dataset curated | Row dari rekaman gagal validasi gaze | Persentase |
|---|---:|---:|
| 15 detik | 3.113 / 3.579 | 87,0% |
| 25 detik | 2.801 / 3.214 | 87,1% |
| 45 detik | 2.204 / 2.527 | 87,2% |

Persentase tersebut mengacu pada status validasi rekaman asal, bukan penilaian akurasi ulang untuk setiap window.

Konfigurasi validasi pada dua rekaman tambahan memakai tiga syarat yang harus terpenuhi bersama:

| Ukuran validasi | Batas operasional |
|---|---:|
| Median error rata-rata per target validasi | ≤180 piksel |
| P95 error rata-rata per target validasi | ≤350 piksel |
| Error terbesar pada target nonpusat (`CornerMax`) | ≤300 piksel |

Kode menghitung median/P95 tersebut dari error rata-rata prediksi pada setiap target validasi, bukan dari seluruh error per frame. Audit juga menyimpan statistik per sampel secara terpisah. Ambang ini adalah konfigurasi prototype, bukan standar universal untuk semua eye tracker atau ukuran layar.

Pada Bangsur, median error sekitar 224,0 piksel sehingga melewati batas 180 piksel, walaupun P95 sekitar 276,2 piksel memenuhi batasnya. Pada Ihsan, median sekitar 277,3 piksel dan P95 sekitar 601,5 piksel. Status gagal keduanya konsisten dengan aturan kode.

### 3.2 Bagian data yang perlu diperhatikan

- Posisi dan variasi posisi: `gaze_x_norm_mean`, `gaze_x_norm_std`, `gaze_y_norm_mean`, dan `gaze_y_norm_std`.
- Fitur turunan gaze, termasuk proporsi eye-state, memerlukan pemeriksaan algoritma serta ketahanan terhadap error/noise. Kelulusan atau kegagalan akurasi posisi saja belum mengukur semua aspek kualitas fitur turunan tersebut.
- EAR, kedipan, dan estimasi iris berasal dari pengukuran landmark dengan aturan validitas masing-masing. Validasi posisi gaze tidak menjadi pengujian langsung untuk akurasi fitur-fitur tersebut.

`good_acceptable` saat ini menyeleksi kelengkapan temporal, validitas frame/landmark, clipping, dan HRV readiness. Builder tidak mensyaratkan lolos validasi spasial gaze. `good_only` juga tidak menambahkan syarat akurasi spasial tersebut. Beralih ke `good_only` saja tidak menyelesaikan persoalan ini.

Jika tim langsung mensyaratkan validasi gaze lolos pada seluruh row curated saat ini, sisanya hanya:

| Window | Row tersisa | Orang |
|---|---:|---:|
| 15 detik | 466 | 2 |
| 25 detik | 413 | 2 |
| 45 detik | 323 | 2 |

Jumlah sumber tersebut tidak mendukung penggantian menyeluruh dataset utama tanpa mempertimbangkan keragaman pengukuran. Tim perlu mengumpulkan rekaman yang valid dan menguji konfigurasi fitur, bukan menghapus 19 rekaman tanpa menilai sensor lain yang masih berguna.

### 3.3 Yang belum terbukti

Audit ini belum membuktikan penyebab kegagalan gaze. Posisi kepala, pencahayaan, pemetaan koordinat layar, cara mengikuti target, resolusi/inference kamera, dan konfigurasi mapper merupakan hal yang perlu diperiksa lewat pilot. Angka gagal validasi tidak cukup untuk menetapkan salah satunya sebagai penyebab.

Tim juga belum menguji apakah masalah gaze menyebabkan skor model rendah pada bundle terbaru. Ablation tanpa koordinat gaze dan, secara terpisah, tanpa fitur turunan gaze merupakan pengujian pemodelan berikutnya.

## 4. Masalah HRV dan baseline

### 4.1 Kualitas selama perekaman berbeda antarrekaman

Perbandingan dua rekaman tambahan pada window 25 detik:

| Ukuran | Ihsan | Bangsur easy–medium |
|---|---:|---:|
| Kandidat window | 519 | 252 |
| Window berlabel tunggal (`all_data`) | 433 | 202 |
| Masuk `good_acceptable` | 101 | 191 |
| Curated / window berlabel | 23,3% | 94,6% |
| Status HRV cache task: ready | 170 | 242 |
| Status HRV cache task: gap | 175 | 6 |
| Status HRV cache task: insufficient_data | 147 | 4 |
| Status HRV cache task: recovering | 27 | 0 |
| Coverage baseline HRV | 71,3% | 87,0% |
| Baseline layak menurut builder | Tidak | Ya |

Status cache HRV mencakup seluruh grid tugas, termasuk window dengan target campuran yang tidak masuk `all_data`. Jumlah curated juga dipengaruhi kualitas eye. Karena itu, tim tidak boleh menganggap seluruh selisih curated semata-mata akibat HRV.

Cache live dapat berbeda dari replay pada paket yang tepat tiba di endpoint. Audit bundle memeriksa 7.460 window: 7.429 cocok dengan replay utama dan 31 sesuai urutan publikasi timer alternatif. Audit tidak menemukan selisih yang tidak terjelaskan. Rekonstruksi CSV memakai satu kebijakan endpoint yang konsisten.

Pengambilan Ihsan menunjukkan bahwa durasi tugas panjang belum menjamin banyak data fusion yang layak. Seluruh row yang sudah lolos ke tiga `good_acceptable` mempunyai HRV usable, RMSSD/SDNN tersedia, dan memenuhi aturan window builder. Kekurangan terjadi pada cakupan rekaman yang dapat dipertahankan, bukan berarti HRV tidak siap tetap lolos ke curated.

### 4.2 `Baseline siap` belum menjamin baseline layak untuk rasio

Recorder mengakhiri kalibrasi ketika durasi RR yang diterima mencapai target dan hasil HRV tersedia. Default targetnya 120 detik RR diterima, dengan batas waktu berjalan 300 detik. Builder juga mensyaratkan coverage baseline ≥80%, status ready, serta RMSSD positif untuk membuat `rmssd_baseline_ratio`.

Contoh Ihsan:

- Recorder menerima sekitar 120,9 detik RR dalam sekitar 169,4 detik waktu kalibrasi.
- Recorder mencatat `status=ready` dan `reason=target_reached`.
- Coverage baseline pada `hrv.csv` adalah `0.7134309894`, atau 71,3%.
- Builder mencatat `baseline_usable=false` dan `baseline_rejection_reason=insufficient_baseline_coverage`.

Tim tidak cukup mengandalkan teks “Baseline siap” atau `hrv_calib_status.json.status=ready`. Periksa juga coverage pada row `phase=baseline` di `hrv.csv`. Status JSON merekam kemajuan/penyelesaian baseline, bukan kualitas HRV live untuk setiap window tugas.

Baseline yang tidak layak tidak otomatis membatalkan RMSSD/SDNN absolut dari window tugas yang memenuhi syarat. Builder mengosongkan rasio baseline pada window tersebut.

Baseline berlangsung bersamaan dengan kalibrasi mata. Tim perlu mencatat kondisi ini dan tidak menyebutnya sebagai pengambilan resting baseline terpisah.

## 5. Nilai hilang, fitur, dan keragaman data

| Pemeriksaan curated | 15 detik | 25 detik | 45 detik |
|---|---:|---:|---:|
| `rmssd_baseline_ratio` kosong | 10,79% | 10,80% | 10,17% |
| `blink_count` dan `blink_rate_per_minute` kosong, masing-masing | 0,34% | 0,22% | 0,40% |
| Orang dengan Low | 19 | 18 | 16 |
| Orang dengan Medium | 18 | 17 | 16 |
| Orang dengan High | 14 | 14 | 13 |

`eye_state_other_ratio` konstan pada ketiga CSV. Tim dapat menghapus fitur konstan di pipeline training. Penanganan rasio baseline yang kosong memerlukan imputasi training-only atau eksperimen tanpa rasio. Jangan mengubah nilai kosong menjadi nol di CSV sumber.

Penggabungan tidak menjadikan identitas sebagai predictor. Tim tetap memakai identitas untuk deduplikasi, pengelompokan evaluasi, dan penilaian keragaman sumber. Penambahan easy/medium tidak menambah keragaman sumber High.

Pada window 25 detik, row berurutan bergeser 5 detik dan berbagi 20 detik pengukuran. Dataset 25 detik mencakup sekitar 369,4 menit interval unik di seluruh rekaman: Low 107,8 menit, Medium 133,0 menit, dan High 128,7 menit. Angka ini bukan 3.214 pengamatan yang independen.

Label mengikuti protokol v2: pengerjaan easy/practice/demo serta fase sesi Low yang terlacak menjadi Low; pengerjaan medium menjadi Medium; pengerjaan hard menjadi High. Tim mempertahankan keputusan protokol tersebut. Proporsi yang berdekatan belum membuktikan bahwa pola fisiologis ketiga label mudah dibedakan. Evaluasi model pada bundle terbaru masih diperlukan.

## 6. Pemeriksaan yang sudah lolos

Pemeriksaan terhadap bundle saat laporan dibuat mencakup:

- Parsing CSV/JSON, checksum, jumlah kelas, dan allowlist 26 fitur sensor.
- ID window unik, label tunggal sepanjang interval, timestamp berzona waktu, panjang window, dan grid 5 detik.
- Tidak ada nilai infinity atau duplikasi persis seluruh vektor fitur sensor pada ketiga curated CSV.
- Konsistensi `good_only` sebagai subset `good_acceptable`, lalu subset `all_data`, dengan nilai yang sama pada window yang sama.
- HRV readiness serta aturan coverage, pasangan RR, usia data, dan kelayakan rasio baseline.
- Perbandingan sensor independen pada 868 sampel window di tiga panjang window.

Pada penggabungan 16 November, pemeriksaan mempertahankan seluruh window lama dengan nilai identik dan menjaga hash data mentah serta arsip v1. Rangkaian tes builder menjalankan 64 tes: 63 lulus, satu dilewati karena izin symlink Windows. Tidak ada training model pada penggabungan tersebut.

Pemeriksaan ini membuktikan konsistensi pengolahan sesuai kode. Tim belum memvalidasi akurasi HW9 terhadap alat referensi melalui pemeriksaan bundle ini, dan belum membuktikan akurasi gaze pada rekaman yang gagal.

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

## 8. Seleksi data lama dan baru setelah pengambilan

| Kondisi sumber/window | Perlakuan |
|---|---|
| Data lama atau baru memenuhi kriteria sensor yang diperlukan | Pertahankan dalam konfigurasi dataset yang sesuai |
| HRV window tidak usable | Keluarkan dari dataset utama fusion yang mensyaratkan HRV, tetapi simpan sumber dan alasan |
| Gaze gagal validasi, HRV layak | Tinjau penggunaan fitur gaze; data dapat tetap berguna untuk HRV-only atau fitur lain setelah pemeriksaan yang sesuai |
| Baseline tidak layak, HRV tugas layak | Pertahankan HRV absolut; rasio baseline tetap kosong |
| Tidak ada event aktivitas yang dapat ditelusuri | Pulihkan dari sumber autentik atau kecualikan; jangan membuat label perkiraan |
| Rekaman ulang orang yang sama | Simpan sebagai acquisition berbeda, satu grup orang untuk evaluasi |

Tim menyimpan data mentah lama maupun baru. Tidak ada penggantian satu-banding-satu atau kewajiban membuang rekaman lama setelah data baru masuk. Jika tim menyetujui aturan kualitas baru, terapkan pada seluruh sumber dengan aturan yang sama, dokumentasikan perubahan, lalu bangun ulang view yang sesuai. Jangan diam-diam mengubah syarat dataset saat membandingkan hasil eksperimen.

Aturan admission `good_acceptable` saat ini tetap berlaku. Laporan ini belum menambahkan gate validasi spasial, menghapus fitur, memotong kuota kelas, mengubah label, atau menyetujui release training. Tim perlu menetapkan aturan penggunaan gaze sebelum eksperimen utama yang mengandalkan gaze.

Seleksi berdasarkan akuisisi/validasi sensor harus mendahului pembacaan hasil test. Akurasi model yang turun bukan alasan menghapus peserta atau window dari test.

## 9. Prioritas tindakan

| Prioritas | Tindakan | Hasil yang perlu diperiksa |
|---|---|---|
| 1 | Pilot setup dan validasi gaze sebelum pengambilan penuh | Hasil validasi memenuhi konfigurasi, kondisi layar/posisi terdokumentasi |
| 2 | Pastikan baseline dan aliran RR layak | Coverage baseline ≥80% untuk rasio; status task siap berkelanjutan, gangguan tercatat |
| 3 | Rekam ketiga tingkat bila sasaran pengambilan mencakup kualitas seluruh kelas | Data layak tidak hanya berasal dari easy atau satu orang |
| 4 | Audit dan rebuild ketiga panjang window | Kontribusi aktual, cakupan, nilai hilang, serta alasan exclusion per sesi |
| 5 | Mulai baseline dan ablation batch-02 setelah penerimaan dataset | Uji fusion, HRV-only, eye-only, tanpa gaze/turunannya, serta tanpa rasio baseline sesuai registry eksperimen |

Pengambilan tambahan tidak wajib demi balancing. Tim masih dapat menjalankan eksperimen dengan bundle saat ini sambil memperlakukan kualitas gaze sebagai faktor yang perlu diuji. Jika kesempatan pengambilan berikutnya tersedia, sasaran yang berguna adalah rekaman dengan pengukuran yang lebih dapat dipertanggungjawabkan, bukan jumlah row tertentu.

Window HRV pendek juga merupakan pilihan representasi yang perlu diuji, bukan jaminan kualitas. Penelitian ECG menunjukkan perbedaan validitas RMSSD dan SDNN pada durasi ultra-short; hasil tersebut tidak otomatis memvalidasi HW9 atau kondisi tugas penelitian ini. Karena itu, tim tetap membandingkan 15/25/45 detik melalui evaluasi yang sesuai.

## 10. Ringkasan sumber yang diproses

Tabel berikut memakai `session_quality_summary.csv` pada varian 25 detik. Median/P95 dibulatkan satu desimal. “Baseline layak” mengikuti builder; jumlah curated memakai good + acceptable. Tidak ada sumber yang diputuskan layak hanya berdasarkan median pada tabel ini karena pemeriksaan validasi memakai tiga syarat.

| Rekaman | Validasi gaze | Median error (px) | P95 error (px) | Baseline HRV | Row curated 25 detik |
|---|---|---:|---:|---|---:|
| Andrean | Gagal | 315,2 | 795,3 | Layak | 292 |
| Andreas | Gagal | 182,3 | 463,2 | Layak | 145 |
| Athaya | Gagal | 523,8 | 1.319,9 | Layak | 221 |
| Baithar | Lolos | 166,1 | 325,9 | Layak | 139 |
| bangsur_easy_medium | Gagal | 224,0 | 276,2 | Layak | 191 |
| Fabian | Lolos | 111,7 | 189,7 | Layak | 274 |
| faqih | Gagal | 200,7 | 1.035,1 | Layak | 265 |
| ihsan | Gagal | 277,3 | 601,5 | Tidak layak | 101 |
| ikhsan | Gagal | 439,2 | 585,6 | Layak | 212 |
| irfan | Gagal | 389,5 | 661,4 | Layak | 17 |
| kaka | Gagal | 287,1 | 747,0 | Layak | 151 |
| Rafi | Gagal | 317,2 | 1.275,9 | Layak | 298 |
| raina | Gagal | 238,2 | 439,3 | Layak | 105 |
| Ramadhan | Gagal | 335,8 | 1.130,9 | Layak | 52 |
| Ramadhan-2 | Gagal | 394,0 | 825,5 | Layak | 0 |
| sam | Gagal | 645,7 | 1.215,0 | Layak | 150 |
| Savitri | Gagal | 287,6 | 1.125,5 | Layak | 258 |
| Shakilla | Gagal | 166,5 | 820,2 | Layak | 94 |
| Siti | Gagal | 1.329,7 | 2.334,2 | Layak | 3 |
| Sullivan | Gagal | 486,5 | 1.636,3 | Tidak layak | 0 |
| Tio | Gagal | 386,3 | 961,4 | Tidak layak | 246 |

Ramadhan dan Ramadhan-2 merupakan satu orang dengan dua rekaman. Ihsan berbeda dari Ikhsan. `bangsur_easy_only` merupakan acquisition berbeda dari orang yang sama dengan `bangsur_easy_medium`, tetapi tidak masuk 21 sumber yang diproses karena pengecualian ledger.

## 11. Sumber dan penelusuran

### Dataset dan audit

- [Laporan pembuatan dataset](QUALITY_REPORT.md), [manifest](manifest.json), dan [audit replay](replay_validation.json).
- CSV utama: [15 detik](window_15s_step_5s/good_acceptable.csv), [25 detik](window_25s_step_5s/good_acceptable.csv), dan [45 detik](window_45s_step_5s/good_acceptable.csv).
- [Ringkasan kualitas sumber 25 detik](window_25s_step_5s/session_quality_summary.csv), [kualitas HRV/window](window_25s_step_5s/sensor_quality_counts.csv), serta [distribusi/durasi](window_25s_step_5s/class_distribution.csv).
- Allowlist fitur dan data dictionary berada pada setiap folder window.

Identitas snapshot tiga CSV utama, SHA-256:

```text
15s: f6a45d947fb67231acd1eb718f57dd5b3b36680bcdc31fed7d0b7e69815aa8bb
25s: bfd4e2f266992f3c271d29fe13fa78033527ce4e4fedcf476912352115aa3db1
45s: b4a476a8d7305bd81ff28235370b45410192f6f692146feb3d941770f2173666
```

### Implementasi dan log yang perlu diperiksa

- [Validasi dan kontrol eye tracking](../../../../eye_tracking_prototype/eye_tracking_prototype.py): `run_validation_screen`, audit `held_out_validation`, dan preview persiapan. [Konfigurasi eye](../../../../eye_tracking_prototype/modules/config.py).
- [Recorder dan baseline HRV](../../../../hrv_monitor_prototype/calibrate.py), [ambang HRV](../../../../hrv_monitor_prototype/config.py), [builder v2](../../build_dataset_v2.py), dan [konfigurasi dataset](../../config.json).
- Pada folder sesi: `eye_summary_calibration.json` menyimpan konfigurasi/percobaan kalibrasi dan held-out validation; periksa percobaan terakhir dengan fase `held_out_validation`, termasuk `passed_thresholds` dan `decision`. `eye_summary.json` dapat memuat ringkasan audit yang sama.
- `hrv_calib_status.json` menyimpan status baseline; row baseline/task pada `hrv.csv` menyimpan coverage/status pengukuran. `rr_raw.csv` menyimpan RR mentah, dan `task_events.csv` menyimpan ledger aktivitas. Beberapa sumber mentah hanya tersedia pada penyimpanan lokal karena aturan Git.
- [Panduan integrasi CodeArrange](../../../../.md/notes/codearrange_integration_guidelines.md) untuk menjalankan, menghubungkan, dan menutup sesi.

### Rujukan pengukuran

- Niehorster dkk., [The fundamentals of eye tracking, Part 7: Determining data quality](https://pmc.ncbi.nlm.nih.gov/articles/PMC13230380/): akurasi, presisi, dan kehilangan data merupakan aspek yang berbeda; pemilihan analisis perlu mempertimbangkan kualitas pengukuran.
- Muñoz dkk., [Validity of (Ultra-)Short Recordings for Heart Rate Variability Measurements](https://pmc.ncbi.nlm.nih.gov/articles/PMC4586373/): validitas HRV ultra-short bergantung parameter dan durasi; studi tersebut memakai ECG dan tidak menjadi validasi langsung untuk hardware penelitian ini.
