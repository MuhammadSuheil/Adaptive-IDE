# Analisis dua sesi Rama — 23 Oktober 2026

Paket ini berisi ringkasan turunan per menit agar dapat dianalisis lebih lanjut. File sumber mentah tidak diubah. Semua cap waktu di keluaran diberi zona WIB (`+07:00`) atau dikelompokkan menurut menit WIB.

## Keluaran

Untuk tiap sesi tersedia: `*_eye_tracking_per_minute.csv`, `*_hrv_per_minute.csv`, `*_multimodal_per_minute.csv`, `*_codearrange_attempts.csv`, dan `*_minute_alignment.csv`. `two_session_quality_summary.csv` memuat ringkasan lintas sesi.

## Metode dan batasan

- Eye Tracking dikelompokkan dari `timestamp_ms` epoch; laju gaze, wajah, confidence, blink, FPS, dan sudut kepala diringkas per menit. Hitungan sudut memakai batas absolut 15° yaw dan 12° pitch sebagai indikator gerak kepala; bukan bukti seseorang berdiri.
- HRV dikelompokkan menurut `window_end` WIB; baris baseline tetap ditandai `phase=baseline`. Window 25 detik dengan langkah 5 detik saling tumpang tindih, sehingga banyak window bukan banyak pengamatan independen.
- Multimodal diringkas menurut `window_end` WIB. Label cognitive state dan rasio gaze mengikuti sumber multimodal; validitasnya bergantung pada gaze calibration dan coverage HRV.
- Attempt dipilih dari `codearrange-web/logs/attempts.csv` dengan `name` yang diawali `Rama`, dan timestamp di dalam rentang frame Eye Tracking sesi. Timestamp sumber UTC dikonversi ke WIB; `elapsed_from_joint_task_start_seconds` mengacu ke penanda kalibrasi bersama.
- `head_pose_shifted` hanya dilaporkan sebagaimana tersimpan. Jika gate posisi kepala dinonaktifkan, nilai false tidak membuktikan kepala tidak bergerak. Kolom sudut yaw/pitch dan hilangnya wajah lebih berguna sebagai indikator, tetapi tidak dapat memastikan tester berdiri.
- Folder sesi kedua diberi nama `20261023_154026_rama02`, sedangkan ID di `session_report.json` dan `eye_summary.json` masih menyebut `20261023_154026_P01_coding_task`. Kecocokan dengan log Rama didasarkan pada rentang waktu, bukan ID yang identik.

## Ringkasan per sesi

## Sesi_1: `20261023_142709_P01_coding_task`

Embedded report ID: `20261023_142709_P01_coding_task`. Joint task origin: `2026-10-23T14:29:48.613019+07:00`. Eye coverage: 123,810 frames at 57.94 FPS; face detection 99.99%; dropped frames 3.36%. Gaze calibration quality 0.22546828398071783; validation passed: `False` (median error 550.425841947138 px; p95 2401.654717127806 px).

HRV baseline: `ready`; 121.724609375 s accepted RR; RMSSD 52.393085108001515 ms. Task windows: 424, with 136 ready (32.1%); mean coverage 63.6%. Raw RR rows: 2225; gaps >5 s: 66; longest gap: 34.0 s. CodeArrange attempts: 21, correct: 2. Fused report mean score: 24.5; the score is a derived output, not a validated result when gaze calibration fails and HRV windows are missing.

### Interval penting

- **14:30–14:36:** HRV hanya 12/82 window ready (14,6%; coverage rata-rata 46%). Eye stream lengkap, tetapi gaze calibration quality 0,225 dan validasi gagal.
- **14:37–14:42:** HRV membaik menjadi 53/72 ready (73,6%; coverage 85%). Gaze tetap perlu dianggap tidak tervalidasi.
- **14:43–14:49:** periode gabungan terburuk: HRV 6/84 ready (7,1%), gaze confidence rata-rata 0,126, dan hanya 26,7% frame diberi label on-screen. Di sini tercatat dua submit Bubble Sort dan Binary Search timeout (14:44:58–14:48:03).
- **14:50–14:56:** HRV 22/84 ready (26,2%; coverage 63%); gaze confidence 0,204. Fibonacci dimulai pada 14:53:45.
- **14:57–15:01:** HRV relatif pulih (36/60 ready; 60%), tetapi gaze masih tak tervalidasi.
- **15:02–15:05:** HRV turun menjadi 7/42 ready (16,7%) dan gaze confidence 0,174; submit terakhir tercatat 15:02:36, sensor berlanjut sampai 15:05.

Gerakan berdiri tidak dapat disimpulkan dari sesi ini; kamera dan pose lebih stabil daripada sesi kedua.

| WIB | Puzzle | Score | Correct | Submission |
|---|---|---:|:---:|---|
| 14:30:40 | Variabel | 42.9% | false | manual |
| 14:31:23 | Bilangan Genap atau Ganjil | 100.0% | true | manual |
| 14:32:43 | Mencari Nilai Maksimum | 58.3% | false | manual |
| 14:35:04 | 4. Perbaiki Penjumlahan Bilangan Genap | 45.5% | false | timeout |
| 14:37:58 | Menghitung Huruf Vokal | 61.5% | false | manual |
| 14:39:32 | Linear Search | 64.3% | false | manual |
| 14:40:30 | Membalik String | 80.0% | false | manual |
| 14:40:56 | Membalik String | 80.0% | false | manual |
| 14:41:16 | Membalik String | 100.0% | true | manual |
| 14:42:15 | Penjumlahan Matriks | 58.3% | false | manual |
| 14:42:45 | Penjumlahan Matriks | 83.3% | false | manual |
| 14:44:58 | Bubble Sort | 21.4% | false | manual |
| 14:46:57 | Bubble Sort | 71.4% | false | manual |
| 14:48:03 | Binary Search | 0.0% | false | timeout |
| 14:53:45 | Fibonacci Iteratif | 58.3% | false | manual |
| 14:57:48 | Fibonacci Iteratif | 41.7% | false | manual |
| 14:58:33 | Fibonacci Iteratif | 50.0% | false | manual |
| 14:59:39 | Fibonacci Iteratif | 25.0% | false | manual |
| 15:00:53 | Fibonacci Iteratif | 50.0% | false | manual |
| 15:01:44 | Fibonacci Iteratif | 58.3% | false | manual |
| 15:02:36 | Fibonacci Iteratif | 66.7% | false | manual |


## Sesi_2_Rama02: `20261023_154026_rama02`

Embedded report ID: `20261023_154026_P01_coding_task`. Joint task origin: `2026-10-23T15:43:16.543744+07:00`. Eye coverage: 105,563 frames at 59.73 FPS; face detection 98.32%; dropped frames 0.39%. Gaze calibration quality 0.5589233808638925; validation passed: `False` (median error 442.6014965795088 px; p95 851.0923703964187 px).

HRV baseline: `ready`; 121.4873046875 s accepted RR; RMSSD 76.75127163095846 ms. Task windows: 350, with 210 ready (60.0%); mean coverage 78.1%. Raw RR rows: 1952; gaps >5 s: 34; longest gap: 66.0 s. CodeArrange attempts: 27, correct: 4. Fused report mean score: 24.6; the score is a derived output, not a validated result when gaze calibration fails and HRV windows are missing.

### Interval penting dan indikasi gerakan

- **15:43–15:55:** HRV relatif baik (118/148 window ready; 79,7% untuk gabungan rentang ini). Wajah mayoritas terdeteksi. Gaze calibration tetap gagal validasi.
- **15:56–15:57:** anomali paling kuat: wajah hilang pada 21,9% frame, ada beberapa rentang tanpa wajah (terpanjang sekitar 12,5 detik), dan tidak ada RR selama **15:56:44.540–15:57:50.510** (66 detik). HRV hanya 1/24 window ready. Terjadi di antara timeout Menghitung Huruf Vokal (15:55:49) dan submit pertama Fibonacci (15:57:40). Konsisten dengan tester berpindah jauh dari kamera/beranjak, tetapi tidak membuktikan berdiri.
- **15:58–15:59:** sudut kepala melewati batas 15° yaw atau 12° pitch di sekitar 30–36% frame wajah; Fibonacci dikirim 15:58:12 dan benar 15:58:45.
- **16:02–16:08:** indikasi perubahan pose kepala berulang, paling tinggi 16:04 (56,5% frame), 16:05 (45,0%), dan 16:06 (51,4%). HRV hanya 8/48 window ready (16,7%) dan ada jeda RR 14–18 detik. Ini bertepatan dengan submit Menggabungkan Array Terurut.
- **16:09–16:12:** HRV pulih sebagian (27/46 ready; 58,7%), tetapi gaze confidence rata-rata 0,113. Tiga submit terakhir adalah Pemeriksaan Anagram; yang terakhir timeout pada 16:12:19.

`head_pose_shifted` tersimpan false karena gate posisi kepala dinonaktifkan. Angka di atas dihitung ulang dari sudut yaw/pitch terhadap batas konfigurasi dan hanya menjadi indikator gerakan kepala. Frame tanpa wajah dan jeda RR memperkuat indikasi bergerak, tetapi kamera wajah saja tidak dapat memastikan orang berdiri. `contact_detected` kosong di semua baris RR sehingga kemungkinan sensor terlepas tidak dapat diuji dari kolom itu.

| WIB | Puzzle | Score | Correct | Submission |
|---|---|---:|:---:|---|
| 15:44:50 | Menghitung Huruf Vokal | 61.5% | false | manual |
| 15:45:50 | Linear Search | 28.6% | false | manual |
| 15:46:43 | Linear Search | 35.7% | false | manual |
| 15:47:13 | Membalik String | 80.0% | false | manual |
| 15:47:24 | Membalik String | 100.0% | true | manual |
| 15:48:07 | Penjumlahan Matriks | 100.0% | true | manual |
| 15:51:07 | Binary Search | 0.0% | false | manual |
| 15:51:12 | Binary Search | 25.0% | false | manual |
| 15:52:42 | Manipulasi Array 2 | 0.0% | false | manual |
| 15:52:45 | Manipulasi Array 2 | 0.0% | false | manual |
| 15:53:00 | Manipulasi Array 2 | 20.0% | false | manual |
| 15:53:47 | Transpose Matriks | 23.1% | false | manual |
| 15:54:13 | Transpose Matriks | 23.1% | false | manual |
| 15:54:36 | Transpose Matriks | 38.5% | false | manual |
| 15:54:54 | Transpose Matriks | 76.9% | false | manual |
| 15:55:17 | Transpose Matriks | 100.0% | true | manual |
| 15:55:49 | Menghitung Huruf Vokal | 23.1% | false | timeout |
| 15:57:40 | Fibonacci Iteratif | 25.0% | false | manual |
| 15:58:12 | Fibonacci Iteratif | 75.0% | false | manual |
| 15:58:45 | Fibonacci Iteratif | 100.0% | true | manual |
| 16:01:30 | Menggabungkan Array Terurut | 23.1% | false | manual |
| 16:01:47 | Menggabungkan Array Terurut | 23.1% | false | manual |
| 16:03:37 | Menggabungkan Array Terurut | 23.1% | false | manual |
| 16:04:26 | Menggabungkan Array Terurut | 23.1% | false | manual |
| 16:11:11 | Pemeriksaan Anagram | 12.5% | false | manual |
| 16:11:26 | Pemeriksaan Anagram | 12.5% | false | manual |
| 16:12:19 | Pemeriksaan Anagram | 12.5% | false | timeout |

