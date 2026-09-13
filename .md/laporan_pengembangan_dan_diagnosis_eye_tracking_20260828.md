# Laporan Konsolidasi Pengembangan dan Diagnosis Eye Tracking

**Proyek:** Adaptive IDE — Eye Tracking Prototype  
**Tanggal:** 28 Agustus 2026  
**Cakupan:** Konsolidasi laporan pembenahan sebelumnya, perubahan yang dibuat dalam sesi ini, analisis seluruh kelompok pengujian, dan diagnosis tiga session terakhir.

---

## 1. Ringkasan Eksekutif

Prototype telah mengalami peningkatan besar pada performa dan fondasi pengukuran:

- Processing meningkat dari sekitar **6–13 FPS** menjadi stabil sekitar **39,5 FPS**.
- Capture, inference, dan UI telah dipisah sehingga grid fullscreen tidak lagi menahan MediaPipe.
- Calibration sekarang menolak hasil buruk, memakai validation, outlier rejection, dan hanya menghitung frame kamera yang unik.
- Black grid sekarang merepresentasikan **seluruh layar fisik**, sementara padding hanya menjadi visualisasi area di luar representasi layar.
- Face detection pada tiga pengujian terakhir mencapai sekitar **99,9%**, sehingga lighting, kacamata, dan deteksi wajah bukan penyebab utama kegagalan terakhir.

Walaupun demikian, akurasi belum siap disebut production-ready lintas pengguna. Hasil masih sangat bergantung pada pengguna, pose kepala, dan area layar yang diuji. Pengguna pertama memperoleh hasil yang dapat diterima ketika menguji area normal dan mendekati sudut, sedangkan pengguna kedua menghasilkan prediksi ekstrem ketika melihat sudut layar dan bias besar ketika kembali melihat area tengah.

Kesimpulan utama:

> Bottleneck FPS, duplicate-frame calibration, cache, dan geometri padding sudah ditangani. Masalah terbesar yang tersisa adalah model gaze yang belum robust terhadap perbedaan anatomi dan perubahan yaw/pitch kepala antar pengguna, serta regresi yang melakukan extrapolation secara tidak terkendali di area sudut.

---

## 2. Kondisi Awal dari Laporan Sebelumnya

Laporan `laporan_pembenahan_eye_tracking.md` mencatat masalah awal berikut:

1. Calibration dapat mengumpulkan sampel walaupun pengguna tidak benar-benar melihat target.
2. Tracking horizontal relatif lebih baik daripada vertikal.
3. Pergerakan kepala mengubah koordinat iris secara signifikan.
4. Kelopak mata dapat menutupi iris ketika melihat ke bawah.
5. Model polynomial dapat mengalami interpolation/extrapolation drift.
6. Processing hanya berjalan sekitar 6–13 FPS walaupun webcam meminta 60 FPS.
7. Banyak parameter masih perlu dipindahkan ke konfigurasi.

Pembenahan awal menggunakan normalisasi iris relatif terhadap eye corner, EAR, hybrid regression, multithreaded webcam capture, EMA, dan konfigurasi YAML. Namun, klaim performa dan generalisasi pada saat itu belum divalidasi dengan data lintas pengguna.

---

## 3. Temuan Awal pada Sesi Ini

### 3.1 Calibration buruk tetap memasuki tracking

Session lama memiliki calibration quality sekitar `0.22–0.35`, bahkan ada yang `0.0`, tetapi aplikasi tetap memulai tracking. Prediksi gaze mencapai puluhan ribu piksel di luar layar.

### 3.2 FPS yang ditampilkan tidak akurat

Timer dimulai sebelum calibration, sedangkan frame counter hanya bertambah saat tracking. Akibatnya, durasi calibration ikut menurunkan angka FPS. Setelah dihitung dari timestamp CSV, processing aktual tetap rendah, tetapi angka HUD memang salah.

### 3.3 Async capture tidak sama dengan async inference

Webcam sudah memiliki capture thread, tetapi `detect_for_video`, CSV, pembuatan grid fullscreen, `imshow`, dan input keyboard masih berada dalam loop yang sama. Session menunjukkan MediaPipe hanya memerlukan sekitar 9–12 ms, sedangkan satu frame loop memerlukan sekitar 107 ms. Hampir 98 ms hilang di rendering/UI.

### 3.4 Padding sebelumnya mengurangi layar fisik

Padding 5% sebelumnya diperlakukan sebagai `off_screen`, walaupun area tersebut masih berada di dalam monitor. Black grid hanya merepresentasikan 90% bagian tengah layar. Hal ini menyebabkan false off-screen di dekat sisi layar.

### 3.5 Calibration menghitung duplicate camera frames

Sebelum pembenahan terakhir, calibration memakai `read_frame()` yang dapat mengembalikan frame kamera yang sama berulang kali. Dengan periode kamera sekitar 25,3 ms dan inference sekitar 10–12 ms, satu frame fisik dapat dihitung dua sampai tiga kali. Stability streak dan 30 sampel per titik menjadi terlalu optimistis.

---

## 4. Perubahan yang Telah Diimplementasikan

### 4.1 Calibration dan validation

- Target calibration ditempatkan secara eksplisit di seluruh layar.
- Instruksi meminta pengguna melihat tepat ke tengah dot dan menjaga kepala tetap diam.
- Frame dengan EAR tidak valid, blink, face missing, atau noise tinggi ditolak.
- Sampel per titik diringkas dengan median dan MAD outlier rejection.
- Feature span horizontal dan vertikal divalidasi.
- Model memakai normalized screen target, bukan langsung mempelajari nilai piksel besar.
- Leave-one-point-out validation menghasilkan median error, P95 error, dan quality.
- Tracking tidak dimulai bila quality berada di bawah threshold.
- Calibration sekarang memakai `frame_id` dan hanya menerima **unique camera frames**.

### 4.2 Mapping dan gaze state

- Iris diproyeksikan ke koordinat lokal mata untuk mengurangi efek head roll.
- State dipisah menjadi `on_screen`, `gaze_outside_screen`, `face_missing`, dan `eyes_invalid_or_blink`.
- Prediction, face validity, dan missing face tidak lagi dianggap sebagai satu jenis kegagalan yang sama.

### 4.3 Pipeline performa

- Capture thread memberi frame terbaru beserta ID dan capture timestamp.
- Inference worker memproses frame unik terbaru.
- Frame lama dilewati dan tidak dimasukkan antrean agar latency tidak meningkat.
- Main thread hanya menangani UI OpenCV dan input keyboard.
- UI dibatasi ke 15 FPS, sedangkan gaze processing berjalan mengikuti kemampuan kamera.
- Static grid background di-cache agar tidak digambar ulang sepenuhnya setiap frame.
- Recalibration, pause, snapshot, debug, quit, dan cleanup disesuaikan dengan lifecycle worker.

### 4.4 Telemetry

CSV dan summary sekarang mencatat:

- Capture FPS dan unique-frame processing FPS.
- Capture age.
- Preprocessing, inference, mapping/metrics, logging, dan total processing time.
- P50 dan P95 setiap processing stage.
- Dropped-frame count dan ratio.
- UI FPS.
- Calibration median/P95 error, feature span, dan jumlah unique calibration frames.

### 4.5 Geometri layar dan padding

- Logical gaze classification memakai seluruh koordinat layar `0..screen_width` dan `0..screen_height`.
- Black grid adalah representasi berskala dari seluruh layar fisik.
- Padding tetap terlihat mengelilingi black grid, tetapi hanya merupakan visual off-screen padding.
- Padding tidak menghapus bagian layar dari section classification.
- Calibration tidak lagi menampilkan padding sebagai bagian dari layar.
- Target calibration saat ini mencakup 7%, 50%, dan 93% layar.

---

## 5. Analisis Tiga Session Terakhir

Urutan session sesuai waktu pengujian:

| Urutan | Session | Interpretasi | Frames | Face rate | FPS | Quality | Median error | P95 error | Unique calibration frames |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `d4bdf090...164558` | Pengguna pertama; hasil sudut masih dapat diterima | 2.046 | 99,90% | 39,49 | 0,759 | 204 px | 267 px | 324 |
| 2 | `4b546c12...164739` | Percobaan calibration pengguna kedua; ditolak | 0 | — | — | 0,000 | 749 px | 1.998 px | 324 |
| 3 | `b45e47f5...164949` | Pengguna kedua; calibration lolos tetapi corner tracking gagal | 3.104 | 99,87% | 39,48 | 0,652 | 296 px | 381 px | 325 |

### 5.1 Session pertama

Distribusi state:

- `on_screen`: 64,6%
- `gaze_outside_screen`: 34,9%
- `eyes_invalid_or_blink`: 0,4%
- `face_missing`: 0,1%

Prediksi smooth mencapai:

- X: `-310` hingga `3.505`
- Y: `-1.755` hingga `3.377`

Sebagian prediction masih keluar layar, terutama ke kanan dan bawah. Namun, P95 calibration error 267 px jauh lebih baik dibanding session lain. Ini sesuai dengan laporan bahwa gaze dapat mendekati padding dan hasilnya masih terasa dapat diterima.

### 5.2 Session kedua

Calibration mengumpulkan 324 unique frames tetapi menghasilkan:

- Quality: `0.0`
- Median validation error: 749 px
- P95 validation error: 1.998 px

Tracking tidak dimulai. Ini adalah perilaku yang benar: calibration buruk tidak lagi dibiarkan masuk ke tracking. Fakta bahwa seluruh frame sudah unik juga membuktikan bahwa kegagalan ini bukan lagi akibat duplicate-frame sampling.

### 5.3 Session ketiga

Distribusi state:

- `on_screen`: 41,0%
- `gaze_outside_screen`: 57,8%
- `eyes_invalid_or_blink`: 1,1%
- `face_missing`: 0,1%

Prediksi smooth mencapai:

- X: `0` hingga `3.358`
- Y: `-850` hingga `3.793`

Jumlah prediction di luar layar berdasarkan arah:

- Kanan: 904 frames
- Bawah: 764 frames
- Atas: 209 frames

Terdapat dua periode off-screen kontinu yang sangat panjang:

- Sekitar 9,3 detik mulai detik ke-1,8.
- Sekitar 11,0 detik mulai detik ke-34,6.

Analisis per lima detik menunjukkan:

- Detik 5–20: median X sekitar `2.937–2.989`, terus terdorong melewati sisi kanan layar.
- Detik 35–45: median Y sekitar `2.278–2.290`, terus terdorong melewati sisi bawah layar.
- Ketika kembali ke area yang dilaporkan sebagai tengah, sekitar detik 50–65, median X hanya sekitar `409–618`, sehingga sistem mengklasifikasikan gaze ke `sidebar`, bukan tengah layar.

Ini bukan sekadar EMA lag. EMA hanya memberi keterlambatan singkat; bias yang bertahan beberapa detik menunjukkan input feature-to-screen mapping telah bergeser terhadap pose saat calibration.

---

## 6. Diagnosis Akhir

### 6.1 Bukan cache

Cache hanya menyimpan dua background visual grid: normal dan off-screen. Cache tidak dibaca oleh calibration, tidak menyimpan mata pengguna, dan tidak mengubah mapper.

### 6.2 Bukan bottleneck performa

Ketiga session menunjukkan capture dan processing sekitar 39,5 FPS tanpa dropped frames. Total processing P95 tetap di bawah periode kamera. FPS sudah bukan penyebab utama error gaze.

### 6.3 Bukan kegagalan umum face detection atau lighting

Session pertama dan ketiga memiliki face detection sekitar 99,9%, dengan face missing hanya sekitar 0,1%. MediaPipe secara konsisten menemukan wajah. Lighting mungkin tetap memengaruhi detail iris, tetapi data tidak mendukung lighting sebagai akar penyebab utama kegagalan terakhir.

### 6.4 Duplicate-frame calibration sudah teratasi

Setiap calibration menggunakan 324–325 unique camera frames, sesuai kebutuhan minimum 270 sampel ditambah stability frames. Session gagal tetap gagal walaupun frame unik, sehingga masalah yang tersisa berada pada kualitas feature dan model.

### 6.5 Cross-user feature shift

Model saat ini mengasumsikan hubungan antara iris lokal, EAR, dan posisi layar relatif konsisten. Kenyataannya hubungan tersebut berubah karena:

- Bentuk mata dan kelopak berbeda antar pengguna.
- Iris visibility berbeda walaupun tanpa kacamata.
- Yaw dan pitch kepala belum dikompensasi; implementasi saat ini terutama mengurangi head roll.
- Pengguna dapat sedikit menggerakkan kepala ketika mencoba melihat sudut.
- Posisi kepala saat tracking dapat bergeser dari posisi saat calibration.

Session ketiga menunjukkan pola khas pose/feature shift: sudut kanan dan bawah menghasilkan extrapolation, lalu posisi yang seharusnya tengah tetap memiliki bias horizontal besar.

### 6.6 Unbounded regression extrapolation

Model X masih menggunakan polynomial orde dua. Calibration hanya memberi sembilan titik median. Ketika feature tracking keluar sedikit dari rentang feature calibration, polynomial dapat menghasilkan koordinat yang bertambah secara tidak terkendali.

Quality validation saat ini hanya menilai sembilan calibration targets dengan leave-one-out. Nilai `0.652` dapat lolos walaupun operational corner behavior buruk. Validation belum menguji:

- Target tambahan yang tidak digunakan untuk fitting.
- Perubahan pose kecil setelah calibration.
- Return-to-center setelah melihat sudut.
- Stability dan error per region dalam waktu tertentu.

### 6.7 Resolusi webcam bukan hardcoded coordinate bug

Capture meminta 640×480 dan MediaPipe memakai copy 320×240 dari konfigurasi YAML. Landmark MediaPipe bersifat normalized, sehingga resolusi tidak menyebabkan mismatch langsung antara kamera dan koordinat layar.

Namun, 320×240 dapat mengurangi detail iris untuk beberapa bentuk mata. Resolusi inference tetap perlu dibandingkan secara terkontrol dengan 480×360 dan 640×480. Mengubahnya tanpa benchmark dapat menurunkan FPS, sehingga belum dapat dinyatakan sebagai akar penyebab dari data saat ini.

---

## 7. Status Kesiapan Prototype

| Komponen | Status | Catatan |
|---|---|---|
| Capture dan processing FPS | Siap | Stabil sekitar 39,5 FPS |
| UI decoupling | Siap | UI tidak menahan inference |
| Duplicate-frame prevention | Siap | Calibration dan tracking memakai unique frame IDs |
| Padding/screen geometry | Siap | Black grid mewakili seluruh layar; padding visual-only |
| Calibration rejection | Berfungsi | Session buruk berhasil ditolak |
| Single-user center tracking | Cukup baik | Terbukti pada beberapa session |
| Corner tracking | Belum stabil | Prediction masih extrapolate keluar layar |
| Cross-user generalization | Belum siap | Pengguna kedua mengalami persistent mapping bias |
| Cognitive-state measurement | Belum tervalidasi | Akurasi gaze region harus stabil lebih dulu |

Prototype belum layak disebut production-ready untuk penggunaan lintas pengguna. Fondasi performa dan observability sudah kuat, tetapi accuracy pipeline masih memerlukan redesign dan validation lintas pengguna.

---

## 8. Rencana Perbaikan Berikutnya

### Prioritas 1 — Simpan data calibration per titik

Untuk setiap target, simpan:

- Seluruh unique eye features setelah outlier rejection.
- Median, standard deviation, MAD, dan jumlah rejected frames.
- Feature masing-masing mata sebelum dirata-ratakan.
- EAR, face confidence, iris visibility proxy, dan head pose.
- Target layar dan prediction model terhadap target tersebut.

Tanpa data ini, session summary hanya menunjukkan hasil akhir tetapi tidak dapat menentukan titik mana yang merusak model.

### Prioritas 2 — Tambahkan head pose dan pose gate

- Estimasikan yaw, pitch, dan roll dari face landmarks.
- Simpan pose baseline saat calibration.
- Tolak calibration sample ketika pose berubah terlalu jauh.
- Saat tracking, tandai `head_pose_out_of_range` daripada mengirim feature tersebut ke polynomial extrapolation.
- Pertimbangkan yaw/pitch sebagai feature mapper setelah dataset lintas pengguna tersedia.

### Prioritas 3 — Ganti mapper yang tidak bounded

Bandingkan dengan held-out target dan scripted test:

1. Regularized linear/ridge model dengan standardized features.
2. Piecewise bilinear interpolation berdasarkan calibration grid.
3. RBF/interpolator dengan output bounded.
4. Direct region classifier untuk kebutuhan section-level Adaptive IDE.

Untuk use case section tracking, direct region classification dapat lebih robust daripada memaksa estimasi pixel gaze yang presisi. Pixel regression tetap dapat dipakai sebagai visual/debug output.

### Prioritas 4 — Validation targets terpisah

Setelah fitting, tampilkan target validation tambahan yang tidak digunakan sebagai training data, termasuk:

- Empat area dekat sudut.
- Center.
- Center-bottom.
- Return-to-center setelah melihat dua sudut.

Calibration hanya diterima bila validation memenuhi seluruh batas berikut:

- Median error ≤150 px pada layar saat ini.
- P95 error ≤300 px.
- Tidak ada region dengan persistent off-screen prediction.
- Return-to-center kembali ke cell tengah dalam waktu ≤500 ms.

### Prioritas 5 — Benchmark resolusi inference

Uji orang dan pose yang sama menggunakan:

- 320×240
- 480×360
- 640×480

Bandingkan calibration P95, scripted region accuracy, inference P95, dan processing FPS. Pilih resolusi terendah yang memenuhi akurasi dan minimal 30 FPS.

### Prioritas 6 — Protokol evaluasi lintas pengguna

Gunakan minimal lima pengguna, tiga calibration per pengguna, dan urutan target yang sama. Catat:

- Calibration success rate.
- Region classification accuracy.
- False off-screen rate ketika melihat layar.
- Face/eye invalid rate.
- Return-to-center recovery.
- Median/P95 error per region dan per pengguna.

---

## 9. Kesimpulan

Sesi pengembangan ini berhasil menyelesaikan masalah performa, telemetry, duplicate frames, invalid calibration flow, dan geometri padding. Tiga session terakhir membuktikan bahwa pembenahan tersebut bekerja: FPS stabil, wajah terdeteksi, frame calibration unik, dan calibration buruk dapat ditolak.

Pengujian yang sama juga mengungkap batas berikutnya dengan jelas. Sistem masih belajar hubungan gaze dari feature yang belum head-pose invariant dan memetakan feature tersebut menggunakan regresi yang dapat extrapolate tanpa batas. Hasil bagus pada pengguna pertama dan area tengah belum menjamin generalisasi ke pengguna lain atau sudut layar.

Tahap berikutnya bukan lagi tuning padding atau FPS. Fokus harus berpindah ke dataset calibration per titik, head-pose gating, validation target terpisah, dan mapper yang bounded atau langsung mengklasifikasikan region.

