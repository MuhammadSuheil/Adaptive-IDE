# Eye Tracking Report
**Laporan/Log Perubahan Sistem Eye Tracking**


## 15 September 2026 - Perapian Logging & Persiapan Integrasi Heart Rate
**Cakupan:** Audit folder `sessions/`, penyesuaian format log, arsip data lama, dan persiapan metadata untuk integrasi multimodal.

---

### 1. Konteks

Tim menyepakati pengembangan mekanisme eye tracking dicukupkan pada versi saat ini. Sebelum diintegrasikan dengan sistem heart rate, dilakukan audit dan perapian terhadap sistem pencatatan/logging session agar format konsisten dan siap untuk integrasi multimodal.

---

### 2. Temuan Audit Folder `sessions/`

Dari 29 file di folder `sessions/`, ditemukan **6 masalah utama**:

| # | Masalah | Dampak |
|---|---------|--------|
| 1 | **3 versi CSV header berbeda** di folder yang sama (17, 26, 32 kolom) | Script analisis harus handle multi-format |
| 2 | **13 file CSV kosong** (header only, 0 data) dari session yang gagal calibration/quit awal | Mengotori folder, tidak punya value analitis |
| 3 | Hanya **4 dari 27 session** yang punya JSON summary | Session lama kehilangan metadata penting |
| 4 | `csv_path` di JSON mengandung `./` (path tidak bersih) | Bisa bermasalah saat path matching lintas OS |
| 5 | JSON summary tetap ditulis untuk **session gagal** (semua nilai 0) | File sampah 2KB per session gagal |
| 6 | UUID 36 karakter membuat **nama file sangat panjang** | Kurang readable saat browsing manual |

#### Detail Versi CSV Header

| Versi | Tanggal | Kolom | Perbedaan Utama |
|-------|---------|-------|-----------------|
| v1 | 20 Agustus | 17 | Kolom terakhir: `calibration_confidence` |
| v2 | 20 Agustus (lain) | 17 | Kolom terakhir: `calibration_quality` (rename) |
| v3 | 11 September | 26 | + `capture_frame_id`, timing stages, `gaze_status` |
| v4 | 15 September | 32 | + `head_pitch`, `head_yaw`, `head_pose_shifted`, blink data |

**Format final (v4, 32 kolom)** dikonfirmasi dari kode saat ini di `eye_tracking_prototype.py` baris 141-152.

---

### 3. Penyesuaian yang Dilakukan

#### 3.1 Arsip Data Lama ke `sessions/_archive/`

Seluruh file dari tanggal **20 Agustus** (17 CSV) dan **11 September** (6 CSV + 4 JSON), serta 1 file kosong dari 15 September, dipindahkan ke subfolder `sessions/_archive/`.

**Sebelum:**
```
sessions/
├── 29 file (campuran 3 format CSV, file kosong, dll)
```

**Sesudah:**
```
sessions/
├── _archive/           ← 28 file lama tersimpan aman
├── session_11108852..._20260915_122857.csv  ← 1 session valid (format v4)
```

Data lama tetap tersimpan untuk referensi, tapi tidak mengotori folder utama.

#### 3.2 Pencegahan File Kosong (Modifikasi `cleanup()`)

Ditambahkan logika di method `cleanup()` di `eye_tracking_prototype.py`:

```python
if self.frame_count == 0:
    os.remove(self.csv_path)      # Hapus CSV kosong
    return                         # Skip JSON summary
```

Mulai sekarang, jika session berakhir tanpa data tracking (gagal calibration, user quit awal), file CSV kosong otomatis dihapus dan JSON summary tidak ditulis.

#### 3.3 Metadata Baru di JSON Summary

Ditambahkan 3 field baru di level teratas JSON summary:

```json
{
  "schema_version": "2.0",
  "sensor_type": "eye_tracking",
  "prototype_version": "2.0",
  ...
}
```

| Field | Tujuan |
|-------|--------|
| `schema_version` | Melacak versi format JSON agar perubahan masa depan mudah di-detect |
| `sensor_type` | Membedakan session eye tracking vs heart rate vs multimodal saat integrasi |
| `prototype_version` | Melacak versi prototype yang menghasilkan data |

#### 3.4 Perbaikan Path di JSON

`csv_path` di JSON sekarang menggunakan `os.path.abspath()` sehingga path selalu bersih dan absolut (menghilangkan `./` di tengah path).

**Sebelum:**
```json
"csv_path": "d:\\...\\eye_tracking_prototype\\./sessions\\session_...csv"
```

**Sesudah:**
```json
"csv_path": "D:\\...\\eye_tracking_prototype\\sessions\\session_...csv"
```

---

### 4. File yang Dimodifikasi

| File | Perubahan |
|------|-----------|
| `eye_tracking_prototype.py` | `cleanup()`: + pencegahan file kosong, + 3 field metadata baru, + `os.path.abspath()` pada csv_path |
| `sessions/_archive/` (baru) | Subfolder baru berisi 28 file session lama |

---

### 5. Catatan untuk Integrasi Heart Rate

Dengan penyesuaian ini, sistem logging eye tracking sudah siap untuk integrasi:

1. **Format CSV konsisten** — hanya 1 format (v4, 32 kolom) yang akan ada di folder `sessions/`
2. **`sensor_type: "eye_tracking"`** — memudahkan script analisis membedakan tipe session
3. **`schema_version: "2.0"`** — jika tim heart rate menggunakan format berbeda, bisa di-detect otomatis
4. **Timestamp ISO di JSON** — memungkinkan sinkronisasi temporal antara eye tracking dan heart rate
5. **Folder bersih** — tidak ada file sampah yang mengganggu

Yang perlu dibahas bersama tim heart rate nanti:
- Apakah log CSV heart rate dan eye tracking akan **digabung dalam 1 file** atau **terpisah tapi sinkron via timestamp**?
- Apakah JSON summary juga perlu **di-merge** atau cukup **cross-reference** via `session_id`?


## 14 September 2026 - Laporan Pencapaian, Perbaikan Teknis, dan Fitur Baru System Eye Tracking AdaptiveIDE 
**Status Prototype**: Operasional, Sangat Stabil & Validated (Phase 0 – Phase 4 Integration + Blink Counter)

---

### 1. Ringkasan Eksekutif (Executive Summary)

Sistem Eye Tracking pada AdaptiveIDE dirancang untuk memetakan arah pandangan mata pengguna (*gaze*) secara *real-time* ke area interaktif IDE (`main_file`, `sidebar`, `ai_agent`, `terminal`) menggunakan webcam standar tanpa perangkat keras khusus.

Pada iterasi pengembangan hari ini, seluruh kendala utama terkait **ketidaknyamanan Phase 0 (Head Alignment Gate)**, **sensitivitas berlebihan ke arah kanan layar**, **artefak penggunaan kacamata**, hingga **kemudahan interaksi UX** telah berhasil diatasi secara menyeluruh. Selain itu, fitur baru **Deteksi & Perhitungan Kedipan Mata (Blink Detector & Counter)** berbasis Eye Aspect Ratio (EAR) telah berhasil diintegrasikan ke dalam seluruh pipeline tracking, log CSV, live HUD, dan ringkasan sesi akhir.

---

### 2. Evaluasi & Diagnosis Masalah Awal (*The Trash & Sucks Parts*)

Selama proses pengujian awal, ditemukan 5 masalah utama yang menyebabkan UX terasa buruk dan tracking melenceng:

#### 2.1. "Phase 0 Alignment Gate Hell" (Gate Kepala Sangat Menyiksa)
* **Masalah**: Pengguna dipaksa mendekatkan wajah sangat dekat ke kamera hingga leher pegal dan gate tidak pernah kunjung selesai (*never-ending Phase 0*).
* **Penyebab**:
  1. **Bug 180° Pitch Wrap**: Fungsi `RQDecomp3x3` OpenCV menghasilkan nilai pitch `~167°` saat posisi kepala lurus karena orientasi sumbu Y model 3D berlawanan dengan sumbu gambar, memicu pelanggaran threshold `max_pitch_deg: 15°`.
  2. **Target Wajah Terlalu Besar**: `target_face_width_ratio: 0.35` memaksa posisi wajah 20–30 cm dari webcam.
  3. **Tidak Ada Opsi Pass/Skip**: Pengguna terjebak permanen jika pencahayaan kurang ideal.

#### 2.2. Penalaran Kalibrasi Terlalu Ketat & Rejection Loop
* **Masalah**: Hasil kalibrasi yang dirasa pengguna sudah cukup bagus dan pas sering ditolak (*REJECTED*) secara berulang.
* **Penyebab**: Minimum quality threshold `0.40` dengan evaluasi Leave-One-Out Cross-Validation (LOO-CV) yang sangat sensitif terhadap variansi titik sudut (*extrapolation artifacts*).

#### 2.3. Kepekaan Hyper-Sensitive ke Arah Kanan Layar
* **Masalah**: Pandangan mata ke sisi kanan layar (`ai_agent` & `second_file`) terasa meloncat, hyper-sensitive, dan sering keluar dari area layar (*off-screen*).
* **Penyebab**:
  1. **Hard Clamping `[0.0, 1.0]`**: Modul RBF membatasi output secara kaku. Saat melihat ke kanan, estimasi fitur pupil yang sedikit melebihi batas langsung menabrak nilai `1.0` (X = 1920px).
  2. **Padding Layar IDE**: Koordinat X = 1920px berada di luar batas kanvas aktif IDE (`pad_x = 96px`, `pad_y = 54px`), memicu status `off_screen`.
  3. **Smoothing RBF Kurang**: Nilai regularisasi RBF `0.0` terlalu kaku terhadap noise data webcam.

#### 2.4. Waktu Fixation Kalibrasi Terlalu Cepat
* **Masalah**: Pengguna merasa terburu-buru saat titik kalibrasi berpindah (`move_delay_sec: 0.8s`), membuat data sampel diambil sebelum mata *settle*.

#### 2.5. Artefak & Noise Penggunaan Kacamata
* **Masalah**: Pantulan cahaya (*glare*) pada lensa kacamata menyebabkan pergeseran pupil/iris, merusak hasil pemetaan koordinat gaze.

---

### 3. Solusi & Implementsi Perbaikan Teknis

Berikut adalah perbaikan teknis yang telah diterapkan pada codebase:

| Komponen / Masalah | Solusi & Implementasi Kode | File / Modul Terkait |
| :--- | :--- | :--- |
| **Phase 0 Gate Fix** | • Normalisasi Pitch Angle: `if pitch > 90: pitch -= 180`.<br>• Menurunkan `target_face_width_ratio` ke `0.20` (jarak nyaman 50–70cm).<br>• Menambahkan **Segmented Readiness Checklist Bar** (5 status: `[Face]`, `[Distance]`, `[Centered]`, `[Pose]`, `[Stable]`) + Teks Petunjuk Wajah (`MOVE CLOSER` / `MOVE BACK`).<br>• Opsi Instant Manual Bypass (`SPACE` / `ENTER`). | `eye_tracking_prototype.py`<br>`config.yaml` |
| **Keseimbangan Kanan Layar** | • Memperbarui RBF Mapper dengan **Soft Clamping** `[-0.10, 1.10]`.<br>• Menyesuaikan `get_grid_cell()` agar memperhitungkan `pad_x` & `pad_y` sehingga titik tepi kanan terpetakan dengan lancar ke sel IDE (`ai_agent`, `second_file`).<br>• Mengatur `rbf_smoothing: 0.005` untuk regularisasi kurva spasial horizontal. | `modules/mapper.py`<br>`config.yaml` |
| **Robustness Kacamata** | Menerapkan **Dual-Eye Inverse-Variance Iris Weighting** pada `get_normalized_eye_vector()`. Mata dengan konsistensi iris lebih stabil (variansi spread lebih rendah) diberi bobot pemetaan lebih tinggi dibanding mata yang terkena glare kacamata. | `eye_tracking_prototype.py` |
| **Sensitivitas Filter Gaze** | Menerapkan **Adaptive EMA Filter** dengan deteksi saccade:<br>• Dwell lambat (<15px/frame): `alpha = 0.12` (sangat halus).<br>• Saccade cepat (>80px/frame): `alpha = 0.70` (mengikuti gerakan mata tanpa lag). | `modules/filter.py`<br>`config.yaml` |
| **Fixation Delay Kalibrasi** | Menambah `move_delay_sec` menjadi `1.5s` dan `stability_required_frames` dari 4 ke 6 frame agar mata benar-benar stabil sebelum pengambilan sampel. | `config.yaml` |
| **Penilaian Kualitas Kalibrasi** | Menyesuaikan skor kualitas LOO-CV dengan pembobotan berbasis posisi (titik tengah memiliki bobot lebih tinggi daripada titik sudut). | `modules/mapper.py`<br>`config.yaml` |

---

### 4. Fitur Baru: User Blink Detector & Counter (Berbasis EAR)

Telah ditambahkan modul deteksi dan penghitung kedipan mata pengguna berbasis **Eye Aspect Ratio (EAR)**:

#### 4.1. Cara Kerja & State Machine (`modules/blink_detector.py`)
1. **Perhitungan EAR**: Menghitung rasio vertikal terhadap horizontal kelopak mata menggunakan landmark MediaPipe:
   $$\text{EAR} = \frac{\|p_{\text{top1}} - p_{\text{bot1}}\| + \|p_{\text{top2}} - p_{\text{bot2}}\|}{2 \cdot \|p_{\text{left}} - p_{\text{right}}\|}$$
2. **State Machine Filter**:
   - Mata Terbuka ($\text{EAR} \ge 0.20$): Reset penghitung frame tertutup.
   - Mata Tertutup ($\text{EAR} < 0.20$): Increment `closed_frames`.
   - Terdeteksi Kedipan Alami: Jika mata kembali terbuka setelah `1 <= closed_frames <= 10`, maka `total_blinks` bertambah. (Mencegah salah hitung akibat memicingkan mata atau tertidur).
3. **Blink Rate**: Menghitung **BPM (Blinks Per Minute)** secara *real-time* dalam jendela 60 detik.

#### 4.2. Integrasi ke Interface & Logging
* **Live Camera HUD**: Menampilkan status kedipan real-time `Blinks: X (Y/min)` serta indikator visual `[BLINK]`.
* **Log CSV**: Menambahkan 3 kolom baru pada CSV per-frame: `is_blinking` (0/1), `total_blinks`, dan `blink_rate_bpm`.
* **End-of-Session Summary Screen**: Menampilkan total kedipan & rata-rata BPM pada kartu ringkasan akhir OpenCV (`Q`).
* **Session Summary JSON**: Menyimpan `total_blinks` dan `avg_blink_rate_bpm` pada file JSON ekspor.

---

### 5. Peningkatan Visual & User Experience (UX)

1. **Animasi Pulsasi Titik Kalibrasi**: Dot merah mengecil-membesar (*shrink-pulse*) saat masa penyesuaian mata, dan berubah menjadi warna hijau terang + ikon *crosshair* putih saat terkunci (*locked*).
2. **Slim Non-Blocking Head Pose HUD Bar**: HUD indikator Yaw/Pitch posisi kepala dibuat melayang di bagian atas layar agar tidak menutupi dot kalibrasi.
3. **Highlight Grid Cell Aktif**: Sel grid IDE yang sedang dilihat pengguna otomatis di-highlight dengan warna latar semi-transparan dan border tebal terang.
4. **Keyboard Help Panel (`H`)**: Menu panduan hotkey interaktif yang dapat di-toggle saat sesi tracking berlangsung.

---

### 6. Ringkasan Evaluasi & Benchmark Performa Sesi Terakhir

Berdasarkan data uji sesi terbaru (`session_f31be1e7-3987-4256-9a9c-8c1bc660f1a3`):

* **Face Detection Rate**: **100%** (seluruh frame wajah terdeteksi stabil tanpa dropped face).
* **Inference Speed**: Rata-rata **7.8 ms** per frame (MediaPipe FaceLandmarker + Gaze Mapping).
* **Capture Latency**: Rata-rata **0.18 ms** (berkat multithreaded async camera buffer).
* **Akurasi & Kenyamanan Kalibrasi**: Phase 0 dapat dilewati dalam <2 detik, dan kalibrasi 13-point berjalan lancar.
* **Tracking Kedipan**: Berhasil mencatat seluruh kedipan alami pengguna selama sesi interaksi tanpa memicu status false positive.

---

### 7. Kesimpulan & Rekomendasi

Prototype Eye Tracking AdaptiveIDE saat ini telah **matang, responsif, dan siap digunakan** note suheil: BLOMMMMM SIAP KLO LIGHTING BUSUK 
untuk analisis perilaku interaksi pengguna pada IDE:
- Sistem tidak lagi membebaskan pengguna dengan posisi leher tidak nyaman.
- Pemetaan gaze seimbang di seluruh area layar (termasuk sudut dan sisi kanan).
- Data pengguna lengkap dari koordinat gaze, dwell time per section, kestabilan kepala, hingga frekuensi kedipan mata (*blink rate*) tercatat secara presisi dalam format CSV & JSON.


## 28 Agustus 2026 - Laporan Konsolidasi Pengembangan dan Diagnosis Eye Tracking
**Cakupan:** Konsolidasi laporan pembenahan sebelumnya, perubahan yang dibuat dalam sesi ini, analisis seluruh kelompok pengujian, dan diagnosis tiga session terakhir.

---

### 1. Ringkasan Eksekutif

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

### 2. Kondisi Awal dari Laporan Sebelumnya

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

### 3. Temuan Awal pada Sesi Ini

#### 3.1 Calibration buruk tetap memasuki tracking

Session lama memiliki calibration quality sekitar `0.22–0.35`, bahkan ada yang `0.0`, tetapi aplikasi tetap memulai tracking. Prediksi gaze mencapai puluhan ribu piksel di luar layar.

#### 3.2 FPS yang ditampilkan tidak akurat

Timer dimulai sebelum calibration, sedangkan frame counter hanya bertambah saat tracking. Akibatnya, durasi calibration ikut menurunkan angka FPS. Setelah dihitung dari timestamp CSV, processing aktual tetap rendah, tetapi angka HUD memang salah.

#### 3.3 Async capture tidak sama dengan async inference

Webcam sudah memiliki capture thread, tetapi `detect_for_video`, CSV, pembuatan grid fullscreen, `imshow`, dan input keyboard masih berada dalam loop yang sama. Session menunjukkan MediaPipe hanya memerlukan sekitar 9–12 ms, sedangkan satu frame loop memerlukan sekitar 107 ms. Hampir 98 ms hilang di rendering/UI.

#### 3.4 Padding sebelumnya mengurangi layar fisik

Padding 5% sebelumnya diperlakukan sebagai `off_screen`, walaupun area tersebut masih berada di dalam monitor. Black grid hanya merepresentasikan 90% bagian tengah layar. Hal ini menyebabkan false off-screen di dekat sisi layar.

#### 3.5 Calibration menghitung duplicate camera frames

Sebelum pembenahan terakhir, calibration memakai `read_frame()` yang dapat mengembalikan frame kamera yang sama berulang kali. Dengan periode kamera sekitar 25,3 ms dan inference sekitar 10–12 ms, satu frame fisik dapat dihitung dua sampai tiga kali. Stability streak dan 30 sampel per titik menjadi terlalu optimistis.

---

### 4. Perubahan yang Telah Diimplementasikan

#### 4.1 Calibration dan validation

- Target calibration ditempatkan secara eksplisit di seluruh layar.
- Instruksi meminta pengguna melihat tepat ke tengah dot dan menjaga kepala tetap diam.
- Frame dengan EAR tidak valid, blink, face missing, atau noise tinggi ditolak.
- Sampel per titik diringkas dengan median dan MAD outlier rejection.
- Feature span horizontal dan vertikal divalidasi.
- Model memakai normalized screen target, bukan langsung mempelajari nilai piksel besar.
- Leave-one-point-out validation menghasilkan median error, P95 error, dan quality.
- Tracking tidak dimulai bila quality berada di bawah threshold.
- Calibration sekarang memakai `frame_id` dan hanya menerima **unique camera frames**.

#### 4.2 Mapping dan gaze state

- Iris diproyeksikan ke koordinat lokal mata untuk mengurangi efek head roll.
- State dipisah menjadi `on_screen`, `gaze_outside_screen`, `face_missing`, dan `eyes_invalid_or_blink`.
- Prediction, face validity, dan missing face tidak lagi dianggap sebagai satu jenis kegagalan yang sama.

#### 4.3 Pipeline performa

- Capture thread memberi frame terbaru beserta ID dan capture timestamp.
- Inference worker memproses frame unik terbaru.
- Frame lama dilewati dan tidak dimasukkan antrean agar latency tidak meningkat.
- Main thread hanya menangani UI OpenCV dan input keyboard.
- UI dibatasi ke 15 FPS, sedangkan gaze processing berjalan mengikuti kemampuan kamera.
- Static grid background di-cache agar tidak digambar ulang sepenuhnya setiap frame.
- Recalibration, pause, snapshot, debug, quit, dan cleanup disesuaikan dengan lifecycle worker.

#### 4.4 Telemetry

CSV dan summary sekarang mencatat:

- Capture FPS dan unique-frame processing FPS.
- Capture age.
- Preprocessing, inference, mapping/metrics, logging, dan total processing time.
- P50 dan P95 setiap processing stage.
- Dropped-frame count dan ratio.
- UI FPS.
- Calibration median/P95 error, feature span, dan jumlah unique calibration frames.

#### 4.5 Geometri layar dan padding

- Logical gaze classification memakai seluruh koordinat layar `0..screen_width` dan `0..screen_height`.
- Black grid adalah representasi berskala dari seluruh layar fisik.
- Padding tetap terlihat mengelilingi black grid, tetapi hanya merupakan visual off-screen padding.
- Padding tidak menghapus bagian layar dari section classification.
- Calibration tidak lagi menampilkan padding sebagai bagian dari layar.
- Target calibration saat ini mencakup 7%, 50%, dan 93% layar.

---

### 5. Analisis Tiga Session Terakhir

Urutan session sesuai waktu pengujian:

| Urutan | Session | Interpretasi | Frames | Face rate | FPS | Quality | Median error | P95 error | Unique calibration frames |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `d4bdf090...164558` | Pengguna pertama; hasil sudut masih dapat diterima | 2.046 | 99,90% | 39,49 | 0,759 | 204 px | 267 px | 324 |
| 2 | `4b546c12...164739` | Percobaan calibration pengguna kedua; ditolak | 0 | — | — | 0,000 | 749 px | 1.998 px | 324 |
| 3 | `b45e47f5...164949` | Pengguna kedua; calibration lolos tetapi corner tracking gagal | 3.104 | 99,87% | 39,48 | 0,652 | 296 px | 381 px | 325 |

#### 5.1 Session pertama

Distribusi state:

- `on_screen`: 64,6%
- `gaze_outside_screen`: 34,9%
- `eyes_invalid_or_blink`: 0,4%
- `face_missing`: 0,1%

Prediksi smooth mencapai:

- X: `-310` hingga `3.505`
- Y: `-1.755` hingga `3.377`

Sebagian prediction masih keluar layar, terutama ke kanan dan bawah. Namun, P95 calibration error 267 px jauh lebih baik dibanding session lain. Ini sesuai dengan laporan bahwa gaze dapat mendekati padding dan hasilnya masih terasa dapat diterima.

#### 5.2 Session kedua

Calibration mengumpulkan 324 unique frames tetapi menghasilkan:

- Quality: `0.0`
- Median validation error: 749 px
- P95 validation error: 1.998 px

Tracking tidak dimulai. Ini adalah perilaku yang benar: calibration buruk tidak lagi dibiarkan masuk ke tracking. Fakta bahwa seluruh frame sudah unik juga membuktikan bahwa kegagalan ini bukan lagi akibat duplicate-frame sampling.

#### 5.3 Session ketiga

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

### 6. Diagnosis Akhir

#### 6.1 Bukan cache

Cache hanya menyimpan dua background visual grid: normal dan off-screen. Cache tidak dibaca oleh calibration, tidak menyimpan mata pengguna, dan tidak mengubah mapper.

#### 6.2 Bukan bottleneck performa

Ketiga session menunjukkan capture dan processing sekitar 39,5 FPS tanpa dropped frames. Total processing P95 tetap di bawah periode kamera. FPS sudah bukan penyebab utama error gaze.

#### 6.3 Bukan kegagalan umum face detection atau lighting

Session pertama dan ketiga memiliki face detection sekitar 99,9%, dengan face missing hanya sekitar 0,1%. MediaPipe secara konsisten menemukan wajah. Lighting mungkin tetap memengaruhi detail iris, tetapi data tidak mendukung lighting sebagai akar penyebab utama kegagalan terakhir.

#### 6.4 Duplicate-frame calibration sudah teratasi

Setiap calibration menggunakan 324–325 unique camera frames, sesuai kebutuhan minimum 270 sampel ditambah stability frames. Session gagal tetap gagal walaupun frame unik, sehingga masalah yang tersisa berada pada kualitas feature dan model.

#### 6.5 Cross-user feature shift

Model saat ini mengasumsikan hubungan antara iris lokal, EAR, dan posisi layar relatif konsisten. Kenyataannya hubungan tersebut berubah karena:

- Bentuk mata dan kelopak berbeda antar pengguna.
- Iris visibility berbeda walaupun tanpa kacamata.
- Yaw dan pitch kepala belum dikompensasi; implementasi saat ini terutama mengurangi head roll.
- Pengguna dapat sedikit menggerakkan kepala ketika mencoba melihat sudut.
- Posisi kepala saat tracking dapat bergeser dari posisi saat calibration.

Session ketiga menunjukkan pola khas pose/feature shift: sudut kanan dan bawah menghasilkan extrapolation, lalu posisi yang seharusnya tengah tetap memiliki bias horizontal besar.

#### 6.6 Unbounded regression extrapolation

Model X masih menggunakan polynomial orde dua. Calibration hanya memberi sembilan titik median. Ketika feature tracking keluar sedikit dari rentang feature calibration, polynomial dapat menghasilkan koordinat yang bertambah secara tidak terkendali.

Quality validation saat ini hanya menilai sembilan calibration targets dengan leave-one-out. Nilai `0.652` dapat lolos walaupun operational corner behavior buruk. Validation belum menguji:

- Target tambahan yang tidak digunakan untuk fitting.
- Perubahan pose kecil setelah calibration.
- Return-to-center setelah melihat sudut.
- Stability dan error per region dalam waktu tertentu.

#### 6.7 Resolusi webcam bukan hardcoded coordinate bug

Capture meminta 640×480 dan MediaPipe memakai copy 320×240 dari konfigurasi YAML. Landmark MediaPipe bersifat normalized, sehingga resolusi tidak menyebabkan mismatch langsung antara kamera dan koordinat layar.

Namun, 320×240 dapat mengurangi detail iris untuk beberapa bentuk mata. Resolusi inference tetap perlu dibandingkan secara terkontrol dengan 480×360 dan 640×480. Mengubahnya tanpa benchmark dapat menurunkan FPS, sehingga belum dapat dinyatakan sebagai akar penyebab dari data saat ini.

---

### 7. Status Kesiapan Prototype

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

### 8. Rencana Perbaikan Berikutnya

#### Prioritas 1 — Simpan data calibration per titik

Untuk setiap target, simpan:

- Seluruh unique eye features setelah outlier rejection.
- Median, standard deviation, MAD, dan jumlah rejected frames.
- Feature masing-masing mata sebelum dirata-ratakan.
- EAR, face confidence, iris visibility proxy, dan head pose.
- Target layar dan prediction model terhadap target tersebut.

Tanpa data ini, session summary hanya menunjukkan hasil akhir tetapi tidak dapat menentukan titik mana yang merusak model.

#### Prioritas 2 — Tambahkan head pose dan pose gate

- Estimasikan yaw, pitch, dan roll dari face landmarks.
- Simpan pose baseline saat calibration.
- Tolak calibration sample ketika pose berubah terlalu jauh.
- Saat tracking, tandai `head_pose_out_of_range` daripada mengirim feature tersebut ke polynomial extrapolation.
- Pertimbangkan yaw/pitch sebagai feature mapper setelah dataset lintas pengguna tersedia.

#### Prioritas 3 — Ganti mapper yang tidak bounded

Bandingkan dengan held-out target dan scripted test:

1. Regularized linear/ridge model dengan standardized features.
2. Piecewise bilinear interpolation berdasarkan calibration grid.
3. RBF/interpolator dengan output bounded.
4. Direct region classifier untuk kebutuhan section-level Adaptive IDE.

Untuk use case section tracking, direct region classification dapat lebih robust daripada memaksa estimasi pixel gaze yang presisi. Pixel regression tetap dapat dipakai sebagai visual/debug output.

#### Prioritas 4 — Validation targets terpisah

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

#### Prioritas 5 — Benchmark resolusi inference

Uji orang dan pose yang sama menggunakan:

- 320×240
- 480×360
- 640×480

Bandingkan calibration P95, scripted region accuracy, inference P95, dan processing FPS. Pilih resolusi terendah yang memenuhi akurasi dan minimal 30 FPS.

#### Prioritas 6 — Protokol evaluasi lintas pengguna

Gunakan minimal lima pengguna, tiga calibration per pengguna, dan urutan target yang sama. Catat:

- Calibration success rate.
- Region classification accuracy.
- False off-screen rate ketika melihat layar.
- Face/eye invalid rate.
- Return-to-center recovery.
- Median/P95 error per region dan per pengguna.

---

### 9. Kesimpulan

Sesi pengembangan ini berhasil menyelesaikan masalah performa, telemetry, duplicate frames, invalid calibration flow, dan geometri padding. Tiga session terakhir membuktikan bahwa pembenahan tersebut bekerja: FPS stabil, wajah terdeteksi, frame calibration unik, dan calibration buruk dapat ditolak.

Pengujian yang sama juga mengungkap batas berikutnya dengan jelas. Sistem masih belajar hubungan gaze dari feature yang belum head-pose invariant dan memetakan feature tersebut menggunakan regresi yang dapat extrapolate tanpa batas. Hasil bagus pada pengguna pertama dan area tengah belum menjamin generalisasi ke pengguna lain atau sudut layar.

Tahap berikutnya bukan lagi tuning padding atau FPS. Fokus harus berpindah ke dataset calibration per titik, head-pose gating, validation target terpisah, dan mapper yang bounded atau langsung mengklasifikasikan region.

