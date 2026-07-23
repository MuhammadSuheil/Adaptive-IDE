# Ringkasan Planning: Adaptive IDE Extension — Eye Tracking Module

> Dokumen ini merupakan ringkasan hasil planning awal untuk modul eye tracking pada proyek penelitian Adaptive IDE Extension berbasis VS Code.

---

## 1. Konteks Proyek

| Aspek | Detail |
|-------|--------|
| Jenis | Skripsi / Tugas Akhir + Riset Tim |
| Platform target | VS Code Extension |
| Pembagian tugas | Eye Tracking (penulis) + Heart Rate (anggota lain) |
| Hardware | Webcam laptop 30fps + Webcam eksternal 60fps |
| Output IDE | Adaptive font size & layout berdasarkan kondisi developer |

---

## 2. Arsitektur Sistem

Sistem terdiri dari dua proses terpisah yang berkomunikasi via WebSocket:

```
Webcam (30/60fps)
    ↓
[Python Service]
    MediaPipe — deteksi iris
    Gaze Estimation — iris → screen (x, y)
    Smoothing Filter — Kalman adaptive fps
    WebSocket Server @ localhost:8765
    ↓ JSON: {x, y, confidence, fps}
[VS Code Extension — Node.js]
    WebSocket Client — terima gaze data
    Hit Detection — screen coords → baris kode
    State Detection — gabung eye + heart rate
    Adaptive Logic — decide aksi
    VS Code API — apply font, layout, highlight
```

### Kenapa WebSocket?

- Real-time dan low latency
- Dua arah
- Mudah diimplementasi antara Python dan Node.js
- Jalan di localhost (tidak perlu internet)

---

## 3. Engine: MediaPipe

**Pilihan engine: MediaPipe (by Google)**

| Kriteria | Detail |
|----------|--------|
| Jenis | Standalone Python library |
| Setup | `pip install mediapipe` (~10 menit) |
| Performa 30fps | Excellent |
| Performa 60fps | Excellent |
| Latency | ~10–20ms |
| Keunggulan utama | 468 face landmarks + iris tracking built-in |
| CPU usage | ~15–25% (30fps) / ~25–40% (60fps) |

### Kenapa bukan yang lain?

- **WebGazer.js** → browser-based, tidak bisa dipakai di VS Code (Electron app)
- **GazeTracking** → hanya deteksi kiri/kanan/tengah, tidak ada screen coords
- **Tobii SDK** → butuh hardware mahal, skip
- **OpenCV + Dlib** → lebih fleksibel tapi setup lebih lama, MediaPipe sudah cukup

### Adaptif 30fps vs 60fps

MediaPipe otomatis mengikuti frame rate input dari OpenCV. Yang perlu disesuaikan adalah **smoothing filter**:
- 30fps → window smoothing lebih lebar
- 60fps → window smoothing lebih ketat

---

## 4. Pipeline Eye Tracking (6 Phase)

### Phase 1: Capture
- OpenCV ambil frame dari webcam
- Output: raw frame per 30/60fps

### Phase 2: Calibration
- User stare at 5 titik di layar
- Build mapping: iris position → screen coordinates
- Perlu recalibrate tiap ganti subjek (beda orang = beda eye geometry)
- Recalibration: **frequent** (prioritas data validity > kenyamanan subjek)
- Estimasi stabilitas: 5–10 menit per calibration

### Phase 3: Filtering (Smoothing)
- Raw gaze data itu noisy dan jittery
- Pakai **Kalman Filter** untuk smooth output
- Trade-off: lebih smooth = +20–30ms latency (masih acceptable)

bisa pake banyak filter nanti user bisa config

### Phase 4: Event Detection
- **Fixation**: gaze diam di satu titik >300ms = intentional look
- **Saccade**: gerakan cepat antar fixation (skip event saat saccade)
- **Hit detection**: screen (x, y) → baris kode ke-N

### Phase 5: Context Enrichment
- Gabungkan dengan heart rate data (dari modul teman)
- Gabungkan dengan editor state (ada error? tipe token?)
- Output: event bermakna, bukan hanya koordinat

### Phase 6: Adaptive Action
- Trigger perubahan UI di VS Code

---

## 5. Hit Detection: Screen → Baris Kode

### Rumus

```
posisi Y relatif ke editor  = gazeY - posisi atas editor
posisi Y dengan scroll      = posisi Y relatif + scrollTop
nomor baris                 = floor(posisi Y dengan scroll ÷ lineHeight)
```

### VS Code API yang dipakai

| API | Fungsi |
|-----|--------|
| `visibleRanges` | Baris mana yang sedang terlihat |
| `lineHeight` | Tinggi satu baris dalam pixel |
| `scrollTop` | Seberapa jauh user sudah scroll |
| `document.lineAt(N)` | Isi teks baris ke-N |
| `diagnostics` | Ada error/warning di baris tersebut? |

### Akurasi Target

- Target: **baris kode atau kelompok 2–5 kata**
- Tidak perlu per kata (50–80px error dari MediaPipe sudah cukup)
- Baris kode tingginya ~18–22px, dengan toleransi ±20px masih acceptable

---

## 6. Python Auto-Start dari Extension

Python service distart **otomatis** oleh VS Code Extension:

```
VS Code buka
    → Extension activate() dipanggil
    → Extension spawn Python child process
    → Python nyalain webcam + MediaPipe + WebSocket server
    → Extension connect sebagai WebSocket client
    → Gaze data mengalir
VS Code ditutup
    → Extension deactivate() dipanggil
    → Python process di-kill (tidak ada zombie process)
```

User tidak perlu buka terminal atau jalankan script manual.

---

## 7. User State Detection

Kombinasi sinyal eye tracking + heart rate menghasilkan 5 state:

### Focused 🟢
```
Sinyal:  Dwell lama (>2 detik) di satu area, HR normal
Aksi:    Zen mode, sembunyikan sidebar, highlight baris aktif,
         tahan notifikasi
```

### Confused 🟡
```
Sinyal:  Dwell lama di baris yang ada error/warning, HR tinggi
Aksi:    Auto-popup error explanation, gede-in font di baris itu,
         suggest dokumentasi terkait
```

### Scanning 🔵
```
Sinyal:  Gaze loncat-loncat cepat (saccade tinggi), HR normal
Aksi:    Minimap lebih besar, breadcrumb lebih prominent,
         font sedikit lebih kecil, tampilkan file outline
```

### Overloaded 🔴
```
Sinyal:  Gaze acak tidak berpola, HR sangat tinggi
Aksi:    Sembunyikan dekorasi noise, font lebih besar + line spacing,
         munculkan break reminder
```

### Vibecoding 🟣 *(angle penelitian tambahan)*
```
Sinyal:  Gaze bolak-balik antara AI panel ↔ editor secara berulang
Aksi:    Highlight diff AI suggestion vs kode asli,
         slow transition saat accept, track rasio kode AI vs manual
```

---

## 8. Adaptive Logic: Threshold itu Tidak Fix

Tiap orang punya baseline berbeda. Perlu **baseline recording** di awal sesi:
- Durasi: 2–3 menit sebelum mulai coding
- Yang direkam: baseline HR + baseline gaze pattern
- Dipakai sebagai referensi threshold per subjek

---

## 9. Hal yang Masih Perlu Dikonfirmasi ke Dospem

- [ ] Jumlah subjek penelitian (estimasi: 15–30 orang) 10 orang minimal
- [ ] Research question final
- [ ] Apakah vibecoding masuk sebagai variabel resmi atau hanya observasi tambahan. 
- [ ] Standar validasi data yang diharapkan
- [ ] Metode analisis data gaze (heatmap? fixation count? dwell time?) 

---

## 10. Tech Stack Summary

| Komponen | Teknologi |
|----------|-----------|
| Eye engine | MediaPipe (Python) |
| Webcam capture | OpenCV (Python) |
| Smoothing | Kalman Filter (Python) |
| Komunikasi | WebSocket (localhost:8765) |
| Extension | VS Code Extension API (TypeScript/Node.js) |
| IDE target | VS Code |

---

Tambahan dari ello
# Catatan Tambahan & Bahan Diskusi: Eye Tracking Module

Berdasarkan *draft planning* (`draft-1.md`) yang telah disusun, arsitektur dan pendekatannya sudah sangat baik. Berikut adalah beberapa poin teknis tambahan yang perlu didiskusikan lebih lanjut bersama tim, terutama karena sistem ini menggabungkan dua modul pengumpulan data yang berbeda (Eye Tracking dan Heart Rate).

## 1. Pentingnya Sinkronisasi Waktu (*Time Sync*)
Karena sistem ini melakukan **Fusi Sensor Multimodal**, pastikan setiap paket data (misalnya dalam format JSON) yang dikirim dari aplikasi Python (Eye Tracking) dan aplikasi Heart Rate menyertakan **Timestamp (dalam milidetik)** yang tersinkronisasi. Jika waktu perekamannya tidak sinkron, data HR dan gerakan mata akan sangat sulit dicocokkan ketika dianalisis oleh algoritma *Machine Learning* untuk menyimpulkan kondisi kognitif *developer*.

## 2. Hit Detection pada Sumbu X (Horizontal)
Pada *Phase 5: Hit Detection* di *draft*, rumus yang dicantumkan saat ini masih berfokus pada Sumbu Y (mencari baris kode ke-berapa). Mungkin perlu dipertimbangkan untuk memperhitungkan rentang Sumbu X (horizontal) juga. Hal ini penting untuk mendeteksi apakah *developer* benar-benar sedang membaca isi baris kode tersebut, ataukah matanya sedang melirik ke arah *Explorer/Sidebar* di sebelah kiri, ataupun menatap *Minimap* di sebelah kanan.


Pertanyaan dari mon
1. apakah scope penelitian ini hanya perubahan tampilan pada IDE atau lebih mendetail seperti tampilan baris mana yang harus di ubah karena error/bug