# Ringkasan Planning: Adaptive IDE Extension — Eye Tracking Module

> Dokumen ini merupakan ringkasan hasil planning untuk modul eye tracking pada proyek penelitian Adaptive IDE Extension berbasis VS Code.
> **Versi: Planning v3** — diperbarui berdasarkan analisis 4 paper akademik.

> 📄 **Dokumen Terkait (dibuat berdasarkan analisis paper):**
> - [`analisis_paper_vs_rencana.md`](./analisis_paper_vs_rencana.md) — Ringkasan poin kunci tiap paper + gap analisis rencana vs paper
> - [`adaptive_features_paper_based.md`](./adaptive_features_paper_based.md) — Katalog lengkap fitur adaptif dengan sitasi paper

> 📚 **Paper yang Dianalisis** (tersedia di folder `/papers`):
> - **[P1]** *Adaptive IDEs and Developer Cognitive Load* — neuroergonomics, load shedding, FlowLight, adaptive scaffolding
> - **[P2]** *Biometrics, Cognitive Load, and DX* — DX 3 pilar, eye-mind hypothesis, Expertise Reversal Effect, CognitIDE
> - **[P3]** *Predicting Cognitive Load in Software Engineering* — NRevisit metric, XGBoost, AOI mapping, research gaps
> - **[P4]** *Towards Decoding Developer Cognition in the Age of AI Assistants* — CUPS taxonomy, vibecoding empiris, VS Code setup standar


## 1. Konteks Proyek

| Aspek | Detail |
|-------|--------|
| Jenis | Skripsi / Tugas Akhir + Riset Tim |
| Platform target | VS Code Extension |
| Pembagian tugas | Eye Tracking (penulis) + Heart Rate (anggota lain) |
| Hardware | Webcam laptop 30fps + Webcam eksternal 60fps |
| Output IDE | Adaptive font size & layout berdasarkan kondisi developer |
| Scope gaze | Per section IDE (bukan per baris/per kata) |
| Angle tambahan | Vibecoding behavior masuk sebagai variabel penelitian |

---

## 2. Arsitektur Sistem

Sistem terdiri dari dua proses terpisah yang berkomunikasi via WebSocket:

```
Webcam (30/60fps)
    ↓
[Python Service]
    MediaPipe — deteksi iris
    Gaze Estimation — iris → screen (x, y)
    Smoothing Filter — pilihan filter dari config
    Grid Classifier — (x, y) → section IDE
    WebSocket Server @ localhost:8765
    ↓ JSON: {section, confidence, fps, dwellTime}
[VS Code Extension — Node.js]
    WebSocket Client — terima gaze data
    Section Detection — validasi section aktif
    State Detection — gabung eye + heart rate
    Adaptive Logic — decide aksi per section
    VS Code API — apply font, layout, highlight
```

### Kenapa WebSocket?

- Real-time dan low latency
- Dua arah
- Mudah diimplementasi antara Python dan Node.js
- Jalan di localhost (tidak perlu internet)

---

## 3. Pendekatan Baru: Grid-Based Gaze Detection

**Update dari dospem:** gaze detection tidak perlu sampai level koordinat pixel atau baris kode. Cukup deteksi **section mana** yang sedang dilihat user di dalam IDE.

### Konsep Grid

Layar dibagi menjadi grid (bisa dikonfigurasi), lalu setiap cell grid di-mapping ke section IDE:

```
Contoh Grid 3x3:
┌─────────────┬─────────────┬─────────────┐
│  Sidebar    │  Main File  │  AI Agent   │
├─────────────┼─────────────┼─────────────┤
│  Sidebar    │  Main File  │  Second File│
├─────────────┼─────────────┼─────────────┤
│  Terminal   │  Terminal   │  Terminal   │
└─────────────┴─────────────┴─────────────┘
```

### Sections yang Dideteksi

| Section | Keterangan |
|---------|------------|
| `sidebar` | File explorer, extensions panel, source control |
| `main_file` | Editor utama tempat developer coding |
| `second_file` | Editor kedua (split view / reference file) |
| `terminal` | Integrated terminal VS Code |
| `ai_agent` | Panel Copilot, ChatGPT, atau AI agent lainnya |

### Cara Kerja Grid Classifier

```
Gaze (x, y) dari MediaPipe
    ↓
Hitung posisi di grid:
    col = floor(x / (screenWidth  ÷ gridCols))
    row = floor(y / (screenHeight ÷ gridRows))
    ↓
Lookup mapping: grid[row][col] → section name
    ↓
Output: { section: "main_file", confidence: 0.91 }
```

### Keuntungan Pendekatan Grid

- Jauh lebih simpel dari pixel-exact hit detection
- Tidak perlu track scroll position, line height, dll
- Lebih robust terhadap gaze error (50–80px tidak jadi masalah)
- Mudah dikustomisasi ukuran gridnya via config
- Sesuai scope skripsi

---

## 4. Engine: MediaPipe

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

MediaPipe otomatis mengikuti frame rate input dari OpenCV. Yang perlu disesuaikan adalah **smoothing filter** (bisa dipilih dari config):
- 30fps → window smoothing lebih lebar
- 60fps → window smoothing lebih ketat

---

## 5. Pipeline Eye Tracking (6 Phase)

### Phase 1: Capture
- OpenCV ambil frame dari webcam
- Frame rate mengikuti webcam yang dipakai (30 atau 60fps)
- Output: raw frame

### Phase 2: Calibration
- User stare at titik-titik kalibrasi di layar (jumlah titik = jumlah cell grid)
- Build mapping: iris position → screen coordinates
- Perlu recalibrate tiap ganti subjek (beda orang = beda eye geometry)
- Recalibration: **frequent** (prioritas data validity > kenyamanan subjek)
- Estimasi stabilitas: 5–10 menit per calibration

### Phase 3: Filtering (Smoothing)
- Raw gaze data itu noisy dan jittery
- Filter bisa dipilih dari config extension (lihat Section 7)
- Trade-off: lebih smooth = latency bertambah

### Phase 4: Grid Classification
- Koordinat (x, y) di-mapping ke cell grid
- Cell grid di-mapping ke section IDE
- Output: nama section + confidence score

### Phase 5: Context Enrichment
- Gabungkan dengan heart rate data (dari modul teman)
- Gabungkan dengan section context (apa yang ada di section itu?)
- Tambahan: apakah gaze sedang di AI agent panel? → sinyal vibecoding
- Output: event bermakna dengan context lengkap

### Phase 6: Adaptive Action
- Trigger perubahan UI di VS Code berdasarkan section + state

---

## 6. Python Auto-Start dari Extension

Python service distart **otomatis** oleh VS Code Extension:

```
VS Code buka
    → Extension activate() dipanggil
    → Extension baca config (grid size, filter, dll)
    → Extension spawn Python child process dengan config sebagai argumen
    → Python nyalain webcam + MediaPipe + WebSocket server
    → Extension connect sebagai WebSocket client
    → Gaze data mengalir
VS Code ditutup
    → Extension deactivate() dipanggil
    → Python process di-kill (tidak ada zombie process)
```

User tidak perlu buka terminal atau jalankan script manual.

---

## 7. Extension Config & Settings

Berdasarkan arahan dospem, extension memiliki **halaman settings** yang bisa dikustomisasi user/peneliti.

### Config yang Tersedia

```jsonc
// settings.json (contoh)
{
  // Ukuran grid untuk gaze zone detection
  "adaptiveIDE.grid.size": "3x3",          // pilihan: "2x2", "3x3", "4x4", "custom"
  "adaptiveIDE.grid.customRows": 3,         // aktif jika size = "custom"
  "adaptiveIDE.grid.customCols": 4,         // aktif jika size = "custom"

  // Smoothing filter untuk gaze data
  "adaptiveIDE.filter.type": "kalman",      // pilihan: "kalman", "ema", "median", "none"
  "adaptiveIDE.filter.kalman.processNoise": 0.1,
  "adaptiveIDE.filter.kalman.measurementNoise": 4.0,
  "adaptiveIDE.filter.ema.alpha": 0.3,     // Exponential Moving Average coefficient
  "adaptiveIDE.filter.median.windowSize": 5,

  // Dwell time threshold (ms) sebelum gaze dianggap intentional
  "adaptiveIDE.dwell.threshold": 300,

  // Webcam settings
  "adaptiveIDE.webcam.fps": 60,             // pilihan: 30, 60
  "adaptiveIDE.webcam.deviceIndex": 0,      // index kamera (0 = default)

  // Section mapping (bisa dikustomisasi sesuai layout IDE user)
  "adaptiveIDE.sections.enabled": [
    "sidebar", "main_file", "second_file", "terminal", "ai_agent"
  ],

  // Adaptive behavior toggle per section
  "adaptiveIDE.adaptive.enabled": true,
  "adaptiveIDE.adaptive.fontSizeChange": true,
  "adaptiveIDE.adaptive.layoutChange": true,
  "adaptiveIDE.adaptive.notificationSuppress": true,

  // Logging untuk keperluan penelitian
  "adaptiveIDE.research.logGazeData": true,
  "adaptiveIDE.research.logFilePath": "./gaze_log.json"
}
```

### Filter Options

| Filter | Karakteristik | Cocok untuk |
|--------|--------------|-------------|
| `kalman` | Akurasi tinggi, latency +20–30ms | Default, most accurate |
| `ema` | Simpel, latency rendah, sedikit jitter | Webcam 60fps |
| `median` | Robust terhadap outlier/blink | Lighting tidak stabil |
| `none` | Raw data, tanpa smoothing | Debug / testing |

---

## 8. User State Detection

Kombinasi sinyal eye tracking (section) + heart rate menghasilkan 5 state:

### Focused 🟢
```
Sinyal:  Dwell lama (>2 detik) di main_file, HR normal
Aksi:    Zen mode, sembunyikan sidebar,
         tahan notifikasi, highlight section aktif
```

### Confused 🟡
```
Sinyal:  Dwell lama di section yang ada error, HR tinggi
Aksi:    Auto-popup error explanation,
         gede-in font di main editor, suggest dokumentasi
```

### Scanning 🔵
```
Sinyal:  Gaze loncat antar section cepat, HR normal
Aksi:    Minimap lebih besar, breadcrumb prominent,
         tampilkan file outline, font sedikit lebih kecil
```

### Overloaded 🔴
```
Sinyal:  Gaze acak tidak berpola antar section, HR sangat tinggi
Aksi:    Sembunyikan dekorasi noise, font lebih besar + line spacing,
         munculkan break reminder
```

### Vibecoding 🟣
```
Sinyal:  Gaze bolak-balik antara ai_agent ↔ main_file berulang
Pattern: ai_agent (baca) → main_file (accept) → ai_agent (prompt) → repeat
Aksi:    Highlight diff AI suggestion vs kode asli,
         slow transition saat accept,
         track & log rasio waktu di ai_agent vs main_file
```

---

## 9. Adaptive Logic: Threshold itu Tidak Fix

Tiap orang punya baseline berbeda. Perlu **baseline recording** di awal sesi:
- Durasi: 2–3 menit sebelum mulai coding
- Yang direkam: baseline HR + baseline gaze pattern per section
- Dipakai sebagai referensi threshold per subjek
- Threshold bisa di-override manual dari config jika diperlukan

---

## 10. Hal yang Masih Perlu Dikonfirmasi ke Dospem

- [ ] Jumlah subjek penelitian (estimasi: 15–30 orang)
- [ ] Research question final
- [ ] Grid size default yang dipakai untuk penelitian (3x3 atau lain?)
- [ ] Section mana saja yang wajib ada vs opsional
- [ ] Metode analisis data gaze (heatmap per section? fixation count? dwell time per section?)
- [ ] Standar validasi data yang diharapkan
- [ ] Format log data untuk keperluan analisis

---

## 11. Tech Stack Summary

| Komponen | Teknologi |
|----------|-----------|
| Eye engine | MediaPipe (Python) |
| Webcam capture | OpenCV (Python) |
| Gaze zone | Grid-based classifier (custom) |
| Smoothing | Kalman / EMA / Median (pilihan via config) |
| Komunikasi | WebSocket (localhost:8765) |
| Extension | VS Code Extension API (TypeScript/Node.js) |
| Config | VS Code settings.json |
| IDE target | VS Code |

---

## 12. Update dari Analisis Paper (Planning v3)

> Bagian ini menambahkan temuan dari analisis 4 paper akademik.
> Detail lengkap lihat [`analisis_paper.md`](./analisis_paper.md).

### 12.1 Metric Eye Tracking yang Ditambahkan

Berdasarkan **[P3]** — *"NRevisit showed a near-perfect correlation (0.78–0.91) with EEG-measured cognitive load"* dan **[P1]** — *"Task-Evoked Pupillary Response (TEPR)"*:

| Metric Baru | Cara Dapat | Justifikasi Paper |
|------------|-----------|-----------------|
| **NRevisit count** per section | Counter kembali ke section yang sudah dikunjungi | [P3] korelasi 0.78–0.91 dengan EEG load |
| **Gaze transition rate** | Jumlah perpindahan section per detik | [P1][P2] saccade analog |
| **Iris size delta** (approx pupil) | Iris landmark distance dari MediaPipe | [P1][P2][P3] pupil = top predictor |

### Pipeline Eye Tracking (Diperbarui)

```
Webcam (30/60fps)
    ↓
[Python Service]
    MediaPipe — deteksi iris + iris size
    Gaze Estimation — iris → screen (x, y)
    Iris Size Delta — estimasi pupil dilation   ← BARU [P1][P3]
    Smoothing Filter — pilihan filter dari config
    Grid Classifier — (x, y) → section IDE
    NRevisit Counter — track section revisits    ← BARU [P3]
    Transition Rate — hitung perpindahan/detik   ← BARU [P1][P2]
    WebSocket Server @ localhost:8765
    ↓ JSON: {section, confidence, fps, dwellTime, nrevisit, transitionRate, irisDelta}
[VS Code Extension — Node.js]
    WebSocket Client — terima gaze data
    State Detection — gabung eye + heart rate (diperkuat signal baru)
    Adaptive Logic — decide aksi per section
    VS Code API — apply font, layout, highlight
```

### 12.2 State Detection Diperkuat

Berdasarkan sinyal baru dari paper, state detection menjadi lebih robust:

```
FOCUSED 🟢   = dwell_main > 2s + nrevisit rendah + transition_rate rendah + iris stabil + HR normal
CONFUSED 🟡  = nrevisit_section > 3 + iris_delta naik + HR naik
SCANNING 🔵  = transition_rate > 1.5/s + avg_dwell < 800ms + HR normal
OVERLOADED 🔴 = semua sinyal semrawut + iris_delta sangat naik + HR sangat tinggi
VIBECODING 🟣 = alternating ai_agent ↔ main_file ≥ 3 siklus + nrevisit ke AI setelah accept
```

### 12.3 Fitur Adaptif IDE (Ringkasan dari Paper)

> Detail lengkap + sitasi paper lihat [`adaptive_features_paper_based.md`](./adaptive_features_paper_based.md)

7 kategori fitur yang bisa di-adaptasi berdasarkan state developer:

| Kategori | Contoh Aksi | Basis Paper |
|---------|------------|------------|
| **Visual/Typography** | Font size, line height, color theme | [P1][P2] |
| **Layout/Panel** | Sidebar, minimap, zen mode, outline | [P1][P2] |
| **Editor Behavior** | Code folding, hover doc, inline hints | [P1][P2][P3] |
| **Notifikasi** | FlowLight pattern suppression, break reminder | [P1][P2] |
| **AI Integration** | Suggestion rate, vibecoding mode, diff highlight | [P1][P4] |
| **Gaze-Driven** | Section glow, research dashboard | [P1][P3][P4] |
| **Adaptive Scaffolding** | Novice vs expert profiling, instructional fading | [P1][P2] |

### 12.4 Ethical Framework (Wajib Ditambahkan)

Berdasarkan **[P1]** — *"Brain data cannot be treated as a standard corporate analytics asset"* dan **[P3]** — *"Critical ethical research gap: neglecting governance frameworks"*:

| Prinsip | Implementasi |
|---------|------------|
| **Informed consent** | Consent dialog saat extension pertama aktif |
| **Local data only** | Log disimpan di mesin lokal, tidak ke cloud |
| **User control** | Pause/stop kapanpun dari status bar |
| **Data anonymization** | Session ID (bukan nama) dalam semua log |
| **Transparency** | User tahu metric apa yang diukur |

### 12.5 Kontribusi Orisinal yang Bisa Diklaim

| Kontribusi | Mengapa Novel |
|-----------|--------------|
| **Vibecoding State** | Belum ada implementasi real-time di IDE nyata [P4][P1] |
| **NRevisit-triggered intervention** | Trigger dari pola kembali, bukan hanya dwell time [P3] |
| **Section-based research dashboard** | Per zone IDE, bukan per pixel/baris [P3][P4] |
| **AI usage ratio tracking** | Proxy comprehension debt secara real-time [P1][P4] |

---

## 13. Hal yang Masih Perlu Dikonfirmasi ke Dospem (Updated)

- [ ] Jumlah subjek penelitian (estimasi: 15–30 orang)
- [ ] Research question final
- [ ] Grid size default yang dipakai untuk penelitian (3x3 atau lain?)
- [ ] Section mana saja yang wajib ada vs opsional
- [ ] Metode analisis data gaze (heatmap per section? fixation count? dwell time per section?)
- [ ] Standar validasi data yang diharapkan
- [ ] Format log data untuk keperluan analisis
- [ ] **[BARU]** Apakah NRevisit counter perlu masuk sebagai metric utama penelitian?
- [ ] **[BARU]** Apakah iris size delta dari MediaPipe cukup akurat untuk penelitian atau hanya fitur tambahan?
- [ ] **[BARU]** Perlu ethical framework formal (IRB approval) atau cukup consent form sederhana?
- [ ] **[BARU]** Vibecoding state — apakah ini masuk sebagai research question tersendiri?

---

*Dokumen ini dibuat sebagai bahan diskusi dengan dosen pembimbing.*
*Versi: Planning v3 — diperbarui berdasarkan analisis 4 paper akademik.*
*Lihat juga: `analisis_paper_vs_rencana.md` dan `adaptive_features_paper_based.md`*
