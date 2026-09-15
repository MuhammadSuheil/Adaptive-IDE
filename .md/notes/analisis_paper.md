# 📊 Analisis Paper vs Rencana: Adaptive IDE Eye Tracking

> Perbandingan rencana `ringkasan_adaptive_ide_eye_tracking.md` (Planning v2)
> dengan temuan dari **6 paper akademik** (termasuk 2 paper eye tracking baru).

---

## 📚 Referensi Paper

| Kode | Judul Lengkap |
|------|--------------|
| **[P1]** | *Adaptive Integrated Development Environments: Mitigating Software Developer Cognitive Load Through Neuroergonomics and Context-Aware AI* |
| **[P2]** | *The Biometric Assessment of Cognitive Load in Software Engineering: A Multi-Dimensional Analysis of Developer Experience and Predictive Modeling* |
| **[P3]** | *Predicting Cognitive Load in Software Engineering via Machine Learning and Biometrics: State of the Art and Research Gaps* |
| **[P4]** | *Towards Decoding Developer Cognition in the Age of AI Assistants* — AlHaque et al., George Mason University & Virginia Tech, arXiv Jan 2025 |
| **[P5]** | *Eye-Tracking System with Low-End Hardware: Development and Evaluation* — Iacobelli et al., Sapienza University of Rome, Information 2023 |
| **[P6]** | *Real-Time Gaze Estimation Using Webcam-Based CNN Models for Human–Computer Interactions* — Vidhya & Resende Faria, University of Hertfordshire, Computers 2025 |

---

## 📄 Ringkasan Poin Kunci Per Paper

### 📖 Paper 1 — Adaptive IDEs and Developer Cognitive Load [P1]

**Fokus:** Fondasi neuroergonomics untuk IDE adaptif + intervensi konkret berbasis biometrik.

| # | Poin Kunci | Relevansi ke Proyek |
|---|-----------|-------------------|
| 1 | IDE konvensional masih **pasif & statis** — paradigma raw information presentation | Justifikasi mengapa proyek ini penting |
| 2 | CLT: 3 tipe load — Intrinsic, **Extraneous** (harus dikurangi), **Germane** (harus dimaksimalkan) | Kerangka teori untuk semua aksi adaptif |
| 3 | 8 sumber cognitive load: Task, Environment, Info, **Tool**, Communication, **Interruption**, Structure, Temporal | Tool + Interruption adalah yang bisa kita tackle |
| 4 | **TEPR**: pupil diameter = top biometrik indicator, korelasi langsung dengan working memory load | Justifikasi menambah pupil size metric |
| 5 | Eye tracking bisa dipetakan ke **Abstract Syntax Tree** → tahu *exact* kode mana yang membingungkan | Inspirasi NRevisit-triggered hover doc |
| 6 | **HRV/LF/HF ratio** = indikator robust cognitive overload | Validasi penggunaan heart rate modul teman |
| 7 | **Load Shedding**: IDE otomatis sembunyikan info rendah prioritas saat developer overloaded | Justifikasi sidebar/panel collapse |
| 8 | **FlowLight**: biometric-based interruptibility indicator, "defer-to-breakpoint" | Justifikasi notification suppression |
| 9 | GenAI = **"difficulty restructurer"** bukan "difficulty eliminator" — verification load muncul | Konteks untuk vibecoding state |
| 10 | **Comprehension Debt** (CD): gap antara yang tahu vs yang dibutuhkan untuk maintain kode | Urgency vibecoding state tracking |
| 11 | **Adaptive Scaffolding + Instructional Fading**: kurangi AI support bertahap saat developer makin ahli | Basis expertise-aware feature |
| 12 | Adaptive IDE: task completion **−22.7%**, cognitive load **−25%**, satisfaction **+41.9%** | Dampak positif yang bisa diklaim |
| 13 | **Etika**: neural data non-replaceable jika bocor — perlu neurorights framework | Gap yang perlu ditambahkan ke skripsi |

---

### 📖 Paper 2 — Biometrics, Cognitive Load, and DX [P2]

**Fokus:** Developer Experience (DX) 3 pilar + biometrik sebagai objektif measure + Expertise Reversal Effect.

| # | Poin Kunci | Relevansi ke Proyek |
|---|-----------|-------------------|
| 1 | DX 3 pilar: **Flow state, Feedback loops, Cognitive load** — saling berkaitan | Framework evaluasi keberhasilan IDE adaptif |
| 2 | DX bagus → 33% lebih likely capai business goals, 20% retention lebih tinggi | Business justification penelitian |
| 3 | **"Eye-mind hypothesis"**: tidak ada lag antara fixation dan mental processing | Validasi bahwa eye tracking = proxy kognitif valid |
| 4 | Fixation duration **naik** = load tinggi, Blink rate **turun** = fokus tinggi | Metrics tambahan yang bisa dikembangkan |
| 5 | **Uwano study**: developer top scan *seluruh* kode dulu sebelum detail → Scanning state ada basis empiris | Validasi scanning state dalam rencana |
| 6 | Expert: gaze **focal** menuju beacon kritis. Novice: gaze linear/broad | Basis developer profiling (novice vs expert) |
| 7 | **HRV rendah** = predictor stress + burnout jangka pendek | Validasi penggunaan HRV dari modul teman |
| 8 | Random Forest akurasi **0.991** untuk klasifikasi cognitive strain selama coding | Referensi jika mau upgrade ke ML classifier |
| 9 | Multimodal fusion (EEG + eye + HRV) → akurasi **98%** vs unimodal 40–96% | Justifikasi kombinasi eye + heart rate |
| 10 | **Expertise Reversal Effect (ERE)**: scaffold yang bantu novice bisa *ganggu* expert | Basis adaptive scaffolding per expertise level |
| 11 | **CognitIDE** (IntelliJ plugin) + **Lab Streaming Layer**: proof-of-concept biometrik IDE real | Validasi feasibility proyek kita |
| 12 | Etika: biometric data unik, non-replaceable — coercion risk di workplace | Memperkuat urgensi ethical framework |

---

### 📖 Paper 3 — Predicting Cognitive Load in Software Engineering [P3]

**Fokus:** Survey komprehensif ML + biometrik untuk prediksi cognitive load, termasuk research gaps.

| # | Poin Kunci | Relevansi ke Proyek |
|---|-----------|-------------------|
| 1 | NASA-TLX (self-report): retrospective, ada recall bias, tidak bisa real-time | Justifikasi pakai biometrik objektif |
| 2 | **EEG = gold standard** biometrik — alpha suppression + theta increase = overload | Konteks: kita pakai proxy yang lebih ringan |
| 3 | **Eye tracking**: pupil size + fixation dispersion = **top predictor** menurut XGBoost | Justifikasi tambah pupil metric dari MediaPipe |
| 4 | **NRevisit** berkorelasi **0.78–0.91** dengan EEG-measured cognitive load | **Justifikasi kuat** untuk tambah NRevisit counter |
| 5 | **EMIP Dataset**: 216+ programmer, 250Hz, benchmark standard eye tracking + code | Referensi dataset yang bisa dibandingkan |
| 6 | **XGBoost** unggul multimodal: akurasi 0.86, F1 0.78 | Referensi jika upgrade ke ML |
| 7 | CNN + LSTM hybrid: **98–99% akurasi** di lab controlled | Upper bound akurasi yang achievable |
| 8 | **Cross-subject generalizability**: akurasi drop 95% → ~70% saat lintas subjek | Justifikasi baseline per subjek di rencana kita |
| 9 | Real-time latency adalah tantangan besar — deep model bisa lambat | Justifikasi pakai rule-based dulu, bukan ML berat |
| 10 | EEG terlalu intrusif untuk real workplace | Justifikasi kenapa pakai webcam + HR wearable |
| 11 | **Eye-tracking mapped to Areas of Interest (AOIs)** dalam IDE | Validasi section-based gaze approach kita |
| 12 | Feature-level fusion (intermediate) = pendekatan paling umum dan sukses | Referensi cara gabung eye + HR features |
| 13 | **Ethical gap kritis**: literatur fokus akurasi, kurang governance & consent | Gap yang perlu diisi di skripsi |
| 14 | Data harus **developer-controlled**, bukan management-controlled | Prinsip desain ethical untuk proyek ini |

---

### 📖 Paper 4 — Towards Decoding Developer Cognition in the Age of AI Assistants [P4]

**Fokus:** Studi observasional — bagaimana AI assistant mempengaruhi cognitive load, CUPS taxonomy.

| # | Poin Kunci | Relevansi ke Proyek |
|---|-----------|-------------------|
| 1 | AI assistant: perceived productivity ≠ actual productivity | Konteks research question yang bisa dieksplorasi |
| 2 | AI meningkatkan output tapi menambah verification/validation cognitive demand | Justifikasi vibecoding state perlu dideteksi |
| 3 | **Tobii Pro Fusion** + **Emotiv EPOC X EEG** (14 ch) = setup hardware standar | Referensi setup yang lebih proper (kita pakai webcam = tradeoff) |
| 4 | **VS Code** + standardized setup = kontrol environment penelitian | Validasi pilihan VS Code platform kita |
| 5 | **CUPS taxonomy**: 12 state coding activity termasuk *Deferring Thought for Later* | Basis empiris vibecoding state kita |
| 6 | *"Deferring Thought for Later"* = accept AI suggestion tanpa verifikasi → vibecoding | Definisi empiris vibecoding behavior |
| 7 | Less experienced developer pakai AI **lebih sering** | Basis developer profiling |
| 8 | **2-menit relaxation baseline** sebelum task untuk establish cognitive baseline | Validasi baseline recording di rencana kita |
| 9 | Studi pakai **between-subjects design** (AI vs non-AI group) | Referensi metodologi penelitian |
| 10 | Interaction data + screen recording untuk post-hoc analysis | Justifikasi research logging + research dashboard |

---

### 📖 Paper 5 — Eye-Tracking System with Low-End Hardware [P5]

**Fokus:** Implementasi nyata MediaPipe Face Mesh untuk iris/gaze tracking berbasis webcam RGB + validasi vs EyeLink 1000 Plus.

| # | Poin Kunci | Relevansi ke Proyek |
|---|-----------|-------------------|
| 1 | MediaPipe **Face Mesh = 468 3D facial landmarks** — iris dapat dideteksi via landmark kontur mata | **Konfirmasi direct** bahwa MediaPipe bisa untuk gaze |
| 2 | **Iris Detection Algorithm (IDA)**: morphological transformations (dilation+erosion) + adaptive threshold + median blur → deteksi iris center | Teknik yang bisa diadopsi untuk pupil size estimation |
| 3 | **9-point calibration** untuk mapping posisi mata → posisi screen | Validasi kebutuhan kalibrasi di rencana kita |
| 4 | Validasi vs EyeLink 1000 Plus (500 Hz, IR): sistem mereka (25 fps, RGB webcam) **bisa distinguish saccades & fixations setara EyeLink** secara horizontal | MediaPipe + webcam cukup reliabel untuk fixation detection |
| 5 | **DTW (Dynamic Time Warping)** = metode validasi trajectory similarity yang digunakan | Referensi metode evaluasi accuracy |
| 6 | Inkonsistensi/confidence dihitung dari **Euclidean distance face orientation** saat calibration vs current frame | Metode proxy confidence signal yang bisa diadopsi |
| 7 | Limitasi: **akurasi horizontal lebih baik dari vertikal** — sistem masih kurang presisi di sumbu y | Limitation yang harus diakui di skripsi kita juga |
| 8 | 25 fps real-time dengan CPU Intel i7 + 4GB RAM (laptop biasa) | Konfirmasi bahwa hardware requirement reasonable |
| 9 | Eksperimen 60 partisipan — 65% kacamata/lensa, berbagai warna iris → sistem tetap robust | Validasi untuk kondisi natural (developer kebanyakan pakai kacamata) |
| 10 | **Adaptif threshold 30% darkest pixel** di eye region = cara segment iris robustly | Detail implementasi yang berguna |

---

### 📖 Paper 6 — Real-Time Gaze Estimation via Webcam CNN [P6]

**Fokus:** CNN-based gaze estimation dengan MediaPipe untuk landmark detection + advanced analytics (heatmap, fixation, blink rate).

| # | Poin Kunci | Relevansi ke Proyek |
|---|-----------|-------------------|
| 1 | MediaPipe 468 landmarks → crop eye region → CNN → gaze coordinates (x, y) | Pipeline webcam-based yang valid dengan akurasi **90.98%** |
| 2 | **Eye Aspect Ratio (EAR)** untuk blink detection: `EAR = (|P2-P6| + |P3-P5|) / (2 × |P1-P4|)` — threshold 0.2 | **Blink rate = metric tambahan yang bisa diimplementasi** |
| 3 | Grid 16 sel (4×4) untuk section mapping screen → predict gaze per grid region | Validasi grid-based section mapping yang sama seperti rencana kita |
| 4 | **Fixation tracking + heatmap + blink rate** = suite analytics yang dikembangkan | Feature set yang bisa diadopsi untuk research dashboard |
| 5 | **MSE 0.0112, R²=0.9953** — akurasi sangat tinggi untuk webcam-based system | Benchmark akurasi yang bisa direferensikan |
| 6 | CNN outperform existing webcam-based methods: **90.98% vs 80–88%** existing | Validasi CNN approach, tapi kita pakai rule-based (trade-off explained) |
| 7 | **MediaPipe + OpenCV** = kombinasi standar — tidak perlu train CNN sendiri untuk basic gaze | Rencana kita sudah menggunakan kombinasi yang tepat |
| 8 | **Grayscale 256×256 eye region** = representasi optimal untuk gaze estimation | Detail implementasi iris region processing |
| 9 | Ethical: consent form, data anonymized, local storage, GDPR-compliant design | Memperkuat ethical framework yang harus ditambahkan |
| 10 | Variability per-individu — anak (8th) vs dewasa berbeda signifikan | Justifikasi baseline per-subjek yang sudah ada di rencana |

---

## ✅ Yang Sudah Tepat dalam Rencana (Dikonfirmasi Paper)

| Aspek Rencana | Dikonfirmasi Oleh |
|--------------|-----------------|
| MediaPipe + OpenCV — webcam-based eye tracking | **Dipakai langsung di P5 & P6** — validasi paling kuat |
| Grid-based gaze → section mapping (bukan pixel) | AOI = pendekatan standar [P3][P4], grid 16 sel dipakai [P6] |
| WebSocket komunikasi Python ↔ Node.js | WebSocket untuk real-time IDE data [P3] |
| 5 state: Focused, Confused, Scanning, Overloaded, Vibecoding | Semua state ada basis teoritis dari paper [P1][P2][P4] |
| Baseline recording 2–3 menit sebelum sesi | Paper 4 melakukan ini secara eksplisit (2 menit relaksasi) [P4] |
| Kalman / EMA / Median filter untuk smoothing | Signal preprocessing = standar di semua paper [P1][P2][P3] |
| Dwell time threshold untuk intentional gaze | Fixation duration sebagai metric utama [P2][P3] |
| Heart rate + eye tracking = multimodal | Multimodal >> unimodal — dikonfirmasi kuat [P2][P3] |
| Threshold per-subjek (adaptive baseline) | Inter-subject variability = masalah kritis [P3][P6] |
| Vibecoding state detection | CUPS taxonomy empiris, AI-black-box pattern [P4][P1] |
| Logging data untuk penelitian | Semua paper menekankan data collection [P3][P4] |
| VS Code sebagai platform target | Dipakai sebagai standar dalam paper [P4] |
| 9-point kalibrasi | Dipakai langsung di P5 untuk mapping eye position → screen [P5] |

---

## ⚠️ Gap yang Perlu Diisi (Berdasarkan Paper)

### Gap 1: Tidak Ada Pupil Size Metric 🟡 PENTING

**Sumber Paper:**
> **[P1]** — *"Task-Evoked Pupillary Response (TEPR) is a critical biometric phenomenon where the diameter of the pupil involuntarily dilates in direct, measurable proportion to the processing demands placed upon working memory."*
> **[P3]** — Pupil size + fixation dispersion = **top 2 predictor** menurut XGBoost feature importance.

**Masalah:** Rencana hanya ambil gaze position (x, y) → section mapping.

#### 🔬 Analisis: Apakah MediaPipe Bisa & Bagus untuk Pupil Size?

**Berdasarkan P5 (Iacobelli et al., 2023):**
- MediaPipe Face Mesh menghasilkan **468 3D facial landmarks** termasuk kontur iris kiri (469–472) dan kanan (474–477)
- Paper ini mengembangkan **Iris Detection Algorithm (IDA)** di atas MediaPipe yang mampu mendeteksi **iris center dengan presisi** via morphological transformations
- Mereka **tidak secara eksplisit mengukur pupil diameter** — fokusnya adalah iris center untuk gaze direction
- Namun karena iris landmarks tersedia, **delta iris diameter bisa dihitung** sebagai proxy pupil size

**Berdasarkan P6 (Vidhya & Resende Faria, 2025):**
- Paper ini menggunakan MediaPipe hanya untuk **landmark detection → eye region crop** lalu CNN untuk gaze
- Tidak mengukur pupil size — fokus gaze coordinate prediction
- Mengimplementasikan **blink rate via EAR** sebagai cognitive metric tambahan

**Verdict MediaPipe untuk Pupil Size:**

| Aspek | Penilaian | Detail |
|-------|-----------|--------|
| **Feasibility** | ✅ Bisa | Iris landmarks 469–477 tersedia di Face Mesh |
| **Kualitas sebagai pupil size** | ⚠️ Terbatas | Ini adalah *iris diameter* bukan *pupil diameter* sejati |
| **Akurasi** | ⚠️ Moderate | Variasi cahaya mempengaruhi deteksi iris boundary |
| **Nilai sebagai *relative* metric** | ✅ Cukup bagus | Delta/perubahan relatif lebih reliable dari nilai absolut |
| **Vs Tobii hardware** | ❌ Jauh berbeda | Tobii pakai infrared + dedicated pupil tracking hardware |
| **Justifikasi untuk skripsi** | ✅ Valid | Sebagai **proxy** dan **pendekatan low-cost** yang diakui limitation-nya |

**Kesimpulan:** MediaPipe **bisa** digunakan untuk estimasi iris size sebagai *proxy* pupil size, tapi harus diframing dengan benar: ini adalah **iris diameter delta** (perubahan relatif dari baseline), bukan true pupil dilation measurement. Cukup bagus untuk trend detection (membesar/mengecil), tidak untuk nilai absolut.

**Implementasi yang Direkomendasikan:**
```python
# iris_right: landmarks 474-477 (titik kiri, atas, kanan, bawah iris kanan)
# iris_left: landmarks 469-472

def calculate_iris_size_delta(landmarks, baseline_iris_size):
    # Iris diameter = jarak antara landmark kiri & kanan iris
    iris_right_left = landmarks[474]   # leftmost point iris kanan
    iris_right_right = landmarks[476]  # rightmost point iris kanan
    current_size = distance(iris_right_left, iris_right_right)
    
    # Normalize relative to face width (untuk kompensasi jarak ke kamera)
    face_width = distance(landmarks[234], landmarks[454])  # cheekbone to cheekbone
    normalized_size = current_size / face_width
    
    # Delta dari baseline (bukan nilai absolut)
    return normalized_size - baseline_iris_size

# Output: iris_size_delta > 0 = possible cognitive load increase
# (TEPR effect — pupil membesar saat cognitive load tinggi)
```

**Catatan Penting:**
- Harus ada **baseline capture** di awal sesi (sama seperti baseline HR)
- Delta bisa dipengaruhi perubahan cahaya lingkungan → tambahkan light normalization atau acknowledge sebagai limitation
- Frame sebagai *"low-cost proxy for TEPR"* di skripsi

---

### Gap 2: Tidak Ada NRevisit Counter 🔴 SANGAT PENTING

**Sumber Paper:**
> **[P3]** — *"NRevisit showed a near-perfect correlation ranging from **0.78 to 0.91** with EEG measurements of cognitive load. This demonstrates that integrating dynamic behavioral metrics with neural baselines provides a vastly superior ground truth."*

**Update Rekomendasi:** Berdasarkan pertimbangan akhir, **NRevisit Counter direkomendasikan untuk diimplementasikan** — ini bukan sekedar opsional. Korelasinya dengan EEG-measured cognitive load (0.78–0.91) sangat kuat dan implementasinya relatif sederhana (pure Python, tidak perlu hardware tambahan).

**Solusi:**
```python
# Di Python service — track section visit history
section_visits = []  # list section yang dikunjungi berurutan
nrevisit_count = {}  # {section_name: berapa kali re-visit}

def update_nrevisit(current_section):
    if len(section_visits) > 0 and current_section in section_visits[:-1]:
        nrevisit_count[current_section] = nrevisit_count.get(current_section, 0) + 1
    section_visits.append(current_section)
```

---

### Gap 3: Gaze Transition Rate Belum Ditracking 🟡

**Sumber Paper:**
> **[P2]** — *"Saccade velocity increases as developer intensely scans related code segments."*
> **[P1]** — *"Irregular, erratic saccades... strongly indicate high intrinsic load and confusion."*

**Solusi:** Hitung berapa kali gaze berpindah section per detik (analog saccade rate):
```python
transition_rate = section_changes_in_window / window_duration_seconds
# Tinggi = scanning atau overloaded, rendah = focused
```

---

### Gap 4: State Detection Terlalu Rule-Based 🟡

**Sumber Paper:**
> **[P3]** — Random Forest + XGBoost jauh outperform simple thresholds. Cross-subject akurasi bisa drop 95% → 70% dengan rule-based biasa.

**Catatan untuk Skripsi:** Rule-based boleh dipakai untuk MVP, tapi harus di-frame dengan tepat:
- Jelaskan bahwa ini simplifikasi yang bisa di-upgrade ke ML
- Bandingkan output rule-based vs NASA-TLX di akhir sesi sebagai validasi
- Citing [P3] bahwa ML adalah arah yang lebih kuat di masa depan

---

### Gap 5: Tidak Ada Ethical Framework 🔴 WAJIB DITAMBAHKAN

**Sumber Paper:**
> **[P1]** — *"Brain data cannot be treated as a standard corporate analytics asset; unlike a compromised password, exposed neural data cannot be rotated, reset, or changed."*
> **[P3]** — *"The scientific literature focuses overwhelmingly on algorithmic optimization while severely neglecting the critical design of robust governance frameworks."*

**Yang Harus Ditambahkan ke Skripsi:**
| Prinsip Etika | Implementasi |
|--------------|-------------|
| Informed consent | Consent form sebelum mulai — jelaskan data apa yang dikumpulkan |
| Data local-only | Semua log tersimpan di mesin lokal subjek, tidak ke cloud |
| User control | Subjek bisa pause/stop kapanpun dari status bar |
| Data anonymization | Session ID, bukan nama subjek dalam log |
| Data deletion right | Subjek bisa hapus data sesi mereka |

---

### Gap 6: Kalibrasi Belum Ada Auto-Quality Monitoring 🟢 Minor

**Sumber Paper:**
> **[P3]** — *"Eye-trackers require developers to remain rigidly within a specific spatial bounding box. Calibration accuracy degrades over time."*

**Solusi:** Tambahkan confidence score dari MediaPipe sebagai proxy kalibrasi:
```python
if face_landmarks.pose_landmarks:
    confidence = face_result.pose_landmarks.confidence  # atau hitung dari landmark spread
    if confidence < THRESHOLD:
        send_recalibration_reminder()
```

---

## 🔧 Update Rencana yang Direkomendasikan (v2 → v3)

> Berdasarkan 4 paper sebelumnya + **2 paper eye tracking baru [P5][P6]**, rencana diupdate dengan justifikasi yang lebih kuat.

### ✅ Status Keputusan Per Gap

| Gap | Keputusan | Alasan |
|-----|-----------|--------|
| Gap 1: Pupil Size via MediaPipe | ✅ **IMPLEMENTASI** sebagai iris size delta | P5 konfirmasi feasibility, framing sebagai TEPR proxy |
| Gap 2: NRevisit Counter | ✅ **IMPLEMENTASI** — bukan opsional | Korelasi 0.78–0.91 dengan EEG terlalu kuat untuk dilewatkan |
| Gap 3: Gaze Transition Rate | ✅ **IMPLEMENTASI** | Simple computation, basis scanning state yang kuat |
| Gap 4: Rule-based vs ML | 🟡 **Rule-based DULU** | OK untuk skripsi, frame sebagai future work |
| Gap 5: Ethical Framework | ✅ **WAJIB DITAMBAHKAN** | P6 pun implement consent form & anonymization |
| Gap 6: Kalibrasi confidence | 🟡 **Medium priority** | Tambah inconsistency score ala P5 |
| **BONUS dari P6: Blink Rate** | ✅ **TAMBAHKAN** | EAR sudah tersedia dari MediaPipe, metric cognitive load tambahan |

---

### 🔄 Pipeline Eye Tracking yang Diperbarui (Planning v3)

```
Webcam Frame (25–30 fps)
    ↓
MediaPipe Face Mesh → 468 3D landmarks
    ↓
┌─────────────────────────────────────────┐
│ Feature Extraction (parallel)            │
│  ├── Iris center (474,476) → gaze (x,y) │
│  ├── Iris diameter → iris_size_delta     │  ← [P5][P1]
│  ├── Eye landmarks (P1-P6) → EAR        │  ← [P6]
│  └── Face orientation → confidence       │  ← [P5]
└─────────────────────────────────────────┘
    ↓
Smoothing Filter (Kalman/EMA/Median)        ← [P1][P2][P3]
    ↓
Grid Classifier → section name (4×4 grid)   ← [P6]
    ↓
┌─────────────────────────────────────────┐
│ Behavioral Metrics (per window 5 detik)  │
│  ├── Dwell time per section             │
│  ├── NRevisit count per section         │  ← [P3] ★ KRITIS
│  ├── Gaze transition rate               │  ← [P2]
│  └── Blink rate (EAR)                  │  ← [P6]
└─────────────────────────────────────────┘
    ↓
Output JSON ke WebSocket → VS Code Extension:
{
  "section": "main_file",
  "confidence": 0.91,
  "dwell_time_ms": 1500,
  "nrevisit_count": 2,
  "transition_rate": 0.3,
  "iris_size_delta": 0.018,
  "blink_rate_per_min": 14,
  "fps": 28
}
```

### 🧠 State Logic yang Diperkuat (Planning v3)

```python
# FOCUSED — [P1][P2][P5][P6]
focused = (
    dwell_main_file > 2000 AND
    nrevisit_main_file < 2 AND      # NRevisit rendah = tidak bingung
    transition_rate < 0.5 AND       # Tidak sering pindah section
    iris_size_delta < 0.015 AND     # Pupil stabil (tidak overloaded)
    blink_rate NEAR baseline AND    # Blink normal
    hr_normal
)

# CONFUSED — diperkuat [P3]
confused = (
    nrevisit_same_section > 3 AND   # ★ SIGNAL TERKUAT (0.78-0.91 corr EEG)
    iris_size_delta > 0.03 AND      # Pupil membesar (TEPR)
    blink_rate < baseline * 0.7 AND # Blink turun = fokus intensif
    hr_elevated
)

# SCANNING — [P2] Uwano study
scanning = (
    transition_rate > 1.5 AND       # Banyak perpindahan section
    avg_dwell < 800 AND
    nrevisit_count < 2 AND          # Belum bolak-balik ke section sama
    hr_normal
)

# OVERLOADED — [P1][P3]
overloaded = (
    iris_size_delta > 0.05 AND      # Pupil sangat membesar
    nrevisit_count > 5 AND          # Bolak-balik banyak
    blink_rate > baseline * 1.5 AND # Blink meningkat = fatigue
    hr_elevated
)
```

---

## 📊 Tabel Perbandingan Final: Rencana vs Best Practice Paper

| Aspek | Planning v2 | Planning v3 (Updated) | Best Practice | Status |
|-------|------------|----------------------|---------------|--------|
| Eye tracking engine | MediaPipe webcam | MediaPipe webcam | Tobii Pro / IR | ⚠️ Limitation diakui |
| Sinyal gaze | Gaze (x, y) + section | Gaze + iris delta + EAR | Gaze + true pupil + blink | ✅ Improved |
| Behavioral metrics | Dwell time | Dwell + NRevisit + transition + blink | Semua di atas + saccade velocity | ✅ Improved |
| NRevisit counter | ❌ Tidak ada | ✅ Ada | Ada (0.78–0.91 corr EEG) | ✅ **Fixed** |
| Blink rate | ❌ Tidak ada | ✅ EAR dari MediaPipe | Ada | ✅ **Fixed** |
| State detection | Rule-based threshold | Rule-based (diperkuat metrik baru) | ML classifier (RF/XGBoost) | 🟡 OK for skripsi |
| Multi-sensor | Eye + HR | Eye + HR (diperkuat) | Eye + HR + EDA + EEG | ✅ Reasonable |
| Kalibrasi | Manual | Manual + inconsistency score | Auto confidence monitoring | 🟡 Improved |
| Ethical framework | ❌ Tidak ada | ✅ Consent + local + anonymization | Full GDPR-compliant | ✅ **Fixed** |
| Baseline per subjek | ✅ Ada | ✅ Ada (+ iris size baseline) | Ada | ✅ Done |
| Vibecoding state | ✅ Ada | ✅ Ada + NRevisit ke AI panel | CUPS taxonomy | ✅ Strengthened |
| MediaPipe validation | ❌ Hanya assumption | ✅ Dikonfirmasi P5 & P6 | - | ✅ **Validated** |

---

## 🏆 Kontribusi Orisinal yang Bisa Diklaim

Berdasarkan gap di paper, ini adalah hal yang belum pernah diimplementasikan dan bisa menjadi **original contribution** skripsimu:

| Kontribusi | Deskripsi | Basis Paper |
|-----------|-----------|------------|
| **Vibecoding State** | Deteksi otomatis pola developer yang bergantung berlebihan pada AI — real-time di IDE | [P4][P1] |
| **Section-based gaze analytics** | Analisis gaze per zona IDE (bukan pixel/line) dengan NRevisit tracking | [P3][P4] |
| **NRevisit-triggered intervention** | Intervensi (hover doc, problem highlight) dipicu oleh NRevisit count, bukan hanya dwell | [P3] |
| **AI usage ratio tracking** | Mengukur rasio waktu developer di AI panel vs main file dalam satu sesi | [P1][P4] |
| **Low-cost TEPR proxy** | Estimasi iris size delta via MediaPipe sebagai pengganti dedicated pupil tracker | [P1][P3][P5] |
| **Multimodal blink+gaze+HR** | Kombinasi blink rate (EAR) + gaze metrics + HR untuk state classification di IDE | [P2][P6] |

---

*Dokumen ini dibuat sebagai analisis komparatif antara rencana Planning v2 dengan temuan dari 6 paper akademik (P1–P4: cognitive load, P5–P6: eye tracking MediaPipe).*
*Referensi paper tersedia di folder `/papers` dan `/papers/eye_tracking`.*
