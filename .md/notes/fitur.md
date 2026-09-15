# 🎛️ Adaptive IDE — Katalog Fitur Adaptif Berbasis Paper

> Setiap fitur di dokumen ini **bersumber langsung dari paper akademik**.
> Sitasi format: **[P1]**, **[P2]**, **[P3]**, **[P4]** sesuai daftar referensi di bawah.

---

## 📚 Referensi Paper

| Kode | Judul | Fokus |
|------|-------|-------|
| **[P1]** | *Adaptive Integrated Development Environments: Mitigating Software Developer Cognitive Load Through Neuroergonomics and Context-Aware AI* | Fondasi teori IDE adaptif, neuroergonomics, intervensi konkret |
| **[P2]** | *The Biometric Assessment of Cognitive Load in Software Engineering: A Multi-Dimensional Analysis of Developer Experience and Predictive Modeling* | DX, biometrik, eye tracking, expertise reversal effect |
| **[P3]** | *Predicting Cognitive Load in Software Engineering via Machine Learning and Biometrics: State of the Art and Research Gaps* | Survey ML + biometrik, gap penelitian, dataset benchmark |
| **[P4]** | *Towards Decoding Developer Cognition in the Age of AI Assistants* — AlHaque et al., George Mason University & Virginia Tech, arXiv Jan 2025 | Studi empiris AI assistant + cognitive load, CUPS taxonomy |

---

## 🧠 Dasar Teori: Mengapa IDE Perlu Beradaptasi?

> **[P1]** — *"The transition from static text editors to context-aware, neuro-adaptive digital partners represents a paradigm shift aimed at optimizing engineering productivity while fiercely safeguarding the mental well-being and cognitive sustainability of the workforce."*

**Cognitive Load Theory (CLT)** — fondasi dari semua fitur adaptif ini:

- **Intrinsic load** = kesulitan inheren task coding — tidak bisa dihilangkan
- **Extraneous load** = friction dari tools/UI — **harus dikurangi oleh IDE adaptif**
- **Germane load** = proses belajar/schema building — **harus dimaksimalkan**

> **[P1]** — *"In controlled studies, dynamic cognitive load evaluation coupled with interface adaptation reduced developer task completion time by **22.7%**, decreased cognitive load by **25.0%**, and increased satisfaction by **41.9%**."*

---

## 🔬 Sinyal Biologis yang Digunakan

### 👁️ Eye Tracking

| Metrik | Makna Kognitif | Paper |
|--------|---------------|-------|
| **Fixation Duration** (naik) | Butuh waktu lebih lama memproses — load tinggi | [P2][P3] |
| **Fixation Rate** (turun) | Fokus dalam, sedikit titik tapi dalam | [P2] |
| **Pupil Diameter** (naik) | Sympathetic arousal — working memory terisi | [P1][P2][P3] |
| **Blink Rate** (turun) | High attention, otak suppress refleks berkedip | [P2] |
| **Saccade Velocity** (naik) | Pencarian intensif, scanning cepat | [P2][P3] |
| **NRevisit Count** (naik) | Developer kembali ke area sama — tanda confusion | [P3] |
| **Gaze Transition Rate** (naik) | Loncat antar section — scanning behavior | [P1][P2] |

> **[P3]** — *"NRevisit showed a near-perfect correlation ranging from 0.78 to 0.91 with EEG measurements of cognitive load."*

> **[P1]** — *"Task-Evoked Pupillary Response (TEPR): pupil involuntarily dilates in direct, measurable proportion to the processing demands placed upon working memory."*

> **[P2]** — *"The 'eye-mind hypothesis': no significant latency between what the eyes fixate on and what the mind processes."*

### ❤️ Heart Rate

ini yang alep sama mumei ntaran aja

| Metrik | Makna | Paper |
|--------|-------|-------|
| **HRV rendah** | High stress + mental exhaustion | [P1][P2] |
| **LF/HF Ratio tinggi** | Sympathetic dominance = cognitive overload | [P1] |
| **HR naik** | Arousal — stress atau excitement | [P2] |

---

## 🔴 5 State Developer — Basis Paper

### 🟢 Focused
**Sinyal:** Dwell panjang di `main_file`, NRevisit rendah, pupil stabil, HR normal

> **[P1]** — *"Smooth, linear saccades indicate low cognitive load and high comprehension."*
> **[P2]** — *"Flow State: deep immersion → high innovation and employee satisfaction."*

### 🟡 Confused
**Sinyal:** NRevisit tinggi, pupil membesar, HR naik, gaze sering kembali ke area error

> **[P3]** — NRevisit berkorelasi 0.78–0.91 dengan EEG cognitive load.
> **[P2]** — Pola novice muncul di expert = sinyal confusion.

### 🔵 Scanning
**Sinyal:** Gaze loncat cepat antar section, saccade velocity tinggi, dwell pendek, HR normal

> **[P2]** — *"Uwano study (2006): High-performing developers spend more time initially scanning the entire source code to build a mental map before investigating specific details."*

### 🔴 Overloaded
**Sinyal:** Gaze acak, pupil sangat besar, HR sangat tinggi, transisi tidak berpola

> **[P1]** — *"Irregular, erratic saccades combined with prolonged fixations strongly indicate high intrinsic load and confusion."*
> **[P1]** — *"Developers typically require nearly 15 minutes of recovery time after a single interruption at peak load."*

### 🟣 Vibecoding
**Sinyal:** Gaze bolak-balik `ai_agent` ↔ `main_file` ≥ 3 siklus berulang

> **[P4]** — CUPS taxonomy: *"Deferring Thought for Later: accepting AI suggestions with the intent to verify them at a later stage"* — form vibecoding empiris.
> **[P1]** — *"AI-as-Black-Box Code Acceptance: complete failure of schema acquisition; developer cannot debug the code when edge cases inevitably fail."*

---

## 🎛️ FITUR ADAPTIF — DENGAN SITASI PAPER

---

### KATEGORI 1: VISUAL / TYPOGRAPHY

#### ✅ 1.1 Font Size Adaptation
**Basis:** [P1] load shedding — *"reduce low-priority visual information when cognitive resources depleted"*; [P2] extraneous load dari cara info disajikan

| State | Aksi | Setting |
|-------|------|---------|
| Focused 🟢 | Normal | `fontSize: 14` |
| Confused 🟡 | **+2–3pt** | `fontSize: 16` |
| Scanning 🔵 | **−1pt** | `fontSize: 13` |
| Overloaded 🔴 | **+3–4pt** | `fontSize: 17` |

```typescript
vscode.workspace.getConfiguration().update('editor.fontSize', newSize);
```

---

#### ✅ 1.2 Line Height / Spacing
**Basis:** [P1] progressive disclosure; [P2] extraneous load dari presentasi informasi

| State | Line Height |
|-------|------------|
| Focused 🟢 | 1.3 (compact) |
| Confused 🟡 | 1.6 (breathable) |
| Scanning 🔵 | 1.2 (ultra-compact) |
| Overloaded 🔴 | 1.8 (very expanded) |

---

#### ✅ 1.3 Color Theme / Visual Stimulation
**Basis:** [P1] — *"Load shedding: systematically degrades or removes low-priority visual information"*

| State | Aksi |
|-------|------|
| Focused 🟢 | Normal theme |
| Overloaded 🔴 | **Low Stimulation Theme** (warna redup) |
| Confused 🟡 | Highlight area NRevisit tinggi |
| Vibecoding 🟣 | Diff highlight AI vs kode asli |

---

#### ✅ 1.4 Syntax Highlight Intensity
**Basis:** [P1] — *"Suppress non-critical static analysis warnings regarding minor style deviations"* saat overloaded

| State | Aksi |
|-------|------|
| Overloaded 🔴 | Semantic dimming — hanya keyword utama |
| Confused 🟡 | Boost highlight pada token di area error |

---

### KATEGORI 2: LAYOUT / PANEL VISIBILITY

#### ✅ 2.1 Sidebar Auto-Collapse / Expand
**Basis:** [P1] — *"Focused: Zen mode, sembunyikan sidebar"*; [P1] Progressive disclosure

| State | Aksi | VS Code Command |
|-------|------|----------------|
| Focused 🟢 | **Auto-collapse** | `workbench.action.closeSidebar` |
| Scanning 🔵 | **Show Explorer** | `workbench.action.showExplorer` |
| Overloaded 🔴 | **Collapse all** | Close all panels |

---

#### ✅ 2.2 Minimap Show/Hide
**Basis:** [P2] — *"Scanning: Minimap lebih besar, breadcrumb prominent"*; [P1] load shedding saat focused/overloaded

| State | Aksi | Setting |
|-------|------|---------|
| Focused 🟢 | **Hide** | `editor.minimap.enabled: false` |
| Scanning 🔵 | **Show + scale 3** | `editor.minimap.enabled: true` |
| Overloaded 🔴 | **Hide** | `editor.minimap.enabled: false` |
| Confused 🟡 | Show + highlight error | Custom decoration |

---

#### ✅ 2.3 Zen Mode / Centered Layout
**Basis:** [P1] — *"FlowLight paradigm: when high-cognitive flow state detected"*; [P2] — *"Actively block notifications and simplify UI when 'flow state' biometric signature is detected"*

| Trigger | Aksi |
|---------|------|
| Focused 🟢 sustained > 3 menit | Auto-enter **Zen Mode** |
| Focused 🟢 baru mulai | Enable **Centered Layout** |

```typescript
vscode.commands.executeCommand('workbench.action.toggleZenMode');
```

---

#### ✅ 2.4 Problems Panel Auto-Show
**Basis:** [P1] — *"Confused: Auto-popup error explanation"*; [P1] Contextual Task Automation — *"IDE initiates task automation for routine monitoring"*

| Trigger | Aksi |
|---------|------|
| Confused 🟡 + gaze di area error | **Auto-show Problems panel** |
| NRevisit tinggi di section bermasalah | Scroll Problems ke error terkait |

```typescript
vscode.commands.executeCommand('workbench.actions.view.problems');
```

---

#### ✅ 2.5 File Outline Panel
**Basis:** [P2] — *"Scanning: tampilkan file outline"*; [P2] Uwano study — developer butuh mental map keseluruhan kode

| Trigger | Aksi |
|---------|------|
| Scanning 🔵 | **Auto-show Outline** |
| Confused 🟡 | Show + highlight current symbol |

---

#### ✅ 2.6 Breadcrumbs Navigation
**Basis:** [P2] — *"Scanning: breadcrumb prominent"*; [P1] mengurangi Tool cognitive load driver

| State | Aksi | Setting |
|-------|------|---------|
| Scanning 🔵 | **Always visible** | `breadcrumbs.enabled: true` |
| Focused 🟢 | **Hidden** | `breadcrumbs.enabled: false` |

---

### KATEGORI 3: EDITOR BEHAVIOR

#### ✅ 3.1 Adaptive Code Folding
**Basis:** [P1] — *"Load shedding: collapse complex architectural dependency maps"*; [P1] — *"Gaze-based IDEs: preemptively expand relevant code blocks, collapse irrelevant surrounding functions"*

| State | Aksi | Command |
|-------|------|---------|
| Focused 🟢 | Smart fold kode di luar area gaze | `editor.foldRecursively` |
| Overloaded 🔴 | **Fold all** | `editor.foldAll` |
| Scanning 🔵 | **Unfold all** | `editor.unfoldAll` |
| Confused 🟡 | Buka fold di area error | `editor.unfold` |

---

#### ✅ 3.2 Gaze-Triggered Hover Documentation (NRevisit-based)
**Basis:** [P1] — *"Comprehension Scaffold: using GenAI to explain undocumented legacy systems"*; [P3] — NRevisit berkorelasi 0.78–0.91 dengan cognitive load; [P3] — *"Eye-tracking mapped to Areas of Interest (AOIs)"*

> Ini adalah **fitur novel** — menggunakan NRevisit count (bukan hanya dwell time) sebagai trigger.

| Trigger | Aksi |
|---------|------|
| Confused 🟡 + NRevisit > 3x pada token tertentu | **Auto-trigger hover doc** |
| Confused 🟡 + NRevisit > 3x pada section | Suggest AI explanation untuk area tersebut |

```typescript
vscode.commands.executeCommand('editor.action.showHover');
```

---

#### ✅ 3.3 Inline Hints & Parameter Hints (Expertise-Aware)
**Basis:** [P1] — *"Adaptive Scaffolding: maximum scaffolding for novice"*; [P2] — *"Expertise Reversal Effect: hints that help novices may impair experts"*

| State | Aksi | Setting |
|-------|------|---------|
| Confused 🟡 | **ON verbose** | `editor.inlayHints.enabled: 'on'` |
| Overloaded 🔴 | **OFF** | `editor.inlayHints.enabled: 'off'` |
| Focused 🟢 | **Minimal** | `editor.inlayHints.enabled: 'onUnlessPressed'` |

---

#### ✅ 3.4 Sticky Scroll
**Basis:** [P2] — Naming & semantic clarity membantu developer menjaga context; [P1] mengurangi Structure cognitive load (navigating hierarchical topologies)

| State | Aksi |
|-------|------|
| Focused 🟢 | **ON** — function header selalu terlihat |
| Overloaded 🔴 | OFF — simplify view |

---

#### ✅ 3.5 Word Wrap
**Basis:** [P1] — *"Load shedding: reduces need for horizontal navigation"*; [P2] extraneous load dari environmental friction

| State | Aksi |
|-------|------|
| Overloaded 🔴 | **ON** |
| Confused 🟡 | **ON** — lebih mudah membaca |
| Focused 🟢 | **OFF** — expert prefer compact |

---

### KATEGORI 4: NOTIFIKASI & INTERUPSI

#### ✅ 4.1 Notification Suppression — FlowLight Pattern
**Basis:** [P1] — *"FlowLight prototype: device illuminates red when high-cognitive flow state detected, signaling colleagues not to interrupt"*; [P1] — *"'Defer-to-breakpoint' policies: automatically intercept and hold non-urgent notifications, releasing them only when cognitive demand drops"*; [P1] — *"Developers require nearly 15 minutes of recovery time after a single interruption at peak load"*

| State | Aksi |
|-------|------|
| Focused 🟢 | **Tahan** semua notifikasi non-critical |
| Focused 🟢 sustained > 3 menit | Aktifkan **Do Not Disturb** |
| Overloaded / Scanning | **Release** notifikasi yang ditahan |

---

#### ✅ 4.2 Break Reminder
**Basis:** [P2] — CognitIDE: *"Wellness Feedback: suggesting breaks when early signs of fatigue or burnout are detected"*; [P2] — *"Lower HRV predicts short-term accumulation of stress, creating a negative feedback loop leading to burnout"*

| Trigger | Aksi |
|---------|------|
| Overloaded 🔴 sustained > 2 menit | **Gentle break reminder** |
| Overloaded 🔴 sustained > 5 menit | Stronger reminder + suggest save |

```typescript
vscode.window.showInformationMessage(
  '💆 You seem cognitively overloaded. A short break would help.',
  'Dismiss', 'Snooze 10min'
);
```

---

### KATEGORI 5: AI ASSISTANT INTEGRATION

#### ✅ 5.1 AI Suggestion Rate Adaptation
**Basis:** [P1] — *"GenAI is a 'difficulty restructurer' not 'difficulty eliminator'. Blindly injecting vast AI-generated logic triggers cognitive overload due to metacognitive demand required to verify foreign code"*; [P2] — Expertise Reversal Effect pada AI assistance

| State | Aksi |
|-------|------|
| Confused 🟡 | Suggestions **lebih frequent + more context** |
| Overloaded 🔴 | Suggestions **dikurangi** — hindari beban validasi tambahan |
| Focused 🟢 | Suggestions **minimal** |

---

#### ✅ 5.2 Vibecoding Mode — Fitur Khusus
**Basis:** [P4] — CUPS taxonomy: *"Deferring Thought for Later: accepting suggestions with the intent to verify them at a later stage"*; [P1] — *"Verification-Bypass Debt: utilizing AI to generate unit tests without human oversight"*; [P1] — *"AI-as-Black-Box Code Acceptance"*

| Fitur Vibecoding | Implementasi | Basis |
|----------------|-------------|-------|
| **Diff highlight** kode asli vs AI | Decoration pada baris yang baru di-accept | [P1] |
| **Slow-reveal animation** | Visual delay sebelum merge | [P1] |
| **AI Usage Timer** di status bar | "AI Panel: 4m 32s / Total: 12m" | [P4] |
| **Verification Prompt** sebelum save | "Have you reviewed this AI suggestion?" | [P1][P4] |
| **NRevisit to AI tracking** | Log berapa kali kembali ke AI setelah accept | [P3][P4] |

```typescript
// Diff decoration untuk AI-accepted code
const aiAcceptedDeco = vscode.window.createTextEditorDecorationType({
  backgroundColor: 'rgba(147, 112, 219, 0.1)',
  borderLeft: '3px solid rgba(147, 112, 219, 0.6)',
  after: { contentText: ' ← AI', color: 'rgba(147,112,219,0.6)' }
});
```

---

#### ✅ 5.3 Context-Aware Documentation Auto-Popup
**Basis:** [P1] — *"Comprehension Scaffold: using GenAI to explain undocumented systems — reduces extraneous load while promoting germane load"*; [P3] — eye tracking mapped to AOIs dalam IDE

| Trigger | Aksi |
|---------|------|
| Confused 🟡 + gaze pada token unfamiliar | Auto-show hover doc |
| Confused 🟡 + NRevisit > 3x area sama | Suggest AI explanation untuk area |

---

### KATEGORI 6: GAZE-DRIVEN PASSIVE FEATURES

#### ✅ 6.1 Section Visual Feedback
**Basis:** [P1] — *"By calculating intersection of developer's gaze with screen, IDE can anticipate which code module developer is visually searching for"*; [P3] — *"Eye-tracking spatially mapped to Areas of Interest (AOIs)"*

| Perilaku | Aksi |
|----------|------|
| Gaze di `main_file` | Subtle highlight border aktif |
| Gaze di `ai_agent` | AI panel focus indicator |

---

#### ✅ 6.2 Research Dashboard (Gaze Heatmap per Section)
**Basis:** [P3] — EMIP dataset: eye-tracking per algorithmic complexity; [P4] — screen recordings + interaction data untuk post-hoc analysis; [P4] — CUPS taxonomy untuk annotasi aktivitas coding

| Data | Format |
|------|--------|
| % waktu per section | Pie / bar chart |
| State timeline | Timeline bar |
| NRevisit per section | Bar chart |
| Transition matrix (A→B) | Heatmap grid |
| AI vs non-AI time ratio | Gauge |

---

### KATEGORI 7: ADAPTIVE SCAFFOLDING — EXPERTISE-AWARE

#### ✅ 7.1 Developer Profiling (Novice vs Expert)
**Basis:** [P2] — *"Expertise Reversal Effect: instructional aids that benefit novices may paradoxically impair experts"*; [P3] — *"A generalized model that fails to factor in developer experience will inherently miscalibrate cognitive load thresholds"*; [P4] — *"Less experienced developers use AI coding assistants more frequently"*

| Level | Sinyal dari Log | Response IDE |
|-------|---------------|-------------|
| **Novice** | NRevisit konsisten tinggi, sering Confused, banyak waktu di AI panel | Inline hints ON, doc agresif, scaffolding penuh |
| **Expert** | Jarang Confused, scan cepat, sedikit pakai AI | Semua scaffolding OFF, advanced tools, minimal intrusion |

#### Scaffolding Matrix per Level

| Action | Novice | Expert | Basis |
|--------|--------|--------|-------|
| Inline type hints | **ON** | OFF | [P2] ERE |
| AI suggestion freq | **High** | Low | [P2][P4] |
| Doc auto-popup | **Agresif** | Minimal | [P1] Scaffolding |
| Error explanation | **Verbose** | Minimal | [P1] Fading |
| Breadcrumbs | **Always** | On demand | [P1] Load shedding |

> **[P1]** — *"Instructional Fading: as competence grows, the IDE initiates fading — transitioning from complete code blocks to Socratic hints, pseudo-code structures, or pointing to documentation."*

---

## 📊 Summary Matrix: State × Adaptive Action

| Feature | 🟢 Focused | 🟡 Confused | 🔵 Scanning | 🔴 Overloaded | 🟣 Vibecoding | Paper |
|---------|:----------:|:-----------:|:-----------:|:-------------:|:-------------:|-------|
| Font size | Normal | **+2pt** | **−1pt** | **+4pt** | Normal | [P1][P2] |
| Line height | 1.3 | 1.6 | 1.2 | **1.8** | Normal | [P1][P2] |
| Sidebar | **Hide** | Show | **Explorer** | **Hide** | Normal | [P1] |
| Minimap | **Hide** | Show+err | **Show big** | **Hide** | Normal | [P1][P2] |
| Problems panel | Hide | **Auto-show** | Hide | Hide | Normal | [P1] |
| Outline panel | Hide | **Show** | **Show** | Hide | Normal | [P2] |
| Zen mode | **On (3m)** | Off | Off | Off | Off | [P1][P2] |
| Notification | **Suppress** | Normal | Normal | **Suppress** | Normal | [P1] |
| Break reminder | Off | Off | Off | **Show** | Off | [P1][P2] |
| Code folding | Smart | Unfold err | **Unfold all** | **Fold all** | Normal | [P1] |
| Inline hints | Minimal | **Verbose** | Normal | **Off** | Normal | [P1][P2] |
| Hover doc | Manual | **Auto (NR>3)** | Manual | Off | Normal | [P1][P3] |
| Sticky scroll | **On** | On | On | Off | Normal | [P2] |
| Word wrap | Off | **On** | Off | **On** | Normal | [P1][P2] |
| AI suggestions | Normal | **Aggressive** | Normal | **Reduced** | Slow reveal | [P1][P4] |
| Diff highlight | Off | Off | Off | Off | **On** | [P1][P4] |
| AI timer | Off | Off | Off | Off | **On** | [P4] |
| Verification prompt | Off | Off | Off | Off | **On** | [P1][P4] |
| Color theme | Normal | Normal | Normal | **Low stim** | Normal | [P1] |
| Breadcrumbs | **Off** | Show | **On** | Off | Normal | [P1][P2] |

---

## ⭐ Fitur Paling Novel — Original Contribution

| # | Fitur | Mengapa Novel | Paper |
|---|-------|--------------|-------|
| 1 | **Vibecoding State Detection** | Belum ada implementasi di IDE nyata — hanya P4 yang meneliti pola ini | [P4][P1] |
| 2 | **NRevisit-triggered hover doc** | Menggunakan pola *kembali ke area yang sama* (bukan hanya dwell) sebagai trigger | [P3] |
| 3 | **Section-based research dashboard** | Per zone IDE, bukan pixel/baris — sesuai arahan dospem | [P3][P4] |
| 4 | **AI Usage Time Ratio tracking** | Ukur waktu di AI panel vs main file sebagai proxy comprehension debt | [P1][P4] |

---

*Dokumen ini dibuat berdasarkan analisis 4 paper akademik.*
*Setiap fitur memiliki justifikasi langsung dari literatur ilmiah dengan kutipan verbatim.*
