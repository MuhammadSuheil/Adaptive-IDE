# Laporan Pembenahan dan Optimasi Prototype Eye Tracking

Dokumen ini mencatat seluruh analisis, temuan masalah, serta langkah-langkah pembenahan dan optimasi yang telah diterapkan pada prototype modul eye tracking (`eye_tracking_prototype.py` dan `config.yaml`).

---

## 1. Identifikasi Masalah Awal

Pada versi prototype awal, ditemukan beberapa kendala utama saat pengujian:
1. **Kalibrasi Mengumpulkan Data Palsu:** Saat fase kalibrasi, sistem tetap merekam titik kalibrasi meskipun pengguna tidak sedang menatap titik tersebut.
2. **Hasil Tracking Horizontal Cukup Baik, Namun Vertikal Berantakan:** Tracking sumbu X (kiri, tengah, kanan) berfungsi relatif stabil, tetapi sumbu Y (atas, tengah, bawah) sering terdistorsi, kecuali pada 4 titik sudut layar.
3. **Kinerja Rendah (~11 FPS):** Meskipun kamera mendukung 30–60 FPS, eksekusi terhambat di kisaran 11 FPS.
4. **Variabel Hardcoded:** Beberapa parameter kalibrasi, pengolahan citra, dan pemetaan landmark ditulis langsung pada kode program (*hardcoded*).

---

## 2. Akar Penyebab Masalah (Root Cause Analysis)

### A. Penyebab Masalah Kalibrasi & Vertikal
- **Sensitivitas Pergerakan Kepala (Head Movement Sensitivity):** Estimasi iris sebelumnya diambil langsung dari koordinat piksel frame kamera tanpa dinormalisasi terhadap sudut mata. Pergeseran posisi kepala menyebabkan nilai koordinat iris berubah drastis meskipun arah pandangan mata tetap.
- **Oklusi Kelopak Mata (Eyelid Occlusion):** Saat mata melihat ke bawah, kelopak mata atas menutupi bagian atas iris, menyebabkan estimasi pusat iris oleh MediaPipe tergeser ke bawah secara tidak proporsional.
- **Overfitting Modeling Regresi Orde 2:** Penggunaan `estimateAffinePartial2D` awal tidak mampu menangani sifat non-linier gerakan bola mata. Ketika diubah ke Polynomial Orde 2 murni, sumbu Y mengalami *interpolation drift* di area tengah layar.
- **Ketidaksesuaian Cermin Kamera (Mirroring Issue):** Input kamera dari OpenCV tidak dibalik secara horizontal (`cv2.flip`), sehingga gerakan mata ke kiri di kamera terpetakan ke kanan di layar.

### B. Penyebab FPS Terbatas (~11 FPS)
- Pemanggilan `landmarker.detect_for_video()` dan pemrosesan `cv2.imshow()` dilakukan secara *synchronous* di dalam *main thread* yang sama dengan *video capture* I/O.

---

## 3. Langkah Pembenahan dan Optimasi yang Telah Diterapkan

### A. Normalisasi Vektor Mata Relatif (Head-Movement Invariant)
Sistem diubah untuk menghitung posisi iris relatif terhadap sudut mata luar dan dalam:
- **Left Eye:** Dihitung dari `landmark 33` (outer) dan `landmark 133` (inner).
- **Right Eye:** Dihitung dari `landmark 362` (inner) dan `landmark 263` (outer).
- **Formulasi:**
  $$\text{Norm}_X = \frac{\text{Iris}_X - \text{EyeCornerOuter}_X}{\text{EyeCornerInner}_X - \text{EyeCornerOuter}_X}$$

### B. Implementasi Fitur Eye Aspect Ratio (EAR)
Ditambahkan kalkulasi EAR untuk mengukur tingkat keterbukaan kelopak mata:
$$\text{EAR} = \frac{|P_2 - P_6| + |P_3 - P_5|}{2 \times |P_1 - P_4|}$$
Nilai EAR ini dimasukkan ke dalam fitur masukan regresi untuk mengompensasi distorsi vertikal akibat pergerakan kelopak mata.

### C. Pemetaan Regresi Hibrida (Hybrid Gaze Mapper)
Sistem pemetaan diubah menggunakan pendekatan **Hybrid Regression**:
- **Sumbu Horizontal (X):** Menggunakan Polynomial Regression Orde 2 ($1, x, y, x^2, y^2, xy$).
- **Sumbu Vertikal (Y):** Menggunakan Ridge Linear Regression yang memadukan $Y_{\text{norm}}$, $\text{EAR}$, dan *interaction term* ($Y_{\text{norm}} \times \text{EAR}$).

### D. Multi-threaded Async Video Capture (`WebcamStream`)
Dibuat kelas `WebcamStream` menggunakan Python `threading.Thread` untuk menangani *ingestion* frame dari webcam secara *asynchronous* di *background thread*. Hal ini mengeliminasi I/O *blocking* sehingga kecepatan frame dapat mencapai target 30–60 FPS.

### E. Modularisasi Penuh via `config.yaml`
Seluruh konfigurasi diekstrak ke dalam file `config.yaml` tanpa ada nilai *hardcoded*:
- `webcam.flip_horizontal: true`
- `webcam.async_capture: true`
- `webcam.fps_target: 60`
- `calibration.mapping_method: "hybrid"`
- `calibration.stability_threshold: 0.015`
- `calibration.stability_required_frames: 6`
- Landmark index mapping untuk iris, sudut mata, dan kelopak mata.

---

## 4. Kelemahan Teridentifikasi & Rencana Lanjutan

- **Penggunaan Kacamata:** Pantulan cahaya pada lensa kacamata (*glare*) atau bingkai kacamata yang menutupi sudut mata dapat memengaruhi akurasi landmark MediaPipe.
- **Rencana Lanjutan:** Integrasi modul komunikasi WebSocket antara Python service ini dengan VS Code Extension utama.
