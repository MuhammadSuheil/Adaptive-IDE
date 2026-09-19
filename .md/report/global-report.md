# Global Report

**Laporan/Log Perubahan scope global**

## Ringkasan sesi terakhir

Prototype eye tracking dan HRV diarahkan menjadi satu alur sesi multimodal. Webcam dan sensor HRV harus aktif bersamaan, tetapi setiap sumber tetap menyimpan log sendiri. Sinkronisasi baru dibuat setelah user menghentikan sesi.

Alur target:

```text
python sync/synchronize_sessions.py
  -> webcam + sensor HRV aktif
  -> konfigurasi/head gate + eye calibration
  -> eye CSV dan HRV CSV direkam terpisah
  -> user menekan Q/Stop
  -> kedua proses dihentikan
  -> synchronized_features.csv dibuat
```

## Lokasi output

```text
eye_tracking_prototype/sessions/
  session_<shared_id>_<timestamp>.csv

hrv_monitor_prototype/sessions/<shared_id>/
  rr_raw.csv
  hrv.csv

sync/sessions_sync/<shared_id>/
  synchronized_features.csv
```

`rr_raw.csv` dipertahankan untuk audit dan pemrosesan ulang. File sumber tidak ditimpa oleh proses sinkronisasi.

## Perubahan kode utama

### Screen resolution

`screen.width` dan `screen.height` sekarang berasal dari `eye_tracking_prototype/config.yaml`, bukan autodetection monitor atau fallback hardcoded. Nilai tersebut mengatur koordinat fullscreen calibration dan berbeda dari resolusi webcam.

### Head positioning / oval UI

`eye_tracking_prototype/eye_tracking_prototype.py` sekarang memiliki UI live untuk menyesuaikan oval sebelum head calibration:

- slider tinggi oval;
- slider lebar oval;
- slider posisi horizontal;
- slider posisi vertikal;
- tombol start calibration;
- tombol reset ke nilai config;
- perubahan hanya berlaku untuk sesi aktif dan tidak menulis balik `config.yaml`;
- peringatan agar lengan dan badan tetap diam.

Head gate juga memiliki toleransi configurable dan grace frame untuk mengurangi reset countdown akibat jitter landmark.

Konfigurasi terkait terdapat di `config.yaml`:

```yaml
head_positioning:
  guide_height_ratio: 0.55
  guide_width_to_height: 0.66
  target_center_x_ratio: 0.5
  target_center_y_ratio: 0.55
  max_outside_ratio: 1.12
  min_vertical_fill_ratio: 0.65
  min_horizontal_fill_ratio: 0.30
  center_tolerance_ratio: 0.25
  max_yaw_deg: 30.0
  max_pitch_deg: 30.0
  invalid_grace_frames: 8
  countdown_seconds: 5
```

Catatan: `target_face_width_ratio`, `alignment_tolerance_ratio`, dan `size_tolerance_ratio` masih ada di config tetapi belum menjadi syarat utama kelulusan gate; keputusan aktual terutama menggunakan landmark terhadap ellipse, pose, dan toleransi baru.

### Coordinator dan synchronization

`sync/synchronize_sessions.py` memiliki dua fungsi:

1. Mode default tanpa argumen: membuat shared session ID, menjalankan `eye_tracking_prototype.py` dan `hrv_monitor.py` sebagai dua subprocess, menunggu eye tracking berhenti, menghentikan HRV, lalu membuat log sinkron.
2. Mode `--sync-only`: membaca dua log yang sudah selesai dan membuat output sinkron tanpa menyalakan sensor.

Contoh mode live:

```powershell
python sync/synchronize_sessions.py
```

Contoh mode offline:

```powershell
python sync/synchronize_sessions.py --sync-only `
  eye_tracking_prototype/sessions/<eye_file>.csv `
  hrv_monitor_prototype/sessions/<session_id>/hrv.csv
```

Sinkronisasi membuat satu baris per window HRV, bukan mengulang nilai HRV pada setiap frame eye. Eye metrics diringkas di interval `window_start` sampai `window_end` yang sama.

### HRV session output

`hrv_monitor_prototype/hrv_monitor.py` menerima `--session-id`, membuat folder sesi, menulis `rr_raw.csv`, dan menulis `hrv.csv` dengan timestamp UTC/window fields yang dapat dipasangkan dengan timestamp epoch eye tracking. Shared ID juga diteruskan ke eye tracker melalui `--session-id`.

## Temuan paper dan interpretasi teknis

Paper yang paling relevan di `papers/eye_tracking/calibration/`:

- `A review on personal calibration issues for video-oculographic-based gaze tracking.pdf`
- `Webcam-based gaze estimation for computer screen interaction.pdf`
- `Towards efficient calibration for webcam eye-tracking in online experiments.pdf`

Review tersebut mendukung penggunaan pose kepala jika pose dipakai dalam model geometris/transformasi 3D atau dikalibrasi pada beberapa pose. Paper webcam juga menekankan kepala stabil saat calibration; kompensasi gerakan kepala memerlukan model tambahan seperti transformasi 3D/Structure from Motion.

Implementasi sekarang belum melakukan head-pose compensation 3D. `GazeMapper` memakai fitur `[iris_x, iris_y, EAR]`; posisi oval tidak menjadi input langsung mapper. Karena itu memindahkan oval vertikal tidak secara langsung memindahkan koordinat gaze, tetapi dapat mengubah posisi/sudut kepala yang diterima. User bisa menundukkan kepala agar masuk oval, lalu perubahan pitch/geometri mata dapat menyebabkan bias gaze, termasuk bias ke bawah.

Rekomendasi desain lanjutan:

- Horizontal position dikunci di `0.50` agar kamera/wajah tetap terpusat.
- Vertical position diberi rentang kecil, misalnya `0.47..0.53`, bukan bebas jauh ke bawah.
- Width/height tetap boleh disesuaikan untuk UX, tetapi perlu batas agar oval tidak terlalu gepeng atau ekstrem.
- Untuk jarak, gunakan `face_width_ratio`/ukuran landmark sebagai validasi relatif (`terlalu dekat`, `ideal`, `terlalu jauh`), bukan menganggap oval sebagai pengukur sentimeter.
- Baseline pose sebaiknya memakai median beberapa frame valid, bukan frame terakhir.
- Jika ingin kompensasi head pose sungguhan, masukkan pitch/yaw ke model dengan sampel kalibrasi pose yang sesuai; menambah fitur tanpa sampel pose beragam tidak cukup.
- Nilai `max_pitch_deg: 30` dan `max_yaw_deg: 30` cukup longgar untuk tracking webcam; uji awal yang lebih ketat sekitar 10--12 derajat lebih aman.

## Validitas data dan pencahayaan

Rencana notifikasi kualitas mencakup:

- pencahayaan ideal/redup/terang;
- backlight atau wajah jauh lebih gelap dari latar;
- kontras wajah rendah;
- wajah tidak terdeteksi;
- kepala bergeser dari baseline;
- sensor HRV kehilangan kontak atau mengalami gap/artifact;
- peringatan agar user tidak menggerakkan badan/lengan.

Gerakan lengan tidak dapat dipastikan hanya dari webcam; indikator yang dapat dipakai adalah artifact RR, contact loss, atau gap sensor. Status kualitas harus dibedakan dari klaim pasti tentang gerakan.

## Verifikasi terakhir

- Syntax `eye_tracking_prototype.py`, `hrv_monitor.py`, dan `sync/synchronize_sessions.py` telah diperiksa dengan `ast.parse`.
- `sync/synchronize_sessions.py --help` dan `hrv_monitor.py --help` berhasil dijalankan.
- Output collision pada sinkronisasi tidak menimpa file lama; default output menggunakan folder session dan nama `synchronized_features.csv`, dengan suffix versi jika diperlukan.

## Catatan pekerjaan berikutnya

1. Uji satu sesi nyata dengan `python sync/synchronize_sessions.py` pada webcam dan sensor HW9.
2. Pastikan eye CSV dan HRV CSV menggunakan shared session ID yang sama.
3. Periksa apakah proses HRV menerima Ctrl-Break/stop dengan bersih di Windows; jika tidak, tambahkan mekanisme stop event lintas proses.
4. Validasi kualitas sinkronisasi menggunakan sesi eye dan HRV yang benar-benar dilakukan pada waktu dan user yang sama. Jangan memasangkan eye log 15 September dengan HRV log 19 September.
5. Implementasikan quality gate pencahayaan dan adaptive distance feedback setelah lifecycle sesi tervalidasi.
6. Pertimbangkan mengganti nama coordinator menjadi `run_synchronized_session.py` jika nama `synchronize_sessions.py` membingungkan; fungsi sinkronisasi offline tetap dapat dipertahankan.
