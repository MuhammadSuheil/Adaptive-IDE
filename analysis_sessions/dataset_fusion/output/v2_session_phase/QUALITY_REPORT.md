# Dataset fusion v2: session phase

## Label dan window

`cognitive_load_label` adalah satu-satunya target. Active easy/practice/demo, pre-task setelah kedua kalibrasi selesai, feedback, rating, jeda antarset, dan post-task diberi Low. Rating/feedback setelah medium/hard juga Low. Active medium diberi Medium; active hard diberi High. Interrupted hanya menunjuk interval setelah event set_ended yang tercatat, tanpa event akhir penelitian berikutnya. Baseline/kalibrasi dan interval tanpa ledger/readiness yang dapat ditelusuri tidak diberi Low.

Setiap bagian interval harus menghasilkan target yang sama. Campuran easy + rating tetap Low; medium + rating tidak masuk dataset. Aturan ini bukan majority vote. Window mixed-phase dapat dipertahankan; `window_phase_mix` dan `window_label_mix` menyimpan rinciannya. Difficulty/set/soal sumber tidak ditimpa agar sama dengan target.

## Rekonstruksi sensor

Window 15/25/45 detik dihitung ulang dari rr_raw.csv dan eye_tracking.csv dengan step 5 detik, memakai origin task dan grid yang sama per sesi. Tidak ada interpolasi, averaging RMSSD tersimpan, atau perubahan timestamp mentah. RR sebelum task sensor tidak masuk replay. Paket yang tiba setelah endpoint tidak dapat memperbaiki window yang telah lewat. Paket yang tiba tepat di endpoint diproses sebelum window diterbitkan. Recovery menunggu waktu rilisnya; pasangan RMSSD tidak melintasi RR terbuang atau segmen terputus.

Syarat HRV tetap coverage 80%, minimal 10 pasangan, umur data maksimal 3 detik. Ambang eye tetap mengikuti config. Admission v2 tidak menggunakan daftar nama: good_only memuat tier good; good_acceptable memuat tier good/acceptable; all_data memuat semua window berlabel termasuk sensor poor. team_review_group hanya metadata historis. Tier menilai kelengkapan/clipping, bukan akurasi spasial gaze; hasil validasi asli tetap dipertahankan.

Baseline recorder ready dipisahkan dari baseline usable builder. Rasio hanya tersedia untuk baseline ready dengan RMSSD positif dan coverage minimal 80%. Penolakan tercatat pada baseline_rejection_reason; nilai hilang tetap kosong, bukan nol.

## Validasi cache HRV 25 detik

Diperiksa 7460 window: 7429 sesuai replay utama; 31 sesuai urutan timer live sebelum paket tepat di endpoint; tidak ada selisih yang tidak terjelaskan. Semua rincian ada di replay_validation.json. Dataset baru memakai aturan endpoint yang konsisten, tidak menyalin angka cache agar tampak identik.

## Distribusi

| Window | Dataset | Low | Medium | High | Total | Peserta |
|---|---|---:|---:|---:|---:|---:|
| window_15s_step_5s | good_only | 522 | 724 | 642 | 1888 | 18 |
| window_15s_step_5s | good_acceptable | 1062 | 1266 | 1251 | 3579 | 19 |
| window_15s_step_5s | all_data | 1935 | 2001 | 1964 | 5900 | 20 |
| window_25s_step_5s | good_only | 489 | 630 | 574 | 1693 | 15 |
| window_25s_step_5s | good_acceptable | 984 | 1093 | 1137 | 3214 | 19 |
| window_25s_step_5s | all_data | 1760 | 1633 | 1696 | 5089 | 20 |
| window_45s_step_5s | good_only | 455 | 381 | 440 | 1276 | 14 |
| window_45s_step_5s | good_acceptable | 908 | 728 | 891 | 2527 | 18 |
| window_45s_step_5s | all_data | 1572 | 1052 | 1250 | 3874 | 20 |

## Penyaringan window

| Window | Kandidat | Target campuran | Kalibrasi/tidak terlacak | Berlabel, HRV tidak layak | Tier poor |
|---|---:|---:|---:|---:|---:|
| 15s | 7502 | 1602 | 0 | 857 | 2321 |
| 25s | 7460 | 2371 | 0 | 674 | 1875 |
| 45s | 7376 | 3502 | 0 | 512 | 1347 |

Kolom HRV/tier poor merupakan diagnostik pada window berlabel dan dapat saling overlap. Tier poor juga dapat berasal dari kualitas eye; rincian kriteria tersimpan di CSV. all_data tidak membuang tier poor. CSV curated hanya menerapkan tier sensor, tanpa pembatasan nama peserta.

## Komposisi fase pada Low good_acceptable

| Window | Fase sumber | Window Low |
|---|---|---:|
| 15s | active_task | 460 |
| 15s | feedback | 93 |
| 15s | interrupted | 3 |
| 15s | mixed | 447 |
| 15s | post_task | 4 |
| 15s | pre_task | 34 |
| 15s | rest | 21 |
| 25s | active_task | 319 |
| 25s | feedback | 40 |
| 25s | interrupted | 1 |
| 25s | mixed | 580 |
| 25s | post_task | 2 |
| 25s | pre_task | 26 |
| 25s | rest | 16 |
| 45s | active_task | 168 |
| 45s | feedback | 4 |
| 45s | mixed | 712 |
| 45s | pre_task | 12 |
| 45s | rest | 12 |

mixed berarti beberapa fase dengan target Low yang sama. Detail per segmen ada pada window_phase_mix; jumlah window bukan durasi unik dan tidak menyatakan semua Low berasal dari pengerjaan easy.

## Pemodelan berikutnya

Gunakan 26 predictor sensor pada feature_columns.json, bukan semua kolom numerik. Target/difficulty/fase, nama peserta, ID soal, waktu, dan kualitas adalah metadata atau label, bukan X. Identitas tetap dipakai untuk grouping/deduplikasi. Imputasi, scaling, seleksi fitur, class weight dan oversampling dilakukan pada training fold, tidak pada CSV sumber. Builder tidak melatih model; sumber tambahan yang diterima dijelaskan pada ringkasan pembaruan. Batch-01 tetap arsip; batch-02 memakai evaluasinya sendiri.

Window overlap bukan observasi independen. Durasi unik dilaporkan per kelas/rekaman pada class_distribution.csv. Mutu sensor dan komposisi kelas dapat berubah antarwindow; jumlah row terbesar bukan bukti window terbaik. Endpoint pembanding dan manifest evaluasi batch-02 dibuat setelah data tambahan diterima.

## Pembaruan sumber pada build ini

Rekaman tanpa penanda aktivitas tidak diberi label Low secara manual. Eksklusi yang dikonfigurasi memakai session ID sehingga perubahan nama folder tidak mengubah keputusan.

| Folder yang tidak dipakai | Session ID | Alasan |
|---|---|---|
| bangsur_easy_only | 20261116_151757_P01_coding_task | excluded_session: missing_task_events_operator_not_connected: operator confirmed that bangsur_easy_only was not connected to CodeArrange; activity labels are unavailable. |

Sumber yang dikecualikan tetap disimpan dan di-hash. Tidak ada event atau ID soal yang dibuat untuk melengkapinya.

### Perubahan good_acceptable

| Window | Low sebelum | Medium sebelum | High sebelum | Tambahan Low | Tambahan Medium | Tambahan High | Total sesudah | Rasio terbesar/terkecil |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 15s | 870 | 1134 | 1251 | 192 | 132 | 0 | 3579 | 1.192 |
| 25s | 797 | 988 | 1137 | 187 | 105 | 0 | 3214 | 1.155 |
| 45s | 732 | 657 | 891 | 176 | 71 | 0 | 2527 | 1.247 |

Rasio tersebut menjelaskan proporsi, bukan ambang kecukupan statistik. Builder mempertahankan semua window yang lolos, tanpa kuota kelas atau pemotongan manual.

### Kontribusi rekaman yang belum ada pada build sebelumnya

| Window | Rekaman | Low | Medium | High | Durasi unik Low (menit) | Durasi unik Medium (menit) |
|---|---|---:|---:|---:|---:|---:|
| 15s | bangsur_easy_medium | 95 | 113 | 0 | 8.67 | 10.92 |
| 25s | bangsur_easy_medium | 96 | 95 | 0 | 8.83 | 10.92 |
| 45s | bangsur_easy_medium | 97 | 68 | 0 | 9.25 | 8.33 |
| 15s | ihsan | 97 | 19 | 0 | 13.42 | 2.83 |
| 25s | ihsan | 91 | 10 | 0 | 14.75 | 2.67 |
| 45s | ihsan | 79 | 3 | 0 | 14.33 | 1.58 |

Durasi unik merupakan gabungan interval yang diterima, bukan jumlah row dikali panjang window. Satu sesi dapat menyumbang jumlah berbeda pada setiap panjang window. Rincian identitas dan delta tersimpan pada update_summary di manifest.

## Proses pembuatan dan pembaruan

Build ini membaca 21 rekaman dari 20 orang pada sessions/saved_session. Inventory memilih satu salinan acquisition dan menyatukan identitas orang yang mempunyai beberapa rekaman. Sumber yang diterima dan dikecualikan dicatat pada inventory serta manifest.

1. Builder memeriksa identitas sensor/ledger, readiness kedua kalibrasi, timestamp, dan grid HRV sumber.
2. Builder mengolah RR mentah menurut urutan paket dan waktu ketersediaan, lalu menghitung ulang HRV pada tiga panjang window.
3. Builder menggabungkan frame eye pada interval absolut yang sama dan menentukan target dari seluruh bagian fase di dalam window.
4. Builder menghitung tier sensor dan membentuk tiga view tanpa membatasi nama peserta atau kategori penilaian tim lama.
5. Builder memeriksa label, nilai sensor, subset, jumlah data, serta checksum sebelum mengganti bundle aktif.

Pembaruan admission memperluas cakupan window yang sudah memenuhi ambang sensor. Kita tidak menambah baris sintetis, mengubah label untuk menyeimbangkan kelas, atau melonggarkan ambang kualitas. Data mentah dan arsip v1 tetap utuh. Manifest mencatat admission_policy=window_quality_only; team_review_group merupakan metadata historis.

## Ambang kualitas window

| Kriteria | Good | Acceptable |
|---|---:|---:|
| HRV | Ready, RMSSD/SDNN finite dan nonnegatif | Sama |
| Coverage temporal eye, bin 100 ms | ≥95% | ≥80% |
| Face dan EAR valid | ≥95% | ≥80% |
| Gaze tersedia/valid menurut recorder | ≥90% | ≥70% |
| Iris valid, nonblinking | ≥90% | ≥70% |
| Clipping pada gaze valid | ≤5% | ≤20% |

Good/acceptable adalah tier kelengkapan/clipping pada window. Hasil validasi spasial gaze tetap merupakan pemeriksaan terpisah dan tidak berubah menjadi lolos karena sebuah window bertier good.

## Kondisi sensor dan nilai hilang

Dari 21 rekaman sumber, validasi gaze asli mencatat 2 lolos, 19 gagal, dan 0 tanpa keputusan. Baseline HRV layak pada 18 rekaman. Angka ini mencakup semua sumber, bukan hanya rekaman yang mempunyai window curated.

Kegagalan validasi gaze tidak otomatis membuktikan EAR, blink, atau HRV pada rekaman yang sama tidak dapat digunakan. Kita mempertahankan hasil pengukuran serta status validasi aslinya; pemilihan kelompok fitur merupakan tahap pemodelan.

- Rasio RMSSD kosong ketika baseline tidak ready, tidak positif, atau coverage di bawah ambang; HRV absolut tetap tersedia bila window-nya layak.
- Blink kosong ketika counter terputus/tidak diketahui atau tidak ada frame. Nol berarti tidak ada penambahan kedipan yang teramati.
- Estimasi iris memerlukan frame valid dan tidak blinking; perubahan iris juga memerlukan baseline iris ready.
- all_data dapat memuat HRV tidak layak atau fitur eye kosong. Kedua view curated tetap memerlukan tier sensor yang lolos.

## Fitur dan kolom yang dipakai

Setiap CSV menyediakan 26 fitur sensor: tiga HRV, dua blink, empat EAR, empat iris, empat head pose, empat gaze, dan lima proporsi eye-state. feature_columns.json menetapkan allowlist dan cognitive_load_label sebagai target. data_dictionary.csv menjelaskan satuan dan definisi tiap kolom.

RMSSD/SDNN memakai milidetik; blink rate memakai kejadian per menit. EAR serta ukuran iris memakai rasio; ukuran iris adalah estimasi landmark relatif terhadap lebar mata, bukan diameter pupil. Head pose memakai derajat; koordinat gaze dinormalisasi dengan ukuran layar dan dapat berada di luar rentang 0–1. Eye-state adalah kategori keluaran algoritma, bukan anotasi manual gerakan mata.

Identitas, difficulty, fase, set/soal, waktu, dan kualitas tetap metadata. Label v2 harus dibaca dari cognitive_load_label, bukan dibuat kembali dari difficulty. Builder tidak melakukan imputasi, scaling, seleksi fitur, history, SMOTE, atau training.

## Pemeriksaan build

Pemeriksaan pembanding memeriksa 868 window sensor pada ketiga panjang window, termasuk sampel awal/tengah/akhir dan perubahan status ready/gap/recovery bila tersedia. Perhitungan pembanding memakai RR hasil preprocessing yang tersedia sebelum endpoint dan agregasi NumPy atas frame eye yang telah diperiksa validitasnya. Pemeriksaan ini menguji penggabungan dan rumus, bukan memvalidasi algoritma pengukuran gaze terhadap ground truth baru.

Validasi seluruh row memeriksa identitas/duplikasi, zona waktu dan durasi, grid 5 detik, target/fase, nilai infinity/negatif yang tidak sah, HRV readiness, baseline ratio, blink rate, dan jumlah proporsi eye-state. Setiap view harus memenuhi good_only ⊆ good_acceptable ⊆ all_data dengan nilai identik pada window yang sama. Checksum dan jumlah data diverifikasi sebelum dan setelah publikasi. Rincian mesin tersimpan pada manifest.json dan replay_validation.json.

## Menambahkan hasil pengambilan berikutnya

1. Tambahkan folder sesi lengkap ke sessions/saved_session: eye_tracking.csv, hrv.csv, rr_raw.csv, task_events.csv, serta file readiness/ringkasan kalibrasi yang digunakan builder. Jangan menimpa rekaman lama.
2. Pastikan ID sensor dan ledger konsisten. Konfirmasikan identitas lintas sesi pada inventory; rekaman ulang orang yang sama harus memakai grup orang yang sama. Nama folder yang berubah tidak boleh membuat acquisition lama dihitung dua kali.
3. Sumber baru tidak perlu masuk daftar proposed_good/questionable. Builder menilai setiap window memakai aturan sensor yang sama. Jika operator memakai alias baru untuk orang lama, perbarui pemetaan identitas yang dapat dikonfirmasi sebelum build.
Sesi tanpa ledger harus dipulihkan penandanya atau dikecualikan secara eksplisit melalui session_exclusions; jangan membuat event atau label pengganti. Sumber bermasalah yang tidak dikecualikan menghentikan publikasi.
4. Jalankan tes, lalu rebuild dari root repository dengan perintah berikut:

```powershell
python -m unittest discover -s analysis_sessions/dataset_fusion -p "test_*.py"
python analysis_sessions/dataset_fusion/build_dataset_v2.py --replace
```

5. Periksa status berhasil, distribusi kelas, kualitas, dan alasan exclusion. Salin bundle beserta manifest dan allowlist ke Drive bila dibutuhkan. Setelah data tambahan lengkap, tetapkan input batch-02 dari bundle terbaru.

Builder membuat staging sementara dan menahan bundle lama sebagai rollback selama pemasangan. Setelah verifikasi berhasil, builder menghapus rollback; tidak ada arsip dataset permanen tambahan. Jika pemulihan juga gagal, builder menampilkan lokasi rollback dan mempertahankannya untuk pemulihan operator.

## Hubungan dengan hasil pemodelan

Hasil batch-00/batch-01 tetap merujuk hash dataset saat masing-masing training. Pembaruan CSV ini tidak melatih ulang model atau mengganti skor lama. Admission berbasis kualitas tetap berlaku setelah sumber tambahan digabung; kenaikan skor model belum diuji pada bundle ini. Tiga window good_acceptable adalah kandidat utama untuk pemodelan berikutnya, sedangkan v1 menjadi referensi historis. Ringkasannya tersedia pada [README output](../README.md).
