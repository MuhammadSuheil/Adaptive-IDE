# Dataset Fusion Eye Tracking dan HRV

## Versi output

```text
output/
  v1_active_task/
  v2_session_phase/
    window_15s_step_5s/
    window_25s_step_5s/
    window_45s_step_5s/
    QUALITY_REPORT.md
    replay_validation.json
    manifest.json
```

`v1_active_task` menyimpan seluruh 13 artefak lama byte-identik, termasuk CSV yang dipakai batch-01. Jangan menimpa arsip ini. `v2_session_phase` adalah hasil penghitungan ulang RR dan agregasi eye, bukan salinan v1 yang hanya diganti label/durasinya.

Setiap varian v2 berisi `all_data.csv`, `good_acceptable.csv`, dan `good_only.csv`, data dictionary, allowlist, distribusi kelas/durasi unik, inventory sesi, kualitas sensor, daftar window dikeluarkan, dan laporan ringkas. Laporan induk membandingkan ketiga panjang window. Belum ada training model pada tahap ini.

## Target v2

| Konteks sumber | cognitive_load_label |
|---|---|
| Active easy, practice, atau demo | Low |
| Active medium | Medium |
| Active hard | High |
| Pre-task setelah kedua kalibrasi selesai | Low |
| Feedback dan rating, termasuk setelah medium/hard | Low |
| Jeda antarset dan post-task sampai akhir rekaman | Low |
| Setelah penutupan set tercatat tanpa akhir penelitian berikutnya (interrupted) | Low |
| Baseline/kalibrasi, readiness tidak diketahui, atau interval tanpa ledger yang dapat ditelusuri | Tidak masuk dataset berlabel |

Seluruh bagian window harus menghasilkan **satu target yang sama**. Easy + rating atau rating + rest dapat tetap Low, walaupun `study_phase=mixed`. Medium + rating atau hard + rest dikeluarkan. Tidak ada majority vote; transisi singkat tetap diperiksa. Event tepat pada endpoint mempunyai durasi nol dalam window sebelumnya.

`cognitive_load_label` merupakan satu-satunya target model. `difficulty_level`, fase, set, soal, identitas, waktu, dan kualitas tetap metadata asal. Contoh: rating setelah hard mempunyai difficulty hard dan target Low. Jangan membentuk ulang target v2 dari difficulty pada notebook.

## Fitur dan kualitas

Allowlist setiap varian menetapkan **26 fitur sensor**: RMSSD, SDNN, rasio baseline RMSSD, kedipan, mean/std EAR, estimasi ukuran iris/perubahannya, head pose, gaze ternormalisasi, dan proporsi Eye_State. Metadata tidak masuk predictor.

Estimasi iris bukan diameter pupil; Eye_State adalah keluaran algoritma. Nilai hilang tetap kosong, tidak diganti nol. Imputasi, scaling, seleksi fitur, class weight, ROS, dan SMOTE merupakan pekerjaan training-fold di pemodelan, bukan modifikasi CSV sumber.

Aturan window dan admission peserta tetap mengikuti `config.json`:

- `good_only`: tier good dan anggota proposed_good (Andrean, Fabian, Faqih, Ikhsan, Kaka).
- `good_acceptable`: tier good/acceptable dan anggota proposed_good/questionable (tambahan Andreas, Athaya, Rafi, Raina, Sam, Savitri, Shakilla, Tio).
- `all_data`: seluruh window berlabel, termasuk tier poor dan peserta unreviewed.
- Nama cocok tanpa membedakan kapitalisasi. Ramadhan/Ramadhan-2 tetap satu grup orang, dengan baseline per rekaman; salinan acquisition dipilih sekali oleh inventory.
- Rekaman baru harus ditinjau dan dimasukkan ke admission yang sesuai sebelum masuk dataset curated; status anggota tim bukan keputusan kualitas.

| Kriteria | Good | Acceptable |
|---|---:|---:|
| HRV | Ready, RMSSD/SDNN finite dan nonnegatif | Sama |
| Coverage temporal eye, bin 100 ms | ≥95% | ≥80% |
| Face + EAR valid | ≥95% | ≥80% |
| Gaze valid | ≥90% | ≥70% |
| Iris valid nonblinking | ≥90% | ≥70% |
| Clipping di antara gaze valid | ≤5% | ≤20% |

HRV memerlukan accepted-RR coverage minimal 80%, minimal 10 pasangan, serta data age maksimal 3 detik. Ambang tidak dilonggarkan untuk window 15 detik. Kualitas good/acceptable menyatakan kelengkapan/clipping, bukan pengganti hasil validasi spasial gaze. Hasil validasi asli tetap metadata.

`baseline_recorder_ready` dipisahkan dari `baseline_usable`. Rasio hanya dihitung jika baseline ready, RMSSD positif, dan coverage ≥80%; alasan penolakan tercatat dan rasio tetap kosong bila tidak layak. Raw HRV yang layak tidak ikut dibuang hanya karena rasio tidak tersedia.

## Rekonstruksi waktu dan RR

Semua varian memakai task-origin sensor yang sama dan grid step 5 detik. Setiap RMSSD/SDNN dihitung dari RR mentah pada interval sebenarnya; fitur eye diagregasi pada batas absolut yang sama, dengan endpoint sensor `(start, end]`.

Replay mempertahankan urutan packet/raw ID, interval ditolak, segment/gap, dan availability recovery. RR sebelum task sensor tidak masuk replay. Paket setelah endpoint tidak digunakan; paket tepat pada endpoint termasuk jika sudah tersedia. Pasangan RMSSD tidak menyambung RR terbuang atau segmen putus. Window parsial terakhir tidak dibuat.

Cache HRV 25 detik divalidasi terhadap rekonstruksi. Pada sumber sekarang, 6.659 dari 6.689 window cocok dengan replay utama; 30 cocok dengan timer live yang menerbitkan window sebelum paket tepat di endpoint. Tidak ada selisih tak terjelaskan. `replay_validation.json` menyimpan rinciannya. Dataset v2 memakai satu aturan endpoint konsisten, bukan memilih aturan berbeda per window untuk meniru cache.

Durasi unik adalah gabungan interval pada tiap rekaman/kelas, bukan penjumlahan durasi row yang overlap. Step lebih pendek menambah overlap, bukan peserta/observasi independen baru.

## Menjalankan builder

Dari root repository:

```powershell
python analysis_sessions/dataset_fusion/build_dataset_v2.py
python -m unittest discover -s analysis_sessions/dataset_fusion -p "test_*.py" -v
```

Default sumber: `sessions/saved_session`. Default tujuan: `output/v2_session_phase`. **Jika tujuan sudah ada, builder menolak menimpanya.** Untuk sumber tambahan atau rebuild verifikasi, pilih folder versi baru:

```powershell
python analysis_sessions/dataset_fusion/build_dataset_v2.py --output analysis_sessions/dataset_fusion/output_next_version/v2_session_phase
```

`--input`, `--config`, dan `--subjects` tersedia. Subjects memakai nama folder persis dan tidak mengubah ambang kualitas. Satu build selalu menghasilkan ketiga panjang window, sehingga CSV dan manifest tidak tercampur antar-run.

Hash seluruh sumber diverifikasi sebelum publikasi. Bundle lengkap dibangun pada staging, manifest ditulis terakhir, lalu direktori dipublikasikan melalui satu rename. Kegagalan parsing, replay, integritas sumber, atau penulisan tidak menerbitkan bundle parsial. Pada Windows, file turunan memakai permission normal folder tujuan.

Builder v1 tetap tersedia untuk audit dengan `--output` eksplisit ke folder rebuild terpisah. Root output versi dan arsip v1 yang sudah ada dilindungi. Logika angka/label historis v1 tidak diubah.

## Pemodelan dan Drive

Baca [models/README.md](../../models/README.md) dan [plan batch-02](../../models/batch-02/PLAN.md). Path Drive mengikuti struktur versi, misalnya:

`drive/MyDrive/Penelitian-Adaptive-IDE/Dataset/v2_session_phase/window_25s_step_5s/good_acceptable.csv`

Sertakan allowlist dan manifest, bukan CSV saja. Dataset v2 yang sekarang memakai rekaman lama yang tersedia; belum berisi pengambilan Low tambahan. Setelah sumber tambahan, identitas lintas sesi, dan admission disahkan, bangun versi penerimaan baru dan bekukan manifest evaluasi batch-02. Jangan memakai notebook batch-01 langsung pada target v2 atau memakai manifest test good-only v1 sebagai evaluasi seluruh good+acceptable batch-02.
