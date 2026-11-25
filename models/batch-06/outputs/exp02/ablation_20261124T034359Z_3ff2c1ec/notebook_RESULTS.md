# Hasil ablasi baseline Batch 06 Exp02

Empat representasi memakai peserta, window, fold, dan Extra Trees yang sama dengan subset eye Exp01. Seluruh fitur tetap ada; yang berubah adalah normalisasi RMSSD dan/atau eye. SDNN tetap mentah. Test tidak dievaluasi; semua hasil merupakan eksplorasi development pada subset kecil.

## OOF peserta yang ditahan dari fitting

| Varian | Accuracy window | Macro-F1 tiga kelas | Mean accuracy orang | Mean Macro-F1 orang | Support Low/Medium/High |
|---|---:|---:|---:|---:|---|
| raw | 35.77% | 33.96% | 37.14% | 20.45% | 264/180/227 |
| rmssd_only | 38.15% | 35.95% | 39.35% | 22.47% | 264/180/227 |
| eye_only | 31.74% | 31.18% | 31.45% | 21.87% | 264/180/227 |
| combined | 41.43% | 39.85% | 41.93% | 31.06% | 264/180/227 |
| dummy | 39.34% | 18.82% | 40.33% | 19.05% | 264/180/227 |

Macro-F1 per orang memakai tiga kelas tetap, termasuk kelas yang tidak hadir pada fold tertentu. Rata-rata orang memberi setiap peserta bobot yang sama; accuracy window memberi setiap baris bobot yang sama.

## Kontras yang dibakukan

| Perlakuan dibanding kontrol | Delta accuracy | Delta Macro-F1 | Delta mean accuracy orang |
|---|---:|---:|---:|
| rmssd_only vs raw | +2.38 pp | +2.00 pp | +2.20 pp |
| eye_only vs raw | -4.02 pp | -2.77 pp | -5.69 pp |
| combined vs raw | +5.66 pp | +5.90 pp | +4.79 pp |
| combined vs rmssd_only | +3.28 pp | +3.90 pp | +2.59 pp |
| combined vs eye_only | +9.69 pp | +8.67 pp | +10.49 pp |

`eye_only vs raw` menilai transformasi eye saat RMSSD tetap asli. `combined vs rmssd_only` menilai penambahan transformasi eye saat RMSSD sudah relatif. Perbedaan keduanya menunjukkan ketergantungan pada representasi lain, bukan efek fisiologis kausal.

## Validation historis

Validation sudah diperiksa pada Exp01 dan tidak merupakan hold-out baru. Semua empat varian dibakukan sebelum fitting; tidak ada pemilihan ulang dari validation.

| Varian | Window | Accuracy | Macro-F1 tiga kelas | Recall Low | Recall Medium | Recall High | Support Low/Medium/High |
|---|---:|---:|---:|---:|---:|---:|---|
| raw | 99 | 52.53% | 33.46% | 20.93% | N/A | 76.79% | 43/0/56 |
| rmssd_only | 99 | 66.67% | 45.26% | 55.81% | N/A | 75.00% | 43/0/56 |
| eye_only | 99 | 59.60% | 39.61% | 37.21% | N/A | 76.79% | 43/0/56 |
| combined | 99 | 65.66% | 44.58% | 53.49% | N/A | 75.00% | 43/0/56 |

**Validation tidak mencakup ketiga kelas. Recall kelas dengan support nol adalah N/A. Macro-F1 tetap dihitung dengan tiga label untuk konsistensi dan memasukkan F1 nol pada kelas tanpa support. Accuracy validation tidak boleh dipromosikan sebagai pengujian tiga kelas lengkap atau pengguna baru.**

## Pemeriksaan kontrol Exp01

Raw dan combined dijalankan ulang dalam Exp02 sebagai kontrol internal. Kontras selalu dihitung dari hasil Exp02 yang baru, bukan mencampur skor lintas run.

| Protokol | Varian | Metrik sama dengan Exp01 |
|---|---|---|
| participant_held_out | raw | True |
| participant_held_out | combined | True |
| historical_block_cv | raw | True |
| historical_block_cv | combined | True |
| validation | combined | True |
| validation | raw | True |

Perbedaan kontrol perlu dibaca bersama versi paket dan numerik runtime. Tidak ada seed, parameter, atau fold yang diubah untuk memaksakan kecocokan.

## Batas kesimpulan

Acuan eye adalah pre_task, bukan pengukuran pupil atau baseline istirahat tervalidasi. Label tetap difficulty/fase; belum ada rating beban mental independen. Window overlap bukan observasi independen. Empat peserta tidak cukup untuk klaim populasi luas; perubahan deskriptif tidak dianggap signifikansi statistik.

Budget: 45 fitting CV dan 4 fitting Train penuh. Model validation disimpan sebagai empat pembanding, bukan pemenang final menurut validation.
