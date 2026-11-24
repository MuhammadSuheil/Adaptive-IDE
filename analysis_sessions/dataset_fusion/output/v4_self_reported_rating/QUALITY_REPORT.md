# Kualitas dataset v4

Target `difficulty_rating` adalah penilaian kesulitan soal yang dipilih peserta pada skala 1–10. Satu rating berlaku untuk satu episode pengerjaan. Skala ini tidak menggantikan label cognitive load v2 atau v3.

## Sumber rating dan cakupan episode

Dari 500 baris di empat log CodeArrange, 164 rating cocok dengan 164 episode dari 17 peserta. Sebanyak 127 rating mengikuti jawaban manual benar dan 37 mengikuti timeout. Dari 358 episode yang direkonstruksi, 194 tidak memperoleh label.

Pencocokan memakai `attempt_id`, `session_id`, set, identitas soal, jenis submission, hasil, dan timestamp. Lihat `rating_matches.csv` untuk sumber file dan baris, termasuk rating yang tidak menemukan pasangan. Nama peserta atau kemiripan judul saja tidak cukup untuk memasangkan log.

| Keputusan pada baris log | Baris |
|---|---:|
| `attempt_not_in_source_cohort` | 23 |
| `matched` | 164 |
| `missing_rating` | 313 |

Sesi sumber tanpa rating yang cocok: Alif, Mon, Shakilla, bangsur_easy_medium, ihsan.

| Penutup episode tanpa label | Episode |
|---|---:|
| `correct` | 89 |
| `set_changed` | 1 |
| `set_ended` | 97 |
| `timeout` | 6 |
| `trace_tidak_selesai` | 1 |

Aplikasi meminta rating setelah jawaban benar atau timeout. Soal yang dilewati, ditinggalkan, atau berakhir tanpa angka rating tidak masuk CSV berlabel. Rating yang tercatat setelah timeout tetap memakai angka pilihan peserta. `episode_ratings.csv` menyimpan episode tanpa rating untuk audit. Cakupan ini membatasi kesimpulan: dataset tidak mewakili semua soal yang sempat dibuka.

## Pemilihan window dan kualitas sensor

Satu window harus utuh dalam satu interval aktif dari episode berating. Retry dan kembali ke soal sebelum episode ditutup tetap bagian dari episode itu. Feedback, rating, istirahat, dan perpindahan soal memutus interval. Dataset tidak menyambung potongan waktu untuk memenuhi durasi window.

V4 memakai pool `all_data.csv` v2 sebelum kuota kelas v2. V4 mempertahankan agregat sensor, filter RR dan eksklusi timestamp v2. Pada CSV curated, fitur gaze yang tidak lolos syarat v2 tetap kosong. Tidak ada imputasi, normalisasi global, resampling, atau pembuatan rating sintetis.

| Dataset 15 detik | Window | Episode | Peserta | Durasi unik (menit) |
|---|---:|---:|---:|---:|
| `good_only` | 1299 | 107 | 16 | 147.83 |
| `good_acceptable` | 1693 | 126 | 17 | 181.58 |
| `all_data` | 2082 | 150 | 17 | 217.33 |

14 episode berating tidak memperoleh window 15 detik dalam pool yang cocok. Dari jumlah itu, 9 tidak memiliki interval aktif tunggal sepanjang 15 detik. Sisanya tidak memperoleh window utuh pada grid 5 detik atau terkendala cakupan dan eksklusi sumber. Filter kualitas mengeluarkan 389 window dari dataset utama. Sebanyak 24 episode yang semula punya window tidak lagi terwakili setelah filter kualitas.

| Eksklusi saat mencocokkan pool 15 detik | Window |
|---|---:|
| `invalid_event_trace` | 6 |
| `missing_rating` | 2952 |
| `not_single_active_episode` | 3192 |

Eksklusi sebelum pool v2 tetap berlaku. Jumlah berikut mencakup seluruh cohort sumber, bukan hanya episode berating.

| Window | Alasan eksklusi v2 | Jumlah |
|---|---|---:|
| 15s | `ambiguous_eye_timestamp` | 7 |
| 15s | `interrupted` | 73 |
| 15s | `missing_task_context` | 158 |
| 15s | `outside_optional_grace` | 104 |
| 25s | `ambiguous_eye_timestamp` | 11 |
| 25s | `interrupted` | 73 |
| 25s | `missing_task_context` | 247 |
| 25s | `outside_optional_grace` | 118 |
| 45s | `ambiguous_eye_timestamp` | 19 |
| 45s | `interrupted` | 73 |
| 45s | `missing_task_context` | 405 |
| 45s | `outside_optional_grace` | 146 |

## Distribusi dataset utama

| Rating | Episode berating | Episode layak | Peserta layak | Window layak | Durasi unik (menit) |
|---:|---:|---:|---:|---:|---:|
| 1 | 39 | 31 | 9 | 186 | 23.00 |
| 2 | 21 | 14 | 8 | 104 | 12.17 |
| 3 | 24 | 21 | 10 | 222 | 23.67 |
| 4 | 26 | 14 | 8 | 163 | 17.25 |
| 5 | 14 | 12 | 10 | 110 | 12.17 |
| 6 | 13 | 11 | 9 | 201 | 22.17 |
| 7 | 8 | 7 | 5 | 251 | 24.83 |
| 8 | 4 | 3 | 3 | 52 | 5.50 |
| 9 | 4 | 4 | 4 | 202 | 20.83 |
| 10 | 11 | 9 | 6 | 202 | 20.00 |

Durasi unik menghitung gabungan rentang window per rekaman. Window dengan step 5 detik bertumpang tindih, sehingga jumlah window tidak sama dengan jumlah rating independen.

## Perbandingan panjang window

| Panjang | Window layak | Episode layak | Peserta pada rating 8 |
|---:|---:|---:|---:|
| 15 detik | 1693 | 126 | 3 |
| 25 detik | 1328 | 100 | 2 |
| 45 detik | 792 | 64 | 2 |

Pada window 15 detik, rating 8 hanya berasal dari 3 episode dan 3 peserta. Nilai tersebut lebih relevan untuk merancang evaluasi daripada jumlah 52 windownya. Varian 25/45 detik hanya dihitung sebagai perbandingan cakupan. Bundle ini tidak menerbitkan CSV sensor untuk kedua varian tersebut.

## Penggunaan untuk pemodelan

Gunakan `good_acceptable.csv` dan daftar sepuluh fitur core pada `feature_columns.json`. Identitas, waktu, soal, difficulty protokol, durasi, kualitas, dan outcome berfungsi sebagai metadata. Rancang split dan bobot dari v4; jangan menyalinnya dari v3. Pertahankan kelompok peserta dan episode saat memeriksa kebocoran antarbagian. Laporan ini tidak menetapkan split, tidak menjalankan training, dan tidak menyatakan bahwa sepuluh kelas telah memadai secara statistik.
