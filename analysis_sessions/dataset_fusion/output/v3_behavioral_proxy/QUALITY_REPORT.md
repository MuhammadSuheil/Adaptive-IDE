# Dataset v3: label eksploratif berbasis perilaku

## Tujuan dan arti label

Low, Medium, dan High pada v3 adalah kategori perkiraan dari kesesuaian waktu pengerjaan aktif dan pola kesalahan manual. Ini bukan pengukuran cognitive load yang sudah tervalidasi. V3 menguji sumber label alternatif tanpa menetapkan easy=Low, medium=Medium, atau hard=High. Skor model terhadap label ini mengukur kemampuan memprediksi proxy perilaku, bukan membuktikan akurasi membaca keadaan mental.

Tidak ditemukan angka rating asli yang cocok dengan sesi aktif pada arsip proyek yang diperiksa. Penanda rating_started/rating_ended tidak menyimpan angka rating. Nilai yang hilang tidak diisi; timeout tidak diubah menjadi rating 10. Sensor, cognitive_load_score heuristik, difficulty, dan hasil classifier tidak dipakai untuk menentukan label.

## Rekonstruksi episode

Satu episode mengikuti satu soal dalam satu sesi set hingga jawaban pertama yang benar atau penutupan tanpa selesai. Coba Lagi dan kembali ke soal yang belum selesai menambahkan interval aktif ke episode yang sama. Membuka kembali soal yang sudah benar membuat episode baru. Waktu feedback, rating, istirahat, dan pengerjaan soal lain tidak dihitung. time_seconds kumulatif pada log bukan durasi soal.

Kesalahan dihitung dari submission manual incorrect yang unik menurut attempt_id. Timeout dicatat terpisah. Durasi episode tidak selesai adalah batas bawah observasi, bukan waktu penyelesaian. Trace tidak lengkap atau bertentangan tidak dilengkapi dengan event/timestamp buatan.

## Pembagian dan referensi waktu

Kandidat awal memakai pembagian hash lama; kandidat berikutnya memakai shuffle deterministik dengan seed 20261124, maksimal 1.000 kandidat termasuk pembagian awal. Validation dan Test masing-masing max(1, round(0.15 × jumlah peserta)); sisanya Train. Semua rekaman satu orang berada pada bagian yang sama. Referensi dan label dihitung ulang dari Train pada setiap kandidat. Pemilihan hanya memeriksa cakupan kelas dan peserta, tanpa training atau skor classifier. Kandidat pertama pada prioritas terbaik dipakai, lalu pembagian dibekukan. Prioritas A memenuhi minimal 3 orang per kelas Train dan 2 pada validation/Test untuk semua window; B memenuhi syarat itu pada 15 detik dan kehadiran semua kelas pada window lain; C memenuhi syarat 15 detik dan kehadiran kelas pada 25 detik; D memenuhi syarat 15 detik saja. Jika semuanya gagal, pembagian awal dipertahankan dengan status unavailable. Jumlah peserta minimum ini adalah pemeriksaan teknis, bukan jaminan kecukupan statistik.

| Bagian | Orang |
|---|---:|
| Train | 16 |
| validation | 3 |
| Test | 3 |

Referensi hanya memakai paparan pertama pada soal yang sama dari peserta Train, bila trace valid dan berhasil diselesaikan secara manual. Paparan pertama yang gagal tidak diganti dengan keberhasilan pada paparan berikutnya. Paparan sebelumnya yang sudah diketahui dalam metadata juga dikeluarkan dari kontributor referensi. Identitas dasar pembanding adalah set, ID soal, dan judul setelah normalisasi nomor urut, kapitalisasi, Unicode NFKC, serta spasi. Perbedaan isi soal yang tidak terekam tidak dapat diperiksa ulang hanya dari judul. Pengecualian adalah tujuh soal dengan bukti seluruh JSON identik saat berpindah easy-2 ke easy-1; keduanya memakai identitas kanonis dari hash isi. Bukti tersimpan pada item_aliases.json, termasuk snapshot, commit, lokasi, dan hash. Kesamaan isi bukan bukti kesamaan config runtime: batas waktu easy pada config historis berubah setelah perluasan bank soal. Config yang benar-benar dijalankan pada setiap rekaman tidak tersedia, sehingga tekanan waktu tidak dapat dipastikan hanya dari Git atau durasi pengerjaan.

Setiap orang menyumbang maksimal satu waktu per soal. Peserta yang sedang dilabeli dikeluarkan dari referensinya sendiri. Minimal lima orang pembanding diperlukan. Batas waktu adalah persentil 33⅓ dan 66⅔ dengan interpolasi linear. Referensi kurang atau kedua batas sama menyebabkan abstention.

## Rubrik yang dibekukan

| Label | Bukti hasil pengerjaan | Bukti waktu aktif |
|---|---|---|
| Low | Submission manual pertama benar, tanpa kesalahan manual | Waktu ≤ batas bawah |
| Medium | Berhasil setelah 1–2 kesalahan manual | Batas bawah < waktu ≤ batas atas |
| High | Minimal 3 kesalahan manual; boleh belum selesai | Waktu teramati > batas atas |
| Tidak berlabel | Bukti lemah, bertentangan, trace invalid, atau referensi kurang | Tidak dipaksa menjadi Medium |

Rubrik adalah definisi operasional eksperimen, bukan ambang psikologis baku. Waktu dan kesalahan tidak otomatis menunjukkan usaha mental: hasil juga dapat dipengaruhi keterampilan, strategi, motivasi, atau penilaian jawaban. Arsip tidak menyimpan seluruh urutan jawaban beserta versi grader historis untuk mengaudit ulang setiap keputusan benar/salah. Tidak dibuat probabilitas keyakinan yang tidak diukur dan tidak dilakukan tuning rubrik berdasarkan akurasi model.

## Cakupan episode

Total 358 episode; 357 mempunyai trace valid. 82 episode diberi label; 276 tetap tidak berlabel.

| Kategori | Episode |
|---|---:|
| Low | 43 |
| Medium | 17 |
| High | 22 |
| unlabeled | 276 |

| Alasan keputusan episode | Episode |
|---|---:|
| concordant_behavioral_evidence | 82 |
| conflicting_or_insufficient_behavior | 112 |
| insufficient_reference | 163 |
| invalid_event_trace | 1 |

| Bagian | Low | Medium | High | Tidak berlabel |
|---|---:|---:|---:|---:|
| Train | 20 | 10 | 16 | 199 |
| validation | 5 | 3 | 4 | 33 |
| Test | 18 | 4 | 2 | 44 |

## Window dan kualitas sensor

Kandidat berasal dari all_data v2 yang terverifikasi, bukan hanya hasil seleksi keseimbangan v2. Window 15/25/45 detik dengan step 5 detik harus seluruhnya berada dalam satu interval aktif yang kontinu dari satu episode berlabel. Interval retry yang terpisah feedback tidak disambung. Pre-task, feedback, rating, rest, post-task, kalibrasi, dan window lintas episode tidak dijadikan tambahan Low. Target episode sama pada ketiga panjang window.

Rumus sensor, ambang kualitas core, pengecualian RR lokal, dan penanganan timestamp ambigu mengikuti bundle v2 sumber. Nilai sensor tidak dikoreksi atau diimputasi. Gaze yang gagal validasi/kualitas tetap missing di CSV curated. Input default tetap sepuluh fitur core, bukan seluruh kolom numerik.

| Window | Dataset | Low | Medium | High | Total | Orang | Durasi unik (menit) |
|---|---|---:|---:|---:|---:|---:|---:|
| 15s | good_only | 111 | 184 | 516 | 811 | 18 | 94.75 |
| 15s | good_acceptable | 130 | 237 | 826 | 1193 | 21 | 129.58 |
| 15s | all_data | 174 | 256 | 1003 | 1433 | 21 | 150.75 |
| 25s | good_only | 75 | 143 | 410 | 628 | 16 | 86.92 |
| 25s | good_acceptable | 82 | 178 | 643 | 903 | 20 | 121.17 |
| 25s | all_data | 109 | 187 | 773 | 1069 | 21 | 141.08 |
| 45s | good_only | 35 | 89 | 227 | 351 | 14 | 64.33 |
| 45s | good_acceptable | 36 | 99 | 340 | 475 | 18 | 87.08 |
| 45s | all_data | 48 | 99 | 399 | 546 | 19 | 100.17 |

all_data menyimpan semua window berlabel termasuk kualitas poor untuk audit. good_acceptable hanya memakai good/acceptable; good_only adalah subset good. Ketiganya tidak dipaksa seimbang. Penyeimbangan disiapkan sebagai dua alternatif khusus Train, tanpa mengganti label. Jumlah baris overlap bukan jumlah pengamatan independen. Ketiga panjang window adalah alternatif dan tidak ditumpuk.

Rubrik High mensyaratkan waktu yang relatif lama. Episode panjang juga menghasilkan lebih banyak sliding window; akibatnya proporsi High pada CSV dapat jauh lebih besar daripada proporsi episode High. Ini adalah pengaruh durasi dan definisi label, bukan bukti bahwa mayoritas peserta mengalami CL High. Periksa jumlah episode dan durasi unik, bukan hanya jumlah baris. Pada pemodelan nanti, kontribusi episode panjang dan kelas mayoritas perlu ditangani di Train serta dievaluasi per episode.

## Ketersediaan kelas untuk evaluasi

| Window | Tiga kelas pada dataset utama | Kelas hilang Train | Kelas hilang validation | Kelas hilang Test |
|---|---|---|---|---|
| 15s | Ada | Tidak ada | Tidak ada | Tidak ada |
| 25s | Ada | Tidak ada | Tidak ada | Tidak ada |
| 45s | Ada | Tidak ada | Tidak ada | Tidak ada |

Jika dataset utama kehilangan suatu kelas, varian itu belum layak untuk eksperimen tiga kelas. Jika validation/Test kehilangan suatu kelas, evaluasi lengkap ketiga kelas pada bagian tersebut tidak tersedia meskipun seluruh CSV secara gabungan mempunyai tiga kelas. Pembagian yang terpilih dari pencarian cakupan dibekukan sebelum pemodelan. Rubrik dan minimum referensi tidak dilonggarkan. 15 detik adalah varian utama; 25/45 detik menjadi analisis sekunder sesuai dukungan peserta. Status tersimpan di manifest.

## Alasan window tidak digunakan

| Window | Alasan | Window |
|---|---|---:|
| 15s | conflicting_or_insufficient_behavior | 1359 |
| 15s | insufficient_reference | 2242 |
| 15s | invalid_event_trace | 6 |
| 15s | not_single_active_episode | 3192 |
| 25s | conflicting_or_insufficient_behavior | 1073 |
| 25s | insufficient_reference | 1785 |
| 25s | invalid_event_trace | 4 |
| 25s | not_single_active_episode | 4148 |
| 45s | conflicting_or_insufficient_behavior | 640 |
| 45s | insufficient_reference | 1223 |
| 45s | not_single_active_episode | 5384 |

excluded_windows.csv merinci window yang tersedia di pool v2 tetapi tidak lolos label/interval v3. Eksklusi awal v2—misalnya timestamp ambigu—tetap berlaku dan diringkas di source_pool_exclusions pada manifest. Builder berhenti jika menemukan window yang berpotensi valid v3 tetapi tidak mempunyai agregat sensor di pool sumber; tidak menciptakan nilai pengganti.

## Penggunaan dan pembaruan

episode_labels.csv menyimpan durasi, interval sumber, attempt, kesalahan, censoring, batas referensi, split, dan alasan keputusan. label_reference.json menyimpan kontributor Train. participant_splits.csv merupakan pembagian yang wajib dipakai. window_assignments.csv menghubungkan baris sensor ke episode dan split; source_v2_label hanya untuk audit, bukan target/predictor baru.

Jangan sekadar mengganti path pada notebook lama yang membuat pembagian sendiri. Mengubah peserta Train memerlukan penghitungan ulang referensi dan label. Jika melakukan cross-validation di dalam Train dan memakai hasilnya untuk memilih model, hitung ulang referensi pada inner-Train; validation/Test akhir tetap tidak boleh menjadi sumber referensi. Statistik preprocessing dan balancing juga fit di Train.

V3 belum memiliki validasi beban mental independen. Konsistensi rubrik, kelengkapan fitur, dan skor classifier tidak membuktikan bahwa proxy ini mengukur CL sebenarnya. Perbandingan dengan rating atau pengukuran independen memerlukan sumber baru yang benar-benar tercatat.

Build: `python -m analysis_sessions.dataset_fusion.build_dataset_v3`

Rebuild setelah sumber v2 diperbarui: `python -m analysis_sessions.dataset_fusion.build_dataset_v3 --replace`

Verifikasi: `python -m analysis_sessions.dataset_fusion.build_dataset_v3 --verify analysis_sessions/dataset_fusion/output/v3_behavioral_proxy`

Sumber dan v2 tetap read-only. Manifest mencatat hash, versi kebijakan, pembagian, jumlah kelas, serta status evaluasi. Keluaran dipublikasikan sebagai satu bundle dengan rollback.

## Rujukan

- [Validitas dan perbedaan rating kesulitan serta usaha mental](https://link.springer.com/article/10.1007/s11251-024-09692-6).
- [Weak supervision sebagai sumber label yang tidak sempurna](https://cs.brown.edu/people/sbach/files/ratner-vldb17.pdf). Rubrik v3 bukan hasil validasi dari paper tersebut dan tidak memakai model label Snorkel.


## Audit perbaikan referensi

| Soal identik yang berpindah set | Bukti |
|---|---|
| simple_ifelse | Snapshot JSON dan hash identik |
| string_concat | Snapshot JSON dan hash identik |
| basic_while_loop | Snapshot JSON dan hash identik |
| basic_for_loop | Snapshot JSON dan hash identik |
| array_sum | Snapshot JSON dan hash identik |
| factorial_iterative | Snapshot JSON dan hash identik |
| array_copy_manual | Snapshot JSON dan hash identik |

Pengulangan easy pada bangsur_easy_medium dan medium pada Ramadhan-2 tidak menjadi kontributor referensi. Pengecualian berlaku konservatif untuk tingkat yang tercatat, karena detail soal paparan sebelumnya tidak lengkap. Episode sensornya tetap dapat berlabel bila memenuhi syarat. Kelayakan referensi tidak dimasukkan sebagai fitur classifier.

26 episode mempunyai paparan sebelumnya yang diketahui. Referensi akhir memakai 141 kontributor episode Train, dengan 0 kontributor pengulangan yang diketahui.

| Alasan tidak menjadi calon referensi | Episode |
|---|---:|
| invalid_event_trace | 1 |
| known_prior_exposure | 26 |
| later_observed_exposure | 15 |
| not_completed_manual | 124 |

| Rekaman | Soal | Trace yang dikeluarkan |
|---|---|---|
| Ramadhan | reverse_array | missing_episode_end |

## Dukungan independen per bagian

Kandidat terpilih berindeks 178 dengan prioritas B. partition_search.json merekam kandidat dan hash pembagian.

| Window | Bagian | Low: baris/orang/episode | Medium: baris/orang/episode | High: baris/orang/episode |
|---|---|---|---|---|
| 15s | Train | 54/7/12 | 134/6/9 | 556/10/16 |
| 15s | validation | 14/3/4 | 53/2/3 | 151/2/4 |
| 15s | Test | 62/3/12 | 50/2/4 | 119/2/2 |
| 25s | Train | 36/3/6 | 100/6/8 | 428/10/16 |
| 25s | validation | 7/1/1 | 43/2/3 | 115/2/4 |
| 25s | Test | 39/3/9 | 35/2/3 | 100/2/2 |
| 45s | Train | 20/1/3 | 55/4/5 | 211/10/15 |
| 45s | validation | 3/1/1 | 27/1/1 | 60/2/4 |
| 45s | Test | 13/2/3 | 17/2/3 | 69/2/2 |

| Window | Minimum dukungan orang terpenuhi | Status |
|---|---|---|
| 15s | True | supported_for_exploratory_evaluation |
| 25s | False | limited_support_secondary_only |
| 45s | False | limited_support_secondary_only |

## Penanganan dominasi pada Train

CSV utama mempertahankan seluruh window layak. Validation/Test tidak dipangkas, diulang, atau dibuat sintetis. Dua perlakuan berikut adalah alternatif:

1. Full Train dari good_acceptable.csv dengan train_sample_weights.csv. Untuk N baris Train, P peserta pada kelas, E episode peserta tersebut pada kelas, dan W window episode: bobot = N / (3 * P * E * W). Setiap kelas mendapat total bobot N/3, lalu kontribusi disetarakan antarorang dalam kelas dan antarepisode orang tersebut. Tidak ada data sintetis.
2. train_balanced.csv. Jumlah tiap kelas mengikuti kelas terkecil pada Train. Kuota dibagi merata bergiliran menurut peserta dan episode, dibatasi ketersediaan. Dalam episode, good didahulukan lalu acceptable, dengan pemilihan tersebar menurut waktu dan tie-break ID.

Bobot tidak termasuk input classifier. Jangan mengalikan bobot full-pool ke Train seimbang. Jika memakai inner cross-validation, hitung ulang referensi, label, bobot, dan seleksi dari inner-Train; berkas bobot ini hanya untuk Train akhir pada pembagian yang dibekukan.

| Window | Train lengkap Low/Medium/High | Train seimbang per kelas | Total seimbang |
|---|---|---:|---:|
| 15s | 54/134/556 | 54 | 162 |
| 25s | 36/100/428 | 36 | 108 |
| 45s | 20/55/211 | 20 | 60 |

## Sensitivitas referensi

Setiap kontributor referensi dikeluarkan satu per satu dan keputusan episode dihitung ulang. Hasil tidak dipakai untuk mengubah label utama atau membuat probabilitas keyakinan.

- 50 episode tetap berlabel sama pada seluruh pengeluaran satu kontributor.
- 32 episode kehilangan label pada setidaknya satu pengeluaran kontributor.
- 0 episode berubah ke kelas lain pada setidaknya satu pengeluaran kontributor.

Kategori kehilangan label dan perubahan kelas dapat beririsan. Kehilangan label dapat berasal dari referensi yang turun di bawah lima orang atau bukti waktu yang tidak lagi selaras. Rincian tersedia pada reference_sensitivity.csv; minimum referensi tidak diturunkan.

## Penutupan audit dan kebutuhan sumber tambahan

revision_changes.csv menelusuri perubahan seluruh episode bila bundle lama diganti. history menyimpan manifest, pembagian, label, serta referensi sebelumnya tanpa menggandakan CSV sensor.

| Bagian | Penanganan |
|---|---|
| Soal identik yang pindah set | Identitas diperbaiki dengan snapshot dan hash |
| Paparan ulang dalam referensi | Dikeluarkan sebagai kontributor; episode layak dipertahankan |
| Kelas hilang pada evaluasi | Pencarian cakupan terbatas dan status dukungan tiap window |
| Dominasi kelas dan episode panjang | Bobot hierarkis atau alternatif Train seimbang |
| Trace tidak lengkap | Dikeluarkan tanpa membuat event akhir |
| Referensi kurang atau perilaku bertentangan | Tetap tidak berlabel |
| Rating beban mental independen | Tidak tersedia pada arsip yang diperiksa |
| Config deployed dan jawaban historis lengkap | Tidak dipastikan dari Git/ledger saja |

Lolos pemeriksaan teknis berarti siap diuji sebagai proxy perilaku dengan dukungan yang dilaporkan, bukan jaminan validitas CL sebenarnya atau accuracy tertentu. Kegunaan fitur tetap memerlukan pengujian pemodelan. Perubahan rumus tanpa bukti sumber baru tidak mengisi informasi yang memang tidak direkam.

Pengambilan tambahan, bila dilakukan, perlu memprioritaskan paparan pertama soal yang sebanding, kualitas sensor dan interval aktif yang cukup, serta rating usaha mental yang benar-benar dicatat dekat dengan pengerjaan termasuk ketika salah atau timeout. Simpan versi soal, config runtime, urutan jawaban, dan identitas lintas sesi. Kebutuhan ditentukan dari peserta/episode per kelas dan referensi per soal, bukan sekadar target tambahan baris Low.
