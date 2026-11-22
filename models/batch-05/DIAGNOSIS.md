# Diagnosis dan Penutupan Batch 05

Tanggal pemeriksaan: 22 November 2026.

## Kesimpulan

Batch 05 selesai, termasuk audit penutup. Model terkunci tetap Extra Trees dengan normalisasi session-robust: accuracy OOF **47,03%**, validation **44,83%**, dan Test referensi **37,92%**. Macro-F1 masing-masing 46,86%, 43,90%, dan 37,59%. Hasil tersebut belum memenuhi sasaran 70%.

Audit tidak menemukan kesalahan hitung pada enam window yang ditelusuri ke sensor mentah. Penghapusan satu fitur iris yang terbukti berulang juga tidak meningkatkan hasil secara berarti. Temuan terkuat adalah gap fitting-held-out yang besar, variasi dasar antarsesi, dan perubahan fitur antarkelas yang tidak searah pada seluruh sesi. Kita belum mempunyai pengukuran beban mental independen yang dapat menjelaskan ketidakcocokan pada setiap window.

Kesimpulan sebelumnya bahwa setelah cleaning masalahnya tinggal pipeline terlalu kuat. Cleaning memeriksa kelayakan teknis. Cleaning tidak membuktikan bahwa sensor yang direkam menyediakan informasi yang cukup untuk membedakan label protokol pada interval baru.

## 1. Cakupan dan perlindungan hasil

Audit memakai tiga `good_acceptable.csv` yang sama dengan notebook, tanpa mengganti nilai, label, proporsi, atau baris. Audit sumber mencakup gabungan **23 rekaman kanonis dari 22 orang** yang menyumbang setidaknya satu window pada ketiga CSV. Window 25 detik sendiri berisi **22 rekaman dari 21 orang**; perbedaan jumlah ini berasal dari cakupan sumber antarwindow. Ramadhan mempunyai dua rekaman, sedangkan ihsan dan ikhsan merupakan dua orang berbeda.

Daftar sumber mengikuti `selected_session_ids`/`selected_folders` pada manifest fusion, lalu dibatasi oleh sumber yang muncul pada CSV. Salinan Baithar tidak dihitung sebagai peserta tambahan. Sesi `bangsur_easy_only` yang tidak mempunyai ledger aktivitas tetap berada di luar cakupan dataset.

Pemeriksaan hash melindungi **1.495 file**, termasuk notebook, tiga CSV utama, manifest fusion, sumber kanonis, dan artefak resmi tahap 1–3. Seluruh hash sebelum/sesudah sama. Audit penutup menjalankan tepat **10 fitting diagnostik Extra Trees**, tanpa evaluasi validation/Test baru, tanpa fitting final, dan tanpa mengganti model terkunci. Uji otomatis memakai fixture kecil terpisah dari data penelitian.

[Hasil audit](outputs/closing_audit/audit_result.json), [pemeriksaan tambahan](outputs/closing_audit/supplement_checks.json), dan [hash sebelum](outputs/closing_audit/protected_hashes_before.json)/[sesudah](outputs/closing_audit/protected_hashes_after.json) menyimpan bukti pelaksanaannya.

## 2. Ketidakseimbangan jumlah kelas bukan penjelasan yang cukup

| Window | Low | Medium | High | Total |
|---|---:|---:|---:|---:|
| 15 detik | 1.687 | 1.763 | 1.803 | 5.253 |
| 25 detik | 1.549 | 1.487 | 1.572 | 4.608 |
| 45 detik | 1.394 | 966 | 1.149 | 3.509 |

Train eksperimen 25 detik berisi 924 Low, 875 Medium, dan 925 High. Validation dan Test juga mencakup tiga kelas dengan jumlah berdekatan. Kesalahan Test tersebar pada Low (158), Medium (142), dan High (124). Penambahan Low saja tidak menjelaskan atau menyelesaikan seluruh kesalahan tersebut.

Jumlah window juga tidak sama dengan jumlah rekaman independen. Pada 25 detik, Low berasal dari 21 sesi, Medium 22, dan High 17. Jumlah sel sesi-blok-kelas masing-masing 80, 78, dan 78. Satu blok dapat memuat beberapa kelas dan window tetap saling overlap di dalam bagian evaluasinya; jumlah tersebut bukan ukuran sampel independen untuk uji statistik.

[Jumlah kelas dan sumber](outputs/closing_audit/class_source_counts.csv) memisahkan jumlah baris, orang, sesi, dan blok. Pembagian/guard antarfitting-held-out tetap mengikuti manifest, bukan pengacakan ulang baris untuk menaikkan skor.

## 3. Bukti tugas: level kesulitan memang berbeda dalam ketepatan jawaban

Ledger kanonis memuat **650 attempt**. Setiap attempt dapat ditautkan ke kunjungan soal terakhir yang belum terputus, pada episode set dan puzzle yang sama. Seluruh outcome yang tersedia adalah `correct` atau `incorrect`.

| Difficulty | Soal yang disubmit per episode | Benar pada attempt pertama | Benar seluruh attempt | Attempt ulang | Median durasi kunjungan terakhir sebelum submit |
|---|---:|---:|---:|---:|---:|
| Practice | 4 | 1/4 (25,00%) | 2/5 (40,00%) | 1 | 11,50 detik |
| Easy | 143 | 84/143 (58,74%) | 125/237 (52,74%) | 94 | 23,99 detik |
| Medium | 123 | 37/123 (30,08%) | 73/226 (32,30%) | 103 | 45,03 detik |
| Hard | 54 | 2/54 (3,70%) | 16/182 (8,79%) | 128 | 52,72 detik |

“Soal per episode” menghitung satu puzzle dalam satu sesi set sebagai satu unit; ini bukan jumlah judul unik di bank soal. Durasi di tabel hanya mencakup `puzzle_started` terakhir hingga `attempt_submitted`, tidak menjumlahkan kunjungan sebelumnya, waktu feedback, atau rating. Navigasi meninggalkan soal dicatat terpisah pada ledger kunjungan, bukan dipaksakan menjadi durasi attempt.

![Ketepatan jawaban berdasarkan ledger](outputs/closing_audit/task_performance.png)

Ketepatan easy lebih tinggi daripada medium dan hard; hard juga mempunyai banyak pengulangan. Hasil ini memberi bukti perilaku bahwa penugasan memang berbeda tingkat keberhasilannya. Ketepatan jawaban tetap merupakan hasil grading CodeArrange, bukan pengukuran langsung beban mental. Audit tidak mengklasifikasikan attempt gagal sebagai bukti sensor rusak atau mengganti label berdasarkan jawaban benar/salah.

Urutan set pada sebagian besar rekaman adalah easy → medium → hard. Bangsur menjalani medium → easy; beberapa rekaman tidak mencakup seluruh level. Karena urutan dan difficulty tidak bervariasi secara terkontrol pada seluruh rekaman, kita tidak dapat memisahkan pengaruh urutan dari difficulty hanya dengan tabel ini. Satu puzzle, `reverse_array`, tercatat pada dua rekaman Ramadhan; hitungan tersebut tidak dibuat menjadi peserta baru.

Bukti rinci: [attempt tertaut](outputs/closing_audit/attempts_linked.csv), [kunjungan soal](outputs/closing_audit/puzzle_visits.csv), [rekap episode](outputs/closing_audit/performance_by_episode.csv), [urutan dan cakupan sesi](outputs/closing_audit/source_inventory.csv), serta [pengulangan puzzle](outputs/closing_audit/question_repetition.csv).

### Rating yang tersedia

Sumber kanonis memuat 167 `rating_started` dan 166 `rating_ended`. Event tersebut membuktikan fase rating, tetapi tidak memuat angka jawaban peserta. Pemeriksaan field rating/mental-effort pada CSV dan JSON sesi tidak menemukan angka rating peserta yang dapat digunakan di audit ini. Satu event akhir rating yang tidak tersedia tidak cukup untuk menentukan penyebabnya atau mengisi nilainya.

JSON sesi memuat 74 field `average_cognitive_load_score` dan 21 field hitungan window fase rating. Skor sensor heuristik dan hitungan fase tersebut bukan jawaban rating peserta, serta bukan validasi beban mental yang independen dari sensor. Audit tidak memakainya sebagai target baru.

[Inventaris CSV rating](outputs/closing_audit/numeric_rating_inventory.csv) dan [inventaris field JSON](outputs/closing_audit/json_evidence_inventory.csv) memisahkan ketiga jenis informasi tersebut. Kesimpulannya terbatas pada berkas yang tersedia di workspace, bukan pernyataan bahwa tim tidak pernah mengumpulkan kuesioner di tempat lain.

## 4. Penelusuran enam window: hitungan sensor dan target cocok

Audit memilih satu prediksi benar dan satu salah untuk setiap kelas dari **prediksi Test yang sudah tersimpan**. Confidence adalah probabilitas prediksi terbesar, bukan ukuran keyakinan yang telah dikalibrasi. Audit mengurutkan confidence menurun, memecahkan seri dengan `window_id`, dan memilih blok berbeda jika tersedia. Tidak ada inference model baru untuk memilih kasus.

Seluruh kasus berada pada 25 detik. Jam pada tabel memakai WIB; tanggal penuh dan `window_id` tersimpan pada berkas pemilihan kasus.

| Kasus | Sumber dan interval | Prediksi | Confidence | Hasil penelusuran fase |
|---|---|---|---:|---|
| Low benar | Siti, 28 Oktober 10:52:10,697–10:52:35,697 | Low | 80,29% | 20,910 detik active easy + 4,090 detik feedback; seluruhnya Low menurut protokol |
| Low salah | Mon, 17 November 14:53:00,741–14:53:25,741 | High | 67,74% | Active easy, `purchase_change` |
| Medium benar | Andreas, 30 Oktober 11:10:36,882–11:11:01,882 | Medium | 78,91% | Active medium, `count_vowels` |
| Medium salah | Siti, 28 Oktober 11:04:25,697–11:04:50,697 | High | 74,61% | Active medium, `matrix_addition` |
| High benar | Andrean, 30 Oktober 10:37:59,532–10:38:24,532 | High | 67,89% | Active hard, `anagram_check` |
| High salah | irfan, 28 Oktober 13:54:33,506–13:54:58,506 | Medium | 67,54% | Active hard, `fibonacci_iterative` |

Audit menghitung ulang EAR dan iris dari frame mentah yang memenuhi syarat, blink dari selisih counter dengan frame pendahulu, serta RMSSD/SDNN dari RR yang diterima secara kausal. Rumus agregasi ditulis terpisah dari agregator dataset. Protokol penerimaan paket RR menggunakan preprocessor sumber yang sudah diaudit; pemeriksaan ini bukan implementasi independen kedua atas seluruh filter BLE.

Ketentuan yang diperiksa:

- Sampel sensor masuk pada `(window_start, window_end]`.
- RR harus sudah tersedia pada akhir window; recovery yang belum dilepas tidak dihitung secara retroaktif.
- Pasangan RMSSD harus mempunyai `raw_id` berurutan dan `segment_id` sama.
- RMSSD dan SDNN memakai satuan milidetik; SDNN/EAR/iris menggunakan standar deviasi sampel (`ddof=1`).
- Blink rate dihitung sebagai jumlah blink × 60 / durasi detik, bukan rata-rata counter kumulatif.
- Iris memakai mode `eye_width_ratio`. Fitur ini merupakan rasio geometri landmark iris terhadap lebar mata, bukan diameter pupil atau rekaman pupil dilation.

Sebanyak **102 pemeriksaan angka cocok**, tanpa selisih yang melewati toleransi `rtol=1e-9`, `atol=1e-8`. Selisih absolut terbesar sekitar **7,11×10⁻¹⁵**. Keenam target juga cocok dengan fase/difficulty dari ledger. Kasus salah prediksi tetap mempunyai hitungan sensor dan target protokol yang cocok; kegagalan model pada contoh tersebut tidak dijelaskan oleh salah agregasi yang diperiksa.

Pemeriksaan seluruh CSV juga menemukan nol pertentangan antara target dan rincian fase yang tersimpan, serta nol window terkurasi yang menyentuh timestamp ambigu Alif. Pemeriksaan penuh ini menguji konsistensi internal metadata dan aturan eksklusi. Penelusuran independen sampai sensor mentah tetap berjumlah enam kasus, bukan seluruh dataset. Hasilnya tidak membuktikan akurasi fisiologis sensor, mengukur latensi perangkat, atau memvalidasi gaze spasial yang memang tidak digunakan pada core10.

[Kasus terpilih](outputs/closing_audit/selected_test_cases.csv), [hitung ulang sensor](outputs/closing_audit/sensor_recalculation.csv), [segmen fase](outputs/closing_audit/case_phase_segments.csv), [cek target kasus](outputs/closing_audit/case_label_checks.csv), dan [cek kebijakan seluruh CSV](outputs/closing_audit/full_dataset_policy_checks.csv) menyimpan bukti. File `*_events.csv`, `*_eye_rows.csv`, dan `*_rr_rows.csv` menyertakan ledger serta baris sensor terkait.

## 5. Variasi antarsesi lebih besar daripada asosiasi label pada ringkasan blok

Audit meringkas setiap sel sesi-blok-kelas dengan median agar deskripsi tidak sekadar memperbanyak suara window yang overlap. Berikut asosiasi marginal pada ringkasan 25 detik:

| Fitur | Eta² terhadap sesi | Eta² terhadap kelas |
|---|---:|---:|
| RMSSD | 83,66% | 0,38% |
| SDNN | 70,83% | 0,76% |
| Blink rate | 77,62% | 0,58% |
| EAR kiri mean | 88,59% | 2,65% |
| Iris ratio mean | 92,23% | 0,83% |

Eta² ini menghitung asosiasi masing-masing faktor secara terpisah. Kedua kolom tidak dijumlahkan, tidak menguraikan kontribusi kausal, dan tidak menjadi batas accuracy classifier nonlinear. Ringkasan blok juga tidak mengubah sel tersebut menjadi observasi yang bebas sepenuhnya dari ketergantungan waktu.

Perbandingan dalam sesi menunjukkan arah perubahan yang bervariasi. Dari 16 sesi dengan Low dan High pada 25 detik, median RMSSD High lebih besar pada 12 sesi dan lebih kecil pada empat sesi; median iris ratio High lebih besar pada tujuh dan lebih kecil pada sembilan. EAR kiri mean High lebih besar pada 11 sesi, lebih kecil pada lima. Audit tidak menetapkan bahwa satu arah fisiologis wajib berlaku untuk setiap orang. Angka tersebut menunjukkan bahwa pemisahan kelas tidak dapat dijelaskan oleh satu ambang arah sederhana yang sama pada seluruh sesi.

Normalisasi session-robust memang membantu OOF Extra Trees dari Macro-F1 34,11% menjadi 46,86%. Namun, model dengan normalisasi tersebut tetap memperoleh validation 43,90% dan Test 37,59%. Jadi perbedaan dasar antarsesi telah ditangani sebagai kontrol dan belum menjadi penjelasan tunggal yang terselesaikan melalui normalisasi.

Bukti: [median blok](outputs/closing_audit/block_feature_medians.csv), [asosiasi marginal](outputs/closing_audit/block_marginal_associations.csv), [kontras dalam sesi](outputs/closing_audit/within_session_contrasts.csv), dan [ringkasan arah](outputs/closing_audit/contrast_direction_summary.csv).

### Low mencakup beberapa konteks dengan distribusi sensor berbeda

| Window | Active saja | Campuran active + nonactive | Nonactive saja |
|---|---:|---:|---:|
| 15 detik | 707 | 695 | 285 |
| 25 detik | 470 | 936 | 143 |
| 45 detik | 229 | 1.121 | 44 |

Pada 25 detik, 15 sesi menyediakan kedua konteks murni untuk perbandingan median blok. EAR kiri mean pada nonactive lebih rendah daripada active pada 13 sesi, lebih tinggi pada dua. Ini menunjukkan heterogenitas konteks di dalam Low. Audit mempertahankan protokol Low yang disepakati dan tidak menyimpulkan bahwa seluruh fase nonactive salah label.

Low salah yang ditelusuri pada Mon justru berasal dari active easy penuh. Pada evaluasi tahap 2, Low active dan mixed sama-sama mempunyai kesalahan besar. Karena itu, menghapus seluruh nonactive saja belum mempunyai bukti akan menyelesaikan masalah kelas Low.

[Jumlah konteks Low](outputs/closing_audit/low_context_counts.csv) dan [median blok menurut konteks](outputs/closing_audit/low_context_block_medians.csv) tersedia untuk audit lanjutan, tanpa menjadikan fase/identitas sebagai fitur classifier.

## 6. Redundansi iris terbukti, tetapi penghapusannya tidak menyelesaikan skor rendah

Syarat kontrol diperiksa **hanya pada fitting setiap fold**, sebelum dan sesudah preprocessing. Pasangan mean iris ratio dan mean delta harus sama-sama bervariasi, mempunyai setidaknya 20 window, serta memenuhi hubungan affine dengan R² ≥ 1 − 10⁻¹⁰.

Sebanyak **86 dari 106 sel sesi-fold yang layak** memenuhi syarat pada ruang mentah dan kedua normalisasi. Sel yang memenuhi syarat melibatkan 21 sesi; angka ini tidak berarti seluruh fold pada 21 sesi tersebut selalu redundan. Hasil memenuhi gate minimal separuh sel dan minimal tiga sesi.

Kontrol menghapus `iris_size_delta_mean` saja. Extra Trees tetap memakai 200 trees, `min_samples_leaf=5`, seed 42, dua thread, serta lima fold Train yang sama. `max_features=8` mempertahankan delapan fitur kandidat per split sebagaimana baseline core10 dengan `max_features=0.8`. Penghapusan fitur tetap dapat mengubah lintasan acak pohon; kontrol ini mengukur akibat penghapusan pada konfigurasi tetap, bukan dekomposisi kausal yang persis.

| Normalisasi | Core10 OOF accuracy | Core9 OOF accuracy | Core10 Macro-F1 | Core9 Macro-F1 | Selisih Macro-F1 |
|---|---:|---:|---:|---:|---:|
| Global | 34,03% | 34,18% | 34,11% | 34,30% | +0,19 poin persentase |
| Session-robust | 47,03% | 46,44% | 46,86% | 46,29% | −0,57 poin persentase |

![Kontrol penghapusan fitur berulang](outputs/closing_audit/redundancy_control.png)

Pada session-robust, Macro-F1 naik pada dua fold dan turun pada tiga fold. Penghapusan fitur tidak memberi perbaikan yang stabil, bahkan OOF gabungannya turun. Temuan redundansi nyata, tetapi hasil kontrol tidak mendukungnya sebagai penyebab utama yang dapat menjelaskan selisih menuju 70%.

Kontrol ini tidak membaca validation/Test dan tidak mengubah pemenang. [Gate redundansi](outputs/closing_audit/redundancy_gate.json), [bukti sesi-fold](outputs/closing_audit/iris_redundancy_by_fit.csv), [metrik/recall fold](outputs/closing_audit/core9_fold_metrics.csv), [selisih berpasangan](outputs/closing_audit/core9_paired_differences.csv), dan [prediksi OOF](outputs/closing_audit/core9_oof_predictions.csv) menyimpan hasilnya. Pemeriksaan terpisah mencocokkan 12 tabel metrik dari prediksi tersebut, tanpa selisih.

## 7. Gap fitting-held-out tetap ada setelah kontrol pipeline

Learning curve Extra Trees tahap 2 menunjukkan rata-rata accuracy fitting sekitar **95,10%**, dibandingkan held-out fold **47,51%** pada pool fitting penuh. Macro-F1 masing-masing **95,08%** dan **47,26%**. Angka ini merupakan rata-rata lima fold, berbeda dari OOF gabungan 47,03%/46,86% karena ukuran fold tidak sama.

Menambah blok fitting dari sekitar 25% hingga 100% menaikkan rata-rata Macro-F1 held-out dari 33,57% menjadi 47,26%, tetapi tidak menghilangkan gap. Hasil tersebut menunjukkan manfaat tambahan blok pada eksperimen ini. Kita tidak dapat mengekstrapolasikannya menjadi jumlah peserta/menit tertentu yang dijamin mencapai 70%.

Pada neural, loss fitting menurun sementara loss monitor kemudian meningkat. Seluruh 120 epoch tetap selesai. Monitor sudah memakai pemisahan dan komposisi kelas yang diperiksa; tiga aturan checkpoint berasal dari lintasan yang sama. MLP terbaik belum melampaui baseline OOF dan hasil validation-nya belum stabil. Jadi berhenti lebih awal, memaksakan epoch terakhir, atau mengganti ke satu arsitektur neural yang diuji tidak menyelesaikan gap tersebut.

[Learning curve](outputs/exp01/stage2_20261122T104602264199Z_f8aacc8829/learning_curve.csv) dan [hasil tahap 3](outputs/exp01/stage3_20261122T115353575899Z_933c7b4bcb/RESULTS.md) mendokumentasikan perbandingan ini. Pada Test, model juga masih rendah pada 648 window yang sumber sesi-kelasnya tersedia pada Train: accuracy **38,73%**. Kekurangan cakupan sumber bukan satu-satunya masalah.

## 8. Yang sudah diketahui dan yang belum dapat ditentukan

| Pertanyaan | Bukti dan keputusan |
|---|---|
| Apakah Low yang sedikit merupakan satu-satunya penyebab? | Tidak didukung: Train hampir seimbang, dan kesalahan besar ada pada ketiga kelas. |
| Apakah hitungan fitur pada contoh salah prediksi rusak? | Tidak ditemukan pada enam kasus/102 angka yang ditelusuri; jangan digeneralisasi menjadi validasi seluruh pengukuran. |
| Apakah menghapus satu fitur iris berulang menyelesaikan masalah? | Tidak pada kontrol yang dijalankan; OOF berubah kecil dan tidak konsisten. |
| Apakah easy/hard tidak berbeda sama sekali? | Ledger menunjukkan ketepatan jawaban berbeda, tetapi ketepatan bukan pengukuran langsung beban mental. |
| Apakah normalisasi, monitor, atau checkpoint belum pernah ditangani? | Sudah diuji dalam tahap 1–3. Manfaat OOF belum bertahan menjadi performa held-out yang memadai. |
| Apakah seluruh sensor buruk atau label tertentu pasti salah? | Bukti audit tidak cukup untuk keputusan tersebut. Kita tidak mempunyai pengukuran independen per window untuk menentukannya. |
| Apakah 70% mustahil? | Belum terbukti. Audit juga tidak memberi dasar untuk menjanjikan 70% melalui search berikutnya pada data yang sama. |

## 9. Keputusan penutupan

Rangkaian batch ini ditutup dengan model dan hasil resmi yang sama. Tidak perlu menjalankan notebook kembali untuk menyelesaikan audit ini. Tim dapat memakai [SUMMARY.md](SUMMARY.md) untuk rekap eksperimen dan dokumen ini untuk diagnosis beserta batas bukti.

Tidak ada dasar dari audit penutup untuk langsung membuka pencarian model besar lagi. Jika tim melanjutkan penelitian, kebutuhan yang belum tersedia adalah bukti independen tentang beban mental pada tugas dan pengukuran sensor yang sesuai dengan fenomena yang ditargetkan. Rating kesulitan yang ditemukan di arsip lain dapat diaudit berdasarkan instrumen/provenance-nya; jangan mengisinya dari skor sensor. Pengumpulan atau perubahan protokol baru memerlukan keputusan penelitian tersendiri.

Target 70% tetap merupakan sasaran, bukan hasil yang sudah dekat atau dijamin. Mengubah split supaya window bertetangga bocor ke fitting, mengubah label mengikuti prediksi, atau memilih model kembali berdasarkan Test tidak menyelesaikan masalah yang diperiksa di sini.

## Bukti pelaksanaan

Sebanyak **43 uji otomatis lolos**, termasuk 10 uji audit penutup. Pemeriksaan syntax, lint, tipe, dan keamanan kode audit juga lolos. Grafik telah diperiksa secara visual. [Kode audit](outputs/closing_audit/audit.py) disimpan bersama bukti; kode tersebut terpisah dari notebook dan bukan persyaratan menjalankan Colab. Runner menolak mengulang paket fitting yang sudah selesai; mode `--supplement` hanya memeriksa bukti tersimpan tanpa training.
