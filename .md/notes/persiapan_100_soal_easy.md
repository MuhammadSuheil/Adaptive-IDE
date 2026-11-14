# Bank 100 Soal Easy dan Persiapan Pilot

## Status

Bank tersedia pada easy-1 sampai easy-4, masing-masing 25 soal. Totalnya 100 soal berbeda: 13 soal lama yang unik dan 87 soal baru. Satu salinan Menyalin Array Secara Manual yang identik telah digabungkan; soal tersebut tetap ada di easy-1.

Setiap set memuat 17 soal arrange, 4 output, dan 4 debug. Keempat set ditujukan pada tingkat easy yang setara. Nomor set bukan jenjang kesulitan.

Kode Java dan seluruh solusi 13 soal lama dipertahankan. Perubahan pada soal lama terbatas pada lokasi/nomor file, nomor yang tertanam pada judul, dan metadata mode. Soal Cetak Array 2D kini memakai mode output; soalnya tetap menyusun output dari program lama. Debug Penjumlahan Bilangan Genap tetap mempunyai dua bug sebagai pengecualian lama.

## Menjalankan sesi

Jalankan server sekali dari folder codearrange-web:

```powershell
node server.js "Nama Peserta" easy-1 --review
```

Hubungkan ID sensor jika dipakai untuk penelitian, lalu pilih set pada menu awal. Setelah satu set selesai, kembali ke menu dan pilih set berikutnya tanpa restart server. Ikuti [panduan integrasi](codearrange_integration_guidelines.md) untuk urutan menjalankan sensor dan menghubungkan ID sesi.

- Target pelaksanaan adalah keempat set per peserta.
- Keempat easy tidak mempunyai batas waktu pengerjaan dan UI timer disembunyikan. Tidak ada waktu tunggu minimum atau penundaan tombol submit.
- Beri istirahat setelah setiap set; gunakan 5 menit sebagai panduan awal dan boleh lebih lama sesuai kebutuhan.
- Jika peserta tidak nyaman atau harus berhenti, gunakan Akhiri set dan catat alasan serta set/soal terakhir. Jangan menganggap soal yang belum dikerjakan sebagai data tersedia.
- Akhiri set hanya menutup set aktif; Akhiri sesi penelitian menutup rangkaian penelitian. Penghentian server dan sensor tetap melalui terminal.

Konfigurasi easy yang sudah disiapkan operator dipertahankan: difficultyLevel easy, showTimerUI off, dan tidak ada timeLimitMinutes. Pada server saat ini, nilai yang tidak diisi maupun nol berarti tidak ada batas waktu. Batas demo, medium, dan hard tidak diubah.

## Batas materi

Soal baru memakai aritmetika dasar, variabel, boolean, percabangan sederhana, satu perulangan, array satu dimensi, dan operasi String dasar. Tidak ada sorting, binary search, rekursi, palindrome, atau algoritma matriks baru.

Soal arrange baru mempunyai 9–13 tile kode, termasuk class dan main. Debug baru mempunyai satu kesalahan serta satu tile pengecoh; sumber program benarnya juga 9–13 tile. Output baru memuat 3–5 baris hasil. Soal lama dipertahankan meskipun bentuknya berbeda.

Konsep dasar memang berulang dalam konteks atau format berbeda. Bank ini bukan 100 algoritma berbeda. Instruksi menyebut tujuan, urutan output, dan langkah penting agar susunan yang benar tidak memerlukan tebakan terhadap maksud penyusun.

## Validasi yang dapat diulang

Jalankan tes web dari codearrange-web:

```powershell
npm test
```

Validasi kompilasi dan hasil Java memerlukan JDK 11 atau lebih baru pada PATH, hanya untuk pemeriksaan penyusun soal. Server web tetap cukup memakai Node.js:

```powershell
node scripts/validate-easy-java.js
```

Pemeriksaan Java mencakup 260 susunan diterima/program output dan 16 kontrol buggy, total 276 eksekusi. Hasil acuan ditulis dalam tests/fixtures/easy-puzzle-cases.json dan tidak diambil dari output program yang sedang diuji. Soal menyalin array lama yang tidak mencetak hasil memakai probe kesamaan array di harness pengujian, tanpa mengubah JSON soal.

Fixture easy-legacy-puzzles.json menyimpan kode dan solusi awal 13 soal lama untuk pemeriksaan preservasi. Tes lain memeriksa ID, urutan file, komposisi mode, label penelitian, tanpa deadline, grading jawaban salah, pergantian empat set, rating, serta log master/per-set. Server pengujian menulis hanya ke folder sementara, bukan sesi peserta.

Dari root repositori, jalankan juga:

```powershell
python -m unittest discover -s tests -p test_study_event_sync.py
python -m unittest discover -s analysis_sessions/dataset_fusion -p test_dataset_fusion.py
```

Tes ledger/sinkronisasi memakai data sintetis. Pemeriksaan tersebut tidak menguji kenyamanan manusia, kualitas alat sensor fisik, atau membuktikan cognitive load rendah.

## Audit alternatif jawaban

Audit tambahan memeriksa pertukaran statement lengkap pada setiap blok, termasuk urutan logika di dalam loop. Audit mempertahankan label baris identik dan struktur blok; kandidat yang memakai variabel di luar scope tidak dilanjutkan. Jalankan dari codearrange-web dengan JDK 11+:

```powershell
node scripts/audit-easy-solutions.js
```

Hasil pada bank saat ini: 554 kandidat susunan dikompilasi dan dieksekusi untuk soal arrange/debug. Seluruh solusi yang tercantum terverifikasi kembali. Total bank tetap 260 susunan diterima termasuk soal output; tidak ditemukan alternatif sesuai instruksi yang perlu ditambahkan dalam ruang pemeriksaan ini.

Tiga kandidat tambahan mencetak hasil contoh yang sama, tetapi melanggar instruksi: satu menukar urutan pengurangan stok rusak/terjual, dua mencetak kekurangan tabungan sebelum pemeriksaan selesai. Ketiganya tetap ditolak. Kesesuaian satu output contoh bukan alasan otomatis menambah solutions.

Alternatif logika yang sudah diterima antara lain menukar penambahan setoran tetap dan day++ pada daily_deposits. Nilai setoran tidak memakai day, sehingga hasilnya tetap benar. Pertukaran yang sama pada while_multiple_sum justru mengubah jumlah dari 30 menjadi 42 dan harus ditolak.

Tes server memakai kasus acuan pada tests/fixtures/easy-order-rubric.json untuk membedakan alternatif sah, hasil salah, dan output sama yang melanggar instruksi. Script audit akan meminta peninjauan jika ada kandidat output-sama yang belum mempunyai keputusan. Audit ini mencakup permutasi statement dalam blok yang ada, bukan pembuktian formal atas seluruh kemungkinan bentuk program Java atau pemindahan baris antarscope.

## Pilot sebelum pengambilan resmi

Bank sudah tersedia, tetapi pilot manusia belum dilakukan. Uji empat set pada dua orang yang memahami pemrograman dasar dan bukan penyusun soal. Utamakan orang di luar sampel resmi; jika peserta resmi terpapar soal saat pilot, catat paparan tersebut.

Catat waktu dan jumlah percobaan per soal, instruksi yang membingungkan, kenyamanan setelah setiap set, kebutuhan istirahat, serta window sensor yang lolos. Revisi soal yang memerlukan materi di luar standar easy atau sering menimbulkan salah tafsir. Jangan mengubah kunci hanya untuk menerima jawaban yang tidak sesuai instruksi.

Pipeline dataset saat ini memakai window HRV 25 detik dengan step 5 detik dan mensyaratkan seluruh rentang berada dalam active_task dengan difficulty yang jelas. Window yang melewati feedback, rating, atau istirahat dikeluarkan. Soal cepat tetap menghasilkan data mentah, tetapi belum tentu menghasilkan window HRV terpilih.

Jangan memperpanjang feedback, memaksa peserta menunggu, atau menambah jebakan untuk mengejar row. Pertahankan batas fase yang benar. Setelah pilot, periksa durasi aktif dan window berkualitas yang tersedia sebelum memperkirakan hasil dari maksimal delapan peserta.

Label easy tetap merupakan kondisi tugas/proxy CL. Menghapus timer dan menambah jumlah soal tidak menjamin beban kognitif rendah sepanjang sesi. Periksa kenyamanan dan dampak sesi panjang sebelum menetapkan protokol resmi.

## Katalog

A = arrange/susun kode; O = susun output; D = debug. Status Lama berarti kode dan solusi berasal dari bank lama. Output/debug pada easy-2 sampai easy-4 sudah diselingkan di antara arrange.

### easy-1

| No. | Soal | Mode | Asal |
|---:|---|:---:|---|
| 01 | [Variabel](../../codearrange-web/puzzles/easy-1/01_variables.json) | A | Lama |
| 02 | [Bilangan Genap atau Ganjil](../../codearrange-web/puzzles/easy-1/02_even_odd.json) | A | Lama |
| 03 | [Mencari Nilai Maksimum](../../codearrange-web/puzzles/easy-1/03_find_max.json) | A | Lama |
| 04 | [Perbaiki Penjumlahan Bilangan Genap](../../codearrange-web/puzzles/easy-1/04_fix_even_sum.json) | D | Lama |
| 05 | [Cetak Array 2D](../../codearrange-web/puzzles/easy-1/05_print_2d_array.json) | O | Lama |
| 06 | [Penjumlahan 1 sampai 10 dengan Loop](../../codearrange-web/puzzles/easy-1/06_for_loop_sum.json) | A | Lama |
| 07 | [Menyalin Array Secara Manual](../../codearrange-web/puzzles/easy-1/07_array_copy_manual.json) | A | Lama |
| 08 | [Percabangan If-Else Sederhana](../../codearrange-web/puzzles/easy-1/08_simple_ifelse.json) | A | Lama |
| 09 | [Penggabungan String](../../codearrange-web/puzzles/easy-1/09_string_concat.json) | A | Lama |
| 10 | [Mencetak Bilangan Terbalik](../../codearrange-web/puzzles/easy-1/10_basic_while_loop.json) | A | Lama |
| 11 | [Mencetak Bilangan](../../codearrange-web/puzzles/easy-1/11_basic_for_loop.json) | A | Lama |
| 12 | [Penjumlahan Array](../../codearrange-web/puzzles/easy-1/12_array_sum.json) | A | Lama |
| 13 | [Faktorial Iteratif](../../codearrange-web/puzzles/easy-1/13_factorial_iterative.json) | A | Lama |
| 14 | [Total Harga dan Uang Kembalian](../../codearrange-web/puzzles/easy-1/14_purchase_change.json) | A | Baru |
| 15 | [Konversi Menit ke Jam](../../codearrange-web/puzzles/easy-1/15_minutes_to_hours.json) | A | Baru |
| 16 | [Luas dan Keliling Persegi Panjang](../../codearrange-web/puzzles/easy-1/16_rectangle_measurements.json) | A | Baru |
| 17 | [Rata-Rata Tiga Nilai](../../codearrange-web/puzzles/easy-1/17_three_scores_average.json) | A | Baru |
| 18 | [Harga Setelah Potongan Nominal](../../codearrange-web/puzzles/easy-1/18_fixed_discount.json) | A | Baru |
| 19 | [Kecukupan Stok Pesanan](../../codearrange-web/puzzles/easy-1/19_stock_availability.json) | A | Baru |
| 20 | [Output Pembaruan Variabel](../../codearrange-web/puzzles/easy-1/20_trace_variable_updates.json) | O | Baru |
| 21 | [Output Percabangan pada Nilai Batas](../../codearrange-web/puzzles/easy-1/21_trace_boundary_branch.json) | O | Baru |
| 22 | [Output Penghitung While](../../codearrange-web/puzzles/easy-1/22_trace_while_counter.json) | O | Baru |
| 23 | [Perbaiki Batas Akhir Perulangan](../../codearrange-web/puzzles/easy-1/23_fix_inclusive_limit.json) | D | Baru |
| 24 | [Perbaiki Total Harga Belanja](../../codearrange-web/puzzles/easy-1/24_fix_purchase_total.json) | D | Baru |
| 25 | [Perbaiki Akses Elemen Pertama](../../codearrange-web/puzzles/easy-1/25_fix_first_array_index.json) | D | Baru |

### easy-2

| No. | Soal | Mode | Asal |
|---:|---|:---:|---|
| 01 | [Konversi Suhu Celsius](../../codearrange-web/puzzles/easy-2/01_celsius_conversion.json) | A | Baru |
| 02 | [Total Durasi Dua Kegiatan](../../codearrange-web/puzzles/easy-2/02_combined_duration.json) | A | Baru |
| 03 | [Tagihan Biaya Dasar dan Pemakaian](../../codearrange-web/puzzles/easy-2/03_service_bill.json) | A | Baru |
| 04 | [Output Deret Bilangan Ganjil](../../codearrange-web/puzzles/easy-2/04_trace_odd_numbers.json) | O | Baru |
| 05 | [Saldo Setelah Dua Transaksi](../../codearrange-web/puzzles/easy-2/05_account_balance.json) | A | Baru |
| 06 | [Luas Segitiga](../../codearrange-web/puzzles/easy-2/06_triangle_area.json) | A | Baru |
| 07 | [Perbaiki Langkah Penghitung While](../../codearrange-web/puzzles/easy-2/07_fix_while_step.json) | D | Baru |
| 08 | [Keliling Lingkaran](../../codearrange-web/puzzles/easy-2/08_circle_circumference.json) | A | Baru |
| 09 | [Nilai Akhir Berbobot](../../codearrange-web/puzzles/easy-2/09_weighted_score.json) | A | Baru |
| 10 | [Output Panjang dan Karakter String](../../codearrange-web/puzzles/easy-2/10_trace_string_access.json) | O | Baru |
| 11 | [Penambahan Stok di Bawah Batas](../../codearrange-web/puzzles/easy-2/11_restock_threshold.json) | A | Baru |
| 12 | [Pemeriksaan Kelipatan Lima](../../codearrange-web/puzzles/easy-2/12_multiple_of_five.json) | A | Baru |
| 13 | [Perbaiki Batas Kelulusan](../../codearrange-web/puzzles/easy-2/13_fix_pass_threshold.json) | D | Baru |
| 14 | [Nilai Dalam Rentang](../../codearrange-web/puzzles/easy-2/14_inclusive_range.json) | A | Baru |
| 15 | [Nilai Terkecil dari Dua Bilangan](../../codearrange-web/puzzles/easy-2/15_smaller_of_two.json) | A | Baru |
| 16 | [Output Perubahan Elemen Array](../../codearrange-web/puzzles/easy-2/16_trace_array_update.json) | O | Baru |
| 17 | [Bilangan Genap dengan Langkah Dua](../../codearrange-web/puzzles/easy-2/17_even_step_loop.json) | A | Baru |
| 18 | [Jumlah Kuadrat Bilangan](../../codearrange-web/puzzles/easy-2/18_sum_of_squares.json) | A | Baru |
| 19 | [Perbaiki Akumulasi Array](../../codearrange-web/puzzles/easy-2/19_fix_array_accumulator.json) | D | Baru |
| 20 | [Jumlah Kelipatan Tiga dengan While](../../codearrange-web/puzzles/easy-2/20_while_multiple_sum.json) | A | Baru |
| 21 | [Menghitung Bilangan Negatif](../../codearrange-web/puzzles/easy-2/21_count_negative_values.json) | A | Baru |
| 22 | [Output While dengan Pengurangan Dua](../../codearrange-web/puzzles/easy-2/22_trace_while_step_two.json) | O | Baru |
| 23 | [Mengalikan Elemen Array dengan Dua](../../codearrange-web/puzzles/easy-2/23_double_array_values.json) | A | Baru |
| 24 | [Merapikan Spasi Tepi Nama](../../codearrange-web/puzzles/easy-2/24_trim_name.json) | A | Baru |
| 25 | [Perbaiki Akses Karakter Terakhir](../../codearrange-web/puzzles/easy-2/25_fix_last_character.json) | D | Baru |

### easy-3

| No. | Soal | Mode | Asal |
|---:|---|:---:|---|
| 01 | [Sentimeter Menjadi Meter dan Sisanya](../../codearrange-web/puzzles/easy-3/01_centimeters_to_meters.json) | A | Baru |
| 02 | [Pendapatan Dua Kategori Tiket](../../codearrange-web/puzzles/easy-3/02_ticket_revenue.json) | A | Baru |
| 03 | [Ongkir Berdasarkan Berat](../../codearrange-web/puzzles/easy-3/03_shipping_cost.json) | A | Baru |
| 04 | [Output Aritmetika Berurutan](../../codearrange-web/puzzles/easy-3/04_trace_arithmetic_steps.json) | O | Baru |
| 05 | [Persentase Kehadiran](../../codearrange-web/puzzles/easy-3/05_attendance_percentage.json) | A | Baru |
| 06 | [Selisih Waktu Dalam Menit](../../codearrange-web/puzzles/easy-3/06_elapsed_minutes.json) | A | Baru |
| 07 | [Perbaiki Penyebut Rata-Rata Array](../../codearrange-web/puzzles/easy-3/07_fix_average_denominator.json) | D | Baru |
| 08 | [Pembagian Permen dan Sisanya](../../codearrange-web/puzzles/easy-3/08_candy_distribution.json) | A | Baru |
| 09 | [Harga Rata-Rata per Unit](../../codearrange-web/puzzles/easy-3/09_unit_price.json) | A | Baru |
| 10 | [Output Perbandingan Boolean](../../codearrange-web/puzzles/easy-3/10_trace_boolean_comparisons.json) | O | Baru |
| 11 | [Bilangan Positif, Negatif, atau Nol](../../codearrange-web/puzzles/easy-3/11_number_sign.json) | A | Baru |
| 12 | [Potongan Harga Anggota](../../codearrange-web/puzzles/easy-3/12_member_discount.json) | A | Baru |
| 13 | [Perbaiki Penyalinan Seluruh Array](../../codearrange-web/puzzles/easy-3/13_fix_complete_array_copy.json) | D | Baru |
| 14 | [Status Kipas Berdasarkan Suhu](../../codearrange-web/puzzles/easy-3/14_fan_temperature.json) | A | Baru |
| 15 | [Indeks dan Nilai Elemen Array](../../codearrange-web/puzzles/easy-3/15_array_index_labels.json) | A | Baru |
| 16 | [Output Sebagian Elemen Array](../../codearrange-web/puzzles/easy-3/16_trace_partial_array.json) | O | Baru |
| 17 | [Menghitung Nilai yang Mencapai Ambang](../../codearrange-web/puzzles/easy-3/17_count_passing_values.json) | A | Baru |
| 18 | [Nilai Minimum Array](../../codearrange-web/puzzles/easy-3/18_array_minimum.json) | A | Baru |
| 19 | [Perbaiki Selisih Dua Nilai](../../codearrange-web/puzzles/easy-3/19_fix_difference_operator.json) | D | Baru |
| 20 | [Hasil Perkalian Elemen Array](../../codearrange-web/puzzles/easy-3/20_array_product.json) | A | Baru |
| 21 | [Akumulasi Setoran Tetap dengan While](../../codearrange-web/puzzles/easy-3/21_daily_deposits.json) | A | Baru |
| 22 | [Output Teks dan Pemisah](../../codearrange-web/puzzles/easy-3/22_trace_text_separators.json) | O | Baru |
| 23 | [Mengubah Teks Menjadi Huruf Kapital](../../codearrange-web/puzzles/easy-3/23_uppercase_word.json) | A | Baru |
| 24 | [Mengambil Bagian Awal String](../../codearrange-web/puzzles/easy-3/24_word_prefix.json) | A | Baru |
| 25 | [Perbaiki Akses Elemen Terakhir](../../codearrange-web/puzzles/easy-3/25_fix_last_array_index.json) | D | Baru |

### easy-4

| No. | Soal | Mode | Asal |
|---:|---|:---:|---|
| 01 | [Biaya Parkir Bertahap](../../codearrange-web/puzzles/easy-4/01_parking_fee.json) | A | Baru |
| 02 | [Potongan Harga Kemudian Pajak](../../codearrange-web/puzzles/easy-4/02_discount_then_tax.json) | A | Baru |
| 03 | [Stok Setelah Tiga Perubahan](../../codearrange-web/puzzles/easy-4/03_inventory_transactions.json) | A | Baru |
| 04 | [Output Penyaringan Array](../../codearrange-web/puzzles/easy-4/04_trace_filtered_array.json) | O | Baru |
| 05 | [Kekurangan Tabungan](../../codearrange-web/puzzles/easy-4/05_savings_shortfall.json) | A | Baru |
| 06 | [Jam Menjadi Hari dan Sisanya](../../codearrange-web/puzzles/easy-4/06_hours_to_days.json) | A | Baru |
| 07 | [Perbaiki Hasil Pembagian Pecahan](../../codearrange-web/puzzles/easy-4/07_fix_fractional_division.json) | D | Baru |
| 08 | [Rata-Rata Gabungan Dua Kelompok](../../codearrange-web/puzzles/easy-4/08_combined_group_average.json) | A | Baru |
| 09 | [Kelompok Penuh dan Anggota Tersisa](../../codearrange-web/puzzles/easy-4/09_full_groups.json) | A | Baru |
| 10 | [Output Pertukaran Dua Nilai](../../codearrange-web/puzzles/easy-4/10_trace_value_swap.json) | O | Baru |
| 11 | [Dua Syarat Kelayakan](../../codearrange-web/puzzles/easy-4/11_two_eligibility_conditions.json) | A | Baru |
| 12 | [Nilai Terbesar dari Dua Bilangan](../../codearrange-web/puzzles/easy-4/12_larger_of_two.json) | A | Baru |
| 13 | [Perbaiki Inisialisasi Minimum](../../codearrange-web/puzzles/easy-4/13_fix_minimum_initialization.json) | D | Baru |
| 14 | [Status Lampu dari Boolean](../../codearrange-web/puzzles/easy-4/14_boolean_light_status.json) | A | Baru |
| 15 | [Tabel Perkalian Pendek](../../codearrange-web/puzzles/easy-4/15_multiplication_table.json) | A | Baru |
| 16 | [Output Jumlah Berjalan dengan While](../../codearrange-web/puzzles/easy-4/16_trace_while_running_total.json) | O | Baru |
| 17 | [Jumlah Berjalan dengan For](../../codearrange-web/puzzles/easy-4/17_for_running_total.json) | A | Baru |
| 18 | [Menambah Semua Elemen Array](../../codearrange-web/puzzles/easy-4/18_increment_array_values.json) | A | Baru |
| 19 | [Perbaiki Total Biaya Bertahap](../../codearrange-web/puzzles/easy-4/19_fix_cumulative_cost.json) | D | Baru |
| 20 | [Menghitung Bilangan Ganjil dalam Array](../../codearrange-web/puzzles/easy-4/20_count_odd_values.json) | A | Baru |
| 21 | [Jumlah Harga yang Memenuhi Ambang](../../codearrange-web/puzzles/easy-4/21_sum_prices_above_threshold.json) | A | Baru |
| 22 | [Output Rincian Potongan Harga](../../codearrange-web/puzzles/easy-4/22_trace_discount_branch.json) | O | Baru |
| 23 | [Mengganti Karakter Pemisah](../../codearrange-web/puzzles/easy-4/23_replace_separator.json) | A | Baru |
| 24 | [Mengambil Karakter Terakhir String](../../codearrange-web/puzzles/easy-4/24_last_string_character.json) | A | Baru |
| 25 | [Perbaiki Penghitung Bilangan Ganjil](../../codearrange-web/puzzles/easy-4/25_fix_odd_counter.json) | D | Baru |

