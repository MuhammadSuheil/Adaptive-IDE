# Batch 06 · Pencapaian, diagnosis, dan arah pengembangan v3 dan v4

**25 November 2026.** V4 exp03 memberikan bukti paling konsisten untuk mengurutkan episode berdasarkan kesulitan yang dirasakan peserta. Pada Test, AUC-nya **0,773** dan model menangkap **4/5** episode berating tinggi, dengan **13 alarm palsu dari 22** episode lainnya. V3 exp11 memperoleh akurasi Test tertinggi di antara tiga model utama (**61,1%**), tetapi hanya menangkap **1/2 High** dan AUC-nya **0,500**. Pilihan model bergantung pada kegunaan skor dan biaya alarm palsu.

## Data dan pertanyaan yang dijawab

| | V3 · proxy perilaku | V4 · rating peserta |
|---|---|---|
| Target biner | High menurut rubrik waktu pengerjaan aktif dan kesalahan manual. Low dan Medium digabung menjadi Non-High. | Rating kesulitan 7–10 menjadi overload dan 1–6 menjadi non-overload. Semua rating ikut evaluasi biner. |
| Data layak | 66 episode, 21 peserta, 1.193 window. Ada 22 High, 28 Low, dan 16 Medium. | 126 episode, 17 peserta, 1.693 window. Ada 23 overload dan 103 non-overload. |
| Pengembangan | LOPO di dalam Train menghasilkan 37 episode dari 15 peserta. Validasi terpisah berisi 11 episode dari 3 peserta. | LOPO penuh pada 17 peserta untuk membandingkan kandidat. Pada evaluasi dengan Test terpisah, model dipilih lewat LOPO pada 13 peserta Dev, mencakup 99 episode. |
| Test | 18 episode dari 3 peserta, termasuk 2 High. | 27 episode dari 4 peserta, termasuk 5 overload. |

Seluruh metrik perbandingan di bawah memberi satu suara kepada satu episode pengerjaan soal. Window sensor berdurasi 15 detik dengan langkah 5 detik dapat saling tumpang tindih. Pada v3, High menyumbang **33,3% episode** tetapi **69,2% window** karena episode High cenderung lebih panjang. Kedua target merupakan pendekatan untuk mempelajari beban kognitif. Rating v4 secara khusus menanyakan kesulitan soal. [Sebaran dataset](dataset/DATASET_README.md)

## Perbandingan hasil biner per episode

| Evaluasi | Model | Benar / total | Akurasi | Balanced accuracy | Macro-F1 | ROC AUC | High terdeteksi / seluruh High | Alarm palsu / seluruh Non-High |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LOPO Train | V3 exp11 | 26/37 | 70,3% | 72,32% | 0,702 | 0,699 | 14/16 | 9/21 |
| LOPO Train | V3 exp13 | 25/37 | 67,6% | 69,20% | 0,675 | 0,690 | 13/16 | 9/21 |
| LOPO penuh | V4 exp03 | 88/126 | 69,8% | 69,73% | 0,624 | 0,781 | 16/23 | 31/103 |
| Validasi terpisah | V3 exp11 | 6/11 | 54,5% | 58,93% | 0,545 | 0,607 | 3/4 | 4/7 |
| Validasi terpisah | V3 exp13 | 9/11 | 81,8% | 85,71% | 0,817 | 0,714 | 4/4 | 2/7 |
| Validasi silang Dev | V4 exp03 | 66/99 | 66,7% | 75,31% | 0,622 | 0,739 | 16/18 | 31/81 |
| Test | V3 exp11 | 11/18 | 61,1% | 56,25% | 0,481 | 0,500 | 1/2 | 6/16 |
| Test | V3 exp13 | 9/18 | 50,0% | 50,00% | 0,411 | 0,406 | 1/2 | 8/16 |
| Test | V4 exp03 | 13/27 | 48,1% | 60,45% | 0,463 | 0,773 | 4/5 | 13/22 |

**Cara membaca metrik.** Akurasi menghitung proporsi keputusan yang benar pada ambang model. Balanced accuracy merata-ratakan recall High dan Non-High agar kedua kelas mendapat bobot sama. Macro-F1 merata-ratakan F1 kedua kelas, dengan F1 memperhitungkan presisi dan recall. ROC AUC menilai seberapa sering episode positif mendapat skor lebih tinggi daripada episode negatif. Dua kolom terakhir memperlihatkan konsekuensi ambang dalam jumlah episode. Pada Test v4, misalnya, model memanggil 17 episode sebagai overload dan hanya 4 yang benar, sehingga presisi overload **4/17 (23,5%)**, walaupun recall mencapai **4/5 (80%)**. [Metrik pengembangan v3](v3/exp13/outputs/iqr_20261125T091155Z_5a63ee6f/metrics.csv) · [Metrik Test v3](v3/report/test_metrics.csv) · [Metrik v4](outputs/exp03_v4_binary_overload/binary_deepdive_summary.json)

## V3 · Pencapaian dan diagnosis

- **Median sensor per episode menjadi acuan terbaik untuk model gabungan mata, HRV, dan posisi kepala.** Pada LOPO, SVM exp11 yang memakai median fitur benar 26/37 kali. Melatih SVM per window lalu merata-ratakan skornya hanya benar 17/37 kali. Pada target awal tiga kelas, regresi logistik tim mencapai akurasi Test 33,3% dan Macro-F1 0,301. Penggabungan Low dan Medium memberi model tujuan yang lebih sempit, yaitu mendeteksi High setelah episode selesai. [Exp11](v3/exp11/README.md) · [Hasil tiga kelas](outputs/exp01_v3_behavioral_proxy/RESULTS.md)
- **Normalisasi personal membantu tiga peserta validasi, tetapi manfaatnya belum konsisten.** Exp13 mengoreksi satu keputusan pada masing-masing peserta validasi. Di Test, kedua model menangkap High yang sama dan exp13 menambah dua alarm palsu secara bersih. Menebak Non-High terus memberi 16/18 benar (88,9%) dan melewatkan kedua High. Satu prediksi High menggeser recall Test sebesar 50 poin persentase. [Exp13](v3/exp13/README.md) · [Prediksi Test](v3/report/test_episode_predictions.csv)
- **Konvergensi dan durasi HRV belum menjelaskan generalisasi.** SVM exp11 dan exp13 konvergen pada 56 dan 49 iterasi. Hinge loss Train per episode versus validasi mencapai 0,423 versus 0,861 untuk exp11 dan 0,443 versus 0,886 untuk exp13. Percobaan HRV 45 detik pada exp12 tetap benar 6/11 di validasi, dengan 4/4 High terdeteksi dan 5/7 alarm palsu. [Kurva training dan validasi](v3/report/training_curve.png) · [Percobaan HRV 45 detik](v3/exp12/README.md)
- **Audit menemukan pengaruh kurasi global pada riwayat sensor.** Peserta, sesi, episode, dan ID window terpisah antar-split. Referensi label, scaler, dan model memakai data fitting. Namun, sumber riwayat berisi 4.299 window v2 yang dipilih melalui penyeimbangan seluruh peserta dan sesi. Memakai seluruh 6.517 window yang lolos kualitas sambil tetap membatasi waktu hingga awal soal mengubah median referensi pada **17/37 episode LOPO, 7/11 validasi, dan 13/18 Test**. Dampaknya terhadap skor dan akurasi masih belum diukur. Validasi juga sudah dipakai untuk merumuskan exp13 dan split Test pernah dinilai pada eksperimen tiga kelas. [Aturan kurasi v2](../../analysis_sessions/dataset_fusion/v2_phase_policy.py) · [Pembentukan riwayat](v3/exp06/history.py) · [Evaluasi Test](v3/report/evaluate_test.py)

## V4 · Pencapaian dan diagnosis

- **Pengurutan rating tinggi bertahan pada empat peserta Test.** AUC LOPO penuh mencapai 0,781 dengan interval bootstrap peserta 95% sebesar 0,647–0,894. Kandidat terbaik LOPO penuh memakai regresi logistik L2 dengan fitur inti, posisi kepala, dan baseline. Seleksi terpisah pada 13 peserta Dev mengunci model sembilan fitur inti untuk Test. Bukti paling relevan untuk resep terkunci ialah AUC **0,739 di Dev dan 0,773 di Test**. [Hasil v4 exp03](outputs/exp03_v4_binary_overload/RESULTS.md)
- **Ambang keputusan dan ketimpangan kelas menjadi kendala utama.** Pada LOPO penuh, menaikkan ambang dari 0,50 ke 0,65 meningkatkan akurasi dari 69,8% ke 80,2% sambil menurunkan deteksi overload dari 16/23 ke 11/23. Pada Test, tebakan semua non-overload mencapai 22/27 benar (81,5%) dengan nol overload terdeteksi. Proporsi salah klasifikasi pada confusion matrix mengukur kesalahan keputusan. Penyebab seperti kualitas HRV memerlukan uji tersendiri. [Confusion matrix v4](outputs/exp03_v4_binary_overload/figures/v4_binary_confusion_matrices_detailed.png)
- **Rating tepat 1–10 masih sulit diprediksi.** Pada Test, regresi exp02 menghasilkan MAE **2,528 poin rating** dibanding 2,969 untuk tebakan rerata, dengan R² **−0,136** dan korelasi peringkat Spearman **−0,196**. Klasifikasi tiga kelompok rating mencapai akurasi **37,0%**, balanced accuracy **36,0%**, dan Macro-F1 **0,345**. Beberapa rating hanya diwakili tiga atau empat episode. Soal yang dirancang mudah juga dapat dirasakan sulit oleh peserta, sehingga rating tinggi pada soal mudah perlu dipertahankan saat targetnya pengalaman peserta. [Hasil v4 exp02](outputs/exp02_v4_self_reported_rating/RESULTS.md) · [Distribusi rating](dataset/DATASET_README.md)

## Kesimpulan dan arah pengembangan

**Prioritaskan v4 exp03 untuk mengurutkan episode yang kemungkinan dirasakan sulit.** AUC Test v4 sebesar 0,773 mengungguli 0,500 pada v3 exp11 dan 0,406 pada v3 exp13 untuk target masing-masing. Untuk keputusan High pada ambang saat ini, ketiganya masih sering memberi alarm palsu. **Pertahankan exp11 sebagai acuan v3** ketika pertanyaannya menyangkut kesulitan yang terlihat dari perilaku pengerjaan. Akurasi validasi exp13 sebesar 81,8% belum berulang pada peserta Test. Perbedaan target, peserta, dan jumlah High membatasi klaim keunggulan langsung antardataset.

1. **V4 lebih dahulu.** Gunakan prediksi di luar fold pada 13 peserta Dev untuk memilih ambang. Tetapkan lebih dahulu berapa banyak overload yang perlu terdeteksi dan berapa alarm palsu yang dapat diterima. Bandingkan ambang tetap 0,50 dengan pemilihan ambang di dalam fold peserta, lalu laporkan presisi, recall, alarm palsu, balanced accuracy, dan AUC per episode. Test yang sudah dibaca menjadi hasil historis.
2. **V3 sebagai uji keandalan.** Ulangi exp11 dan exp13 dengan riwayat dari window berkualitas yang sudah tersedia pada saat episode dimulai. Hitung perubahan skor dan keputusan pada episode yang sama, dengan fitting dan ambang yang dibekukan. Langkah ini dapat dilakukan dari rekaman yang sudah ada dan langsung mengukur dampak kurasi global.
3. **Untuk memilih model beban kognitif, samakan pertanyaan evaluasinya.** Pada episode yang memiliki label perilaku dan rating, hasilkan prediksi dari peserta yang disisihkan dengan split yang sama, lalu bandingkan kedua model terhadap target yang sama. Laporkan berapa peserta dan episode positif yang ikut. Dengan begitu tim dapat menilai apakah skor mengikuti kesulitan yang dilaporkan peserta, kesulitan yang tercermin pada perilaku, atau keduanya.
