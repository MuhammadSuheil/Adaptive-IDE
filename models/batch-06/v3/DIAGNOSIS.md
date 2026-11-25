# Diagnosis hasil baseline Batch 06

Pemeriksaan 24 November 2026 WIB. Run `baseline_20261123T192338Z_8d8cee5e` selesai dengan 165 fitting CV dan empat fitting validation, tanpa evaluasi Test. Baseline meningkatkan beberapa hasil agregat, tetapi belum menghasilkan classifier tiga tingkat yang kuat pada orang yang ditahan dari fitting.

## 1. Manfaat baseline pada OOF

| Subset dan model | Window / orang | Accuracy raw | Accuracy relatif | Delta accuracy | Macro-F1 raw | Macro-F1 relatif |
|---|---|---:|---:|---:|---:|---:|
| RMSSD, Logistic Regression | 2.454 / 19 | 37,73% | 39,41% | +1,67 pp | 35,59% | 38,15% |
| RMSSD, Extra Trees | 2.454 / 19 | 35,66% | 37,33% | +1,67 pp | 34,03% | 35,10% |
| RMSSD dan eye, Logistic Regression | 671 / 4 | 34,43% | 31,59% | −2,83 pp | 32,56% | 26,78% |
| RMSSD dan eye, Extra Trees | 671 / 4 | 35,77% | 41,43% | +5,66 pp | 33,96% | 39,85% |

OOF menggabungkan satu prediksi per window Train dari fold yang menahan seluruh data orang tersebut. Angka ini merupakan hasil development; OOF yang sama memilih satu dari dua keluarga model untuk tiap representasi. Peningkatan dinilai dari pasangan raw-relative dalam subset yang sama, bukan dari perbandingan skor dua subset yang mempunyai peserta berbeda.

RMSSD Logistic Regression menambah 41 jawaban benar: 926 menjadi 967 dari 2.454. Recall High naik dari 12,80% menjadi 21,74%, sedangkan Medium turun dari 39,59% menjadi 34,93%. Low hampir tetap, 60,26% menjadi 60,61%. Pemisahan tiga label masih lemah, khususnya High pada orang baru.

Eye Extra Trees menambah 38 jawaban benar: 240 menjadi 278 dari 671. Low tetap 132/264, Medium naik 73/180 menjadi 99/180, dan High naik 35/227 menjadi 47/227. Ada manfaat pada Medium/High, tetapi recall High relatif masih hanya 20,70%.

Dummy pada eye memperoleh 39,34% accuracy dengan semua prediksi Low. Model relatif 41,43% menambah 14 prediksi benar, atau 2,09 poin accuracy. Macro-F1 relatif 39,85% lebih tinggi daripada Dummy 18,82%, sehingga model mengenali lebih dari satu kelas, tetapi kemampuan keseluruhannya masih terbatas.

Sumber: [leaderboard](outputs/exp01/baseline_20261123T192338Z_8d8cee5e/cv_leaderboard.csv), [perbandingan pasangan](outputs/exp01/baseline_20261123T192338Z_8d8cee5e/paired_comparison.csv), dan [metrik dengan dukungan kelas](outputs/exp01/baseline_20261123T192338Z_8d8cee5e/analysis/metrics_with_class_support.csv).

## 2. Manfaat antarorang belum merata

Pada RMSSD Logistic Regression, accuracy naik untuk 11 orang, turun untuk enam, dan tetap untuk dua. Accuracy gabungan window naik 37,73% menjadi 39,41%, tetapi rata-rata accuracy dengan bobot sama setiap orang turun **40,51% menjadi 38,61%**. Perubahan besar tidak merata terhadap jumlah window.

Contohnya, Shakilla turun 15/26 menjadi 0/26 benar dan Ramadhan turun 30/52 menjadi 13/52. Andreas naik 19/71 menjadi 41/71, Mon naik 80/171 menjadi 105/171, dan Baithar naik 77/201 menjadi 98/201. Ini penelusuran heterogenitas, bukan alasan menghapus sumber yang memburuk atau mengganti labelnya.

Macro-F1 rata-rata per orang pada konfigurasi yang sama naik kecil, 27,59% menjadi 28,30%. Jadi manfaat juga bergantung pada metrik: pengenalan beberapa kelas membaik tanpa menghasilkan rata-rata accuracy per pengguna yang lebih tinggi. Macro-F1 per orang tetap memakai tiga kelas, termasuk kelas yang tidak hadir pada fold tertentu.

Pada eye Extra Trees, accuracy naik pada dua orang, tetap pada satu, dan turun pada satu. Macro-F1 naik pada keempat orang:

| Orang | Accuracy raw | Accuracy relatif | Macro-F1 raw | Macro-F1 relatif |
|---|---:|---:|---:|---:|
| Baithar | 28,72% | 39,49% | 23,87% | 36,05% |
| Fabian | 26,94% | 26,94% | 14,27% | 18,03% |
| Mon | 47,37% | 60,23% | 21,43% | 41,88% |
| kaka | 45,54% | 41,07% | 22,25% | 28,29% |

Ini sinyal manfaat Macro-F1 pada kontrol kecil tersebut, bukan bukti bahwa normalisasi eye akan konsisten pada populasi luas. Logistic Regression eye pada orang yang sama justru memburuk.

![Perubahan accuracy per orang](outputs/exp01/baseline_20261123T192338Z_8d8cee5e/analysis/figures/participant_accuracy_changes.png)

Sumber: [fold_metrics.csv](outputs/exp01/baseline_20261123T192338Z_8d8cee5e/fold_metrics.csv) dan [participant_metrics.csv](outputs/exp01/baseline_20261123T192338Z_8d8cee5e/analysis/participant_metrics.csv).

## 3. Validation eye 65,66% tidak mencakup tiga kelas

Validation eye memiliki **43 Low, nol Medium, dan 56 High**, dari tiga orang yang juga ada pada Train. Laporan otomatis belum menampilkan support kelas pada tabel validation. Recall Medium nol berasal dari support nol dan `zero_division=0`; tidak ada contoh Medium yang dapat dinilai. Model tetap menghasilkan tujuh prediksi Medium pada window Low/High. Jadi menyatakan seluruh Medium salah atau model tidak pernah memprediksi Medium merupakan pembacaan yang keliru.

| Label sebenarnya | Prediksi Low | Prediksi Medium | Prediksi High | Support |
|---|---:|---:|---:|---:|
| Low | 23 | 2 | 18 | 43 |
| Medium | 0 | 0 | 0 | 0 |
| High | 9 | 5 | 42 | 56 |

Accuracy 65/99 = 65,66% dan Macro-F1 tiga kelas 44,58% cocok dengan prediksi. Macro-F1 masih memasukkan Medium dengan F1 nol. Menghitung ulang hanya dua kelas akan mengubah metrik dan tidak menjadikannya hasil classifier tiga kelas yang tervalidasi.

Kenaikan dari raw 52/99 menjadi relatif 65/99 terkonsentrasi pada satu sumber:

| Orang | Komposisi validation | Benar raw | Benar relatif |
|---|---|---:|---:|
| Baithar | 39 High | 39/39 | 38/39 |
| Fabian | 6 Low, 17 High | 4/23 | 4/23 |
| Mon | 37 Low | 9/37 | 23/37 |

Perbaikan total +13 berasal dari +14 pada Mon dan −1 pada Baithar. Komposisi kelas sangat terkait peserta. Tabel ini tidak memisahkan efek keakraban sesi, konteks waktu, dan distribusi kelas secara kausal.

Angka 65,66% sah untuk 99 window tersebut, tetapi belum menilai tiga kelas secara lengkap maupun pengguna baru. OOF tiga kelas untuk generalisasi orang tetap 41,43% pada empat orang. Ketiadaan Medium bukan satu-satunya perbedaan dua evaluasi; peserta, interval, dan peran datanya juga berubah.

Sumber: [prediksi validation](outputs/exp01/baseline_20261123T192338Z_8d8cee5e/validation_predictions.csv), [support kelas](outputs/exp01/baseline_20261123T192338Z_8d8cee5e/analysis/metrics_with_class_support.csv), dan [rincian sumber/fase](outputs/exp01/baseline_20261123T192338Z_8d8cee5e/analysis/validation_source_phase_metrics.csv).

## 4. Validation RMSSD dan gap fitting

Pada 721 window validation, tersedia 198 Low, 269 Medium, dan 254 High. RMSSD Logistic Regression raw benar 262, relatif 284: accuracy naik 36,34% menjadi 39,39%, Macro-F1 36,11% menjadi 39,24%. Low bertambah satu benar, Medium turun empat, dan High bertambah 25. Manfaat untuk sebagian pemisahan High belum menjadi pengenalan tiga kelas yang kuat.

Fitting accuracy Logistic Regression tercatat 52,85% raw dan 52,69% relatif; model linear juga belum memisahkan Train dengan baik. Pada eye Extra Trees, fitting accuracy 94,63% raw dan 94,49% relatif, sedangkan OOF peserta 35,77% dan 41,43%. Gap fitting-held-out tetap besar setelah normalisasi. Nilai fitting adalah catatan run, bukan hasil reproduksi inference pada audit ini.

## 5. Kontrol waktu dan batas atribusi

RMSSD relatif menaikkan Macro-F1 CV blok Logistic Regression 34,48% menjadi 35,86%, tetapi Extra Trees turun 34,55% menjadi 33,56%. Manfaat RMSSD tidak universal pada dua model dan dua jenis evaluasi.

Eye Extra Trees menaikkan accuracy CV blok 35,47% menjadi 39,34%, membaik pada lima fold; Macro-F1 34,26% menjadi 38,38%. Fold tersebut tetap berbagi empat sumber orang yang sama, bukan lima kelompok populasi independen.

Cabang `hrv_eye` mengubah RMSSD dan eye secara bersamaan. Kontrolnya belum mengisolasi kontribusi eye. Membandingkan langsung cabang RMSSD 19 orang dengan cabang eye empat orang juga tidak mengisolasi modalitas. Yang didukung adalah manfaat transformasi gabungan pada Extra Trees dan subset empat orang ini.

## 6. Arah berikutnya

1. **Laporkan manfaat terbatas.** Sertakan OOF tiga kelas, metrik dengan bobot sama per orang, recall, dan jumlah sumber bersama angka validation. Jangan mempromosikan 65,66% sebagai hasil tiga kelas pada pengguna baru.
2. **Isolasi transformasi pada subset identik.** Jika dilanjutkan, bandingkan raw, RMSSD-only, eye-only, dan gabungan pada empat orang, baris/fold, serta parameter Extra Trees yang sama. Kontrol ini belum dijalankan.
3. **Lengkapi dukungan kelas setiap bagian.** Laporan berikut perlu menandai kelas yang tidak dapat dinilai. Jika development diubah agar tiga kelas tersedia, jadikan eksperimen baru dengan aturan metadata yang dibakukan, tanpa memilih split berdasarkan skor atau mengganti hasil lama.
4. **Perluas pengambilan acuan awal eye.** Empat orang membatasi kesimpulan. Tambahan data perlu menambah sumber dengan acuan awal layak dan tiga level tugas; memperbanyak window overlap tidak menggantikan keragaman orang.

Normalisasi tidak menggantikan pengukuran beban mental independen. Label tetap difficulty/fase, SDNN tetap mentah, dan acuan eye tetap pre_task. Hasil tidak membuktikan seluruh sensor rusak, seluruh label salah, atau bahwa search model berikutnya akan mencapai 70%.

## 7. Integritas pemeriksaan

Metrik 20 tabel OOF, empat validation, dan 165 fold beserta JSON pekerjaan dihitung ulang dan cocok. Tiga hash CSV, empat checksum bundle, delapan delta pasangan, 33 manifest fold, serta 19 referensi baseline diperiksa. Label dan grup orang sesuai CSV. Fold peserta memisahkan orang, fold waktu memenuhi guard 45 detik, dan ID Test tidak muncul pada prediksi. Empat acuan eye terpisah dari dukungan interval yang dinilai.

Source cell notebook eksekusi identik dengan notebook persiapan dan tidak mempunyai output exception. Plan mencatat NumPy 2.4.6, Pandas 2.2.3, scikit-learn 1.6.1, dan Python 3.13.18. Pemeriksaan memakai prediksi tersimpan, bukan inference silang versi; selisih versi tidak ditetapkan sebagai penyebab hasil.

Tidak ada fitting ulang, inference model, perubahan dataset, atau penggantian model terpilih. Joblib hanya dibaca sebagai byte untuk checksum. Berkas asli ZIP dipertahankan dan notebook eksekusi diarsipkan terpisah. [audit.json](outputs/exp01/baseline_20261123T192338Z_8d8cee5e/analysis/audit.json) mencatat cakupan pemeriksaan.
