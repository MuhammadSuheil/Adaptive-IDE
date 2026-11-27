# Ringkasan Hasil Batch 07 — Keandalan Riwayat V3, Kalibrasi Ambang V4, & Evaluasi Silang V3 vs V4

Tanggal eksekusi lokal: 25 November 2026 (`Python 3.10.11`, `NumPy 2.2.6`, `Pandas 2.3.3`, `scikit-learn 1.7.2`). Revisi pelaporan: 27 November 2026 berdasarkan [DIAGNOSIS.md](DIAGNOSIS.md).

> **Catatan revisi.** Versi pertama ringkasan ini menonjolkan akurasi 77,8%, 81,0%, 81,5%, dan 72,2%. Keempat angka itu **di bawah atau sama dengan tebakan kelas mayoritas** pada data yang sama: V4 81,7% (LOPO) dan 81,5% (Test), V3 88,9% (Test). Selain itu, kebijakan ambang dan kandidat untuk angka tersebut dipilih setelah hasil Full LOPO/Test terlihat. Ringkasan ini sekarang memakai ROC AUC dan balanced accuracy sebagai metrik utama, selalu menyertakan pembanding mayoritas, dan menandai angka Test historis sebagai eksploratif. Angka mentah di `metrics.csv` tidak berubah.

## 1. Kesimpulan Utama

Batch 07 menjalankan empat eksperimen (`exp01`–`exp04`) secara lokal. Semua input diverifikasi dengan SHA-256, peserta dipisah ketat per fold, dan label V3 dihitung ulang di setiap fold. Protokol evaluasinya adalah yang paling sehat sejauh ini. Hasilnya:

1. **V4 (rating 7–10 vs 1–6) memberikan sinyal sensor yang nyata tetapi sedang.**
   - `win_logreg_core9_global` mencapai **AUC episode 0,778 (CI 95% bootstrap peserta 0,63–0,89)** pada LOPO 17 peserta (126 episode, 23 overload). Balanced accuracy-nya **0,741** pada ambang 0,50.
   - Pada Dev LOPO 13 peserta, yang tidak memuat peserta Test, AUC **0,739** dan balanced accuracy **0,753**.
   - Skor sensor sendirian tidak lebih baik daripada konteks tugas: durasi episode saja memberi AUC 0,816, dan tingkat set soal 0,776. Namun skor sensor **menambah informasi yang signifikan** di atas tingkat soal, durasi, dan waktu dalam sesi (AUC dalam sampel 0,884 → 0,927; uji likelihood-ratio p ≈ 0,0001). Ini temuan terkuat penelitian sejauh ini. Rinciannya ada di [DIAGNOSIS §5](DIAGNOSIS.md#5-v4-sinyal-nyata-di-atas-konteks-tetapi-sebagian-bersifat-antar-peserta).
   - Sebagian sinyal berasal dari perbedaan antar-peserta. AUC rata-rata di dalam peserta hanya **0,645**, dan hanya 10 dari 17 peserta yang memiliki kedua kelas.

2. **Kalibrasi ambang dalam-fold mengurangi alarm palsu, tetapi tidak menaikkan balanced accuracy secara konsisten.**
   - Pada Dev LOPO 13 peserta (bagian yang seharusnya dipakai untuk memilih), kebijakan terbaik adalah `fixed_050`/`nested_youden` (balanced accuracy 0,753). `nested_fa_cap25_max_rec` hanya **0,636** (peringkat ke-8): alarm palsu turun dari 31/81 ke 23/81, tetapi recall turun dari 16/18 ke 10/18.
   - Pada Full LOPO 17 peserta, `nested_fa_cap25_max_rec` memberi balanced accuracy **0,763 (CI 0,63–0,87)** dengan alarm palsu 22/103 dan recall 17/23. Akurasinya 77,8%, **di bawah tebakan selalu Non-Overload (81,7%)**. Full LOPO memuat peserta Test, sehingga keunggulan kebijakan ini teramati setelah seluruh data dilihat.
   - Pada 4 peserta Test historis (27 episode, 5 overload), `win_logreg_full14_global` + `nested_fa_cap25` memberi balanced accuracy **0,732 (CI 0,50–0,83)**. Akurasinya 81,5%, **sama persis dengan tebakan selalu Non-Overload**. Dengan kandidat pemenang Dev (`core9`) dan kebijakan yang sama, Test hanya mencapai balanced accuracy 0,673 (akurasi 59,3%).

3. **Riwayat dari `all_data.csv` lebih benar secara metodologis, tetapi model V3 belum terbukti mengukur beban kognitif.**
   - Mengganti riwayat terkurasi global `good_acceptable.csv` dengan riwayat berbasis kualitas dari `all_data.csv` menaikkan AUC LOPO V3 pada keenam kombinasi (+0,033 s.d. +0,092; LOPO 37 episode, 16 High). Selisih ini lebih kecil daripada galat baku AUC pada ukuran sampel tersebut (sekitar 0,08). Riwayat bersih dipertahankan karena menghilangkan seleksi global, bukan karena kenaikan AUC-nya.
   - Akurasi validasi 81,8% pada `exp13` batch 06 terbukti merupakan efek kurasi global. Dengan riwayat bersih, nilainya kembali ke 6/11 (54,5%).
   - **Skor SVM V3 mengikuti durasi episode.** Skor berkorelasi ρ = 0,63 dengan durasi bahkan di antara episode yang sama-sama Non-High, dan tidak menambah informasi di atas durasi (p = 0,79). Padahal label V3 sendiri didefinisikan dari durasi (durasi saja AUC 0,976).
   - Pada Test historis V3 (3 peserta, 18 episode, 2 High), AUC SVM `center_only` **0,500** dan hanya **1 dari 8** pasangan dalam-peserta terurut benar. Angka "13/18 benar (72,2%)" dengan `nested_recall75` berada di bawah tebakan selalu Non-High (16/18, 88,9%) dan dipilih dari 18 kombinasi yang dinilai pada Test.
   - Ketepatan urutan dalam-peserta 14/16 (87,5%) pada LOPO berasal dari hanya 5 peserta dan tidak bertahan pada Test.

4. **Evaluasi silang V3 vs V4 (exp04) menunjukkan kedua label searah, tetapi belum memvalidasi konstruk beban kognitif.**
   - Label perilaku V3 dan rating V4 bersepakat pada **34/40 episode (85,0%)**. Rating berkorelasi dengan durasi aktif (ρ = +0,633) dan jumlah kesalahan (ρ = +0,640), dan rating ditanyakan tepat setelah peserta selesai atau timeout. Kesepakatan ini menunjukkan kedua label mengukur "soal yang sulit bagi peserta", bukan pengukuran fisiologis independen.
   - Pada 40 episode irisan, `V4_LogReg_Core9` adalah **satu-satunya model yang batas bawah CI AUC-nya di atas 0,50 pada keempat definisi target** (misalnya rating ≥ 7: AUC 0,774, CI 0,56–0,95).
   - Ensemble `Dual_V3SVM_V4LR` (AUC 0,814 pada target konkordan, 7 positif) mempunyai CI 0,43–1,00, sehingga belum didukung secara statistik.

---

## 2. Tabel Komparasi Utama Batch 07

Kolom **Mayoritas** adalah akurasi tebakan selalu negatif pada episode yang sama. Bandingkan kolom **Akurasi** dengan kolom ini, bukan dengan 50%.

| Eksperimen | Dataset & evaluasi | Konfigurasi | Dipilih dari Dev? | ROC AUC | Balanced Acc | Akurasi | Mayoritas | Recall positif | Alarm palsu |
|---|---|---|:---:|---:|---:|---:|---:|---:|---:|
| Exp01 | V3 · LOPO Train (15p, 37 ep) | `extra_trees \| center_only \| fixed` | — | 0,786 | 67,56% | 64,86% | 56,8% | 14/16 | 11/21 |
| Exp01 | V3 · LOPO Train (15p, 37 ep) | `svm_rbf \| center_only \| fixed` | — | 0,768 | 69,20% | 67,57% | 56,8% | 13/16 | 9/21 |
| Exp02 | V3 · LOPO Train (15p, 37 ep) | `svm_rbf \| personal_iqr \| nested_bal_acc` | — | 0,783 | 68,45% | 67,57% | 56,8% | 12/16 | 8/21 |
| Exp02 | V3 · Validasi (3p, 11 ep) | `svm_rbf \| center_only \| nested_bal_acc` | — | 0,750 | 66,07% | 63,64% | 63,6% | 3/4 | 3/7 |
| Exp02 | V3 · Test historis (3p, 18 ep) ⚠ | `svm_rbf \| center_only \| fixed` | — | **0,500** | 53,13% | 55,56% | **88,9%** | 1/2 | 7/16 |
| **Exp03** | **V4 · Dev LOPO (13p, 99 ep)** | **`win_logreg_core9_global \| fixed_050`** | **Ya (pemenang)** | **0,739** | **75,31%** | 66,67% | 81,8% | 16/18 | 31/81 |
| Exp03 | V4 · Dev LOPO (13p, 99 ep) | `win_logreg_core9_global \| nested_fa_cap25` | Tidak (peringkat 8) | 0,739 | 63,58% | 68,69% | 81,8% | 10/18 | 23/81 |
| **Exp03** | **V4 · Full LOPO (17p, 126 ep)** | **`win_logreg_core9_global \| fixed_050`** | **Ya** | **0,778** | **74,06%** | 65,87% | 81,7% | 20/23 | 40/103 |
| Exp03 | V4 · Full LOPO (17p, 126 ep) | `win_logreg_core9_global \| nested_fa_cap25` | Tidak (pasca-hasil) | 0,778 | 76,28% | 77,78% | 81,7% | 17/23 | 22/103 |
| Exp03 | V4 · Full LOPO (17p, 126 ep) | `win_logreg_full14_global \| nested_youden` | Tidak (pasca-hasil) | 0,781 | 73,15% | 80,95% | 81,7% | 14/23 | 15/103 |
| Exp03 | V4 · Test historis (4p, 27 ep) ⚠ | `win_logreg_core9_global \| fixed_050` | Ya | 0,773 | 60,45% | 48,15% | 81,5% | 4/5 | 13/22 |
| Exp03 | V4 · Test historis (4p, 27 ep) ⚠ | `win_logreg_full14_global \| nested_fa_cap25` | Tidak (pasca-hasil) | 0,745 | 73,18% | 81,48% | 81,5% | 3/5 | 3/22 |
| Exp04 | Irisan V3∩V4 · LOPO (15p, 40 ep) | `V4_LogReg_Core9` → rating ≥ 7 | — | 0,774 | 72,58% | 57,50% | 77,5% | — | — |
| Exp04 | Irisan V3∩V4 · LOPO (15p, 40 ep) | `V3_SVM_RBF` → V3 High | — | 0,727 | 62,23% | 57,50% | 72,5% | 8/11 | 14/29 |

⚠ Test historis telah dievaluasi sejak batch 06 dan hanya memuat 2–5 episode positif. CI 95% AUC Test V4 adalah 0,25–0,92. Angka-angka ini eksploratif dan tidak dapat dipakai sebagai bukti generalisasi.

**Pembanding tanpa sensor pada Full LOPO V4 (126 episode):** waktu sejak episode pertama dalam sesi AUC 0,819, durasi aktif 0,816, jumlah window 0,808, tingkat set soal 0,776, pernah timeout 0,736. Model sensor (0,778) setara dengan pembanding ini, dan kontribusinya baru terlihat ketika keduanya digabung (lihat §1 butir 1).

---

## 3. Rekomendasi Operasional untuk Sistem Adaptive-IDE

Rekomendasi di bawah menggantikan versi sebelumnya, yang menyarankan ambang 0,58–0,62 dan skema dual-trigger. Keduanya belum tervalidasi.

1. **Kandidat model utama:** `win_logreg_core9_global` (logistik L2 per-window, bobot setara-episode, probabilitas dirata-ratakan per episode). Kandidat ini menang di Dev, unggul konsisten di Full LOPO, dan tangguh di exp04. `full14` cukup dipakai sebagai ablasi.
2. **Ambang belum boleh dikunci.** Ambang terpilih per fold berubah cukup jauh antar-subsampel peserta (misalnya `nested_youden` 0,40–0,75 pada `full14`). Batch 08 harus memilih kebijakan ambang di dalam fold, dengan aturan yang ditetapkan sebelum run.
3. **Model V3 belum layak menjadi pemicu intervensi.** Skornya belum dapat dibedakan dari durasi episode. Durasi dan jumlah kesalahan sudah dapat dibaca IDE tanpa sensor. V3 dipertahankan sebagai label validasi silang.
4. **Model saat ini menilai setelah episode selesai.** Kegunaan untuk intervensi saat tugas berjalan (deteksi pada 30–120 detik pertama) belum diukur dan menjadi pertanyaan Batch 08.
5. **Pelaporan:** gunakan ROC AUC, balanced accuracy, PR-AUC, recall pada batas alarm palsu, dan AUC dalam peserta, masing-masing dengan CI per peserta. Selalu sertakan pembanding mayoritas dan pembanding konteks tanpa sensor. Akurasi saja tidak bermakna ketika 82% episode adalah Non-Overload.

Prioritas Batch 08 dan keputusan dataset tercantum di [DIAGNOSIS.md §10–11](DIAGNOSIS.md#10-prioritas-batch-08-batch-terakhir).
