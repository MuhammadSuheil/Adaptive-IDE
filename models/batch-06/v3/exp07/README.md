# Exp07 · kapan IDE perlu merespons High?

Eksperimen v3 ini menguji dua keputusan untuk IDE. Jika IDE memberi bantuan hanya pada High, Low dan Medium boleh diberi target operasional yang sama, **Non-High**. Jika IDE juga memberi petunjuk ringan pada Medium, ia masih perlu membedakan tiga kelas. Pada paket v3, Medium berarti soal selesai dengan satu atau dua kesalahan manual dan waktu aktif di rentang tengah referensi soal. Menggabungkannya menghilangkan kesempatan untuk mengenali kondisi tersebut secara tersendiri. CSV asli tetap menyimpan ketiga label untuk analisis.

## Ukuran kelas dan peserta

| Split resmi | Orang terdaftar / dengan window berlabel layak | Episode Low / Medium / High | Episode Non-High / High | Window Non-High / High |
|---|---:|---:|---:|---:|
| Train | 16 / 15 | 12 / 9 / 16 | 21 / 16 | 188 / 556 |
| Validation | 3 / 3 | 4 / 3 / 4 | 7 / 4 | 67 / 151 |
| Test | 3 / 3 | 12 / 4 / 2 | 16 / 2 | 112 / 119 |
| Total | 22 / 21 | 28 / 16 / 22 | 44 / 22 | 367 / 826 |

Proporsi Non-High adalah 66,7% per episode dan 30,8% per window. Episode High yang panjang memberi lebih banyak window 15 detik dengan step 5 detik. Model di sini memprediksi sesudah episode selesai dan setiap episode evaluasi bernilai satu. Dengan komposisi Test ini, menebak Non-High terus akan benar 16/18 (88,9%) sekaligus melewatkan kedua High. Angka tersebut dihitung dari distribusi label. Exp07 tidak menilai model pada Test.

Split v3 menyediakan sekitar 15% peserta masing-masing untuk validation dan Test, dibulatkan menjadi tiga dari 22. Tim memilih pembagian dengan memperhatikan cakupan peserta per kelas, sebelum fitting. V4 dahulu memakai protokol berbeda: 17 orang ditahan bergiliran, satu per fold. [Kontrak dan split v3](../dataset/v3_behavioral_proxy/README.md)

## Evaluasi dan resep

Exp07 melakukan LOPO di dalam Train v3. Untuk tiap peserta yang ditahan, notebook membangun ulang referensi waktu per soal dari peserta inner-Train yang memenuhi syarat, mengecualikan referensi diri, lalu menghitung label sebelum pemetaan biner. Episode tanpa cukup bukti tetap tanpa label. Kandidat window berasal dari seluruh window v2 berkualitas good/acceptable di dalam satu episode aktif. Ini memungkinkan perubahan kelayakan episode antar-fold. Dengan split resmi penuh, implementasi mencocokkan label pada 358 episode dan ID seluruh 1.193 window berlabel layak dengan paket v3.

Dari 16 fold Train, 15 menyumbang total 37 episode held-out berlabel. Satu peserta tidak mempunyai endpoint yang layak. Fitting per fold mempunyai 26–37 episode berlabel. Sesudah LOPO, resep yang sama dilatih pada Train lengkap dan dievaluasi pada 11 episode validation. Validation ini sudah diperiksa dalam eksperimen sebelumnya.

- **Tiga kelas.** Dua SVC RBF memakai median fitur eye mentah atau median core + head relatif. Riwayat relatif mengikuti exp06 dan setiap referensi berakhir sebelum episode. StandardScaler dan pembobotan dipelajari dari fitting. Parameter tetap C=1 dan gamma=scale.
- **Pemetaan hasil tiga kelas.** Prediksi Low dan Medium yang sudah dibuat model tiga kelas ditampilkan sebagai Non-High. Batas keputusan High model itu tetap sama.
- **Training biner langsung.** Fitting memakai label Non-High dan High sejak awal. Bobot dihitung kembali untuk dua kelas, peserta dalam masing-masing kelas, dan episode per peserta. Ini mengubah batas keputusan High.
- **Dua tahap dengan tiga keluaran.** Gate eye biner memutuskan High. Untuk episode yang lolos sebagai Non-High, SVC core + head relatif yang dilatih pada Low dan Medium memisahkan kedua kelas tersebut.

## Hasil per episode

| Target dan resep | Benar LOPO Train | Benar validation | High terdeteksi pada validation | Alarm High palsu pada validation |
|---|---:|---:|---:|---:|
| Non-High/High, kelas mayoritas fitting | 21/37 (56,8%) | 7/11 (63,6%) | 0/4 | 0 |
| Non-High/High, eye tiga kelas kemudian dipetakan | 10/37 (27,0%) | 9/11 (81,8%) | 3/4 | 1 |
| Non-High/High, eye biner langsung | 10/37 (27,0%) | 7/11 (63,6%) | 3/4 | 3 |
| Non-High/High, core + head relatif tiga kelas kemudian dipetakan | 23/37 (62,2%) | 7/11 (63,6%) | 3/4 | 3 |
| Non-High/High, core + head relatif biner langsung | **26/37 (70,3%)** | **6/11 (54,5%)** | **3/4** | **4** |
| Tiga kelas, core + head relatif | 18/37 (48,6%) | 5/11 (45,5%) | 3/4 | 3 |
| Tiga kelas, dua tahap | 7/37 (18,9%) | 5/11 (45,5%) | 3/4 | 3 |

Pada model eye tiga kelas, ketiga Medium validation diprediksi Low. Saat keluaran digabung, skor validation naik dari 6/11 ke 9/11 karena tiga kasus itu sekarang memenuhi target Non-High. Ia tetap melewatkan satu dari empat High. Akurasi binernya 27,0% pada LOPO Train.

Model biner relatif menangkap 14/16 High pada LOPO dengan 9 alarm palsu dari 21 Non-High. Pada validation model menangkap 3/4 High, dengan 4 alarm palsu dari 7 Non-High (2 Low dan 2 Medium). Precision High menjadi 3/7 dan recall High 3/4. Model ini mengutamakan sensitivitas, dengan risiko IDE sering menginterupsi pengguna yang tidak membutuhkan bantuan tingkat High. Belum ada resep yang mempertahankan accuracy 70% pada kedua evaluasi.

Jika satu-satunya aksi adalah memberi bantuan pada High, jadikan model biner relatif kandidat pengembangan. Tetapkan lebih dulu berapa banyak alarm palsu yang masih dapat diterima dan berapa banyak High yang minimal harus terdeteksi. Uji pemilihan ambang di dalam fitting berbasis peserta, lalu laporkan precision, recall, dan alarm palsu pada peserta yang ditahan. Jika Medium memerlukan aksi tersendiri, pertahankan tiga label. Gate eye pada percobaan dua tahap belum stabil untuk dijadikan pengendali aksi.

## Menjalankan di Colab

Buka [notebook exp07](EyeHeartCognitiveLoad.ipynb) pada CPU dan pilih **Run all**. Di bawah `DATASET_ROOT = PROJECT_ROOT / 'Dataset'`, sediakan paket `v3_behavioral_proxy/` dan dua CSV di `v2_session_phase/window_15s_step_5s/` dari repo saat ini: [all_data.csv](../../../analysis_sessions/dataset_fusion/output/v2_session_phase/window_15s_step_5s/all_data.csv) untuk kandidat episode dan [good_acceptable.csv](../../../analysis_sessions/dataset_fusion/output/v2_session_phase/window_15s_step_5s/good_acceptable.csv) untuk riwayat. Notebook memeriksa snapshot v2 dengan menoleransi perbedaan akhir baris LF/CRLF. Ekspor Colab exp06 terakhir memakai revisi v2 lebih lama. Gunakan kedua file tertaut untuk mereproduksi exp07.

Hasil dikemas menjadi satu ZIP di `models/batch-06/exp07/outputs/`. Paket memuat prediksi beserta label tiga kelas asli, metrik, matriks kesalahan berdasarkan label asli, cakupan fold, distribusi data, dan manifest input. [ZIP run lokal terverifikasi](outputs/v3_20261124T111830Z_4b184fc1.zip) · [Tabel metrik lengkap](outputs/v3_20261124T111830Z_4b184fc1/metrics.csv)
