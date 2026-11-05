EDA data coding: mata, HRV, waktu, dan soal

Program membaca `sessions/saved_session` dan menghasilkan grafik PNG, CSV analisis, serta indeks HTML offline. Tidak memerlukan kamera, sensor, server web, atau koneksi internet saat dijalankan. Data sumber tidak ditulis ulang; hash diperiksa sebelum/sesudah proses.

Jalankan dari akar repositori:

```powershell
python analysis_sessions/saved_session_eda/generate_eda.py
```

Buka `analysis_sessions/saved_session_eda/output/index.html` di browser. Pilih peserta untuk grafik dan tabel per soal. Klik grafik untuk membuka gambar penuh. Proses seluruh sesi memerlukan beberapa menit, tergantung perangkat.

Contoh memilih peserta atau folder keluaran:

```powershell
python analysis_sessions/saved_session_eda/generate_eda.py --subjects ikhsan Shakilla Sullivan
python analysis_sessions/saved_session_eda/generate_eda.py --subjects Ramadhan Ramadhan-2 --bin-seconds 2 --dpi 140 --output analysis_sessions/eda_ramadhan
```

`--input` dapat menunjuk folder kumpulan sesi atau satu folder peserta. Nama `--subjects` sama dengan nama folder, termasuk huruf besar/kecil. Output wajib terpisah dari sumber. Menjalankan ulang akan memperbarui berkas keluaran bernama sama; jangan menyimpan berkas kerja manual di folder keluaran. Halaman indeks hanya memuat peserta pada perintah terakhir; grafik lama peserta lain tidak otomatis dihapus.

Dependensi sudah tersedia pada lingkungan pengembangan yang dipakai untuk membuat program ini. Untuk lingkungan baru:

```powershell
python -m pip install -r analysis_sessions/saved_session_eda/requirements.txt
```

Grafik per peserta:

1. Ringkasan waktu: HRV, cakupan, EAR, kedipan, iris, gaze, fase, dan submit.
2. Urutan soal: interval coding aktif, percobaan ulang, submit benar/salah, dan rentang tanpa rekaman mata.
3. Gaze x/y dengan P10–P90, clipping, sampel tidak valid, label Eye_State, dan pose kepala.
4. RR mentah, HRV siap, rasio baseline, serta status HRV.
5. Heatmap prediksi gaze per level.
6. Heatmap setiap soal; otomatis dibagi halaman agar seluruh soal terlihat.
7. Durasi, RMSSD, EAR, iris, kedipan, dan clipping per soal.
8. Distribusi fitur per level.
9. Korelasi Spearman pada jendela aktif dengan HRV siap, beserta jumlah pasangan tersedia.
10. Titik validasi, sampel prediksi, dan kesalahan per titik.
11. FPS, ketersediaan fitur, pasangan RR, dan usia data HRV.
12. Skor/label heuristik tersimpan, dengan penanda bahwa itu bukan ground truth. Jika timeline tidak ada, grafik memberi keterangan kosong.
13. Panel perubahan sensor selama setiap soal: HRV siap, EAR, delta iris, dan kedipan pada waktu lokal soal; otomatis dibagi halaman. Kunjungan ulang tetap mengikuti waktu dan bagian tanpa sensor tidak diisi.

Grafik antarrekaman: kualitas sensor, validasi gaze, pola fitur per level, dan matriks HRV per soal. Dua rekaman Ramadhan tetap ditampilkan terpisah dengan identitas peserta yang sama pada CSV ringkasan.

Folder yang memakai ID sesi asli sama diperiksa agar salinan tidak menggandakan peserta atau statistik. Secara default folder tanpa akhiran ` - Copy` dipilih jika pengukurannya merupakan prefix identik dan tambahan pada salinan hanya fase `post_task`, dengan ledger, RR mentah, dan kalibrasi identik. `session_inventory.csv` mencatat seluruh folder dan alasan pemilihan; varian yang tidak dipilih juga ditampilkan pada indeks HTML. Sumber tidak dihapus. Pilih satu varian memakai `--subjects "Baithar - Copy"` bila ingin memeriksanya secara terpisah. Varian ber-ID sama tetapi pengukurannya berbeda memerlukan pemilihan eksplisit.

CSV utama per peserta:

- `eye_time_bins.csv`: ringkasan frame per bin waktu, default lima detik. Ini mencakup seluruh fase dalam rekaman.
- `aligned_hrv_eye_windows.csv`: sumber HRV tugas dengan label ledger akhir dan agregasi fitur mata pada jendela yang sama.
- `question_summary.csv`: ringkasan soal; pengulangan soal dalam set/level yang sama digabung. Identitasnya adalah `(difficulty_level, puzzle_set, puzzle_id)`.
- `level_summary.csv`: ringkasan per level, tanpa menyamakan frame dengan peserta independen.
- `phase_intervals.csv`: batas fase beserta waktu WIB dan menit relatif.
- `events_for_eda.csv`: kejadian sumber yang cocok dengan ID sesi asli, ditambah waktu relatif.
- `correlation_spearman.csv` dan `correlation_pair_counts.csv`: korelasi serta jumlah pasangan lengkap untuk tiap pasangan fitur.

Aturan pembacaan:

- Nol waktu adalah `both_calibrated_at_utc`; jika tidak ada, frame mata pertama. Waktu negatif menunjukkan kejadian sebelum nol itu. Titik HRV ditempatkan di akhir jendelanya, bukan pengukuran sesaat.
- Grafik diagnostik RR/HRV memperluas sumbu waktu bila RR mentah dimulai sebelum kejadian web pertama. Panel waktu per soal memakai nol lokal soal; EAR kiri berwarna biru dan kanan oranye.
- Folder peserta boleh berbeda dari ID sesi sumber. Program memakai ID asli pada sensor untuk membaca ledger; tidak menjalankan penulis anotasi/fusi lama.
- Label fase dan soal dihitung ulang dari ledger lengkap. Durasi ledger dapat melebihi cakupan sensor; bagian tanpa sensor tetap terlihat.
- HRV selain `ready` tetap kosong, termasuk pada Sullivan; tidak diganti nol. Tidak ada baseline valid/positif berarti rasio tidak tersedia.
- Ringkasan level memakai HRV siap yang seluruh jendelanya berada pada coding aktif level tersebut. Ringkasan soal lebih ketat: seluruh jendela juga harus berada pada satu soal dan satu set. Jendela campuran disimpan di CSV, tetapi dikeluarkan dari statistik soal. Soal cepat dapat tidak memiliki jendela murni 25 detik.
- Kedipan dihitung dari pertambahan penghitung, ditempatkan pada waktu kemunculannya. Laju per soal dibagi durasi coding aktif yang beririsan dengan rentang rekaman mata; tidak memakai rata-rata penghitung kumulatif atau menyamakan laju rolling 60 detik dengan laju jendela 25 detik.
- Persentase kualitas mata berbobot frame. Heatmap menghitung frame, bukan durasi fiksasi; skala warna setiap panel mandiri dan area luar layar ditampilkan.
- `Eye_State`, ukuran iris, gaze, dan keluaran kognitif merupakan estimasi algoritme. Grafik bukan bukti bahwa lokasi tatapan atau kondisi kognitif benar. Ukuran iris bukan diameter pupil terukur.
- Korelasi membutuhkan minimal lima pasangan numerik lengkap per pasangan fitur. Tidak ada p-value karena ketergantungan waktu dan overlap jendela belum dimodelkan. Nilai/kolom konstan atau tidak tersedia tetap kosong.
- Semua HRV memakai jendela 25 detik dengan langkah 5 detik pada arsip ini. Jangan menganggap baris/frame atau kedua sesi Ramadhan sebagai sampel peserta independen.
- EDA tidak menentukan soal yang "paling membebani" sebagai kesimpulan kognitif dan tidak melatih model. Perbandingan ini bersifat deskriptif.

Pemeriksaan otomatis:

```powershell
python -m unittest discover -s analysis_sessions/saved_session_eda -p "test_*.py" -v
```
