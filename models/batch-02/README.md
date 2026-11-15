# Notebook Batch-02

Paket ini menyiapkan eksperimen sebelum pengambilan tambahan selesai. Notebook belum berisi hasil training. Persetujuan dataset pada `registry/dataset_release.json` masih pending.

## Isi paket

| Folder | Tujuan |
|---|---|
| exp00 | Pemeriksaan dataset, registry, pengesahan release, dan persiapan cache sensor |
| exp01 | Baseline pada window 15/25/45 detik |
| exp02–exp03 | Kontribusi sensor, representasi, normalisasi, dan missing values |
| exp04–exp05 | Perbandingan window, kepadatan training, dan history sensor kausal |
| exp06–exp08 | Seleksi fitur, balancing, dan algoritma/regularisasi |
| exp09–exp10 | Fusion, ensemble, ordinal, dan hierarchy supervised |
| exp11–exp13 | MLP, FT-Transformer, TabPFN-2 lokal, serta GRU kecil |
| exp14 | Seleksi nested, pemilihan window, sensitivitas, dan refit penggunaan |
| shared/batch02 | Implementasi yang dipakai oleh notebook lokal dan Colab |
| tests | Tes teknis tanpa fitting classifier penelitian |
| technical_validation | Catatan pemeriksaan paket, bukan hasil eksperimen |

Detail metode dan parameter terdapat pada [PLAN.md](PLAN.md). Nomor eksperimen adalah keluarga perlakuan, bukan kewajiban menjalankan seluruh kombinasi kartesian.

## Yang dapat dilakukan sekarang

Buka exp00 untuk melihat kondisi dataset dan gate. Biarkan `APPROVE_RELEASE=False` dan `PREPARE_STREAM_CACHE=False` selama data tambahan belum digabung. Jangan mengubah file gate menjadi approved secara manual untuk melewati pemeriksaan.

Cell training pada exp01–exp14 memerlukan release yang sudah disahkan. Penolakan karena gate pending merupakan pengaman, bukan kegagalan model. Tidak ada mode smoke/full yang harus dijalankan bergantian.

## Setelah data tambahan siap

1. Gabungkan sumber dan rebuild v2 memakai prosedur dataset. Tinjau ketiga good_acceptable, proporsi kelas, dan sumber yang lolos kualitas.
2. Buka exp00. Jalankan pemeriksaan, tinjau registry identitas, lalu isi `PERSON_ALIASES` bila beberapa sesi berasal dari orang yang sama. Setelah pemeriksaan operator, set `CONFIRM_IDENTITY_REGISTRY=True`.
3. Isi `ADDITIONAL_SESSION_IDS` dengan ID sesi tambahan yang muncul pada dataset, serta `APPROVED_BY`. Sesudah tim menyetujui proporsi, set `PROPORTIONS_APPROVED=True` dan `APPROVE_RELEASE=True`, lalu jalankan cell pengesahan sekali.
4. Di lokal yang mempunyai sumber mentah dan kode proyek, set `PREPARE_STREAM_CACHE=True` untuk membuat cache aliran sensor. Cache mencakup endpoint yang tidak mempunyai target tunggal dan dicocokkan dengan fitur CSV. Cache ini diperlukan untuk history, centering kausal, thinning, dan sequence.
5. Jalankan notebook eksperimen yang dipilih. Exp14 menjalankan pemilihannya sendiri melalui inner validation; jangan memasukkan pemenang dari outer leaderboard eksperimen lain.

Pengesahan membekukan tiga CSV, allowlist, registry, dan split di `inputs/<release_id>`. Dataset v2 aktif tetap dapat diperbarui pada kesempatan berikutnya, tetapi run yang sudah dimulai menggunakan snapshot. Mengganti data/kode/dependency tidak boleh dilanjutkan sebagai resume run lama.

Cell pengesahan/cache tidak menimpa snapshot yang sudah ada. Saat mengulang exp00 setelah persiapan berhasil, kembalikan kedua flag tersebut ke False. Periksa folder staging jika persiapan cache terputus; jangan menghapus sumber mentah.

## Google Colab dan lokal

Salin folder batch-02 lengkap sekali, termasuk shared, requirements.txt, dan notebook, ke `drive/MyDrive/Penelitian-Adaptive-IDE/models/batch-02`. Sumber dataset berada di `Dataset/v2_session_phase`. Setelah pengesahan lokal, salin juga registry release dan `inputs/<release_id>`, termasuk streams, agar seluruh notebook membaca input yang sama. Sumber mentah tidak perlu diunggah untuk training jika cache telah tersedia.

Notebook mendeteksi root lokal atau Drive. Isi `PROJECT_ROOT` pada cell awal hanya bila folder tidak terdeteksi. `DATASET_ROOT=None` memakai lokasi default; penggantian lokasi dapat dilakukan di cell konfigurasi. Lokasi output adalah `models/batch-02/outputs`.

`DATASET_ROOT` menentukan sumber pemeriksaan/pengesahan exp00. Menggantinya tidak mengganti release aktif pada notebook training; sahkan release baru bila input penelitian berubah. Restart kernel setelah memperbarui paket shared.

Colab memasang dependency inti yang versinya belum cocok. Bila paket yang sudah dimuat berubah, restart runtime sekali dan jalankan dari awal. Lokal memakai environment terpisah; pilih kernel environment batch-02. Runtime harus mendukung wheel dependency yang dikunci; pemeriksaan Windows saat persiapan memakai Python 3.14.6. Runtime Colab belum diuji.

Untuk cabang neural, aktifkan `INSTALL_OPTIONAL=True` pada notebook terkait bila dependency belum tersedia. Versi optional tetap tercatat pada requirements.txt. Ketiadaan GPU tidak menggagalkan jalur CPU: FT-Transformer awal dan TabPFN diberi skipped_hardware. MLP dan GRU juga memerlukan dukungan data/group stopping yang memadai.

TabPFN memerlukan checkpoint V2 lokal, hash yang telah diperiksa, dan persetujuan lisensi pada Config. Paket tidak mengunduh checkpoint atau mengirim data ke hosted inference. Tanpa checkpoint tersebut, cabangnya berstatus skipped_checkpoint. Kode dan bobot TabPFN mempunyai ketentuan lisensi terpisah; baca [sumber resmi](https://github.com/PriorLabs/TabPFN).

## Hasil dan resume

Setiap notebook menyimpan hasil pada `outputs/expXX/<run_id>`. Untuk melanjutkan proses yang terputus, isi `RESUME_DIR` dengan folder run tersebut. Data, kode, dependency, dan konfigurasi harus cocok. Jangan menjalankan dua penulis pada folder run yang sama.

Artefak mencakup identitas run, workload, skor inner, konfigurasi terpilih, prediksi OOF, metrik kelas, confusion matrix, grafik, serta pipeline per-fold. JSON inner cache disimpan lokal dan tidak masuk Git; skor/parameter inner tetap diekspor pada search_results.json. File CSV/JSON laporan dan grafik tetap dapat dipush; bobot, cache, dan snapshot input tidak.

`complete_oof=false` berarti cakupan prediksi belum lengkap. Cabang skipped tidak dihitung sebagai skor nol. `complete_with_skips` menunjukkan ada cabang yang tidak memenuhi syarat; baca events.json atau search_results.json. Skor refit seluruh data bukan skor test baru.

Exp14 memisahkan evaluasi native masing-masing window dari evaluasi prosedur pemilihan window. Hasil yang kedua berada pada joint_window_oof.csv dan joint_window_metrics.json karena panjang window yang terpilih dapat berbeda antar-fold. Model penggunaan tersedia pada deployment; preferred_window.json menjelaskan pilihan atau fallback kompatibilitas 25 detik.

## Pemeriksaan teknis

Dengan environment yang memiliki dependency inti, jalankan:

```bash
python -m unittest discover -s models/batch-02/tests -v
```

Tes memakai data sintetis dan model tiruan untuk memeriksa alur evaluasi/artefak. Tes neural terbatas pada forward pass tanpa backward atau optimizer step. Hasil sementara tes tidak disimpan sebagai output penelitian.

Pemeriksaan paket belum membuktikan fitting classifier, konvergensi, training neural, inference checkpoint TabPFN, atau full nested evaluation akan berhasil pada data final. Catatan yang sudah dan belum diuji tersedia pada [technical_validation/CHECKS.md](technical_validation/CHECKS.md).
