# Panduan Hari H: CodeArrange dan Perekaman Sensor

Panduan ini ditujukan untuk operator penelitian yang menjalankan CodeArrange, eye tracking, dan HRV pada satu perangkat. CodeArrange boleh dijalankan lebih dahulu, tetapi pengaitan sesi sensor dilakukan manual sebelum peserta mulai mengerjakan soal.

## Persiapan Awal CodeArrange

1. Pastikan folder set soal yang akan digunakan tersedia di `codearrange-web/puzzles/` dan tentukan urutan set, misalnya easy → medium → hard.
2. Jalankan CodeArrange dari terminal dengan nama peserta dan set awal. Contoh:

   ```bash
   cd codearrange-web
   node server.js "Nama Peserta" demo --review
   ```

   Ganti `demo` dengan set awal yang disepakati. `--review` digunakan jika sesi memakai dialog rating soal.
3. Buka alamat lokal yang ditampilkan server. Halaman awal CodeArrange dapat dibiarkan terbuka sementara sistem sensor disiapkan.

## Menghubungkan sesi sensor

1. Jalankan sistem perekaman multimodal eye tracking dan HRV.
2. Tunggu sampai sistem menampilkan **ID Sesi sensor** dan folder sesi sudah dibuat. Salin ID tersebut persis seperti yang ditampilkan.
3. Di halaman awal CodeArrange, tempel ID pada kolom **ID Sesi Sensor (untuk penelitian)**, lalu tekan **Hubungkan**.
4. Pastikan status halaman menunjukkan bahwa CodeArrange terhubung ke ID sesi yang benar.
5. Setelah terhubung, pilih set soal dari dropdown **Pilih Set Soal**, kemudian tekan **Mulai**.

CodeArrange tidak menghubungkan sesi sensor secara otomatis. Jika peserta mulai sebelum ID sesi dihubungkan, soal tetap dapat dikerjakan dan attempt tetap dicatat, tetapi event penelitian untuk pengerjaan tersebut tidak tercatat ke sesi sensor. Karena itu, operator perlu memastikan status terhubung sebelum menekan **Mulai**.

## Perpindahan set dan jeda

Setelah peserta menyelesaikan satu set, gunakan **Kembali ke Menu Awal**. Di menu awal, pilih set berikutnya lalu tekan **Mulai**. Server tetap berjalan sehingga tidak perlu dihentikan atau dijalankan ulang.

Perekaman sensor berjalan terus selama jeda antarset. Jangan menghentikan sensor saat peserta beristirahat. Jeda setelah set ditutup dan sebelum set berikutnya dimulai akan diberi fase `rest` saat data sensor dianalisis.

Jika muncul pesan bahwa penanda akhir set gagal disimpan, kembali ke menu awal dan tekan **Coba simpan akhir set**. Jangan mulai set berikutnya sebelum penanda berhasil disimpan; server akan menolak pergantian set selama set sebelumnya masih terbuka.

## Mengakhiri sesi

Setelah semua set selesai:

1. Di CodeArrange, tekan **Akhiri sesi penelitian** dan konfirmasi.
2. Pastikan sesi ditandai sudah diakhiri.
3. Hentikan proses perekaman sensor sesuai prosedur sistem sensor. Penghentian sensor akan menjalankan atau memungkinkan proses fusi data sesuai alur multimodal yang digunakan.
4. Periksa folder sesi sensor untuk memastikan file hasil perekaman dan hasil fusi tersedia.

## Arti label fase pada hasil

- `active_task`: peserta sedang menyusun dan mengerjakan soal, sejak soal tampil sampai jawaban dikirim.
- `feedback`: peserta membaca hasil jawaban setelah submit, termasuk benar/salah dan penjelasan.
- `rating`: peserta sedang memberi rating soal. Ini berbeda dari feedback hasil jawaban.
- `rest`: jeda setelah satu set berakhir dan sebelum set berikutnya dimulai.
- `mixed`: window HRV melintasi lebih dari satu fase atau konteks soal/set; rincian durasi tiap bagian tersedia pada `window_phase_mix`.
- `unlabeled`: fase tidak dapat dipastikan, misalnya event penelitian tidak tersedia atau tidak cocok dengan sesi sensor.

Timeline tetap menyimpan seluruh window. Ringkasan pengerjaan per set hanya memakai window HRV berkualitas layak yang seluruh rentangnya berada dalam `active_task`; window campuran, feedback, rating, jeda, dan kualitas sensor yang tidak layak dihitung sebagai pengecualian.

## Pemecahan masalah

**ID sesi sensor tidak ditemukan**

Pastikan sistem sensor sudah membuat folder sesi, ID disalin tanpa salah ketik, dan CodeArrange menggunakan root folder sesi yang benar. Jika folder sesi berada di luar `sessions/`, jalankan server dengan `CODEARRANGE_SESSIONS_ROOT` diarahkan ke folder induk sesi tersebut.

**Status belum menunjukkan terhubung**

Periksa pesan pada halaman, pastikan ID sesi benar dan folder tersedia, lalu tekan **Hubungkan** lagi. Jangan mulai peserta sampai status koneksi berhasil.

**Penanda akhir set gagal disimpan**

Pastikan server CodeArrange masih aktif dan folder sesi sensor masih dapat diakses. Dari menu awal tekan **Coba simpan akhir set**. Jika masih gagal, jangan pilih set baru; catat kendala dan minta bantuan tim teknis agar batas aktivitas tetap jelas.

**Sesi tanpa eye tracking/HRV**

CodeArrange tetap dapat digunakan untuk latihan atau pengambilan data tanpa sensor. Biarkan kolom ID sesi kosong. Attempt dan log CodeArrange tetap tersimpan, tetapi tidak ada event penelitian yang ditautkan ke folder sensor.

**Browser di-refresh saat pengerjaan**

CodeArrange berusaha memulihkan set dan soal dari status sesi yang tersimpan. Setelah refresh, periksa kembali tampilan soal dan status sesi sebelum peserta melanjutkan. Jika sesi set terputus atau penanda akhir belum tersimpan, ikuti pesan pada halaman dan jangan berpindah set sebelum statusnya jelas.
