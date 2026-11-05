Audit data mata dan HRV — 5 November 2026

Pemeriksaan mencakup **19 rekaman unik dari 18 peserta**, **859.234 frame mata**, dan **6.689 jendela HRV tugas** pada `sessions/saved_session`. Ramadhan dan Ramadhan-2 adalah peserta yang sama. Data dapat digunakan untuk eksperimen ML dengan pemilihan fitur, penyaringan kualitas, dan label yang tepat.

Folder sumber berjumlah 20. `Baithar - Copy` memakai ID sesi yang sama dengan `Baithar`; pengukuran mata/HRV merupakan prefix identik, dengan tambahan hanya 756 frame mata dan 7 jendela HRV pascatugas, sekitar 33 detik. Ledger, RR mentah, dan kalibrasinya identik. Statistik memakai folder `Baithar` satu kali; kedua folder tetap disimpan. `session_inventory.csv` mencatat alasan pemilihan.

Ringkasan JSON Baithar masih mencatat 49.559 frame dan 433 jendela tugas, sedangkan CSV utama berisi 48.803 frame dan 426 jendela. Audit/EDA memakai jumlah aktual CSV. Selisih pada salinan berada di fase pascatugas dan tidak menambah pengukuran coding.

**Kriteria penilaian.** Audit ini memakai **median <450 px dan P95 <900 px** untuk penyaringan awal lokasi tatapan. Batas P95 diperlebar menjadi dua kali batas median sebagai toleransi operasional eksplorasi; ini bukan ambang ilmiah yang telah divalidasi. Kedua syarat harus terpenuhi. Kesalahan sudut tetap dilaporkan sebagai masalah lokal, tetapi tidak menjadi syarat kelulusan penyaringan dua angka ini.

Pengaturan asli yang tersimpan adalah **median ≤180 px, P95 ≤350 px, dan sudut maksimum ≤300 px**. **Baithar dan Fabian lulus aturan asli pada percobaan validasi terakhir; 17 rekaman lainnya gagal dan dilanjutkan manual melalui Space/Enter.** Penilaian audit tidak mengubah `passed_thresholds`, berkas pengukuran, atau konfigurasi program.

| Istilah | Arti dan cara membaca |
|---|---|
| Kualitas kalibrasi | Skor konsistensi model, bukan persentase tatapan benar. Batas awal program 15%; skor lebih tinggi tidak otomatis menjamin lokasi benar. |
| Median | Kesalahan tipikal dari lima titik validasi. **<450 px memenuhi kriteria audit**; ≤300 px masuk kelompok median lebih rendah menurut kriteria tim. |
| P95 | Ukuran mendekati kesalahan terbesar dari lima titik. **<900 px memenuhi kriteria audit**. Median kecil dapat disertai P95 besar. |
| Sudut maksimum | Kesalahan terbesar dari empat titik non-tengah. Menunjukkan area yang bermasalah; batas asli program 300 px tetap menjadi informasi pembanding. |
| Clipping | Prediksi mencapai pembatas koordinat model. Gerakan dapat terpotong dan tampak lebih diam. |

Nilai median/P95 dihitung dari kesalahan rata-rata sepuluh prediksi pada masing-masing titik, bukan seluruh frame coding. P95 tidak berarti 95% frame tugas akurat. Semua konfigurasi layar memakai 2880×1800 px; grid nominal 3×3 mempunyai sel 960×600 px. Kesalahan 450 px masih dapat memindahkan prediksi ke bagian layar lain.

**Hasil validasi dan masalah mata setiap rekaman.** Angka dibulatkan untuk tampilan; penyaringan memakai nilai asli. Persentase clipping di tabel menggunakan prediksi mentah valid sepanjang rekaman. Persentase pada layar menggunakan frame tugas aktif. “Lolos angka” belum berarti seluruh lokasi tatapan dapat dipercaya.

| Rekaman | Median px | P95 px | Sudut maks. px | Median <450 dan P95 <900 | Masalah mata yang ditemukan |
|---|---:|---:|---:|---|---|
| Andrean | 315 | 795 | 324 | Lolos angka | Titik tengah meleset 913 px, tetapi clipping seluruh rekaman relatif rendah 1,9%; 91,1% frame aktif pada layar. |
| Andreas | 182 | 463 | 495 | Lolos angka | Validasi diulang; angka memakai percobaan terakhir. Clipping 17,1% dan hanya 64,1% frame aktif pada layar; kestabilan saat tugas perlu diperiksa. |
| Athaya | 524 | 1320 | 1466 | Tidak | Skor kalibrasi 0%; target kiri atas diprediksi jauh ke kanan-bawah. Posisi tidak memenuhi kriteria audit. |
| Baithar | 166 | 326 | 293 | Lolos angka | Lulus aturan asli, tetapi clipping saat perekaman 29,6% dan hanya 49,4% frame aktif pada layar. Validasi awal belum menjamin kestabilan sepanjang tugas. |
| Fabian | 112 | 190 | 201 | Lolos angka | Lulus aturan asli; clipping 3,2% dan 90,2% frame aktif pada layar. Layak diprioritaskan untuk peninjauan gaze, dengan tetap memeriksa tiap soal. |
| faqih | 201 | 1035 | 222 | Tidak | Median rendah, tetapi titik tengah meleset 1239 px. Area tengah perlu diperiksa khusus. |
| ikhsan | 439 | 586 | 614 | Lolos angka | Titik kiri bawah sekitar 610 px terlalu tinggi; clipping 13,4%. Kondisi kacamata dicatat, penyebabnya belum terbukti. |
| irfan | 390 | 661 | 693 | Lolos angka | Clipping 61,7%; tatapan tidak valid 41,7% sepanjang rekaman; hanya 10,3% frame aktif diklasifikasikan pada layar. Risiko tinggi meskipun lolos angka validasi. |
| kaka | 287 | 747 | 454 | Lolos angka | Titik tengah meleset 820 px; tatapan tidak valid 17,9% sepanjang rekaman. |
| Rafi | 317 | 1276 | 981 | Tidak | Titik tengah meleset 1350 px. Timeline gabungan dan laporan sesi belum tersedia. |
| raina | 238 | 439 | 452 | Lolos angka | Tatapan tidak valid 30,1% sepanjang rekaman; hanya 55,5% frame aktif pada layar. Delta iris aktif tersedia 68,6%. |
| Ramadhan | 336 | 1131 | 447 | Tidak | Titik tengah meleset 1302 px; skor kalibrasi 17,1%; perekaman terputus saat medium. |
| Ramadhan-2 | 394 | 826 | 449 | Lolos angka | Clipping 68,9%; hanya 16,2% frame aktif pada layar. Risiko tinggi meskipun median/P95 memenuhi kriteria. |
| sam | 646 | 1215 | 719 | Tidak | Titik tengah meleset 1339 px; hard hanya mempunyai 9,392 detik rekaman mata. |
| Savitri | 288 | 1125 | 445 | Tidak | Median memenuhi, tetapi titik tengah meleset 1295 px dan P95 melewati batas. Clipping 7,6%; 78,5% frame aktif pada layar. |
| Shakilla | 166 | 820 | 187 | Lolos angka | Titik tengah meleset 978 px meskipun empat titik lainnya sekitar 80–187 px. Hanya medium; kamera berbeda. |
| Siti | 1330 | 2334 | 2458 | Tidak | Kesalahan spasial besar; clipping 72,8%; hanya 12,3% frame aktif pada layar. |
| Sullivan | 487 | 1636 | 1875 | Tidak | Seluruh prediksi validasi target kanan bawah memiliki y=-180; clipping 53,4%; hanya 26,6% frame aktif pada layar. |
| Tio | 386 | 961 | 409 | Tidak | Median memenuhi, tetapi P95 melewati 900 px; titik tengah meleset 1100 px. |

**Sepuluh rekaman lolos penyaringan angka:** Andrean, Andreas, Baithar, Fabian, Ikhsan, Irfan, Kaka, Raina, Ramadhan-2, Shakilla. Untuk percobaan lokasi tatapan pada area besar, prioritaskan peninjauan **Fabian dan Andrean**, kemudian **Ikhsan, Kaka, Raina, dan Shakilla**, dengan masalah lokal tetap dilaporkan. **Andreas dan Baithar memerlukan pemeriksaan kestabilan saat tugas; Irfan dan Ramadhan-2 ditahan dari subset lokasi awal** karena clipping tinggi. Sembilan rekaman lainnya tidak memenuhi salah satu atau kedua batas audit; fitur mata nonspasialnya tetap dapat diperiksa. Urutan peninjauan ini merupakan keputusan operasional, bukan ambang kualitas baru atau sertifikasi lokasi.

**Arti clipping tinggi pada Irfan dan Ramadhan-2.** Program membatasi keluaran koordinat hingga 10% di luar ukuran layar. Pada layar 2880×1800 px, koordinat yang disimpan dibatasi menjadi **x antara -288 dan 3168**, serta **y antara -180 dan 1980**. Misalnya, prediksi x=-350, -700, dan -900 semuanya disimpan sebagai x=-288. Prediksi berbeda menjadi angka yang sama, sehingga gerakan pada arah itu dapat hilang.

| Rekaman | Prediksi mentah valid yang mencapai salah satu pembatas | Arah yang paling sering mencapai pembatas |
|---|---:|---|
| Irfan | 61,7% | **57,8%** seluruh prediksi valid berada di **x=-288**, di kiri layar. |
| Ramadhan-2 | 68,9% | **68,2%** seluruh prediksi valid berada di **y=1980**, di bawah layar. |

Persentase dihitung dari prediksi valid sepanjang rekaman, termasuk fase selain coding aktif. Prediksi valid di sini berarti koordinat tersedia dan diterima alur pencatat, bukan sudah terbukti tepat. Satu prediksi dapat mencapai pembatas x dan y sekaligus, sehingga persentase per arah tidak boleh langsung dijumlahkan.

Artinya, keluaran pemetaan pada kedua rekaman sering mencapai batas yang ditetapkan program. Ini **tidak membuktikan peserta benar-benar melihat ke kiri/bawah selama persentase tersebut**, dan penyebab pastinya belum diketahui. Dampaknya paling langsung pada koordinat, bagian layar, perubahan gaze, dan `Eye_State`: gerakan dapat tampak lebih diam karena terpotong. EAR/kedipan dan HRV tetap diperiksa terpisah; tidak otomatis dibuang bersama koordinat.

Pada percobaan validasi terakhir masing-masing sesi, titik tengah menjadi titik terburuk pada **11 dari 19 rekaman**, dan kesalahannya >700 px pada 12 rekaman. Periksa bersama prosedur menatap titik, waktu tunggu, tampilan validasi, skala layar, posisi kepala, dan model pemetaan. Pola ini belum membuktikan satu penyebab tertentu.

**Penggunaan gaze dan `Eye_State` untuk ML.** Keduanya boleh diuji sebagai fitur. Pergeseran tetap dapat membuat lokasi salah tetapi mempertahankan pola gerakan. Sebaliknya, clipping, jitter, perubahan skala, dan penyaringan dapat mengubah pola tersebut. `Eye_State` dihitung dari koordinat gaze yang sudah dipetakan dan diperhalus, sehingga bukan pengukuran independen.

Kode menandai fiksasi ketika enam sampel terakhir berkumpul dalam radius 50 px, dan sakade ketika perpindahan antarsampel ≥80 px. Koordinat di luar layar masih dapat memperoleh label fiksasi. Pada frame berlabel fiksasi saat tugas aktif, koordinat mentah berada di pembatas sebanyak **77,0% untuk Irfan, 82,3% Ramadhan-2, 84,0% Siti, dan 81,7% Sullivan**. Pada Shakilla angkanya 2,9% dan Faqih 3,8%. Ini indikator risiko distorsi, bukan persentase fiksasi yang pasti salah.

| Fitur mata | Keputusan penggunaan |
|---|---|
| EAR kiri/kanan | Calon fitur pada seluruh 19 rekaman. Tersedia secara numerik pada ≥99,68% frame aktif; ketersediaan bukan bukti akurasi. Saring wajah hilang dan lonjakan. |
| Kedipan | Calon fitur dengan akurasi detektor belum diverifikasi. Gunakan pertambahan kejadian per jendela, bukan penghitung kumulatif. Ambang EAR semua arsip 0,20; batas durasi berbasis frame perlu memperhatikan FPS. |
| Pose kepala | Calon tambahan. Saring nilai saat wajah hilang. Penjaga posisi kepala dinonaktifkan pada seluruh arsip; `head_pose_shifted=0` bukan bukti kepala tidak bergerak. |
| Ukuran/delta iris | Eksperimen terpisah; estimasi landmark iris relatif terhadap lebar mata, bukan diameter pupil. Nilai hilang jangan diganti nol. Pencatatannya ikut bergantung pada gerbang validitas tatapan. |
| Gaze dan `Eye_State` | Uji sebagai tambahan pada model dasar, dengan penanda kualitas dan analisis sensitivitas per peserta. `gaze_x_raw`/`gaze_y_raw` sudah melewati kalibrasi dan pembatas, bukan landmark mentah. |
| Section, dwell, transisi | Gunakan hanya jika pemetaan dan area layar mendukung. Grid statis bernama sidebar/terminal belum membuktikan lokasi elemen CodeArrange. Dwell grid bukan otomatis durasi fiksasi. |
| Confidence | Jangan dianggap peluang tatapan benar; dihitung dari jarak ke pusat sel grid. |

Masalah fitur tambahan saat tugas aktif: sampel EAR >1 ditemukan pada Faqih, Kaka, Raina, Sam, dan Sullivan; tinjau sebagai kemungkinan lonjakan. Delta iris aktif tersedia paling sedikit pada Irfan (56,4%) dan Raina (68,6%). Jumlah kedipan terdeteksi Raina 802 dan Ramadhan-2 91 sepanjang rekaman masing-masing; perbedaan durasi dan detektor harus diperiksa sebelum menghubungkannya dengan keadaan kognitif. Tidak ada penurunan penghitung kedipan, tetapi itu tidak memvalidasi deteksinya. Semua folder merujuk gambar potongan mata yang tidak tersedia, sehingga pemeriksaan visual ulang terbatas.

**HRV dan cakupan tugas.** Gunakan jendela tugas berstatus `ready`: cakupan ≥80%, minimal 10 pasangan interval, usia data ≤3 detik, tidak ada jeda paket >3 detik, dan tidak dalam pemulihan. Persentase siap di bawah mencakup perekaman tugas termasuk jeda/umpan balik. Jumlah easy/medium/hard hanya jendela siap yang seluruhnya berlabel tugas aktif dalam satu set; jendelanya tetap tumpang tindih.

| Rekaman | HRV tugas siap | Jendela coding siap: easy / medium / hard | Keputusan |
|---|---:|---|---|
| Andrean | 98,5% | 25 / 86 / 136 | Simpan HRV siap dan calon fitur mata; ketiga level tersedia. |
| Andreas | 99,5% | 23 / 66 / 91 | Simpan HRV siap; periksa kestabilan gaze saat coding. Demo tidak menambah jendela coding siap. |
| Athaya | 98,2% | 16 / 65 / 91 | Simpan HRV siap dan calon EAR/kedipan; posisi tidak lolos audit. |
| Baithar | 95,5% | 9 / 77 / 99 | Gunakan satu rekaman utama; evaluasi gaze selama tugas meskipun validasi awal lulus. |
| Fabian | 99,8% | 45 / 81 / 122 | Simpan HRV siap; prioritaskan peninjauan fitur gaze karena validasi asli lulus dan clipping rendah. |
| faqih | 96,6% | 29 / 81 / 128 | Simpan HRV siap dan calon EAR/kedipan; batasi interpretasi lokasi. |
| ikhsan | 92,8% | — / 133 / 139 | Simpan medium/hard; gaze area besar untuk eksperimen bersyarat. |
| irfan | 97,1% | 29 / 80 / 109 | Simpan HRV siap dan calon EAR/kedipan; tahan lokasi dari subset awal. |
| kaka | 73,2% | 22 / 63 / 48 | Simpan jendela siap; perhatikan kehilangan HRV pada hard. Ada 9 jendela latihan di luar angka coding. |
| Rafi | 94,9% | 28 / 106 / 143 | Data mentah dapat diproses; timeline gabungan perlu dibuat secara aman. |
| raina | 98,7% | 26 / 45 / 62 | Simpan jendela siap; periksa sampel mata hilang dan detektor kedipan. |
| Ramadhan | 93,7% | 2 / 4 parsial / — | Data parsial; easy selesai, medium terinterupsi. |
| Ramadhan-2 | 96,3% | — / 58 / 78 | Gunakan referensi sendiri; medium adalah pengulangan. Tahan lokasi dari subset awal. |
| sam | 95,2% | 53 / 123 / 0 | Gunakan easy/medium; hard tidak terekam lengkap. |
| Savitri | 99,3% | 7 / 78 / 121 | Simpan HRV siap dan calon EAR/kedipan; P95 gaze tidak memenuhi kriteria audit. |
| Shakilla | 95,5% | — / 88 / — | Medium saja; catat kondisi kamera dan masalah tengah. |
| Siti | 83,1% | 24 / 95 / 92 | Simpan HRV siap dan calon EAR/kedipan; posisi berisiko tinggi. |
| Sullivan | 0,0% | 0 / 0 / 0 | Tidak masuk fusi yang mensyaratkan HRV siap; calon mata saja dan hasil tugas tetap disimpan. |
| Tio | 88,6% | 15 / 75 / 115 | Simpan jendela siap; referensi HRV hanya 61,3% cakupannya. |

Referensi HRV memakai aturan berbeda: mengumpulkan 120 detik RR diterima dalam maksimal 300 detik. Sullivan mencapai sekitar 120 detik dalam 284 detik, cakupan 42,3%; label referensi `ready` tidak berarti rekaman kontinu. HRV tugasnya terdiri dari 137 `gap`, 86 `insufficient_data`, 10 `recovering`. RMSSD/SDNN adalah fitur fisiologis, bukan skor kualitas atau label kognitif. Nilai rendah sendiri tidak menetapkan overload.

**Catatan khusus peserta dari tim dan hasil pemeriksaan.** Informasi kondisi pengujian berasal dari tim; hasil pengukuran di kolom berikutnya berasal dari arsip.

| Peserta | Kondisi yang dilaporkan tim | Temuan dan keputusan penggunaan |
|---|---|---|
| **Ikhsan** | Memakai kacamata. Pada pengujian prototipe, tim sering menemukan hasil mata buruk pada pengguna kacamata. | Median 439 px dan P95 586 px memenuhi penyaringan audit, tetapi target kiri bawah bergeser sekitar 610 px ke atas. EAR aktif tersedia 100%; hanya 68,0% frame aktif diklasifikasikan pada layar. HRV siap 92,8%. Pertahankan EAR/kedipan dan HRV siap; gaze untuk eksperimen bersyarat. CSV tidak membuktikan kacamata sebagai penyebabnya. |
| **Shakilla** | Hanya medium; kamera laptop 30 FPS. Peserta lain memakai kamera eksternal yang ditargetkan 60 FPS/1080p menurut tim. | Medium sekitar 12 menit, enam pengiriman jawaban tercatat salah. Log kameranya melaporkan 640×480 dan tangkapan 29,85 FPS. Median 166 px/P95 820 px memenuhi penyaringan audit, tetapi titik tengah meleset 978 px. Ada 88 jendela coding dengan HRV siap. Gunakan sebagai data medium dan tandai kondisi perangkat. |
| **Athaya** | Kalibrasi dengan skor 0% diterima agar pengujian dapat dilanjutkan; kualitas tatapan diragukan. | Median 524 px/P95 1320 px tidak memenuhi kriteria audit; kesalahan sudut maksimum 1466 px. EAR aktif tetap tersedia 100%, HRV siap 98,2%, dan ketiga level terekam. Gunakan sebagai calon fitur nonspasial + HRV; tahan koordinat dari subset lokasi awal. |
| **Sullivan** | Menyelesaikan tugas dengan cepat. Ringkasan gaze dominan di luar layar terlihat janggal karena kalibrasi dianggap cukup baik saat pengujian. | Skor kalibrasi tersimpan 17,1%; median 487 px/P95 1636 px. Semua prediksi validasi target kanan bawah memiliki y=-180. Ringkasan di luar layar tidak cukup untuk menyatakan ia terdistraksi. Tugas memang selesai: 20 jawaban benar, satu salah. HRV tugas siap 0%; simpan hasil performa dan calon fitur mata nonspasial, keluarkan dari fusi yang mensyaratkan HRV siap. |
| **Ramadhan / Ramadhan-2** | Menurut tim, HRV terputus dan sistem berhenti di tengah pengujian; peserta mengulang mulai medium. | Rekaman pertama memuat easy selesai dan medium parsial, berhenti sekitar 11:40 WIB. Ini konsisten dengan interupsi, tetapi CSV tidak menetapkan penyebab shutdown. Rekaman kedua memuat medium ulang dan hard; HRV siap 96,3%, tetapi clipping koordinat 68,9%. Simpan keduanya sebagai satu peserta dengan dua rekaman, referensi masing-masing 33,42 dan 32,80 ms. Gunakan HRV siap dan calon fitur nonspasial; tandai pengulangan medium dan tahan koordinat rekaman kedua dari subset awal. |

Metadata kamera peserta selain Shakilla melaporkan 2560×1440 dan tangkapan teramati sekitar 28–30 FPS. Angka ini perlu dibedakan dari target perangkat 60 FPS/1080p; laporan driver 60 FPS tidak membuktikan laju aktual 60 FPS. Laju baris tersimpan sekitar 21,3–29,8 FPS. Analisis memakai metadata tiap rekaman, bukan mengasumsikan semua perangkat beroperasi sesuai target.

**Label dan rancangan ML.** Tingkat kesulitan serta kejadian tugas tersedia. Nilai rating numerik tidak ada di `task_events.csv`. Log jawaban yang tersedia berisi 170 baris, tetapi tidak ada kecocokan ID dengan 439 pengiriman jawaban pada sesi yang diperiksa. Label `CONFUSED`, `DISTRACTED`, dan `OVERLOADED` pada timeline dibuat oleh aturan program dari sensor yang sama; memakai label itu melatih peniruan aturan, bukan membuktikan keadaan mental peserta.

Eksperimen yang dapat dilakukan adalah prediksi kondisi tugas `easy/medium/hard`, dengan klaim sesuai target tersebut. Bandingkan **HRV+EAR/kedipan**, lalu tambahan **Eye_State**, **perubahan gaze**, dan **keduanya**, pada pembagian peserta yang sama. Jangan memasukkan tingkat kesulitan atau identitas soal/set sebagai fitur ketika tingkat kesulitan merupakan target.

Semua HRV menggunakan jendela 25 detik dengan langkah 5 detik, sehingga overlap 80%. Ada **3.331 calon jendela coding dengan HRV siap**, di luar 9 jendela latihan, dari 17 peserta. Angka itu belum menyaring kualitas mata tambahan dan bukan jumlah contoh independen. Pisahkan latih/uji berdasarkan peserta; Ramadhan dan Ramadhan-2 selalu satu kelompok. Imputasi, penskalaan, pemilihan fitur, dan penyetelan model dilakukan pada data latih saja. [Dokumentasi validasi silang berdasarkan kelompok](https://scikit-learn.org/stable/modules/cross_validation.html).

**Integritas data.** Seluruh 18 timeline utama yang tersedia cocok dengan HRV dan jumlah frame mata dalam rentang waktunya; Rafi tidak mempunyai timeline sumber. Pasangan CSV mata lengkap/ringkas diperiksa terpisah dari pemilihan folder salinan; jangan menghitung kedua format atau folder varian sebagai contoh independen. Tidak ditemukan waktu mata mundur atau duplikasi waktu/ID tangkapan/ID kejadian pada 19 rekaman utama.

Laporan Ikhsan, Raina, Ramadhan-2, dan Sullivan dibuat sebelum kejadian akhir studi masuk. Status `interrupted` pada laporan tidak selalu berarti perekaman rusak. Penyesuaian ledger akhir hanya mengubah bagian akhir menjadi `post_task`; label kesulitan saat tugas aktif tetap sama. Pemeriksaan hash mencakup 215 CSV/JSON arsip sesi, termasuk folder varian, serta satu log jawaban tambahan; sumber tidak berubah selama analisis.

**Perbaikan untuk pengambilan data berikutnya.** Softbox dan kamera cepat membantu memperoleh gambar, tetapi tidak menjamin pemetaan gambar mata ke lokasi layar benar. FPS mengukur frekuensi pengamatan; akurasi mengukur seberapa dekat prediksi dengan target. Prediksi dapat cepat dan halus tetapi tetap meleset. [Penjelasan akurasi dan presisi dari Tobii](https://help.tobii.com/hc/en-us/articles/213534825-Let-s-talk-accuracy-and-precision).

Belum ada bukti yang menetapkan satu penyebab seluruh kesalahan. Temuan paling konkret adalah **input pemrosesan 320×240 pada semua arsip, penjaga posisi kepala dinonaktifkan, dan validasi yang selalu dimulai dari tengah dengan jeda 0,6 detik serta sepuluh sampel per titik**. Prioritaskan pemeriksaan alur pengukuran ini sebelum menambah kecerahan lampu atau membeli kamera.

| Prioritas | Temuan atau dugaan yang harus dibedakan | Tindakan konkret tim |
|---|---|---|
| **1. Periksa validasi titik tengah** | Tengah menjadi titik terburuk pada 11/19 rekaman dan selalu diuji pertama. Penyebab waktu adaptasi belum terbukti. | Pada uji internal, acak urutan titik dan ulangi tengah di akhir. Bandingkan jeda 0,6 detik dengan 1,5–2 detik serta pengambilan sampel stabil selama 1–2 detik. Angka ini rancangan percobaan, bukan jaminan perbaikan. Simpan waktu target tampil, waktu/ID frame, serta sebaran setiap titik. |
| **2. Kendalikan posisi kepala** | `head_positioning.enabled` bernilai false pada seluruh arsip. Pemetaan saat ini memakai fitur mata dan EAR, tanpa pose kepala sebagai masukan mapper. | Uji penjaga posisi kepala sebelum dipakai pada peserta. Tetapkan posisi kamera, jarak duduk, dan tinggi kursi yang nyaman serta konsisten. Ambil referensi kepala aktual; beri peringatan saat berubah besar dan periksa ulang kalibrasi. Penjaga posisi bukan kompensasi gerakan otomatis. |
| **3. Periksa detail dan bentuk gambar yang diproses** | Frame kamera diperkecil menjadi 320×240 sebelum landmark dihitung. Resolusi kamera tinggi tidak berarti model menerima gambar sebesar itu. | Simpan contoh frame asli dan input pemrosesan. Bandingkan dengan input lebih besar sambil mengukur error dan FPS. Pertahankan rasio gambar: misalnya 640×360 untuk frame 16:9 atau 640×480 untuk 4:3. Resize langsung dari 16:9 ke 4:3 mengubah bentuk gambar; pengaruhnya perlu diuji, belum terbukti menjadi penyebab. |
| **4. Cocokkan koordinat layar** | Konfigurasi memakai ukuran manual 2880×1800. Kecocokan dengan tampilan fisik belum diverifikasi dari arsip. | Catat monitor aktif, resolusi, skala Windows/DPI, ukuran jendela fullscreen, dan zoom browser. Periksa posisi titik uji terhadap koordinat yang dicatat. Pastikan transformasi dan pembalikan kamera konsisten antara kalibrasi, validasi, dan perekaman. |
| **5. Periksa softbox pada gambar mata** | Tim melaporkan pencahayaan terang, tetapi potongan mata tidak tersedia untuk menilai pantulan dan ketajaman. | Periksa kedua mata pada frame asli, terutama pengguna kacamata. Uji sudut softbox agar area iris tidak tertutup pantulan atau terlalu putih. Periksa fokus dan gerakan kabur. Bila perangkat mendukung, uji exposure/gain/fokus yang stabil setelah setelan sesuai; jangan menganggap makin terang selalu makin baik. |
| **6. Ukur kamera yang benar-benar diterima** | Driver melaporkan 60 FPS, tetapi log menunjukkan sekitar 28–30 FPS tangkapan. Sebagian permintaan resolusi/FourCC juga tidak dipenuhi backend. | Catat ukuran frame aktual, format kamera, backend, FPS tangkapan dan pemrosesan, keterlambatan frame, serta frame terlewat. Gunakan mode yang terbukti stabil pada uji internal; peningkatan FPS tidak boleh mengorbankan detail mata atau menambah error posisi. |
| **7. Pantau kestabilan selama coding** | Irfan dan Ramadhan-2 lolos angka validasi audit tetapi sering mencapai pembatas saat tugas. | Tambahkan pemeriksaan titik singkat sebelum setiap level dan sesudah perubahan posisi. Catat sebagai fase pemeriksaan tersendiri agar tidak masuk label coding. Tampilkan status kualitas per jendela; jangan hanya melihat ringkasan akhir. Saat menatap titik di dalam layar, prediksi yang terus berada di pembatas harus diperiksa sebelum melanjutkan penggunaan spasial. |
| **8. Perkuat HRV dan pemulihan koneksi** | Sullivan tidak punya HRV tugas siap. Kode perekam berhenti ketika sensor terputus; orkestrator mengakhiri sesi saat salah satu proses keluar. | Uji HRV saat mengetik, bukan hanya duduk diam. Pantau penerimaan RR, cakupan, pasangan interval, dan usia data. Tambahkan penanganan putus-sambung atau pencatatan interupsi yang jelas, simpan data parsial, dan pertahankan pemisahan segmen/baseline. Jangan menyambung jeda seolah perekaman kontinu. |
| **9. Lengkapi arsip dan label** | Potongan mata hilang, beberapa kejadian akhir datang setelah laporan, dan log rating yang cocok belum tersedia. | Arsipkan potongan mata/frame uji sesuai persetujuan peserta, snapshot konfigurasi, versi kode, metadata kamera/monitor, log jawaban dan rating dengan ID yang cocok. Akhiri studi di web sebelum menutup perekam, tunggu kejadian tersimpan, lalu buat laporan. |

**Uji internal sebelum merekrut peserta berikutnya:** gunakan 2–3 anggota tim, sertakan kondisi berkacamata. Ubah satu faktor setiap percobaan dan pertahankan faktor lain. Nilai semua titik, jitter, clipping saat target berada di layar, kelengkapan HRV, serta kestabilan sesudah beberapa menit mengetik. Jumlah ini untuk mencari masalah teknis, bukan validasi ilmiah atau penentuan ukuran sampel penelitian.

Untuk masalah tengah, interpretasi percobaan harus jelas: jika kesalahan berpindah mengikuti titik yang diuji pertama, prioritaskan waktu adaptasi/prosedur awal; jika tengah tetap buruk di urutan mana pun, prioritaskan pemetaan dan koordinat lokal. Jika hasil memburuk setelah kepala bergeser, prioritaskan kestabilan atau kompensasi pose. Jika input lebih besar tidak memperbaiki error, jangan mempertahankannya hanya karena resolusinya lebih tinggi.

Tetapkan aturan pengambilan data sebelum sesi utama. Median <450/P95 <900 tetap merupakan penyaringan eksploratif; periksa juga tiap titik dan kestabilan saat tugas. Catat keputusan operator jika tetap melanjutkan dan tandai fitur mana yang boleh dianalisis. Perubahan lampu, kamera, kacamata, resolusi, transformasi gambar, atau posisi perangkat memerlukan pemeriksaan ulang. Ini saran prosedur/perangkat lunak; konfigurasi dan kode belum diubah oleh audit ini.

**Berkas pendukung.** `session_inventory.csv` memuat pemilihan 19 rekaman dari 20 folder; `subject_summary.csv` dan `validation_targets.csv` memuat angka kalibrasi/validasi; `task_sets.csv` memuat cakupan tugas; `ml_eye_feature_summary.csv`, `ml_eye_features_by_phase.csv`, dan `ml_eye_window_quality.csv` memuat ketersediaan fitur; `gaze_eye_state_clipping_review.csv` memuat hubungan clipping dengan label fiksasi. `integrity_checks.csv` dan manifest hash mendokumentasikan konsistensi sumber. Berkas kualitas tersebut belum merupakan dataset pelatihan berlabel yang sudah dibersihkan.

Grafik EDA waktu/soal dapat dibuat dengan `python analysis_sessions/saved_session_eda/generate_eda.py`. Buka `analysis_sessions/saved_session_eda/output/index.html` untuk memilih peserta, grafik tiap soal, dan perbandingan antarsesi. Panduan tersedia di `analysis_sessions/saved_session_eda/README.md`; keluaran dibuat terpisah dari data mentah.
