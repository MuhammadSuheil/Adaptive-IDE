# Rencana eksperimen baseline Batch 06

## Pertanyaan

Apakah normalisasi terhadap referensi awal sensor meningkatkan klasifikasi label protokol Low, Medium, dan High dibanding sepuluh fitur mentah pada data, model, dan fold yang sama?

Eksperimen menguji representasi yang dapat dihitung sebelum prediksi pada sesi baru. Statistik seluruh sesi atau tugas pengguna tidak dipakai sebagai baseline. Pengukuran beban mental independen belum tersedia sehingga hasil tidak memvalidasi pengalaman mental tiap window.

## Input dan acuan

Notebook mount `MyDrive/Penelitian-Adaptive-IDE` dan membaca tiga `good_acceptable.csv` dari `Dataset/v2_session_phase`. Training memakai 25 detik, step 5 detik. Hash dihitung dari file Drive; dataset lokal tidak disalin otomatis atau menggantikan input Drive.

Baseline RMSSD berasal dari recorder dengan status ready, coverage >=80%, nilai positif, dan waktu selesai tidak melewati awal tugas/practice pertama. Bukti waktu disertakan untuk 23 sumber kanonis; nilai dan coverage harus cocok dengan CSV. Sumber yang belum diaudit tidak memperoleh acuan sintetis.

Acuan eye berasal dari window `pre_task` murni paling awal pada CSV 25 detik yang seluruhnya mendahului awal set/practice pertama. Fitur harus memenuhi kontrak core terkurasi. Seluruh window supervised yang tumpang tindih dengan interval referensi dikeluarkan; awal window penilaian minimal satu detik setelah akhir referensi.

SDNN tidak dinormalkan karena baseline independennya tidak ada di CSV. Tidak ada rekonstruksi SDNN dari data tugas atau substitusi angka dari sesi lain. Acuan eye adalah kondisi pre_task, bukan baseline istirahat tervalidasi. Durasi baseline RMSSD recorder lebih panjang daripada window tugas; pengujian ini menilai manfaat representasi relatif dengan keterbatasan tersebut.

## Model dan transformasi

| Model | Konfigurasi tetap |
|---|---|
| Logistic Regression | C=1; max_iter=3000; tanpa class weight |
| Extra Trees | 200 trees; min_samples_leaf=5; max_features=0.8; seed=42; dua thread |
| Dummy | most_frequent dari fitting |

Model di atas dilatih pada representasi raw dan baseline-relative; Dummy hanya satu kali per fold. RobustScaler dihitung dari baris fitting setelah transformasi. Tidak ada tuning, ensemble, balancing, pemilihan seed, atau penambahan identitas sebagai predictor.

## Pembagian dan pemilihan model

Hash tiga CSV yang cocok dengan batch-05 mempertahankan split historis. Pada setiap subset yang layak, evaluasi utama memakai leave-one-participant-out di dalam Train saja. Seluruh rekaman orang yang sama masuk kelompok yang sama. Kontrol sekunder memakai lima fold waktu historis pada Train dengan guard 45 detik. Subset minimal empat orang Train dan fitting tiga kelas; jika syarat gagal, subset dilewati.

Hash yang berbeda menghasilkan split orang baru menggunakan GroupShuffleSplit seed 42 untuk Test 20%, lalu seed 43 untuk validation 25% dari sisanya. Pembagian tidak diulang untuk memilih distribusi atau hasil yang lebih menguntungkan. Baseline yang belum mempunyai bukti sumber tetap dikeluarkan; cakupan yang tersisa dilaporkan.

Seluruh prediksi OOF raw/relative memakai endpoint identik. Pemilihan keluarga model tiap representasi memakai Macro-F1 OOF peserta, accuracy sebagai pembeda, dan syarat recall positif pada ketiga kelas. Model dikunci sebelum validation; validation tidak mengganti pemenang. Tidak ada evaluasi Test atau refit Train + validation.

## Budget dan keluaran

Dengan cakupan lokal saat persiapan, cabang RMSSD mempunyai 19 fold peserta + lima fold waktu, sehingga maksimal 120 fitting CV. Cabang eye mempunyai empat fold peserta + lima fold waktu, sehingga maksimal 45 fitting CV. Maksimal empat fitting Train penuh untuk kandidat validation. Total maksimal 169 fitting pada data historis ini; konfigurasi tidak diperluas setelah melihat skor. Jumlah fold menyesuaikan jumlah orang yang layak pada data Drive.

Plan dan daftar endpoint fold ditulis sebelum fitting. Metrik, prediksi, baseline, dan hash disimpan untuk audit. Tidak ada angka penelitian batch-06 sebelum notebook dijalankan pada dataset Drive. Uji end-to-end memakai fixture sintetis kecil yang terpisah dari data penelitian.

## Kriteria interpretasi

Laporkan accuracy OOF, Macro-F1 OOF, recall tiap kelas, jumlah orang/window, dan jumlah fold yang accuracy-nya naik. Baca manfaat baseline dari delta pasangan raw-relative pada subset/protokol yang sama. Jangan menyimpulkan manfaat dengan membandingkan empat orang pada cabang eye terhadap seluruh dataset batch lain.

OOF memilih model sehingga merupakan hasil development. Fold peserta dapat tidak memuat ketiga kelas dan window tetap overlap di dalam fold. Statistik perubahan deskriptif tidak dianggap uji signifikansi berbasis ribuan window independen. Jika perbaikan hanya terlihat pada empat orang atau tidak bertahan pada validation, hasil dilaporkan sebagai manfaat terbatas, bukan pencapaian target accuracy umum.
