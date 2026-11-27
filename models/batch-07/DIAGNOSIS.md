# Diagnosis Hasil Batch 07

Tanggal pemeriksaan: 27 November 2026.

Dokumen ini menilai ulang klaim [SUMMARY.md](SUMMARY.md) dari prediksi episode yang tersimpan di keempat folder `outputs/`. Tidak ada training ulang, perubahan dataset, atau perubahan notebook. Semua angka baru dihitung oleh [diagnosis/audit_batch07.py](diagnosis/audit_batch07.py) dan disimpan di [diagnosis/diagnosis_checks.json](diagnosis/diagnosis_checks.json).

## Kesimpulan

Batch 07 adalah batch dengan protokol evaluasi paling sehat sejauh ini. Pemisahan per peserta (LOPO), label V3 yang dihitung ulang per fold, ambang yang dipilih di dalam fold, dan riwayat sensor yang bebas kurasi global adalah kemajuan nyata dibanding batch 01–05. Namun **angka utama yang dilaporkan menyesatkan**, dan kedua jalur label tidak sama kuatnya:

1. **Akurasi 77,8%, 81,5%, dan 72,2% berada di bawah atau sama dengan tebakan "selalu Non-Overload/Non-High".** Proporsi kelas mayoritas adalah 81,7% (V4 LOPO), 81,5% (Test V4), dan 88,9% (Test V3). Akurasi tidak boleh menjadi metrik utama untuk target yang positifnya hanya 18% episode.
2. **V4 (rating 7–10) memiliki sinyal sensor yang nyata tetapi sedang.** AUC LOPO 17 peserta 0,778 (CI 95% bootstrap peserta 0,63–0,89), balanced accuracy 0,75–0,76. Skor sensor tetap menambah informasi setelah dikontrol tingkat soal, durasi, dan posisi waktu dalam sesi (p ≈ 0,0001). Ini temuan terkuat seluruh penelitian.
3. **Model V3 tidak dapat dibedakan dari detektor durasi episode.** Skor SVM berkorelasi ρ = 0,63 dengan durasi bahkan di dalam episode yang labelnya sama-sama Non-High. Skor itu juga tidak menambah informasi apa pun di atas durasi (p = 0,79). Di Test historis, AUC tepat 0,50 dan hanya 1 dari 8 pasangan dalam-peserta terurut benar.
4. **Pemilihan kandidat dan kebijakan ambang dilakukan setelah melihat Full LOPO dan Test.** Kebijakan `nested_fa_cap25_max_rec` hanya peringkat ke-8 pada Dev (balanced accuracy 0,636 dibanding 0,753 untuk `fixed_050`). Kandidat `full14` untuk headline Test juga bukan pemenang Dev.
5. **Test historis tidak informatif.** Test berisi 4 peserta dengan 5 episode overload (V4), atau 3 peserta dengan 2 episode High (V3). Data ini sudah dilihat sejak batch 06. CI 95% balanced accuracy Test V4 adalah 0,50–0,83, dan CI AUC 0,25–0,92.
6. **Sebagian sinyal V4 berasal dari perbedaan antar-peserta, bukan perubahan kondisi dalam diri peserta.** AUC rata-rata di dalam peserta 0,645, sedangkan AUC gabungan 0,778. Rata-rata skor per orang berkorelasi ρ = 0,54 dengan proporsi overload orang tersebut.
7. **Kedua model terbaik bekerja per episode setelah tugas selesai**, sedangkan IDE adaptif perlu memutuskan selama tugas berjalan. Batch 07 belum mengukur deteksi dini.

**Penilaian keseluruhan:** Batch 07 sudah layak menjadi dasar model final, tetapi hanya jalur V4. Targetnya pun perlu dinyatakan dengan AUC dan balanced accuracy yang realistis (sekitar 0,75–0,80 dan 0,70–0,76), bukan akurasi 70–80%. Batch 08 tidak perlu mencari model baru yang lebih kompleks. Yang dibutuhkan adalah **satu protokol yang dibekukan sepenuhnya**, evaluasi jujur terhadap pembanding non-sensor, dan formulasi deteksi yang sesuai penggunaan di IDE.

## Cakupan pemeriksaan

| Eksperimen | Artefak yang diperiksa | Unit |
|---|---|---|
| Exp01 | `models_20261125T124030Z_fe00ac84` | 37 episode LOPO Train V3, 11 validasi |
| Exp02 | `exp02_v3_20261125T152650Z_6705f25b` | 1.584 baris prediksi (2 sumber riwayat × 2 representasi × 3 model × 3 kebijakan) |
| Exp03 | `exp03_v4_20261125T153235Z_d3e8d715` | 7.560 baris prediksi (3 protokol × 6 kandidat × 5 kebijakan) |
| Exp04 | `exp04_unified_20261125T152940Z_ebddb53d` | 640 baris prediksi pada 40 episode irisan |

Kode eksperimen (`experiment.py`, `build_notebook.py`) juga dibaca untuk memastikan pemisahan peserta, penghitungan ulang label V3 per fold, pembobotan, dan pemilihan ambang. **Tidak ditemukan kebocoran peserta pada tingkat fitting**. Kebocoran yang ditemukan berada pada tingkat pemilihan konfigurasi dan pelaporan, bukan di dalam kode fold.

Metrik dihitung ulang dengan numpy/scipy (AUC Mann–Whitney, balanced accuracy, bootstrap per peserta 1.000 ulangan, uji likelihood-ratio logistik satu derajat bebas). Nilainya cocok dengan `metrics.csv` masing-masing run.

## 1. Akurasi lebih rendah daripada tebakan kelas mayoritas

| Klaim | Akurasi dilaporkan | Tebakan selalu negatif | Balanced accuracy | Keterangan |
|---|---:|---:|---:|---|
| V4 Full LOPO, `core9`, `nested_fa_cap25` | 77,8% (98/126) | **81,7%** (103/126) | 0,763 (CI 0,63–0,87) | Model lebih buruk dari tebakan konstan dalam akurasi |
| V4 Full LOPO, `full14`, `nested_youden` | 81,0% | **81,7%** | 0,732 | Sama |
| V4 Test historis, `full14`, `nested_fa_cap25` | 81,5% (22/27) | **81,5%** (22/27) | 0,732 (CI 0,50–0,83) | Identik dengan tebakan konstan |
| V3 Test historis, `svm_rbf`, `nested_recall75` | 72,2% (13/18) | **88,9%** (16/18) | 0,625 | 1 dari 2 High terdeteksi, 4 alarm palsu |
| Exp04 `V4_LogReg_Core9`, target V3 High / V4 ≥ 7 (40 episode) | 57,5% / 57,5% | 72,5% / 77,5% | 0,679 / 0,726 | Sama |

Kedua angka pada video monev (72,22% dan 81,48%) berasal dari baris di atas. Keduanya **tidak menunjukkan keunggulan atas tebakan tanpa sensor**. SUMMARY juga menyebut konfigurasi V4 "melampaui target akurasi 74%–78%". Kalimat ini perlu dikoreksi karena target akurasi tidak bermakna ketika kelas mayoritas sudah 82%.

Metrik yang bermakna untuk target ini adalah **ROC AUC, PR-AUC, balanced accuracy, dan recall pada batas alarm palsu tertentu**, disertai CI per peserta. Balanced accuracy Batch 07 (0,73–0,76 pada LOPO V4) memang lebih tinggi daripada semua batch sebelumnya. Inilah angka yang layak dilaporkan.

## 2. Pemilihan setelah melihat hasil (*selection after the fact*)

Exp03 menghasilkan 6 kandidat × 5 kebijakan ambang × 3 protokol = 90 baris hasil. Kode mengunci kandidat terbaik Dev berdasarkan AUC, tetapi headline README/SUMMARY memilih kombinasi lain:

| Peringkat Dev (balanced accuracy, 13 peserta) | Nilai |
|---|---:|
| 1. `win_logreg_core9_global` · `nested_youden` | 0,753 |
| 1. `win_logreg_core9_global` · `fixed_050` | 0,753 |
| 3. `win_logreg_core9_global` · `nested_recall75_min_fa` | 0,738 |
| … | … |
| 8. `win_logreg_core9_global` · **`nested_fa_cap25_max_rec`** (dijadikan headline) | **0,636** |

- **Kebijakan headline tidak dipilih dari Dev.** `nested_fa_cap25` baru tampak unggul di Full LOPO 17 peserta. Full LOPO memuat keempat peserta Test, jadi Full LOPO dan Test sama-sama ikut menentukan pilihan.
- **Kandidat headline Test berbeda dari pemenang Dev.** `full14` dipakai untuk angka 81,5%, padahal `core9` peringkat satu di Dev (AUC 0,739 vs 0,729). Dengan `core9` dan kebijakan yang sama, akurasi Test hanya 59,3% (balanced accuracy 0,673).
- **Exp02 V3 berpola sama.** Delapan belas kombinasi dievaluasi di Test historis, lalu angka 72,2% dipilih dari `svm_rbf · center_only · nested_recall75`. Pada kombinasi lain dengan riwayat bersih, akurasi Test berkisar 50–72%.

Ambang di dalam fold memang dipilih tanpa melihat peserta evaluasi. **Kebocorannya terletak satu tingkat di atasnya**: memilih kebijakan dan kandidat mana yang dilaporkan. Dengan 37–126 episode, memilih yang terbaik dari puluhan kombinasi memberi bias optimistis yang besarnya sebanding dengan selisih antar-kombinasi itu sendiri.

## 3. Test historis terlalu kecil dan sudah terpakai

| Test | Peserta | Episode | Positif | CI 95% AUC (bootstrap peserta) |
|---|---:|---:|---:|---|
| V4 `core9` | 4 | 27 | 5 | 0,375–0,923 |
| V4 `full14` | 4 | 27 | 5 | 0,250–0,923 |
| V3 | 3 | 18 | 2 | Tidak stabil; hanya 2 orang mempunyai kedua kelas |

Satu episode positif menggeser recall Test V4 sebesar 20 poin persentase, dan Test V3 sebesar 50 poin. Peserta Test V3 dan V4 sudah dievaluasi pada batch 06 (exp01–exp03 dan laporan v3), jadi bagian ini bukan lagi uji independen. **Estimasi terbaik yang tersedia adalah LOPO pada seluruh peserta**, dengan syarat seluruh pemilihan model berlangsung di dalam fold (nested). Test historis cukup dilaporkan sebagai lampiran.

## 4. V3: skor sensor mengikuti durasi episode

Label V3 didefinisikan dari durasi aktif relatif terhadap peserta lain dan jumlah kesalahan. Durasi saja sudah memberi AUC **0,976** terhadap label tersebut. Masalahnya: skor sensor ternyata juga mengikuti durasi.

| Pemeriksaan (LOPO Train, `svm_rbf · center_only`, 37 episode) | Hasil |
|---|---:|
| Spearman skor vs durasi aktif, semua episode | ρ = 0,598 |
| Spearman skor vs durasi, **hanya episode Non-High** | **ρ = 0,627 (p = 0,002)** |
| Spearman skor vs durasi, hanya episode High | ρ = 0,306 |
| Tambahan informasi skor di atas log durasi | χ² = 0,07, **p = 0,79** |
| Tambahan informasi skor di atas tingkat soal | χ² = 4,43, p = 0,035 |

Jika model menangkap beban kognitif, skornya seharusnya tidak naik bersama durasi pada episode yang labelnya sama. Penyebab yang paling mungkin terletak pada representasi:

- Fitur episode adalah **median seluruh window dikurangi median riwayat sebelum soal dimulai**.
- Sinyal yang bergeser pelan selama sesi (EAR menurun karena kelelahan, HRV berubah karena duduk lama) membuat selisih tersebut membesar sesuai panjang episode.
- Episode panjang memang cenderung berlabel High. Jadi model dapat "menebak durasi" dari besarnya pergeseran tanpa mengukur beban.

Ini berbeda dari kebocoran IQR/slope pada batch 06, tetapi mekanismenya serupa: **durasi episode bocor melalui representasi fitur**. Di Test historis, AUC SVM `center_only` tepat 0,500 dan urutan dalam-peserta hanya 1/8 pasangan benar. Klaim "87,5% ketepatan urutan dalam-peserta" pada LOPO berasal dari 5 orang dan 16 pasangan, dan tidak bertahan di Test.

Masalah tambahan V3:

- **Ukuran sampel:** hanya 37 episode berlabel dari 15 peserta di Train. Sebanyak 8 episode LOPO tidak memiliki minimal tiga window riwayat, sehingga memakai fallback "dikurangi median fitting". Representasinya jadi berbeda dari episode lain.
- **Cakupan label:** 276 dari 358 episode tidak berlabel karena rubrik tiga kelas menolak bukti yang bertentangan.
- **Kegunaan untuk IDE:** durasi dan jumlah kesalahan sudah dapat dibaca IDE tanpa sensor. Model sensor yang hanya mereproduksi durasi tidak menambah nilai.

**Diagnosis:** dalam bentuknya sekarang, V3 tidak layak menjadi target utama model sensor. V3 tetap berguna sebagai validasi silang perilaku untuk skor V4.

## 5. V4: sinyal nyata di atas konteks, tetapi sebagian bersifat antar-peserta

### 5.1 Pembanding non-sensor

| Prediktor tunggal tanpa sensor (Full LOPO, 126 episode) | AUC |
|---|---:|
| Waktu sejak episode pertama sesi | 0,819 |
| Durasi aktif episode | 0,816 |
| Jumlah window episode | 0,808 |
| Tingkat set soal (easy/medium/hard) | 0,776 |
| Pernah timeout | 0,736 |
| **Model sensor `win_logreg_core9_global`** | **0,778** |

Skor sensor sendirian **tidak lebih baik** daripada konteks tugas sederhana. Overload terkonsentrasi pada soal hard yang selalu dikerjakan terakhir (11/18 episode hard ber-rating ≥ 7, dibanding 3/55 easy) dan pada episode timeout (14/28 overload, dibanding 9/98 tanpa timeout). Namun skor sensor **membawa informasi tambahan**:

| Model logistik | AUC dalam sampel | Uji penambahan skor sensor |
|---|---:|---|
| Tingkat soal | 0,776 → **0,866** | χ² = 14,6, p = 0,0001 |
| Tingkat soal + log durasi + waktu dalam sesi | 0,884 → **0,927** | χ² = 14,9, p = 0,0001 |

Di dalam kelas yang sama, skor V4 tidak berkorelasi dengan durasi (ρ = 0,06 Non-Overload; 0,02 Overload). Di dalam tiap tingkat soal, AUC tetap 0,71 (medium), 0,77 (hard), dan 0,94 (easy, hanya 3 positif). Jadi, berbeda dari V3, **V4 bukan detektor durasi**. Ini bukti terkuat bahwa fitur EAR, kedipan, iris, dan HRV memuat informasi yang terkait dengan persepsi beban.

### 5.2 Sinyal antar-peserta (*trait*) vs perubahan dalam diri (*state*)

| Protokol `core9` | AUC gabungan | AUC rata-rata dalam peserta | Peserta dengan kedua kelas | Pasangan dalam-peserta benar |
|---|---:|---:|---:|---:|
| Dev LOPO 13p | 0,739 | 0,609 | 7 | 64,3% dari 70 |
| Full LOPO 17p | 0,778 | 0,645 | 10 | 69,2% dari 104 |

Sebagian kemampuan membedakan berasal dari perbedaan level dasar antar-orang. Orang yang rata-rata skornya tinggi juga cenderung sering memberi rating ≥ 7 (ρ = 0,54). Bukti dari batch sebelumnya konsisten dengan ini: normalisasi `unsupervised_session_robust` menurunkan AUC ke sekitar 0,42–0,48 (batch 06 exp03), dan `causal_session_robust` di batch 07 hanya 0,61–0,65. Konsekuensinya bagi IDE: model akan cenderung "selalu waspada" pada sebagian pengguna dan "selalu tenang" pada pengguna lain. Dalam data, Andreas (5 dari 8 episode overload) mendapat rata-rata skor 0,73, sedangkan Savitri (0 dari 11) 0,12. Pada tingkat individu, AUC dalam peserta sangat bervariasi: 0,00 (kaka), 0,33 (Andreas), 0,36 (faqih), sampai 0,89–0,93 (Fabian, Rafi, Baithar).

### 5.3 Sampel efektif

- Hanya 23 episode overload, dari 12 orang. Andreas menyumbang 5, sam dan Fabian masing-masing 3.
- Tujuh dari 17 peserta hanya memiliki satu kelas, sehingga tidak dapat dinilai di dalam peserta.
- Ambang terpilih per fold berubah cukup besar. Contohnya `full14 · nested_macro_f1` berkisar 0,625–0,800, dan `nested_youden` 0,400–0,750 pada Full LOPO. Nilai ambang "0,58–0,62" di rekomendasi SUMMARY belum stabil antar-subsampel peserta.

## 6. Formulasi belum sesuai penggunaan IDE

Kedua model yang direkomendasikan menghasilkan satu keputusan **setelah episode selesai**: median seluruh window (V3) atau rata-rata probabilitas seluruh window (V4). Untuk IDE adaptif, bantuan berguna justru saat pengguna sedang kesulitan. Dua fakta memperberat ini:

- Episode overload adalah episode panjang (durasi saja AUC 0,82). Sebelum episode selesai, sebagian besar informasi pembeda belum tersedia.
- Waktu yang sudah berjalan pada tugas saat ini **dapat** diketahui IDE secara real-time. Pembanding yang jujur untuk deteksi dini adalah "durasi berjalan + sensor" vs "durasi berjalan saja".

Model per-window V4 sudah bisa dipakai secara daring (rata-rata berjalan), tetapi Batch 07 belum mengukur performanya pada 30/60/120 detik pertama episode.

## 7. Exp04: konkordansi bukan validasi konstruk independen

Kesepakatan 85% (34/40) antara label perilaku V3 dan rating V4 wajar karena keduanya digerakkan oleh durasi dan kegagalan. Rating berkorelasi ρ = 0,63 dengan durasi dan 0,64 dengan jumlah kesalahan, dan rating ditanyakan tepat setelah peserta berhasil atau timeout. Kesepakatan ini menunjukkan kedua label mengukur "soal yang sulit bagi peserta". Ia belum membuktikan bahwa keduanya mengukur beban kognitif fisiologis.

Pada 40 episode irisan, CI 95% AUC hampir semua model mencakup 0,5 atau sangat lebar (contoh `Dual_V3SVM_V4LR` 0,81 dengan CI 0,43–1,00). Model gabungan "dual-trigger" belum mempunyai dukungan statistik yang cukup. Temuan exp04 yang paling bisa dipegang: **`V4_LogReg_Core9` satu-satunya model yang CI AUC-nya berada di atas 0,5 pada keempat definisi target**. Ini konsisten dengan bagian 5.

## 8. Temuan yang perlu dipertahankan

| Temuan | Bukti | Keputusan untuk Batch 08 |
|---|---|---|
| LOPO per peserta + pemilihan ambang di dalam fold | Kode exp03 `run_lopo_with_nested_thresholds` | Pertahankan, lalu perluas ke pemilihan kandidat dan kebijakan |
| Logistik L2 per-window, bobot setara-episode, rata-rata per episode | Unggul konsisten di Dev, Full LOPO, dan exp04 | Model utama |
| `core9` (tanpa `iris_size_delta_mean`) | Setara atau lebih baik daripada `full14` di Dev; redundansi iris terbukti di batch 05 | Fitur utama; `full14` sebagai ablasi |
| Riwayat dari `all_data.csv` (kualitas saja, bukan kurasi seimbang global) | Exp02: kurasi global memengaruhi hasil validasi exp13 | Wajib bila memakai riwayat |
| Penghitungan ulang label V3 per fold | Kode `fold_labels` | Wajib bila V3 dipakai |

## 9. Hal yang tidak terbukti

- **Tidak terbukti bahwa sensor tidak berguna.** Uji di bagian 5.1 menunjukkan kontribusi signifikan di atas konteks tugas.
- **Tidak terbukti bahwa target AUC di atas 0,80 mustahil.** Dengan 17 peserta dan 23 positif, CI masih ±0,13. Data saat ini hanya tidak cukup untuk membedakan 0,75 dari 0,85.
- **Detak jantung rata-rata bukan fitur yang bisa langsung ditambahkan.** Perkiraan HR dari `hrv_rr_count` memberi AUC univariat sekitar 0,70 (arah terbalik, HR lebih rendah pada overload), tetapi sinyal ini hilang setelah dikontrol tingkat soal dan waktu dalam sesi (p = 0,77). HR menurun sepanjang sesi (ρ = −0,37 dengan menit sesi). Fitur apa pun yang bergeser seiring waktu akan tampak prediktif karena soal hard selalu terakhir. Uji kontrol waktu-sesi wajib untuk setiap fitur baru.
- **Konfigurasi terbaik Batch 07 belum terbukti terbaik.** Selisih AUC antar-kandidat teratas (0,73–0,79) lebih kecil daripada lebar CI masing-masing.

## 10. Prioritas Batch 08 (batch terakhir)

Batch 08 sebaiknya menjadi **konfirmasi terkunci**, bukan eksplorasi. Semua keputusan di bawah ditulis di PLAN dan dibekukan sebelum notebook dijalankan.

### 10.1 Protokol evaluasi tunggal

1. **Target utama:** V4 overload, rating ≥ 7 vs 1–6, per episode. V3 High menjadi target sekunder/validasi silang saja.
2. **Evaluasi utama:** nested LOPO pada **seluruh 17 peserta V4**. Pemilihan kandidat, C, dan kebijakan ambang dilakukan di inner-LOPO setiap outer fold. Hanya prediksi outer yang dilaporkan. Tidak ada lagi "Test historis" sebagai angka utama.
3. **Metrik utama yang ditetapkan di awal:** ROC AUC episode dan balanced accuracy pada satu kebijakan ambang yang dipilih di inner fold. Metrik pendamping: PR-AUC (baseline 0,18), recall pada alarm palsu ≤ 25%, AUC dalam peserta, dan presisi overload. Semua disertai CI bootstrap per peserta dan uji permutasi label di dalam fold (minimal 1.000 permutasi) untuk p-value terhadap peluang acak.
4. **Pembanding wajib pada fold yang sama:** (a) tebakan mayoritas; (b) konteks tanpa sensor, yaitu durasi berjalan, waktu dalam sesi, dan timeout; (c) konteks + sensor. Klaim kontribusi sensor didasarkan pada (c) vs (b), bukan sensor vs nol.

### 10.2 Eksperimen yang benar-benar perlu (maksimal 4)

| # | Pertanyaan | Desain |
|---|---|---|
| E1 | Konfirmasi model utama | `win_logreg_core9_global` dengan nested C dan nested kebijakan ambang, plus pembanding 10.1(4). Satu run, tanpa variasi lain. |
| E2 | Deteksi dini untuk IDE | Model E1 dievaluasi pada rata-rata probabilitas 30/60/90/120 detik pertama episode (hanya episode yang sudah berjalan selama itu), dibanding durasi-berjalan saja. Inilah kegunaan nyata bagi IDE. |
| E3 | Mengurangi ketergantungan pada perbedaan antar-orang | Normalisasi personal berbasis pre-task/riwayat kausal yang **dipelajari di dalam fold** (misalnya rata-rata fitur dikurangi median riwayat sesi + skala global), dibanding `global_robust`. Laporkan AUC gabungan dan AUC dalam peserta. Pilih menurut AUC dalam peserta bila AUC gabungan setara. |
| E4 | Memakai seluruh informasi rating | Ordinal/regresi pada rating 1–10 (atau rating yang di-z per peserta) yang dievaluasi pada target biner ≥ 7, dibanding logistik biner. Biner hanya memakai 23 positif; rating penuh memakai 126 nilai. Sensitivitas ambang overload (≥ 6, ≥ 7, ≥ 8) dilaporkan sebagai analisis sekunder, bukan untuk memilih ambang terbaik. |

Yang **tidak** perlu dilakukan lagi: arsitektur neural, GRU/TCN, ensemble besar, pencarian kebijakan ambang baru, varian window 25/45 detik, atau model V3 baru dengan representasi median-minus-riwayat. Batch 02–06 sudah menunjukkan bahwa pendekatan tersebut overfit atau tidak stabil pada ukuran sampel ini.

### 10.3 Sasaran yang realistis untuk laporan akhir

Berdasarkan CI saat ini, sasaran jujur Batch 08 adalah:

- **AUC episode nested-LOPO ≥ 0,75**, dengan batas bawah CI > 0,60.
- **Balanced accuracy ≥ 0,70.**
- **Kontribusi sensor signifikan di atas konteks** (uji permutasi atau likelihood-ratio pada prediksi outer).
- **AUC dalam peserta > 0,60.**

Jika E2 menunjukkan bahwa sensor tidak menambah informasi pada 60–90 detik pertama, laporan perlu menyatakan bahwa model saat ini cocok untuk penilaian pasca-tugas, belum untuk intervensi real-time.

### 10.4 Koreksi dokumentasi

- SUMMARY Batch 07 dan skrip video monev: ganti akurasi dengan balanced accuracy/AUC, sertakan tebakan mayoritas, dan beri label "eksploratif" pada angka Test historis.
- Rekomendasi ambang "0,58–0,62" dan "dual-trigger" ditandai belum tervalidasi.

## 11. Keputusan dataset untuk Batch 08

**Keputusan: tidak membuat dua dataset sensor baru.** Batch 08 tetap membaca paket `v3_behavioral_proxy` dan `v4_self_reported_rating` sebagai sumber yang tidak diubah. Label biner dibentuk oleh **satu modul label bersama** yang dipakai semua notebook Batch 08, dan hasil labelnya disimpan sebagai artefak audit di folder output.

Alasan:

1. **Label biner adalah transformasi deterministik dari label yang sudah ada.** V4 overload = `difficulty_rating ≥ 7`. V3 High = `cognitive_load_label == "High"`. CSV baru hanya menduplikasi ribuan baris fitur yang sama dengan satu kolom berbeda, dan menambah satu sumber yang harus dijaga sinkron hash-nya.
2. **Label V3 tidak boleh dibekukan dalam file.** Referensi persentil V3 dihitung ulang dari peserta fitting di setiap fold (`fold_labels`). File statis akan memakai referensi dari seluruh Train, termasuk peserta yang sedang dievaluasi. Hasilnya kebocoran label kecil, persis hal yang sudah dihindari batch 07.
3. **Ambang overload adalah keputusan pemodelan.** E4 menguji sensitivitas ≥ 6/7/8 dan rating kontinu. Dataset yang dibekukan pada ≥ 7 justru membatasi eksperimen itu.
4. **Kode builder dataset tidak lagi ada di repo.** Sumber `build_dataset_v2/v3/v4.py` terhapus pada commit `9061a10ff` dan hanya tersisa `__pycache__`. Membuat versi dataset baru berarti memulihkan dan memverifikasi ulang seluruh builder: biaya besar tanpa informasi baru.

Agar tetap tertib seperti dataset resmi, modul label Batch 08 harus:

- Menuliskan nama kebijakan dan versi, misalnya `v4_overload_rating_ge7_v1` dan `v3_high_vs_rest_foldwise_v1`.
- Memeriksa SHA-256 paket sumber seperti batch 07.
- Menyimpan `labels_used.csv` (episode, peserta, fold, label) dan ringkasan jumlah kelas per fold di setiap output.

Dataset baru **hanya** dibenarkan jika Batch 08 mengubah isi data, bukan sekadar pemetaan label. Contohnya rubrik V3 biner baru yang menambah episode berlabel, fitur baru dari RR atau frame mentah, atau penambahan peserta. Perubahan seperti itu dibangun lewat builder resmi dengan manifest dan laporan kualitas. Dengan diagnosis di atas, perubahan semacam itu tidak menjadi prioritas Batch 08.

## Lampiran: reproduksi

```powershell
python models/batch-07/diagnosis/audit_batch07.py
```

Skrip hanya membaca `episode_predictions.csv` Batch 07 dan metadata episode V3/V4 (`episode_labels.csv`, `episode_ratings.csv`), lalu menulis [diagnosis_checks.json](diagnosis/diagnosis_checks.json). Kebutuhannya numpy, pandas, dan scipy; scikit-learn tidak diperlukan. Uji likelihood-ratio memakai skor sensor out-of-fold sebagai kovariat tetap, sehingga tidak ada fitting model sensor baru.
