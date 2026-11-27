# Diagnosis Hasil Batch 08

Tanggal pemeriksaan: 27 November 2026. Dasar: run resmi pada [SUMMARY.md](SUMMARY.md) dan [ROUNDS.md](ROUNDS.md).

## Kesimpulan

Batch 08 adalah hasil pemodelan terbaik dan paling terkontrol dalam penelitian ini:

- **Model sensor-only (A-ens):** AUC 0,826 dan balanced accuracy 0,768.
- **Model deploy (B3):** AUC 0,890, balanced accuracy 0,840, dan akurasi 84,9%, melampaui tebakan mayoritas 81,7%.
- **Evaluasi:** nested LOPO 17 peserta. Model referensi Batch 07 direproduksi persis, dan audit independen lolos.
- **Uji permutasi:** pipeline nested lengkap memberi p = 0,005 (0/200 permutasi ≥ AUC asli), jadi sinyal sensor bukan kebetulan.

Hasil ini sebaiknya dibaca sebagai **bukti kelayakan (feasibility) pada satu kohort kecil**, bukan sebagai akurasi yang sudah tervalidasi untuk populasi developer. Keterbatasan utamanya terletak pada data dan desain protokol, bukan lagi pada pipeline pemodelan.

## 1. Ukuran sampel menentukan ketidakpastian

- 126 episode dari 17 peserta, dengan hanya **23 episode overload dari 12 orang**. Andreas menyumbang 5, sam 3, dan Fabian 3.
- CI 95% bootstrap per peserta lebar: A-ens AUC 0,69–0,93; B3 0,77–0,97.
- Keunggulan A-ens atas Batch 07 (ΔAUC +0,048, CI −0,05 s.d. +0,14) **belum signifikan**. Yang signifikan adalah B3 vs Batch 07 (ΔAUC +0,112, p = 0,029) dan kontribusi sensor di atas konteks (p ≈ 10⁻⁶).
- Satu episode overload yang berubah prediksinya menggeser recall sebesar 4,3 poin persentase.

**Implikasi:** sajikan hasil sebagai rentang (CI), bukan angka tunggal. Klaim "lebih baik dari Batch 07" untuk model sensor-only perlu diberi keterangan "secara titik estimasi".

## 2. Derajat kebebasan peneliti (iterasi)

Sesuai arahan user, iterasi berlangsung tiga ronde ([ROUNDS.md](ROUNDS.md)).

- **Yang terkontrol:** seleksi konfigurasi, kalibrasi, ambang, dan ensemble selalu terjadi di inner-LOPO. Model utama ronde 3 (A-ens, B3) ditetapkan sebelum hasilnya terlihat. Uji permutasi mengulang seluruh pipeline ronde 3.
- **Yang tidak terkontrol:** penyempitan pool dari 648 ke 16 konfigurasi (ronde 2) memakai pola seleksi ronde 1 dari seluruh 17 peserta. Keputusan antar-ronde ini tidak tercakup oleh nested LOPO maupun uji permutasi.
- **Konteks lintas batch:** kohort yang sama sudah dipakai sepanjang Batch 01–08. Pilihan desain yang bertahan (core9, logistik, agregasi episode, label V4) sebagian merupakan hasil pembelajaran dari kohort ini.

**Implikasi:** angka nested LOPO adalah estimasi terbaik yang tersedia, tetapi kemungkinan sedikit optimistis untuk pengguna baru. Satu-satunya cara menghilangkan bias ini adalah data uji baru yang tidak pernah disentuh (§8).

## 3. Sinyal antar-peserta vs perubahan dalam diri

| Model | AUC gabungan | AUC dalam peserta (10 orang dengan kedua kelas) | Korelasi rata-rata skor per orang vs proporsi overload orang tersebut |
|---|---:|---:|---:|
| Batch 07 | 0,778 | 0,645 | ρ = 0,544 |
| A-ens | 0,826 | 0,712 | ρ = 0,644 |
| B3 | 0,890 | 0,885 | ρ = 0,665 |

Kemampuan membedakan episode **di dalam orang yang sama** membaik. Namun model juga makin mengenali orang yang cenderung sering memberi rating tinggi. AUC dalam peserta A-ens sangat bervariasi: 1,00 pada Andrean, Baithar, Fabian, dan Siti, tetapi 0,25 (Rafi), 0,00 (kaka, 4 episode), dan 0,50 (Tio). Tujuh peserta hanya memiliki satu kelas sehingga tidak dapat dinilai di dalam diri.

**Implikasi:** pada pengguna baru, sebagian alarm dapat mencerminkan karakteristik individu, bukan kondisi saat itu. Normalisasi hybrid (relatif terhadap riwayat sesi sendiri) sudah membantu. Kalibrasi ambang per pengguna dari beberapa tugas awal layak dilakukan saat integrasi IDE.

## 4. Peran konteks waktu dan perancu protokol

- Waktu aktif yang berjalan sendirian sudah memberi AUC 0,818 dan AUC dalam peserta 0,909. Sebagian besar kekuatan model deploy B3 berasal dari variabel ini. Sensor menambah informasi yang signifikan, tetapi bukan satu-satunya sumber.
- Protokol selalu berurutan easy → medium → hard, dan soal hard ditutup oleh timeout 15 menit. Overload terkonsentrasi pada soal hard (11/18) dan episode timeout (14/28).
- Uji kontrol sudah dilakukan: skor sensor tetap menambah informasi di atas tingkat soal, durasi, dan menit dalam sesi (p ≈ 10⁻⁶), dan skor A-ens tidak berkorelasi dengan durasi di dalam kelas yang sama. Kontrol statistik ini tidak sepenuhnya setara dengan urutan soal yang diacak.
- **Untuk alarm online, sensor saja belum lebih baik daripada timer** (recall 69,6% vs 78,3% pada alarm palsu 22,3%). Keunggulan baru muncul pada gabungan B3 (82,6% vs 56,5% pada alarm palsu 13,6%).

**Implikasi:** model deploy mengasumsikan IDE tahu kapan sebuah "tugas" dimulai. Di CodeArrange batas ini jelas. Di IDE nyata, batas tugas perlu didefinisikan, misalnya fokus pada satu file/fungsi, sejak error terakhir, atau tugas yang dipilih pengguna. Definisi ini memengaruhi fitur waktu berjalan secara langsung.

## 5. Arti label

- Overload adalah **rating kesulitan subjektif ≥ 7** yang diberikan setelah jawaban benar atau timeout. Ini bukan pengukuran beban kognitif langsung (misalnya NASA-TLX atau dual-task).
- Rating dapat dipengaruhi oleh hasil (gagal/timeout), dan skala tiap orang berbeda. Target z-rating per orang konsisten buruk (ronde 1), sehingga perbedaan skala belum bisa ditangani lewat transformasi sederhana.
- V3 (proxy perilaku) terbukti tidak dapat dipelajari dari sensor tanpa kebocoran durasi (exp04: AUC LOPO 0,43). V3 hanya dipakai sebagai validasi silang.

## 6. Keterbatasan sensor yang tidak berubah sejak Batch 07

- Fitur gaze dan eye-state tidak dipakai karena validasi kalibrasi hanya lolos pada 4 dari 23 rekaman.
- "Ukuran iris" adalah rasio geometri landmark, bukan diameter pupil.
- HRV hanya time-domain (RMSSD, SDNN) pada window 15 s. Detak jantung rata-rata tidak ditambahkan karena sinyalnya sepenuhnya dijelaskan waktu dalam sesi (Batch 07 DIAGNOSIS §9).
- Satu perangkat dan satu ruangan; tidak ada variasi kamera atau pencahayaan.

## 7. Hal yang sudah terbukti dan dapat dipegang

| Temuan | Bukti |
|---|---|
| Pipeline bebas kebocoran peserta dan reprodusibel | Assert fit/held per fold; audit independen; run notebook identik dengan run skrip; reproduksi Batch 07 tepat |
| Sinyal sensor nyata untuk overload V4 | Permutasi pipeline lengkap p = 0,005 (0/200 permutasi ≥ AUC asli); uji likelihood-ratio di atas konteks p ≈ 10⁻⁶ |
| Rating ≥ 7 adalah definisi overload terbaik | Sensitivitas ≥6 / ≥7 / ≥8: AUC 0,669 / 0,802 / 0,776 |
| Normalisasi relatif terhadap riwayat sesi sendiri, agregasi window teratas, dan ensemble konfigurasi bermanfaat | Seleksi inner 17/17 fold (ronde 1); ensemble +0,024 AUC atas seleksi tunggal pada fold yang sama |
| Deteksi dini memungkinkan | AUC 0,85–0,90 pada 30–90 detik pertama (B3); alarm median 63 s, ±2,5 menit sebelum episode berakhir |
| V3 tidak dapat dipelajari tanpa kebocoran durasi | exp04 |

## 8. Saran setelah Batch 08 (di luar pemodelan)

Pemodelan pada data saat ini sudah mendekati batas yang dapat dibuktikan. Peningkatan bermakna berikutnya membutuhkan **data baru**, bukan model baru:

1. **Uji eksternal.** Rekam 8–10 peserta baru dengan protokol yang sama, lalu evaluasi `final_model/b8_overload_model.json` **tanpa melatih ulang**. Ini satu-satunya cara menghilangkan bias §2.
2. **Acak urutan soal** (atau minimal beberapa soal hard di awal), supaya overload tidak terikat pada posisi waktu dalam sesi.
3. **Tambahkan pengukuran beban independen**, misalnya NASA-TLX singkat per set, agar label tidak hanya bergantung pada rating kesulitan.
4. **Pilot integrasi IDE.** Definisikan batas tugas (§4) dan uji ambang 0,525 dengan kalibrasi per pengguna dari beberapa tugas pertama.
5. **Perbaiki validasi gaze** saat pengambilan data, agar fitur spasial (gaze, fiksasi, transisi) dapat diuji. Pada data saat ini fitur tersebut tidak dapat dipakai.
