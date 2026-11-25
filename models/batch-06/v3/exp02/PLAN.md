# Rencana ablasi normalisasi Exp02

## Tujuan

Memisahkan kontribusi transformasi RMSSD dan eye terhadap hasil Extra Trees pada subset empat orang yang dipakai Exp01. Perbandingan mengubah normalisasi, sementara fitur, data, baseline, model, seed, serta fold tetap.

## Sumber dan pembagian

Snapshot dibekukan dari run `baseline_20261123T192338Z_8d8cee5e`. Hash CSV harus cocok. Train tetap 671 window dengan 264 Low, 180 Medium, dan 227 High; validation tetap 99 window dengan 43 Low, nol Medium, dan 56 High. Bukti sumber baseline dan identitas setiap window tertanam di notebook.

Evaluasi utama memakai empat fold peserta dari Exp01. Kontrol sekunder memakai lima fold blok waktu dengan guard 45 detik. Daftar fitting dan held-out diperiksa sebelum fitting; setiap endpoint Train mendapat tepat satu prediksi OOF per protokol/varian. ID Test dilarang pada fitting dan prediksi.

## Model dan perlakuan

Extra Trees: 200 trees, min_samples_leaf=5, max_features=0.8, seed=42, dua thread. RobustScaler hanya mempelajari statistik dari fitting fold. Tidak ada search, balancing, pemilihan seed, penghapusan fitur, atau classifier per identitas.

Empat perlakuan adalah raw, RMSSD-only, eye-only, dan gabungan. RMSSD relatif memakai RMSSD/baseline − 1. Eye memakai transformasi Exp01 terhadap satu window pre_task awal yang sama. SDNN tetap mentah. Dummy memakai kelas mayoritas fitting dan satu pekerjaan per fold.

Semua perlakuan dibakukan untuk CV dan validation sebelum fitting. Tidak ada keluarga model baru atau pemenang yang dipilih dari validation. Model Train penuh dari keempat perlakuan disimpan sebagai pembanding.

## Metrik dan kontras

Laporkan accuracy window, Macro-F1 dengan tiga label tetap, recall/support tiap kelas, serta rata-rata accuracy/Macro-F1 per orang dengan bobot sama. Recall tanpa support adalah N/A; Macro-F1 tetap memasukkan kelas tanpa support dengan F1 nol untuk konsistensi metrik. Metrik rata-rata per orang tetap memakai tiga kelas pada setiap orang.

Lima kontras dibakukan: RMSSD-only vs raw, eye-only vs raw, gabungan vs raw, gabungan vs RMSSD-only, dan gabungan vs eye-only. Semua delta dihitung dari skor dalam run Exp02. Perbandingan raw/combined dengan skor Exp01 hanya pemeriksaan kontrol, bukan sumber delta campuran antar-run.

## Budget dan batas interpretasi

Sembilan fold × (empat varian Extra Trees + satu Dummy) = 45 fitting CV. Empat fitting Train penuh mengevaluasi seluruh perlakuan pada validation historis. Total maksimal 49 fitting, tanpa perluasan setelah membaca hasil. Test tidak dievaluasi dan tidak ada refit Train + validation.

Validation sudah diperiksa sebelumnya dan tidak mempunyai Medium; hasilnya tidak menjadi bukti generalisasi tiga kelas lengkap. Empat orang membatasi klaim populasi. Label masih difficulty/fase, acuan eye masih pre_task, dan normalisasi iris bukan pengukuran pupil. Selisih deskriptif tidak dianggap signifikansi statistik dari window independen.

## Status persiapan

Preflight lokal berhasil memverifikasi snapshot, hash CSV, waktu/nilai acuan, pemisahan peserta/waktu, serta empat matriks fitur 671 × 10. Training penelitian Exp02 belum dijalankan. Uji end-to-end menggunakan fixture sintetis terpisah; angkanya tidak menjadi hasil penelitian.
