# Pemeriksaan Teknis Paket Batch-02

Tanggal pemeriksaan: 15 November 2026. Runtime lokal: Windows, Python 3.14.6.

Paket sudah melalui pemeriksaan teknis tanpa fitting classifier penelitian atau training neural. Hasil pada dokumen ini bukan metrik cognitive load.

## Hasil pemeriksaan

| Pemeriksaan | Hasil |
|---|---|
| Notebook | 15 file exp00–exp14 lolos validasi nbformat |
| Cell kode | 105 cell lolos parsing Python |
| Setup lokal | Cell setup, environment, dan konfigurasi berhasil dieksekusi tanpa instalasi ulang |
| Konstruktor model | 20 konstruktor inti menerima parameter yang disiapkan; tidak dipanggil fit |
| Tes otomatis | 44 tes lulus |
| Preprocessing | Median/scaling training-only, fitur kosong/konstan, log rasio, MI dan PCA diuji pada data sintetis kecil |
| Balancing | Bobot kelas dan SMOTE seluruh minoritas diuji tanpa melatih classifier |
| History | Prefix, perubahan annotation, gap/grid, cache sensor berubah, serta urutan antarstream diperiksa |
| Split/evaluasi | Endpoint UTC berpasangan, pemisahan grup, kelas absent, dan probabilitas invalid diuji |
| Artefak/resume | Alur exp01 memakai model tiruan; resume memuat hasil tersimpan tanpa fitting ulang |
| Seleksi akhir | Alur exp14, joint-window, refit, sensitivitas, dan ekspor artefak diperiksa dengan model tiruan dan pencarian satu kontrol; aturan pencarian lengkap diuji terpisah dengan skor fixture |
| Neural | Forward MLP/FT-Transformer/GRU diuji tanpa backward atau optimizer step |
| Pemeriksaan sumber | Ketiga CSV aktif dibaca dan checksum dicocokkan; dataset tidak diubah |
| Lint dan format | Ruff check dan format check lulus |
| Keamanan | Bandit medium/high lulus |
| Tipe dasar | Mypy lulus dengan ignore-missing-imports dan follow-imports=silent |
| Sintaks modul | compileall lulus |

Simpan interpretasi tes integrasi dengan tepat: model tiruan mengembalikan prediksi yang ditentukan fixture. Angka hasil tes sementara tidak mengukur kemampuan classifier dan tidak disimpan sebagai output penelitian.

## Kondisi data dan gate

| Window / step | Low | Medium | High | Total |
|---|---:|---:|---:|---:|
| 15 / 5 detik | 870 | 1134 | 1251 | 3255 |
| 25 / 5 detik | 797 | 988 | 1137 | 2922 |
| 45 / 5 detik | 732 | 657 | 891 | 2280 |

Ini jumlah sebelum pengambilan tambahan. Gate tambahan, proporsi, dan pembekuan release tetap pending. Belum ada snapshot input atau hasil training batch-02 pada paket ini.

## Belum diuji

- Fitting, konvergensi, dan kinerja classifier nyata.
- Training MLP, FT-Transformer, atau GRU.
- Eksekusi GPU dan inference/persistensi checkpoint TabPFN-2.
- Evaluasi nested lengkap dengan kandidat/model nyata.
- Pengesahan serta rekonstruksi cache untuk sumber tambahan yang belum tersedia.
- Runtime Google Colab.

Uji teknis tidak menjamin seluruh kombinasi training akan bebas masalah pada runtime dan data final. Notebook akan memeriksa release, dependency, dukungan data, serta checkpoint sebelum menjalankan cabangnya.

## Catatan perbaikan selama validasi

- Ekspor grafik memakai FigureCanvasAgg agar tidak membutuhkan Tcl/Tk atau jendela GUI.
- Identitas cache history mencakup nilai sensor; perubahan input tidak memakai hasil lama.
- Pengambilan sequence mempertahankan konteks sesi meskipun row beberapa sesi terseling.
- Pemeriksaan environment mendeteksi modul versi lama yang masih dimuat setelah pemasangan dependency.
- Snapshot/resume menolak perubahan checksum data dan identitas run.

Detail versi dependency, jumlah kandidat, dan hash kode tersimpan pada [report.json](report.json). Panduan menjalankan paket terdapat pada [README](../README.md).
