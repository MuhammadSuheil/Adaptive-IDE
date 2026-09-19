# RR interval dan HRV

Satu perintah menjalankan kalibrasi pribadi, perekaman RR, preprocessing, dan
perhitungan RMSSD–SDNN secara live. **Sesi baru hanya menghasilkan dua file log:**
`rr_raw.csv` dan `hrv.csv`. HRV dan statusnya tetap tampil di terminal.

**Untuk penggunaan normal, cukup jalankan `calibrate.py`.** File lain tetap
diperlukan sebagai modul yang dipanggil selama sesi; tidak perlu menjalankan
recording, preprocessing, dan calculate pada terminal terpisah.

## Menjalankan

Python 3.10 atau lebih baru. Dari root repository:

```powershell
python -m pip install -r hrv_monitor_prototype/requirements.txt
python -m hrv_monitor_prototype.calibrate
```

Jika terminal berada di folder lain, pindah dahulu ke root project:

```powershell
cd E:\Skripsi\ngodong\Adaptive-IDE
```

Peserta duduk tenang selama kalibrasi. Ketika baseline siap, terminal memberikan
instruksi untuk mulai coding. Tekan Ctrl+C untuk mengakhiri sesi.

```powershell
# Otomatis berhenti setelah 180 detik tugas, di luar durasi kalibrasi.
python -m hrv_monitor_prototype.calibrate --duration 180

# Gunakan alamat sensor jika ada beberapa perangkat dengan nama yang cocok.
python -m hrv_monitor_prototype.calibrate --address ALAMAT_BLE
```

Nama sensor default mengandung `HW9`; ubah dengan `--name NAMA`.
`--output FOLDER` mengganti folder induk sesi. Setiap run membuat folder sesi baru;
rekaman lama tidak ditimpa atau dihapus. Koneksi terputus mengakhiri sesi.
Secara default, kedua log tersimpan di `hrv_monitor_prototype/sessions/NAMA_SESI/`.
Lokasi folder dicetak di terminal setelah sensor tersambung.

## Dua file log

| File | Isi |
|---|---|
| `rr_raw.csv` | Semua RR asli dari kalibrasi dan tugas, timestamp penerimaan, dan fase |
| `hrv.csv` | Hasil baseline dan HRV setiap window tugas, beserta status/kecukupan data |

**Raw RR** menyimpan:

- `raw_id`, `packet_id`: urutan RR dan paket asal.
- `received_at`: timestamp penerimaan paket pada komputer, UTC.
- `elapsed_seconds`: detik sejak perekaman sesi dimulai, memakai jam monotonic.
- `rr_units`: nilai asli BLE dalam unit 1/1024 detik.
- `rr_original_ms`: konversi RR menjadi milidetik, `rr_units × 1000 / 1024`.
- `contact_detected`: status kontak jika didukung sensor.
- `phase`: `calibration` atau `task`.

Raw ditulis sebelum preprocessing. RR nol, di luar rentang, atau yang kemudian
ditolak oleh filter tetap tersimpan. Satu paket dapat berisi beberapa RR dengan
timestamp penerimaan yang sama; timestamp ini bukan waktu pengukuran setiap denyut.
Field BPM pada format BLE hanya dilewati dan tidak digunakan sebagai fitur.
Paket tanpa RR tidak menghasilkan RR buatan. Paket rusak yang tidak bisa dibaca
dilaporkan di terminal. Payload paket BLE tidak lagi disimpan ke file.

**Log HRV** menyimpan kolom `phase`:

- `baseline`: satu hasil kalibrasi per sesi. Jika kalibrasi gagal, status
  `unavailable` dan nilai HRV kosong; alasannya tetap tercatat.
- `task`: hasil pada setiap window tugas setelah kalibrasi berhasil.

Kolom lainnya: `window_start`, `window_end`, `window_seconds`, `rr_count`,
`rr_pairs`, `rmssd_ms`, `sdnn_ms`, `status`, `reason`, `accepted_rr_seconds`,
`coverage`, dan `data_age_seconds`.

Semua timestamp HRV memakai UTC. Pada baris baseline, batas waktu mencakup masa
kalibrasi; `accepted_rr_seconds` menunjukkan durasi RR yang dipakai. Pada baris
tugas, batas waktu menunjukkan window analisis. Window tugas dimulai dari akhir
kalibrasi. Raw memakai elapsed seluruh sesi; rebasing waktu untuk analisis tugas
hanya dilakukan di memori.

Hasil preprocessing, paket BLE, baseline JSON, salinan raw tugas, dan log teks
terpisah tidak dibuat saat sesi live. Baseline HRV sudah masuk `hrv.csv`.
Kedua file CSV di-flush setelah penulisan agar bisa dibaca selama sesi berlangsung.

## Pengaturan

Edit [config.py](config.py) sebelum memulai sesi. Nilai saat README diperbarui:

```python
FILTER_ENABLED = False
MEDIAN_THRESHOLD_MS = 250
WINDOW_SECONDS = 25
STEP_SECONDS = 5
BASELINE_TARGET_SECONDS = 120
BASELINE_MAX_SECONDS = 300
RR_GAP_SECONDS = 3
RECOVERY_RR_COUNT = 5
HRV_MIN_COVERAGE = 0.80
HRV_MIN_PAIRS = 10
HRV_MAX_DATA_AGE_SECONDS = 3
```

Saat ini filter median **nonaktif** (`False`). Ubah menjadi `True` jika ingin
mengaktifkannya. Pemeriksaan rentang RR, waktu, kontak, pemulihan, dan kecukupan
data tetap berjalan pada kedua pengaturan. Perubahan config dibaca pada saat
program dijalankan kembali; tidak mengubah rekaman yang sudah tersimpan.

Window dan step harus memenuhi `0 < step <= window`. Coverage adalah proporsi
durasi window yang terwakili RR diterima: 0.80 pada window 25 detik berarti
minimal 20 detik data. Pertimbangkan minimum pasangan jika window diperkecil.
Angka filter/pemulihan/kecukupan tersebut adalah **parameter awal prototype**,
bukan standar universal atau hasil validasi khusus HW9.

## Kalibrasi pribadi

Setiap sesi dimulai dengan kalibrasi sambil duduk tenang. Targetnya 120 detik RR
yang diterima setelah preprocessing. Jeda tidak menambah durasi data; waktu
tunggu otomatis diperpanjang sampai paling lama 300 detik total.

Data sebelum jeda tetap dihitung, sehingga kalibrasi tidak mengulang dari nol.
RR yang masih dalam pemulihan belum dipakai. Ketika target terpenuhi dan kedua
fitur dapat dihitung, baseline disimpan pada baris `baseline` di `hrv.csv`, lalu
perekaman tugas dimulai. Baseline adalah satu RMSSD dan satu SDNN dari seluruh
RR kalibrasi yang dipakai, bukan rata-rata window bertumpang tindih.

Jika batas waktu habis, baseline dinyatakan belum tersedia dan tugas tidak dimulai.
Jalankan ulang perintah untuk mengulang kalibrasi dalam sesi baru. Rekaman yang
gagal tetap tersimpan. Baseline sesi sebelumnya tidak digunakan kembali.

## Filtering dan pemulihan

Preprocessing berjalan di memori dengan aturan berikut:

1. Estimasikan waktu denyut dari durasi RR dan timestamp penerimaan. Selisih
   penjumlahan waktu dengan penerimaan lebih dari 2 detik menyebabkan penambatan
   ulang dan segmen baru. Ketelitian waktu dibatasi buffering/latensi BLE.
2. Kecualikan RR di luar 300–2000 ms, kontak sensor yang dinyatakan hilang,
   waktu paket yang tidak dapat dipercaya, dan interval yang melintasi awal fase.
3. Dalam keadaan normal, bandingkan RR dengan median maksimal 11 observasi
   plausible sebelumnya, setelah tersedia minimal 5 observasi. Selisih lebih
   dari 250 ms menyebabkan pengecualian. Observasi yang hanya ditolak oleh median
   tetap masuk riwayat agar perubahan yang menetap dapat diikuti.
4. Setelah jeda RR lebih dari 3 detik, mulai pemulihan. Tampung RR baru dan
   evaluasi ulang terhadap median kelompok. Tunggu minimal 5 RR yang lolos
   evaluasi sebelum kembali ke keadaan normal. Buffer pemulihan dibatasi.

Contoh **ketika filter median aktif**: kelompok `953, 625, 648, 617, 640` ms
memiliki median 640 ms, sehingga RR 953 ms ditolak oleh ambang 250 ms. Baru empat
RR yang diterima; masih perlu data tambahan. Jika filter median nonaktif, kelima
RR dapat menyelesaikan pemulihan jika pemeriksaan dasarnya terpenuhi. Kecukupan
window tetap diperiksa sebelum HRV ditampilkan.
RR yang lolos pemulihan tidak dimasukkan kembali ke window yang sudah diterbitkan.

`FILTER_ENABLED = False` hanya mematikan pengecualian median. Pemeriksaan dasar,
pemulihan, dan kecukupan data tetap berjalan. Tidak ada interpolasi atau RR
pengganti. Filter sederhana ini belum menjamin artefak yang tersisa bisa dihilangkan.

## HRV live dan terminal

Laporan window pertama terbit 25 detik setelah baseline siap, lalu setiap 5 detik.
Sebelum menampilkan angka, program memeriksa:

- Tidak sedang jeda atau pemulihan.
- Minimal 80% durasi window terisi RR yang diterima. Rentang tumpang tindih tidak
  dihitung dua kali, dan bagian interval di luar window tidak menambah coverage.
- Minimal 10 pasangan RR berurutan.
- Umur RR diterima terakhir maksimal 3 detik pada akhir window.

Status yang dicatat: `ready`, `gap`, `recovering`, atau `insufficient_data`.
Jika syarat gagal, RMSSD dan SDNN kosong di CSV dan tampil sebagai `tidak tersedia`
di terminal. Jumlah RR, durasi, dan alasan tetap tercatat. Angka 0 tetap ditampilkan
sebagai 0.00 ms jika syarat terpenuhi. `ready` berarti memenuhi aturan prototype,
bukan bukti akurasi fisiologis atau label beban kognitif.

RMSSD memakai akar rata-rata kuadrat selisih pasangan RR berurutan dalam segmen
yang sama. Pasangan tidak menyambung melewati RR terbuang atau jeda yang terdeteksi.
SDNN memakai standar deviasi sampel (`n − 1`). Keduanya dalam milidetik.

Timer tetap berjalan ketika RR tidak masuk selama koneksi masih aktif. Setiap
window diterbitkan satu kali; paket terlambat tidak mengubah hasil yang sudah
terbit. Window akhir yang belum lengkap dilewati. HRV tampil kembali otomatis
ketika data mencukupi.

## Peran file dan ketergantungannya

| File | Tanggung jawab |
|---|---|
| [record_rr.py](record_rr.py) | Mencari/menghubungkan sensor, membaca RR, memberi timestamp, dan menulis `rr_raw.csv` |
| [preprocess_rr.py](preprocess_rr.py) | Estimasi waktu denyut, filter, dan pemulihan setelah jeda |
| [calculate_hrv.py](calculate_hrv.py) | Rumus RMSSD–SDNN dan pemeriksaan kecukupan setiap window |
| [calibrate.py](calibrate.py) | Mengatur kalibrasi, menghubungkan seluruh proses, menampilkan terminal, dan menulis `hrv.csv` |
| [config.py](config.py) | Pengaturan yang digunakan saat program dimulai |

Alur pemanggilan dalam satu sesi:

```text
calibrate.py
  ├─ record_rr.py       → sensor BLE → RR asli → rr_raw.csv
  ├─ preprocess_rr.py   → filtering dan pemulihan di memori
  ├─ calculate_hrv.py   → RMSSD, SDNN, dan status window
  └─ terminal + hrv.csv → hasil baseline dan tugas
```

**`record_rr.py` bukan file yang tidak terpakai.** `calibrate.py` mengimpor
`record_sensor` untuk menerima RR dari sensor dan `SESSION_ROOT` untuk lokasi
penyimpanan. Jika file tersebut dihapus, sesi baru melalui `calibrate.py` akan
gagal dengan `ModuleNotFoundError`. Preprocessing dan calculate masih bisa
digunakan untuk input yang sudah tersimpan, tetapi tidak menggantikan perekam BLE.

Untuk pengujian sensor tanpa kalibrasi atau HRV:

```powershell
python -m hrv_monitor_prototype.record_rr
```

`record_rr.py` dapat dijalankan sendiri untuk merekam raw tanpa kalibrasi; mode
tersebut menghasilkan `rr_raw.csv` saja dengan fase `raw`. Utilitas ekspor CSV
offline pada preprocessing dan calculate tetap tersedia jika dijalankan secara
eksplisit; tidak digunakan untuk menghasilkan file tambahan pada sesi live.
Perhitungan offline memakai seluruh input yang diberikan, sehingga pisahkan fase
kalibrasi dan tugas jika menyiapkan data analisis ulang dari raw sesi lengkap.

## Pengujian

```powershell
python -m unittest discover -s hrv_monitor_prototype/tests -v
```

Pengujian mencakup rumus, raw preservation, filter/pemulihan, jeda, kecukupan data,
kalibrasi, terminal, dan kepastian bahwa sesi live hanya menghasilkan dua file.
Pengujian koneksi serta karakteristik HW9 tetap memerlukan perangkat asli.
