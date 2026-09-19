# True: gunakan filter median. False: lewati filter median.
FILTER_ENABLED = False

# RR ditolak jika selisih terhadap median melebihi nilai ini (milidetik).
MEDIAN_THRESHOLD_MS = 250

# Panjang data yang dianalisis dan pergeseran window (detik).
# Harus memenuhi: 0 < STEP_SECONDS <= WINDOW_SECONDS.
WINDOW_SECONDS = 25
STEP_SECONDS = 5

# Kalibrasi sambil duduk tenang, diulang untuk setiap sesi.
# Target adalah durasi RR yang diterima; batas maksimum adalah waktu berjalan.
BASELINE_TARGET_SECONDS = 120
BASELINE_MAX_SECONDS = 300

# Pemulihan dan kecukupan data: parameter awal prototype, belum divalidasi.
RR_GAP_SECONDS = 3
RECOVERY_RR_COUNT = 5
HRV_MIN_COVERAGE = 0.80  # Proporsi durasi window yang diwakili RR diterima.
HRV_MIN_PAIRS = 10
HRV_MAX_DATA_AGE_SECONDS = 3
