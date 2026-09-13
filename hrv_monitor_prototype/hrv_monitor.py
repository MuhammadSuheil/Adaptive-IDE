import asyncio
import time
import csv
import datetime
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import welch
from bleak import BleakClient, BleakScanner

HR_MEASUREMENT_CHAR_UUID = "00002A37-0000-1000-8000-00805f9b34fb"
CALIBRATION_SECONDS      = 60
WINDOW_SIZE              = 30
LF_BAND                  = (0.04, 0.15)
ANOMALY_THRESHOLD_PCT    = 0.20

CSV_FILENAME     = "cognitive_load_logs.csv"
RAW_CSV_FILENAME = "raw_rr_log.csv"


class CognitiveLoadProcessor:
    def __init__(self):
        self.rr_data          = []
        self.baseline_lfs     = []
        self.baseline_mean    = 0.0
        self.baseline_std     = 0.0
        self.is_calibrated    = False
        self.start_time       = time.time()
        self.last_valid_rr    = None
        self.valid_rr_history = []
        self.rr_time_cursor   = 0.0

        session_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(CSV_FILENAME, mode='w', newline='') as f:
            w = csv.writer(f)
            w.writerow([f"--- SESI: {session_timestamp} ---", "", "", ""])
            w.writerow(["Timestamp_Seconds", "LF_Power", "Z_Score"])

        with open(RAW_CSV_FILENAME, mode='w', newline='') as f:
            w = csv.writer(f)
            w.writerow([f"--- SESI: {session_timestamp} ---", ""])
            w.writerow(["RR_Timestamp_Seconds", "RR_Interval_Seconds"])

    def _write_raw(self, rr_timestamp, rr_value):
        with open(RAW_CSV_FILENAME, mode='a', newline='') as f:
            csv.writer(f).writerow([f"{rr_timestamp:.4f}", f"{rr_value:.4f}"])

    def _write_processed(self, current_time, lf_power, z_score):
        with open(CSV_FILENAME, mode='a', newline='') as f:
            csv.writer(f).writerow([
                f"{current_time:.2f}",
                f"{lf_power:.4f}",
                f"{z_score:.4f}"
            ])

    def add_rr_interval(self, raw_rr_value):
        self.rr_time_cursor += raw_rr_value
        rr_timestamp = self.rr_time_cursor

        self._write_raw(rr_timestamp, raw_rr_value)

        is_anomaly = False
        if self.last_valid_rr is not None:
            pct_change = abs(raw_rr_value - self.last_valid_rr) / self.last_valid_rr
            if pct_change > ANOMALY_THRESHOLD_PCT:
                is_anomaly = True
 
        if is_anomaly:
            if len(self.valid_rr_history) >= 5:
                clean_rr = float(np.median(self.valid_rr_history))
            elif self.last_valid_rr is not None:
                clean_rr = self.last_valid_rr
            else:
                clean_rr = raw_rr_value
            print(f"[ARTEFAK] RR={raw_rr_value:.4f}s -> median={clean_rr:.4f}s")
        else:
            clean_rr = raw_rr_value
            self.last_valid_rr = clean_rr
            self.valid_rr_history.append(clean_rr)
            if len(self.valid_rr_history) > 10:
                self.valid_rr_history.pop(0)

        print(f"Filtered RR: {clean_rr:.4f}s | Timestamp: {rr_timestamp:.4f}s")
        system_time = time.time() - self.start_time
        self.rr_data.append((system_time, clean_rr))
        self.process_stream()

    def calculate_lf_power(self, data_window):
        if len(data_window) < 10:
            return None

        rrs   = np.array([x[1] for x in data_window])
        times = np.cumsum(rrs)

        t_even = np.arange(times[0], times[-1], 0.25)
        if len(t_even) < 8:
            return None

        try:
            rr_even = interp1d(times, rrs, kind='cubic')(t_even)
        except Exception as e:
            print(f"[MATH ERROR] Interpolation failed: {e}")
            return None

        nperseg = min(64, len(rr_even) // 2)
        if nperseg < 4:
            return None

        freqs, psd = welch(rr_even, fs=4.0, nperseg=nperseg)
        lf_mask    = (freqs >= LF_BAND[0]) & (freqs <= LF_BAND[1])
        lf_power   = np.trapezoid(psd[lf_mask], freqs[lf_mask])

        return float(lf_power)

    def process_stream(self):
        current_time = time.time() - self.start_time
        recent_data  = [d for d in self.rr_data if current_time - d[0] <= WINDOW_SIZE]
        lf_power     = self.calculate_lf_power(recent_data)

        if lf_power is None:
            return

        if current_time < CALIBRATION_SECONDS:
            self.baseline_lfs.append(lf_power)
            print(f"[CALIBRATING] {int(current_time)}/{CALIBRATION_SECONDS}s | LF: {lf_power:.4f}")
            return

        if not self.is_calibrated:
            self.baseline_mean = float(np.mean(self.baseline_lfs))
            self.baseline_std  = float(np.std(self.baseline_lfs))
            self.is_calibrated = True
            print("\n" + "="*40)
            print(f"CALIBRATION COMPLETE.")
            print(f"Baseline Mean LF : {self.baseline_mean:.4f}")
            print(f"Baseline STD LF  : {self.baseline_std:.4f}")
            print("="*40 + "\n")
            return

        z_score = (lf_power - self.baseline_mean) / (self.baseline_std + 1e-6)

        print(f"LF: {lf_power:.4f} | Z: {z_score:+.2f}")
        self._write_processed(current_time, lf_power, z_score)


processor = CognitiveLoadProcessor()


def hr_measurement_handler(sender, data):
    flags               = data[0]
    rr_interval_present = (flags & 0x10) >> 4

    offset = 1
    if (flags & 0x01) == 0:
        offset += 1
    else:
        offset += 2
    if flags & 0x08:
        offset += 2

    if rr_interval_present:
        while offset + 1 < len(data):
            rr_raw     = int.from_bytes(data[offset:offset+2], byteorder='little')
            rr_seconds = rr_raw / 1024.0
            if 0.3 < rr_seconds < 2.0:
                processor.add_rr_interval(rr_seconds)
            offset += 2
    else:
        print("HR data diterima, RR interval tidak tersedia di paket ini.")


async def run():
    print("Scanning...")
    devices     = await BleakScanner.discover(timeout=5.0)
    hw9_address = next(
        (d.address for d in devices if d.name and "HW9" in d.name), None
    )

    if not hw9_address:
        print("Perangkat tidak ditemukan.")
        return

    async with BleakClient(hw9_address) as client:
        print(f"Terhubung. Duduk diam selama {CALIBRATION_SECONDS} detik untuk kalibrasi.")
        await client.start_notify(HR_MEASUREMENT_CHAR_UUID, hr_measurement_handler)
        while True:
            await asyncio.sleep(1)


if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\nMonitor dihentikan.")