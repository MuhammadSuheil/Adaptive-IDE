"""Batch 08 · inferensi model final dari JSON (tanpa scikit-learn, hanya numpy).

Model deploy B3 = stacking( Track A-ens sensor , konteks waktu aktif berjalan ).

Pemakaian di IDE (per episode/tugas yang sedang berjalan):
    model = OverloadModel.load("models/batch-08/final_model/b8_overload_model.json")
    p = model.predict_episode(windows, history_reference)   # windows: list dict per window 15 s
    alarm = p >= model.threshold

`windows`: urutan window 15 s (step 5 s) sejak tugas dimulai. Setiap window memuat fitur sensor (lihat
`model.spec["features_required"]`) dan `elapsed_active_s` = waktu aktif tugas sampai akhir window.
`history_reference`: median fitur ≤12 window disjoint berkualitas yang selesai sebelum tugas dimulai pada sesi yang
sama (minimal 3); isi None bila belum ada, dan model memakai median fitting sebagai fallback (sama seperti training).
"""

import json
from pathlib import Path

import numpy as np


def _robust(x, med, iqr):
    x = np.where(np.isfinite(x), x, med)
    return np.clip((x - med) / iqr, -10.0, 10.0)


def _agg(values, how):
    v = np.sort(np.asarray(values, float))
    if how == "mean":
        return float(v.mean())
    if how == "median":
        return float(np.median(v))
    return float(v[len(v) // 2:].mean())  # top50


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class OverloadModel:
    def __init__(self, spec):
        self.spec = spec
        self.threshold = spec["stacker"]["threshold_youden"]

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def _config_window_scores(self, cfg, X, ref):
        """Skor mentah per window untuk satu konfigurasi. X: dict fitur → array (n_window)."""
        cols = cfg["features"]
        raw = np.column_stack([X[c] for c in cols])
        blocks = [_robust(raw, np.array(cfg["med"]), np.array(cfg["iqr"]))]
        if cfg.get("hybrid"):
            h = cfg["hybrid"]
            r = np.array([ref.get(c, np.nan) if ref else np.nan for c in h["columns"]], float)
            r = np.where(np.isfinite(r), r, np.array(h["fallback"]))
            delta = np.column_stack([X[c] for c in h["columns"]]) - r
            blocks.append(_robust(delta, np.array(h["med"]), np.array(h["iqr"])))
        if cfg.get("aux"):
            a = cfg["aux"]
            s = _robust(raw, np.array(a["med"]), np.array(a["iqr"])) @ np.array(a["coef"]) + a["intercept"]
            blocks.append(_robust(s.reshape(-1, 1), np.array([a["score_med"]]), np.array([a["score_iqr"]])))
        Z = np.hstack(blocks)
        lin = Z @ np.array(cfg["coef"]) + cfg["intercept"]
        return _sigmoid(lin) if cfg["kind"] == "logistic" else lin

    def scores(self, windows, history_reference=None):
        """Kembalikan (skor sensor A-ens, skor konteks, probabilitas overload) untuk window yang tersedia."""
        if not windows:
            raise ValueError("Minimal satu window.")
        feats = set(self.spec["features_required"])
        X = {c: np.array([w.get(c, np.nan) for w in windows], float) for c in feats}
        zsum = 0.0
        for cfg in self.spec["sensor_configs"]:
            ep = _agg(self._config_window_scores(cfg, X, history_reference), cfg["agg"])
            zsum += (ep - cfg["oof_mu"]) / cfg["oof_sd"]
        sensor = zsum / len(self.spec["sensor_configs"])
        c = self.spec["context"]
        e = np.log(np.array([w["elapsed_active_s"] for w in windows], float)).reshape(-1, 1)
        ctx = _agg(_sigmoid(_robust(e, np.array(c["med"]), np.array(c["iqr"])) @ np.array(c["coef"]) + c["intercept"]),
                   c["agg"])
        s = self.spec["stacker"]
        z = (s["coef_sensor"] * (sensor - s["mu_sensor"]) / s["sd_sensor"]
             + s["coef_context"] * (ctx - s["mu_context"]) / s["sd_context"] + s["intercept"])
        return sensor, ctx, float(_sigmoid(z))

    def predict_episode(self, windows, history_reference=None):
        return self.scores(windows, history_reference)[2]

    def sensor_only_probability(self, windows, history_reference=None):
        """Probabilitas Track A-ens (sensor saja), memakai kalibrasi OOF sensor."""
        sensor = self.scores(windows, history_reference)[0]
        k = self.spec["sensor_calibration"]
        return float(_sigmoid(k["coef"] * sensor + k["intercept"]))
