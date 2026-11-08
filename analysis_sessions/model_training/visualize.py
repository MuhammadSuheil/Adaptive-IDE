"""Explain the five-level reference, model agreement and feature dependence."""
from __future__ import annotations

import argparse
import base64
from html import escape
import json
from pathlib import Path
import sys

from matplotlib.figure import Figure
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from analysis_sessions.model_training.data import DIAGNOSTICS, LEVELS, NOTE, digest, load_dataset, write_json
from analysis_sessions.model_training.reference import weight_key
from analysis_sessions.model_training.presentation import FAMILY_NAMES, FEATURE_NAMES, LEVEL_NAMES, NOTE_ID, VARIANT_NAMES, display_frame

MODEL_NAMES = {"svm": "RBF-SVM (SVR)", "random_forest": "Random Forest", "gradient_boosting": "Gradient boosting"}
COLORS = {"svm": "#3266b2", "random_forest": "#148472", "gradient_boosting": "#b66114"}
LEVEL_COLORS = ("#5097bd", "#78b8a6", "#dfc25a", "#df9754", "#bd6170")
CHARTS = {
    "data_coverage": ("1. Jendela mana yang dapat diberi skor?", "Jendela dapat diberi skor jika rasio RMSSD terhadap acuan istirahat, frekuensi kedipan, dan proporsi estimasi fiksasinya valid. Fitur lain yang kosong dapat diisi dengan median data latih. Jendela yang dikeluarkan tetap dihitung. Ketiga dataset saling mencakup, sehingga jumlah jendelanya tidak boleh dijumlahkan."),
    "reference_bands": ("2. Bagaimana lima tingkat beban ditentukan?", "Grafik ini memakai acuan seluruh data latih untuk model akhir. Skor kontinu menggabungkan komponen HRV dan mata, lalu dibagi menjadi lima tingkat menggunakan empat batas persentil acuan latih. Evaluasi peserta uji memakai acuan tersendiri yang tidak menyertakan peserta tersebut. Batas ini bersifat relatif, bukan ambang beban kognitif universal. Histogram menghitung jendela; penentuan batas memberi setiap peserta bobot yang sama."),
    "missing_features": ("3. Pengukuran apa saja yang masuk ke model?", "Semua prediktor HRV dan mata yang dikonfigurasi ditampilkan. Angka pada sel menunjukkan persentase nilai yang hilang. EAR, geometri iris, statistik tatapan, estimasi status mata, dan konteks kepala merupakan kandidat masukan. Proporsi status mata belum terklasifikasi atau lainnya hanya menjadi diagnostik. Estimasi iris bukan pengukuran diameter pupil."),
    "model_agreement": ("4. Seberapa sesuai prediksi model dengan acuan sementara?", "Macro-F1 rata-rata per peserta yang lebih tinggi berarti kesesuaian lebih baik; galat skor yang lebih rendah juga lebih baik. Setiap peserta uji memiliki bobot yang sama. Macro-F1 mencakup lima label; F1 kelas yang tidak muncul ditetapkan nol. Hasil ini mengukur peniruan aturan sensor yang dipilih, bukan akurasi terhadap pengukuran beban kognitif independen. Skor kedua cabang digabungkan sebelum tingkat beban ditentukan."),
    "confusion_matrices": ("5. Tingkat mana yang sering tertukar?", "Baris menunjukkan tingkat acuan sementara; kolom menunjukkan prediksi model. Setiap sel memuat persentase terhadap total baris acuan dan jumlah jendela. Diagonal yang kuat berarti prediksi sesuai dengan acuan. Sembilan panel memakai jendela uji yang sama, urutan label tetap, dan skala warna yang sama."),
    "participant_agreement": ("6. Apakah kesesuaiannya konsisten antarpeserta?", "Setiap sel adalah macro-F1 lima label untuk satu peserta yang seluruh rekamannya dipisahkan dari pelatihan. Label kolom memuat jumlah jendela uji; sebagian peserta mungkin memiliki sedikit jendela atau tidak mencakup semua tingkat acuan. Variasi antarpeserta perlu diperhatikan karena jendela yang tumpang tindih bukan sampel independen. Nilai ini bukan interval kepercayaan atau pengukuran beban yang sudah divalidasi."),
    "feature_dependence": ("7. Kelompok fitur mana yang memengaruhi prediksi?", "Semua kolom dalam satu kelompok fitur digeser secara melingkar di dalam data peserta uji yang sama. Hubungan antarkolom dalam kelompok tetap terjaga, tetapi keselarasan dengan sinyal lain berubah. Nilai positif berarti galat skor meningkat; nilai negatif berarti perubahan tersebut kebetulan memperbaiki prediksi. Ini merupakan diagnostik ketergantungan model, bukan bukti sebab-akibat atau validitas fitur sebagai ukuran beban kognitif. Fitur berkorelasi dapat saling menggantikan; nilai kecil tidak berarti rekamannya sia-sia."),
    "branch_contributions": ("8. Bagaimana kontribusi kedua cabang berubah sepanjang waktu?", "Grafik memakai kandidat dengan kesesuaian tertinggi dan peserta dengan jendela uji terbanyak. Panel atas membandingkan estimasi skor gabungan dengan acuan sementara. Panel bawah menampilkan kontribusi HRV dan mata setelah dikalikan bobot; jumlahnya sama dengan estimasi skor gabungan. Setiap titik merangkum satu jendela. Garis penghubung tidak menambahkan pengukuran pada celah rekaman. Prediksi berasal dari evaluasi peserta uji, bukan model akhir yang diuji pada data latihnya sendiri."),
    "weight_sensitivity": ("9. Apa yang berubah pada pembobotan 50/50?", "Grafik menunjukkan seberapa sering prediksi berganti tingkat ketika kedua cabang model yang sama digabungkan dengan bobot alternatif. Setiap rasio memiliki batas tingkat yang dihitung dari acuan latih yang sama. Pelatihan ulang tidak diperlukan karena target setiap cabang tidak bergantung pada bobot penggabungan. Banyaknya label yang berubah tidak membuktikan rasio mana yang lebih akurat mengukur beban kognitif sebenarnya."),
}


def comparison_table(summaries):
    return pd.DataFrame([{"Training dataset": VARIANT_NAMES[row["variant"]], "Model": MODEL_NAMES[row["model"]],
                          "Participant macro-F1 agreement": row["participant_macro_f1"], "Participant score MAE": row["participant_score_mae"],
                          "Exact reference agreement": row["exact_agreement"], "Within one level": row["within_one_level"],
                          "Ordinal error": row["ordinal_mae"], "Prediction time (ms)": row["mean_single_window_ms"]} for row in summaries])


def explain(summaries):
    best = summaries[0]
    return (f"Kesesuaian terhadap acuan tertinggi diperoleh {MODEL_NAMES[best['model']]} pada dataset {VARIANT_NAMES[best['variant']]}; "
            f"macro-F1 rata-rata peserta {best['participant_macro_f1']:.3f}, dengan galat skor rata-rata peserta {best['participant_score_mae']:.2f} poin pada skala 0–100. "
            f"Evaluasi mencakup {best['test_windows']} jendela berkualitas baik dari {best['test_participants']} peserta uji. "
            "Hasil ini menunjukkan kandidat untuk meniru acuan sementara, bukan model pengukur beban kognitif yang sudah divalidasi.")


def style_axis(axis):
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#e0e5eb", linewidth=.7)
    axis.set_axisbelow(True)


def heat_labels(axis, values, percent=False, counts=None):
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            text = "—" if not np.isfinite(value) else f"{value:.0%}" if percent else f"{value:.2f}"
            if counts is not None:
                text += f"\n({counts[row, col]})"
            axis.text(col, row, text, ha="center", va="center", fontsize=8,
                      color="white" if np.isfinite(value) and value > .6 else "#1c2938")


def make_figures(directory, dataset, config):
    directory = Path(directory)
    summaries = json.loads((directory / "summary.json").read_text())
    reference = json.loads((directory / "reference.json").read_text())
    metrics = json.loads((directory / "participant_metrics.json").read_text())
    predictions = pd.read_csv(directory / "held_out_predictions.csv")
    labels = pd.read_csv(directory / "reference_labels.csv", keep_default_na=False)
    coverage = pd.read_csv(directory / "data_coverage.csv")
    variants, models = tuple(VARIANT_NAMES), tuple(MODEL_NAMES)
    keys = [(variant, model) for variant in variants for model in models]
    row_names = [f"{VARIANT_NAMES[variant]} · {MODEL_NAMES[model]}" for variant, model in keys]
    positions = np.arange(3)
    figures = {}

    fig = Figure(figsize=(11, 4.5), layout="constrained")
    axis = fig.subplots()
    scorable = coverage.set_index("variant").loc[list(variants), "scorable_windows"].to_numpy()
    excluded = coverage.set_index("variant").loc[list(variants), "unscorable_windows"].to_numpy()
    first = axis.bar(positions, scorable, label="Scorable", color="#288c80")
    axis.bar_label(first, padding=3)
    axis.bar(positions, excluded, bottom=scorable, label="Insufficient reference signal", color="#c6cdd4")
    axis.set(xticks=positions, xticklabels=list(VARIANT_NAMES.values()), ylabel="Windows", title="Data latih yang tersedia dan jendela yang dikeluarkan")
    axis.margins(y=.15)
    axis.legend()
    style_axis(axis)
    figures["data_coverage"] = fig

    fig = Figure(figsize=(12, 4.5), layout="constrained")
    left, right = fig.subplots(1, 2)
    valid_labels = labels.loc[labels.variant.eq("good_only") & labels.signal_status.eq("scorable")]
    boundaries = reference["boundaries"][weight_key(config["primary_hrv_weight"]) ]
    edges = [0, *boundaries, 100]
    for index, (level, color) in enumerate(zip(LEVELS, LEVEL_COLORS)):
        left.axvspan(edges[index], edges[index + 1], color=color, alpha=.25, label=LEVEL_NAMES[level])
    left.hist(valid_labels.reference_score.astype(float), bins=25, color="#405c76", alpha=.8)
    for boundary in boundaries:
        left.axvline(boundary, color="#44505a", linestyle="--", linewidth=.8)
    left.set(xlim=(0, 100), xlabel="Provisional combined reference score", ylabel="Good windows", title="Skor acuan latih dan lima tingkat relatif")
    left.legend(fontsize=8, ncols=3)
    eye = config["reference_eye_weights"]
    primary = config["primary_hrv_weight"]
    values = [primary, (1 - primary) * eye["blink_suppression"], (1 - primary) * eye["fixation_proportion"]]
    bars = right.bar(["HRV\nreference", "Blink\nreference", "Fixation\nreference"], values, color=("#3469ae", "#2a9181", "#86b79d"))
    right.bar_label(bars, labels=[f"{value:.0%}" for value in values], padding=3)
    right.set(ylim=(0, 1), ylabel="Weight in the provisional reference", title="Definisi acuan\n(bukan bobot fitur hasil pembelajaran)")
    style_axis(right)
    figures["reference_bands"] = fig

    missing = np.array([dataset["views"][variant][dataset["features"]].isna().mean().to_numpy() for variant in variants]).T
    fig = Figure(figsize=(10, max(7, len(missing) * .28)), layout="constrained")
    axis = fig.subplots()
    image = axis.imshow(missing, vmin=0, vmax=1, cmap="YlOrRd", aspect="auto")
    axis.set(xticks=positions, xticklabels=list(VARIANT_NAMES.values()), yticks=np.arange(len(missing)), yticklabels=[FEATURE_NAMES.get(name, name) for name in dataset["features"]], title="Nilai yang hilang pada setiap kandidat prediktor")
    heat_labels(axis, missing, percent=True)
    fig.colorbar(image, ax=axis, fraction=.03, label="Missing fraction (1 = 100%)")
    figures["missing_features"] = fig

    fig = Figure(figsize=(13, 4.5), layout="constrained")
    axes = fig.subplots(1, 2)
    for axis, metric, title in zip(axes, ("participant_macro_f1", "participant_score_mae"), ("Kesesuaian lima tingkat: lebih tinggi lebih baik", "Galat skor acuan: lebih rendah lebih baik")):
        for index, model in enumerate(models):
            values = [next(row[metric] for row in summaries if row["variant"] == variant and row["model"] == model) for variant in variants]
            bars = axis.bar(positions + (index - 1) * .25, values, width=.23, color=COLORS[model], label=MODEL_NAMES[model])
            axis.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)
        axis.set(xticks=positions, xticklabels=list(VARIANT_NAMES.values()), title=title,
                 ylabel="Participant-average macro-F1" if metric == "participant_macro_f1" else "Mean absolute score error (0–100 points)")
        if metric == "participant_macro_f1":
            axis.set_ylim(0, 1)
        else:
            axis.margins(y=.2)
        style_axis(axis)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncols=3)
    figures["model_agreement"] = fig

    fig = Figure(figsize=(15, 14), layout="constrained")
    axes = fig.subplots(3, 3)
    tick_labels = [LEVEL_NAMES[value].replace(" ", "\n") for value in LEVELS]
    for row_index, variant in enumerate(variants):
        for col_index, model in enumerate(models):
            axis = axes[row_index, col_index]
            counts = np.array(next(row["confusion_matrix"] for row in summaries if row["variant"] == variant and row["model"] == model))
            totals = counts.sum(axis=1, keepdims=True)
            values = np.divide(counts, totals, out=np.zeros_like(counts, dtype=float), where=totals != 0)
            image = axis.imshow(values, vmin=0, vmax=1, cmap="Blues")
            heat_labels(axis, values, percent=True, counts=counts)
            axis.set(xticks=np.arange(5), xticklabels=tick_labels, yticks=np.arange(5), yticklabels=[LEVEL_NAMES[level] for level in LEVELS],
                     xlabel="Model level", ylabel="Provisional reference level", title=f"{VARIANT_NAMES[variant]}\n{MODEL_NAMES[model]}")
    fig.colorbar(image, ax=axes, fraction=.02, label="Fraction within the reference row")
    figures["confusion_matrices"] = fig

    people = sorted({row["held_out_participant_id"] for row in metrics}, key=str.casefold)
    counts = predictions.loc[predictions.variant.eq(variants[0]) & predictions.model.eq(models[0])].participant_id.value_counts()
    values = np.array([[next(row["macro_f1_agreement"] for row in metrics if row["variant"] == variant and row["model"] == model and row["held_out_participant_id"] == person) for person in people] for variant, model in keys])
    fig = Figure(figsize=(max(11, len(people) + 5), 7), layout="constrained")
    axis = fig.subplots()
    image = axis.imshow(values, vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
    heat_labels(axis, values)
    axis.set(yticks=np.arange(9), yticklabels=row_names, xticks=np.arange(len(people)), xticklabels=[f"{person}\n(n={counts[person]})" for person in people], title="Kesesuaian lima tingkat untuk setiap peserta uji")
    fig.colorbar(image, ax=axis, fraction=.03, label="Macro-F1 agreement")
    figures["participant_agreement"] = fig

    sensitivity = pd.read_csv(directory / "family_sensitivity.csv")
    fig = Figure(figsize=(12, 7), layout="constrained")
    axis = fig.subplots()
    if sensitivity.empty:
        axis.text(.5, .5, "Diagnostik kelompok fitur dinonaktifkan pada eksperimen ini", ha="center", va="center")
        axis.set_axis_off()
    else:
        families = list(dataset["groups"])
        values = np.array([[sensitivity.loc[sensitivity.variant.eq(variant) & sensitivity.model.eq(model) & sensitivity.family.eq(family), "mae_increase"].mean() for family in families] for variant, model in keys])
        limit = max(1., float(np.nanmax(np.abs(values)))) if np.isfinite(values).any() else 1.
        image = axis.imshow(values, vmin=-limit, vmax=limit, cmap="RdBu_r", aspect="auto")
        for row in range(len(keys)):
            for column in range(len(families)):
                value = values[row, column]
                axis.text(column, row, f"{value:+.2f}" if np.isfinite(value) else "—", ha="center", va="center", fontsize=9,
                          color="white" if abs(value) > .6 * limit else "#172533")
        axis.set(yticks=np.arange(9), yticklabels=row_names, xticks=np.arange(len(families)), xticklabels=[FAMILY_NAMES[name].replace(" ", "\n") for name in families], title="Perubahan galat uji setelah satu kelompok fitur digeser")
        fig.colorbar(image, ax=axis, fraction=.03, label="Increase in reference-score MAE (points)")
    figures["feature_dependence"] = fig

    best = summaries[0]
    chosen = predictions.loc[predictions.variant.eq(best["variant"]) & predictions.model.eq(best["model"])]
    person = chosen.participant_id.value_counts().index[0]
    chosen = chosen.loc[chosen.participant_id.eq(person)].sort_values("window_end")
    times = pd.to_datetime(chosen.window_end, utc=True, format="mixed")
    minutes = (times - times.iloc[0]).dt.total_seconds() / 60
    fig = Figure(figsize=(12, 7), layout="constrained")
    top, bottom = fig.subplots(2, 1, sharex=True)
    top.plot(minutes, chosen.reference_score, label="Provisional reference", color="#737d87", linestyle="--", linewidth=1)
    top.plot(minutes, chosen.combined_score, label="Model estimate", color="#3274a7", linewidth=1)
    top.set(ylim=(0, 100), ylabel="Combined score", title=f"Peserta uji {person}: {MODEL_NAMES[best['model']]} / {VARIANT_NAMES[best['variant']]}")
    top.legend()
    bottom.plot(minutes, chosen.hrv_contribution, label=f"HRV contribution ({config['primary_hrv_weight']:.0%} weight)", color="#3469ae")
    bottom.plot(minutes, chosen.eye_contribution, label=f"Eye contribution ({1-config['primary_hrv_weight']:.0%} weight)", color="#188b77")
    bottom.set(xlabel="Minutes since first retained test window ended", ylabel="Weighted score points", ylim=(0, 100))
    bottom.legend()
    for axis in (top, bottom):
        style_axis(axis)
    figures["branch_contributions"] = fig

    fig = Figure(figsize=(11, 4.5), layout="constrained")
    axis = fig.subplots()
    for index, model in enumerate(models):
        values = [next(row["weight_label_change_fraction"] for row in summaries if row["variant"] == variant and row["model"] == model) for variant in variants]
        bars = axis.bar(positions + (index - 1) * .25, values, width=.23, color=COLORS[model], label=MODEL_NAMES[model])
        axis.bar_label(bars, labels=[f"{value:.1%}" for value in values], padding=3, fontsize=8)
    axis.set(xticks=positions, xticklabels=list(VARIANT_NAMES.values()), ylim=(0, 1), ylabel="Fraction of predictions changing level",
             title=f"Sensitivitas: bobot HRV {config['primary_hrv_weight']:.0%} dibandingkan {config['sensitivity_hrv_weight']:.0%}")
    axis.legend()
    style_axis(axis)
    figures["weight_sensitivity"] = fig
    return figures


def render_report(directory, dataset, config):
    directory = Path(directory)
    destination = directory / "visualizations"
    destination.mkdir(exist_ok=True)
    summaries = json.loads((directory / "summary.json").read_text())
    inventory = [{"family": family, "column": column, "role": "HRV predictor" if family == "HRV" else "Eye context predictor" if family == "Head context" else "Eye predictor"} for family, columns in dataset["groups"].items() for column in columns]
    inventory.extend({"family": "Signal quality", "column": column, "role": "Diagnostic only; excluded from prediction"} for column in DIAGNOSTICS)
    pd.DataFrame(inventory).to_csv(directory / "feature_inventory.csv", index=False)
    sections, paths = [], {}
    for key, figure in make_figures(directory, dataset, config).items():
        path = destination / f"{key}.png"
        figure.savefig(path, dpi=140, facecolor="white")
        figure.clear()
        paths[key] = path
        title, caption = CHARTS[key]
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        sections.append(f'<section><h2>{escape(title)}</h2><p>{escape(caption)}</p><img alt="{escape(title)}" src="data:image/png;base64,{encoded}"></section>')
    table = comparison_table(summaries).to_html(index=False, border=0, float_format=lambda value: f"{value:.3f}")
    coverage = display_frame(pd.read_csv(directory / "data_coverage.csv")).to_html(index=False, border=0)
    description = (f"{config['primary_hrv_weight']:.0%} HRV + {1-config['primary_hrv_weight']:.0%} informasi mata. "
                   "Dua cabang regresi memprediksi skor masing-masing. Skor digabungkan sebelum label Low / Mid low / Mid / Mid high / High ditentukan. "
                   "Tingkat kesulitan tugas tidak digunakan sebagai target maupun prediktor.")
    html = f'''<!doctype html><html lang="id"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Prototipe beban lima tingkat — {escape(directory.name)}</title><style>
body{{font:16px/1.6 system-ui,sans-serif;margin:0;background:#f2f6f8;color:#1b2c3c}}main{{max-width:1300px;margin:auto;padding:30px}}section,.intro{{background:white;border:1px solid #dce5eb;padding:25px;border-radius:10px;margin:24px 0}}h1,h2{{line-height:1.25}}img{{max-width:100%;height:auto}}.scroll{{overflow:auto}}table{{border-collapse:collapse;font-size:13px;width:100%}}td,th{{padding:10px;border-bottom:1px solid #e2e8ec;text-align:left}}th{{background:#eef4f7}}.notice{{border-left:5px solid #b58228;padding-left:16px}}code{{background:#eef4f7;padding:2px 5px}}</style></head><body><main>
<h1>Prototipe estimasi beban kognitif sementara lima tingkat</h1><p>{escape(directory.name)} · Jendela {dataset['window_seconds']:g} detik</p><div class="intro"><p class="notice"><strong>{escape(NOTE_ID)}</strong> Label acuan dibentuk dari kelompok sensor yang juga menjadi masukan model. Kesesuaian tinggi berarti model berhasil meniru definisi tersebut; hasil ini belum membuktikan bahwa peserta mengalami beban kognitif tinggi.</p><p>{escape(description)}</p><p>{escape(explain(summaries))}</p><h2>Cakupan data latih</h2><div class="scroll">{coverage}</div><h2>Perbandingan terhadap acuan sementara</h2><div class="scroll">{table}</div><p>Macro-F1 adalah skor kesesuaian 0–1, bukan persentase prediksi benar. MAE skor adalah galat absolut rata-rata pada skala skor 0–100. Galat urutan tingkat menghitung jarak prediksi dari tingkat acuan. Kesesuaian tepat dan selisih maksimal satu tingkat dinyatakan sebagai proporsi. Rata-rata per peserta mencegah rekaman panjang mendominasi peringkat utama.</p><p>Label tingkat (Low, Mid low, Mid, Mid high, High), nama dataset (Good only, Good + acceptable, All data), nama fitur, dan label kolom ditampilkan dalam bahasa Inggris. Penjelasan serta panduan membaca hasil tetap dalam bahasa Indonesia.</p></div>
{''.join(sections)}<section><h2>Cara kerja acuan</h2><p>Acuan empiris hanya dibentuk dari jendela data latih berkualitas baik. Setiap peserta memiliki total bobot yang sama. Rasio RMSSD terhadap acuan istirahat yang lebih rendah menghasilkan skor acuan HRV lebih tinggi. Frekuensi kedipan yang lebih rendah dan proporsi estimasi fiksasi yang lebih tinggi menghasilkan komponen acuan mata lebih tinggi; keduanya berbobot sama secara bawaan. Arah hubungan ini merupakan asumsi eksperimen. Model mata juga dapat memakai EAR, geometri iris, statistik tatapan, pola status mata, dan konteks kepala untuk memprediksi skor acuan mata. Fitur-fitur tersebut tidak semuanya diberi aturan arah beban tinggi secara sembarang.</p><p>Empat persentil berbobot dari acuan latih menghasilkan lima tingkat relatif. Pada setiap pembagian evaluasi dan validasi internal, acuan hanya dibentuk dari peserta latih. Semua varian dataset memakai acuan yang sama dalam pembagian evaluasi yang sama, serta jendela uji yang identik. Model akhir memakai acuan bersama yang dibentuk ulang dari seluruh jendela latih berkualitas baik. Skor evaluasi berasal dari model peserta uji yang terpisah, bukan dari pengujian model akhir pada data latihnya sendiri.</p><h2>Arti setiap berkas keluaran</h2><ul><li><code>reference.json</code>: distribusi empiris, batas tingkat, dan bobot acuan tetap untuk model akhir.</li><li><code>reference_labels.csv</code>: skor serta tingkat acuan yang dapat ditelusuri, termasuk baris yang tidak dapat diberi skor. Label ini merupakan hasil aturan, bukan pengukuran beban independen.</li><li><code>held_out_predictions.csv</code>: skor acuan dan prediksi tiap cabang, kontribusi berbobot, tingkat akhir, serta hasil pembobotan alternatif untuk setiap jendela uji.</li><li><code>comparison.csv</code>, <code>summary.json</code>: ringkasan kesesuaian terhadap acuan.</li><li><code>participant_metrics.csv/json</code>: hasil per peserta dan matriks kekeliruan 5×5.</li><li><code>training_audit.json</code>: peserta pembentuk acuan, pembagian validasi berdasarkan peserta, dan skor pemilihan hiperparameter tiap cabang.</li><li><code>data_coverage.csv</code>, <code>evaluation_folds.csv</code>: jumlah data yang dikeluarkan dan kelayakan tiap pembagian evaluasi.</li><li><code>feature_inventory.csv</code>, <code>family_sensitivity.csv</code>: peran fitur serta diagnostik ketergantungan model.</li><li><code>models/*.joblib</code>: sembilan paket model akhir. Masing-masing memuat model HRV, model mata, dan acuan tetap; berkas JSON pendamping menjelaskan metadatanya.</li></ul><h2>Batas penafsiran hasil</h2><p>Kalkulator acuan sudah dapat menghasilkan labelnya sendiri tanpa pembelajaran mesin. Model-model ini mempelajari pendekatan terhadap aturan tersebut; model belum menemukan ukuran beban yang tervalidasi secara independen dari rekaman tanpa label beban. Kualitas data, jumlah jendela, dan komposisi peserta berubah bersamaan. Kalibrasi mata, urutan tugas, perbedaan fisiologi, dan kondisi perangkat dapat memengaruhi sinyal. Dataset All data dapat menyertakan jendela berkualitas buruk yang masih dapat diberi skor; saat inferensi, sistem tidak mengeluarkan prediksi untuk baris yang diberi penanda kualitas buruk.</p><p>Setiap ukuran jendela memiliki acuan dan perbandingan tersendiri. Label High dari ukuran jendela yang berbeda tidak otomatis memiliki arti universal yang sama. Program ini belum memasang kebijakan adaptasi aplikasi atau memakai probabilitas sebagai ambang adaptasi. Jendela 25 detik tetap membutuhkan pengamatan 25 detik meskipun komputasi prediksinya cepat. Pengukuran beban yang independen diperlukan untuk memvalidasi akurasi terhadap beban kognitif sebenarnya.</p></section></main></body></html>'''
    report = destination / "report.html"
    report.write_text(html, encoding="utf-8")
    (directory / "REPORT.md").write_text(f"# Hasil estimasi beban sementara lima tingkat\n\n{NOTE_ID}\n\n{description}\n\n{explain(summaries)}\n\nBuka [laporan visual](visualizations/report.html) untuk melihat seluruh grafik dan penjelasannya.\n", encoding="utf-8")
    return {"report": report, "charts": paths}


def visualize_run(directory):
    directory = Path(directory).resolve()
    path = directory / "run_manifest.json"
    manifest = json.loads(path.read_text())
    if manifest.get("status") != "complete":
        raise ValueError("Only completed runs can be rendered")
    # Regenerating charts must never re-sign modified model/result artifacts.
    for name, expected in manifest.get("output_sha256", {}).items():
        if "/visualizations/" in name or name.endswith("/REPORT.md"):
            continue
        if digest(directory / name) != expected:
            raise ValueError(f"Result hash mismatch: {name}")
    outputs = {}
    for dataset_id, spec in manifest["datasets"].items():
        for filename in ("summary.json", "participant_metrics.json", "held_out_predictions.csv", "reference.json", "reference_labels.csv", "family_sensitivity.csv", "data_coverage.csv"):
            key = f"{dataset_id}/{filename}"
            if key in manifest.get("output_sha256", {}) and digest(directory / key) != manifest["output_sha256"][key]:
                raise ValueError(f"Result hash mismatch: {key}")
        dataset = load_dataset(spec["directory"], manifest["config"])
        if dataset["hashes"] != spec["input_hashes"]:
            raise ValueError("Input data no longer matches this run")
        outputs[dataset_id] = render_report(directory / dataset_id, dataset, manifest["config"])
    manifest["visualization_code_sha256"] = digest(Path(__file__))
    manifest["output_sha256"] = {str(item.relative_to(directory)).replace("\\", "/"): digest(item) for item in sorted(directory.rglob("*")) if item.is_file() and item != path}
    write_json(path, manifest)
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    for name, result in visualize_run(args.run).items():
        print(f"{name}: {result['report']}")


if __name__ == "__main__":
    main()
