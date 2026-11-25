"""Build the report from frozen development runs and the outputs of evaluate_test.py."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymupdf
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

OUT = Path(__file__).resolve().parent
V3 = OUT.parent
BATCH = V3.parent
DATA = BATCH.parents[1] / "analysis_sessions/dataset_fusion/output"
RUN = V3 / "exp13/outputs/iqr_20261125T091155Z_5a63ee6f"
LOSS = V3 / "exp09/outputs/loss_20261124T120138Z_dfccaf89/loss.csv"
MODELS = {"exp11_center": "Exp11", "personal_iqr": "Exp13"}
COLORS = ["#307ca0", "#ddb55c", "#b95442"]
PROTOCOLS = {"Train_LOPO": "LOPO Train", "validation": "Validasi", "Test": "Test"}


def notebook_functions(experiment):
    nb = json.loads((V3 / experiment / "EyeHeartCognitiveLoad.ipynb").read_text(encoding="utf8"))
    ns = {}
    for i in [3, 5]:
        exec(compile("".join(nb["cells"][i]["source"]), f"{experiment}-cell-{i}", "exec"), ns)
    return ns


def loss_curves(pred):
    ns = notebook_functions("exp13")
    iteration_curve = notebook_functions("exp09")["iteration_curve"]
    x, refs, meta, records, train, val, bank, _ = ns["load_cohort"](DATA)
    spread, _ = ns["historical_iqr"](bank, meta, refs)
    labels = ns["fold_labels"](records, train).reindex(x.index)
    fit = labels.ne("") & meta.participant_id.isin(train)
    held = labels.ne("") & meta.participant_id.isin(val)
    y = ns["binary"](labels[fit])
    people = meta.loc[fit, "participant_id"]
    weights = 1 / (people.groupby(y).transform("nunique") * y.groupby([y, people]).transform("size"))
    weights /= weights.mean()
    delta = x - refs.fillna(x.loc[fit].median())
    typical = spread.loc[fit].groupby(people).median().median()
    factor = (2 / (1 + spread.div(typical.where(typical.gt(0))))).fillna(1)
    curves = []
    for name, features in [("exp11_center", delta), ("personal_iqr", delta * factor)]:
        scaler = StandardScaler().fit(features.loc[fit], sample_weight=weights)
        tr, va = scaler.transform(features.loc[fit]), scaler.transform(features.loc[held])
        model = SVC(C=1, kernel="rbf", gamma="scale").fit(tr, y, sample_weight=weights)
        held_pred = pred[(pred.protocol == "validation") & (pred.recipe == name)].set_index("episode_id").loc[x.index[held]]
        assert model.fit_status_ == 0
        np.testing.assert_allclose(ns["high_score"](model, va), held_pred.score, rtol=0, atol=1e-12)
        curve = iteration_curve(model, tr, y, weights, va, held_pred)
        if name == "exp11_center":
            np.testing.assert_allclose(curve.iloc[:, 2:], pd.read_csv(LOSS).iloc[:, 2:], atol=1e-12)
        curves.append(curve.assign(model=MODELS[name]))
    curves = pd.concat(curves, ignore_index=True)
    curves.to_csv(OUT / "training_curve.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True, layout="constrained")
    for (model, g), ax in zip(curves.groupby("model"), axes):
        for col, label, color, style in [("train_weighted", "Train berbobot", "#137963", "-"),
                                         ("train_episode", "Train per episode", "#4a6b84", "--"),
                                         ("validation_episode", "Validasi per episode", "#af463b", "-")]:
            ax.plot(g.iterations, g[col], style, color=color, label=label)
        ax.set(title=f"{model} · {len(g)} iterasi", xlabel="Iterasi solver SVM", ylim=(0, None))
        ax.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Rata-rata hinge loss")
    axes[1].legend(fontsize=9)
    save(fig, "training_curve")
    return curves.groupby("model").tail(1)


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def distributions():
    folder = BATCH / "dataset/v3_behavioral_proxy"
    splits = pd.read_csv(folder / "participant_splits.csv").sort_values(["split", "participant_id"])
    windows = folder / "window_15s_step_5s"
    clean = pd.read_csv(windows / "good_acceptable.csv", usecols=["window_id", "participant_id", "cognitive_load_label"])
    assigned = pd.read_csv(windows / "window_assignments.csv")
    rows = clean.merge(assigned, on=["window_id", "participant_id", "cognitive_load_label"], validate="one_to_one")
    assert len(rows) == len(clean) == 1193
    episodes = rows.drop_duplicates("episode_id")
    classes = ["Low", "Medium", "High"]
    counts = pd.crosstab(episodes.participant_id, episodes.cognitive_load_label).reindex(columns=classes)
    counts = splits.set_index("participant_id").join(counts).fillna(0)
    assert len(episodes) == 66 and len(counts) == 22
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.4), layout="constrained", height_ratios=[1, 2.55])
    for n, (title, frame) in enumerate([("Episode", episodes), ("Window", rows)]):
        left = 0
        for cls, col in zip(classes, COLORS):
            length = frame.cognitive_load_label.eq(cls).mean() * 100
            axes[0].barh(n, length, left=left, color=col, label=cls if n == 0 else None)
            axes[0].text(left + length/2, n, f"{length:.0f}%", ha="center", va="center",
                         color="black" if cls == "Medium" else "white", weight="bold")
            left += length
    axes[0].set(yticks=[0, 1], yticklabels=["Episode", "Window"], xlim=(0, 100), xticks=[])
    axes[0].legend(loc="upper center", bbox_to_anchor=(.5, 1.54), ncol=3, frameon=False)
    ordered = pd.concat([counts[counts.split.eq(s)] for s in ["Train", "validation", "Test"]])
    bottom = np.zeros(len(ordered))
    for cls, color in zip(classes, COLORS):
        axes[1].bar(np.arange(len(ordered)), ordered[cls], bottom=bottom, color=color, width=.8)
        bottom += ordered[cls].to_numpy()
    for boundary in (15.5, 18.5):
        axes[1].axvline(boundary, color="#7d8588", linewidth=.8)
    axes[1].set(xticks=range(22), xticklabels=[str(i) for i in range(1, 23)], ylabel="Episode layak",
                xlabel="Peserta dalam split resmi · 1–16 Train, 17–19 validasi, 20–22 Test")
    axes[1].grid(axis="y", alpha=.18)
    save(fig, "sebaran_data")
    return rows


def diagnostic_plots(pred, metrics):
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 5), layout="constrained")
    for i, (name, title) in enumerate(MODELS.items()):
        for j, (protocol, label) in enumerate(PROTOCOLS.items()):
            g = pred[(pred.recipe == name) & (pred.protocol == protocol)]
            cm = confusion_matrix(g.actual, g.predicted, labels=["Non-High", "High"])
            row = metrics[(metrics.recipe == name) & (metrics.protocol == protocol)].iloc[0]
            np.testing.assert_array_equal(cm.ravel(), [row.tn, row.fp, row.fn, row.tp])
            np.testing.assert_allclose([np.trace(cm)/cm.sum(), roc_auc_score(g.actual.eq("High"), g.score)],
                                       [row.accuracy, row.auc_pooled])
            frac = cm / cm.sum(axis=1, keepdims=True)
            ax = axes[i,j]
            ax.imshow(frac, cmap="Blues", vmin=0, vmax=1, aspect="auto")
            for (r, c), val in np.ndenumerate(cm):
                ax.text(c, r, f"{val}  ({frac[r,c]:.0%})", ha="center", va="center",
                        color="white" if frac[r,c] > .55 else "#152d45", weight="bold")
            ax.set(title=f"{title} · {label}",
                   xticks=[0,1], xticklabels=["Non-High","High"], yticks=[0,1], yticklabels=["Non-High","High"])
            if i == 1:
                ax.set_xlabel("Prediksi")
            if j == 0:
                ax.set_ylabel("Label aktual")
    save(fig, "confusion_matrix")


CSS = """body{font-family:sans-serif;font-size:11.2pt;color:#203548}
 h1{font-size:24pt;color:#113e4c;margin:0 0 8pt} h2{font-size:15pt;color:#17596b;margin:12pt 0 4pt}
 p{margin:4pt 0 7pt;line-height:1.35} table{border-collapse:collapse;width:100%;font-size:10.3pt;margin:6pt 0}
 td,th{padding:5pt 3pt;border-bottom:0.6pt solid #cddada;text-align:left}
 th{background:#e4eeee;color:#164b58}"""


def table(head, rows):
    cells = "".join("<th>" + c + "</th>" for c in head)
    body = "".join("<tr>" + "".join(f"<td>{v}</td>" for v in row) + "</tr>" for row in rows)
    return "<table><tr>" + cells + "</tr>" + body + "</table>"


def page(doc, title, sections):
    p = doc.new_page(width=595, height=842)
    p.insert_text((39, 32), "ADAPTIVE IDE     /     BATCH 06", fontsize=9, color=(.19,.39,.44))
    p.insert_text((39, 817), f"Riset beban kognitif     /     25 November 2026                                     {p.number + 1}",
                  fontsize=9, color=(.27,.37,.41))
    y = 49
    for item in [f"<h1>{title}</h1>", *sections]:
        if isinstance(item, tuple):
            image, height = item
            p.insert_image(pymupdf.Rect(38, y, 557, y+height), filename=str(OUT / image))
            y += height + 9
        else:
            spare, scale = p.insert_htmlbox((38,y,557,791), item, css=CSS, scale_low=1)
            if spare < 0 or scale != 1:
                raise ValueError(f"Page overflow: {title} at {y}")
            y = 791 - spare + 4
    assert y < 801, (title,y)


def main():
    pred = pd.concat([pd.read_csv(RUN / "episode_predictions.csv"), pd.read_csv(OUT / "test_episode_predictions.csv")])
    metrics = pd.concat([pd.read_csv(RUN / "metrics.csv"), pd.read_csv(OUT / "test_metrics.csv")])
    curves = loss_curves(pred)
    rows = distributions()
    diagnostic_plots(pred, metrics)
    with pymupdf.open() as doc:
        write_pages(doc, metrics, curves, rows)
        doc.set_metadata({"title":"Evaluasi cognitive load berbasis perilaku · Batch 06 v3",
                          "subject":"Evaluasi pengembangan, Test v3 dan perbandingan dengan rating v4"})
        doc.subset_fonts()
        doc.save(OUT / "Laporan_v3.pdf", garbage=4, deflate=True)


def dec(value, digits=1):
    return f"{value:.{digits}f}".replace(".", ",")


def write_pages(doc, metrics, curves, windows):
    rows = []
    for r in metrics.itertuples():
        rows.append([PROTOCOLS[r.protocol], MODELS[r.recipe],
                     f"{r.tp+r.tn}/{r.episodes} · {dec(100*r.accuracy)}%", dec(100*r.balanced_accuracy,2)+"%",
                     dec(r.macro_f1,3), dec(r.auc_pooled,3)])
    page(doc, "Evaluasi model beban kognitif v3", [
        "<p><b>Pengembangan dan Test resmi · Mata, HRV, dan posisi kepala</b></p>"
        "<p>Pada tiga peserta Test, exp11 benar 11/18 (61,1%) dan exp13 benar 9/18 (50,0%). Keduanya menangkap satu dari dua High. "
        "Kenaikan exp13 ke 81,8% pada validasi tidak terulang pada Test. Hasil ini belum mendukung klaim akurasi umum 78%.</p>",
        "<h2>Akurasi pada episode peserta yang disisihkan</h2>",
        table(["Evaluasi","Model","Benar · akurasi","Bal. acc.","Macro-F1","AUC"], rows),
        "<p>LOPO menahan satu peserta Train secara bergantian. Sebanyak 15 peserta menyumbang 37 episode. "
        "Validasi menahan tiga peserta sekaligus dengan 11 episode. Satu perubahan keputusan di validasi setara dengan 9,1 poin persentase. "
        "Tebakan Non-High untuk semua episode memberi 56,8% pada LOPO dan 63,6% pada validasi, dengan recall High nol.</p>",
        "<h2>Apa yang dipelajari model?</h2><p>Target High diturunkan dari waktu pengerjaan aktif relatif terhadap peserta lain pada soal sejenis dan bukti kesalahan manual. "
        "Non-High menggabungkan Low dan Medium. Model membuat satu prediksi setelah episode selesai. Hasil mengukur kesesuaian terhadap proxy perilaku. "
        "Usaha mental yang dirasakan peserta memerlukan ukuran independen.</p>",
        "<h2>Protokol Test v3</h2><p>Kedua resep dan ambang nol dikunci sebelum evaluasi biner. Fitting memakai 37 episode Train. "
        "Prediksi pada 11 episode validasi mereproduksi skor yang diarsipkan untuk kedua model. "
        "Split Test pernah dipakai pada eksperimen tiga kelas terdahulu. Dengan dua High, satu perubahan keputusan menggeser recall sebesar 50 poin persentase.</p>"])
    dist = []
    for split, assigned in [("Train",16),("validation",3),("Test",3)]:
        g = windows[windows.split == split]
        e = g.drop_duplicates("episode_id")
        dist.append([split, f"{e.participant_id.nunique()}/{assigned}", *[str(e.cognitive_load_label.eq(c).sum()) for c in ("Low","Medium","High")], len(e), len(g)])
    page(doc, "Sebaran peserta, episode, dan window", [
        "<p>Dari 358 episode tercatat, 82 mendapat label perilaku. Sebanyak 66 memiliki setidaknya satu window gabungan layak. "
        "Terdapat 1.193 window berdurasi 15 detik dengan langkah 5 detik. Satu peserta Train yang dialokasikan tidak menyumbang episode layak pada split resmi.</p>",
        table(["Split","Peserta layak/alokasi","Low","Medium","High","Episode","Window"],dist),
        ("sebaran_data.png",283),
        "<p><b>Durasi mengubah komposisi kelas.</b> High menyumbang 22/66 episode (33,3%) tetapi 826/1.193 window (69,2%). "
        "Episode panjang menghasilkan lebih banyak window yang berbagi label. Karena itu setiap episode diberi satu suara dalam evaluasi. "
        "Pada Test, menebak Non-High untuk semua episode memberi 16/18 benar (88,9%) sekaligus melewatkan dua High.</p>",
        "<p>Grafik peserta memakai nomor urut dalam split untuk berbagi hasil tanpa nama. Distribusi berasal dari label resmi Train lengkap. "
        "Pada LOPO, referensi label dihitung ulang dari peserta fitting untuk setiap fold, sehingga kelayakan tiap episode dapat berubah.</p>"])
    end_rows = [[r.model,r.iterations,dec(r.train_weighted,3),dec(r.train_episode,3),dec(r.validation_episode,3)] for r in curves.itertuples()]
    page(doc, "Kurva training dan validasi", [
        "<p>Kedua model memakai SVM kernel RBF (C = 1, gamma = scale), dilatih pada 37 episode Train. "
        "Waktu, jumlah kesalahan, identitas dan rating tidak menjadi fitur model. Sumbu mendatar menunjukkan iterasi solver. "
        "Pada tiap batas iterasi, model di-fit ulang dengan urutan data, bobot, dan parameter yang sama.</p>",
        ("training_curve.png",206),
        table(["Model","Iterasi akhir","Train berbobot","Train per episode","Validasi per episode"],end_rows),
        "<p><b>Hinge loss</b> memberi penalti pada prediksi yang salah arah maupun prediksi benar yang terlalu dekat dengan batas keputusan. "
        "Nilai lebih rendah berarti margin lebih sesuai dengan label. Kurva Train berbobot mengikuti bobot fitting. Kurva per episode memberi tiap episode bobot sama.</p>",
        "<p>Loss validasi akhir lebih tinggi daripada loss Train untuk kedua model. Exp13 mengenali lebih banyak episode validasi pada ambang nol, "
        "sementara rata-rata penalti marginnya sedikit lebih besar daripada exp11. Akurasi dan margin memberi dua informasi yang berbeda. "
        "Konvergensi solver membuktikan proses optimisasi selesai. Besarnya gap dan ukuran validasi membatasi kesimpulan tentang generalisasi.</p>",
        "<p>Skor akhir seluruh 11 episode validasi per model dicocokkan dengan run exp13 tersimpan. Semua titik kurva acuan juga dicocokkan dengan kurva exp09 yang memakai resep exp11.</p>"])
    page(doc, "Alarm palsu dan High yang terlewat", [
        "<p>Setiap sel memuat jumlah episode dan persentase terhadap label aktual pada barisnya. Ambang kedua model adalah skor keputusan nol.</p>",
        ("confusion_matrix.png",249),
        "<p><b>Test.</b> Exp11 mendeteksi 1/2 High dan memberi 6/16 alarm palsu. Exp13 mendeteksi High yang sama dan memberi 8/16 alarm palsu. "
        "Prediksi selalu Non-High mendapat akurasi 88,9%, balanced accuracy 50%, dan recall High nol. "
        "Di dalam peserta, kedua model hanya mengurutkan 1/8 pasangan High dan Non-High dengan benar. Enam pasangan berasal dari satu orang.</p>",
        "<p><b>Validasi.</b> Exp13 memperbaiki dua alarm palsu dan menangkap satu High yang dulu terlewat. Tiga koreksi terjadi pada tiga peserta berbeda. "
        "Dua alarm palsu tersisa berasal dari satu peserta. Pada LOPO, satu alarm palsu terkoreksi, satu alarm palsu baru muncul, dan satu High tambahan terlewat.</p>",
        "<p><b>Riwayat personal.</b> Exp11 mengurangi median riwayat sebelum soal. Exp13 juga menyesuaikan perubahan terhadap variasi personal (IQR) "
        "dengan minimal empat window riwayat yang saling lepas. Riwayat cukup tersedia untuk 28/37 episode LOPO dan 11/11 validasi. "
        "Di antara 28 episode dengan riwayat cukup, alarm palsu tetap 9/13 dan High terdeteksi turun dari 14/15 ke 13/15.</p>",
        "<p><b>Urutan dalam peserta.</b> Kedua model mengurutkan 12/16 pasangan High dan Non-High pada LOPO dan 7/7 di validasi dengan benar. "
        "Hanya lima peserta LOPO dan dua peserta validasi memiliki kedua kelas. Enam dari tujuh pasangan validasi berasal dari satu orang.</p>"])
    page(doc, "Apa yang berubah selama eksperimen?", [
        "<p>Perbandingan berikut memakai label biner dan sensor gabungan pada 37 episode LOPO dan 11 episode validasi. "
        "Akurasi menyatakan jumlah episode yang diprediksi benar dibagi jumlah episode evaluasi.</p>",
        table(["Resep","LOPO Train","Validasi"], [
            ["Exp07/09/11 · median per episode, SVM","26/37 · 70,3%","6/11 · 54,5%"],
            ["Exp08 · ambang dipilih dalam Train","26/37 · 70,3%","7/11 · 63,6%"],
            ["Exp10 · regresi logistik per window dari v4","11/37 · 29,7%","9/11 · 81,8%"],
            ["Exp11 B · rata-rata skor SVM per window","17/37 · 45,9%","6/11 · 54,5%"],
            ["Exp12 · HRV 15 detik pada waktu akhir berpasangan","22/37 · 59,5%","6/11 · 54,5%"],
            ["Exp12 · HRV 45 detik pada waktu akhir berpasangan","24/37 · 64,9%","6/11 · 54,5%"],
            ["Exp13 · skala variasi sensor personal","25/37 · 67,6%","9/11 · 81,8%"]]),
        "<p>Exp12 memakai HRV 15 detik sebagai cadangan bila pasangan 45 detik tidak tersedia. Pada validasi, model 45 detik menangkap 4/4 High dan "
        "memberi 5/7 alarm palsu. Akurasi 81,8% muncul juga pada transfer exp10, tetapi skor LOPO-nya jauh lebih rendah.</p>",
        "<h2>Kontribusi modalitas</h2><p>Pada ablation exp09, hanya posisi kepala memberi 28/37 benar pada LOPO dan 7/11 pada validasi. "
        "Mata + HRV tanpa posisi kepala menghasilkan 20/37 dan 6/11. Model gabungan lebih baik mengurutkan pasangan di dalam peserta pada LOPO "
        "(12/16 versus 9/16 untuk kepala saja). Gerakan yang terkait konteks tugas masih dapat memengaruhi skor.</p>",
        "<h2>Jejak model tiga kelas</h2><p>Extra Trees exp03 memperoleh 5/11 pada validasi dan 4/18 (22,2%) pada Test. "
        "Pada eksperimen tim exp01, regresi logistik tiga kelas yang dikunci memperoleh 7/11 dan 6/18 (33,3%). "
        "Menggabungkan Medium ke Non-High mengubah pertanyaan evaluasi sehingga angka ini dilaporkan terpisah.</p>"])
    v4 = json.loads((BATCH / "outputs/exp03_v4_binary_overload/binary_deepdive_summary.json").read_bytes())
    target = "full_1_6_vs_7_10"
    locked = v4["dev_to_heldout_test_summary"][target]
    v4_results = {"Train_LOPO": ("LOPO penuh", v4["best_lopo_operational_per_task"][target], 23),
                  "validation": ("Dev LOPO*", locked["dev_lopo"], 18),
                  "Test": ("Test", locked["held_out_test"], 5)}
    comparison = []
    for protocol, title in PROTOCOLS.items():
        v4_title, v, positives = v4_results[protocol]
        for r in metrics[metrics.protocol.eq(protocol)].itertuples():
            comparison.append([title, "V3 " + MODELS[r.recipe], f"{r.tp+r.tn}/{r.episodes} · {dec(100*r.accuracy)}%",
                               dec(100*r.balanced_accuracy,2)+"%", dec(r.auc_pooled,3),
                               f"{r.tp}/{r.tp+r.fn}", f"{r.fp}/{r.fp+r.tn}"])
        n = v["n_episodes"]
        negatives = n - positives
        comparison.append([v4_title, "V4 Exp03", f"{round(n*v['ep_acc'])}/{n} · {dec(100*v['ep_acc'])}%",
                           dec(100*v["ep_bal_acc"],2)+"%", dec(v["ep_auc"],3),
                           f"{round(positives*v['ep_sens'])}/{positives}",
                           f"{round(negatives*(1-v['ep_spec']))}/{negatives}"])
    page(doc, "Perbandingan v3 dan v4", [
        "<p><b>LOPO, validasi, dan Test per episode.</b> V3 mendefinisikan High dari perilaku pengerjaan. "
        "V4 mendefinisikan overload dari rating kesulitan 7–10 dan non-overload dari rating 1–6.</p>",
        table(["Evaluasi","Model","Benar · akurasi","Bal. acc.","AUC","High benar","Alarm palsu"], comparison),
        "<p><b>* Validasi v4.</b> Tim memakai validasi silang LOPO pada 13 peserta pengembangan (99 episode) untuk memilih model yang kemudian diuji pada empat peserta Test. "
        "V4 tidak memiliki split validasi terpisah seperti tiga peserta validasi v3. Baris LOPO penuh v4 merangkum 17 peserta (126 episode), sedangkan LOPO Train v3 mencakup 15 peserta dengan episode layak.</p>",
        "<p>AUC menilai urutan skor pasangan positif–negatif. Pada LOPO penuh v4, AUC 0,781 (interval 95% 0,647–0,894). "
        "Menaikkan ambang dari 0,50 ke 0,65 meningkatkan akurasi dari 69,8% ke 80,2%, sementara overload terdeteksi turun dari 16/23 ke 11/23. "
        "Selalu memprediksi non-overload memberi 103/126 benar dan melewatkan semua overload.</p>",
        "<p>Test v3 memuat dua High perilaku dari 18 episode. Test v4 memuat lima overload dari 27 episode. "
        "Model Test v4 memakai sembilan fitur inti, sedangkan kandidat terbaik LOPO penuh v4 juga memakai posisi kepala dan baseline.</p>",
        "<h2>Skor v3 terhadap rating peserta yang sama</h2><p>Di antara 21 episode LOPO v3 dengan rating, tujuh berating ≥7. "
        "Exp11 mencapai AUC rating 0,827 (Spearman 0,525) dan exp13 0,786 (Spearman 0,393). "
        "Pada validasi, keduanya mencapai AUC rating 0,778, dengan hanya satu rating ≥7 dari sepuluh episode. "
        "Pencocokan ini dilakukan setelah prediksi dan memberi bukti terbatas tentang kesesuaian skor perilaku dengan pengalaman peserta.</p>",
        "<p><b>Sumber Batch 06.</b> Metrik exp13, v3/report/test_metrics.csv, dan ringkasan biner tim di outputs/exp03_v4_binary_overload/binary_deepdive_summary.json.</p>"])


if __name__ == "__main__":
    main()
