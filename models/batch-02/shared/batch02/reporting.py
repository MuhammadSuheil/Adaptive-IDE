"""Reports are derived from saved predictions, never fabricated during setup."""

from pathlib import Path

import numpy as np
import pandas as pd

from .artifacts import write_json
from .metrics import LABELS, classification_metrics
from .streams import nonoverlap_indices


def transform_metadata(bundle):
    fitted = getattr(bundle, "fitted", None)
    processor = getattr(fitted, "preprocessing", None)
    result = {"feature_names": list(bundle.feature_names)}
    if processor is not None:
        result["preprocessing"] = {
            "columns": processor.columns_,
            "medians": processor.medians_.to_dict(),
            "scaler": processor.scaler,
            "offset": processor.offset_.tolist(),
            "scale": processor.scale_.tolist(),
            "selector": processor.selector,
            "selected_indices": processor.indices_.tolist(),
            "winsor_bounds": processor.bounds_.to_dict()
            if processor.bounds_ is not None
            else None,
        }
        if processor.pca_ is not None:
            result["preprocessing"]["pca_components"] = (
                processor.pca_.components_.tolist()
            )
            result["preprocessing"]["explained_variance_ratio"] = (
                processor.pca_.explained_variance_ratio_.tolist()
            )
    if getattr(bundle, "children", ()):
        result["children"] = [transform_metadata(child) for child in bundle.children]
    if getattr(bundle, "fallback", None) is not None:
        result["fallback"] = transform_metadata(bundle.fallback)
    return result


def prediction_frame(rows, values, kind, comparison, recipe, seconds, support):
    names = [
        "window_id",
        "session_id",
        "person_id",
        "window_start",
        "window_end",
        "study_phase",
        "window_phase_mix",
        "baseline_usable",
        "gaze_validation_passed",
        "collection_wave",
    ]
    result = rows[[n for n in names if n in rows]].copy().reset_index(drop=True)
    result["y_true"], result["y_pred"] = rows.y.to_numpy(), values.argmax(1)
    (
        result["comparison"],
        result["recipe"],
        result["window_seconds"],
        result["support"],
    ) = comparison, recipe.key, seconds, support
    for k, label in enumerate(LABELS):
        result[f"{'prob' if kind == 'probability' else 'score'}_{label}"] = values[:, k]
    return result


def summarize_predictions(predictions, directory, expected):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(directory / "oof_predictions.csv", index=False)
    summaries = []
    metrics_all = {}
    for (seconds, comparison, support), rows in predictions.groupby(
        ["window_seconds", "comparison", "support"], sort=True
    ):
        if rows.window_id.duplicated().any():
            raise ValueError("Prediksi OOF berulang pada comparison/window yang sama.")
        pcols = [f"prob_{name}" for name in LABELS]
        probabilities = (
            rows[pcols].to_numpy()
            if set(pcols).issubset(rows) and rows[pcols].notna().all().all()
            else None
        )
        metrics = classification_metrics(rows.y_true, rows.y_pred, probabilities)
        complete = len(rows) == expected[(int(seconds), comparison, support)]
        key = f"{seconds}/{support}/{comparison}"
        metrics_all[key] = {**metrics, "rows": len(rows), "complete_oof": complete}
        summaries.append(
            {
                "window_seconds": seconds,
                "comparison": comparison,
                "support": support,
                "rows": len(rows),
                "complete_oof": complete,
                "macro_f1": metrics["macro_f1"],
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
            }
        )
        subdir = directory / f"window_{seconds}s" / support / comparison
        subdir.mkdir(parents=True, exist_ok=True)
        matrix = np.array(metrics["confusion_matrix"])
        pd.DataFrame(matrix, index=LABELS, columns=LABELS).to_csv(
            subdir / "confusion_matrix.csv"
        )
        pd.DataFrame(metrics["classes"]).T.to_csv(subdir / "class_metrics.csv")
        per_group = {
            person: classification_metrics(part.y_true, part.y_pred)
            for person, part in rows.groupby("person_id")
        }
        write_json(subdir / "metrics_by_person.json", per_group)
        from .diagnostics import audit_predictions

        audit_predictions(rows, subdir)
        selected = rows.reset_index(drop=True).iloc[
            nonoverlap_indices(rows.reset_index(drop=True))
        ]
        write_json(
            subdir / "nonoverlap_metrics.json",
            classification_metrics(selected.y_true, selected.y_pred),
        )
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        fig = Figure(figsize=(4.8, 4.2))
        FigureCanvasAgg(fig)
        ax = fig.subplots()
        normalized = np.divide(
            matrix,
            matrix.sum(1, keepdims=True),
            out=np.zeros((3, 3), float),
            where=matrix.sum(1, keepdims=True) > 0,
        )
        ax.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
        ax.set(
            xticks=range(3),
            yticks=range(3),
            xticklabels=LABELS,
            yticklabels=LABELS,
            xlabel="Prediksi",
            ylabel="Label",
            title=f"{seconds}s · {comparison}",
        )
        for i in range(3):
            for j in range(3):
                ax.text(
                    j,
                    i,
                    f"{matrix[i, j]}\n{normalized[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="white" if normalized[i, j] > 0.5 else "black",
                )
        figure_dir = subdir / "figures"
        figure_dir.mkdir(exist_ok=True)
        fig.tight_layout()
        fig.savefig(figure_dir / "confusion_matrix.png", dpi=160)
        fig.clear()
    write_json(directory / "metrics.json", metrics_all)
    table = pd.DataFrame(summaries)
    table.to_csv(directory / "summary.csv", index=False)
    return table


def write_run_notes(directory, table, experiment):
    lines = [
        f"# {experiment}: Hasil Evaluasi",
        "",
        "Angka berikut berasal dari prediksi OOF tersimpan. Baris dengan complete_oof=false belum menjadi hasil evaluasi lengkap.",
        "",
        "| Window | Cakupan | Konfigurasi | Row | OOF lengkap | Accuracy | Macro-F1 |",
        "|---|---|---|---:|---|---:|---:|",
    ]
    for row in table.itertuples():
        lines.append(
            f"| {row.window_seconds} | {row.support} | {row.comparison} | {row.rows} | {row.complete_oof} | {row.accuracy:.4f} | {row.macro_f1:.4f} |"
        )
    lines.extend(
        [
            "",
            "Definisi konfigurasi dan hasil pemilihan inner tersedia pada folder per-fold. metrics.json memuat metrik kelas; audit_subsets.json memuat analisis fase/ketersediaan. Grafik ada pada subfolder figures.",
            "",
            "Hasil native antarwindow mempunyai cakupan test berbeda. Evaluasi prosedur pemilihan window, bila eligible, disimpan terpisah pada joint_window_metrics.json.",
        ]
    )
    (directory / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if experiment == "exp14":
        (directory / "DIAGNOSIS.md").write_text(
            "# Pemeriksaan Hasil Batch-02\n\n"
            "Gunakan confusion matrix untuk mengidentifikasi pasangan kelas yang tertukar, metrik train/test untuk gap generalisasi, dan sensitivity.json untuk efek jumlah grup serta seed. "
            "negative_controls.json mencatat informasi pada pola missing dan warm-up; kontrol tersebut tidak menjadi kandidat final. "
            "coverage.json memisahkan ketersediaan sensor dan target tunggal.\n\n"
            "Angka-angka ini merupakan hasil observasi. Penjelasan sebab-akibat tetap memerlukan pemeriksaan sumber dan kontras eksperimen yang bersesuaian. "
            "Status serta alasan cabang yang tidak dijalankan tercatat pada search_results.json dan events.json jika tersedia.\n",
            encoding="utf-8",
        )
