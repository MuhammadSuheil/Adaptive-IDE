"""Source cells following exp08 load_cohort in the standalone notebook."""

import warnings
import matplotlib.pyplot as plt
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

RECIPES = {"exp08_rbf": ("rbf", SENSORS), "linear_all": ("linear", SENSORS),
           "rbf_eye_hrv": ("rbf", CORE), "rbf_head": ("rbf", HEAD)}


def fold_data(x, refs, meta, records, fitting, held):
    if fitting & held:
        raise ValueError("Peserta fitting dan evaluasi tumpang tindih.")
    labels = fold_labels(records, fitting).reindex(x.index)
    fit_ids = labels.index[labels.ne("") & meta.participant_id.isin(fitting)]
    held_ids = labels.index[labels.ne("") & meta.participant_id.isin(held)]
    y = binary(labels.loc[fit_ids])
    people = meta.participant_id.loc[fit_ids]
    w = 1 / (people.groupby(y).transform("nunique") * y.groupby([y, people]).transform("size"))
    w /= w.mean()
    relative = x - refs.fillna(x.loc[fit_ids].median())
    scaler = StandardScaler().fit(relative.loc[fit_ids], sample_weight=w)
    def scaled(ids):
        return pd.DataFrame(scaler.transform(relative.loc[ids]), index=ids, columns=x.columns)
    if not len(held_ids):
        return None
    results = meta.loc[held_ids, ["participant_id"]].assign(original=labels[held_ids], actual=binary(labels[held_ids]))
    return scaled(fit_ids), y, w, scaled(held_ids), results


def high_score(model, x):
    return model.decision_function(x) * (1 if model.classes_[1] == "High" else -1)


def hinge(y, margin):
    return np.maximum(0, 1 - np.where(y.eq("High"), 1, -1) * margin)


def iteration_curve(final_model, x, y, w, x_val, val):
    # Each capped refit starts with identical data, order, scaling, weights and solver settings.
    n = int(final_model.n_iter_[0])
    rows = []
    for cap in range(1, n + 1):
        if cap < n:
            model = SVC(kernel="rbf", C=1, gamma="scale", max_iter=cap)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                model.fit(x, y, sample_weight=w)
        else:
            model = final_model
        train_loss = hinge(y, high_score(model, x))
        val_loss = hinge(val.actual, high_score(model, x_val))
        rows.append(dict(iterations=int(model.n_iter_[0]), converged=model.fit_status_ == 0,
                         train_weighted=np.average(train_loss, weights=w), train_episode=train_loss.mean(),
                         validation_episode=val_loss.mean()))
    return pd.DataFrame(rows)


def plot_loss(curve, output):
    with plt.rc_context({"font.size": 11, "font.family": "DejaVu Sans"}):
        fig, ax = plt.subplots(figsize=(9, 5.2), layout="constrained")
        for column, label, color, style in [
                ("validation_episode", "Validation · 11 episode", "#ae392a", "-"),
                ("train_episode", "Train · 37 episode", "#4f6880", "--"),
                ("train_weighted", "Train · bobot fitting", "#086653", "-")]:
            ax.plot(curve.iterations, curve[column], label=label, color=color, linestyle=style, linewidth=2.2)
            ax.scatter(curve.iterations.iloc[-1], curve[column].iloc[-1], color=color, s=27, zorder=4)
        ax.axhline(1, color="#697078", linestyle=":", linewidth=1, label="Skor konstan nol")
        ax.set(xlabel="Iterasi solver SVC", ylabel="Rata-rata hinge loss", ylim=(0, None))
        ax.set_title("Loss model Eye + HRV + posisi kepala", loc="left", weight="bold", pad=12)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.18)
        ax.legend(frameon=False, fontsize=10, ncol=2)
        fig.savefig(output, dpi=180, bbox_inches="tight")
        plt.show()
        plt.close(fig)


def evaluate(predictions):
    summary, paired = [], []
    for (protocol, recipe), g in predictions.groupby(["protocol", "recipe"]):
        if g.episode_id.duplicated().any():
            raise ValueError("Prediksi episode berulang.")
        actual = g.actual.eq("High")
        aucs, counts = [], []
        for participant, part in g.groupby("participant_id"):
            high = part.actual.eq("High")
            n_pairs = int(high.sum() * (~high).sum())
            if n_pairs:
                auc = roc_auc_score(high, part.score)
                aucs.append(auc)
                counts.append(n_pairs)
                paired.append(dict(protocol=protocol, recipe=recipe, participant_id=participant,
                                   pairs=n_pairs, concordance=auc))
        summary.append(dict(protocol=protocol, recipe=recipe, episodes=len(g),
                            accuracy=g.actual.eq(g.predicted).mean(),
                            balanced_accuracy=balanced_accuracy_score(g.actual, g.predicted),
                            macro_f1=f1_score(g.actual, g.predicted, average="macro"),
                            auc_pooled=roc_auc_score(actual, g.score), hinge=hinge(g.actual, g.score).mean(),
                            paired_people=len(aucs), within_person_pairs=sum(counts),
                            concordance_person_mean=np.mean(aucs) if aucs else np.nan,
                            concordance_pair_mean=np.average(aucs, weights=counts) if aucs else np.nan))
    return pd.DataFrame(summary), pd.DataFrame(paired)


def run(dataset_root, output_root):
    x, refs, meta, records, train, validation, _, hashes = load_cohort(dataset_root)
    folds = [("Train_LOPO", p, train - {p}, {p}) for p in sorted(train)]
    folds.append(("validation", "validation", train, validation))
    predictions = []
    for protocol, fold, fitting, held in folds:
        data = fold_data(x, refs, meta, records, fitting, held)
        if data is None:
            continue
        fit_x, y, weights, held_x, result = data
        for recipe, (kernel, features) in RECIPES.items():
            model = SVC(C=1, kernel=kernel, gamma="scale").fit(fit_x[features], y, sample_weight=weights)
            if model.fit_status_:
                raise ValueError("Solver belum konvergen.")
            score = high_score(model, held_x[features])
            predictions.append(result.assign(protocol=protocol, fold=fold, recipe=recipe, score=score,
                                             predicted=model.predict(held_x[features])).reset_index())
            if protocol == "validation" and recipe == "exp08_rbf":
                curve = iteration_curve(model, fit_x[features], y, weights, held_x[features], result)
    pred = pd.concat(predictions, ignore_index=True)
    summary, pairs = evaluate(pred)
    out = output_root / ("loss_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8])
    out.mkdir(parents=True, exist_ok=False)
    plot_loss(curve, out / "loss.png")
    for name, frame in [("loss", curve), ("metrics", summary), ("episode_predictions", pred),
                        ("within_person", pairs)]:
        frame.to_csv(out / (name + ".csv"), index=False)
    save_json(out / "manifest.json", {"inputs": hashes, "sklearn": sklearn.__version__, "pandas": pd.__version__,
              "numpy": np.__version__, "recipes": RECIPES, "C": 1, "fit_tol": 0.001, "gamma": "scale"})
    print(curve.tail(1).to_string(index=False))
    print(summary.to_string(index=False))
    archive = shutil.make_archive(str(out), "zip", root_dir=out.parent, base_dir=out.name)
    print("ZIP:", archive)
    return Path(archive)
