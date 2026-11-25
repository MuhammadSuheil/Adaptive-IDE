"""Score the official v3 Test split with the two frozen exp13 notebook recipes."""

import json

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from build_report import DATA, OUT, RUN, notebook_functions


def run():
    ns = notebook_functions("exp13")
    x, refs, meta, records, train, val, bank, hashes = ns["load_cohort"](DATA)
    manifest = json.loads((RUN / "manifest.json").read_text(encoding="utf8"))
    if any(digest != manifest["inputs"][key] for key, digest in hashes.items()):
        raise ValueError("Snapshot input differs from the frozen exp13 run.")
    official, _, _ = ns["read_bundle"](DATA, "v3")
    windows = official.loc[official.split.eq("Test")]
    people = frozenset(windows.participant_id)
    assert people.isdisjoint(train | val) and len(people) == 3
    # build_pool already matched official labelled windows against the quality-filtered v2 pool.
    keys = ["window_id", "participant_id", "session_id"]
    source = pd.read_csv(bank.parent / "all_data.csv", usecols=keys + ns["HEAD"])
    windows = windows.merge(source, on=keys, how="left", validate="one_to_one")
    assert len(windows) == 231 and np.isfinite(windows[ns["SENSORS"]]).all().all()
    test_x = windows.groupby("episode_id")[ns["SENSORS"]].median()
    known = pd.DataFrame(records).set_index("episode_id")
    test_meta = known.loc[test_x.index, meta.columns]
    assert test_meta.split.eq("Test").all() and len(test_x) == 18
    streams = ns["history_streams"](bank)
    centers = []
    for row in test_meta.itertuples():
        stream = streams.get((row.participant_id, row.session_id))
        start = pd.to_datetime(row.start_ms, unit="ms", utc=True)
        center, _, _ = ns["prior_reference"](stream, start)
        centers.append(center)
    x = pd.concat([x, test_x], verify_integrity=True)
    meta = pd.concat([meta, test_meta], verify_integrity=True)
    refs = pd.concat([refs, pd.DataFrame(centers, index=test_x.index)], verify_integrity=True)
    spread, history = ns["historical_iqr"](bank, meta, refs)
    labels = ns["fold_labels"](records, train).reindex(x.index)
    pd.testing.assert_series_equal(labels.loc[test_x.index], windows.groupby("episode_id").actual.first(),
                                   check_names=False)
    # Validation and Test share one inference batch. Only train participants enter fitting.
    frames, _ = ns["evaluate_fold"](x, refs, meta, labels, spread, train, people | val)
    both = pd.concat(frames, ignore_index=True)
    old = pd.read_csv(RUN / "episode_predictions.csv").query("protocol == 'validation'")
    order = ["recipe", "episode_id", "participant_id", "actual", "predicted"]
    check = both[both.participant_id.isin(val)].set_index(order).score.sort_index()
    pd.testing.assert_series_equal(check, old.set_index(order).score.sort_index(), rtol=0, atol=1e-12)
    pred = both[both.participant_id.isin(people)].copy().assign(protocol="Test")
    pred = pred.join(history, on="episode_id", validate="many_to_one")
    assert pred.groupby("recipe").episode_id.nunique().eq(18).all() and len(pred) == 36
    stats, _ = ns["evaluate"](pred)
    counts = []
    for name, g in pred.groupby("recipe"):
        tn, fp, fn, tp = confusion_matrix(g.actual, g.predicted, labels=ns["TWO"]).ravel()
        counts.append(dict(recipe=name, tn=tn, fp=fp, fn=fn, tp=tp, people=g.participant_id.nunique()))
    stats = stats.merge(pd.DataFrame(counts), on="recipe", validate="one_to_one")
    pred.to_csv(OUT / "test_episode_predictions.csv", index=False)
    stats.to_csv(OUT / "test_metrics.csv", index=False)
    print(stats.to_string(index=False))


if __name__ == "__main__":
    run()
