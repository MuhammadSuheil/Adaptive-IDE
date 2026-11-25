"""Post-hoc comparison of already-held-out v3 scores with separately recorded v4 ratings."""

import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


def compare(predictions, ratings):
    rows = []
    for (protocol, model), group in predictions.groupby(["protocol", "model"]):
        data = group.merge(ratings, on=["episode_id", "participant_id"], how="left", validate="one_to_one")
        rated = data.dropna(subset=["difficulty_rating"])
        high = rated.difficulty_rating.ge(7)
        rho = (spearmanr(rated.score, rated.difficulty_rating).statistic
               if rated.score.nunique() > 1 and rated.difficulty_rating.nunique() > 1 else None)
        rows.append(dict(protocol=protocol, model=model, eligible=len(group), matched=len(rated),
                         participants=rated.participant_id.nunique(), rating_7_10=high.sum(),
                         spearman=rho, auc_rating_7_10=roc_auc_score(high, rated.score) if high.nunique() == 2 else None))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("ratings", type=Path)
    args = parser.parse_args()
    ratings = pd.read_csv(args.ratings, usecols=["episode_id", "participant_id", "difficulty_rating"])
    print(compare(pd.read_csv(args.predictions), ratings).to_csv(index=False), end="")
