"""Count work from final split manifests without benchmarking model training."""

from .registry import experiment_windows, model_bank, recipes_for


def estimate_work(experiment, manifests):
    result = []
    for seconds in experiment_windows(experiment):
        if experiment == "exp00":
            continue
        folds = [
            f
            for f in manifests["windows"][str(seconds)]["folds"]
            if f["status"] == "ready"
        ]
        if experiment == "exp14":
            candidates = 120
        else:
            candidates = len(recipes_for(experiment))
            if experiment == "exp09":
                candidates += len([r for r in model_bank() if r.model != "svm"])
        inner = sum(len(f["inner"]) for f in folds)
        result.append(
            {
                "window_seconds": seconds,
                "outer_folds": len(folds),
                "candidate_limit_or_count": candidates,
                "base_inner_fits": inner * candidates,
                "base_outer_refits": len(folds)
                * (1 if experiment == "exp14" else candidates),
                "extra_work": "fusion/binary submodels, neural stopping/refit, paired evaluation, sensitivities, and deployment are additional",
                "runtime_estimate_seconds": None,
                "runtime_estimate_reason": "No real-model training benchmark has been run.",
            }
        )
    return result
