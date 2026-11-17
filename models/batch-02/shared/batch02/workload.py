"""Count work from final split manifests without benchmarking model training."""

from .registry import experiment_windows, model_bank, recipes_for


def estimate_work(experiment, manifests, *, paired_manifests=None, capabilities=None):
    result = []
    for seconds in experiment_windows(experiment):
        if experiment == "exp00":
            continue
        supports = ("native", "paired") if experiment == "exp04" else ("native",)
        for support in supports:
            source = paired_manifests if support == "paired" else manifests
            if source is None:
                result.append(
                    {
                        "window_seconds": seconds,
                        "support": support,
                        "status": "requires_paired_manifest",
                    }
                )
                continue
            result.append(
                _support_work(experiment, source, seconds, support, capabilities)
            )
    return result


def _support_work(experiment, manifests, seconds, support, capabilities):
    folds = [
        f for f in manifests["windows"][str(seconds)]["folds"] if f["status"] == "ready"
    ]
    if experiment == "exp14":
        candidates = 120
        outer, skipped = 1, []
    else:
        recipes = [r for r in recipes_for(experiment) if r.support == support]
        skipped = [
            r
            for r in recipes
            if capabilities and capabilities.get(r.model, "").startswith("skipped")
        ]
        candidates = outer = len(recipes) - len(skipped)
        if experiment == "exp09":
            candidates += len([r for r in model_bank() if r.model != "svm"])
    inner = sum(len(f["inner"]) for f in folds)
    return {
        "window_seconds": seconds,
        "support": support,
        "status": "estimated",
        "outer_folds": len(folds),
        "candidate_limit_or_count": candidates,
        "base_inner_fits": inner * candidates,
        "base_outer_refits": len(folds) * outer,
        "ineligible_candidates": [r.key for r in skipped],
        "extra_work": "fusion/binary submodels, neural stopping/refit, paired evaluation, sensitivities, and deployment are additional",
        "runtime_estimate_seconds": None,
        "runtime_estimate_reason": "No real-model training benchmark has been run.",
    }
