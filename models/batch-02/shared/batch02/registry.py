"""Deterministic candidate definitions, shared by notebooks and nested search."""

from dataclasses import asdict, dataclass, field, replace

from .artifacts import digest

HRV = ("rmssd_ms", "sdnn_ms", "rmssd_baseline_ratio")
CORE = (
    *HRV,
    "blink_rate_per_minute",
    "ear_left_mean",
    "ear_left_std",
    "ear_right_mean",
    "ear_right_std",
)
TEMPORAL = (
    "rmssd_ms",
    "sdnn_ms",
    "rmssd_baseline_ratio",
    "blink_rate_per_minute",
    "ear_left_mean",
    "ear_right_mean",
)
SENSOR_PREFIXES = (
    "rmssd_",
    "sdnn_",
    "blink_",
    "ear_",
    "iris_",
    "head_",
    "gaze_",
    "eye_state_",
)


@dataclass(frozen=True)
class Recipe:
    model: str = "logistic"
    features: str = "F1"
    baseline_ratio: bool = False
    scaler: str = "standard"
    missing: str = "median"
    normalization: str = "N0"
    history: int = 0
    selector: str = "none"
    balance: str = "none"
    indicator: bool = False
    winsor: bool = False
    thin: int = 5
    support: str = "native"
    decision: tuple = (1.0, 1.0, 1.0)
    params: dict = field(default_factory=dict)
    children: tuple = ()

    def __post_init__(self):
        defaults = {
            "logistic": {"C": 1.0},
            "svm": {"C": 1.0, "gamma": "scale"},
            "xgboost": {
                "max_depth": 3,
                "learning_rate": 0.05,
                "min_child_weight": 5,
                "reg_lambda": 5,
            },
            "hgb": {"max_leaf_nodes": 7, "learning_rate": 0.05},
            "catboost": {"depth": 4, "l2_leaf_reg": 5},
            "random_forest": {"max_depth": 5},
            "extra_trees": {"max_depth": 5},
        }
        object.__setattr__(
            self, "params", {**defaults.get(self.model, {}), **self.params}
        )
        object.__setattr__(self, "decision", tuple(self.decision))
        if self.selector == "pca" or self.balance == "smote":
            object.__setattr__(self, "scaler", "standard")

    @property
    def key(self):
        return f"{self.model}_{digest(asdict(self))[:16]}"


def feature_group(group, allowlist, baseline_ratio=True):
    if any(not n.startswith(SENSOR_PREFIXES) for n in allowlist):
        raise ValueError("Allowlist mengandung metadata/non-sensor.")
    base = [n for n in allowlist if n not in ("blink_count", "eye_state_other_ratio")]
    groups = {
        "F0": list(allowlist),
        "F1": base,
        "F2": [n for n in base if n in HRV or not n.endswith("_mean")],
        "F3": [n for n in base if not n.startswith(("gaze_", "eye_state_"))],
        "F4": [n for n in base if n in CORE],
        "F5": [n for n in base if n in HRV],
        "F6": [n for n in base if n not in HRV],
        "F7": [n for n in base if not n.startswith("iris_")],
    }
    if group not in groups:
        raise ValueError(f"Kelompok fitur tidak dikenal: {group}")
    return [n for n in groups[group] if baseline_ratio or n != "rmssd_baseline_ratio"]


def unique(recipes):
    return list({r.key: r for r in recipes}.values())


def recipe_from_dict(value):
    data = dict(value)
    data["children"] = tuple(
        recipe_from_dict(child) for child in data.get("children", ())
    )
    return Recipe(**data)


def model_bank():
    result = [Recipe(params={"C": c}) for c in (0.1, 1.0, 10.0)]
    result += [
        Recipe("svm", params={"C": c, "gamma": "scale"}) for c in (0.1, 1.0, 10.0)
    ]
    result += [Recipe("svm", params={"C": 1.0, "gamma": 0.1})]
    result += [
        Recipe("xgboost", scaler="none", params=p)
        for p in (
            {},
            {"max_depth": 2, "min_child_weight": 10},
            {
                "max_depth": 3,
                "min_child_weight": 10,
                "reg_lambda": 10,
                "learning_rate": 0.1,
            },
        )
    ]
    result += [
        Recipe(
            "hgb", scaler="none", params={"max_leaf_nodes": leaf, "learning_rate": lr}
        )
        for leaf, lr in ((7, 0.05), (15, 0.05), (7, 0.1))
    ]
    result += [
        Recipe("catboost", scaler="none", params={"depth": depth, "l2_leaf_reg": l2})
        for depth, l2 in ((4, 5), (6, 10))
    ]
    result += [
        Recipe(model, scaler="none", params={"max_depth": depth})
        for model in ("random_forest", "extra_trees")
        for depth in (5, None)
    ]
    return result


def controls():
    return [Recipe(), Recipe("xgboost", scaler="none")]


def recipes_for(experiment):
    base = controls()
    if experiment == "exp01":
        result = [
            Recipe("dummy", features="F0", baseline_ratio=True),
            Recipe(features="F0", baseline_ratio=True),
            Recipe("xgboost", features="F0", baseline_ratio=True, scaler="none"),
            Recipe(),
        ]
    elif experiment == "exp02":
        result = [
            replace(r, features=f"F{g}", baseline_ratio=ratio)
            for r in base
            for g in range(8)
            for ratio in ((False,) if g == 6 else (False, True))
        ]
    elif experiment == "exp03":
        result = base + [
            replace(r, normalization=n) for r in base for n in ("N2", "N3")
        ]
        result += [replace(r, scaler="robust") for r in base]
        result += [replace(r, indicator=True) for r in base] + [
            replace(r, winsor=True) for r in base
        ]
        result += [Recipe("xgboost", scaler="none", missing="native")]
    elif experiment == "exp04":
        result = [
            replace(r, support=support, thin=thin)
            for r in base
            for support in ("native", "paired")
            for thin in (5, 10)
        ]
    elif experiment == "exp05":
        result = [replace(r, history=h) for r in base for h in (0, 15, 30)]
    elif experiment == "exp06":
        result = [
            replace(r, selector=s)
            for r in base
            for s in ("none", "corr", "mi8", "mi12", "pca")
        ]
    elif experiment == "exp07":
        result = [
            replace(r, balance=b)
            for r in base
            for b in ("none", "weight", "ros", "smote", "recording")
        ]
        result += [
            replace(r, decision=tuple(1.25 if i == j else 1.0 for i in range(3)))
            for r in base
            for j in range(3)
        ]
    elif experiment == "exp08":
        result = model_bank()
    elif experiment == "exp09":
        result = base + [
            Recipe("late_fusion", params={"branch": model, "alpha": alpha})
            for model in ("logistic", "xgboost")
            for alpha in (0.25, 0.5, 0.75)
        ]
        result += [Recipe("vote")]
    elif experiment == "exp10":
        result = base + [Recipe("ordinal", params={"C": c}) for c in (0.1, 1.0, 10.0)]
        result += [
            Recipe("hierarchy", params={"branch": m}) for m in ("logistic", "xgboost")
        ]
    elif experiment == "exp11":
        result = [
            Recipe("mlp", features=f, params={"lr": lr})
            for f in ("F1", "F4")
            for lr in (0.001, 0.0003)
        ]
        result += [Recipe("ft_transformer", features=f) for f in ("F1", "F4")]
    elif experiment == "exp12":
        result = [
            Recipe("tabpfn", features=f, missing="native", scaler="none")
            for f in ("F1", "F4")
        ]
    elif experiment == "exp13":
        result = [
            Recipe("gru", features="F4"),
            Recipe("mlp", features="F4"),
            *[replace(r, features="F4", history=h) for r in base for h in (0, 15, 30)],
        ]
    else:
        raise ValueError(f"Keluarga eksperimen tidak dikenal: {experiment}")
    return unique(result)


def experiment_windows(experiment):
    if experiment not in {f"exp{i:02d}" for i in range(15)}:
        raise ValueError(f"Keluarga eksperimen tidak dikenal: {experiment}")
    return (15, 25, 45)
