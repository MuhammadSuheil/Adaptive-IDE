import copy
import hashlib
import inspect
import io
import json
import math
import platform
import sys
import tempfile
import threading
import time
import warnings
from collections import OrderedDict
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

import cloudpickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight

LABELS = ["Low", "Medium", "High"]
CORE = [
    "rmssd_ms",
    "sdnn_ms",
    "blink_rate_per_minute",
    "ear_left_mean",
    "ear_left_std",
    "ear_right_mean",
    "ear_right_std",
    "iris_size_estimate_mean",
    "iris_size_estimate_std",
    "iris_size_delta_mean",
]
HEAD = ["head_pitch_mean", "head_pitch_std", "head_yaw_mean", "head_yaw_std"]
GROUPS = {"core": CORE, "core_head": CORE + HEAD, "eye": CORE[2:], "hrv": CORE[:2]}
SEED = 42
CV_SEEDS = (42, 137)
FINAL_SEEDS = (42, 43, 44, 45, 46)
BLOCK_SECONDS = 240
SUPPORT_SECONDS = 105
CODE_VERSION = "batch04-exp02-performance-1.1.0"
BUDGET = {"tabular_trials": 40, "reference_trials": 60, "neural_trials": 30}

try:
    from IPython.display import display
except ImportError:
    display = print


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def metrics(y, labels):
    report = classification_report(
        y, labels, labels=[0, 1, 2], output_dict=True, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y, labels)),
        "macro_f1": float(
            f1_score(y, labels, labels=[0, 1, 2], average="macro", zero_division=0)
        ),
        "balanced_accuracy": float(balanced_accuracy_score(y, labels)),
        "minimum_recall": min(float(report[str(i)]["recall"]) for i in range(3)),
    }


def rank_key(record):
    score = record["validation"]
    return (
        -score["accuracy"],
        -score["macro_f1"],
        -score["minimum_recall"],
        record.get("accuracy_std", 0),
        record.get("complexity", 0),
        record["id"],
    )


def csv_bytes(frame):
    return frame.to_csv(index=False).encode("utf-8")


def array_bytes(array):
    stream = io.BytesIO()
    np.save(stream, np.asarray(array), allow_pickle=False)
    return stream.getvalue()


def figure_bytes(figure):
    stream = io.BytesIO()
    figure.savefig(stream, format="png", dpi=150, bbox_inches="tight")
    plt.close(figure)
    return stream.getvalue()
