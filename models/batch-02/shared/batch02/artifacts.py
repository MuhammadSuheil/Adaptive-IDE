"""Atomic, hash-addressed artifacts and explicit run state."""

import hashlib
import importlib.metadata
import json
import math
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def digest(value):
    return hashlib.sha256(
        json.dumps(clean(value), sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(clean(value), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def environment():
    names = (
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "imbalanced-learn",
        "xgboost",
        "catboost",
        "torch",
        "tabpfn",
        "joblib",
        "matplotlib",
    )
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    versions["python"] = sys.version
    versions["platform"] = platform.platform()
    versions["threadpoolctl"] = importlib.metadata.version("threadpoolctl")
    return versions


def code_hash():
    root = Path(__file__).parent
    return digest({p.name: file_hash(p) for p in sorted(root.glob("*.py"))})


def open_run(root, identity, resume):
    root = Path(root).resolve()
    key = digest(identity)
    if resume is not None:
        path = Path(resume).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Resume harus berada pada output eksperimen yang sama.")
        if read_json(path / "identity.json")["hash"] != key:
            raise ValueError(
                "Resume ditolak: hash identitas data/kode/config/environment berubah."
            )
        return path
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    path = root / f"{stamp}_{key[:12]}"
    path.mkdir(parents=True, exist_ok=False)
    write_json(path / "identity.json", {"hash": key, "identity": identity})
    write_json(path / "status.json", {"status": "not_started"})
    return path
