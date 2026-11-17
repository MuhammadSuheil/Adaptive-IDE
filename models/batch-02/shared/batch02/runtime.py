"""Environment setup/preflight without fitting or checkpoint downloads."""

import importlib
import importlib.metadata
import importlib.util
import subprocess
import sys
from pathlib import Path


def requirements(batch):
    core, optional = [], {}
    for raw in (
        (Path(batch) / "requirements.txt").read_text(encoding="utf-8").splitlines()
    ):
        if raw.startswith("# optional-"):
            group, requirement = raw[2:].split(": ", 1)
            optional.setdefault(group.removeprefix("optional-"), []).append(requirement)
        elif raw.strip() and not raw.startswith("#"):
            core.append(raw.strip())
    return core, optional


def ensure_environment(batch, install=False, optional=()):
    core, groups = requirements(batch)
    wanted = core + [spec for group in optional for spec in groups[group]]
    missing = []
    module_names = {"scikit-learn": "sklearn", "imbalanced-learn": "imblearn"}
    for specification in wanted:
        name, version = specification.split("==")
        try:
            present = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            present = None
        loaded_version = getattr(
            sys.modules.get(module_names.get(name, name)), "__version__", None
        )
        if (
            loaded_version is not None
            and present is not None
            and str(loaded_version) != present
        ):
            raise RuntimeError(
                f"Restart runtime: modul {name} yang dimuat berbeda dari paket terpasang."
            )
        if present is None or present.split("+", 1)[0] != version:
            missing.append(specification)
    if missing and install:
        loaded = any(
            module_names.get(spec.split("==")[0], spec.split("==")[0]) in sys.modules
            for spec in missing
        )
        subprocess.run([sys.executable, "-m", "pip", "install", *wanted], check=True)  # nosec B603
        if loaded:
            raise RuntimeError(
                "Dependency berubah setelah modul dimuat. Restart runtime sekali, lalu jalankan notebook dari awal."
            )
        importlib.invalidate_caches()
        return {"status": "installed", "requirements": wanted}
    return {
        "status": "ready" if not missing else "dependency_mismatch",
        "missing": missing,
        "python": sys.version,
    }


def require_core_environment(batch):
    result = ensure_environment(batch)
    if result["status"] != "ready":
        raise RuntimeError(
            f"Dependency belum sesuai requirements.txt: {result['missing']}. Jalankan cell Environment terlebih dahulu."
        )


def capabilities(config):
    result = {
        "mlp": "skipped_dependency",
        "gru": "skipped_dependency",
        "ft_transformer": "skipped_dependency",
        "tabpfn": "skipped_dependency",
    }
    _, optional = requirements(config.batch)
    torch_version = optional["neural"][0].split("==")[1]
    if (
        importlib.util.find_spec("torch")
        and importlib.metadata.version("torch").split("+", 1)[0] == torch_version
    ):
        import torch

        result.update(
            mlp="eligible_subject_to_support", gru="eligible_subject_to_support"
        )
        result["ft_transformer"] = "eligible_subject_to_support"
        result["neural_device"] = "cuda" if torch.cuda.is_available() else "cpu"
        if (
            importlib.util.find_spec("tabpfn")
            and importlib.metadata.version("tabpfn")
            == optional["tabpfn"][0].split("==")[1]
        ):
            result["tabpfn"] = (
                "eligible_subject_to_checkpoint"
                if torch.cuda.is_available()
                else "skipped_hardware"
            )
            if config.tabpfn_checkpoint is None or not config.tabpfn_license_confirmed:
                result["tabpfn"] = "skipped_checkpoint"
    return result
