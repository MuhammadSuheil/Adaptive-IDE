class FeatureCache(OrderedDict):
    def __init__(self, limit_bytes=128 * 1024**2):
        super().__init__()
        self.limit_bytes = limit_bytes
        self.bytes_used = 0

    @staticmethod
    def size(value):
        if isinstance(value, np.ndarray):
            return max(value.nbytes, sys.getsizeof(value))
        if isinstance(value, (list, tuple)):
            return sys.getsizeof(value) + sum(FeatureCache.size(v) for v in value)
        return sys.getsizeof(value)

    def __getitem__(self, key):
        value = super().__getitem__(key)
        self.move_to_end(key)
        return value

    def __setitem__(self, key, value):
        if key in self:
            self.bytes_used -= self.size(super().__getitem__(key)) + self.size(key)
            super().__delitem__(key)
        size = self.size(key) + self.size(value)
        if size > self.limit_bytes:
            return
        while self and self.bytes_used + size > self.limit_bytes:
            old_key, old_value = self.popitem(last=False)
            self.bytes_used -= self.size(old_key) + self.size(old_value)
        super().__setitem__(key, value)
        self.bytes_used += size

    def __deepcopy__(self, memo):
        return type(self)(self.limit_bytes)


def feature_cache(frame):
    cache = frame.attrs.get("_cl_features")
    if not isinstance(cache, FeatureCache):
        cache = frame.attrs["_cl_features"] = FeatureCache()
    return cache


def frame_signature(frame, columns=()):
    return hashlib.sha256(
        frame[list(columns)].to_numpy(float).tobytes()
        + frame._end.to_numpy(dtype=np.int64).tobytes()
        + pd.util.hash_pandas_object(frame.session_id, index=False).to_numpy().tobytes()
        + pd.util.hash_pandas_object(frame.index, index=False).to_numpy().tobytes()
    ).hexdigest()


_uncached_history_indices = history_indices


def history_indices(frame, seconds):
    key = ("history_indices", frame_signature(frame), seconds)
    cache = feature_cache(frame)
    if key not in cache:
        value = _uncached_history_indices(frame, seconds)
        cache[key] = value
        return value
    return cache[key]


def available_groups(frame):
    head_ready = set(HEAD).issubset(frame.columns)
    if head_ready:
        head_ready = bool(
            np.isfinite(
                frame[HEAD].apply(pd.to_numeric, errors="coerce").to_numpy(float)
            ).all()
        )
    return ["core", "core_head"] if head_ready else ["core"]


def represent(frame, recipe):
    columns = GROUPS[recipe["group"]]
    dependencies = list(
        dict.fromkeys(columns + (CORE if recipe.get("derived") else []))
    )
    fingerprint = frame_signature(frame, dependencies)
    key = (
        id(frame),
        fingerprint,
        tuple(columns),
        recipe.get("history", 0),
        recipe.get("representation", "current"),
        recipe.get("derived", False),
    )
    cache = feature_cache(frame)
    if key in cache:
        return cache[key]
    raw = frame[columns].to_numpy(float)
    if not np.isfinite(raw).all():
        raise ValueError(
            f"Fitur {recipe['group']} tidak lengkap; jangan imputasi gaze gagal."
        )
    parts, names = [raw], list(columns)
    if recipe.get("derived"):
        source = frame[CORE]
        extra = np.column_stack(
            [
                source.rmssd_ms / np.maximum(source.sdnn_ms, 1e-6),
                2
                * (source.ear_left_mean - source.ear_right_mean)
                / np.maximum(source.ear_left_mean + source.ear_right_mean, 1e-6),
                (source.ear_left_std + source.ear_right_std)
                / np.maximum(source.ear_left_mean + source.ear_right_mean, 1e-6),
                source.iris_size_estimate_std
                / np.maximum(source.iris_size_estimate_mean, 1e-6),
            ]
        )
        parts.append(extra)
        names += ["hrv_ratio", "ear_gap", "ear_cv", "iris_cv"]
    if recipe.get("representation") == "history":
        context = history_indices(frame, recipe["history"])
        median, dispersion, trend = [], [], []
        for i, ids in enumerate(context):
            past = ids[:-1]
            values = raw[past] if len(past) >= 2 else raw[[i]]
            median.append(np.median(values, axis=0))
            dispersion.append(np.std(values, axis=0))
            trend.append((values[-1] - values[0]) / (max(1, len(values) - 1) * 5))
        parts += [np.asarray(median), np.asarray(dispersion), np.asarray(trend)]
        names += (
            ["past_median_" + c for c in columns]
            + ["past_std_" + c for c in columns]
            + ["past_slope_" + c for c in columns]
        )
    result = (np.concatenate(parts, axis=1), names)
    cache[key] = result
    return result


class SensorTransform:
    def __init__(self, recipe):
        self.recipe = copy.deepcopy(recipe)

    def _matrix(self, frame):
        columns = GROUPS[self.recipe["group"]]
        dependencies = list(
            dict.fromkeys(columns + (CORE if self.recipe.get("derived") else []))
        )
        key = (
            "matrix",
            frame_signature(frame, dependencies),
            self.recipe["group"],
            self.recipe["normalization"],
            self.recipe.get("history", 0),
            self.recipe.get("representation", "current"),
            self.recipe.get("derived", False),
        )
        cache = feature_cache(frame)
        if key in cache:
            x, self.names = cache[key]
            return x
        x, self.names = represent(frame, self.recipe)
        x = x.copy()
        normalization = self.recipe["normalization"]
        if normalization == "log_robust":
            for column in ("rmssd_ms", "sdnn_ms"):
                if column in self.names:
                    pos = self.names.index(column)
                    x[:, pos] = np.log1p(x[:, pos])
        if normalization == "causal":
            raw = frame[GROUPS[self.recipe["group"]]].to_numpy(float)
            for i, ids in enumerate(history_indices(frame, self.recipe["history"])):
                if len(ids) >= 3:
                    x[i, : raw.shape[1]] -= np.median(raw[ids[:-1]], axis=0)
        x.setflags(write=False)
        cache[key] = (x, self.names)
        return x

    def fit(self, frame, ids):
        ids = np.asarray(ids, int)
        x = self._matrix(frame)
        self.global_center = np.median(x[ids], axis=0)
        self.global_scale = np.maximum(
            np.quantile(x[ids], 0.75, axis=0) - np.quantile(x[ids], 0.25, axis=0), 1e-6
        )
        self.centers, self.scales = {}, {}
        sid = frame.session_id.to_numpy()
        for session in np.unique(sid[ids]):
            subset = x[ids[sid[ids] == session]]
            self.centers[session] = np.median(subset, axis=0)
            scale = np.quantile(subset, 0.75, axis=0) - np.quantile(
                subset, 0.25, axis=0
            )
            self.scales[session] = np.where(scale > 1e-6, scale, self.global_scale)
        normalized = self._normalize(frame, x)
        self.scaler = (
            RobustScaler()
            if self.recipe["normalization"]
            in ("robust", "log_robust", "session_robust")
            else StandardScaler()
        ).fit(normalized[ids])
        scaled = self.scaler.transform(normalized[ids])
        self.selector = None
        if self.recipe.get("selector") == "mi8":
            self.selector = SelectKBest(
                lambda a, b: mutual_info_classif(a, b, random_state=SEED),
                k=min(8, scaled.shape[1]),
            ).fit(scaled, frame.iloc[ids]._y)
        elif self.recipe.get("selector") == "pca95":
            self.selector = PCA(n_components=0.95, svd_solver="full").fit(scaled)
        return self

    def _normalize(self, frame, x):
        if self.recipe["normalization"] not in ("session_center", "session_robust"):
            return x
        center = np.stack(
            [self.centers.get(s, self.global_center) for s in frame.session_id]
        )
        result = x - center
        if self.recipe["normalization"] == "session_robust":
            result /= np.stack(
                [self.scales.get(s, self.global_scale) for s in frame.session_id]
            )
        return result

    def transform(self, frame, ids):
        x = self.scaler.transform(
            self._normalize(frame, self._matrix(frame))[np.asarray(ids, int)]
        )
        if self.selector is not None:
            x = self.selector.transform(x)
        if not np.isfinite(x).all():
            raise ValueError("Hasil transformasi nonfinite.")
        return x.astype(np.float32)
