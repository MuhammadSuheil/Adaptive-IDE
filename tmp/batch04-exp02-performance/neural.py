def build_network(recipe, input_dim):
    import torch
    from torch import nn

    cfg, family = recipe["network"], recipe["family"]
    width, depth, dropout = cfg["width"], cfg["depth"], cfg["dropout"]

    class Network(nn.Module):
        def __init__(self):
            super().__init__()
            if family == "tabm":
                from tabm import TabM

                self.body = TabM.make(
                    n_num_features=input_dim,
                    d_out=3,
                    d_block=width,
                    n_blocks=depth,
                    dropout=dropout,
                    k=16,
                )
            elif family == "gru":
                self.body = nn.GRU(
                    input_dim,
                    width,
                    num_layers=depth,
                    batch_first=True,
                    dropout=dropout if depth > 1 else 0.0,
                )
                self.output = nn.Linear(width, 3)
            elif family == "tcn":
                self.project = nn.Linear(input_dim, width)
                self.body = nn.ModuleList(
                    [
                        nn.Conv1d(width, width, 3, padding=2 * d, dilation=d)
                        for d in (1, 2, 4)
                        for _ in range(depth)
                    ]
                )
                self.drop = nn.Dropout(dropout)
                self.output = nn.Linear(width, 3)
            elif family == "transformer":
                self.project = nn.Linear(1, width)
                self.feature = nn.Parameter(torch.randn(1, input_dim, width) * 0.02)
                self.token = nn.Parameter(torch.zeros(1, 1, width))
                layer = nn.TransformerEncoderLayer(
                    width,
                    4,
                    width * 2,
                    dropout,
                    batch_first=True,
                    activation="gelu",
                    norm_first=True,
                )
                self.body = nn.TransformerEncoder(
                    layer, depth, enable_nested_tensor=False
                )
                self.output = nn.Linear(width, 3)
            else:
                self.project = nn.Linear(input_dim, width)
                self.body = nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.LayerNorm(width),
                            nn.Linear(width, width),
                            nn.GELU(),
                            nn.Dropout(dropout),
                        )
                        for _ in range(depth)
                    ]
                )
                self.output = nn.Linear(width, 3)

        def forward(self, x, lengths=None):
            if family == "tabm":
                return self.body(x)
            if family == "gru":
                packed = nn.utils.rnn.pack_padded_sequence(
                    x, lengths.cpu(), batch_first=True, enforce_sorted=False
                )
                _, hidden = self.body(packed)
                return self.output(hidden[-1])
            if family == "tcn":
                h = self.project(x).transpose(1, 2)
                for conv in self.body:
                    h = h + self.drop(
                        torch.nn.functional.gelu(conv(h)[..., : h.shape[-1]])
                    )
                return self.output(
                    h.transpose(1, 2)[
                        torch.arange(len(x), device=x.device), lengths - 1
                    ]
                )
            if family == "transformer":
                h = self.project(x.unsqueeze(-1)) + self.feature
                h = torch.cat([self.token.expand(len(x), -1, -1), h], dim=1)
                return self.output(self.body(h)[:, 0])
            h = torch.nn.functional.gelu(self.project(x))
            for block in self.body:
                h = h + block(h) if family == "residual" else block(h)
            return self.output(h)

    return Network()


def sequence_arrays(pre, frame, ids, history):
    context = history_indices(frame, history)
    groups = [context[int(i)] for i in ids]
    lengths = np.array([len(g) for g in groups], dtype=np.int64)
    max_len = int(lengths.max()) if len(lengths) else 1
    all_x = pre.transform(frame, np.arange(len(frame)))
    x = np.zeros((len(ids), max_len, all_x.shape[1]), dtype=np.float32)
    for i, group in enumerate(groups):
        x[i, : len(group)] = all_x[group]
    return x, lengths


def neural_arrays(pre, recipe, frame, ids):
    if recipe["family"] in ("gru", "tcn"):
        return sequence_arrays(pre, frame, ids, recipe["history"])
    return pre.transform(frame, ids), np.ones(len(ids), np.int64)


def neural_probabilities(network, x, lengths, batch=512):
    import torch

    device = next(network.parameters()).device
    result = []
    network.eval()
    with torch.no_grad():
        for start in range(0, len(x), batch):
            logits = network(
                torch.as_tensor(x[start : start + batch], device=device),
                torch.as_tensor(lengths[start : start + batch], device=device),
            )
            p = logits.softmax(dim=-1)
            if p.ndim == 3:
                p = p.mean(dim=1)
            result.append(p.cpu().numpy())
    return np.concatenate(result) if result else np.empty((0, 3))


def fit_neural(recipe, frame, ids, seed, epochs=None, trial=None):
    import torch
    from torch import nn

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = recipe["network"]
    ids = np.asarray(ids, int)
    fit, monitor = (
        monitor_indices(frame, ids) if epochs is None else (ids, np.array([], int))
    )
    fit_context = fit
    if recipe["family"] in ("gru", "tcn"):
        context = history_indices(frame, recipe["history"])
        fit_context = np.unique(np.concatenate([context[i] for i in fit]))
    pre = SensorTransform(recipe).fit(frame, fit_context)
    x, lengths = neural_arrays(pre, recipe, frame, fit)
    y = frame.iloc[fit]._y.to_numpy()
    network = build_network(recipe, x.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(
        network.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    maximum = int(epochs or cfg["max_epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=maximum)
    weights = None
    if recipe["balance"] == "weight":
        counts = np.bincount(y, minlength=3)
        weights = torch.tensor(
            len(y) / (3 * counts), dtype=torch.float32, device=device
        )
    loss_fn = nn.CrossEntropyLoss(weight=weights, label_smoothing=cfg["smoothing"])
    if len(monitor):
        monitor_x, monitor_lengths = neural_arrays(pre, recipe, frame, monitor)
        monitor_y = frame.iloc[monitor]._y.to_numpy()
    rng = np.random.default_rng(seed)
    best, best_state, stale, best_epoch, history = math.inf, None, 0, maximum, []
    for epoch in range(1, maximum + 1):
        network.train()
        order = rng.permutation(len(fit))
        total_loss = 0.0
        for start in range(0, len(order), cfg["batch"]):
            batch = order[start : start + cfg["batch"]]
            xx = torch.as_tensor(x[batch], device=device)
            yy = torch.as_tensor(y[batch], dtype=torch.long, device=device)
            logits = network(xx, torch.as_tensor(lengths[batch], device=device))
            if logits.ndim == 3:
                yy = yy[:, None].expand(-1, logits.shape[1]).reshape(-1)
                logits = logits.reshape(-1, 3)
            loss = loss_fn(logits, yy)
            if not torch.isfinite(loss):
                raise ValueError("Loss neural nonfinite.")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(network.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(batch)
        scheduler.step()
        row = {"epoch": epoch, "train_loss": total_loss / len(fit)}
        if len(monitor):
            p = neural_probabilities(network, monitor_x, monitor_lengths)
            monitored = float(log_loss(monitor_y, p, labels=[0, 1, 2]))
            row.update(
                monitor_loss=monitored,
                monitor_accuracy=float(accuracy_score(monitor_y, p.argmax(axis=1))),
            )
            if monitored < best - 1e-5:
                best, best_epoch, stale = monitored, epoch, 0
                best_state = {
                    k: v.detach().cpu().clone() for k, v in network.state_dict().items()
                }
            else:
                stale += 1
            if trial is not None:
                trial.report(
                    -monitored, step=trial.user_attrs.get("fold_offset", 0) + epoch
                )
                if epoch >= 20 and trial.should_prune():
                    import optuna

                    raise optuna.TrialPruned()
        history.append(row)
        if epoch == 1 or epoch % 20 == 0:
            print(f"epoch {epoch}/{maximum} loss={row['train_loss']:.4f}", flush=True)
        if len(monitor) and stale >= cfg["patience"]:
            break
    if best_state is not None:
        network.load_state_dict(best_state)
    network.to("cpu").eval()
    fit_prediction = neural_probabilities(network, x, lengths).argmax(axis=1)
    return {
        "kind": "neural",
        "recipe": copy.deepcopy(recipe),
        "pre": pre,
        "network": network,
        "input_dim": x.shape[-1],
        "best_epoch": int(best_epoch),
        "history": history,
        "training_accuracy": float(accuracy_score(y, fit_prediction)),
    }


def predict_neural(model, frame, ids):
    x, lengths = neural_arrays(model["pre"], model["recipe"], frame, ids)
    p = neural_probabilities(model["network"], x, lengths)
    return {"labels": p.argmax(axis=1), "scores": p, "score_type": "probability"}
