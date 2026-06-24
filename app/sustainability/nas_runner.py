"""
A bounded, *real* carbon-aware NAS run for the interactive API.

This trains actual ScalableMLP architectures on a real dataset and scores each
against the carbon-cost proxy + stability constraints — the same machinery as
the full research runner (``run_nas.py``), but with a reduced search space,
fewer epochs and a capped training sample so it finishes in seconds and can be
driven from a background API job. It does no plotting or printing.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent

# A small but genuine slice of the full grid: a few capacities, one early exit,
# and both a full- and low-precision mode so the Pareto trade-off is visible.
QUICK_SEARCH_SPACE = [
    {"hidden_scale": 1.0},
    {"hidden_scale": 0.6},
    {"hidden_scale": 0.35},
]
QUICK_EXIT_LEVELS = [2]
QUICK_PRECISION_MODES = ["fp32", "int8"]


def _architecture_label(hidden_scale: float) -> str:
    h1 = max(32, int(256 * hidden_scale))
    h2 = max(16, int(128 * hidden_scale))
    h3 = max(8, int(64 * hidden_scale))
    return f"MLP {h1}-{h2}-{h3}"


def _make_evaluate_fn(X_train_t, y_train_t, y_test_series, epochs: int):
    import torch
    from sklearn.metrics import brier_score_loss, roc_auc_score
    from torch.utils.data import DataLoader, TensorDataset

    from .metrics import ks_statistic

    def evaluate_model_fn(model, X_tensor, exit_level, precision):
        model.train()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=1e-3, weight_decay=1e-4
        )
        loss_fn = torch.nn.BCEWithLogitsLoss()
        loader = DataLoader(
            TensorDataset(X_train_t, y_train_t),
            batch_size=256,
            shuffle=True,
            drop_last=True,  # keep BatchNorm happy on the final batch
        )

        best_loss = float("inf")
        patience, patience_counter = 3, 0
        for _ in range(epochs):
            epoch_loss, n_batches = 0.0, 0
            for x_batch, y_batch in loader:
                optimizer.zero_grad()
                logits = model(x_batch.float(), exit_level=exit_level)
                loss = loss_fn(logits, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            avg_loss = epoch_loss / max(1, n_batches)
            if avg_loss < best_loss - 1e-4:
                best_loss, patience_counter = avg_loss, 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        model.eval()
        with torch.no_grad():
            logits = model(X_tensor.float(), exit_level=exit_level)
            probs = torch.sigmoid(logits).cpu().numpy()

        metrics = {
            "auc": float(roc_auc_score(y_test_series, probs)),
            "ks": float(ks_statistic(y_test_series.values, probs)),
            "brier": float(brier_score_loss(y_test_series, probs)),
            "val_loss": float(best_loss),
        }
        return probs, metrics

    return evaluate_model_fn


def run_quick_nas(
    epochs: int = 6,
    train_cap: int = 4000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Train a small grid of architectures and return scored candidates.

    Returns a JSON-serialisable dict. Raises on missing dependencies/data so
    the caller can report a clean error.
    """
    import numpy as np
    import torch

    from .carbon_aware_nas import carbon_aware_nas
    from .preprocessing import load_and_preprocess
    from .reference_model import REFERENCE_MODELS

    started = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)

    data_path = DATA_DIR / "Bank_data.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"NAS dataset not found: {data_path}")

    X_train, X_test, y_train, y_test, preprocessor = load_and_preprocess(
        str(data_path)
    )
    preprocessor.fit(X_train)
    X_train_p = preprocessor.transform(X_train)
    X_test_p = preprocessor.transform(X_test)
    if hasattr(X_train_p, "toarray"):
        X_train_p = X_train_p.toarray()
    if hasattr(X_test_p, "toarray"):
        X_test_p = X_test_p.toarray()

    # Cap the training sample so an interactive run stays in the seconds range.
    if train_cap and X_train_p.shape[0] > train_cap:
        idx = np.random.RandomState(seed).choice(
            X_train_p.shape[0], size=train_cap, replace=False
        )
        X_train_p = X_train_p[idx]
        y_train_capped = y_train.iloc[idx]
    else:
        y_train_capped = y_train

    X_train_t = torch.tensor(X_train_p, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_capped.values, dtype=torch.float32)
    X_test_t = torch.tensor(X_test_p, dtype=torch.float32)

    evaluate_fn = _make_evaluate_fn(X_train_t, y_train_t, y_test, epochs)
    reference = REFERENCE_MODELS["logistic_regression"]

    results = carbon_aware_nas(
        X_test_t,
        y_test,
        reference,
        preprocessing_latency_ms=12.27,
        exit_latencies_ms={1: 0.10, 2: 0.20, 3: 0.25},
        evaluate_model_fn=evaluate_fn,
        verbose=False,
        search_space=QUICK_SEARCH_SPACE,
        exit_levels=QUICK_EXIT_LEVELS,
        precision_modes=QUICK_PRECISION_MODES,
    )

    fallback = bool(results and results[0].get("constraint_violation"))
    candidates: List[Dict[str, Any]] = []
    for r in results:
        m = r["metrics"]
        scale = r["architecture"]["hidden_scale"]
        candidates.append(
            {
                "architecture": _architecture_label(scale),
                "hidden_scale": scale,
                "exit_level": r["exit_level"],
                "precision": r["precision"],
                "auc": round(m["auc"], 4),
                "ks": round(m["ks"], 4),
                "brier": round(m["brier"], 4),
                "val_loss": round(m.get("val_loss", 0.0), 4),
                "carbon_cost": round(r["carbon_cost"], 2),
                "passes": not fallback,
            }
        )

    return {
        "dataset": "Bank",
        "configs_tested": (
            len(QUICK_SEARCH_SPACE)
            * len(QUICK_EXIT_LEVELS)
            * len(QUICK_PRECISION_MODES)
        ),
        "passed_constraints": 0 if fallback else len(candidates),
        "fallback": fallback,
        "reference": reference,
        "epochs": epochs,
        "train_samples": int(X_train_t.shape[0]),
        "candidates": candidates,
        "elapsed_seconds": round(time.time() - started, 2),
    }


def _safe_run_quick_nas(
    epochs: int = 6, train_cap: int = 4000
) -> Dict[str, Any]:
    """Wrapper that converts any failure into a structured error dict."""
    try:
        return {"status": "done", **run_quick_nas(epochs, train_cap)}
    except Exception as exc:  # pragma: no cover - depends on torch/data
        logger.error("Quick NAS failed: %s", exc)
        return {"status": "error", "error": str(exc)}


def latest_result_holder() -> Dict[str, Optional[Any]]:
    """A tiny mutable holder the API uses to stash the most recent run."""
    return {"state": "idle", "result": None, "started_at": None}
