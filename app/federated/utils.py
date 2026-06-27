"""Utilities to run a minimal multi-client federated simulation."""

from __future__ import annotations

import logging
import os
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .client import FederatedClient
from .config import FLConfig
from .server import FederatedServer

logger = logging.getLogger(__name__)

# The shared real dataset, partitioned across clients for a realistic
# (non-IID) federated run. Lives alongside the sustainability NAS data.
_BANK_DATA = (
    Path(__file__).resolve().parent.parent / "sustainability" / "Bank_data.csv"
)
# Cap total training rows so an interactive run stays in the seconds range.
_REAL_DATA_CAP = 6000


class DefaultFLModel(nn.Module):
    """Simple MLP used for federated simulation."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def default_model_builder(config: FLConfig) -> Callable[[], nn.Module]:
    return lambda: DefaultFLModel(
        input_size=config.input_size,
        hidden_size=config.hidden_size,
    )


def create_client_loaders(
    config: FLConfig,
) -> List[tuple[DataLoader, DataLoader]]:
    """Create synthetic per-client train/validation datasets."""
    rng = np.random.default_rng(config.random_seed)
    loaders: List[tuple[DataLoader, DataLoader]] = []

    for _ in range(config.number_of_clients):
        samples = config.batch_size * 4
        client_shift = rng.normal(0.0, 0.5, config.input_size)
        x = rng.normal(
            client_shift, 1.0, size=(samples, config.input_size)
        ).astype(np.float32)
        w = rng.normal(0.0, 1.0, size=(config.input_size,))
        logits = x @ w + rng.normal(0.0, 0.3, size=(samples,))
        y = (1.0 / (1.0 + np.exp(-logits)) > 0.5).astype(np.float32)

        split_idx = int(samples * (1.0 - config.validation_split))
        split_idx = max(config.batch_size, min(split_idx, samples - 1))

        x_train = torch.from_numpy(x[:split_idx])
        y_train = torch.from_numpy(y[:split_idx])
        x_val = torch.from_numpy(x[split_idx:])
        y_val = torch.from_numpy(y[split_idx:])

        train_ds = TensorDataset(x_train, y_train)
        val_ds = TensorDataset(x_val, y_val)

        loaders.append(
            (
                DataLoader(
                    train_ds,
                    batch_size=config.batch_size,
                    shuffle=True,
                ),
                DataLoader(
                    val_ds,
                    batch_size=config.batch_size,
                    shuffle=False,
                ),
            )
        )

    return loaders


def create_real_client_loaders(
    config: FLConfig,
) -> Tuple[List[tuple], int]:
    """Partition the real Bank credit dataset across clients (non-IID).

    Clients are sharded along the numeric feature most correlated with the
    target, so each holds a different slice of the population — the realistic
    federated setting where institutions serve distinct cohorts. Returns the
    per-client (train, val) loaders and the preprocessed feature dimension.
    """
    from app.sustainability.preprocessing import load_and_preprocess

    if not _BANK_DATA.exists():
        raise FileNotFoundError(f"Federated dataset missing: {_BANK_DATA}")

    X_train, _X_test, y_train, _y_test, pre = load_and_preprocess(
        str(_BANK_DATA)
    )
    if len(X_train) > _REAL_DATA_CAP:
        X_train = X_train.iloc[:_REAL_DATA_CAP]
        y_train = y_train.iloc[:_REAL_DATA_CAP]

    pre.fit(X_train)
    x = pre.transform(X_train)
    if hasattr(x, "toarray"):
        x = x.toarray()
    x = x.astype(np.float32)
    y = y_train.values.astype(np.float32)

    # Pick the numeric feature most correlated with default to shard along,
    # so the partition is genuinely non-IID (not just random quantity skew).
    sort_col, best_corr = None, 0.0
    for col in X_train.select_dtypes(include="number").columns:
        series = X_train[col].astype(float)
        if series.std() == 0:
            continue
        filled = series.fillna(series.median())
        corr = abs(np.corrcoef(filled, y_train)[0, 1])
        if not np.isnan(corr) and corr > best_corr:
            best_corr, sort_col = corr, col
    if sort_col is not None:
        filled = X_train[sort_col].astype(float)
        order = np.argsort(filled.fillna(filled.median()).values)
        x, y = x[order], y[order]

    rng = np.random.default_rng(config.random_seed)
    loaders: List[tuple] = []
    for shard in np.array_split(np.arange(len(y)), config.number_of_clients):
        idx = shard.copy()
        rng.shuffle(idx)
        split = int(len(idx) * (1.0 - config.validation_split))
        split = max(1, min(split, len(idx) - 1)) if len(idx) > 1 else len(idx)
        tr, val = idx[:split], idx[split:]

        train_ds = TensorDataset(
            torch.from_numpy(x[tr]), torch.from_numpy(y[tr])
        )
        train_loader = DataLoader(
            train_ds, batch_size=config.batch_size, shuffle=True
        )
        val_loader = None
        if len(val) > 0:
            val_ds = TensorDataset(
                torch.from_numpy(x[val]), torch.from_numpy(y[val])
            )
            val_loader = DataLoader(
                val_ds, batch_size=config.batch_size, shuffle=False
            )
        loaders.append((train_loader, val_loader))

    return loaders, int(x.shape[1])


def _build_loaders(config: FLConfig) -> Tuple[List[tuple], str, str]:
    """Prefer the real dataset; fall back to synthetic. Returns (loaders,
    data_source, dataset_label) and updates config.input_size for real data."""
    mode = os.getenv("PULSELEDGER_FEDERATED_DATA", "auto").strip().lower()
    if mode != "synthetic":
        try:
            loaders, input_dim = create_real_client_loaders(config)
            config.input_size = input_dim
            return loaders, "bank_credit", "Bank credit (non-IID)"
        except Exception as exc:
            logger.warning(
                "Real federated data unavailable, using synthetic: %s", exc
            )
    return create_client_loaders(config), "synthetic", "Synthetic"


def run_federated_simulation(
    config: Optional[FLConfig] = None,
    model_builder: Optional[Callable[[], nn.Module]] = None,
) -> Dict[str, object]:
    """Run a minimal FedAvg simulation across multiple clients."""
    config = config or FLConfig()

    # Choose data first so the model is sized to the real feature dimension.
    loaders, data_source, dataset_label = _build_loaders(config)

    if model_builder is None:
        model_builder = default_model_builder(config)

    server = FederatedServer(model_builder=model_builder)

    clients = [
        FederatedClient(
            client_id=f"client_{i+1}",
            model_builder=model_builder,
            config=config,
        )
        for i in range(config.number_of_clients)
    ]

    best_val_loss = float("inf")
    best_round = -1
    rounds_without_improvement = 0
    stopped_early = False
    best_global_state = server.global_state()

    for round_idx in range(config.aggregation_rounds):
        global_state = server.global_state()
        round_results = []

        for client, (train_loader, val_loader) in zip(clients, loaders):
            client.load_global_weights(global_state)
            round_results.append(
                client.train_local(
                    train_loader,
                    validation_loader=val_loader,
                )
            )

        round_metrics = server.aggregate_round(
            round_number=round_idx,
            client_results=round_results,
        )

        current_val = round_metrics.average_val_loss
        improvement = best_val_loss - current_val
        if improvement > config.early_stopping_min_delta:
            best_val_loss = current_val
            best_round = round_idx
            rounds_without_improvement = 0
            best_global_state = server.global_state()
        else:
            rounds_without_improvement += 1

        if (
            config.enable_early_stopping
            and rounds_without_improvement >= config.early_stopping_patience
        ):
            stopped_early = True
            break

    best_model_path = Path(config.best_model_path)
    best_model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_global_state, best_model_path)

    return {
        "config": asdict(config),
        "round_metrics": [asdict(metric) for metric in server.round_history],
        "global_keys": sorted(list(server.global_state().keys())),
        "best_round": best_round,
        "best_val_loss": float(best_val_loss),
        "stopped_early": stopped_early,
        "best_model_path": str(best_model_path),
        "data_source": data_source,
        "dataset": dataset_label,
    }


def main() -> None:
    results = run_federated_simulation()
    print("Federated simulation completed")
    print(f"Rounds: {len(results['round_metrics'])}")
    print(f"Final model keys: {len(results['global_keys'])}")
    print(f"Best round: {results['best_round']}")
    print(f"Best val loss: {results['best_val_loss']:.6f}")
    print(f"Stopped early: {results['stopped_early']}")
    print(f"Best model path: {results['best_model_path']}")
    if results["round_metrics"]:
        print("Round metrics:")
        for row in results["round_metrics"]:
            print(
                "  "
                f"round={row['round_number']}, "
                f"clients={row['participating_clients']}, "
                f"avg_loss={row['average_client_loss']:.6f}, "
                f"avg_acc={row['average_client_accuracy']:.4f}, "
                f"val_loss={row['average_val_loss']:.6f}, "
                f"val_acc={row['average_val_accuracy']:.4f}"
            )
        last = results["round_metrics"][-1]
        print(
            "Last round -> "
            f"clients: {last['participating_clients']}, "
            f"avg_loss: {last['average_client_loss']:.6f}, "
            f"avg_acc: {last['average_client_accuracy']:.4f}, "
            f"val_loss: {last['average_val_loss']:.6f}, "
            f"val_acc: {last['average_val_accuracy']:.4f}"
        )


if __name__ == "__main__":
    main()
