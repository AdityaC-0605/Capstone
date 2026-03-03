"""Utilities to run a minimal multi-client federated simulation."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .client import FederatedClient
from .config import FLConfig
from .server import FederatedServer


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
        x = rng.normal(client_shift, 1.0, size=(samples, config.input_size)).astype(
            np.float32
        )
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


def run_federated_simulation(
    config: Optional[FLConfig] = None,
    model_builder: Optional[Callable[[], nn.Module]] = None,
) -> Dict[str, object]:
    """Run a minimal FedAvg simulation across multiple clients."""
    config = config or FLConfig()

    if model_builder is None:
        model_builder = default_model_builder(config)

    server = FederatedServer(model_builder=model_builder)
    loaders = create_client_loaders(config)

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
