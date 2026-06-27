"""Federated simulation trains on the real Bank dataset (non-IID shards)."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from app.federated.config import FLConfig  # noqa: E402
from app.federated.utils import (  # noqa: E402
    create_real_client_loaders,
    run_federated_simulation,
)


def test_real_client_loaders_partition_across_clients():
    config = FLConfig(number_of_clients=4)
    loaders, input_dim = create_real_client_loaders(config)
    assert len(loaders) == 4
    assert input_dim > 0
    # Every client gets a usable training loader.
    for train_loader, _val in loaders:
        assert len(train_loader.dataset) > 0


def test_run_uses_real_data_and_sizes_model(tmp_path):
    config = FLConfig(
        number_of_clients=3,
        aggregation_rounds=3,
        local_epochs=2,
        best_model_path=str(tmp_path / "fed.pt"),
    )
    result = run_federated_simulation(config)
    assert result["data_source"] == "bank_credit"
    assert "Bank" in result["dataset"]
    # Model was sized to the real preprocessed feature dimension (not the
    # synthetic default of 20).
    assert result["config"]["input_size"] != 20
    assert len(result["round_metrics"]) >= 1
    for r in result["round_metrics"]:
        assert 0.0 <= r["average_val_accuracy"] <= 1.0


def test_synthetic_fallback_via_env(tmp_path, monkeypatch):
    monkeypatch.setenv("PULSELEDGER_FEDERATED_DATA", "synthetic")
    config = FLConfig(
        number_of_clients=2,
        aggregation_rounds=2,
        best_model_path=str(tmp_path / "fed.pt"),
    )
    result = run_federated_simulation(config)
    assert result["data_source"] == "synthetic"
    assert result["config"]["input_size"] == 20
