"""Minimal federated learning package."""

from .aggregation import fedavg
from .client import ClientResult, FederatedClient
from .config import FLConfig
from .server import FederatedServer, RoundMetrics

__all__ = [
    "FLConfig",
    "ClientResult",
    "FederatedClient",
    "RoundMetrics",
    "FederatedServer",
    "fedavg",
]
