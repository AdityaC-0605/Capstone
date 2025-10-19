"""
Federated learning components for privacy-preserving collaborative training.

This module provides a comprehensive federated learning framework including:
- Federated server coordination with FedAvg and FedProx algorithms
- Secure communication protocols with encryption and authentication
- Client management and selection strategies
- REST API for federated learning coordination
- Simulation utilities for testing and development
"""

from .communication import (
    FederatedCommunicationManager,
    FederatedMessage,
    MessageSerializer,
    MessageType,
    SecureCommunicator,
    create_heartbeat_message,
    create_registration_message,
)
from .federated_client import (
    ClientConfig,
    ClientStatus,
    DifferentialPrivacyManager,
    FederatedClient,
    GradientCompressor,
    LocalTrainer,
    TrainingResult,
    create_federated_client,
    simulate_federated_client,
)
from .federated_server import (
    AggregationMethod,
    ClientInfo,
    ClientSelector,
    ClientStatus,
    FederatedAggregator,
    FederatedConfig,
    FederatedRound,
    FederatedServer,
    ModelUpdate,
    SecureAggregator,
)
from .privacy_mechanisms import (
    AdvancedDifferentialPrivacy,
    AsynchronousFederatedLearning,
    ConvergenceMonitor,
    PrivacyBudget,
    PrivacyConfig,
    PrivacyMechanism,
    PrivacyPreservationManager,
    SecureAggregationProtocol,
)
from .server_api import (
    FederatedServerAPI,
    create_federated_server_api,
    run_federated_server_async,
)
from .utils import (
    FederatedClientSimulator,
    FederatedLearningSimulator,
    analyze_privacy_impact,
    compare_aggregation_methods,
    create_federated_simulation,
)

__all__ = [
    # Core server components
    "FederatedServer",
    "FederatedConfig",
    "ModelUpdate",
    "ClientInfo",
    "FederatedRound",
    "AggregationMethod",
    "ClientStatus",
    "FederatedAggregator",
    "ClientSelector",
    "SecureAggregator",
    # Communication components
    "FederatedCommunicationManager",
    "MessageType",
    "FederatedMessage",
    "MessageSerializer",
    "SecureCommunicator",
    "create_registration_message",
    "create_heartbeat_message",
    # API components
    "FederatedServerAPI",
    "create_federated_server_api",
    "run_federated_server_async",
    # Client components
    "FederatedClient",
    "ClientConfig",
    "LocalTrainer",
    "DifferentialPrivacyManager",
    "GradientCompressor",
    "TrainingResult",
    "ClientStatus",
    "create_federated_client",
    "simulate_federated_client",
    # Privacy mechanisms
    "PrivacyPreservationManager",
    "PrivacyConfig",
    "AdvancedDifferentialPrivacy",
    "SecureAggregationProtocol",
    "ConvergenceMonitor",
    "AsynchronousFederatedLearning",
    "PrivacyMechanism",
    "PrivacyBudget",
    # Utilities
    "FederatedClientSimulator",
    "FederatedLearningSimulator",
    "create_federated_simulation",
    "compare_aggregation_methods",
    "analyze_privacy_impact",
]
