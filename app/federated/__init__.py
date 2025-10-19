"""
Federated learning components for privacy-preserving collaborative training.

This module provides a comprehensive federated learning framework including:
- Federated server coordination with FedAvg and FedProx algorithms
- Secure communication protocols with encryption and authentication
- Client management and selection strategies
- REST API for federated learning coordination
- Simulation utilities for testing and development
"""

from .federated_server import (
    FederatedServer,
    FederatedConfig,
    ModelUpdate,
    ClientInfo,
    FederatedRound,
    AggregationMethod,
    ClientStatus,
    FederatedAggregator,
    ClientSelector,
    SecureAggregator,
)

from .communication import (
    FederatedCommunicationManager,
    MessageType,
    FederatedMessage,
    MessageSerializer,
    SecureCommunicator,
    create_registration_message,
    create_heartbeat_message,
)

from .server_api import (
    FederatedServerAPI,
    create_federated_server_api,
    run_federated_server_async,
)

from .federated_client import (
    FederatedClient,
    ClientConfig,
    LocalTrainer,
    DifferentialPrivacyManager,
    GradientCompressor,
    TrainingResult,
    ClientStatus,
    create_federated_client,
    simulate_federated_client,
)

from .privacy_mechanisms import (
    PrivacyPreservationManager,
    PrivacyConfig,
    AdvancedDifferentialPrivacy,
    SecureAggregationProtocol,
    ConvergenceMonitor,
    AsynchronousFederatedLearning,
    PrivacyMechanism,
    PrivacyBudget,
)

from .utils import (
    FederatedClientSimulator,
    FederatedLearningSimulator,
    create_federated_simulation,
    compare_aggregation_methods,
    analyze_privacy_impact,
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
