"""
Federated Learning Utilities.

This module provides utility functions for federated learning operations,
including client simulation, model aggregation testing, and performance monitoring.
"""

import json
import logging
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

try:
    from ..core.logging import get_logger
    from ..models.dnn_model import DNNConfig, DNNModel
    from .communication import FederatedCommunicationManager, MessageType
    from .federated_server import (
        AggregationMethod,
        ClientInfo,
        FederatedConfig,
        FederatedRound,
        FederatedServer,
        ModelUpdate,
    )
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).parent.parent))

    from core.logging import get_logger
    from federated.communication import (
        FederatedCommunicationManager,
        MessageType,
    )
    from federated.federated_server import (
        AggregationMethod,
        ClientInfo,
        FederatedConfig,
        FederatedRound,
        FederatedServer,
        ModelUpdate,
    )
    from models.dnn_model import DNNConfig, DNNModel

logger = get_logger(__name__)


class SimpleTestModel(nn.Module):
    """Simple test model for federated learning simulation."""

    def __init__(self, input_size: int = 20, hidden_size: int = 64):
        super(SimpleTestModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class FederatedClientSimulator:
    """Simulates federated clients for testing and development."""

    def __init__(
        self,
        client_id: str,
        data_size: int,
        model_config: Optional[DNNConfig] = None,
    ):
        self.client_id = client_id
        self.data_size = data_size
        self.model_config = model_config or DNNConfig()
        self.local_model: Optional[nn.Module] = None
        self.local_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self.training_history: List[Dict[str, float]] = []

        # Performance characteristics
        self.connection_quality = random.uniform(0.7, 1.0)
        self.computational_power = random.uniform(
            0.5, 1.5
        )  # Relative to baseline
        self.privacy_budget = random.uniform(5.0, 10.0)

        logger.debug(
            f"Created client simulator {client_id} with {data_size} samples"
        )

    def generate_synthetic_data(
        self, input_size: int = 20, noise_level: float = 0.1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate synthetic training data for the client.

        Args:
            input_size: Number of input features
            noise_level: Amount of noise to add to data

        Returns:
            Tuple of (features, labels)
        """
        # Generate base data with client-specific characteristics
        np.random.seed(hash(self.client_id) % 2**32)

        # Create client-specific data distribution
        client_bias = np.random.normal(0, 0.5, input_size)

        X = np.random.normal(client_bias, 1.0, (self.data_size, input_size))

        # Add noise to simulate real-world data heterogeneity
        X += np.random.normal(0, noise_level, X.shape)

        # Generate labels with some client-specific pattern
        weights = np.random.normal(0, 1, input_size)
        y_prob = 1 / (
            1
            + np.exp(-(X @ weights + np.random.normal(0, 0.1, self.data_size)))
        )
        y = (y_prob > 0.5).astype(float)

        self.local_data = (torch.FloatTensor(X), torch.FloatTensor(y))

        logger.debug(
            f"Generated {self.data_size} synthetic samples for client {self.client_id}"
        )
        return self.local_data

    def initialize_local_model(
        self, global_weights: Dict[str, torch.Tensor]
    ) -> bool:
        """
        Initialize local model with global weights.

        Args:
            global_weights: Global model weights

        Returns:
            True if successful, False otherwise
        """
        try:
            # Create simple local model for testing
            self.local_model = SimpleTestModel(input_size=20, hidden_size=64)

            # Load global weights
            if global_weights:
                with torch.no_grad():
                    for name, param in self.local_model.named_parameters():
                        if name in global_weights:
                            param.copy_(global_weights[name])

            logger.debug(
                f"Initialized local model for client {self.client_id}"
            )
            return True

        except Exception as e:
            logger.error(
                f"Failed to initialize local model for client {self.client_id}: {e}"
            )
            return False

    def train_local_model(
        self, epochs: int = 5, learning_rate: float = 0.01
    ) -> ModelUpdate:
        """
        Train the local model and create a model update.

        Args:
            epochs: Number of local training epochs
            learning_rate: Learning rate for training

        Returns:
            ModelUpdate object
        """
        if self.local_model is None or self.local_data is None:
            raise ValueError("Local model or data not initialized")

        start_time = datetime.now()

        # Setup training
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.SGD(
            self.local_model.parameters(), lr=learning_rate
        )

        X_train, y_train = self.local_data

        # Training loop
        self.local_model.train()
        epoch_losses = []

        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.local_model(X_train)
            loss = criterion(outputs.squeeze(), y_train)
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        # Calculate training metrics
        training_time = (datetime.now() - start_time).total_seconds()
        training_loss = np.mean(epoch_losses)

        # Simulate validation metrics
        self.local_model.eval()
        with torch.no_grad():
            val_outputs = self.local_model(X_train)
            val_probs = torch.sigmoid(val_outputs.squeeze())
            val_preds = (val_probs > 0.5).float()
            accuracy = (val_preds == y_train).float().mean().item()

        validation_metrics = {"accuracy": accuracy, "loss": training_loss}

        # Simulate energy consumption (based on computational power and training time)
        base_energy = 0.1  # kWh per hour baseline
        energy_consumed = (
            base_energy * (training_time / 3600) * self.computational_power
        )

        # Get model weights
        model_weights = {
            name: param.clone().detach()
            for name, param in self.local_model.named_parameters()
        }

        # Create model update
        model_update = ModelUpdate(
            client_id=self.client_id,
            round_number=0,  # Will be set by server
            model_weights=model_weights,
            data_size=self.data_size,
            training_loss=training_loss,
            validation_metrics=validation_metrics,
            training_time=training_time,
            energy_consumed=energy_consumed,
        )

        # Add to training history
        self.training_history.append(
            {
                "epoch": len(self.training_history),
                "loss": training_loss,
                "accuracy": accuracy,
                "training_time": training_time,
                "energy_consumed": energy_consumed,
            }
        )

        logger.debug(
            f"Client {self.client_id} completed local training: "
            f"loss={training_loss:.4f}, accuracy={accuracy:.4f}"
        )

        return model_update

    def apply_differential_privacy(
        self, model_update: ModelUpdate, epsilon: float = 1.0
    ) -> ModelUpdate:
        """
        Apply differential privacy to model update.

        Args:
            model_update: Original model update
            epsilon: Privacy parameter

        Returns:
            Model update with differential privacy applied
        """
        if epsilon <= 0:
            logger.warning(f"Invalid epsilon value: {epsilon}")
            return model_update

        # Calculate noise scale based on epsilon
        sensitivity = 1.0  # Simplified sensitivity calculation
        noise_scale = sensitivity / epsilon

        # Add noise to model weights
        noisy_weights = {}
        for name, weight in model_update.model_weights.items():
            noise = torch.normal(0, noise_scale, weight.shape)
            noisy_weights[name] = weight + noise

        # Update model update
        model_update.model_weights = noisy_weights
        model_update.differential_privacy_applied = True
        model_update.epsilon_used = epsilon

        # Update privacy budget
        self.privacy_budget -= epsilon

        logger.debug(
            f"Applied differential privacy to client {self.client_id} "
            f"with epsilon={epsilon}, remaining budget={self.privacy_budget:.2f}"
        )

        return model_update


class FederatedLearningSimulator:
    """Simulates a complete federated learning scenario."""

    def __init__(self, config: Optional[FederatedConfig] = None):
        self.config = config or FederatedConfig()
        self.server = FederatedServer(self.config)
        self.clients: Dict[str, FederatedClientSimulator] = {}
        self.simulation_results: List[Dict[str, Any]] = []

    def add_client(
        self,
        client_id: str,
        data_size: int,
        model_config: Optional[DNNConfig] = None,
    ) -> bool:
        """
        Add a simulated client to the federation.

        Args:
            client_id: Unique client identifier
            data_size: Size of client's training data
            model_config: Model configuration

        Returns:
            True if successful, False otherwise
        """
        try:
            # Create client simulator
            client_sim = FederatedClientSimulator(
                client_id, data_size, model_config
            )
            self.clients[client_id] = client_sim

            # Register with server
            success = self.server.register_client(
                client_id=client_id,
                public_key="dummy_public_key",  # Simplified for simulation
                ip_address="127.0.0.1",
                port=8000 + len(self.clients),
                capabilities={"data_size": data_size},
            )

            if success and client_id in self.server.clients:
                self.server.clients[client_id].data_size = data_size

            logger.info(f"Added client {client_id} with {data_size} samples")
            return success

        except Exception as e:
            logger.error(f"Failed to add client {client_id}: {e}")
            return False

    def initialize_global_model(self, input_size: int = 20) -> bool:
        """
        Initialize the global model.

        Args:
            input_size: Number of input features

        Returns:
            True if successful, False otherwise
        """
        try:
            # Create simple global model for testing
            global_model = SimpleTestModel(
                input_size=input_size, hidden_size=64
            )

            # Set global model in server
            success = self.server.set_global_model(global_model)

            if success:
                logger.info(
                    f"Initialized global model with {input_size} input features"
                )

            return success

        except Exception as e:
            logger.error(f"Failed to initialize global model: {e}")
            return False

    def generate_client_data(
        self, input_size: int = 20, noise_level: float = 0.1
    ) -> bool:
        """
        Generate synthetic data for all clients.

        Args:
            input_size: Number of input features
            noise_level: Amount of noise to add to data

        Returns:
            True if successful, False otherwise
        """
        try:
            for client_id, client_sim in self.clients.items():
                client_sim.generate_synthetic_data(input_size, noise_level)

                # Initialize local model with global weights
                global_weights = self.server.get_global_model_weights()
                client_sim.initialize_local_model(global_weights)

            logger.info(
                f"Generated synthetic data for {len(self.clients)} clients"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to generate client data: {e}")
            return False

    def simulate_federated_round(
        self,
        round_number: int,
        local_epochs: int = 5,
        apply_dp: bool = False,
        dp_epsilon: float = 1.0,
    ) -> Optional[FederatedRound]:
        """
        Simulate a complete federated learning round.

        Args:
            round_number: Round number
            local_epochs: Number of local training epochs
            apply_dp: Whether to apply differential privacy
            dp_epsilon: Differential privacy parameter

        Returns:
            FederatedRound object if successful, None otherwise
        """
        try:
            logger.info(f"Starting federated round {round_number}")

            # Start round on server
            round_info = self.server.start_federated_round()
            if round_info is None:
                logger.warning("Failed to start federated round")
                return None

            # Get participating clients
            participating_clients = round_info.participating_clients
            client_updates = []

            # Simulate client training
            for client_id in participating_clients:
                if client_id in self.clients:
                    client_sim = self.clients[client_id]

                    # Update local model with latest global weights
                    global_weights = self.server.get_global_model_weights()
                    client_sim.initialize_local_model(global_weights)

                    # Train local model
                    model_update = client_sim.train_local_model(
                        epochs=local_epochs, learning_rate=0.01
                    )

                    # Set round number
                    model_update.round_number = round_number

                    # Apply differential privacy if requested
                    if apply_dp and client_sim.privacy_budget > dp_epsilon:
                        model_update = client_sim.apply_differential_privacy(
                            model_update, dp_epsilon
                        )

                    client_updates.append(model_update)

            # Complete round on server
            success = self.server.complete_federated_round(client_updates)

            if success:
                completed_round = self.server.round_history[-1]

                # Record simulation results
                round_results = {
                    "round_number": round_number,
                    "participating_clients": len(participating_clients),
                    "total_data_size": completed_round.total_data_size,
                    "average_loss": completed_round.aggregated_metrics.get(
                        "average_loss", 0.0
                    ),
                    "round_duration": completed_round.get_duration(),
                    "total_energy_consumed": completed_round.total_energy_consumed,
                    "differential_privacy_applied": apply_dp,
                }

                self.simulation_results.append(round_results)

                logger.info(
                    f"Completed federated round {round_number}: "
                    f"avg_loss={round_results['average_loss']:.4f}, "
                    f"duration={round_results['round_duration']:.2f}s"
                )

                return completed_round
            else:
                logger.error(
                    f"Failed to complete federated round {round_number}"
                )
                return None

        except Exception as e:
            logger.error(f"Error in federated round {round_number}: {e}")
            return None

    def run_simulation(
        self,
        num_rounds: int = 10,
        local_epochs: int = 5,
        apply_dp: bool = False,
        dp_epsilon: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Run a complete federated learning simulation.

        Args:
            num_rounds: Number of federated rounds
            local_epochs: Number of local training epochs per round
            apply_dp: Whether to apply differential privacy
            dp_epsilon: Differential privacy parameter

        Returns:
            Simulation results dictionary
        """
        logger.info(
            f"Starting federated learning simulation with {num_rounds} rounds"
        )

        start_time = datetime.now()
        successful_rounds = 0

        for round_num in range(num_rounds):
            round_result = self.simulate_federated_round(
                round_number=round_num,
                local_epochs=local_epochs,
                apply_dp=apply_dp,
                dp_epsilon=dp_epsilon,
            )

            if round_result is not None:
                successful_rounds += 1

                # Check for convergence
                if self.server.check_convergence():
                    logger.info(f"Convergence achieved at round {round_num}")
                    break
            else:
                logger.warning(f"Round {round_num} failed")

        total_time = (datetime.now() - start_time).total_seconds()

        # Compile final results
        final_results = {
            "simulation_config": {
                "num_clients": len(self.clients),
                "num_rounds": num_rounds,
                "successful_rounds": successful_rounds,
                "local_epochs": local_epochs,
                "differential_privacy": apply_dp,
                "dp_epsilon": dp_epsilon if apply_dp else None,
            },
            "performance_metrics": {
                "total_simulation_time": total_time,
                "average_round_time": (
                    total_time / successful_rounds
                    if successful_rounds > 0
                    else 0
                ),
                "convergence_achieved": self.server.metrics[
                    "convergence_achieved"
                ],
                "total_energy_consumed": self.server.metrics[
                    "total_energy_consumed"
                ],
            },
            "round_results": self.simulation_results,
            "server_status": self.server.get_server_status(),
        }

        logger.info(
            f"Simulation completed: {successful_rounds}/{num_rounds} rounds successful"
        )
        return final_results

    def plot_simulation_results(self, save_path: Optional[str] = None):
        """
        Plot simulation results.

        Args:
            save_path: Optional path to save the plot
        """
        if not self.simulation_results:
            logger.warning("No simulation results to plot")
            return

        try:
            import matplotlib.pyplot as plt

            # Extract data for plotting
            rounds = [r["round_number"] for r in self.simulation_results]
            losses = [r["average_loss"] for r in self.simulation_results]
            energy = [
                r["total_energy_consumed"] for r in self.simulation_results
            ]

            # Create subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

            # Plot loss convergence
            ax1.plot(
                rounds, losses, "b-", marker="o", linewidth=2, markersize=6
            )
            ax1.set_xlabel("Federated Round")
            ax1.set_ylabel("Average Training Loss")
            ax1.set_title("Federated Learning Convergence")
            ax1.grid(True, alpha=0.3)

            # Plot energy consumption
            ax2.plot(
                rounds, energy, "g-", marker="s", linewidth=2, markersize=6
            )
            ax2.set_xlabel("Federated Round")
            ax2.set_ylabel("Energy Consumed (kWh)")
            ax2.set_title("Energy Consumption per Round")
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                logger.info(f"Simulation plot saved to {save_path}")

            plt.show()

        except ImportError:
            logger.warning(
                "Matplotlib not available, skipping plot generation"
            )
        except Exception as e:
            logger.error(f"Error plotting simulation results: {e}")


# Utility functions
def create_federated_simulation(
    num_clients: int = 5,
    data_sizes: Optional[List[int]] = None,
    config: Optional[FederatedConfig] = None,
) -> FederatedLearningSimulator:
    """
    Create a federated learning simulation with multiple clients.

    Args:
        num_clients: Number of clients to create
        data_sizes: List of data sizes for each client
        config: Federated learning configuration

    Returns:
        Configured FederatedLearningSimulator
    """
    if data_sizes is None:
        # Generate random data sizes
        data_sizes = [random.randint(100, 1000) for _ in range(num_clients)]

    if len(data_sizes) != num_clients:
        raise ValueError("Number of data sizes must match number of clients")

    # Create simulator
    simulator = FederatedLearningSimulator(config)

    # Add clients
    for i in range(num_clients):
        client_id = f"client_{i+1}"
        simulator.add_client(client_id, data_sizes[i])

    return simulator


def compare_aggregation_methods(
    num_clients: int = 5, num_rounds: int = 10
) -> Dict[str, Any]:
    """
    Compare different federated aggregation methods.

    Args:
        num_clients: Number of clients
        num_rounds: Number of rounds to simulate

    Returns:
        Comparison results
    """
    methods = [AggregationMethod.FEDAVG, AggregationMethod.FEDPROX]
    results = {}

    for method in methods:
        logger.info(f"Testing aggregation method: {method.value}")

        # Create configuration for this method
        config = FederatedConfig(
            aggregation_method=method,
            max_rounds=num_rounds,
            min_clients_per_round=2,
            max_clients_per_round=num_clients,
        )

        # Create and run simulation
        simulator = create_federated_simulation(num_clients, config=config)
        simulator.initialize_global_model()
        simulator.generate_client_data()

        simulation_results = simulator.run_simulation(num_rounds)
        results[method.value] = simulation_results

    return results


def analyze_privacy_impact(
    epsilon_values: List[float] = [0.1, 1.0, 5.0, 10.0],
    num_clients: int = 5,
    num_rounds: int = 10,
) -> Dict[str, Any]:
    """
    Analyze the impact of differential privacy on federated learning.

    Args:
        epsilon_values: List of epsilon values to test
        num_clients: Number of clients
        num_rounds: Number of rounds

    Returns:
        Privacy impact analysis results
    """
    results = {}

    # Baseline without differential privacy
    logger.info("Running baseline simulation without differential privacy")
    baseline_sim = create_federated_simulation(num_clients)
    baseline_sim.initialize_global_model()
    baseline_sim.generate_client_data()
    baseline_results = baseline_sim.run_simulation(num_rounds, apply_dp=False)
    results["baseline"] = baseline_results

    # Test different epsilon values
    for epsilon in epsilon_values:
        logger.info(f"Testing differential privacy with epsilon={epsilon}")

        dp_sim = create_federated_simulation(num_clients)
        dp_sim.initialize_global_model()
        dp_sim.generate_client_data()
        dp_results = dp_sim.run_simulation(
            num_rounds, apply_dp=True, dp_epsilon=epsilon
        )
        results[f"epsilon_{epsilon}"] = dp_results

    return results
