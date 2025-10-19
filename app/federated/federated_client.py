"""
Federated Learning Client Implementation.

This module implements the federated client for privacy-preserving collaborative training.
It includes local training, encrypted model update transmission, client-side differential privacy,
and gradient compression for communication efficiency.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import asyncio
import aiohttp
import json
import time
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import logging
import copy
import gzip
import base64

# Cryptographic imports
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

try:
    from .federated_server import ModelUpdate, FederatedConfig
    from .communication import (
        FederatedCommunicationManager,
        MessageType,
        FederatedMessage,
        MessageSerializer,
        SecureCommunicator,
        create_registration_message,
    )
    from ..core.logging import get_logger, get_audit_logger
    from ..core.interfaces import BaseModel, TrainingMetrics
    from ..data.cross_validation import validate_model_cv
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).parent.parent))

    from federated.federated_server import ModelUpdate, FederatedConfig
    from federated.communication import (
        FederatedCommunicationManager,
        MessageType,
        FederatedMessage,
        MessageSerializer,
        SecureCommunicator,
        create_registration_message,
    )
    from core.logging import get_logger, get_audit_logger
    from core.interfaces import BaseModel, TrainingMetrics

logger = get_logger(__name__)
audit_logger = get_audit_logger()


class ClientStatus(Enum):
    """Client status enumeration."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    TRAINING = "training"
    UPLOADING = "uploading"
    WAITING = "waiting"
    ERROR = "error"


@dataclass
class ClientConfig:
    """Configuration for federated client."""

    # Client identification
    client_id: str
    server_host: str = "localhost"
    server_port: int = 8080

    # Training parameters
    local_epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 0.01
    optimizer: str = "sgd"  # "sgd", "adam", "adamw"

    # Privacy parameters
    enable_differential_privacy: bool = True
    dp_epsilon: float = 1.0
    dp_delta: float = 1e-5
    dp_clip_norm: float = 1.0
    privacy_budget: float = 10.0

    # Communication parameters
    enable_compression: bool = True
    compression_ratio: float = 0.1  # For gradient compression
    max_retries: int = 3
    timeout_seconds: int = 30
    heartbeat_interval: int = 60  # seconds

    # Security parameters
    enable_encryption: bool = True
    verify_server_certificate: bool = True

    # Performance parameters
    use_gpu: bool = True
    mixed_precision: bool = True
    gradient_accumulation_steps: int = 1

    # Data parameters
    data_path: Optional[str] = None
    validation_split: float = 0.2
    shuffle_data: bool = True

    # Logging and monitoring
    log_level: str = "INFO"
    save_local_model: bool = False
    local_model_path: str = "models/local"


@dataclass
class TrainingResult:
    """Result of local training."""

    success: bool
    model_update: Optional[ModelUpdate] = None
    training_loss: float = 0.0
    validation_metrics: Dict[str, float] = field(default_factory=dict)
    training_time: float = 0.0
    energy_consumed: float = 0.0
    data_size: int = 0
    epochs_completed: int = 0
    convergence_achieved: bool = False
    message: str = ""


class DifferentialPrivacyManager:
    """Manages differential privacy for federated client."""

    def __init__(self, config: ClientConfig):
        self.config = config
        self.privacy_budget_used = 0.0
        self.noise_multiplier = self._calculate_noise_multiplier()

    def _calculate_noise_multiplier(self) -> float:
        """Calculate noise multiplier based on privacy parameters."""
        # Simplified calculation - in practice, use more sophisticated methods
        # like the moments accountant or RDP accountant
        if self.config.dp_epsilon <= 0:
            return float("inf")

        # Basic noise multiplier calculation
        return (
            self.config.dp_clip_norm
            * np.sqrt(2 * np.log(1.25 / self.config.dp_delta))
            / self.config.dp_epsilon
        )

    def clip_gradients(self, model: nn.Module) -> float:
        """
        Clip gradients to bound sensitivity.

        Args:
            model: Model with gradients to clip

        Returns:
            Gradient norm before clipping
        """
        total_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2

        total_norm = total_norm ** (1.0 / 2)

        # Clip gradients
        clip_coef = self.config.dp_clip_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.data.mul_(clip_coef)

        return total_norm

    def add_noise(self, model: nn.Module) -> float:
        """
        Add differential privacy noise to gradients.

        Args:
            model: Model to add noise to

        Returns:
            Amount of noise added
        """
        if not self.config.enable_differential_privacy:
            return 0.0

        if self.privacy_budget_used >= self.config.privacy_budget:
            logger.warning("Privacy budget exhausted, not adding noise")
            return 0.0

        noise_scale = self.noise_multiplier * self.config.dp_clip_norm
        total_noise = 0.0

        for param in model.parameters():
            if param.grad is not None:
                noise = torch.normal(
                    0, noise_scale, param.grad.shape, device=param.grad.device
                )
                param.grad.data.add_(noise)
                total_noise += noise.norm().item()

        # Update privacy budget
        self.privacy_budget_used += self.config.dp_epsilon

        logger.debug(
            f"Added DP noise: scale={noise_scale:.4f}, total_noise={total_noise:.4f}"
        )
        return total_noise

    def get_remaining_budget(self) -> float:
        """Get remaining privacy budget."""
        return max(0.0, self.config.privacy_budget - self.privacy_budget_used)

    def can_participate(self) -> bool:
        """Check if client can participate in training (has privacy budget)."""
        return self.get_remaining_budget() >= self.config.dp_epsilon


class GradientCompressor:
    """Handles gradient compression for communication efficiency."""

    def __init__(self, config: ClientConfig):
        self.config = config

    def compress_gradients(
        self, model: nn.Module
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """
        Compress gradients for efficient transmission.

        Args:
            model: Model with gradients to compress

        Returns:
            Tuple of (compressed_gradients, compression_metadata)
        """
        if not self.config.enable_compression:
            # No compression - return original gradients
            gradients = {
                name: param.grad.clone()
                for name, param in model.named_parameters()
                if param.grad is not None
            }
            return gradients, {"compression_method": "none"}

        # Top-k compression
        return self._topk_compression(model)

    def _topk_compression(
        self, model: nn.Module
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """Apply top-k gradient compression."""
        compressed_gradients = {}
        compression_metadata = {
            "compression_method": "topk",
            "compression_ratio": self.config.compression_ratio,
            "shapes": {},
            "indices": {},
        }

        for name, param in model.named_parameters():
            if param.grad is not None:
                grad = param.grad.data.flatten()
                k = max(1, int(len(grad) * self.config.compression_ratio))

                # Get top-k elements by magnitude
                _, indices = torch.topk(torch.abs(grad), k)
                compressed_grad = torch.zeros_like(grad)
                compressed_grad[indices] = grad[indices]

                # Store compressed gradient and metadata
                compressed_gradients[name] = compressed_grad.reshape(param.grad.shape)
                compression_metadata["shapes"][name] = param.grad.shape
                compression_metadata["indices"][name] = indices.cpu().numpy()

        return compressed_gradients, compression_metadata

    def decompress_gradients(
        self, compressed_gradients: Dict[str, torch.Tensor], metadata: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        """
        Decompress gradients (if needed for local operations).

        Args:
            compressed_gradients: Compressed gradient tensors
            metadata: Compression metadata

        Returns:
            Decompressed gradients
        """
        if metadata.get("compression_method") == "none":
            return compressed_gradients

        # For top-k compression, gradients are already in the right format
        return compressed_gradients


class LocalTrainer:
    """Handles local model training for federated client."""

    def __init__(self, config: ClientConfig):
        self.config = config
        self.dp_manager = DifferentialPrivacyManager(config)
        self.compressor = GradientCompressor(config)

    def train_local_model(
        self,
        model: nn.Module,
        train_data: DataLoader,
        val_data: Optional[DataLoader] = None,
    ) -> TrainingResult:
        """
        Train model locally for specified epochs.

        Args:
            model: Model to train
            train_data: Training data loader
            val_data: Optional validation data loader

        Returns:
            TrainingResult with training metrics and model update
        """
        start_time = time.time()

        try:
            # Check privacy budget
            if (
                self.config.enable_differential_privacy
                and not self.dp_manager.can_participate()
            ):
                return TrainingResult(
                    success=False, message="Insufficient privacy budget for training"
                )

            # Setup training
            device = torch.device(
                "cuda" if self.config.use_gpu and torch.cuda.is_available() else "cpu"
            )
            model = model.to(device)

            # Setup optimizer
            optimizer = self._create_optimizer(model)
            criterion = nn.BCEWithLogitsLoss()

            # Training loop
            model.train()
            epoch_losses = []

            for epoch in range(self.config.local_epochs):
                epoch_loss = 0.0
                num_batches = 0

                for batch_idx, (data, target) in enumerate(train_data):
                    data, target = data.to(device), target.to(device)

                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output.squeeze(), target.float())
                    loss.backward()

                    # Apply differential privacy
                    if self.config.enable_differential_privacy:
                        grad_norm = self.dp_manager.clip_gradients(model)
                        noise_added = self.dp_manager.add_noise(model)

                        if batch_idx == 0:  # Log once per epoch
                            logger.debug(
                                f"Epoch {epoch}: grad_norm={grad_norm:.4f}, noise={noise_added:.4f}"
                            )

                    optimizer.step()

                    epoch_loss += loss.item()
                    num_batches += 1

                avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
                epoch_losses.append(avg_epoch_loss)

                logger.debug(f"Local epoch {epoch}: loss={avg_epoch_loss:.4f}")

            # Calculate training metrics
            training_time = time.time() - start_time
            final_loss = np.mean(epoch_losses) if epoch_losses else 0.0

            # Validation metrics
            val_metrics = {}
            if val_data is not None:
                val_metrics = self._evaluate_model(model, val_data, device)

            # Create model update
            model_weights = {
                name: param.cpu().clone().detach()
                for name, param in model.named_parameters()
            }

            # Compress gradients if needed
            compressed_gradients, compression_metadata = (
                self.compressor.compress_gradients(model)
            )

            # Estimate energy consumption (simplified)
            energy_consumed = self._estimate_energy_consumption(training_time, device)

            model_update = ModelUpdate(
                client_id=self.config.client_id,
                round_number=0,  # Will be set by server
                model_weights=model_weights,
                data_size=len(train_data.dataset),
                training_loss=final_loss,
                validation_metrics=val_metrics,
                training_time=training_time,
                energy_consumed=energy_consumed,
                differential_privacy_applied=self.config.enable_differential_privacy,
                epsilon_used=(
                    self.config.dp_epsilon
                    if self.config.enable_differential_privacy
                    else 0.0
                ),
            )

            return TrainingResult(
                success=True,
                model_update=model_update,
                training_loss=final_loss,
                validation_metrics=val_metrics,
                training_time=training_time,
                energy_consumed=energy_consumed,
                data_size=len(train_data.dataset),
                epochs_completed=self.config.local_epochs,
                message="Local training completed successfully",
            )

        except Exception as e:
            training_time = time.time() - start_time
            logger.error(f"Local training failed: {e}")

            return TrainingResult(
                success=False,
                training_time=training_time,
                message=f"Local training failed: {str(e)}",
            )

    def _create_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """Create optimizer based on configuration."""
        if self.config.optimizer.lower() == "adam":
            return optim.Adam(model.parameters(), lr=self.config.learning_rate)
        elif self.config.optimizer.lower() == "adamw":
            return optim.AdamW(model.parameters(), lr=self.config.learning_rate)
        else:  # Default to SGD
            return optim.SGD(
                model.parameters(), lr=self.config.learning_rate, momentum=0.9
            )

    def _evaluate_model(
        self, model: nn.Module, val_data: DataLoader, device: torch.device
    ) -> Dict[str, float]:
        """Evaluate model on validation data."""
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        criterion = nn.BCEWithLogitsLoss()

        with torch.no_grad():
            for data, target in val_data:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output.squeeze(), target.float())

                total_loss += loss.item()

                # Calculate accuracy
                predicted = (torch.sigmoid(output.squeeze()) > 0.5).float()
                total += target.size(0)
                correct += (predicted == target).sum().item()

        accuracy = correct / total if total > 0 else 0.0
        avg_loss = total_loss / len(val_data) if len(val_data) > 0 else 0.0

        return {"accuracy": accuracy, "loss": avg_loss}

    def _estimate_energy_consumption(
        self, training_time: float, device: torch.device
    ) -> float:
        """Estimate energy consumption during training."""
        # Simplified energy estimation
        base_power = 100  # Watts for CPU
        if device.type == "cuda":
            base_power = 250  # Watts for GPU

        # Convert to kWh
        energy_kwh = (base_power * training_time / 3600) / 1000
        return energy_kwh


class FederatedClient:
    """
    Main federated learning client class.

    Handles communication with federated server, local training,
    and privacy-preserving model updates.
    """

    def __init__(self, config: ClientConfig):
        self.config = config
        self.status = ClientStatus.DISCONNECTED
        self.local_trainer = LocalTrainer(config)
        self.communication_manager = FederatedCommunicationManager(
            FederatedConfig(), SecureCommunicator()
        )

        # Client state
        self.server_url = f"http://{config.server_host}:{config.server_port}"
        self.authentication_token: Optional[str] = None
        self.current_round = 0
        self.local_model: Optional[nn.Module] = None
        self.training_data: Optional[DataLoader] = None
        self.validation_data: Optional[DataLoader] = None

        # Performance tracking
        self.training_history: List[Dict[str, Any]] = []
        self.communication_stats = {
            "messages_sent": 0,
            "messages_received": 0,
            "bytes_sent": 0,
            "bytes_received": 0,
            "connection_failures": 0,
        }

        logger.info(f"Federated client {config.client_id} initialized")

    async def connect_to_server(self) -> bool:
        """
        Connect and register with the federated server.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.status = ClientStatus.CONNECTING

            # Create registration message
            registration_msg = create_registration_message(
                client_id=self.config.client_id,
                public_key=self.communication_manager.communicator.get_public_key_pem(),
                ip_address="127.0.0.1",  # Simplified for demo
                port=8000,
                capabilities={
                    "differential_privacy": self.config.enable_differential_privacy,
                    "compression": self.config.enable_compression,
                    "gpu_available": torch.cuda.is_available(),
                },
            )

            # Send registration request
            registration_url = f"{self.server_url}/api/v1/register"
            response = await self.communication_manager.send_message(
                registration_msg, registration_url
            )

            if response and response.payload.get("success"):
                self.authentication_token = response.payload.get("authentication_token")
                self.status = ClientStatus.CONNECTED

                # Add server's public key
                server_public_key = response.payload.get("server_public_key")
                if server_public_key:
                    self.communication_manager.communicator.add_peer_public_key(
                        "server", server_public_key
                    )

                logger.info(f"Client {self.config.client_id} connected to server")

                # Start heartbeat
                asyncio.create_task(self._heartbeat_loop())

                return True
            else:
                self.status = ClientStatus.ERROR
                logger.error(
                    f"Failed to register with server: {response.payload if response else 'No response'}"
                )
                return False

        except Exception as e:
            self.status = ClientStatus.ERROR
            logger.error(f"Connection failed: {e}")
            return False

    async def disconnect_from_server(self) -> bool:
        """
        Disconnect from the federated server.

        Returns:
            True if disconnection successful, False otherwise
        """
        try:
            if self.status == ClientStatus.CONNECTED:
                # Send unregistration request
                unregister_url = (
                    f"{self.server_url}/api/v1/unregister/{self.config.client_id}"
                )
                # Implementation would send unregister request here

            self.status = ClientStatus.DISCONNECTED
            self.authentication_token = None

            logger.info(f"Client {self.config.client_id} disconnected from server")
            return True

        except Exception as e:
            logger.error(f"Disconnection failed: {e}")
            return False

    def set_local_data(self, X: pd.DataFrame, y: pd.Series) -> bool:
        """
        Set local training data for the client.

        Args:
            X: Feature data
            y: Target data

        Returns:
            True if data set successfully, False otherwise
        """
        try:
            # Split into train/validation
            from sklearn.model_selection import train_test_split

            X_train, X_val, y_train, y_val = train_test_split(
                X,
                y,
                test_size=self.config.validation_split,
                random_state=42,
                stratify=y,
            )

            # Convert to tensors
            X_train_tensor = torch.FloatTensor(X_train.values)
            y_train_tensor = torch.FloatTensor(y_train.values)
            X_val_tensor = torch.FloatTensor(X_val.values)
            y_val_tensor = torch.FloatTensor(y_val.values)

            # Create data loaders
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

            self.training_data = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=self.config.shuffle_data,
            )
            self.validation_data = DataLoader(
                val_dataset, batch_size=self.config.batch_size, shuffle=False
            )

            logger.info(
                f"Local data set: {len(train_dataset)} training, {len(val_dataset)} validation samples"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to set local data: {e}")
            return False

    def set_local_model(self, model: nn.Module) -> bool:
        """
        Set the local model for training.

        Args:
            model: PyTorch model

        Returns:
            True if model set successfully, False otherwise
        """
        try:
            self.local_model = copy.deepcopy(model)
            logger.info(f"Local model set for client {self.config.client_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to set local model: {e}")
            return False

    async def get_global_model(self) -> Optional[Dict[str, torch.Tensor]]:
        """
        Retrieve the latest global model from server.

        Returns:
            Global model weights if successful, None otherwise
        """
        try:
            if not self.authentication_token:
                logger.error("Not authenticated with server")
                return None

            # Request global model
            global_model_url = f"{self.server_url}/api/v1/global-model?client_id={self.config.client_id}"

            # Create request message
            request_msg = FederatedMessage(
                message_type=MessageType.MODEL_REQUEST,
                sender_id=self.config.client_id,
                recipient_id="server",
                timestamp=datetime.now(),
                message_id=f"model_request_{secrets.token_hex(8)}",
                payload={"client_id": self.config.client_id},
            )

            response = await self.communication_manager.send_message(
                request_msg, global_model_url
            )

            if response and response.payload.get("success"):
                encoded_weights = response.payload.get("model_weights_encoded")
                if encoded_weights:
                    global_weights = MessageSerializer.deserialize_model_weights(
                        encoded_weights
                    )
                    logger.info(
                        f"Retrieved global model with {len(global_weights)} parameters"
                    )
                    return global_weights

            logger.warning("Failed to retrieve global model")
            return None

        except Exception as e:
            logger.error(f"Error retrieving global model: {e}")
            return None

    async def participate_in_round(self) -> bool:
        """
        Participate in a federated learning round.

        Returns:
            True if participation successful, False otherwise
        """
        try:
            if not self.local_model or not self.training_data:
                logger.error("Local model or training data not set")
                return False

            if not self.local_trainer.dp_manager.can_participate():
                logger.warning("Insufficient privacy budget to participate")
                return False

            self.status = ClientStatus.TRAINING

            # Get latest global model
            global_weights = await self.get_global_model()
            if global_weights:
                # Update local model with global weights
                with torch.no_grad():
                    for name, param in self.local_model.named_parameters():
                        if name in global_weights:
                            param.copy_(global_weights[name])

            # Perform local training
            training_result = self.local_trainer.train_local_model(
                self.local_model, self.training_data, self.validation_data
            )

            if not training_result.success:
                logger.error(f"Local training failed: {training_result.message}")
                self.status = ClientStatus.ERROR
                return False

            # Send model update to server
            self.status = ClientStatus.UPLOADING
            success = await self._send_model_update(training_result.model_update)

            if success:
                self.status = ClientStatus.WAITING
                self.current_round += 1

                # Record training history
                self.training_history.append(
                    {
                        "round": self.current_round,
                        "training_loss": training_result.training_loss,
                        "validation_metrics": training_result.validation_metrics,
                        "training_time": training_result.training_time,
                        "energy_consumed": training_result.energy_consumed,
                        "privacy_budget_used": (
                            self.config.dp_epsilon
                            if self.config.enable_differential_privacy
                            else 0.0
                        ),
                        "timestamp": datetime.now().isoformat(),
                    }
                )

                logger.info(f"Successfully participated in round {self.current_round}")
                return True
            else:
                self.status = ClientStatus.ERROR
                return False

        except Exception as e:
            logger.error(f"Error participating in round: {e}")
            self.status = ClientStatus.ERROR
            return False

    async def _send_model_update(self, model_update: ModelUpdate) -> bool:
        """Send model update to server."""
        try:
            if not self.authentication_token:
                logger.error("Not authenticated with server")
                return False

            # Create model update message
            update_msg = self.communication_manager.create_model_update_message(
                sender_id=self.config.client_id,
                recipient_id="server",
                model_update=model_update,
            )

            # Send to server
            update_url = f"{self.server_url}/api/v1/model-update"
            response = await self.communication_manager.send_message(
                update_msg, update_url
            )

            if response and response.payload.get("success"):
                logger.info("Model update sent successfully")
                self.communication_stats["messages_sent"] += 1
                return True
            else:
                logger.error(
                    f"Failed to send model update: {response.payload if response else 'No response'}"
                )
                self.communication_stats["connection_failures"] += 1
                return False

        except Exception as e:
            logger.error(f"Error sending model update: {e}")
            self.communication_stats["connection_failures"] += 1
            return False

    async def _heartbeat_loop(self):
        """Send periodic heartbeat to server."""
        while self.status in [
            ClientStatus.CONNECTED,
            ClientStatus.TRAINING,
            ClientStatus.WAITING,
        ]:
            try:
                await asyncio.sleep(self.config.heartbeat_interval)

                if self.authentication_token:
                    heartbeat_url = f"{self.server_url}/api/v1/heartbeat"
                    # Send heartbeat (simplified implementation)
                    logger.debug(
                        f"Sending heartbeat from client {self.config.client_id}"
                    )

            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                break

    def get_client_status(self) -> Dict[str, Any]:
        """Get comprehensive client status."""
        return {
            "client_id": self.config.client_id,
            "status": self.status.value,
            "current_round": self.current_round,
            "authenticated": self.authentication_token is not None,
            "privacy_budget_remaining": self.local_trainer.dp_manager.get_remaining_budget(),
            "training_history": self.training_history,
            "communication_stats": self.communication_stats,
            "local_data_size": (
                len(self.training_data.dataset) if self.training_data else 0
            ),
            "config": {
                "differential_privacy": self.config.enable_differential_privacy,
                "compression": self.config.enable_compression,
                "local_epochs": self.config.local_epochs,
                "batch_size": self.config.batch_size,
            },
        }

    async def run_client_loop(self, max_rounds: int = 10) -> bool:
        """
        Run the main client loop for federated learning.

        Args:
            max_rounds: Maximum number of rounds to participate in

        Returns:
            True if completed successfully, False otherwise
        """
        try:
            # Connect to server
            if not await self.connect_to_server():
                return False

            logger.info(f"Starting federated learning for {max_rounds} rounds")

            # Participate in federated rounds
            for round_num in range(max_rounds):
                if not self.local_trainer.dp_manager.can_participate():
                    logger.warning(f"Privacy budget exhausted after {round_num} rounds")
                    break

                logger.info(f"Participating in round {round_num + 1}/{max_rounds}")

                success = await self.participate_in_round()
                if not success:
                    logger.error(f"Failed to participate in round {round_num + 1}")
                    break

                # Wait between rounds (in practice, server would coordinate this)
                await asyncio.sleep(5)

            # Disconnect from server
            await self.disconnect_from_server()

            logger.info(
                f"Federated learning completed after {len(self.training_history)} rounds"
            )
            return True

        except Exception as e:
            logger.error(f"Client loop error: {e}")
            return False


# Utility functions
def create_federated_client(
    client_id: str, config_dict: Optional[Dict[str, Any]] = None
) -> FederatedClient:
    """
    Create a federated client with configuration.

    Args:
        client_id: Unique client identifier
        config_dict: Configuration dictionary

    Returns:
        Configured FederatedClient instance
    """
    config_dict = config_dict or {}
    config_dict["client_id"] = client_id

    config = ClientConfig(**config_dict)
    return FederatedClient(config)


async def simulate_federated_client(
    client_id: str,
    X: pd.DataFrame,
    y: pd.Series,
    server_host: str = "localhost",
    server_port: int = 8080,
    max_rounds: int = 5,
) -> Dict[str, Any]:
    """
    Simulate a federated client for testing.

    Args:
        client_id: Client identifier
        X: Training features
        y: Training targets
        server_host: Server host
        server_port: Server port
        max_rounds: Maximum rounds to participate

    Returns:
        Simulation results
    """
    # Create client configuration
    config = ClientConfig(
        client_id=client_id,
        server_host=server_host,
        server_port=server_port,
        local_epochs=3,
        enable_differential_privacy=True,
        dp_epsilon=1.0,
    )

    # Create client
    client = FederatedClient(config)

    # Set local data
    client.set_local_data(X, y)

    # Create simple model
    model = nn.Sequential(
        nn.Linear(X.shape[1], 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
    )
    client.set_local_model(model)

    # Run client simulation
    success = await client.run_client_loop(max_rounds)

    return {
        "success": success,
        "client_status": client.get_client_status(),
        "rounds_completed": len(client.training_history),
    }
