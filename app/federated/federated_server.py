"""
Federated Learning Server Implementation.

This module implements the federated server coordination system for privacy-preserving
collaborative training across multiple banking institutions. It includes client management,
secure model aggregation using FedAvg algorithm, and authentication mechanisms.
"""

import torch
import torch.nn as nn
import numpy as np
import asyncio
import json
import hashlib
import hmac
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import logging
from collections import defaultdict, OrderedDict
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy

# Cryptographic imports
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import secrets

try:
    from ..core.interfaces import BaseModel, TrainingMetrics
    from ..core.logging import get_logger, get_audit_logger
    from ..core.config import Config
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    
    from core.interfaces import BaseModel, TrainingMetrics
    from core.logging import get_logger, get_audit_logger
    from core.config import Config

logger = get_logger(__name__)
audit_logger = get_audit_logger()


class ClientStatus(Enum):
    """Client connection status."""
    DISCONNECTED = "disconnected"
    CONNECTED = "connected"
    TRAINING = "training"
    READY = "ready"
    ERROR = "error"


class AggregationMethod(Enum):
    """Federated aggregation methods."""
    FEDAVG = "fedavg"
    FEDPROX = "fedprox"
    FEDOPT = "fedopt"
    SCAFFOLD = "scaffold"


@dataclass
class ClientInfo:
    """Information about a federated client."""
    client_id: str
    public_key: str
    ip_address: str
    port: int
    status: ClientStatus = ClientStatus.DISCONNECTED
    last_seen: datetime = field(default_factory=datetime.now)
    data_size: int = 0
    model_version: int = 0
    capabilities: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Security and privacy
    authentication_token: Optional[str] = None
    privacy_budget: float = 1.0
    differential_privacy_epsilon: float = 1.0
    
    # Communication
    connection_quality: float = 1.0
    average_response_time: float = 0.0
    failed_connections: int = 0
    
    def is_active(self, timeout_minutes: int = 30) -> bool:
        """Check if client is considered active."""
        return (datetime.now() - self.last_seen).total_seconds() < timeout_minutes * 60
    
    def update_last_seen(self):
        """Update the last seen timestamp."""
        self.last_seen = datetime.now()


@dataclass
class ModelUpdate:
    """Represents a model update from a federated client."""
    client_id: str
    round_number: int
    model_weights: Dict[str, torch.Tensor]
    data_size: int
    training_loss: float
    validation_metrics: Dict[str, float]
    training_time: float
    energy_consumed: float = 0.0
    
    # Privacy and security
    is_encrypted: bool = False
    differential_privacy_applied: bool = False
    epsilon_used: float = 0.0
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    model_hash: Optional[str] = None
    signature: Optional[str] = None
    
    def calculate_hash(self) -> str:
        """Calculate hash of model weights for integrity verification."""
        weight_bytes = b""
        for key in sorted(self.model_weights.keys()):
            weight_bytes += self.model_weights[key].cpu().numpy().tobytes()
        return hashlib.sha256(weight_bytes).hexdigest()
    
    def verify_integrity(self) -> bool:
        """Verify the integrity of the model update."""
        if self.model_hash is None:
            return False
        return self.calculate_hash() == self.model_hash


@dataclass
class FederatedRound:
    """Information about a federated training round."""
    round_number: int
    start_time: datetime
    end_time: Optional[datetime] = None
    participating_clients: List[str] = field(default_factory=list)
    client_updates: List[ModelUpdate] = field(default_factory=list)
    aggregated_metrics: Dict[str, float] = field(default_factory=dict)
    global_model_hash: Optional[str] = None
    
    # Performance metrics
    total_data_size: int = 0
    average_training_time: float = 0.0
    total_energy_consumed: float = 0.0
    convergence_metrics: Dict[str, float] = field(default_factory=dict)
    
    def is_complete(self) -> bool:
        """Check if the round is complete."""
        return self.end_time is not None
    
    def get_duration(self) -> float:
        """Get round duration in seconds."""
        if self.end_time is None:
            return (datetime.now() - self.start_time).total_seconds()
        return (self.end_time - self.start_time).total_seconds()


@dataclass
class FederatedConfig:
    """Configuration for federated learning server."""
    # Basic settings
    server_host: str = "localhost"
    server_port: int = 8080
    max_clients: int = 100
    min_clients_per_round: int = 2
    max_clients_per_round: int = 10
    
    # Training settings
    aggregation_method: AggregationMethod = AggregationMethod.FEDAVG
    max_rounds: int = 100
    target_accuracy: float = 0.85
    convergence_threshold: float = 0.001
    
    # Client management
    client_timeout_minutes: int = 30
    max_failed_connections: int = 3
    client_selection_strategy: str = "random"  # "random", "performance", "data_size"
    
    # Security settings
    require_authentication: bool = True
    enable_encryption: bool = True
    require_differential_privacy: bool = False
    max_privacy_budget: float = 10.0
    
    # Performance settings
    async_aggregation: bool = True
    max_concurrent_clients: int = 50
    model_compression: bool = True
    
    # Monitoring
    enable_metrics_collection: bool = True
    log_client_updates: bool = True
    save_round_history: bool = True


class SecureAggregator:
    """Handles secure aggregation of model updates."""
    
    def __init__(self, config: FederatedConfig):
        self.config = config
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        self.public_key = self.private_key.public_key()
    
    def get_public_key_pem(self) -> str:
        """Get server public key in PEM format."""
        return self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        ).decode('utf-8')
    
    def decrypt_model_update(self, encrypted_update: bytes) -> Dict[str, torch.Tensor]:
        """Decrypt an encrypted model update."""
        try:
            decrypted_data = self.private_key.decrypt(
                encrypted_update,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            # Deserialize the decrypted model weights
            return torch.load(io.BytesIO(decrypted_data), weights_only=True)
        except Exception as e:
            logger.error(f"Failed to decrypt model update: {e}")
            raise
    
    def verify_client_signature(self, client_public_key: str, data: bytes, signature: bytes) -> bool:
        """Verify client signature for authentication."""
        try:
            public_key = serialization.load_pem_public_key(
                client_public_key.encode('utf-8'),
                backend=default_backend()
            )
            public_key.verify(
                signature,
                data,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except Exception as e:
            logger.warning(f"Signature verification failed: {e}")
            return False


class FederatedAggregator:
    """Implements various federated aggregation algorithms."""
    
    @staticmethod
    def federated_averaging(updates: List[ModelUpdate]) -> Dict[str, torch.Tensor]:
        """
        Implement FedAvg algorithm for model aggregation.
        
        Args:
            updates: List of model updates from clients
            
        Returns:
            Aggregated model weights
        """
        if not updates:
            raise ValueError("No updates provided for aggregation")
        
        # Calculate total data size for weighted averaging
        total_data_size = sum(update.data_size for update in updates)
        
        if total_data_size == 0:
            logger.warning("Total data size is 0, using uniform weights")
            weights = [1.0 / len(updates) for _ in updates]
        else:
            weights = [update.data_size / total_data_size for update in updates]
        
        # Initialize aggregated weights with zeros
        aggregated_weights = {}
        first_update = updates[0]
        
        for key in first_update.model_weights.keys():
            aggregated_weights[key] = torch.zeros_like(first_update.model_weights[key])
        
        # Weighted aggregation
        for i, update in enumerate(updates):
            weight = weights[i]
            for key in aggregated_weights.keys():
                if key in update.model_weights:
                    aggregated_weights[key] += weight * update.model_weights[key]
                else:
                    logger.warning(f"Missing weight key {key} in update from client {update.client_id}")
        
        logger.info(f"Aggregated {len(updates)} model updates using FedAvg")
        return aggregated_weights
    
    @staticmethod
    def federated_proximal(updates: List[ModelUpdate], global_model: Dict[str, torch.Tensor], 
                          mu: float = 0.01) -> Dict[str, torch.Tensor]:
        """
        Implement FedProx algorithm for handling heterogeneous data.
        
        Args:
            updates: List of model updates from clients
            global_model: Current global model weights
            mu: Proximal term coefficient
            
        Returns:
            Aggregated model weights
        """
        # Start with FedAvg aggregation
        aggregated_weights = FederatedAggregator.federated_averaging(updates)
        
        # Apply proximal term
        for key in aggregated_weights.keys():
            if key in global_model:
                aggregated_weights[key] = (
                    aggregated_weights[key] + mu * global_model[key]
                ) / (1 + mu)
        
        logger.info(f"Aggregated {len(updates)} model updates using FedProx (mu={mu})")
        return aggregated_weights
    
    @staticmethod
    def scaffold_aggregation(updates: List[ModelUpdate], control_variates: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Implement SCAFFOLD algorithm for variance reduction.
        
        Args:
            updates: List of model updates from clients
            control_variates: Current control variates
            
        Returns:
            Tuple of (aggregated_weights, updated_control_variates)
        """
        # This is a simplified implementation of SCAFFOLD
        # In practice, you would need client-specific control variates
        aggregated_weights = FederatedAggregator.federated_averaging(updates)
        
        # Update control variates (simplified)
        updated_control_variates = {}
        for key in control_variates.keys():
            if key in aggregated_weights:
                updated_control_variates[key] = 0.1 * (aggregated_weights[key] - control_variates[key])
        
        logger.info(f"Aggregated {len(updates)} model updates using SCAFFOLD")
        return aggregated_weights, updated_control_variates


class ClientSelector:
    """Handles client selection strategies for federated rounds."""
    
    def __init__(self, config: FederatedConfig):
        self.config = config
    
    def select_clients(self, available_clients: List[ClientInfo], 
                      round_number: int) -> List[ClientInfo]:
        """
        Select clients for the current federated round.
        
        Args:
            available_clients: List of available clients
            round_number: Current round number
            
        Returns:
            Selected clients for training
        """
        # Filter active clients
        active_clients = [
            client for client in available_clients 
            if client.is_active() and client.status in [ClientStatus.CONNECTED, ClientStatus.READY]
        ]
        
        if len(active_clients) < self.config.min_clients_per_round:
            logger.warning(f"Not enough active clients: {len(active_clients)} < {self.config.min_clients_per_round}")
            return []
        
        # Apply selection strategy
        if self.config.client_selection_strategy == "random":
            return self._random_selection(active_clients)
        elif self.config.client_selection_strategy == "performance":
            return self._performance_based_selection(active_clients)
        elif self.config.client_selection_strategy == "data_size":
            return self._data_size_based_selection(active_clients)
        else:
            logger.warning(f"Unknown selection strategy: {self.config.client_selection_strategy}")
            return self._random_selection(active_clients)
    
    def _random_selection(self, clients: List[ClientInfo]) -> List[ClientInfo]:
        """Random client selection."""
        import random
        num_clients = min(len(clients), self.config.max_clients_per_round)
        return random.sample(clients, num_clients)
    
    def _performance_based_selection(self, clients: List[ClientInfo]) -> List[ClientInfo]:
        """Select clients based on performance metrics."""
        # Sort by average response time and connection quality
        scored_clients = []
        for client in clients:
            score = (
                client.connection_quality * 0.6 + 
                (1.0 / (client.average_response_time + 1.0)) * 0.4
            )
            scored_clients.append((client, score))
        
        # Sort by score (descending)
        scored_clients.sort(key=lambda x: x[1], reverse=True)
        
        num_clients = min(len(clients), self.config.max_clients_per_round)
        return [client for client, _ in scored_clients[:num_clients]]
    
    def _data_size_based_selection(self, clients: List[ClientInfo]) -> List[ClientInfo]:
        """Select clients based on data size."""
        # Sort by data size (descending)
        clients_sorted = sorted(clients, key=lambda x: x.data_size, reverse=True)
        
        num_clients = min(len(clients), self.config.max_clients_per_round)
        return clients_sorted[:num_clients]


class FederatedServer:
    """
    Main federated learning server class.
    
    Coordinates federated training across multiple clients, handles secure aggregation,
    and manages the global model state.
    """
    
    def __init__(self, config: Optional[FederatedConfig] = None):
        self.config = config or FederatedConfig()
        self.clients: Dict[str, ClientInfo] = {}
        self.global_model: Optional[nn.Module] = None
        self.global_model_weights: Dict[str, torch.Tensor] = {}
        self.current_round = 0
        self.round_history: List[FederatedRound] = []
        self.is_running = False
        
        # Components
        self.aggregator = FederatedAggregator()
        self.secure_aggregator = SecureAggregator(self.config)
        self.client_selector = ClientSelector(self.config)
        
        # Threading and async
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_concurrent_clients)
        self.round_lock = threading.Lock()
        
        # Metrics and monitoring
        self.metrics = {
            'total_rounds': 0,
            'total_clients_served': 0,
            'average_round_time': 0.0,
            'total_energy_consumed': 0.0,
            'convergence_achieved': False
        }
        
        logger.info(f"Federated server initialized with config: {self.config}")
    
    def register_client(self, client_id: str, public_key: str, ip_address: str, 
                       port: int, capabilities: Optional[Dict[str, Any]] = None) -> bool:
        """
        Register a new federated client.
        
        Args:
            client_id: Unique client identifier
            public_key: Client's public key for authentication
            ip_address: Client's IP address
            port: Client's port number
            capabilities: Optional client capabilities
            
        Returns:
            True if registration successful, False otherwise
        """
        try:
            if client_id in self.clients:
                logger.warning(f"Client {client_id} already registered, updating info")
            
            # Generate authentication token
            auth_token = secrets.token_urlsafe(32) if self.config.require_authentication else None
            
            client_info = ClientInfo(
                client_id=client_id,
                public_key=public_key,
                ip_address=ip_address,
                port=port,
                status=ClientStatus.CONNECTED,
                capabilities=capabilities or {},
                authentication_token=auth_token
            )
            
            self.clients[client_id] = client_info
            
            # Log registration
            audit_logger.log_model_operation(
                user_id=client_id,
                model_id="federated_server",
                operation="client_registration",
                success=True,
                details={
                    "client_id": client_id,
                    "ip_address": ip_address,
                    "port": port,
                    "capabilities": capabilities
                }
            )
            
            logger.info(f"Client {client_id} registered successfully from {ip_address}:{port}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register client {client_id}: {e}")
            return False
    
    def unregister_client(self, client_id: str) -> bool:
        """
        Unregister a federated client.
        
        Args:
            client_id: Client identifier to unregister
            
        Returns:
            True if unregistration successful, False otherwise
        """
        try:
            if client_id not in self.clients:
                logger.warning(f"Client {client_id} not found for unregistration")
                return False
            
            del self.clients[client_id]
            
            audit_logger.log_model_operation(
                user_id=client_id,
                model_id="federated_server",
                operation="client_unregistration",
                success=True,
                details={"client_id": client_id}
            )
            
            logger.info(f"Client {client_id} unregistered successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unregister client {client_id}: {e}")
            return False
    
    def authenticate_client(self, client_id: str, token: str) -> bool:
        """
        Authenticate a client using their token.
        
        Args:
            client_id: Client identifier
            token: Authentication token
            
        Returns:
            True if authentication successful, False otherwise
        """
        if not self.config.require_authentication:
            return True
        
        if client_id not in self.clients:
            logger.warning(f"Authentication failed: Client {client_id} not registered")
            return False
        
        client = self.clients[client_id]
        if client.authentication_token != token:
            logger.warning(f"Authentication failed: Invalid token for client {client_id}")
            client.failed_connections += 1
            return False
        
        client.update_last_seen()
        return True
    
    def set_global_model(self, model: nn.Module) -> bool:
        """
        Set the initial global model.
        
        Args:
            model: PyTorch model to use as global model
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.global_model = copy.deepcopy(model)
            self.global_model_weights = {
                name: param.clone().detach() 
                for name, param in model.named_parameters()
            }
            
            logger.info(f"Global model set with {len(self.global_model_weights)} parameters")
            return True
            
        except Exception as e:
            logger.error(f"Failed to set global model: {e}")
            return False
    
    def get_global_model_weights(self) -> Dict[str, torch.Tensor]:
        """
        Get current global model weights.
        
        Returns:
            Dictionary of model weights
        """
        return copy.deepcopy(self.global_model_weights)
    
    def aggregate_updates(self, client_updates: List[ModelUpdate]) -> Dict[str, torch.Tensor]:
        """
        Aggregate model updates from clients.
        
        Args:
            client_updates: List of model updates from clients
            
        Returns:
            Aggregated model weights
        """
        if not client_updates:
            logger.warning("No client updates to aggregate")
            return self.global_model_weights
        
        # Verify update integrity
        valid_updates = []
        for update in client_updates:
            if update.verify_integrity():
                valid_updates.append(update)
            else:
                logger.warning(f"Invalid update from client {update.client_id}, skipping")
        
        if not valid_updates:
            logger.error("No valid updates to aggregate")
            return self.global_model_weights
        
        # Apply aggregation method
        if self.config.aggregation_method == AggregationMethod.FEDAVG:
            aggregated_weights = self.aggregator.federated_averaging(valid_updates)
        elif self.config.aggregation_method == AggregationMethod.FEDPROX:
            aggregated_weights = self.aggregator.federated_proximal(
                valid_updates, self.global_model_weights
            )
        else:
            logger.warning(f"Unsupported aggregation method: {self.config.aggregation_method}")
            aggregated_weights = self.aggregator.federated_averaging(valid_updates)
        
        # Update global model weights
        self.global_model_weights = aggregated_weights
        
        # Update global model if available
        if self.global_model is not None:
            with torch.no_grad():
                for name, param in self.global_model.named_parameters():
                    if name in aggregated_weights:
                        param.copy_(aggregated_weights[name])
        
        logger.info(f"Aggregated {len(valid_updates)} model updates")
        return aggregated_weights
    
    def start_federated_round(self) -> Optional[FederatedRound]:
        """
        Start a new federated training round.
        
        Returns:
            FederatedRound object if successful, None otherwise
        """
        with self.round_lock:
            if self.current_round >= self.config.max_rounds:
                logger.info("Maximum rounds reached, stopping federated training")
                return None
            
            # Select clients for this round
            available_clients = list(self.clients.values())
            selected_clients = self.client_selector.select_clients(available_clients, self.current_round)
            
            if len(selected_clients) < self.config.min_clients_per_round:
                logger.warning(f"Not enough clients for round {self.current_round}")
                return None
            
            # Create new round
            federated_round = FederatedRound(
                round_number=self.current_round,
                start_time=datetime.now(),
                participating_clients=[client.client_id for client in selected_clients]
            )
            
            # Update client status
            for client in selected_clients:
                client.status = ClientStatus.TRAINING
                client.model_version = self.current_round
            
            self.round_history.append(federated_round)
            
            logger.info(f"Started federated round {self.current_round} with {len(selected_clients)} clients")
            return federated_round
    
    def complete_federated_round(self, client_updates: List[ModelUpdate]) -> bool:
        """
        Complete the current federated round with client updates.
        
        Args:
            client_updates: List of model updates from clients
            
        Returns:
            True if round completed successfully, False otherwise
        """
        try:
            with self.round_lock:
                if not self.round_history:
                    logger.error("No active round to complete")
                    return False
                
                current_round = self.round_history[-1]
                if current_round.is_complete():
                    logger.warning(f"Round {current_round.round_number} already completed")
                    return False
                
                # Aggregate model updates
                aggregated_weights = self.aggregate_updates(client_updates)
                
                # Update round information
                current_round.end_time = datetime.now()
                current_round.client_updates = client_updates
                current_round.total_data_size = sum(update.data_size for update in client_updates)
                current_round.average_training_time = np.mean([update.training_time for update in client_updates])
                current_round.total_energy_consumed = sum(update.energy_consumed for update in client_updates)
                
                # Calculate aggregated metrics
                if client_updates:
                    current_round.aggregated_metrics = {
                        'average_loss': np.mean([update.training_loss for update in client_updates]),
                        'total_data_size': current_round.total_data_size,
                        'participating_clients': len(client_updates),
                        'round_duration': current_round.get_duration()
                    }
                
                # Update client status
                for update in client_updates:
                    if update.client_id in self.clients:
                        client = self.clients[update.client_id]
                        client.status = ClientStatus.READY
                        client.performance_metrics.update(update.validation_metrics)
                        client.update_last_seen()
                
                # Update server metrics
                self.metrics['total_rounds'] += 1
                self.metrics['average_round_time'] = (
                    self.metrics['average_round_time'] * (self.current_round) + 
                    current_round.get_duration()
                ) / (self.current_round + 1)
                self.metrics['total_energy_consumed'] += current_round.total_energy_consumed
                
                self.current_round += 1
                
                logger.info(f"Completed federated round {current_round.round_number} with {len(client_updates)} updates")
                
                # Log round completion
                audit_logger.log_model_operation(
                    user_id="federated_server",
                    model_id="global_model",
                    operation="round_completion",
                    success=True,
                    details={
                        "round_number": current_round.round_number,
                        "participating_clients": len(client_updates),
                        "total_data_size": current_round.total_data_size,
                        "round_duration": current_round.get_duration(),
                        "energy_consumed": current_round.total_energy_consumed
                    }
                )
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to complete federated round: {e}")
            return False
    
    def check_convergence(self) -> bool:
        """
        Check if the federated training has converged.
        
        Returns:
            True if converged, False otherwise
        """
        if len(self.round_history) < 2:
            return False
        
        # Get last two rounds
        current_round = self.round_history[-1]
        previous_round = self.round_history[-2]
        
        if not current_round.is_complete() or not previous_round.is_complete():
            return False
        
        # Check loss improvement
        current_loss = current_round.aggregated_metrics.get('average_loss', float('inf'))
        previous_loss = previous_round.aggregated_metrics.get('average_loss', float('inf'))
        
        if previous_loss == 0:
            return False
        
        improvement = abs(previous_loss - current_loss) / previous_loss
        
        converged = improvement < self.config.convergence_threshold
        if converged:
            self.metrics['convergence_achieved'] = True
            logger.info(f"Convergence achieved at round {self.current_round} with improvement {improvement:.6f}")
        
        return converged
    
    def get_server_status(self) -> Dict[str, Any]:
        """
        Get comprehensive server status information.
        
        Returns:
            Dictionary containing server status
        """
        active_clients = [
            client for client in self.clients.values() 
            if client.is_active()
        ]
        
        return {
            'server_info': {
                'host': self.config.server_host,
                'port': self.config.server_port,
                'is_running': self.is_running,
                'current_round': self.current_round,
                'max_rounds': self.config.max_rounds
            },
            'clients': {
                'total_registered': len(self.clients),
                'active_clients': len(active_clients),
                'client_status': {
                    client.client_id: {
                        'status': client.status.value,
                        'last_seen': client.last_seen.isoformat(),
                        'data_size': client.data_size,
                        'model_version': client.model_version
                    }
                    for client in self.clients.values()
                }
            },
            'training': {
                'rounds_completed': len([r for r in self.round_history if r.is_complete()]),
                'convergence_achieved': self.metrics['convergence_achieved'],
                'total_energy_consumed': self.metrics['total_energy_consumed'],
                'average_round_time': self.metrics['average_round_time']
            },
            'security': {
                'authentication_required': self.config.require_authentication,
                'encryption_enabled': self.config.enable_encryption,
                'differential_privacy_required': self.config.require_differential_privacy
            }
        }
    
    def save_checkpoint(self, filepath: str) -> bool:
        """
        Save server state to checkpoint file.
        
        Args:
            filepath: Path to save checkpoint
            
        Returns:
            True if successful, False otherwise
        """
        try:
            checkpoint = {
                'config': self.config,
                'current_round': self.current_round,
                'global_model_weights': self.global_model_weights,
                'clients': {
                    client_id: {
                        'client_id': client.client_id,
                        'public_key': client.public_key,
                        'ip_address': client.ip_address,
                        'port': client.port,
                        'data_size': client.data_size,
                        'performance_metrics': client.performance_metrics
                    }
                    for client_id, client in self.clients.items()
                },
                'metrics': self.metrics,
                'timestamp': datetime.now().isoformat()
            }
            
            torch.save(checkpoint, filepath)
            logger.info(f"Server checkpoint saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            return False
    
    def load_checkpoint(self, filepath: str) -> bool:
        """
        Load server state from checkpoint file.
        
        Args:
            filepath: Path to checkpoint file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            checkpoint = torch.load(filepath, weights_only=False)
            
            self.current_round = checkpoint['current_round']
            self.global_model_weights = checkpoint['global_model_weights']
            self.metrics = checkpoint['metrics']
            
            # Restore client information (without sensitive data)
            for client_id, client_data in checkpoint['clients'].items():
                if client_id not in self.clients:
                    # Create minimal client info for restoration
                    client_info = ClientInfo(
                        client_id=client_data['client_id'],
                        public_key=client_data['public_key'],
                        ip_address=client_data['ip_address'],
                        port=client_data['port'],
                        status=ClientStatus.DISCONNECTED,
                        data_size=client_data['data_size'],
                        performance_metrics=client_data['performance_metrics']
                    )
                    self.clients[client_id] = client_info
            
            logger.info(f"Server checkpoint loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return False
    
    def shutdown(self):
        """Gracefully shutdown the federated server."""
        logger.info("Shutting down federated server...")
        
        self.is_running = False
        
        # Update all clients to disconnected status
        for client in self.clients.values():
            client.status = ClientStatus.DISCONNECTED
        
        # Shutdown thread pool
        self.executor.shutdown(wait=True)
        
        logger.info("Federated server shutdown complete")


# Utility functions for federated learning
def create_federated_server(config_dict: Optional[Dict[str, Any]] = None) -> FederatedServer:
    """
    Create a federated server with configuration.
    
    Args:
        config_dict: Configuration dictionary
        
    Returns:
        Configured FederatedServer instance
    """
    if config_dict:
        config = FederatedConfig(**config_dict)
    else:
        config = FederatedConfig()
    
    return FederatedServer(config)


def simulate_federated_round(server: FederatedServer, 
                           client_updates: List[ModelUpdate]) -> bool:
    """
    Simulate a complete federated round for testing.
    
    Args:
        server: FederatedServer instance
        client_updates: List of simulated client updates
        
    Returns:
        True if round completed successfully
    """
    # Start round
    round_info = server.start_federated_round()
    if round_info is None:
        return False
    
    # Complete round with updates
    return server.complete_federated_round(client_updates)