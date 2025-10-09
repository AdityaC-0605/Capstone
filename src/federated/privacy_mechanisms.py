"""
Privacy Preservation Mechanisms for Federated Learning.

This module implements comprehensive privacy preservation mechanisms including
differential privacy, secure aggregation protocols, gradient encryption,
privacy budget tracking, and convergence monitoring for federated learning.
"""

import torch
import torch.nn as nn
import numpy as np
import hashlib
import hmac
import secrets
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
import copy

# Cryptographic imports
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

try:
    from .federated_server import ModelUpdate, FederatedConfig
    from ..core.logging import get_logger, get_audit_logger
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    
    from federated.federated_server import ModelUpdate, FederatedConfig
    from core.logging import get_logger, get_audit_logger

logger = get_logger(__name__)
audit_logger = get_audit_logger()


class PrivacyMechanism(Enum):
    """Types of privacy mechanisms."""
    DIFFERENTIAL_PRIVACY = "differential_privacy"
    SECURE_AGGREGATION = "secure_aggregation"
    HOMOMORPHIC_ENCRYPTION = "homomorphic_encryption"
    GRADIENT_COMPRESSION = "gradient_compression"
    NOISE_INJECTION = "noise_injection"


@dataclass
class PrivacyConfig:
    """Configuration for privacy preservation mechanisms."""
    # Differential Privacy
    enable_differential_privacy: bool = True
    dp_epsilon: float = 1.0
    dp_delta: float = 1e-5
    dp_clip_norm: float = 1.0
    dp_noise_multiplier: Optional[float] = None
    
    # Privacy Budget Management
    privacy_budget: float = 10.0
    budget_tracking: bool = True
    budget_allocation_strategy: str = "uniform"  # "uniform", "adaptive", "exponential"
    
    # Secure Aggregation
    enable_secure_aggregation: bool = True
    min_clients_for_aggregation: int = 3
    dropout_resilience: bool = True
    
    # Gradient Encryption
    enable_gradient_encryption: bool = True
    encryption_key_size: int = 2048
    
    # Convergence Monitoring
    enable_convergence_monitoring: bool = True
    convergence_threshold: float = 0.001
    convergence_patience: int = 5
    
    # Asynchronous FL
    enable_async_fl: bool = False
    async_staleness_bound: int = 3
    async_weight_decay: float = 0.9

@dataclass
class PrivacyBudget:
    """Privacy budget tracking for differential privacy."""
    total_budget: float
    used_budget: float = 0.0
    remaining_budget: float = field(init=False)
    allocations: List[Dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
        self.remaining_budget = self.total_budget - self.used_budget
    
    def allocate_budget(self, amount: float, purpose: str) -> bool:
        """Allocate privacy budget for a specific purpose."""
        if amount <= self.remaining_budget:
            self.used_budget += amount
            self.remaining_budget -= amount
            self.allocations.append({
                "amount": amount,
                "purpose": purpose,
                "timestamp": datetime.now().isoformat()
            })
            return True
        return False
    
    def can_allocate(self, amount: float) -> bool:
        """Check if budget can be allocated."""
        return amount <= self.remaining_budget
    
    def get_utilization(self) -> float:
        """Get budget utilization percentage."""
        return (self.used_budget / self.total_budget) * 100 if self.total_budget > 0 else 0.0


class AdvancedDifferentialPrivacy:
    """Advanced differential privacy mechanisms with multiple noise types."""
    
    def __init__(self, config: PrivacyConfig):
        self.config = config
        self.privacy_budget = PrivacyBudget(config.privacy_budget)
        self.noise_multiplier = config.dp_noise_multiplier or self._calculate_noise_multiplier()
        
    def _calculate_noise_multiplier(self) -> float:
        """Calculate noise multiplier using advanced composition."""
        if self.config.dp_epsilon <= 0:
            return float('inf')
        
        # Advanced composition for multiple queries
        # This is a simplified version - in practice, use RDP or moments accountant
        return self.config.dp_clip_norm * np.sqrt(2 * np.log(1.25 / self.config.dp_delta)) / self.config.dp_epsilon
    
    def add_gaussian_noise(self, gradients: Dict[str, torch.Tensor], 
                          sensitivity: float = None) -> Tuple[Dict[str, torch.Tensor], float]:
        """Add Gaussian noise for differential privacy."""
        if not self.config.enable_differential_privacy:
            return gradients, 0.0
        
        if not self.privacy_budget.can_allocate(self.config.dp_epsilon):
            logger.warning("Insufficient privacy budget for noise addition")
            return gradients, 0.0
        
        sensitivity = sensitivity or self.config.dp_clip_norm
        noise_scale = self.noise_multiplier * sensitivity
        
        noisy_gradients = {}
        total_noise = 0.0
        
        for name, grad in gradients.items():
            noise = torch.normal(0, noise_scale, grad.shape, device=grad.device)
            noisy_gradients[name] = grad + noise
            total_noise += noise.norm().item()
        
        # Update privacy budget
        self.privacy_budget.allocate_budget(self.config.dp_epsilon, "gaussian_noise")
        
        logger.debug(f"Added Gaussian noise: scale={noise_scale:.4f}, total={total_noise:.4f}")
        return noisy_gradients, total_noise
    
    def add_laplace_noise(self, gradients: Dict[str, torch.Tensor], 
                         sensitivity: float = None) -> Tuple[Dict[str, torch.Tensor], float]:
        """Add Laplace noise for differential privacy."""
        if not self.config.enable_differential_privacy:
            return gradients, 0.0
        
        if not self.privacy_budget.can_allocate(self.config.dp_epsilon):
            logger.warning("Insufficient privacy budget for Laplace noise")
            return gradients, 0.0
        
        sensitivity = sensitivity or self.config.dp_clip_norm
        noise_scale = sensitivity / self.config.dp_epsilon
        
        noisy_gradients = {}
        total_noise = 0.0
        
        for name, grad in gradients.items():
            # Generate Laplace noise
            noise = torch.from_numpy(
                np.random.laplace(0, noise_scale, grad.shape)
            ).float().to(grad.device)
            noisy_gradients[name] = grad + noise
            total_noise += noise.norm().item()
        
        # Update privacy budget
        self.privacy_budget.allocate_budget(self.config.dp_epsilon, "laplace_noise")
        
        logger.debug(f"Added Laplace noise: scale={noise_scale:.4f}, total={total_noise:.4f}")
        return noisy_gradients, total_noise
    
    def clip_gradients_adaptive(self, gradients: Dict[str, torch.Tensor], 
                               percentile: float = 50.0) -> Tuple[Dict[str, torch.Tensor], float]:
        """Adaptive gradient clipping based on gradient distribution."""
        all_norms = []
        for grad in gradients.values():
            all_norms.extend(grad.flatten().abs().tolist())
        
        if not all_norms:
            return gradients, 0.0
        
        # Calculate adaptive clipping threshold
        clip_threshold = np.percentile(all_norms, percentile)
        clip_threshold = max(clip_threshold, self.config.dp_clip_norm)
        
        clipped_gradients = {}
        total_norm = 0.0
        
        for name, grad in gradients.items():
            grad_norm = grad.norm()
            total_norm += grad_norm.item()
            
            if grad_norm > clip_threshold:
                clipped_gradients[name] = grad * (clip_threshold / grad_norm)
            else:
                clipped_gradients[name] = grad
        
        logger.debug(f"Adaptive gradient clipping: threshold={clip_threshold:.4f}")
        return clipped_gradients, total_norm
    
    def get_privacy_spent(self) -> Dict[str, Any]:
        """Get comprehensive privacy spending information."""
        return {
            "total_budget": self.privacy_budget.total_budget,
            "used_budget": self.privacy_budget.used_budget,
            "remaining_budget": self.privacy_budget.remaining_budget,
            "utilization_percent": self.privacy_budget.get_utilization(),
            "allocations": self.privacy_budget.allocations,
            "epsilon_per_query": self.config.dp_epsilon,
            "delta": self.config.dp_delta
        }


class SecureAggregationProtocol:
    """Secure aggregation protocol for federated learning."""
    
    def __init__(self, config: PrivacyConfig):
        self.config = config
        self.client_keys: Dict[str, rsa.RSAPublicKey] = {}
        self.server_private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=config.encryption_key_size,
            backend=default_backend()
        )
        self.server_public_key = self.server_private_key.public_key()
        
    def register_client(self, client_id: str, public_key: rsa.RSAPublicKey):
        """Register a client's public key for secure aggregation."""
        self.client_keys[client_id] = public_key
        logger.debug(f"Registered client {client_id} for secure aggregation")
    
    def generate_shared_secrets(self, participating_clients: List[str]) -> Dict[str, bytes]:
        """Generate shared secrets for secure aggregation."""
        if len(participating_clients) < self.config.min_clients_for_aggregation:
            raise ValueError(f"Need at least {self.config.min_clients_for_aggregation} clients")
        
        shared_secrets = {}
        
        for client_id in participating_clients:
            # Generate a random secret for each client
            secret = secrets.token_bytes(32)  # 256-bit secret
            shared_secrets[client_id] = secret
        
        logger.info(f"Generated shared secrets for {len(participating_clients)} clients")
        return shared_secrets
    
    def encrypt_model_update(self, model_update: ModelUpdate, 
                           shared_secret: bytes) -> Dict[str, Any]:
        """Encrypt model update using shared secret."""
        try:
            # Serialize model weights
            import pickle
            serialized_weights = pickle.dumps(model_update.model_weights)
            
            # Generate key from shared secret
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=b'federated_learning_salt',
                iterations=100000,
                backend=default_backend()
            )
            key = kdf.derive(shared_secret)
            
            # Encrypt with AES
            iv = secrets.token_bytes(16)
            cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
            encryptor = cipher.encryptor()
            
            # Pad data to block size
            padding_length = 16 - (len(serialized_weights) % 16)
            padded_data = serialized_weights + bytes([padding_length] * padding_length)
            
            encrypted_data = encryptor.update(padded_data) + encryptor.finalize()
            
            return {
                "encrypted_weights": base64.b64encode(encrypted_data).decode('utf-8'),
                "iv": base64.b64encode(iv).decode('utf-8'),
                "client_id": model_update.client_id,
                "data_size": model_update.data_size,
                "training_loss": model_update.training_loss,
                "validation_metrics": model_update.validation_metrics
            }
            
        except Exception as e:
            logger.error(f"Failed to encrypt model update: {e}")
            raise
    
    def decrypt_model_update(self, encrypted_update: Dict[str, Any], 
                           shared_secret: bytes) -> Dict[str, torch.Tensor]:
        """Decrypt model update using shared secret."""
        try:
            # Derive key from shared secret
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=b'federated_learning_salt',
                iterations=100000,
                backend=default_backend()
            )
            key = kdf.derive(shared_secret)
            
            # Decrypt with AES
            encrypted_data = base64.b64decode(encrypted_update["encrypted_weights"])
            iv = base64.b64decode(encrypted_update["iv"])
            
            cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
            decryptor = cipher.decryptor()
            
            padded_data = decryptor.update(encrypted_data) + decryptor.finalize()
            
            # Remove padding
            padding_length = padded_data[-1]
            serialized_weights = padded_data[:-padding_length]
            
            # Deserialize model weights
            import pickle
            model_weights = pickle.loads(serialized_weights)
            
            return model_weights
            
        except Exception as e:
            logger.error(f"Failed to decrypt model update: {e}")
            raise
    
    def secure_aggregate(self, encrypted_updates: List[Dict[str, Any]], 
                        shared_secrets: Dict[str, bytes]) -> Dict[str, torch.Tensor]:
        """Perform secure aggregation of encrypted model updates."""
        if len(encrypted_updates) < self.config.min_clients_for_aggregation:
            raise ValueError("Insufficient clients for secure aggregation")
        
        # Decrypt all updates
        decrypted_updates = []
        total_data_size = 0
        
        for update in encrypted_updates:
            client_id = update["client_id"]
            if client_id in shared_secrets:
                try:
                    weights = self.decrypt_model_update(update, shared_secrets[client_id])
                    decrypted_updates.append({
                        "weights": weights,
                        "data_size": update["data_size"]
                    })
                    total_data_size += update["data_size"]
                except Exception as e:
                    logger.warning(f"Failed to decrypt update from {client_id}: {e}")
                    if not self.config.dropout_resilience:
                        raise
        
        if not decrypted_updates:
            raise ValueError("No valid updates to aggregate")
        
        # Perform weighted aggregation
        aggregated_weights = {}
        first_update = decrypted_updates[0]["weights"]
        
        for key in first_update.keys():
            aggregated_weights[key] = torch.zeros_like(first_update[key])
        
        for update in decrypted_updates:
            weight = update["data_size"] / total_data_size
            for key in aggregated_weights.keys():
                if key in update["weights"]:
                    aggregated_weights[key] += weight * update["weights"][key]
        
        logger.info(f"Securely aggregated {len(decrypted_updates)} model updates")
        return aggregated_weights


class ConvergenceMonitor:
    """Monitor federated learning convergence with privacy preservation."""
    
    def __init__(self, config: PrivacyConfig):
        self.config = config
        self.loss_history: List[float] = []
        self.convergence_history: List[bool] = []
        self.patience_counter = 0
        self.best_loss = float('inf')
        
    def update_loss(self, loss: float) -> bool:
        """Update loss and check for convergence."""
        self.loss_history.append(loss)
        
        if not self.config.enable_convergence_monitoring:
            return False
        
        # Check for improvement
        if loss < self.best_loss - self.config.convergence_threshold:
            self.best_loss = loss
            self.patience_counter = 0
            converged = False
        else:
            self.patience_counter += 1
            converged = self.patience_counter >= self.config.convergence_patience
        
        self.convergence_history.append(converged)
        
        if converged:
            logger.info(f"Convergence detected after {len(self.loss_history)} rounds")
        
        return converged
    
    def get_convergence_metrics(self) -> Dict[str, Any]:
        """Get convergence monitoring metrics."""
        if not self.loss_history:
            return {"status": "no_data"}
        
        recent_losses = self.loss_history[-5:] if len(self.loss_history) >= 5 else self.loss_history
        
        return {
            "current_loss": self.loss_history[-1],
            "best_loss": self.best_loss,
            "rounds_completed": len(self.loss_history),
            "patience_counter": self.patience_counter,
            "converged": self.convergence_history[-1] if self.convergence_history else False,
            "recent_average_loss": np.mean(recent_losses),
            "loss_trend": "decreasing" if len(self.loss_history) > 1 and 
                         self.loss_history[-1] < self.loss_history[-2] else "increasing"
        }
    
    def should_stop_training(self) -> bool:
        """Determine if training should stop based on convergence."""
        return (self.convergence_history and self.convergence_history[-1] and 
                self.config.enable_convergence_monitoring)


class AsynchronousFederatedLearning:
    """Asynchronous federated learning with staleness handling."""
    
    def __init__(self, config: PrivacyConfig):
        self.config = config
        self.client_versions: Dict[str, int] = {}
        self.global_version = 0
        self.pending_updates: Dict[str, ModelUpdate] = {}
        self.update_lock = threading.Lock()
        
    def register_client_update(self, client_id: str, model_update: ModelUpdate, 
                             client_version: int) -> bool:
        """Register an asynchronous client update."""
        if not self.config.enable_async_fl:
            return False
        
        staleness = self.global_version - client_version
        
        if staleness > self.config.async_staleness_bound:
            logger.warning(f"Update from {client_id} too stale (staleness: {staleness})")
            return False
        
        with self.update_lock:
            # Apply staleness-aware weighting
            staleness_weight = self.config.async_weight_decay ** staleness
            
            # Weight the model update
            for name, param in model_update.model_weights.items():
                model_update.model_weights[name] = param * staleness_weight
            
            self.pending_updates[client_id] = model_update
            self.client_versions[client_id] = client_version
        
        logger.debug(f"Registered async update from {client_id} with staleness {staleness}")
        return True
    
    def get_pending_updates(self) -> List[ModelUpdate]:
        """Get all pending updates for aggregation."""
        with self.update_lock:
            updates = list(self.pending_updates.values())
            self.pending_updates.clear()
            return updates
    
    def increment_global_version(self):
        """Increment global model version."""
        self.global_version += 1
        logger.debug(f"Global model version incremented to {self.global_version}")
    
    def get_async_stats(self) -> Dict[str, Any]:
        """Get asynchronous FL statistics."""
        return {
            "global_version": self.global_version,
            "active_clients": len(self.client_versions),
            "pending_updates": len(self.pending_updates),
            "client_versions": dict(self.client_versions),
            "staleness_bound": self.config.async_staleness_bound,
            "weight_decay": self.config.async_weight_decay
        }


class PrivacyPreservationManager:
    """Main manager for all privacy preservation mechanisms."""
    
    def __init__(self, config: Optional[PrivacyConfig] = None):
        self.config = config or PrivacyConfig()
        
        # Initialize components
        self.dp_manager = AdvancedDifferentialPrivacy(self.config)
        self.secure_aggregation = SecureAggregationProtocol(self.config)
        self.convergence_monitor = ConvergenceMonitor(self.config)
        self.async_fl = AsynchronousFederatedLearning(self.config)
        
        # Privacy tracking
        self.privacy_events: List[Dict[str, Any]] = []
        
        logger.info("Privacy preservation manager initialized")
    
    def apply_privacy_mechanisms(self, model_update: ModelUpdate, 
                                mechanism_types: List[PrivacyMechanism]) -> ModelUpdate:
        """Apply multiple privacy mechanisms to a model update."""
        protected_update = copy.deepcopy(model_update)
        privacy_applied = []
        
        for mechanism in mechanism_types:
            if mechanism == PrivacyMechanism.DIFFERENTIAL_PRIVACY:
                if self.config.enable_differential_privacy:
                    noisy_weights, noise_amount = self.dp_manager.add_gaussian_noise(
                        protected_update.model_weights
                    )
                    protected_update.model_weights = noisy_weights
                    protected_update.differential_privacy_applied = True
                    protected_update.epsilon_used = self.config.dp_epsilon
                    privacy_applied.append("differential_privacy")
        
        # Log privacy application
        self.privacy_events.append({
            "timestamp": datetime.now().isoformat(),
            "client_id": model_update.client_id,
            "mechanisms_applied": privacy_applied,
            "privacy_budget_used": self.config.dp_epsilon if "differential_privacy" in privacy_applied else 0.0
        })
        
        return protected_update
    
    def get_privacy_report(self) -> Dict[str, Any]:
        """Generate comprehensive privacy report."""
        return {
            "differential_privacy": self.dp_manager.get_privacy_spent(),
            "secure_aggregation": {
                "enabled": self.config.enable_secure_aggregation,
                "registered_clients": len(self.secure_aggregation.client_keys),
                "min_clients": self.config.min_clients_for_aggregation
            },
            "convergence_monitoring": self.convergence_monitor.get_convergence_metrics(),
            "asynchronous_fl": self.async_fl.get_async_stats(),
            "privacy_events": len(self.privacy_events),
            "recent_events": self.privacy_events[-10:] if self.privacy_events else []
        }
    
    def validate_privacy_guarantees(self) -> Dict[str, bool]:
        """Validate that privacy guarantees are maintained."""
        validations = {}
        
        # Check differential privacy budget
        dp_valid = (self.dp_manager.privacy_budget.remaining_budget >= 0 and
                   self.dp_manager.privacy_budget.used_budget <= self.dp_manager.privacy_budget.total_budget)
        validations["differential_privacy_budget"] = dp_valid
        
        # Check secure aggregation requirements
        sa_valid = (not self.config.enable_secure_aggregation or 
                   len(self.secure_aggregation.client_keys) >= self.config.min_clients_for_aggregation)
        validations["secure_aggregation_clients"] = sa_valid
        
        # Overall privacy guarantee
        validations["overall_privacy_guarantee"] = all(validations.values())
        
        return validations