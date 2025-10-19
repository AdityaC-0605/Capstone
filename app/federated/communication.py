"""
Federated Learning Communication Protocols.

This module implements secure communication protocols for federated learning,
including message serialization, encryption, and network communication utilities.
"""

import asyncio
import base64
import gzip
import hashlib
import hmac
import io
import json
import logging
import secrets
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import aiohttp
import torch
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

try:
    from ..core.logging import get_logger
    from .federated_server import ClientInfo, FederatedConfig, ModelUpdate
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).parent.parent))

    from core.logging import get_logger
    from federated.federated_server import ClientInfo, FederatedConfig, ModelUpdate

logger = get_logger(__name__)


class MessageType(Enum):
    """Types of federated learning messages."""

    CLIENT_REGISTRATION = "client_registration"
    CLIENT_UNREGISTRATION = "client_unregistration"
    MODEL_REQUEST = "model_request"
    MODEL_RESPONSE = "model_response"
    TRAINING_START = "training_start"
    MODEL_UPDATE = "model_update"
    AGGREGATION_COMPLETE = "aggregation_complete"
    ROUND_COMPLETE = "round_complete"
    SERVER_STATUS = "server_status"
    HEARTBEAT = "heartbeat"
    ERROR = "error"


@dataclass
class FederatedMessage:
    """Base class for federated learning messages."""

    message_type: MessageType
    sender_id: str
    recipient_id: str
    timestamp: datetime
    message_id: str
    payload: Dict[str, Any]

    # Security fields
    signature: Optional[str] = None
    encrypted: bool = False
    compression: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for serialization."""
        return {
            "message_type": self.message_type.value,
            "sender_id": self.sender_id,
            "recipient_id": self.recipient_id,
            "timestamp": self.timestamp.isoformat(),
            "message_id": self.message_id,
            "payload": self.payload,
            "signature": self.signature,
            "encrypted": self.encrypted,
            "compression": self.compression,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FederatedMessage":
        """Create message from dictionary."""
        return cls(
            message_type=MessageType(data["message_type"]),
            sender_id=data["sender_id"],
            recipient_id=data["recipient_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            message_id=data["message_id"],
            payload=data["payload"],
            signature=data.get("signature"),
            encrypted=data.get("encrypted", False),
            compression=data.get("compression", False),
        )


class MessageSerializer:
    """Handles serialization and deserialization of federated messages."""

    @staticmethod
    def serialize_model_weights(weights: Dict[str, torch.Tensor]) -> str:
        """
        Serialize PyTorch model weights to base64 string.

        Args:
            weights: Dictionary of model weights

        Returns:
            Base64 encoded serialized weights
        """
        try:
            # Serialize to bytes
            buffer = io.BytesIO()
            torch.save(weights, buffer)
            serialized_bytes = buffer.getvalue()

            # Compress if beneficial
            compressed_bytes = gzip.compress(serialized_bytes)
            if len(compressed_bytes) < len(serialized_bytes):
                serialized_bytes = compressed_bytes
                compression_used = True
            else:
                compression_used = False

            # Encode to base64
            encoded = base64.b64encode(serialized_bytes).decode("utf-8")

            logger.debug(
                f"Serialized model weights: {len(weights)} tensors, "
                f"{len(encoded)} chars, compression: {compression_used}"
            )

            return encoded

        except Exception as e:
            logger.error(f"Failed to serialize model weights: {e}")
            raise

    @staticmethod
    def deserialize_model_weights(encoded_weights: str) -> Dict[str, torch.Tensor]:
        """
        Deserialize model weights from base64 string.

        Args:
            encoded_weights: Base64 encoded serialized weights

        Returns:
            Dictionary of model weights
        """
        try:
            # Decode from base64
            serialized_bytes = base64.b64decode(encoded_weights.encode("utf-8"))

            # Try decompression first
            try:
                decompressed_bytes = gzip.decompress(serialized_bytes)
                serialized_bytes = decompressed_bytes
            except:
                # Not compressed, use as is
                pass

            # Deserialize weights
            buffer = io.BytesIO(serialized_bytes)
            weights = torch.load(buffer, weights_only=True)

            logger.debug(f"Deserialized model weights: {len(weights)} tensors")
            return weights

        except Exception as e:
            logger.error(f"Failed to deserialize model weights: {e}")
            raise

    @staticmethod
    def serialize_message(message: FederatedMessage) -> str:
        """
        Serialize a federated message to JSON string.

        Args:
            message: FederatedMessage to serialize

        Returns:
            JSON string representation
        """
        try:
            message_dict = message.to_dict()
            json_str = json.dumps(message_dict, indent=None, separators=(",", ":"))

            if message.compression:
                # Compress JSON if requested
                compressed = gzip.compress(json_str.encode("utf-8"))
                json_str = base64.b64encode(compressed).decode("utf-8")

            return json_str

        except Exception as e:
            logger.error(f"Failed to serialize message: {e}")
            raise

    @staticmethod
    def deserialize_message(json_str: str) -> FederatedMessage:
        """
        Deserialize a federated message from JSON string.

        Args:
            json_str: JSON string representation

        Returns:
            FederatedMessage object
        """
        try:
            # Try decompression first
            try:
                compressed_bytes = base64.b64decode(json_str.encode("utf-8"))
                decompressed = gzip.decompress(compressed_bytes)
                json_str = decompressed.decode("utf-8")
            except:
                # Not compressed, use as is
                pass

            message_dict = json.loads(json_str)
            return FederatedMessage.from_dict(message_dict)

        except Exception as e:
            logger.error(f"Failed to deserialize message: {e}")
            raise


class SecureCommunicator:
    """Handles secure communication between federated participants."""

    def __init__(self, private_key: Optional[rsa.RSAPrivateKey] = None):
        if private_key is None:
            self.private_key = rsa.generate_private_key(
                public_exponent=65537, key_size=2048, backend=default_backend()
            )
        else:
            self.private_key = private_key

        self.public_key = self.private_key.public_key()
        self.peer_public_keys: Dict[str, rsa.RSAPublicKey] = {}

    def get_public_key_pem(self) -> str:
        """Get public key in PEM format."""
        return self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        ).decode("utf-8")

    def add_peer_public_key(self, peer_id: str, public_key_pem: str):
        """Add a peer's public key for secure communication."""
        try:
            public_key = serialization.load_pem_public_key(
                public_key_pem.encode("utf-8"), backend=default_backend()
            )
            self.peer_public_keys[peer_id] = public_key
            logger.debug(f"Added public key for peer {peer_id}")
        except Exception as e:
            logger.error(f"Failed to add public key for peer {peer_id}: {e}")
            raise

    def encrypt_message(self, message: str, recipient_id: str) -> str:
        """
        Encrypt a message for a specific recipient.

        Args:
            message: Message to encrypt
            recipient_id: ID of the recipient

        Returns:
            Base64 encoded encrypted message
        """
        if recipient_id not in self.peer_public_keys:
            raise ValueError(f"No public key found for recipient {recipient_id}")

        try:
            recipient_key = self.peer_public_keys[recipient_id]

            # For large messages, use hybrid encryption
            if len(message.encode("utf-8")) > 190:  # RSA key size limit
                return self._hybrid_encrypt(message, recipient_key)
            else:
                # Direct RSA encryption for small messages
                encrypted_bytes = recipient_key.encrypt(
                    message.encode("utf-8"),
                    padding.OAEP(
                        mgf=padding.MGF1(algorithm=hashes.SHA256()),
                        algorithm=hashes.SHA256(),
                        label=None,
                    ),
                )
                return base64.b64encode(encrypted_bytes).decode("utf-8")

        except Exception as e:
            logger.error(f"Failed to encrypt message for {recipient_id}: {e}")
            raise

    def decrypt_message(self, encrypted_message: str, sender_id: str) -> str:
        """
        Decrypt a message from a specific sender.

        Args:
            encrypted_message: Base64 encoded encrypted message
            sender_id: ID of the sender

        Returns:
            Decrypted message
        """
        try:
            encrypted_bytes = base64.b64decode(encrypted_message.encode("utf-8"))

            # Check if this is hybrid encryption
            if len(encrypted_bytes) > 256:  # Likely hybrid encryption
                return self._hybrid_decrypt(encrypted_bytes)
            else:
                # Direct RSA decryption
                decrypted_bytes = self.private_key.decrypt(
                    encrypted_bytes,
                    padding.OAEP(
                        mgf=padding.MGF1(algorithm=hashes.SHA256()),
                        algorithm=hashes.SHA256(),
                        label=None,
                    ),
                )
                return decrypted_bytes.decode("utf-8")

        except Exception as e:
            logger.error(f"Failed to decrypt message from {sender_id}: {e}")
            raise

    def _hybrid_encrypt(self, message: str, recipient_key: rsa.RSAPublicKey) -> str:
        """Hybrid encryption for large messages."""
        # Generate symmetric key
        symmetric_key = secrets.token_bytes(32)  # 256-bit key
        iv = secrets.token_bytes(16)  # 128-bit IV

        # Encrypt message with symmetric key
        cipher = Cipher(
            algorithms.AES(symmetric_key), modes.CBC(iv), backend=default_backend()
        )
        encryptor = cipher.encryptor()

        # Pad message to block size
        message_bytes = message.encode("utf-8")
        padding_length = 16 - (len(message_bytes) % 16)
        padded_message = message_bytes + bytes([padding_length] * padding_length)

        encrypted_message = encryptor.update(padded_message) + encryptor.finalize()

        # Encrypt symmetric key with RSA
        encrypted_key = recipient_key.encrypt(
            symmetric_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None,
            ),
        )

        # Combine encrypted key, IV, and encrypted message
        combined = encrypted_key + iv + encrypted_message
        return base64.b64encode(combined).decode("utf-8")

    def _hybrid_decrypt(self, encrypted_data: bytes) -> str:
        """Hybrid decryption for large messages."""
        # Extract components
        encrypted_key = encrypted_data[:256]  # RSA key size
        iv = encrypted_data[256:272]  # 16 bytes IV
        encrypted_message = encrypted_data[272:]

        # Decrypt symmetric key
        symmetric_key = self.private_key.decrypt(
            encrypted_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None,
            ),
        )

        # Decrypt message
        cipher = Cipher(
            algorithms.AES(symmetric_key), modes.CBC(iv), backend=default_backend()
        )
        decryptor = cipher.decryptor()
        decrypted_padded = decryptor.update(encrypted_message) + decryptor.finalize()

        # Remove padding
        padding_length = decrypted_padded[-1]
        decrypted_message = decrypted_padded[:-padding_length]

        return decrypted_message.decode("utf-8")

    def sign_message(self, message: str) -> str:
        """
        Sign a message with private key.

        Args:
            message: Message to sign

        Returns:
            Base64 encoded signature
        """
        try:
            signature = self.private_key.sign(
                message.encode("utf-8"),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH,
                ),
                hashes.SHA256(),
            )
            return base64.b64encode(signature).decode("utf-8")

        except Exception as e:
            logger.error(f"Failed to sign message: {e}")
            raise

    def verify_signature(self, message: str, signature: str, sender_id: str) -> bool:
        """
        Verify a message signature.

        Args:
            message: Original message
            signature: Base64 encoded signature
            sender_id: ID of the sender

        Returns:
            True if signature is valid, False otherwise
        """
        if sender_id not in self.peer_public_keys:
            logger.warning(f"No public key found for sender {sender_id}")
            return False

        try:
            sender_key = self.peer_public_keys[sender_id]
            signature_bytes = base64.b64decode(signature.encode("utf-8"))

            sender_key.verify(
                signature_bytes,
                message.encode("utf-8"),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH,
                ),
                hashes.SHA256(),
            )
            return True

        except Exception as e:
            logger.warning(f"Signature verification failed for sender {sender_id}: {e}")
            return False


class FederatedCommunicationManager:
    """Manages communication between federated server and clients."""

    def __init__(
        self,
        config: FederatedConfig,
        secure_communicator: Optional[SecureCommunicator] = None,
    ):
        self.config = config
        self.communicator = secure_communicator or SecureCommunicator()
        self.session: Optional[aiohttp.ClientSession] = None
        self.message_handlers: Dict[MessageType, Callable] = {}

    async def start_session(self):
        """Start HTTP session for communication."""
        if self.session is None:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)

    async def close_session(self):
        """Close HTTP session."""
        if self.session is not None:
            await self.session.close()
            self.session = None

    def register_message_handler(self, message_type: MessageType, handler: Callable):
        """Register a handler for a specific message type."""
        self.message_handlers[message_type] = handler
        logger.debug(f"Registered handler for message type: {message_type.value}")

    async def send_message(
        self, message: FederatedMessage, target_url: str
    ) -> Optional[FederatedMessage]:
        """
        Send a federated message to a target URL.

        Args:
            message: Message to send
            target_url: Target URL

        Returns:
            Response message if any, None otherwise
        """
        try:
            await self.start_session()

            # Serialize message
            serialized_message = MessageSerializer.serialize_message(message)

            # Encrypt if required
            if self.config.enable_encryption and message.recipient_id != "broadcast":
                encrypted_message = self.communicator.encrypt_message(
                    serialized_message, message.recipient_id
                )
                message.encrypted = True
                message.payload = {"encrypted_data": encrypted_message}
                serialized_message = MessageSerializer.serialize_message(message)

            # Sign message
            if message.sender_id:
                signature = self.communicator.sign_message(serialized_message)
                message.signature = signature
                serialized_message = MessageSerializer.serialize_message(message)

            # Send HTTP request
            headers = {
                "Content-Type": "application/json",
                "X-Federated-Message-Type": message.message_type.value,
                "X-Federated-Sender": message.sender_id,
            }

            async with self.session.post(
                target_url, data=serialized_message, headers=headers
            ) as response:
                if response.status == 200:
                    response_data = await response.text()
                    if response_data:
                        response_message = MessageSerializer.deserialize_message(
                            response_data
                        )

                        # Verify signature if present
                        if response_message.signature and response_message.sender_id:
                            if not self.communicator.verify_signature(
                                response_data,
                                response_message.signature,
                                response_message.sender_id,
                            ):
                                logger.warning(
                                    f"Invalid signature in response from {response_message.sender_id}"
                                )
                                return None

                        # Decrypt if encrypted
                        if response_message.encrypted:
                            encrypted_data = response_message.payload.get(
                                "encrypted_data"
                            )
                            if encrypted_data:
                                decrypted_data = self.communicator.decrypt_message(
                                    encrypted_data, response_message.sender_id
                                )
                                response_message = (
                                    MessageSerializer.deserialize_message(
                                        decrypted_data
                                    )
                                )

                        return response_message
                else:
                    logger.error(
                        f"HTTP error {response.status} when sending message to {target_url}"
                    )

        except Exception as e:
            logger.error(f"Failed to send message to {target_url}: {e}")

        return None

    async def handle_incoming_message(
        self, request_data: str, sender_ip: str
    ) -> Optional[FederatedMessage]:
        """
        Handle an incoming federated message.

        Args:
            request_data: Raw request data
            sender_ip: IP address of sender

        Returns:
            Response message if any, None otherwise
        """
        try:
            # Deserialize message
            message = MessageSerializer.deserialize_message(request_data)

            # Verify signature
            if message.signature and message.sender_id:
                if not self.communicator.verify_signature(
                    request_data, message.signature, message.sender_id
                ):
                    logger.warning(
                        f"Invalid signature from {message.sender_id} at {sender_ip}"
                    )
                    return self._create_error_response(message, "Invalid signature")

            # Decrypt if encrypted
            if message.encrypted:
                encrypted_data = message.payload.get("encrypted_data")
                if encrypted_data:
                    decrypted_data = self.communicator.decrypt_message(
                        encrypted_data, message.sender_id
                    )
                    message = MessageSerializer.deserialize_message(decrypted_data)

            # Handle message based on type
            if message.message_type in self.message_handlers:
                handler = self.message_handlers[message.message_type]
                response = await handler(message, sender_ip)
                return response
            else:
                logger.warning(
                    f"No handler for message type: {message.message_type.value}"
                )
                return self._create_error_response(message, "Unknown message type")

        except Exception as e:
            logger.error(f"Failed to handle incoming message from {sender_ip}: {e}")
            return None

    def _create_error_response(
        self, original_message: FederatedMessage, error_message: str
    ) -> FederatedMessage:
        """Create an error response message."""
        return FederatedMessage(
            message_type=MessageType.ERROR,
            sender_id="server",
            recipient_id=original_message.sender_id,
            timestamp=datetime.now(),
            message_id=f"error_{secrets.token_hex(8)}",
            payload={
                "error": error_message,
                "original_message_id": original_message.message_id,
            },
        )

    def create_model_update_message(
        self, sender_id: str, recipient_id: str, model_update: ModelUpdate
    ) -> FederatedMessage:
        """
        Create a model update message.

        Args:
            sender_id: ID of the sender
            recipient_id: ID of the recipient
            model_update: ModelUpdate object

        Returns:
            FederatedMessage containing the model update
        """
        # Serialize model weights
        serialized_weights = MessageSerializer.serialize_model_weights(
            model_update.model_weights
        )

        payload = {
            "client_id": model_update.client_id,
            "round_number": model_update.round_number,
            "model_weights": serialized_weights,
            "data_size": model_update.data_size,
            "training_loss": model_update.training_loss,
            "validation_metrics": model_update.validation_metrics,
            "training_time": model_update.training_time,
            "energy_consumed": model_update.energy_consumed,
            "differential_privacy_applied": model_update.differential_privacy_applied,
            "epsilon_used": model_update.epsilon_used,
            "timestamp": model_update.timestamp.isoformat(),
        }

        return FederatedMessage(
            message_type=MessageType.MODEL_UPDATE,
            sender_id=sender_id,
            recipient_id=recipient_id,
            timestamp=datetime.now(),
            message_id=f"model_update_{secrets.token_hex(8)}",
            payload=payload,
        )

    def parse_model_update_message(self, message: FederatedMessage) -> ModelUpdate:
        """
        Parse a model update message.

        Args:
            message: FederatedMessage containing model update

        Returns:
            ModelUpdate object
        """
        payload = message.payload

        # Deserialize model weights
        model_weights = MessageSerializer.deserialize_model_weights(
            payload["model_weights"]
        )

        return ModelUpdate(
            client_id=payload["client_id"],
            round_number=payload["round_number"],
            model_weights=model_weights,
            data_size=payload["data_size"],
            training_loss=payload["training_loss"],
            validation_metrics=payload["validation_metrics"],
            training_time=payload["training_time"],
            energy_consumed=payload.get("energy_consumed", 0.0),
            differential_privacy_applied=payload.get(
                "differential_privacy_applied", False
            ),
            epsilon_used=payload.get("epsilon_used", 0.0),
            timestamp=datetime.fromisoformat(payload["timestamp"]),
        )


# Utility functions
def create_registration_message(
    client_id: str,
    public_key: str,
    ip_address: str,
    port: int,
    capabilities: Optional[Dict[str, Any]] = None,
) -> FederatedMessage:
    """Create a client registration message."""
    payload = {
        "client_id": client_id,
        "public_key": public_key,
        "ip_address": ip_address,
        "port": port,
        "capabilities": capabilities or {},
    }

    return FederatedMessage(
        message_type=MessageType.CLIENT_REGISTRATION,
        sender_id=client_id,
        recipient_id="server",
        timestamp=datetime.now(),
        message_id=f"registration_{secrets.token_hex(8)}",
        payload=payload,
    )


def create_heartbeat_message(client_id: str) -> FederatedMessage:
    """Create a heartbeat message."""
    return FederatedMessage(
        message_type=MessageType.HEARTBEAT,
        sender_id=client_id,
        recipient_id="server",
        timestamp=datetime.now(),
        message_id=f"heartbeat_{secrets.token_hex(8)}",
        payload={"status": "alive"},
    )
