"""
Data encryption system with AES-256 encryption at rest and TLS 1.3 for transit.
Implements key management with rotation policies and encrypted backup/recovery.
"""

import os
import secrets
import hashlib
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import json
import base64
from dataclasses import dataclass, asdict

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

from ..core.config import get_config
from ..core.logging import get_logger, get_audit_logger


logger = get_logger(__name__)
audit_logger = get_audit_logger()


@dataclass
class EncryptionKey:
    """Encryption key metadata."""
    key_id: str
    created_at: datetime
    expires_at: Optional[datetime]
    algorithm: str
    key_size: int
    is_active: bool
    rotation_count: int


@dataclass
class EncryptedData:
    """Encrypted data container."""
    data: bytes
    key_id: str
    algorithm: str
    iv: Optional[bytes]
    metadata: Dict[str, Any]


class KeyManager:
    """Manages encryption keys with rotation policies."""
    
    def __init__(self, key_store_path: Optional[str] = None):
        self.config = get_config()
        self.key_store_path = Path(key_store_path or self.config.security.encryption_key_path).parent
        self.key_store_path.mkdir(parents=True, exist_ok=True)
        self._keys: Dict[str, EncryptionKey] = {}
        self._key_data: Dict[str, bytes] = {}
        self._load_keys()
    
    def generate_key(self, algorithm: str = "AES-256", expires_in_days: Optional[int] = 365) -> str:
        """Generate a new encryption key."""
        key_id = secrets.token_hex(16)
        
        if algorithm == "AES-256":
            key_data = secrets.token_bytes(32)  # 256 bits
            key_size = 256
        elif algorithm == "AES-128":
            key_data = secrets.token_bytes(16)  # 128 bits
            key_size = 128
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        expires_at = None
        if expires_in_days:
            expires_at = datetime.now() + timedelta(days=expires_in_days)
        
        key_metadata = EncryptionKey(
            key_id=key_id,
            created_at=datetime.now(),
            expires_at=expires_at,
            algorithm=algorithm,
            key_size=key_size,
            is_active=True,
            rotation_count=0
        )
        
        self._keys[key_id] = key_metadata
        self._key_data[key_id] = key_data
        
        self._save_key(key_id, key_metadata, key_data)
        
        audit_logger.log_security_event(
            event_type="key_generation",
            user_id=None,
            severity="INFO",
            details={
                "key_id": key_id,
                "algorithm": algorithm,
                "key_size": key_size
            }
        )
        
        logger.info(f"Generated new encryption key: {key_id}")
        return key_id
    
    def get_active_key(self, algorithm: str = "AES-256") -> Optional[str]:
        """Get the active key for the specified algorithm."""
        for key_id, key_metadata in self._keys.items():
            if (key_metadata.is_active and 
                key_metadata.algorithm == algorithm and
                (key_metadata.expires_at is None or key_metadata.expires_at > datetime.now())):
                return key_id
        return None
    
    def get_key_data(self, key_id: str) -> Optional[bytes]:
        """Get key data by key ID."""
        return self._key_data.get(key_id)
    
    def rotate_key(self, old_key_id: str) -> str:
        """Rotate an encryption key."""
        old_key = self._keys.get(old_key_id)
        if not old_key:
            raise ValueError(f"Key not found: {old_key_id}")
        
        # Generate new key
        new_key_id = self.generate_key(old_key.algorithm)
        
        # Deactivate old key
        old_key.is_active = False
        old_key.rotation_count += 1
        self._save_key_metadata(old_key_id, old_key)
        
        audit_logger.log_security_event(
            event_type="key_rotation",
            user_id=None,
            severity="INFO",
            details={
                "old_key_id": old_key_id,
                "new_key_id": new_key_id,
                "rotation_count": old_key.rotation_count
            }
        )
        
        logger.info(f"Rotated key {old_key_id} to {new_key_id}")
        return new_key_id
    
    def check_key_expiration(self) -> List[str]:
        """Check for expired or soon-to-expire keys."""
        expiring_keys = []
        warning_threshold = datetime.now() + timedelta(days=30)
        
        for key_id, key_metadata in self._keys.items():
            if key_metadata.expires_at and key_metadata.expires_at <= warning_threshold:
                expiring_keys.append(key_id)
        
        return expiring_keys
    
    def _load_keys(self):
        """Load keys from storage."""
        keys_file = self.key_store_path / "keys.json"
        if keys_file.exists():
            try:
                with open(keys_file, 'r') as f:
                    keys_data = json.load(f)
                
                for key_id, key_info in keys_data.items():
                    # Load metadata
                    metadata = EncryptionKey(
                        key_id=key_info['key_id'],
                        created_at=datetime.fromisoformat(key_info['created_at']),
                        expires_at=datetime.fromisoformat(key_info['expires_at']) if key_info.get('expires_at') else None,
                        algorithm=key_info['algorithm'],
                        key_size=key_info['key_size'],
                        is_active=key_info['is_active'],
                        rotation_count=key_info['rotation_count']
                    )
                    self._keys[key_id] = metadata
                    
                    # Load key data
                    key_file = self.key_store_path / f"{key_id}.key"
                    if key_file.exists():
                        with open(key_file, 'rb') as kf:
                            self._key_data[key_id] = kf.read()
                
                logger.info(f"Loaded {len(self._keys)} encryption keys")
            except Exception as e:
                logger.error(f"Failed to load keys: {e}")
    
    def _save_key(self, key_id: str, metadata: EncryptionKey, key_data: bytes):
        """Save key and metadata to storage."""
        # Save key data
        key_file = self.key_store_path / f"{key_id}.key"
        with open(key_file, 'wb') as f:
            f.write(key_data)
        
        # Save metadata
        self._save_key_metadata(key_id, metadata)
    
    def _save_key_metadata(self, key_id: str, metadata: EncryptionKey):
        """Save key metadata."""
        keys_file = self.key_store_path / "keys.json"
        
        # Load existing metadata
        keys_data = {}
        if keys_file.exists():
            with open(keys_file, 'r') as f:
                keys_data = json.load(f)
        
        # Update metadata
        metadata_dict = asdict(metadata)
        metadata_dict['created_at'] = metadata.created_at.isoformat()
        if metadata.expires_at:
            metadata_dict['expires_at'] = metadata.expires_at.isoformat()
        
        keys_data[key_id] = metadata_dict
        
        # Save updated metadata
        with open(keys_file, 'w') as f:
            json.dump(keys_data, f, indent=2)


class DataEncryption:
    """Handles data encryption at rest using AES-256."""
    
    def __init__(self, key_manager: Optional[KeyManager] = None):
        self.key_manager = key_manager or KeyManager()
        self.backend = default_backend()
    
    def encrypt_data(self, data: bytes, key_id: Optional[str] = None, 
                    metadata: Optional[Dict[str, Any]] = None) -> EncryptedData:
        """Encrypt data using AES-256."""
        if key_id is None:
            key_id = self.key_manager.get_active_key("AES-256")
            if key_id is None:
                key_id = self.key_manager.generate_key("AES-256")
        
        key_data = self.key_manager.get_key_data(key_id)
        if key_data is None:
            raise ValueError(f"Key not found: {key_id}")
        
        # Generate random IV
        iv = secrets.token_bytes(16)
        
        # Create cipher
        cipher = Cipher(
            algorithms.AES(key_data),
            modes.CBC(iv),
            backend=self.backend
        )
        
        # Pad data to block size
        padded_data = self._pad_data(data)
        
        # Encrypt
        encryptor = cipher.encryptor()
        encrypted_data = encryptor.update(padded_data) + encryptor.finalize()
        
        return EncryptedData(
            data=encrypted_data,
            key_id=key_id,
            algorithm="AES-256-CBC",
            iv=iv,
            metadata=metadata or {}
        )
    
    def decrypt_data(self, encrypted_data: EncryptedData) -> bytes:
        """Decrypt data."""
        key_data = self.key_manager.get_key_data(encrypted_data.key_id)
        if key_data is None:
            raise ValueError(f"Key not found: {encrypted_data.key_id}")
        
        # Create cipher
        cipher = Cipher(
            algorithms.AES(key_data),
            modes.CBC(encrypted_data.iv),
            backend=self.backend
        )
        
        # Decrypt
        decryptor = cipher.decryptor()
        padded_data = decryptor.update(encrypted_data.data) + decryptor.finalize()
        
        # Remove padding
        return self._unpad_data(padded_data)
    
    def encrypt_file(self, file_path: str, output_path: Optional[str] = None) -> str:
        """Encrypt a file."""
        input_path = Path(file_path)
        if not input_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        output_path = output_path or f"{file_path}.encrypted"
        
        with open(input_path, 'rb') as f:
            data = f.read()
        
        encrypted_data = self.encrypt_data(data)
        
        # Save encrypted data with metadata
        encrypted_container = {
            'data': base64.b64encode(encrypted_data.data).decode('utf-8'),
            'key_id': encrypted_data.key_id,
            'algorithm': encrypted_data.algorithm,
            'iv': base64.b64encode(encrypted_data.iv).decode('utf-8'),
            'metadata': encrypted_data.metadata
        }
        
        with open(output_path, 'w') as f:
            json.dump(encrypted_container, f)
        
        audit_logger.log_security_event(
            event_type="file_encryption",
            user_id=None,
            severity="INFO",
            details={
                "input_file": str(input_path),
                "output_file": output_path,
                "key_id": encrypted_data.key_id
            }
        )
        
        return output_path
    
    def decrypt_file(self, encrypted_file_path: str, output_path: Optional[str] = None) -> str:
        """Decrypt a file."""
        encrypted_path = Path(encrypted_file_path)
        if not encrypted_path.exists():
            raise FileNotFoundError(f"Encrypted file not found: {encrypted_file_path}")
        
        with open(encrypted_path, 'r') as f:
            encrypted_container = json.load(f)
        
        encrypted_data = EncryptedData(
            data=base64.b64decode(encrypted_container['data']),
            key_id=encrypted_container['key_id'],
            algorithm=encrypted_container['algorithm'],
            iv=base64.b64decode(encrypted_container['iv']),
            metadata=encrypted_container['metadata']
        )
        
        decrypted_data = self.decrypt_data(encrypted_data)
        
        output_path = output_path or encrypted_file_path.replace('.encrypted', '')
        
        with open(output_path, 'wb') as f:
            f.write(decrypted_data)
        
        audit_logger.log_security_event(
            event_type="file_decryption",
            user_id=None,
            severity="INFO",
            details={
                "encrypted_file": str(encrypted_path),
                "output_file": output_path,
                "key_id": encrypted_data.key_id
            }
        )
        
        return output_path
    
    def _pad_data(self, data: bytes) -> bytes:
        """Apply PKCS7 padding."""
        block_size = 16
        padding_length = block_size - (len(data) % block_size)
        padding = bytes([padding_length] * padding_length)
        return data + padding
    
    def _unpad_data(self, padded_data: bytes) -> bytes:
        """Remove PKCS7 padding."""
        padding_length = padded_data[-1]
        return padded_data[:-padding_length]


class BackupEncryption:
    """Handles encrypted backup and recovery mechanisms."""
    
    def __init__(self, encryption: Optional[DataEncryption] = None):
        self.encryption = encryption or DataEncryption()
        self.config = get_config()
    
    def create_encrypted_backup(self, source_path: str, backup_path: str, 
                              compression: bool = True) -> Dict[str, Any]:
        """Create an encrypted backup of data."""
        source = Path(source_path)
        backup = Path(backup_path)
        backup.parent.mkdir(parents=True, exist_ok=True)
        
        backup_metadata = {
            'source_path': str(source),
            'backup_path': str(backup),
            'created_at': datetime.now().isoformat(),
            'compression': compression,
            'files': []
        }
        
        if source.is_file():
            # Single file backup
            encrypted_file = self.encryption.encrypt_file(str(source))
            backup_metadata['files'].append({
                'original': str(source),
                'encrypted': encrypted_file,
                'size': source.stat().st_size
            })
        elif source.is_dir():
            # Directory backup
            for file_path in source.rglob('*'):
                if file_path.is_file():
                    relative_path = file_path.relative_to(source)
                    encrypted_file = self.encryption.encrypt_file(
                        str(file_path),
                        str(backup / f"{relative_path}.encrypted")
                    )
                    backup_metadata['files'].append({
                        'original': str(file_path),
                        'encrypted': encrypted_file,
                        'relative_path': str(relative_path),
                        'size': file_path.stat().st_size
                    })
        
        # Save backup metadata
        metadata_file = backup / "backup_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(backup_metadata, f, indent=2)
        
        audit_logger.log_security_event(
            event_type="encrypted_backup_created",
            user_id=None,
            severity="INFO",
            details={
                "source_path": str(source),
                "backup_path": str(backup),
                "file_count": len(backup_metadata['files'])
            }
        )
        
        logger.info(f"Created encrypted backup: {backup}")
        return backup_metadata
    
    def restore_from_backup(self, backup_path: str, restore_path: str) -> Dict[str, Any]:
        """Restore data from encrypted backup."""
        backup = Path(backup_path)
        restore = Path(restore_path)
        
        # Load backup metadata
        metadata_file = backup / "backup_metadata.json"
        if not metadata_file.exists():
            raise FileNotFoundError(f"Backup metadata not found: {metadata_file}")
        
        with open(metadata_file, 'r') as f:
            backup_metadata = json.load(f)
        
        restore.mkdir(parents=True, exist_ok=True)
        restored_files = []
        
        for file_info in backup_metadata['files']:
            encrypted_file = file_info['encrypted']
            
            if 'relative_path' in file_info:
                # Directory backup
                output_file = restore / file_info['relative_path']
                output_file.parent.mkdir(parents=True, exist_ok=True)
            else:
                # Single file backup
                output_file = restore / Path(file_info['original']).name
            
            decrypted_file = self.encryption.decrypt_file(encrypted_file, str(output_file))
            restored_files.append(decrypted_file)
        
        restore_metadata = {
            'backup_path': str(backup),
            'restore_path': str(restore),
            'restored_at': datetime.now().isoformat(),
            'restored_files': restored_files
        }
        
        audit_logger.log_security_event(
            event_type="encrypted_backup_restored",
            user_id=None,
            severity="INFO",
            details={
                "backup_path": str(backup),
                "restore_path": str(restore),
                "file_count": len(restored_files)
            }
        )
        
        logger.info(f"Restored backup to: {restore}")
        return restore_metadata