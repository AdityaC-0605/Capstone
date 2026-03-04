"""
Main security manager that orchestrates all security components.
"""

from typing import Any, Dict, Optional

from ..core.config import get_config
from ..core.logging import get_audit_logger, get_logger
from .anonymization import AnonymizationConfig, AnonymizationPipeline
from .auth import AuthenticationManager, Permission, Role
from .encryption import BackupEncryption, DataEncryption, KeyManager
from .gdpr_compliance import GDPRComplianceManager

logger = get_logger(__name__)
audit_logger = get_audit_logger()


class SecurityManager:
    """Central security manager for the sustainable credit risk AI system."""

    def __init__(self):
        self.config = get_config()

        # Initialize security components
        self.key_manager = KeyManager()
        self.data_encryption = DataEncryption(self.key_manager)
        self.backup_encryption = BackupEncryption(self.data_encryption)

        # Initialize anonymization
        anonymization_config = AnonymizationConfig(
            k_value=self.config.security.differential_privacy_epsilon or 5,
            epsilon=self.config.security.differential_privacy_epsilon or 1.0,
        )
        self.anonymization_pipeline = AnonymizationPipeline(
            anonymization_config
        )

        # Initialize authentication and authorization
        self.auth_manager = AuthenticationManager()

        # Initialize GDPR compliance
        self.gdpr_manager = GDPRComplianceManager()

        logger.info("Security manager initialized successfully")

    def encrypt_sensitive_data(
        self, data: bytes, metadata: Optional[Dict[str, Any]] = None
    ):
        """Encrypt sensitive data."""
        return self.data_encryption.encrypt_data(data, metadata=metadata)

    def decrypt_sensitive_data(self, encrypted_data):
        """Decrypt sensitive data."""
        return self.data_encryption.decrypt_data(encrypted_data)

    def anonymize_dataset(self, data, strategy: Dict[str, Any]):
        """Anonymize a dataset according to the specified strategy."""
        return self.anonymization_pipeline.anonymize_dataset(data, strategy)

    def authenticate_user(
        self, username: str, password: str, mfa_token: Optional[str] = None
    ):
        """Authenticate a user."""
        return self.auth_manager.authenticate_user(
            username, password, mfa_token
        )

    def check_user_permission(self, user, permission: Permission) -> bool:
        """Check if a user has a specific permission."""
        return self.auth_manager.check_permission(user, permission)

    def process_gdpr_request(
        self, request_type: str, data_subject_id: str, **kwargs
    ):
        """Process GDPR data subject requests."""
        return self.gdpr_manager.process_data_subject_request(
            request_type, data_subject_id, **kwargs
        )

    def create_encrypted_backup(self, source_path: str, backup_path: str):
        """Create an encrypted backup."""
        return self.backup_encryption.create_encrypted_backup(
            source_path, backup_path
        )

    def rotate_encryption_keys(self):
        """Rotate encryption keys as part of security maintenance."""
        # Check for keys that need rotation
        expiring_keys = self.key_manager.check_key_expiration()

        rotated_keys = []
        for key_id in expiring_keys:
            try:
                new_key_id = self.key_manager.rotate_key(key_id)
                rotated_keys.append((key_id, new_key_id))
                logger.info(f"Rotated key {key_id} to {new_key_id}")
            except Exception as e:
                logger.error(f"Failed to rotate key {key_id}: {e}")

        return rotated_keys

    def generate_security_report(self) -> Dict[str, Any]:
        """Generate comprehensive security status report."""
        # Get GDPR compliance report
        gdpr_report = self.gdpr_manager.generate_compliance_report()

        # Get key management status
        expiring_keys = self.key_manager.check_key_expiration()

        # Get authentication statistics
        total_users = len(self.auth_manager.users)
        active_sessions = len(
            [s for s in self.auth_manager.sessions.values() if s.is_active]
        )

        return {
            "report_generated_at": gdpr_report["report_generated_at"],
            "encryption": {
                "total_keys": len(self.key_manager._keys),
                "expiring_keys": len(expiring_keys),
                "active_keys": len(
                    [k for k in self.key_manager._keys.values() if k.is_active]
                ),
            },
            "authentication": {
                "total_users": total_users,
                "active_sessions": active_sessions,
                "total_api_keys": len(self.auth_manager.api_key_manager.keys),
            },
            "gdpr_compliance": gdpr_report,
            "security_recommendations": self._generate_security_recommendations(
                expiring_keys
            ),
        }

    def _generate_security_recommendations(self, expiring_keys) -> list:
        """Generate security recommendations based on current state."""
        recommendations = []

        if expiring_keys:
            recommendations.append(
                {
                    "priority": "high",
                    "category": "key_management",
                    "message": f"{len(expiring_keys)} encryption keys are expiring soon and should be rotated",
                }
            )

        # Check for inactive users
        inactive_users = [
            user
            for user in self.auth_manager.users.values()
            if user.last_login and (user.last_login.days > 90)
        ]

        if inactive_users:
            recommendations.append(
                {
                    "priority": "medium",
                    "category": "user_management",
                    "message": f"{len(inactive_users)} users have been inactive for over 90 days",
                }
            )

        return recommendations


# Global security manager instance
security_manager = SecurityManager()


def get_security_manager() -> SecurityManager:
    """Get the global security manager instance."""
    return security_manager
