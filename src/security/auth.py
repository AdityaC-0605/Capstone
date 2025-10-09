"""
Access control and authentication system with RBAC, API key management,
JWT tokens, audit logging, and multi-factor authentication.
"""

import secrets
import hashlib
import hmac
import time
import pyotp
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import jwt
import bcrypt
from pathlib import Path
import json

from ..core.config import get_config
from ..core.logging import get_logger, get_audit_logger


logger = get_logger(__name__)
audit_logger = get_audit_logger()


class Role(Enum):
    """User roles in the system."""
    ADMIN = "admin"
    DATA_SCIENTIST = "data_scientist"
    ANALYST = "analyst"
    VIEWER = "viewer"
    API_USER = "api_user"


class Permission(Enum):
    """System permissions."""
    READ_DATA = "read_data"
    WRITE_DATA = "write_data"
    DELETE_DATA = "delete_data"
    TRAIN_MODEL = "train_model"
    DEPLOY_MODEL = "deploy_model"
    VIEW_PREDICTIONS = "view_predictions"
    MANAGE_USERS = "manage_users"
    VIEW_AUDIT_LOGS = "view_audit_logs"
    MANAGE_KEYS = "manage_keys"
    EXPORT_DATA = "export_data"


@dataclass
class User:
    """User entity."""
    user_id: str
    username: str
    email: str
    password_hash: str
    roles: Set[Role]
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    last_login: Optional[datetime] = None
    failed_login_attempts: int = 0
    mfa_enabled: bool = False
    mfa_secret: Optional[str] = None
    api_keys: List[str] = field(default_factory=list)


@dataclass
class APIKey:
    """API key entity."""
    key_id: str
    key_hash: str
    user_id: str
    name: str
    permissions: Set[Permission]
    created_at: datetime
    expires_at: Optional[datetime]
    last_used: Optional[datetime]
    is_active: bool = True
    usage_count: int = 0


@dataclass
class Session:
    """User session."""
    session_id: str
    user_id: str
    created_at: datetime
    expires_at: datetime
    ip_address: str
    user_agent: str
    is_active: bool = True


class RoleBasedAccessControl:
    """Role-based access control system."""
    
    def __init__(self):
        self.role_permissions = self._initialize_role_permissions()
    
    def _initialize_role_permissions(self) -> Dict[Role, Set[Permission]]:
        """Initialize default role permissions."""
        return {
            Role.ADMIN: {
                Permission.READ_DATA,
                Permission.WRITE_DATA,
                Permission.DELETE_DATA,
                Permission.TRAIN_MODEL,
                Permission.DEPLOY_MODEL,
                Permission.VIEW_PREDICTIONS,
                Permission.MANAGE_USERS,
                Permission.VIEW_AUDIT_LOGS,
                Permission.MANAGE_KEYS,
                Permission.EXPORT_DATA
            },
            Role.DATA_SCIENTIST: {
                Permission.READ_DATA,
                Permission.WRITE_DATA,
                Permission.TRAIN_MODEL,
                Permission.VIEW_PREDICTIONS,
                Permission.EXPORT_DATA
            },
            Role.ANALYST: {
                Permission.READ_DATA,
                Permission.VIEW_PREDICTIONS,
                Permission.EXPORT_DATA
            },
            Role.VIEWER: {
                Permission.READ_DATA,
                Permission.VIEW_PREDICTIONS
            },
            Role.API_USER: {
                Permission.READ_DATA,
                Permission.VIEW_PREDICTIONS
            }
        }
    
    def has_permission(self, user_roles: Set[Role], permission: Permission) -> bool:
        """Check if user roles have the specified permission."""
        for role in user_roles:
            if permission in self.role_permissions.get(role, set()):
                return True
        return False
    
    def get_user_permissions(self, user_roles: Set[Role]) -> Set[Permission]:
        """Get all permissions for user roles."""
        permissions = set()
        for role in user_roles:
            permissions.update(self.role_permissions.get(role, set()))
        return permissions
    
    def add_role_permission(self, role: Role, permission: Permission):
        """Add permission to a role."""
        if role not in self.role_permissions:
            self.role_permissions[role] = set()
        self.role_permissions[role].add(permission)
    
    def remove_role_permission(self, role: Role, permission: Permission):
        """Remove permission from a role."""
        if role in self.role_permissions:
            self.role_permissions[role].discard(permission)


class PasswordManager:
    """Handles password hashing and validation."""
    
    def __init__(self):
        self.min_length = 8
        self.require_uppercase = True
        self.require_lowercase = True
        self.require_digits = True
        self.require_special = True
    
    def hash_password(self, password: str) -> str:
        """Hash a password using bcrypt."""
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    
    def verify_password(self, password: str, password_hash: str) -> bool:
        """Verify a password against its hash."""
        return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))
    
    def validate_password_strength(self, password: str) -> Tuple[bool, List[str]]:
        """Validate password strength."""
        errors = []
        
        if len(password) < self.min_length:
            errors.append(f"Password must be at least {self.min_length} characters long")
        
        if self.require_uppercase and not any(c.isupper() for c in password):
            errors.append("Password must contain at least one uppercase letter")
        
        if self.require_lowercase and not any(c.islower() for c in password):
            errors.append("Password must contain at least one lowercase letter")
        
        if self.require_digits and not any(c.isdigit() for c in password):
            errors.append("Password must contain at least one digit")
        
        if self.require_special and not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            errors.append("Password must contain at least one special character")
        
        return len(errors) == 0, errors


class MultiFactorAuth:
    """Multi-factor authentication using TOTP."""
    
    def __init__(self):
        self.issuer_name = "Sustainable Credit Risk AI"
    
    def generate_secret(self) -> str:
        """Generate a new MFA secret."""
        return pyotp.random_base32()
    
    def generate_qr_code_url(self, user_email: str, secret: str) -> str:
        """Generate QR code URL for MFA setup."""
        totp = pyotp.TOTP(secret)
        return totp.provisioning_uri(
            name=user_email,
            issuer_name=self.issuer_name
        )
    
    def verify_token(self, secret: str, token: str, window: int = 1) -> bool:
        """Verify TOTP token."""
        totp = pyotp.TOTP(secret)
        return totp.verify(token, valid_window=window)
    
    def generate_backup_codes(self, count: int = 10) -> List[str]:
        """Generate backup codes for MFA."""
        return [secrets.token_hex(4).upper() for _ in range(count)]


class APIKeyManager:
    """Manages API keys for authentication."""
    
    def __init__(self):
        self.keys: Dict[str, APIKey] = {}
        self._load_keys()
    
    def generate_api_key(self, user_id: str, name: str, permissions: Set[Permission],
                        expires_in_days: Optional[int] = None) -> Tuple[str, str]:
        """Generate a new API key."""
        # Generate key
        key = secrets.token_urlsafe(32)
        key_id = secrets.token_hex(16)
        
        # Hash key for storage
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        
        # Calculate expiration
        expires_at = None
        if expires_in_days:
            expires_at = datetime.now() + timedelta(days=expires_in_days)
        
        # Create API key object
        api_key = APIKey(
            key_id=key_id,
            key_hash=key_hash,
            user_id=user_id,
            name=name,
            permissions=permissions,
            created_at=datetime.now(),
            expires_at=expires_at,
            last_used=None,
            is_active=True,
            usage_count=0
        )
        
        self.keys[key_id] = api_key
        self._save_keys()
        
        audit_logger.log_security_event(
            event_type="api_key_generated",
            user_id=user_id,
            severity="INFO",
            details={
                "key_id": key_id,
                "name": name,
                "permissions": [p.value for p in permissions]
            }
        )
        
        return key, key_id
    
    def validate_api_key(self, key: str) -> Optional[APIKey]:
        """Validate an API key."""
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        
        for api_key in self.keys.values():
            if (api_key.key_hash == key_hash and 
                api_key.is_active and
                (api_key.expires_at is None or api_key.expires_at > datetime.now())):
                
                # Update usage
                api_key.last_used = datetime.now()
                api_key.usage_count += 1
                self._save_keys()
                
                return api_key
        
        return None
    
    def revoke_api_key(self, key_id: str, user_id: str) -> bool:
        """Revoke an API key."""
        if key_id in self.keys and self.keys[key_id].user_id == user_id:
            self.keys[key_id].is_active = False
            self._save_keys()
            
            audit_logger.log_security_event(
                event_type="api_key_revoked",
                user_id=user_id,
                severity="INFO",
                details={"key_id": key_id}
            )
            
            return True
        return False
    
    def list_user_keys(self, user_id: str) -> List[APIKey]:
        """List API keys for a user."""
        return [key for key in self.keys.values() if key.user_id == user_id]
    
    def _load_keys(self):
        """Load API keys from storage."""
        config = get_config()
        keys_file = Path(config.keys_path) / "api_keys.json"
        
        if keys_file.exists():
            try:
                with open(keys_file, 'r') as f:
                    keys_data = json.load(f)
                
                for key_id, key_info in keys_data.items():
                    api_key = APIKey(
                        key_id=key_info['key_id'],
                        key_hash=key_info['key_hash'],
                        user_id=key_info['user_id'],
                        name=key_info['name'],
                        permissions={Permission(p) for p in key_info['permissions']},
                        created_at=datetime.fromisoformat(key_info['created_at']),
                        expires_at=datetime.fromisoformat(key_info['expires_at']) if key_info.get('expires_at') else None,
                        last_used=datetime.fromisoformat(key_info['last_used']) if key_info.get('last_used') else None,
                        is_active=key_info['is_active'],
                        usage_count=key_info['usage_count']
                    )
                    self.keys[key_id] = api_key
                
                logger.info(f"Loaded {len(self.keys)} API keys")
            except Exception as e:
                logger.error(f"Failed to load API keys: {e}")
    
    def _save_keys(self):
        """Save API keys to storage."""
        config = get_config()
        keys_path = Path(config.keys_path)
        keys_path.mkdir(parents=True, exist_ok=True)
        keys_file = keys_path / "api_keys.json"
        
        keys_data = {}
        for key_id, api_key in self.keys.items():
            keys_data[key_id] = {
                'key_id': api_key.key_id,
                'key_hash': api_key.key_hash,
                'user_id': api_key.user_id,
                'name': api_key.name,
                'permissions': [p.value for p in api_key.permissions],
                'created_at': api_key.created_at.isoformat(),
                'expires_at': api_key.expires_at.isoformat() if api_key.expires_at else None,
                'last_used': api_key.last_used.isoformat() if api_key.last_used else None,
                'is_active': api_key.is_active,
                'usage_count': api_key.usage_count
            }
        
        with open(keys_file, 'w') as f:
            json.dump(keys_data, f, indent=2)


class JWTManager:
    """Manages JWT tokens for session authentication."""
    
    def __init__(self):
        self.config = get_config()
        self.secret_key = self.config.security.jwt_secret_key or secrets.token_hex(32)
        self.algorithm = "HS256"
        self.expiration_hours = self.config.security.jwt_expiration_hours
    
    def generate_token(self, user_id: str, roles: Set[Role], 
                      additional_claims: Optional[Dict[str, Any]] = None) -> str:
        """Generate a JWT token."""
        now = datetime.utcnow()
        payload = {
            'user_id': user_id,
            'roles': [role.value for role in roles],
            'iat': now,
            'exp': now + timedelta(hours=self.expiration_hours),
            'iss': 'sustainable-credit-risk-ai'
        }
        
        if additional_claims:
            payload.update(additional_claims)
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def validate_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Validate and decode a JWT token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("JWT token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid JWT token: {e}")
            return None
    
    def refresh_token(self, token: str) -> Optional[str]:
        """Refresh a JWT token."""
        payload = self.validate_token(token)
        if payload:
            # Generate new token with same claims but updated expiration
            user_id = payload['user_id']
            roles = {Role(role) for role in payload['roles']}
            return self.generate_token(user_id, roles)
        return None


class AuthenticationManager:
    """Main authentication and authorization manager."""
    
    def __init__(self):
        self.users: Dict[str, User] = {}
        self.sessions: Dict[str, Session] = {}
        self.rbac = RoleBasedAccessControl()
        self.password_manager = PasswordManager()
        self.mfa = MultiFactorAuth()
        self.api_key_manager = APIKeyManager()
        self.jwt_manager = JWTManager()
        self._load_users()
    
    def create_user(self, username: str, email: str, password: str, 
                   roles: Set[Role]) -> Tuple[bool, str]:
        """Create a new user."""
        # Validate password
        is_valid, errors = self.password_manager.validate_password_strength(password)
        if not is_valid:
            return False, "; ".join(errors)
        
        # Check if user exists
        if any(user.username == username or user.email == email for user in self.users.values()):
            return False, "User already exists"
        
        # Create user
        user_id = secrets.token_hex(16)
        password_hash = self.password_manager.hash_password(password)
        
        user = User(
            user_id=user_id,
            username=username,
            email=email,
            password_hash=password_hash,
            roles=roles
        )
        
        self.users[user_id] = user
        self._save_users()
        
        audit_logger.log_security_event(
            event_type="user_created",
            user_id=user_id,
            severity="INFO",
            details={
                "username": username,
                "email": email,
                "roles": [role.value for role in roles]
            }
        )
        
        return True, user_id
    
    def authenticate_user(self, username: str, password: str, 
                         mfa_token: Optional[str] = None) -> Optional[User]:
        """Authenticate a user with username/password and optional MFA."""
        user = self._find_user_by_username(username)
        if not user or not user.is_active:
            audit_logger.log_security_event(
                event_type="authentication_failed",
                user_id=user.user_id if user else None,
                severity="WARNING",
                details={"username": username, "reason": "user_not_found_or_inactive"}
            )
            return None
        
        # Check password
        if not self.password_manager.verify_password(password, user.password_hash):
            user.failed_login_attempts += 1
            self._save_users()
            
            audit_logger.log_security_event(
                event_type="authentication_failed",
                user_id=user.user_id,
                severity="WARNING",
                details={"username": username, "reason": "invalid_password"}
            )
            return None
        
        # Check MFA if enabled
        if user.mfa_enabled:
            if not mfa_token or not self.mfa.verify_token(user.mfa_secret, mfa_token):
                audit_logger.log_security_event(
                    event_type="authentication_failed",
                    user_id=user.user_id,
                    severity="WARNING",
                    details={"username": username, "reason": "invalid_mfa_token"}
                )
                return None
        
        # Reset failed attempts and update last login
        user.failed_login_attempts = 0
        user.last_login = datetime.now()
        self._save_users()
        
        audit_logger.log_security_event(
            event_type="authentication_successful",
            user_id=user.user_id,
            severity="INFO",
            details={"username": username}
        )
        
        return user
    
    def create_session(self, user: User, ip_address: str, user_agent: str) -> str:
        """Create a new session for a user."""
        session_id = secrets.token_hex(32)
        session = Session(
            session_id=session_id,
            user_id=user.user_id,
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=24),
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        self.sessions[session_id] = session
        return session_id
    
    def validate_session(self, session_id: str) -> Optional[User]:
        """Validate a session and return the user."""
        session = self.sessions.get(session_id)
        if not session or not session.is_active or session.expires_at < datetime.now():
            return None
        
        return self.users.get(session.user_id)
    
    def check_permission(self, user: User, permission: Permission) -> bool:
        """Check if a user has a specific permission."""
        return self.rbac.has_permission(user.roles, permission)
    
    def enable_mfa(self, user_id: str) -> Tuple[str, str]:
        """Enable MFA for a user."""
        user = self.users.get(user_id)
        if not user:
            raise ValueError("User not found")
        
        secret = self.mfa.generate_secret()
        qr_url = self.mfa.generate_qr_code_url(user.email, secret)
        
        user.mfa_secret = secret
        user.mfa_enabled = True
        self._save_users()
        
        audit_logger.log_security_event(
            event_type="mfa_enabled",
            user_id=user_id,
            severity="INFO",
            details={"username": user.username}
        )
        
        return secret, qr_url
    
    def _find_user_by_username(self, username: str) -> Optional[User]:
        """Find a user by username."""
        for user in self.users.values():
            if user.username == username:
                return user
        return None
    
    def _load_users(self):
        """Load users from storage."""
        config = get_config()
        users_file = Path(config.keys_path) / "users.json"
        
        if users_file.exists():
            try:
                with open(users_file, 'r') as f:
                    users_data = json.load(f)
                
                for user_id, user_info in users_data.items():
                    user = User(
                        user_id=user_info['user_id'],
                        username=user_info['username'],
                        email=user_info['email'],
                        password_hash=user_info['password_hash'],
                        roles={Role(role) for role in user_info['roles']},
                        is_active=user_info['is_active'],
                        created_at=datetime.fromisoformat(user_info['created_at']),
                        last_login=datetime.fromisoformat(user_info['last_login']) if user_info.get('last_login') else None,
                        failed_login_attempts=user_info['failed_login_attempts'],
                        mfa_enabled=user_info['mfa_enabled'],
                        mfa_secret=user_info.get('mfa_secret'),
                        api_keys=user_info.get('api_keys', [])
                    )
                    self.users[user_id] = user
                
                logger.info(f"Loaded {len(self.users)} users")
            except Exception as e:
                logger.error(f"Failed to load users: {e}")
    
    def _save_users(self):
        """Save users to storage."""
        config = get_config()
        keys_path = Path(config.keys_path)
        keys_path.mkdir(parents=True, exist_ok=True)
        users_file = keys_path / "users.json"
        
        users_data = {}
        for user_id, user in self.users.items():
            users_data[user_id] = {
                'user_id': user.user_id,
                'username': user.username,
                'email': user.email,
                'password_hash': user.password_hash,
                'roles': [role.value for role in user.roles],
                'is_active': user.is_active,
                'created_at': user.created_at.isoformat(),
                'last_login': user.last_login.isoformat() if user.last_login else None,
                'failed_login_attempts': user.failed_login_attempts,
                'mfa_enabled': user.mfa_enabled,
                'mfa_secret': user.mfa_secret,
                'api_keys': user.api_keys
            }
        
        with open(users_file, 'w') as f:
            json.dump(users_data, f, indent=2)