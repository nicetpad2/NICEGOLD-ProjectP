#!/usr/bin/env python3
"""
Single User Authentication System
ระบบ Authentication สำหรับผู้ใช้คนเดียว Production-Ready
"""

import hashlib
import json
import logging
import os
import secrets
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

import jwt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class UserSession:
    """User session data structure"""
    username: str
    token: str
    created_at: datetime
    expires_at: datetime
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "username": self.username,
            "token": self.token,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "ip_address": self.ip_address,
            "user_agent": self.user_agent
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserSession":
        """Create from dictionary"""
        return cls(
            username=data["username"],
            token=data["token"],
            created_at=datetime.fromisoformat(data["created_at"]),
            expires_at=datetime.fromisoformat(data["expires_at"]),
            ip_address=data.get("ip_address"),
            user_agent=data.get("user_agent")
        )

class SingleUserAuth:
    """
    Production-ready single user authentication system
    Features:
    - Secure password hashing with PBKDF2
    - JWT token-based sessions
    - Session management and validation
    - Security logging and monitoring
    - Configurable session timeout
    """
    
    def __init__(self, 
                 config_dir: str = "config/auth",
                 session_timeout_hours: int = 24,
                 max_login_attempts: int = 5,
                 lockout_duration_minutes: int = 30):
        """
        Initialize single user authentication system
        
        Args:
            config_dir: Directory to store auth configuration
            session_timeout_hours: Session timeout in hours
            max_login_attempts: Maximum failed login attempts before lockout
            lockout_duration_minutes: Lockout duration in minutes
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        self.user_file = self.config_dir / "user.json"
        self.sessions_file = self.config_dir / "sessions.json"
        self.security_log = self.config_dir / "security.log"
        
        # Security settings
        self.session_timeout = timedelta(hours=session_timeout_hours)
        self.max_login_attempts = max_login_attempts
        self.lockout_duration = timedelta(minutes=lockout_duration_minutes)
        
        # JWT settings
        self.jwt_secret = self._get_or_create_jwt_secret()
        self.jwt_algorithm = "HS256"
        
        # Initialize data
        self.user_data = self._load_user_data()
        self.sessions = self._load_sessions()
        self.login_attempts = {}
        
        # Security logging
        self._setup_security_logging()
        
        logger.info("✅ Single User Authentication system initialized")
    
    def _setup_security_logging(self):
        """Setup security event logging"""
        self.security_logger = logging.getLogger("security")
        handler = logging.FileHandler(self.security_log)
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.security_logger.addHandler(handler)
        self.security_logger.setLevel(logging.INFO)
    
    def _get_or_create_jwt_secret(self) -> str:
        """Get or create JWT secret key"""
        secret_file = self.config_dir / "jwt_secret.key"
        
        if secret_file.exists():
            with open(secret_file, 'r') as f:
                return f.read().strip()
        else:
            # Generate new secret
            secret = secrets.token_urlsafe(64)
            with open(secret_file, 'w') as f:
                f.write(secret)
            # Secure file permissions
            os.chmod(secret_file, 0o600)
            return secret
    
    def _load_user_data(self) -> Dict[str, Any]:
        """Load user data from file"""
        if self.user_file.exists():
            try:
                with open(self.user_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading user data: {e}")
                return {}
        return {}
    
    def _save_user_data(self):
        """Save user data to file"""
        try:
            with open(self.user_file, 'w') as f:
                json.dump(self.user_data, f, indent=2)
            # Secure file permissions
            os.chmod(self.user_file, 0o600)
        except Exception as e:
            logger.error(f"Error saving user data: {e}")
    
    def _load_sessions(self) -> Dict[str, UserSession]:
        """Load active sessions from file"""
        if self.sessions_file.exists():
            try:
                with open(self.sessions_file, 'r') as f:
                    sessions_data = json.load(f)
                    sessions = {}
                    for token, data in sessions_data.items():
                        try:
                            sessions[token] = UserSession.from_dict(data)
                        except Exception as e:
                            logger.warning(f"Invalid session data for token {token}: {e}")
                    return sessions
            except Exception as e:
                logger.error(f"Error loading sessions: {e}")
        return {}
    
    def _save_sessions(self):
        """Save active sessions to file"""
        try:
            sessions_data = {}
            for token, session in self.sessions.items():
                sessions_data[token] = session.to_dict()
            
            with open(self.sessions_file, 'w') as f:
                json.dump(sessions_data, f, indent=2)
            # Secure file permissions
            os.chmod(self.sessions_file, 0o600)
        except Exception as e:
            logger.error(f"Error saving sessions: {e}")
    
    def _hash_password(self, password: str, salt: bytes) -> str:
        """Hash password using PBKDF2"""
        return hashlib.pbkdf2_hmac(
            'sha256', 
            password.encode('utf-8'), 
            salt, 
            100000  # iterations
        ).hex()
    
    def _is_locked_out(self, ip_address: str) -> bool:
        """Check if IP is locked out due to failed attempts"""
        if ip_address not in self.login_attempts:
            return False
        
        attempts_data = self.login_attempts[ip_address]
        if attempts_data['count'] >= self.max_login_attempts:
            lockout_time = datetime.fromisoformat(attempts_data['last_attempt'])
            if datetime.now() - lockout_time < self.lockout_duration:
                return True
            else:
                # Reset attempts after lockout period
                del self.login_attempts[ip_address]
        
        return False
    
    def _record_login_attempt(self, ip_address: str, success: bool):
        """Record login attempt for rate limiting"""
        if success:
            # Clear failed attempts on successful login
            if ip_address in self.login_attempts:
                del self.login_attempts[ip_address]
        else:
            # Record failed attempt
            if ip_address not in self.login_attempts:
                self.login_attempts[ip_address] = {'count': 0, 'last_attempt': None}
            
            self.login_attempts[ip_address]['count'] += 1
            self.login_attempts[ip_address]['last_attempt'] = datetime.now().isoformat()
    
    def setup_user(self, username: str, password: str) -> bool:
        """
        Setup the single user account
        
        Args:
            username: Username for the account
            password: Password for the account
            
        Returns:
            bool: True if user setup successful
        """
        try:
            if self.user_data:
                logger.warning("⚠️ User already exists. Use change_password() to update.")
                return False
            
            # Generate salt and hash password
            salt = secrets.token_bytes(32)
            password_hash = self._hash_password(password, salt)
            
            # Store user data
            self.user_data = {
                "username": username,
                "password_hash": password_hash,
                "salt": salt.hex(),
                "created_at": datetime.now().isoformat(),
                "last_login": None,
                "login_count": 0
            }
            
            self._save_user_data()
            
            # Log security event
            self.security_logger.info(f"User account created: {username}")
            logger.info(f"✅ User '{username}' created successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error setting up user: {e}")
            return False
    
    def authenticate(self, username: str, password: str, 
                    ip_address: str = "unknown", 
                    user_agent: str = "unknown") -> Optional[str]:
        """
        Authenticate user and create session
        
        Args:
            username: Username
            password: Password
            ip_address: Client IP address
            user_agent: Client user agent
            
        Returns:
            str: JWT token if authentication successful, None otherwise
        """
        try:
            # Check if IP is locked out
            if self._is_locked_out(ip_address):
                self.security_logger.warning(
                    f"Login attempt from locked out IP: {ip_address}"
                )
                return None
            
            # Validate user exists
            if not self.user_data:
                self.security_logger.warning(
                    f"Login attempt with no user configured: {username} from {ip_address}"
                )
                self._record_login_attempt(ip_address, False)
                return None
            
            # Validate username
            if username != self.user_data["username"]:
                self.security_logger.warning(
                    f"Invalid username attempt: {username} from {ip_address}"
                )
                self._record_login_attempt(ip_address, False)
                return None
            
            # Validate password
            salt = bytes.fromhex(self.user_data["salt"])
            password_hash = self._hash_password(password, salt)
            
            if password_hash != self.user_data["password_hash"]:
                self.security_logger.warning(
                    f"Invalid password attempt for {username} from {ip_address}"
                )
                self._record_login_attempt(ip_address, False)
                return None
            
            # Authentication successful
            self._record_login_attempt(ip_address, True)
            
            # Clean up expired sessions
            self._cleanup_expired_sessions()
            
            # Create new session
            token = jwt.encode({
                "username": username,
                "iat": datetime.utcnow(),
                "exp": datetime.utcnow() + self.session_timeout,
                "ip": ip_address
            }, self.jwt_secret, algorithm=self.jwt_algorithm)
            
            session = UserSession(
                username=username,
                token=token,
                created_at=datetime.now(),
                expires_at=datetime.now() + self.session_timeout,
                ip_address=ip_address,
                user_agent=user_agent
            )
            
            self.sessions[token] = session
            self._save_sessions()
            
            # Update user login stats
            self.user_data["last_login"] = datetime.now().isoformat()
            self.user_data["login_count"] = self.user_data.get("login_count", 0) + 1
            self._save_user_data()
            
            # Log successful login
            self.security_logger.info(
                f"Successful login: {username} from {ip_address}"
            )
            logger.info(f"✅ User '{username}' authenticated successfully")
            
            return token
            
        except Exception as e:
            logger.error(f"❌ Authentication error: {e}")
            self.security_logger.error(f"Authentication error: {e}")
            return None
    
    def validate_token(self, token: str, ip_address: str = None) -> bool:
        """
        Validate JWT token and session
        
        Args:
            token: JWT token to validate
            ip_address: Client IP address for additional security
            
        Returns:
            bool: True if token is valid
        """
        try:
            # Check if session exists
            if token not in self.sessions:
                return False
            
            session = self.sessions[token]
            
            # Check if session expired
            if datetime.now() > session.expires_at:
                self._remove_session(token)
                return False
            
            # Validate JWT token
            try:
                payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])
                
                # Additional IP validation if provided
                if ip_address and session.ip_address != ip_address:
                    self.security_logger.warning(
                        f"Token used from different IP: {session.ip_address} vs {ip_address}"
                    )
                    # Optionally strict: return False
                
                return True
                
            except jwt.ExpiredSignatureError:
                self._remove_session(token)
                return False
            except jwt.InvalidTokenError:
                self._remove_session(token)
                return False
            
        except Exception as e:
            logger.error(f"Token validation error: {e}")
            return False
    
    def get_session_info(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Get session information
        
        Args:
            token: JWT token
            
        Returns:
            dict: Session information or None
        """
        if token in self.sessions and self.validate_token(token):
            session = self.sessions[token]
            return {
                "username": session.username,
                "created_at": session.created_at.isoformat(),
                "expires_at": session.expires_at.isoformat(),
                "ip_address": session.ip_address,
                "user_agent": session.user_agent
            }
        return None
    
    def logout(self, token: str) -> bool:
        """
        Logout user by removing session
        
        Args:
            token: JWT token to logout
            
        Returns:
            bool: True if logout successful
        """
        try:
            if token in self.sessions:
                session = self.sessions[token]
                self.security_logger.info(
                    f"User logout: {session.username} from {session.ip_address}"
                )
                self._remove_session(token)
                logger.info(f"✅ User logged out successfully")
                return True
            return False
        except Exception as e:
            logger.error(f"Logout error: {e}")
            return False
    
    def logout_all_sessions(self) -> int:
        """
        Logout all active sessions
        
        Returns:
            int: Number of sessions logged out
        """
        try:
            count = len(self.sessions)
            self.sessions.clear()
            self._save_sessions()
            
            self.security_logger.info(f"All sessions logged out: {count} sessions")
            logger.info(f"✅ All sessions logged out: {count} sessions")
            
            return count
        except Exception as e:
            logger.error(f"Logout all error: {e}")
            return 0
    
    def change_password(self, current_password: str, new_password: str) -> bool:
        """
        Change user password
        
        Args:
            current_password: Current password for verification
            new_password: New password to set
            
        Returns:
            bool: True if password changed successfully
        """
        try:
            if not self.user_data:
                logger.error("No user account exists")
                return False
            
            # Verify current password
            salt = bytes.fromhex(self.user_data["salt"])
            current_hash = self._hash_password(current_password, salt)
            
            if current_hash != self.user_data["password_hash"]:
                self.security_logger.warning("Password change failed: invalid current password")
                return False
            
            # Generate new salt and hash for new password
            new_salt = secrets.token_bytes(32)
            new_password_hash = self._hash_password(new_password, new_salt)
            
            # Update user data
            self.user_data["password_hash"] = new_password_hash
            self.user_data["salt"] = new_salt.hex()
            self.user_data["password_changed_at"] = datetime.now().isoformat()
            
            self._save_user_data()
            
            # Logout all existing sessions for security
            self.logout_all_sessions()
            
            self.security_logger.info(f"Password changed for user: {self.user_data['username']}")
            logger.info("✅ Password changed successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"Password change error: {e}")
            return False
    
    def _remove_session(self, token: str):
        """Remove session from active sessions"""
        if token in self.sessions:
            del self.sessions[token]
            self._save_sessions()
    
    def _cleanup_expired_sessions(self):
        """Remove expired sessions"""
        now = datetime.now()
        expired_tokens = []
        
        for token, session in self.sessions.items():
            if now > session.expires_at:
                expired_tokens.append(token)
        
        for token in expired_tokens:
            del self.sessions[token]
        
        if expired_tokens:
            self._save_sessions()
            logger.info(f"Cleaned up {len(expired_tokens)} expired sessions")
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get authentication system status
        
        Returns:
            dict: System status information
        """
        self._cleanup_expired_sessions()
        
        return {
            "user_configured": bool(self.user_data),
            "username": self.user_data.get("username", "Not configured"),
            "active_sessions": len(self.sessions),
            "last_login": self.user_data.get("last_login"),
            "login_count": self.user_data.get("login_count", 0),
            "session_timeout_hours": self.session_timeout.total_seconds() / 3600,
            "max_login_attempts": self.max_login_attempts,
            "lockout_duration_minutes": self.lockout_duration.total_seconds() / 60,
            "failed_attempt_ips": len(self.login_attempts)
        }

# Global instance for single user authentication
auth_manager = SingleUserAuth()

def require_auth(func):
    """
    Decorator to require authentication for functions
    """
    def wrapper(*args, **kwargs):
        token = kwargs.get('auth_token')
        if not token or not auth_manager.validate_token(token):
            raise PermissionError("Authentication required")
        return func(*args, **kwargs)
    return wrapper

# CLI functions for management
def create_admin_user():
    """Interactive setup for admin user"""
    print("🔐 Setting up Single User Authentication")
    print("=" * 50)
    
    status = auth_manager.get_system_status()
    if status["user_configured"]:
        print(f"❌ User already configured: {status['username']}")
        print("Use change_password() to update password")
        return False
    
    username = input("Enter username: ").strip()
    if not username:
        print("❌ Username cannot be empty")
        return False
    
    import getpass
    password = getpass.getpass("Enter password: ")
    password_confirm = getpass.getpass("Confirm password: ")
    
    if password != password_confirm:
        print("❌ Passwords do not match")
        return False
    
    if len(password) < 8:
        print("❌ Password must be at least 8 characters")
        return False
    
    success = auth_manager.setup_user(username, password)
    if success:
        print("✅ Admin user created successfully!")
        print(f"Username: {username}")
        print("You can now login to the system")
    else:
        print("❌ Failed to create user")
    
    return success

def interactive_login():
    """Interactive login for testing"""
    print("🔑 User Login")
    print("=" * 30)
    
    username = input("Username: ").strip()
    import getpass
    password = getpass.getpass("Password: ")
    
    token = auth_manager.authenticate(username, password, "127.0.0.1", "CLI-Test")
    
    if token:
        print("✅ Login successful!")
        print(f"Token: {token[:20]}...")
        
        # Show session info
        session_info = auth_manager.get_session_info(token)
        if session_info:
            print(f"Session expires: {session_info['expires_at']}")
        
        return token
    else:
        print("❌ Login failed")
        return None

def show_system_status():
    """Show authentication system status"""
    print("📊 Authentication System Status")
    print("=" * 40)
    
    status = auth_manager.get_system_status()
    
    for key, value in status.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python single_user_auth.py setup    - Setup admin user")
        print("  python single_user_auth.py login    - Test login")
        print("  python single_user_auth.py status   - Show system status")
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    if command == "setup":
        create_admin_user()
    elif command == "login":
        interactive_login()
    elif command == "status":
        show_system_status()
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
