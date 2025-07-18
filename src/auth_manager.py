from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
import json
import os
import secrets
"""Simple authentication manager using PBKDF2 hashing."""


# [Patch v6.9.47] Login system module
HASH_ITERATIONS = 100_000
SALT_BYTES = 16
HASH_NAME = "sha256"
SESSION_TIMEOUT = timedelta(hours = 1)


@dataclass
class Session:
    """Dataclass representing a user session."""

    username: str
    token: str
    expires_at: datetime


class AuthManager:
    """Manage user registration, authentication and sessions."""

    def __init__(self, user_file: str = "users.json") -> None:
        self.user_file = user_file
        self.sessions: dict[str, Session] = {}
        self._load_users()

    def _load_users(self) -> None:
        if os.path.exists(self.user_file):
            with open(self.user_file, "r", encoding = "utf - 8") as f:
                self.users = json.load(f)
        else:
            self.users = {}

    def _save_users(self) -> None:
        with open(self.user_file, "w", encoding = "utf - 8") as f:
            json.dump(self.users, f)

    def _hash_password(self, password: str, salt: bytes) -> bytes:
        return hashlib.pbkdf2_hmac(
            HASH_NAME, password.encode("utf - 8"), salt, HASH_ITERATIONS
        )

    def register(self, username: str, password: str) -> None:
        """Register a new user with hashed password."""
        if username in self.users:
            raise ValueError("username already exists")
        salt = secrets.token_bytes(SALT_BYTES)
        pwd_hash = self._hash_password(password, salt)
        self.users[username] = {"salt": salt.hex(), "hash": pwd_hash.hex()}
        self._save_users()

    def authenticate(self, username: str, password: str) -> Session:
        """Validate credentials and create a session."""
        user = self.users.get(username)
        if not user:
            raise ValueError("invalid username or password")
        salt = bytes.fromhex(user["salt"])
        stored_hash = bytes.fromhex(user["hash"])
        if secrets.compare_digest(self._hash_password(password, salt), stored_hash):
            token = secrets.token_urlsafe()
            session = Session(
                username = username, 
                token = token, 
                expires_at = datetime.utcnow() + SESSION_TIMEOUT, 
            )
            self.sessions[token] = session
            return session
        raise ValueError("invalid username or password")

    def validate_session(self, token: str) -> bool:
        """Check if a session token is valid and not expired."""
        session = self.sessions.get(token)
        if not session:
            return False
        if datetime.utcnow() > session.expires_at:
            self.sessions.pop(token, None)
            return False
        return True

    def logout(self, token: str) -> None:
        """Remove a session token."""
        self.sessions.pop(token, None)