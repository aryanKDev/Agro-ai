"""
AgroAI — Authentication Utilities
===================================
bcrypt password hashing and input validation helpers.
"""

import re
import bcrypt
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Input Validators
# ---------------------------------------------------------------------------

EMAIL_REGEX = re.compile(r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$')

def validate_email(email: str) -> tuple[bool, str]:
    """Return (True, '') if email is valid, else (False, error_message)."""
    if not email or not isinstance(email, str):
        return False, "Email is required."
    email = email.strip().lower()
    if not EMAIL_REGEX.match(email):
        return False, "Please enter a valid email address."
    if len(email) > 254:
        return False, "Email address is too long."
    return True, ""


def validate_password(password: str) -> tuple[bool, str]:
    """Return (True, '') if password meets requirements, else (False, error_message)."""
    if not password or not isinstance(password, str):
        return False, "Password is required."
    if len(password) < 8:
        return False, "Password must be at least 8 characters long."
    if len(password) > 128:
        return False, "Password is too long (max 128 characters)."
    return True, ""


def validate_name(name: str) -> tuple[bool, str]:
    """Return (True, '') if name is valid, else (False, error_message)."""
    if not name or not isinstance(name, str):
        return False, "Name is required."
    name = name.strip()
    if len(name) < 2:
        return False, "Name must be at least 2 characters."
    if len(name) > 100:
        return False, "Name is too long (max 100 characters)."
    return True, ""


# ---------------------------------------------------------------------------
# Password Hashing
# ---------------------------------------------------------------------------

def hash_password(plain_password: str) -> str:
    """Hash a plain-text password using bcrypt. Returns the hashed string."""
    salt = bcrypt.gensalt(rounds=12)
    hashed = bcrypt.hashpw(plain_password.encode("utf-8"), salt)
    return hashed.decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plain-text password against a bcrypt hash. Returns True if match."""
    try:
        return bcrypt.checkpw(
            plain_password.encode("utf-8"),
            hashed_password.encode("utf-8")
        )
    except Exception as e:
        logger.error(f"Password verification error: {e}")
        return False
