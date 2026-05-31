"""
AgroAI — Authentication Routes Blueprint
==========================================
Provides:
  POST   /api/auth/register  — Create new user account
  POST   /api/auth/login     — Authenticate and receive JWT
  POST   /api/auth/logout    — Stateless logout (client clears token)
  GET    /api/auth/profile   — Get current user profile (protected)
  PUT    /api/auth/profile   — Update user name (protected)
"""

import logging
import datetime
from flask import Blueprint, request, jsonify
from flask_jwt_extended import (
    create_access_token,
    jwt_required,
    get_jwt_identity,
)
import database
from auth.auth_utils import (
    validate_email,
    validate_password,
    validate_name,
    hash_password,
    verify_password,
)

logger = logging.getLogger(__name__)

auth_bp = Blueprint("auth", __name__, url_prefix="/api/auth")


# ---------------------------------------------------------------------------
# POST /api/auth/register
# ---------------------------------------------------------------------------
@auth_bp.route("/register", methods=["POST"])
def register():
    """
    Register a new user account.
    Body: { "name": str, "email": str, "password": str }
    """
    data = request.get_json(silent=True) or {}

    name     = (data.get("name")     or "").strip()
    email    = (data.get("email")    or "").strip().lower()
    password = (data.get("password") or "").strip()

    # ── Validate inputs ────────────────────────────────────────────────────
    ok, err = validate_name(name)
    if not ok:
        return jsonify({"success": False, "message": err}), 400

    ok, err = validate_email(email)
    if not ok:
        return jsonify({"success": False, "message": err}), 400

    ok, err = validate_password(password)
    if not ok:
        return jsonify({"success": False, "message": err}), 400

    # ── Check for duplicate email ─────────────────────────────────────────
    if not database.is_connected:
        return jsonify({"success": False, "message": "Database unavailable. Please try again later."}), 503

    existing = database.get_user_by_email(email)
    if existing:
        return jsonify({"success": False, "message": "An account with this email already exists."}), 409

    # ── Create user ───────────────────────────────────────────────────────
    hashed_pw = hash_password(password)
    user_id = database.create_user({
        "name":      name,
        "email":     email,
        "password":  hashed_pw,
        "role":      "user",
        "createdAt": datetime.datetime.utcnow(),
        "lastLogin": datetime.datetime.utcnow(),
    })

    if not user_id:
        return jsonify({"success": False, "message": "Failed to create account. Please try again."}), 500

    logger.info(f"[AUTH] New user registered: {email}")
    return jsonify({"success": True, "message": "Account created successfully! Please login."}), 201


# ---------------------------------------------------------------------------
# POST /api/auth/login
# ---------------------------------------------------------------------------
@auth_bp.route("/login", methods=["POST"])
def login():
    """
    Authenticate a user and return a JWT access token.
    Body: { "email": str, "password": str }
    """
    data = request.get_json(silent=True) or {}

    email    = (data.get("email")    or "").strip().lower()
    password = (data.get("password") or "").strip()

    if not email or not password:
        return jsonify({"success": False, "message": "Email and password are required."}), 400

    if not database.is_connected:
        return jsonify({"success": False, "message": "Database unavailable. Please try again later."}), 503

    # ── Look up user ───────────────────────────────────────────────────────
    user = database.get_user_by_email(email)
    if not user:
        # Use a vague message to prevent email enumeration
        return jsonify({"success": False, "message": "Invalid email or password."}), 401

    if not verify_password(password, user["password"]):
        return jsonify({"success": False, "message": "Invalid email or password."}), 401

    # ── Update lastLogin ───────────────────────────────────────────────────
    database.update_user(str(user["_id"]), {"lastLogin": datetime.datetime.utcnow()})

    # ── Issue JWT ──────────────────────────────────────────────────────────
    user_id_str = str(user["_id"])
    access_token = create_access_token(identity=user_id_str)

    logger.info(f"[AUTH] User logged in: {email}")
    return jsonify({
        "success": True,
        "token": access_token,
        "user": {
            "id":    user_id_str,
            "name":  user["name"],
            "email": user["email"],
            "role":  user.get("role", "user"),
        }
    }), 200


# ---------------------------------------------------------------------------
# POST /api/auth/logout
# ---------------------------------------------------------------------------
@auth_bp.route("/logout", methods=["POST"])
def logout():
    """
    Stateless logout — the client must delete the stored token.
    Returns 200 OK so the frontend knows it can safely clear storage.
    """
    return jsonify({"success": True, "message": "Logged out successfully."}), 200


# ---------------------------------------------------------------------------
# GET /api/auth/profile
# ---------------------------------------------------------------------------
@auth_bp.route("/profile", methods=["GET"])
@jwt_required()
def get_profile():
    """
    Return the current authenticated user's profile.
    Requires: Authorization: Bearer <token>
    """
    user_id = get_jwt_identity()

    if not database.is_connected:
        return jsonify({"success": False, "message": "Database unavailable."}), 503

    user = database.get_user_by_id(user_id)
    if not user:
        return jsonify({"success": False, "message": "User not found."}), 404

    total_scans = database.count_user_scans(user_id)

    created_at = user.get("createdAt")
    last_login = user.get("lastLogin")

    return jsonify({
        "success": True,
        "user": {
            "id":          user_id,
            "name":        user["name"],
            "email":       user["email"],
            "role":        user.get("role", "user"),
            "createdAt":   created_at.isoformat() + "Z" if created_at else None,
            "lastLogin":   last_login.isoformat() + "Z" if last_login else None,
            "totalScans":  total_scans,
        }
    }), 200


# ---------------------------------------------------------------------------
# PUT /api/auth/profile
# ---------------------------------------------------------------------------
@auth_bp.route("/profile", methods=["PUT"])
@jwt_required()
def update_profile():
    """
    Update allowed user profile fields (name only in Phase 1A).
    Requires: Authorization: Bearer <token>
    Body: { "name": str }
    """
    user_id = get_jwt_identity()
    data = request.get_json(silent=True) or {}

    updates = {}

    # Name update
    new_name = (data.get("name") or "").strip()
    if new_name:
        ok, err = validate_name(new_name)
        if not ok:
            return jsonify({"success": False, "message": err}), 400
        updates["name"] = new_name

    if not updates:
        return jsonify({"success": False, "message": "No valid fields to update."}), 400

    if not database.is_connected:
        return jsonify({"success": False, "message": "Database unavailable."}), 503

    success = database.update_user(user_id, updates)
    if not success:
        return jsonify({"success": False, "message": "Failed to update profile."}), 500

    logger.info(f"[AUTH] Profile updated for user_id: {user_id}")
    return jsonify({"success": True, "message": "Profile updated successfully."}), 200
