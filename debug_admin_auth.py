"""
AgroAI — debug_admin_auth.py
============================
Paste your JWT token when prompted.
This script will:
  1. Decode the JWT (without verifying signature) to show the raw payload
  2. Verify it with the REAL secret key from .env
  3. Look up the user in MongoDB
  4. Print the exact role value in the DB
  5. Explain why admin access is/isn't granted

Run with:
    python debug_admin_auth.py
"""

import os, sys, json, datetime
from dotenv import load_dotenv

load_dotenv()

# ── 1. Get token from user ────────────────────────────────────────────────────
print("\n" + "=" * 64)
print("  AgroAI Admin Auth Debugger")
print("=" * 64)
token = input("\nPaste your JWT token (from localStorage.agroai_jwt): ").strip()
if not token:
    print("[ERROR] No token provided.")
    sys.exit(1)

# ── 2. Decode without verification (inspect payload) ─────────────────────────
try:
    import base64, json as _json
    parts = token.split(".")
    if len(parts) != 3:
        print("[ERROR] Token doesn't look like a JWT (expected 3 parts separated by '.')")
        sys.exit(1)

    # Add padding for base64
    payload_b64 = parts[1] + "=" * (-len(parts[1]) % 4)
    payload_raw = base64.urlsafe_b64decode(payload_b64).decode("utf-8")
    payload = _json.loads(payload_raw)

    print("\n[STEP 1] Raw JWT Payload (decoded without signature verification):")
    print(json.dumps(payload, indent=2, default=str))

    sub = payload.get("sub")
    iat = payload.get("iat")
    exp = payload.get("exp")

    print(f"\n  sub (user_id) : {sub!r}")
    print(f"  iat (issued)  : {datetime.datetime.utcfromtimestamp(iat)} UTC" if iat else "  iat: missing")

    if exp:
        exp_dt = datetime.datetime.utcfromtimestamp(exp)
        now = datetime.datetime.utcnow()
        if now > exp_dt:
            print(f"  exp (expired) : {exp_dt} UTC  ← ⚠️  TOKEN IS EXPIRED! This is your 401 cause.")
        else:
            remaining = exp_dt - now
            print(f"  exp (valid)   : {exp_dt} UTC  (expires in {remaining})")
    else:
        print("  exp: missing")

except Exception as e:
    print(f"[ERROR] Failed to decode token: {e}")
    sys.exit(1)

# ── 3. Verify with real secret ────────────────────────────────────────────────
print("\n[STEP 2] Verifying token with JWT_SECRET_KEY from .env...")
jwt_secret = os.getenv("JWT_SECRET_KEY", "")
if not jwt_secret:
    print("  [ERROR] JWT_SECRET_KEY not found in .env")
    sys.exit(1)

print(f"  JWT_SECRET_KEY : {jwt_secret[:10]}...{jwt_secret[-4:]}  (length: {len(jwt_secret)})")

try:
    import jwt as pyjwt
    decoded = pyjwt.decode(token, jwt_secret, algorithms=["HS256"])
    print(f"  ✅ Signature VALID — token is genuine")
    print(f"  Verified sub: {decoded.get('sub')!r}")
except pyjwt.ExpiredSignatureError:
    print("  ❌ TOKEN EXPIRED — this is the 401 cause. Log out and log back in.")
    sys.exit(0)
except pyjwt.InvalidSignatureError:
    print("  ❌ INVALID SIGNATURE — token was signed with a DIFFERENT secret key.")
    print("     This is the 401 cause. The server restarted with a different JWT_SECRET_KEY.")
    print("     Fix: Log out and log back in to get a new token.")
    sys.exit(0)
except pyjwt.DecodeError as e:
    print(f"  ❌ DECODE ERROR: {e}")
    sys.exit(0)
except ImportError:
    print("  [WARN] PyJWT not installed. Skipping signature verification. Run: pip install PyJWT")

# ── 4. Look up user in MongoDB ────────────────────────────────────────────────
print("\n[STEP 3] Looking up user in MongoDB...")
MONGO_URI = os.getenv("MONGO_URI", "")
if not MONGO_URI:
    print("[ERROR] MONGO_URI not set in .env")
    sys.exit(1)

try:
    from pymongo import MongoClient
    from pymongo.server_api import ServerApi
    from bson import ObjectId

    client = MongoClient(MONGO_URI, server_api=ServerApi("1"), serverSelectionTimeoutMS=5000)
    client.admin.command("ping")
    db    = client["plant_disease_db"]
    users = db["users"]

    user_id = sub  # from JWT payload

    if not ObjectId.is_valid(user_id):
        print(f"  ❌ user_id {user_id!r} is NOT a valid ObjectId — get_user_by_id() returns None.")
        print("     This means the JWT identity is not a valid MongoDB ObjectId string.")
        sys.exit(0)

    user = users.find_one({"_id": ObjectId(user_id)}, {"password": 0})

    if not user:
        print(f"  ❌ NO USER FOUND in DB for _id={user_id}")
        print("     get_user_by_id() returns None → admin check fails → 403")
        sys.exit(0)

    print(f"  ✅ User found:")
    print(f"     _id   : {str(user['_id'])}")
    print(f"     name  : {user.get('name')}")
    print(f"     email : {user.get('email')}")
    role = user.get("role")
    print(f"     role  : {role!r}  ← this is what the admin check compares to 'admin'")

    print("\n[STEP 4] Admin check simulation:")
    print(f"  user.get('role') == 'admin'  →  {role!r} == 'admin'  →  {role == 'admin'}")

    if role == "admin":
        print("\n  ✅ Admin check PASSES — the decorator WOULD allow access.")
        print("     The 401 is coming from @jwt_required() itself, NOT from the role check.")
        print("     Most likely cause: EXPIRED TOKEN or WRONG SECRET KEY.")
        print("     Fix: Log out, log back in, then retry the admin API.")
    else:
        if role is None:
            print("\n  ❌ ROLE IS MISSING from the user document.")
            print("     Fix: Run `python create_admin.py <email>` to set role='admin'.")
        else:
            print(f"\n  ❌ ROLE IS '{role}' (not 'admin').")
            print("     Fix: Run `python create_admin.py <email>` to promote this user.")

    client.close()

except Exception as e:
    print(f"  [ERROR] MongoDB lookup failed: {e}")

print("\n" + "=" * 64 + "\n")
