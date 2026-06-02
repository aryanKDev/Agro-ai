"""
AgroAI — create_admin.py
========================
CLI utility to promote an existing user to admin role.

Usage:
    python create_admin.py <email>

Example:
    python create_admin.py aryan@example.com
"""

import sys
import os
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.environ.get("MONGO_URI", "")
if not MONGO_URI:
    print("[ERROR] MONGO_URI not set in .env")
    sys.exit(1)

try:
    from pymongo import MongoClient
    from pymongo.server_api import ServerApi
except ImportError:
    print("[ERROR] pymongo not installed. Run: pip install pymongo")
    sys.exit(1)


def main():
    if len(sys.argv) < 2:
        print("Usage: python create_admin.py <email>")
        sys.exit(1)

    email = sys.argv[1].strip().lower()

    print(f"[Admin] Connecting to MongoDB...")
    client = MongoClient(MONGO_URI, server_api=ServerApi("1"))
    db     = client["plant_disease_db"]
    users  = db["users"]

    user = users.find_one({"email": email})
    if not user:
        print(f"[ERROR] No user found with email: {email}")
        sys.exit(1)

    current_role = user.get("role", "user")
    if current_role == "admin":
        print(f"[INFO] User '{user['name']}' ({email}) is already an admin.")
        sys.exit(0)

    result = users.update_one(
        {"email": email},
        {"$set": {"role": "admin"}}
    )

    if result.modified_count == 1:
        print(f"[SUCCESS] User '{user['name']}' ({email}) has been promoted to ADMIN.")
        print("[INFO] They can now log in and access the Admin Dashboard.")
    else:
        print("[ERROR] Update failed. No documents were modified.")
        sys.exit(1)


if __name__ == "__main__":
    main()
