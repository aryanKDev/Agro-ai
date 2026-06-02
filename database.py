"""
AgroAI — MongoDB Database Connection Manager
=============================================
Handles connection to MongoDB Atlas and provides CRUD operations for:
  - scans       collection (user-scoped + legacy support)
  - users       collection (SaaS multi-user)
  - feedbacks   collection (Phase 2B — real feedback system)
"""

import os
import logging
import datetime
from bson import ObjectId
from pymongo import MongoClient, ASCENDING
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from dotenv import load_dotenv

# Set up logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

MONGO_URI             = os.getenv("MONGO_URI")
DB_NAME               = "agroai"
COLLECTION_NAME       = "scans"
USERS_COLLECTION      = "users"
FEEDBACKS_COLLECTION  = "feedbacks"
CHAT_HISTORY_COLLECTION = "chat_history"  # Phase 3A RAG

db_client        = None
db               = None
scans_col        = None
users_col        = None
feedbacks_col    = None
chat_history_col = None  # Phase 3A RAG
is_connected     = False


def init_db():
    """Initialize MongoDB Atlas Connection."""
    global db_client, db, scans_col, users_col, feedbacks_col, chat_history_col, is_connected

    if not MONGO_URI:
        logger.warning("MONGO_URI not defined in .env. MongoDB features will run in offline simulation mode.")
        is_connected = False
        return False

    try:
        db_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        db_client.admin.command('ping')

        db               = db_client[DB_NAME]
        scans_col        = db[COLLECTION_NAME]
        users_col        = db[USERS_COLLECTION]
        feedbacks_col    = db[FEEDBACKS_COLLECTION]
        chat_history_col = db[CHAT_HISTORY_COLLECTION]  # Phase 3A RAG

        # Ensure unique email index on users collection
        users_col.create_index([("email", ASCENDING)], unique=True)
        # Index feedbacks by userId for fast per-user queries
        feedbacks_col.create_index([("userId", ASCENDING)])
        # Index chat_history by userId + timestamp (Phase 3A RAG)
        chat_history_col.create_index([("userId", ASCENDING), ("timestamp", -1)])

        is_connected = True
        logger.info("Successfully connected to MongoDB Atlas!")
        return True

    except (ConnectionFailure, ServerSelectionTimeoutError) as e:
        logger.error(f"Failed to connect to MongoDB Atlas: {e}")
        is_connected = False
        db_client = db = scans_col = users_col = feedbacks_col = chat_history_col = None
        return False
    except Exception as e:
        logger.error(f"Unexpected database initialization error: {e}")
        is_connected = False
        return False


# Initialize at startup
init_db()


def check_connection():
    """Ensure database connection is still active."""
    global is_connected
    if not is_connected or db_client is None:
        return init_db()
    try:
        db_client.admin.command('ping')
        return True
    except Exception:
        is_connected = False
        return False


# ===========================================================================
# FEEDBACKS COLLECTION  (Phase 2B)
# ===========================================================================

def save_feedback(user_id: str, name: str, email: str, rating: int, message: str) -> str | None:
    """
    Save a user feedback document to the feedbacks collection.
    Returns the inserted _id string, or None on failure.
    """
    if not check_connection():
        logger.warning("Database offline. Cannot save feedback.")
        return None
    try:
        doc = {
            "name":      name,
            "email":     email,
            "rating":    int(rating),
            "message":   message,
            "createdAt": datetime.datetime.utcnow(),
        }
        if user_id and ObjectId.is_valid(user_id):
            doc["userId"] = ObjectId(user_id)
        result = feedbacks_col.insert_one(doc)
        logger.info(f"Feedback saved | ID: {result.inserted_id} | user: {user_id} | rating: {rating}")
        return str(result.inserted_id)
    except Exception as e:
        logger.error(f"Failed to save feedback: {e}")
        return None


def get_my_feedbacks(user_id: str) -> list:
    """
    Fetch all feedback submitted by a specific user, newest first.
    Returns a list of feedback dicts.
    """
    if not check_connection() or not ObjectId.is_valid(user_id):
        return []
    try:
        cursor = feedbacks_col.find(
            {"userId": ObjectId(user_id)}
        ).sort("createdAt", -1).limit(50)
        results = []
        for doc in cursor:
            results.append({
                "id":        str(doc["_id"]),
                "name":      doc.get("name", "Anonymous"),
                "email":     doc.get("email", ""),
                "rating":    doc.get("rating", 0),
                "message":   doc.get("message", ""),
                "createdAt": doc["createdAt"].isoformat() + "Z",
            })
        return results
    except Exception as e:
        logger.error(f"Failed to fetch user feedbacks: {e}")
        return []


def get_feedback_stats() -> dict:
    """
    Compute global aggregate feedback statistics.
    Returns: { total, avg_rating, distribution: {1..5: count} }
    """
    empty = {"total": 0, "avg_rating": None, "distribution": {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}}
    if not check_connection():
        return empty
    try:
        docs = list(feedbacks_col.find({}, {"rating": 1}))
        if not docs:
            return empty
        total = len(docs)
        ratings = [d.get("rating", 0) for d in docs]
        avg = round(sum(ratings) / total, 1)
        dist = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        for r in ratings:
            if r in dist:
                dist[r] += 1
        return {"total": total, "avg_rating": avg, "distribution": dist}
    except Exception as e:
        logger.error(f"Failed to compute feedback stats: {e}")
        return empty


# ===========================================================================
# USERS COLLECTION
# ===========================================================================

def create_user(user_doc: dict) -> str | None:
    """
    Insert a new user document into the users collection.
    Returns the inserted _id as a string, or None on failure.
    """
    if not check_connection():
        return None
    try:
        result = users_col.insert_one(user_doc)
        logger.info(f"User created with ID: {result.inserted_id}")
        return str(result.inserted_id)
    except Exception as e:
        logger.error(f"Failed to create user: {e}")
        return None


def get_user_by_email(email: str) -> dict | None:
    """Find a user document by email (case-insensitive). Returns full doc or None."""
    if not check_connection():
        return None
    try:
        return users_col.find_one({"email": email.lower().strip()})
    except Exception as e:
        logger.error(f"Failed to fetch user by email: {e}")
        return None


def get_user_by_id(user_id: str) -> dict | None:
    """Find a user document by string ObjectId. Returns full doc or None."""
    if not check_connection():
        return None
    try:
        if not ObjectId.is_valid(user_id):
            return None
        return users_col.find_one({"_id": ObjectId(user_id)})
    except Exception as e:
        logger.error(f"Failed to fetch user by ID: {e}")
        return None


def update_user(user_id: str, fields: dict) -> bool:
    """
    Update specific fields of a user document.
    Returns True on success, False otherwise.
    """
    if not check_connection():
        return False
    try:
        if not ObjectId.is_valid(user_id):
            return False
        result = users_col.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": fields}
        )
        return result.modified_count > 0
    except Exception as e:
        logger.error(f"Failed to update user: {e}")
        return False


def count_user_scans(user_id: str) -> int:
    """Return the total number of scans belonging to a user."""
    if not check_connection():
        return 0
    try:
        if not ObjectId.is_valid(user_id):
            return 0
        return scans_col.count_documents({"userId": ObjectId(user_id)})
    except Exception as e:
        logger.error(f"Failed to count user scans: {e}")
        return 0


# ===========================================================================
# SCANS COLLECTION
# ===========================================================================

def save_scan(disease, confidence, severity, is_healthy,
              image_data_url=None, filename=None, user_id=None,
              weather_snapshot=None, risk_level=None, risk_score=None):
    """
    Save a disease prediction result to the scans collection.
    Now supports weatherSnapshot, riskLevel, riskScore (Phase 1E).
    """
    if not check_connection():
        logger.warning("Database offline. Skipping MongoDB save.")
        return None

    try:
        scan_doc = {
            "disease":      disease,
            "confidence":   float(confidence),
            "severity":     severity,
            "isHealthy":    bool(is_healthy),
            "imageDataUrl": image_data_url,
            "filename":     filename,
            "timestamp":    datetime.datetime.utcnow(),
        }

        if user_id and ObjectId.is_valid(user_id):
            scan_doc["userId"] = ObjectId(user_id)

        # Phase 1E — weather + risk enrichment
        if weather_snapshot and isinstance(weather_snapshot, dict):
            scan_doc["weatherSnapshot"] = weather_snapshot
        if risk_level:
            scan_doc["riskLevel"] = risk_level
        if risk_score is not None:
            scan_doc["riskScore"] = int(risk_score)

        result = scans_col.insert_one(scan_doc)
        logger.info(f"Scan saved | ID: {result.inserted_id} | user: {user_id} | risk: {risk_level}")
        return str(result.inserted_id)

    except Exception as e:
        logger.error(f"Failed to save scan to database: {e}")
        return None


def get_scans(limit=100, user_id=None):
    """
    Fetch scan records, sorted newest first.
    Returns extended fields including weatherSnapshot, riskLevel, riskScore.
    """
    if not check_connection():
        logger.warning("Database offline. Cannot fetch scans from MongoDB.")
        return []

    try:
        if user_id and ObjectId.is_valid(user_id):
            query = {"userId": ObjectId(user_id)}
        else:
            query = {"userId": {"$exists": False}}

        cursor = scans_col.find(query).sort("timestamp", -1).limit(limit)
        scans = []
        for doc in cursor:
            scans.append({
                "id":              str(doc["_id"]),
                "disease":         doc["disease"],
                "confidence":      doc["confidence"],
                "severity":        doc["severity"],
                "isHealthy":       doc.get("isHealthy", False),
                "imageDataUrl":    doc.get("imageDataUrl"),
                "filename":        doc.get("filename"),
                "timestamp":       doc["timestamp"].isoformat() + "Z",
                "numericId":       int(doc["timestamp"].timestamp() * 1000),
                "weatherSnapshot": doc.get("weatherSnapshot"),
                "riskLevel":       doc.get("riskLevel"),
                "riskScore":       doc.get("riskScore"),
            })
        return scans

    except Exception as e:
        logger.error(f"Failed to fetch scans from database: {e}")
        return []


def delete_scan(scan_id, user_id=None):
    """
    Delete a single scan by its document ID.
    If user_id is provided, also verify ownership before deleting.

    Returns: True if deleted, False otherwise.
    """
    if not check_connection():
        logger.warning("Database offline. Cannot delete scan.")
        return False

    try:
        query = {}
        if ObjectId.is_valid(scan_id):
            query["_id"] = ObjectId(scan_id)
        else:
            query["_id"] = scan_id

        # Scope deletion to owner if user_id is provided
        if user_id and ObjectId.is_valid(user_id):
            query["userId"] = ObjectId(user_id)

        result = scans_col.delete_one(query)
        deleted = result.deleted_count > 0
        if deleted:
            logger.info(f"Deleted scan: {scan_id}")
        else:
            logger.warning(f"Scan not found or permission denied for deletion: {scan_id}")
        return deleted

    except Exception as e:
        logger.error(f"Failed to delete scan: {e}")
        return False


def clear_all(user_id=None):
    """
    Clear scan documents.
    If user_id is provided, clear only that user's scans.
    Otherwise clear all legacy (anonymous) scans.
    """
    if not check_connection():
        logger.warning("Database offline. Cannot clear scans.")
        return False

    try:
        if user_id and ObjectId.is_valid(user_id):
            query = {"userId": ObjectId(user_id)}
        else:
            query = {"userId": {"$exists": False}}

        result = scans_col.delete_many(query)
        logger.info(f"Cleared scans from MongoDB. Deleted {result.deleted_count} documents.")
        return True

    except Exception as e:
        logger.error(f"Failed to clear scans collection: {e}")
        return False


# ===========================================================================
# DASHBOARD STATS  (Phase 1B)
# ===========================================================================

def get_dashboard_stats(user_id: str) -> dict:
    """
    Compute personalised dashboard statistics for a user.
    Returns (Phase 2C extended):
        totalScans, healthyPlants, diseasedPlants, lastScan, topDisease,
        riskBreakdown, recentActivity, avgConfidence, highestRiskScan,
        monthlyProgress, scanActivityTrend
    """
    empty = {
        "totalScans": 0, "healthyPlants": 0, "diseasedPlants": 0,
        "lastScan": None, "topDisease": None,
        "riskBreakdown": {"HIGH": 0, "MEDIUM": 0, "LOW": 0},
        "recentActivity": [],
        "avgConfidence": None,
        "highestRiskScan": None,
        "monthlyProgress": [],
        "scanActivityTrend": [],
    }

    if not check_connection() or not ObjectId.is_valid(user_id):
        return empty

    try:
        oid = ObjectId(user_id)
        all_scans = list(
            scans_col.find({"userId": oid})
                     .sort("timestamp", -1)
                     .limit(1000)
        )

        total    = len(all_scans)
        healthy  = sum(1 for s in all_scans if s.get("isHealthy", False))
        diseased = total - healthy

        # Last scan
        last_scan = None
        if all_scans:
            s = all_scans[0]
            last_scan = {
                "disease":   s.get("disease", "Unknown"),
                "date":      s["timestamp"].isoformat() + "Z",
                "isHealthy": s.get("isHealthy", False),
                "severity":  s.get("severity", "LOW"),
            }

        # Top disease (among diseased only)
        disease_counts: dict = {}
        for s in all_scans:
            if not s.get("isHealthy", False):
                d = s.get("disease", "Unknown")
                disease_counts[d] = disease_counts.get(d, 0) + 1
        top_disease = max(disease_counts, key=disease_counts.get) if disease_counts else None

        # Risk breakdown
        risk_breakdown = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
        highest_risk_scan = None
        highest_risk_score = -1
        for s in all_scans:
            rl = s.get("riskLevel", "LOW")
            if rl in risk_breakdown:
                risk_breakdown[rl] += 1
            # Track highest risk scan (by score)
            rs = s.get("riskScore") or 0
            if rs > highest_risk_score:
                highest_risk_score = rs
                highest_risk_scan = {
                    "disease":   s.get("disease", "Unknown"),
                    "riskLevel": rl,
                    "riskScore": rs,
                    "date":      s["timestamp"].isoformat() + "Z",
                }

        # Average confidence (Phase 2C)
        conf_values = [s.get("confidence", 0) for s in all_scans if s.get("confidence") is not None]
        avg_confidence = round(sum(conf_values) / len(conf_values), 1) if conf_values else None

        # Recent activity (last 10)
        recent = []
        for s in all_scans[:10]:
            recent.append({
                "id":        str(s["_id"]),
                "disease":   s.get("disease", "Unknown"),
                "isHealthy": s.get("isHealthy", False),
                "severity":  s.get("severity", "LOW"),
                "riskLevel": s.get("riskLevel", "LOW"),
                "confidence":s.get("confidence"),
                "date":      s["timestamp"].isoformat() + "Z",
            })

        # Monthly progress — last 6 calendar months (Phase 2C)
        now = datetime.datetime.utcnow()
        monthly_progress = []
        for i in range(5, -1, -1):  # 5 months ago → current
            month_dt = (now.replace(day=1) - datetime.timedelta(days=1)).replace(day=1)
            # Calculate target month
            target_month = now.month - i
            target_year  = now.year
            while target_month <= 0:
                target_month += 12
                target_year  -= 1
            month_label = datetime.date(target_year, target_month, 1).strftime("%b %Y")
            m_scans = [
                s for s in all_scans
                if s["timestamp"].month == target_month and s["timestamp"].year == target_year
            ]
            monthly_progress.append({
                "month":    month_label,
                "scans":    len(m_scans),
                "healthy":  sum(1 for s in m_scans if s.get("isHealthy", False)),
                "diseased": sum(1 for s in m_scans if not s.get("isHealthy", False)),
            })

        # 30-Day scan activity trend (Phase 2C)
        scan_activity_trend = []
        for i in range(29, -1, -1):
            day_dt    = now - datetime.timedelta(days=i)
            day_label = day_dt.strftime("%d %b")
            count     = sum(
                1 for s in all_scans
                if s["timestamp"].date() == day_dt.date()
            )
            scan_activity_trend.append({"date": day_label, "count": count})

        return {
            "totalScans":        total,
            "healthyPlants":     healthy,
            "diseasedPlants":    diseased,
            "lastScan":          last_scan,
            "topDisease":        top_disease,
            "riskBreakdown":     risk_breakdown,
            "recentActivity":    recent,
            "avgConfidence":     avg_confidence,
            "highestRiskScan":   highest_risk_scan,
            "monthlyProgress":   monthly_progress,
            "scanActivityTrend": scan_activity_trend,
        }

    except Exception as e:
        logger.error(f"Failed to compute dashboard stats: {e}")
        return empty


# ===========================================================================
# CHAT HISTORY COLLECTION  (Phase 3A — RAG Agriculture Expert)
# ===========================================================================

def save_chat_message(
    user_id: str | None,
    question: str,
    answer: str,
    sources: list,
    mode: str,
) -> str | None:
    """
    Persist a RAG chat exchange to the chat_history collection.

    Args:
        user_id : ObjectId string of the authenticated user (None for guests)
        question: User's question text
        answer  : RAG or fallback answer text
        sources : List of source dicts [{document, page, category}]
        mode    : "rag" | "fallback"

    Returns:
        Inserted document _id string, or None on failure.
    """
    if not check_connection():
        logger.warning("[chat_history] Database offline. Cannot save chat message.")
        return None
    try:
        doc = {
            "question":  question,
            "answer":    answer,
            "sources":   sources or [],
            "mode":      mode,
            "timestamp": datetime.datetime.utcnow(),
        }
        if user_id and ObjectId.is_valid(user_id):
            doc["userId"] = ObjectId(user_id)

        result = chat_history_col.insert_one(doc)
        logger.info(f"[chat_history] Saved | id={result.inserted_id} | mode={mode} | user={user_id}")
        return str(result.inserted_id)
    except Exception as e:
        logger.error(f"[chat_history] Failed to save message: {e}")
        return None


def get_chat_history(user_id: str | None, limit: int = 10) -> list:
    """
    Fetch the last `limit` RAG chat messages for a user, newest first.

    Args:
        user_id: ObjectId string of the authenticated user (None returns [])
        limit  : Maximum number of messages to return (default 10)

    Returns:
        List of chat message dicts, or [] if offline / no history.
    """
    if not check_connection():
        return []
    try:
        query = (
            {"userId": ObjectId(user_id)}
            if user_id and ObjectId.is_valid(user_id)
            else {"userId": {"$exists": False}}
        )
        cursor = (
            chat_history_col.find(query)
            .sort("timestamp", -1)
            .limit(limit)
        )
        results = []
        for doc in cursor:
            results.append({
                "id":        str(doc["_id"]),
                "question":  doc.get("question", ""),
                "answer":    doc.get("answer", ""),
                "sources":   doc.get("sources", []),
                "mode":      doc.get("mode", "rag"),
                "timestamp": doc["timestamp"].isoformat() + "Z",
            })
        return results
    except Exception as e:
        logger.error(f"[chat_history] Failed to fetch history: {e}")
        return []
