"""
AgroAI — MongoDB Database Connection Manager
=============================================
Handles connection to MongoDB Atlas and provides CRUD operations for:
  - scans               collection (user-scoped + legacy support)
  - users               collection (SaaS multi-user)
  - feedbacks           collection (Phase 2B — real feedback system)
  - plant_tracks        collection (Phase 4B — tracked plant records)
  - tracked_plant_scans collection (Phase 4B — per-plant scan history)
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
CHAT_HISTORY_COLLECTION    = "chat_history"   # Phase 3A RAG
PLANT_TRACKS_COLLECTION    = "plant_tracks"   # Phase 4B
TRACKED_SCANS_COLLECTION   = "tracked_plant_scans"  # Phase 4B

db_client           = None
db                  = None
scans_col           = None
users_col           = None
feedbacks_col       = None
chat_history_col    = None  # Phase 3A RAG
plant_tracks_col    = None  # Phase 4B
tracked_scans_col   = None  # Phase 4B
is_connected        = False


def init_db():
    """Initialize MongoDB Atlas Connection."""
    global db_client, db, scans_col, users_col, feedbacks_col, chat_history_col, \
           plant_tracks_col, tracked_scans_col, is_connected

    if not MONGO_URI:
        logger.warning("MONGO_URI not defined in .env. MongoDB features will run in offline simulation mode.")
        is_connected = False
        return False

    try:
        db_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        db_client.admin.command('ping')

        db                = db_client[DB_NAME]
        scans_col         = db[COLLECTION_NAME]
        users_col         = db[USERS_COLLECTION]
        feedbacks_col     = db[FEEDBACKS_COLLECTION]
        chat_history_col  = db[CHAT_HISTORY_COLLECTION]   # Phase 3A RAG
        plant_tracks_col  = db[PLANT_TRACKS_COLLECTION]   # Phase 4B
        tracked_scans_col = db[TRACKED_SCANS_COLLECTION]  # Phase 4B

        # Ensure unique email index on users collection
        users_col.create_index([("email", ASCENDING)], unique=True)
        # Index feedbacks by userId for fast per-user queries
        feedbacks_col.create_index([("userId", ASCENDING)])
        # Index chat_history by userId + timestamp (Phase 3A RAG)
        chat_history_col.create_index([("userId", ASCENDING), ("timestamp", -1)])
        # Phase 4B — indexes for plant tracking collections
        plant_tracks_col.create_index([("userId", ASCENDING), ("createdAt", -1)])
        tracked_scans_col.create_index([("plantId", ASCENDING), ("scanDate", -1)])
        tracked_scans_col.create_index([("userId",  ASCENDING), ("scanDate", -1)])

        is_connected = True
        logger.info("Successfully connected to MongoDB Atlas!")
        return True

    except (ConnectionFailure, ServerSelectionTimeoutError) as e:
        logger.error(f"Failed to connect to MongoDB Atlas: {e}")
        is_connected = False
        db_client = db = scans_col = users_col = feedbacks_col = chat_history_col = \
            plant_tracks_col = tracked_scans_col = None
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



# ===========================================================================
# ADMIN ANALYTICS  (Phase 4A — Admin Intelligence Dashboard)
# ===========================================================================

def get_admin_overview() -> dict:
    """
    Platform-level KPIs for the Admin Overview tab.
    Returns:
        totalUsers, activeUsers (last 30d), totalScans, totalFeedback,
        totalRagQueries, avgScansPerUser
    """
    empty = {
        "totalUsers": 0, "activeUsers": 0, "totalScans": 0,
        "totalFeedback": 0, "totalRagQueries": 0, "avgScansPerUser": 0,
    }
    if not check_connection():
        return empty
    try:
        total_users    = users_col.count_documents({})
        total_scans    = scans_col.count_documents({})
        total_feedback = feedbacks_col.count_documents({})
        total_rag      = chat_history_col.count_documents({})

        # Active users = distinct userId values in scans in last 30 days
        cutoff = datetime.datetime.utcnow() - datetime.timedelta(days=30)
        pipeline_active = [
            {"$match": {"timestamp": {"$gte": cutoff}, "userId": {"$exists": True}}},
            {"$group": {"_id": "$userId"}},
            {"$count": "count"},
        ]
        r = list(scans_col.aggregate(pipeline_active))
        active_users = r[0]["count"] if r else 0

        avg_scans = round(total_scans / total_users, 1) if total_users else 0

        return {
            "totalUsers":     total_users,
            "activeUsers":    active_users,
            "totalScans":     total_scans,
            "totalFeedback":  total_feedback,
            "totalRagQueries": total_rag,
            "avgScansPerUser": avg_scans,
        }
    except Exception as e:
        logger.error(f"[Admin] get_admin_overview error: {e}")
        return empty


def get_admin_agriculture() -> dict:
    """
    Agriculture intelligence aggregation across all scans.
    Returns:
        topDiseases, diseaseDistribution, highRiskCount, avgRiskScore,
        weatherImpactSummary, mostCommonCropIssues, severityBreakdown
    """
    empty = {
        "topDiseases": [], "diseaseDistribution": [],
        "highRiskCount": 0, "avgRiskScore": 0,
        "weatherImpactSummary": {}, "mostCommonCropIssues": [],
        "severityBreakdown": {"HIGH": 0, "MEDIUM": 0, "LOW": 0},
    }
    if not check_connection():
        return empty
    try:
        # Disease frequency (excluding healthy)
        pipeline_disease = [
            {"$match": {"isHealthy": {"$ne": True}}},
            {"$group": {"_id": "$disease", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": 15},
        ]
        disease_docs = list(scans_col.aggregate(pipeline_disease))
        top_diseases = [{"disease": d["_id"], "count": d["count"]} for d in disease_docs]
        disease_distribution = top_diseases[:10]

        # High risk count
        high_risk = scans_col.count_documents({"riskLevel": "HIGH"})

        # Average risk score
        pipeline_risk = [
            {"$match": {"riskScore": {"$exists": True, "$ne": None}}},
            {"$group": {"_id": None, "avg": {"$avg": "$riskScore"}}},
        ]
        r = list(scans_col.aggregate(pipeline_risk))
        avg_risk = round(r[0]["avg"], 1) if r else 0

        # Severity breakdown
        sev_pipeline = [
            {"$group": {"_id": "$severity", "count": {"$sum": 1}}},
        ]
        sev_docs = list(scans_col.aggregate(sev_pipeline))
        sev_breakdown = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
        for s in sev_docs:
            if s["_id"] in sev_breakdown:
                sev_breakdown[s["_id"]] = s["count"]

        # Weather impact summary — avg humidity/temp across HIGH risk scans
        pipeline_weather = [
            {"$match": {"riskLevel": "HIGH", "weatherSnapshot": {"$exists": True}}},
            {"$group": {
                "_id": None,
                "avgTemp":     {"$avg": "$weatherSnapshot.temperature"},
                "avgHumidity": {"$avg": "$weatherSnapshot.humidity"},
                "avgRain":     {"$avg": "$weatherSnapshot.rainChance"},
                "count":       {"$sum": 1},
            }},
        ]
        w = list(scans_col.aggregate(pipeline_weather))
        weather_summary = {}
        if w:
            weather_summary = {
                "avgTemp":     round(w[0].get("avgTemp", 0) or 0, 1),
                "avgHumidity": round(w[0].get("avgHumidity", 0) or 0, 1),
                "avgRain":     round(w[0].get("avgRain", 0) or 0, 1),
                "count":       w[0].get("count", 0),
            }

        # Most common crop issues (extract crop from disease name)
        crop_issues = {}
        for d in disease_docs:
            name = d["_id"] or ""
            parts = name.split("___")
            if len(parts) >= 2:
                crop = parts[0].replace("_", " ").title()
                crop_issues[crop] = crop_issues.get(crop, 0) + d["count"]
        sorted_crops = sorted(crop_issues.items(), key=lambda x: x[1], reverse=True)[:8]
        most_common_crops = [{"crop": c, "count": n} for c, n in sorted_crops]

        return {
            "topDiseases":           top_diseases,
            "diseaseDistribution":   disease_distribution,
            "highRiskCount":         high_risk,
            "avgRiskScore":          avg_risk,
            "weatherImpactSummary":  weather_summary,
            "mostCommonCropIssues":  most_common_crops,
            "severityBreakdown":     sev_breakdown,
        }
    except Exception as e:
        logger.error(f"[Admin] get_admin_agriculture error: {e}")
        return empty


def _contains_hindi(text: str) -> bool:
    """Return True if text contains Devanagari Unicode characters (Hindi)."""
    return any('\u0900' <= ch <= '\u097F' for ch in (text or ""))


def get_admin_rag() -> dict:
    """
    RAG Analytics: query volumes, top questions, top sources,
    success rate, fallback rate, chat volume trend (last 30 days).
    """
    empty = {
        "totalRagQueries": 0, "ragSuccessRate": 0, "fallbackRate": 0,
        "topQuestions": [], "topSources": [], "categoryDistribution": [],
        "chatVolumeTrend": [],
    }
    if not check_connection():
        return empty
    try:
        all_chats = list(
            chat_history_col.find({}, {"question": 1, "mode": 1, "sources": 1, "timestamp": 1})
        )
        total = len(all_chats)
        if not total:
            return empty

        rag_count      = sum(1 for c in all_chats if c.get("mode") == "rag")
        fallback_count = total - rag_count
        success_rate   = round(rag_count / total * 100, 1) if total else 0
        fallback_rate  = round(fallback_count / total * 100, 1) if total else 0

        # Top questions (by exact match frequency)
        q_freq = {}
        for c in all_chats:
            q = (c.get("question") or "").strip()
            if q:
                q_freq[q] = q_freq.get(q, 0) + 1
        top_questions = [
            {"question": q[:120], "count": n}
            for q, n in sorted(q_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        ]

        # Top sources and category distribution
        source_freq   = {}
        category_freq = {}
        for c in all_chats:
            for src in (c.get("sources") or []):
                doc = src.get("document", "")
                cat = src.get("category", "General")
                if doc:
                    source_freq[doc] = source_freq.get(doc, 0) + 1
                if cat:
                    category_freq[cat] = category_freq.get(cat, 0) + 1
        top_sources = [
            {"document": k[:60], "count": v}
            for k, v in sorted(source_freq.items(), key=lambda x: x[1], reverse=True)[:8]
        ]
        category_distribution = [
            {"category": k, "count": v}
            for k, v in sorted(category_freq.items(), key=lambda x: x[1], reverse=True)[:8]
        ]

        # Chat volume trend (last 30 days)
        now = datetime.datetime.utcnow()
        trend = []
        for i in range(29, -1, -1):
            day_dt    = now - datetime.timedelta(days=i)
            day_label = day_dt.strftime("%d %b")
            count     = sum(
                1 for c in all_chats
                if c.get("timestamp") and c["timestamp"].date() == day_dt.date()
            )
            trend.append({"date": day_label, "count": count})

        return {
            "totalRagQueries":      total,
            "ragSuccessRate":       success_rate,
            "fallbackRate":         fallback_rate,
            "topQuestions":         top_questions,
            "topSources":           top_sources,
            "categoryDistribution": category_distribution,
            "chatVolumeTrend":      trend,
        }
    except Exception as e:
        logger.error(f"[Admin] get_admin_rag error: {e}")
        return empty


def get_admin_feedback() -> dict:
    """
    Feedback Analytics: avg rating, count, distribution,
    latest 10 entries, keyword frequency, trend (last 30 days).
    """
    empty = {
        "avgRating": None, "totalFeedback": 0,
        "ratingDistribution": {1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
        "latestFeedback": [], "keywordFrequency": [], "feedbackTrend": [],
    }
    if not check_connection():
        return empty
    try:
        all_fb = list(feedbacks_col.find().sort("createdAt", -1))
        total  = len(all_fb)
        if not total:
            return empty

        ratings = [f.get("rating", 0) for f in all_fb]
        avg     = round(sum(ratings) / total, 2) if ratings else None
        dist    = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        for r in ratings:
            if r in dist:
                dist[r] += 1

        latest = []
        for f in all_fb[:10]:
            latest.append({
                "name":    f.get("name", "Anonymous"),
                "rating":  f.get("rating", 0),
                "message": f.get("message", "")[:200],
                "date":    f["createdAt"].isoformat() + "Z" if f.get("createdAt") else None,
            })

        # Keyword frequency — common words in messages (simple tokenizer)
        STOPWORDS = {
            "the","a","an","is","it","to","and","or","in","of","for","on","with",
            "this","that","i","my","me","we","are","was","be","so","very","app",
            "agroai","good","really","great","works","use","its","at","as","by","not"
        }
        word_freq = {}
        for f in all_fb:
            msg = (f.get("message") or "").lower()
            for word in msg.split():
                word = word.strip(".,!?;:'\"")
                if len(word) >= 4 and word not in STOPWORDS:
                    word_freq[word] = word_freq.get(word, 0) + 1
        keyword_frequency = [
            {"word": w, "count": c}
            for w, c in sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:20]
        ]

        # Feedback trend (last 30 days — count per day)
        now = datetime.datetime.utcnow()
        trend = []
        for i in range(29, -1, -1):
            day_dt    = now - datetime.timedelta(days=i)
            day_label = day_dt.strftime("%d %b")
            count     = sum(
                1 for f in all_fb
                if f.get("createdAt") and f["createdAt"].date() == day_dt.date()
            )
            trend.append({"date": day_label, "count": count})

        return {
            "avgRating":           avg,
            "totalFeedback":       total,
            "ratingDistribution":  dist,
            "latestFeedback":      latest,
            "keywordFrequency":    keyword_frequency,
            "feedbackTrend":       trend,
        }
    except Exception as e:
        logger.error(f"[Admin] get_admin_feedback error: {e}")
        return empty


def get_admin_languages() -> dict:
    """
    Language Analytics derived from chat_history (Hindi Unicode detection).
    Returns:
        englishCount, hindiCount, englishPct, hindiPct, mostUsedLanguage,
        languageTrend (last 30 days)
    """
    empty = {
        "englishCount": 0, "hindiCount": 0,
        "englishPct": 0, "hindiPct": 0,
        "mostUsedLanguage": "English", "languageTrend": [],
    }
    if not check_connection():
        return empty
    try:
        all_chats = list(
            chat_history_col.find({}, {"question": 1, "timestamp": 1})
        )
        total = len(all_chats)
        if not total:
            return empty

        hindi_count   = sum(1 for c in all_chats if _contains_hindi(c.get("question", "")))
        english_count = total - hindi_count
        hindi_pct     = round(hindi_count / total * 100, 1) if total else 0
        english_pct   = round(100 - hindi_pct, 1)
        most_used     = "Hindi" if hindi_count > english_count else "English"

        # Language trend — last 30 days
        now = datetime.datetime.utcnow()
        trend = []
        for i in range(29, -1, -1):
            day_dt    = now - datetime.timedelta(days=i)
            day_label = day_dt.strftime("%d %b")
            day_chats = [
                c for c in all_chats
                if c.get("timestamp") and c["timestamp"].date() == day_dt.date()
            ]
            en = sum(1 for c in day_chats if not _contains_hindi(c.get("question", "")))
            hi = len(day_chats) - en
            trend.append({"date": day_label, "english": en, "hindi": hi})

        return {
            "englishCount":    english_count,
            "hindiCount":      hindi_count,
            "englishPct":      english_pct,
            "hindiPct":        hindi_pct,
            "mostUsedLanguage": most_used,
            "languageTrend":   trend,
        }
    except Exception as e:
        logger.error(f"[Admin] get_admin_languages error: {e}")
        return empty



# ===========================================================================
# ADMIN USER MANAGEMENT  (Phase 4C — Admin User Management Dashboard)
# ===========================================================================

def get_admin_users() -> list:
    """
    Return all registered users with aggregated platform usage statistics.

    For each user, computes:
        - totalScans       : count of docs in 'scans' collection for that user
        - trackedPlants    : count of docs in 'plant_tracks' collection
        - feedbackCount    : count of docs in 'feedbacks' collection
        - ragQueries       : count of docs in 'chat_history' collection
        - lastActivity     : latest timestamp across scans / chat / feedback
        - status           : 'active' if lastActivity within 7 days, else 'inactive'

    SECURITY: Sensitive fields (password, passwordHash, tokens) are NEVER returned.

    Returns:
        List of user dicts sorted by createdAt descending (newest first).
        Returns [] if database is offline.
    """
    if not check_connection():
        logger.warning("[Admin] get_admin_users: database offline")
        return []
    try:
        now = datetime.datetime.utcnow()
        active_cutoff = now - datetime.timedelta(days=7)

        # Fetch all users — exclude every sensitive field explicitly
        raw_users = list(
            users_col.find(
                {},
                {
                    # Exclude all password/token fields — MongoDB projection
                    "password":     0,
                    "passwordHash": 0,
                    "token":        0,
                    "refreshToken": 0,
                    "resetToken":   0,
                    "secret":       0,
                }
            ).sort("createdAt", -1)
        )

        result = []
        for u in raw_users:
            uid = u["_id"]

            # ── Per-user counts (each is a fast indexed count) ──────────────
            total_scans = scans_col.count_documents({"userId": uid})
            tracked_plants = plant_tracks_col.count_documents({"userId": uid})
            feedback_count = feedbacks_col.count_documents({"userId": uid})
            rag_queries = chat_history_col.count_documents({"userId": uid})

            # ── Last activity: latest timestamp across three collections ─────
            last_ts = None

            # Latest scan timestamp
            last_scan = scans_col.find_one(
                {"userId": uid}, {"timestamp": 1}, sort=[("timestamp", -1)]
            )
            if last_scan and last_scan.get("timestamp"):
                ts = last_scan["timestamp"]
                if last_ts is None or ts > last_ts:
                    last_ts = ts

            # Latest chat_history timestamp
            last_chat = chat_history_col.find_one(
                {"userId": uid}, {"timestamp": 1}, sort=[("timestamp", -1)]
            )
            if last_chat and last_chat.get("timestamp"):
                ts = last_chat["timestamp"]
                if last_ts is None or ts > last_ts:
                    last_ts = ts

            # Latest feedback timestamp
            last_fb = feedbacks_col.find_one(
                {"userId": uid}, {"createdAt": 1}, sort=[("createdAt", -1)]
            )
            if last_fb and last_fb.get("createdAt"):
                ts = last_fb["createdAt"]
                if last_ts is None or ts > last_ts:
                    last_ts = ts

            # ── Status: active if last activity ≤ 7 days ────────────────────
            status = "inactive"
            if last_ts and last_ts >= active_cutoff:
                status = "active"

            # ── Build the safe output dict (NO sensitive fields) ─────────────
            result.append({
                "_id":          str(uid),
                "name":         u.get("name", "Unknown"),
                "email":        u.get("email", ""),
                "role":         u.get("role", "user"),
                "createdAt":    u["createdAt"].isoformat() + "Z" if u.get("createdAt") else None,
                "totalScans":   total_scans,
                "trackedPlants": tracked_plants,
                "feedbackCount": feedback_count,
                "ragQueries":   rag_queries,
                "lastActivity": last_ts.isoformat() + "Z" if last_ts else None,
                "status":       status,
            })

        logger.info(f"[Admin] get_admin_users: returned {len(result)} users")
        return result

    except Exception as e:
        logger.error(f"[Admin] get_admin_users error: {e}")
        return []


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


# ===========================================================================
# PHASE 4B — PLANT PROGRESS TRACKING
# ===========================================================================

def create_plant_track(user_id: str, plant_name: str) -> str | None:
    """
    Create a new tracked plant record for a user.

    Args:
        user_id   : ObjectId string of the authenticated user
        plant_name: Human-readable name for the plant (e.g. "My Tomato Plant")

    Returns:
        Inserted plant _id string, or None on failure.
    """
    if not check_connection() or not ObjectId.is_valid(user_id):
        return None
    try:
        doc = {
            "userId":          ObjectId(user_id),
            "plantName":       plant_name.strip()[:80],
            "createdAt":       datetime.datetime.utcnow(),
            "latestDisease":   None,
            "latestScanDate":  None,
            "totalScans":      0,
        }
        result = plant_tracks_col.insert_one(doc)
        logger.info(f"[Phase4B] Plant track created | id={result.inserted_id} | name={plant_name!r}")
        return str(result.inserted_id)
    except Exception as e:
        logger.error(f"[Phase4B] Failed to create plant track: {e}")
        return None


def add_tracked_scan(
    plant_id: str,
    user_id:  str,
    disease:  str,
    confidence: float,
    risk_score: int | None,
    weather_snapshot: dict | None,
    image_url: str | None,
) -> str | None:
    """
    Append a scan snapshot to tracked_plant_scans and update the parent
    plant_tracks document with the latest status.

    Returns the inserted scan _id string, or None on failure.
    """
    if not check_connection():
        return None
    if not ObjectId.is_valid(plant_id) or not ObjectId.is_valid(user_id):
        return None
    try:
        now = datetime.datetime.utcnow()
        scan_doc = {
            "plantId":         ObjectId(plant_id),
            "userId":          ObjectId(user_id),
            "disease":         disease,
            "confidence":      float(confidence),
            "riskScore":       int(risk_score) if risk_score is not None else 0,
            "weatherSnapshot": weather_snapshot or {},
            "imageUrl":        image_url or "",
            "scanDate":        now,
        }
        result = tracked_scans_col.insert_one(scan_doc)

        # Update parent track with latest scan metadata
        plant_tracks_col.update_one(
            {"_id": ObjectId(plant_id), "userId": ObjectId(user_id)},
            {
                "$set": {
                    "latestDisease":  disease,
                    "latestScanDate": now,
                },
                "$inc": {"totalScans": 1},
            },
        )
        logger.info(f"[Phase4B] Tracked scan added | plant={plant_id} | disease={disease!r}")
        return str(result.inserted_id)
    except Exception as e:
        logger.error(f"[Phase4B] Failed to add tracked scan: {e}")
        return None


def get_user_plants(user_id: str) -> list:
    """
    Return all tracked plants for a user, newest first.
    Each entry includes summary info derived from tracked_plant_scans.
    """
    if not check_connection() or not ObjectId.is_valid(user_id):
        return []
    try:
        oid    = ObjectId(user_id)
        tracks = list(
            plant_tracks_col.find({"userId": oid}).sort("createdAt", -1).limit(200)
        )
        results = []
        for t in tracks:
            pid = t["_id"]
            # Fetch last 2 scans to compute trend
            recent = list(
                tracked_scans_col.find({"plantId": pid})
                                 .sort("scanDate", -1)
                                 .limit(2)
            )
            trend = _compute_trend(recent)
            results.append({
                "id":             str(pid),
                "plantName":      t.get("plantName", "Unknown"),
                "createdAt":      t["createdAt"].isoformat() + "Z",
                "latestDisease":  t.get("latestDisease"),
                "latestScanDate": t["latestScanDate"].isoformat() + "Z" if t.get("latestScanDate") else None,
                "totalScans":     t.get("totalScans", 0),
                "trend":          trend,
                "latestRiskScore": recent[0].get("riskScore", 0) if recent else 0,
            })
        return results
    except Exception as e:
        logger.error(f"[Phase4B] Failed to fetch user plants: {e}")
        return []


def get_plant_history(plant_id: str, user_id: str) -> dict:
    """
    Return the full scan history + analytics for a single tracked plant.
    Verifies ownership via user_id.

    Returns:
        {
          plant: { id, plantName, createdAt, totalScans },
          scans: [ { scanDate, disease, confidence, riskScore, weatherSnapshot, imageUrl } ],
          analytics: { avgConfidence, avgRiskScore, recoveryRate, totalScans, highRiskCount }
        }
    """
    empty = {"plant": None, "scans": [], "analytics": {}}
    if not check_connection():
        return empty
    if not ObjectId.is_valid(plant_id) or not ObjectId.is_valid(user_id):
        return empty
    try:
        pid  = ObjectId(plant_id)
        uid  = ObjectId(user_id)
        # Verify ownership
        track = plant_tracks_col.find_one({"_id": pid, "userId": uid})
        if not track:
            return empty

        raw_scans = list(
            tracked_scans_col.find({"plantId": pid, "userId": uid})
                             .sort("scanDate", 1)  # oldest-first for charting
                             .limit(500)
        )

        scans = []
        for s in raw_scans:
            scans.append({
                "id":              str(s["_id"]),
                "scanDate":        s["scanDate"].isoformat() + "Z",
                "disease":         s.get("disease", "Unknown"),
                "confidence":      s.get("confidence", 0),
                "riskScore":       s.get("riskScore", 0),
                "healthScore":     max(0, 100 - s.get("riskScore", 0)),
                "weatherSnapshot": s.get("weatherSnapshot", {}),
                "imageUrl":        s.get("imageUrl", ""),
            })

        # Analytics
        total = len(scans)
        avg_conf = round(sum(s["confidence"] for s in scans) / total, 1) if total else 0
        avg_risk = round(sum(s["riskScore"]  for s in scans) / total, 1) if total else 0
        high_risk_count = sum(1 for s in scans if s["riskScore"] >= 70)

        # Recovery rate: % improvement from first to last scan
        recovery_rate = 0
        if total >= 2:
            first_risk = scans[0]["riskScore"]
            last_risk  = scans[-1]["riskScore"]
            if first_risk > 0:
                recovery_rate = round(max(0, (first_risk - last_risk) / first_risk * 100), 1)

        return {
            "plant": {
                "id":        str(track["_id"]),
                "plantName": track.get("plantName", "Unknown"),
                "createdAt": track["createdAt"].isoformat() + "Z",
                "totalScans": track.get("totalScans", total),
            },
            "scans": scans,
            "analytics": {
                "totalScans":     total,
                "avgConfidence":  avg_conf,
                "avgRiskScore":   avg_risk,
                "highRiskCount":  high_risk_count,
                "recoveryRate":   recovery_rate,
            },
        }
    except Exception as e:
        logger.error(f"[Phase4B] Failed to get plant history: {e}")
        return empty


def get_plant_analytics(user_id: str) -> dict:
    """
    Aggregate tracking analytics across all of a user's plants.

    Returns:
        totalTracked, avgRecoveryRate, highRiskPlants, mostImprovedPlant
    """
    empty = {"totalTracked": 0, "avgRecoveryRate": 0, "highRiskPlants": 0, "mostImprovedPlant": None}
    if not check_connection() or not ObjectId.is_valid(user_id):
        return empty
    try:
        uid    = ObjectId(user_id)
        tracks = list(plant_tracks_col.find({"userId": uid}))
        total  = len(tracks)
        if not total:
            return empty

        recovery_rates = []
        high_risk      = 0
        best_plant     = None
        best_recovery  = -1

        for t in tracks:
            pid = t["_id"]
            scans = list(
                tracked_scans_col.find({"plantId": pid})
                                 .sort("scanDate", 1)
                                 .limit(500)
            )
            if len(scans) >= 2:
                first_risk = scans[0].get("riskScore", 0)
                last_risk  = scans[-1].get("riskScore", 0)
                rate = round(max(0, (first_risk - last_risk) / max(first_risk, 1) * 100), 1)
                recovery_rates.append(rate)
                if rate > best_recovery:
                    best_recovery = rate
                    best_plant    = t.get("plantName", "Unknown")
            if scans and scans[-1].get("riskScore", 0) >= 70:
                high_risk += 1

        avg_recovery = round(sum(recovery_rates) / len(recovery_rates), 1) if recovery_rates else 0

        return {
            "totalTracked":     total,
            "avgRecoveryRate":  avg_recovery,
            "highRiskPlants":   high_risk,
            "mostImprovedPlant": best_plant,
        }
    except Exception as e:
        logger.error(f"[Phase4B] Failed to compute plant analytics: {e}")
        return empty


def _compute_trend(recent_scans: list) -> str:
    """Return 'recovering', 'worsening', or 'stable' based on last 2 scan riskScores."""
    if len(recent_scans) < 2:
        return "stable"
    # recent_scans[0] is the newest, [1] is older
    delta = recent_scans[1].get("riskScore", 0) - recent_scans[0].get("riskScore", 0)
    if delta >= 10:
        return "recovering"
    if delta <= -10:
        return "worsening"
    return "stable"
