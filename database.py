"""
AgroAI — MongoDB Database Connection Manager
=============================================
This module handles connection to MongoDB Atlas, checks health on startup,
and provides CRUD operations for storing and fetching crop scans.
"""

import os
import logging
import datetime
from bson import ObjectId
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from dotenv import load_dotenv

# Set up logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "agroai"
COLLECTION_NAME = "scans"

db_client = None
db = None
scans_col = None
is_connected = False

def init_db():
    """Initialize MongoDB Atlas Connection."""
    global db_client, db, scans_col, is_connected
    
    if not MONGO_URI:
        logger.warning("MONGO_URI not defined in .env. MongoDB features will run in offline simulation mode.")
        is_connected = False
        return False
        
    try:
        # 5-second timeout for server selection so startup doesn't hang if offline
        db_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        # Force a connection check
        db_client.admin.command('ping')
        
        db = db_client[DB_NAME]
        scans_col = db[COLLECTION_NAME]
        is_connected = True
        logger.info("Successfully connected to MongoDB Atlas!")
        return True
    except (ConnectionFailure, ServerSelectionTimeoutError) as e:
        logger.error(f"Failed to connect to MongoDB Atlas: {e}")
        is_connected = False
        db_client = None
        db = None
        scans_col = None
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

def save_scan(disease, confidence, severity, is_healthy, image_data_url=None, filename=None):
    """
    Save a disease prediction result to the scans collection.
    
    Returns:
        The inserted document ID as a string, or None if connection failed.
    """
    if not check_connection():
        logger.warning("Database offline. Skipping MongoDB save.")
        return None
        
    try:
        scan_doc = {
            "disease": disease,
            "confidence": float(confidence),
            "severity": severity,
            "isHealthy": bool(is_healthy),
            "imageDataUrl": image_data_url, # Base64 data URL
            "filename": filename,
            "timestamp": datetime.datetime.utcnow() # Saved as ISO Date in Mongo
        }
        
        result = scans_col.insert_one(scan_doc)
        logger.info(f"Scan saved to MongoDB with ID: {result.inserted_id}")
        return str(result.inserted_id)
    except Exception as e:
        logger.error(f"Failed to save scan to database: {e}")
        return None

def get_scans(limit=100):
    """
    Fetch the list of previous scans, sorted newest first.
    
    Returns:
        List of dicts representing scans, formatted for frontend compatibility.
    """
    if not check_connection():
        logger.warning("Database offline. Cannot fetch scans from MongoDB.")
        return []
        
    try:
        cursor = scans_col.find().sort("timestamp", -1).limit(limit)
        scans = []
        for doc in cursor:
            # Map MongoDB fields to match frontend's expected properties
            scans.append({
                "id": str(doc["_id"]),
                "disease": doc["disease"],
                "confidence": doc["confidence"],
                "severity": doc["severity"],
                "isHealthy": doc.get("isHealthy", False),
                "imageDataUrl": doc.get("imageDataUrl"),
                "filename": doc.get("filename"),
                # Return standard JS-compatible timestamp representation
                "timestamp": doc["timestamp"].isoformat() + "Z",
                # Also include milliseconds ID since some frontend parts may parse date from ID
                "numericId": int(doc["timestamp"].timestamp() * 1000)
            })
        return scans
    except Exception as e:
        logger.error(f"Failed to fetch scans from database: {e}")
        return []

def delete_scan(scan_id):
    """
    Delete a single scan by its document ID.
    
    Returns:
        True if deleted, False otherwise.
    """
    if not check_connection():
        logger.warning("Database offline. Cannot delete scan.")
        return False
        
    try:
        # Try deleting with ObjectId or string ID just in case
        if ObjectId.is_valid(scan_id):
            result = scans_col.delete_one({"_id": ObjectId(scan_id)})
        else:
            result = scans_col.delete_one({"_id": scan_id})
            
        deleted = result.deleted_count > 0
        if deleted:
            logger.info(f"Deleted scan: {scan_id}")
        else:
            logger.warning(f"Scan not found for deletion: {scan_id}")
        return deleted
    except Exception as e:
        logger.error(f"Failed to delete scan: {e}")
        return False

def clear_all():
    """
    Clear all documents in the scans collection.
    
    Returns:
        True if successful, False otherwise.
    """
    if not check_connection():
        logger.warning("Database offline. Cannot clear scans.")
        return False
        
    try:
        result = scans_col.delete_many({})
        logger.info(f"Cleared all scans from MongoDB. Deleted {result.deleted_count} documents.")
        return True
    except Exception as e:
        logger.error(f"Failed to clear scans collection: {e}")
        return False
