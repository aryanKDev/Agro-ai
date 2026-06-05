"""
Plant Disease Detection System - Production-Ready Flask Backend
================================================================
Author notes:
  - Uses google-generativeai SDK (pip install google-generativeai) with gemini-1.5-flash
  - disease_info.json is queried FIRST for factual questions (no API call)
  - Gemini is called only for open-ended / conversational queries
  - 1 retry after 5 s on quota / server errors
  - Full graceful fallback to local DB if Gemini is unavailable
  - python-dotenv for safe API key loading
  - Structured logging for easy demo/viva monitoring
  - Flask-JWT-Extended for multi-user SaaS authentication (Phase 1A)
"""

import os
import json
import time
import logging
import datetime
import numpy as np
import tensorflow as tf
from PIL import Image
from io import BytesIO
from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
from flask_jwt_extended import JWTManager, jwt_required, get_jwt_identity, verify_jwt_in_request
from dotenv import load_dotenv
import requests
import base64
import database
import pdf_generator
import risk_engine
from services.weather_service  import get_weather, get_weather_by_coords
from services.farming_insights import generate_farming_insights
from services.rag_service      import get_rag_service          # Phase 3A RAG

# ---------------------------------------------------------------------------
# Google Generative AI SDK  (pip install google-generativeai)
# ---------------------------------------------------------------------------
import google.generativeai as genai

# ---------------------------------------------------------------------------
# Logging – visible in terminal during demo / viva
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Load .env and configure Gemini
# ---------------------------------------------------------------------------
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    logger.warning("GOOGLE_API_KEY not set in .env – chatbot will run in offline mode only.")

GEMINI_MODEL = "gemini-1.5-flash"

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    logger.info(f"Gemini SDK configured | model: {GEMINI_MODEL}")
else:
    logger.warning("Gemini SDK not configured (no API key).")

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder=BASE_DIR, template_folder=BASE_DIR)
CORS(app, supports_credentials=True)

# ---------------------------------------------------------------------------
# JWT Configuration
# ---------------------------------------------------------------------------
app.config["JWT_SECRET_KEY"] = os.getenv(
    "JWT_SECRET_KEY", "agroai_fallback_secret_change_in_production_2026"
)
_expires_seconds = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRES", 86400))
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = datetime.timedelta(seconds=_expires_seconds)

jwt = JWTManager(app)

# ---------------------------------------------------------------------------
# Register Auth Blueprint
# ---------------------------------------------------------------------------
from auth.auth_routes import auth_bp
app.register_blueprint(auth_bp)
logger.info("Auth blueprint registered at /api/auth")

# ---------------------------------------------------------------------------
# Load ML model and disease info database
# ---------------------------------------------------------------------------
MODEL_PATH        = os.path.join(BASE_DIR, "trained_plant_disease_model.keras")
DISEASE_INFO_PATH = os.path.join(BASE_DIR, "disease_info.json")

try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
    logger.info("ML model loaded.")

    if not os.path.exists(DISEASE_INFO_PATH):
        raise FileNotFoundError(f"disease_info.json not found: {DISEASE_INFO_PATH}")
    with open(DISEASE_INFO_PATH, "r", encoding="utf-8") as f:
        disease_info_db: dict = json.load(f)
    logger.info(f"Disease DB loaded: {len(disease_info_db)} entries.")

except Exception as e:
    logger.critical(f"Startup failure: {e}")
    exit(1)

CLASS_NAMES = list(disease_info_db.keys())

# In-memory conversational session store  { session_id -> list[dict] }
chat_sessions: dict = {}

# In-memory weather cache  { city_lower -> {data, expires_at} }
_weather_cache: dict = {}

# ---------------------------------------------------------------------------
# Phase 3A — RAG Service (loaded once at startup)
# ---------------------------------------------------------------------------
try:
    _rag = get_rag_service()
    if _rag.is_ready():
        logger.info("RAG service ready — FAISS vectorstore loaded.")
    else:
        logger.warning(
            "RAG service initialised but vectorstore not found. "
            "Run: python ingest.py  to build the knowledge base index."
        )
except Exception as _rag_err:
    logger.error(f"RAG service failed to initialise: {_rag_err}")
    _rag = None


# ===========================================================================
# Helper: Extract current user from JWT (optional — does not fail if no token)
# ===========================================================================
def get_optional_user_id() -> str | None:
    """
    Try to extract the JWT identity from the request.
    Returns the user_id string if authenticated, else None.
    Does NOT raise an error if no token is present.
    """
    try:
        verify_jwt_in_request(optional=True)
        return get_jwt_identity()
    except Exception:
        return None


# ===========================================================================
def create_thumbnail_base64(image_bytes: bytes, max_size=(256, 256)) -> str:
    """Generate a small JPEG base64 data URL from image bytes for database storage."""
    try:
        img = Image.open(BytesIO(image_bytes))
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.thumbnail(max_size)
        buffered = BytesIO()
        img.save(buffered, format="JPEG", quality=80)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{img_str}"
    except Exception as e:
        logger.error(f"Failed to generate thumbnail: {e}")
        return None


def model_prediction(image_bytes: bytes) -> tuple:
    """Run Keras model and return (predicted class name, confidence %)."""
    img = Image.open(BytesIO(image_bytes))
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize((128, 128))
    arr = tf.keras.preprocessing.image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    preds = model.predict(arr)
    idx = int(np.argmax(preds))
    confidence = float(np.max(preds) * 100)
    class_name = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else "Unknown_Class"
    return class_name, round(confidence, 1)


def _fmt_bullets(text: str) -> str:
    """Convert numbered list lines into clean bullet lines."""
    lines = text.strip().split("\n")
    out = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        import re
        if re.match(r'^\d+\.\s', line):
            line = "• " + re.sub(r'^\d+\.\s*', '', line)
        out.append(line)
    return "\n".join(out)


def build_local_response(db_key: str, user_message: str) -> str | None:
    """
    Return a rich, emoji-formatted farmer-friendly answer from disease_info.json.
    Returns None if the disease is not in the DB.
    """
    info = disease_info_db.get(db_key)
    if not info:
        return None

    disease_display = db_key.split("___")[-1].replace("_", " ").strip()
    msg_lower = user_message.lower()

    header = f"🌱 *Powered by Local Plant Expert Mode*"

    if any(kw in msg_lower for kw in ["symptom", "sign", "look like", "how does it look"]):
        symptoms = _fmt_bullets(info.get("symptoms", "No data available."))
        return (
            f"{header}\n\n"
            f"🔍 **Symptoms of {disease_display}**\n\n"
            f"{symptoms}\n\n"
            f"_Tip: Ask me about treatment or prevention too!_"
        )
    elif any(kw in msg_lower for kw in ["treat", "cure", "medication", "medicine", "spray", "fungicide", "control"]):
        treatment = _fmt_bullets(info.get("treatment", "No data available."))
        return (
            f"{header}\n\n"
            f"💊 **Treatment for {disease_display}**\n\n"
            f"{treatment}\n\n"
            f"_Always wear protective gear when applying any spray or fungicide._"
        )
    elif any(kw in msg_lower for kw in ["prevent", "avoid", "stop", "protect", "safe"]):
        prevention = _fmt_bullets(info.get("prevention", "No data available."))
        return (
            f"{header}\n\n"
            f"🛡️ **Prevention Tips for {disease_display}**\n\n"
            f"{prevention}\n\n"
            f"_Prevention is always better than cure!_"
        )
    elif any(kw in msg_lower for kw in ["cause", "why", "reason", "how does", "origin"]):
        symptoms = _fmt_bullets(info.get("symptoms", "N/A"))
        prevention = _fmt_bullets(info.get("prevention", "N/A"))
        return (
            f"{header}\n\n"
            f"🌿 **About {disease_display}**\n\n"
            f"This disease usually shows up as:\n{symptoms}\n\n"
            f"🛡️ **How to stop it from spreading:**\n{prevention}"
        )
    else:
        symptoms   = _fmt_bullets(info.get("symptoms",   "N/A"))
        treatment  = _fmt_bullets(info.get("treatment",  "N/A"))
        prevention = _fmt_bullets(info.get("prevention", "N/A"))
        return (
            f"{header}\n\n"
            f"📋 **{disease_display} — Complete Guide**\n\n"
            f"🔍 **Symptoms**\n{symptoms}\n\n"
            f"💊 **Treatment**\n{treatment}\n\n"
            f"🛡️ **Prevention**\n{prevention}"
        )


def call_gemini_with_retry(
    system_instruction: str,
    history: list,
    user_message: str,
    max_retries: int = 1,
    retry_delay: int = 5
) -> tuple[str, list]:
    """
    Send a message to Gemini 1.5 Flash with chat history.
    Uses google-generativeai SDK (genai.GenerativeModel).
    Retries once (after retry_delay seconds) on quota / server errors.
    Returns (reply_text, updated_history_list) or raises on failure.
    """
    if not GOOGLE_API_KEY:
        raise RuntimeError("Gemini SDK not configured (no API key).")

    model_instance = genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        system_instruction=system_instruction,
        generation_config=genai.types.GenerationConfig(
            max_output_tokens=1024,
            temperature=0.7,
        ),
    )

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            chat = model_instance.start_chat(history=history)
            response = chat.send_message(user_message)

            reply = getattr(response, "text", None)
            if not reply:
                raise ValueError("Gemini returned an empty response.")

            updated_history = [
                {"role": m.role, "parts": [p.text for p in m.parts]}
                for m in chat.history
            ]
            return reply, updated_history

        except Exception as exc:
            last_error = exc
            err_str = str(exc)
            logger.warning(f"Gemini attempt {attempt + 1}/{max_retries + 1} failed: {err_str}")

            retriable = any(
                kw in err_str.upper()
                for kw in ["429", "RESOURCE_EXHAUSTED", "QUOTA", "503", "500", "UNAVAILABLE"]
            )
            if attempt < max_retries and retriable:
                logger.info(f"Retrying in {retry_delay}s…")
                time.sleep(retry_delay)
            else:
                break

    raise last_error


# ===========================================================================
# Routes
# ===========================================================================

@app.route("/")
def home():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(BASE_DIR, filename)


@app.route("/predict", methods=["POST"])
def predict():
    """
    POST /predict
    Accepts:  multipart/form-data  → field 'file'  (uploaded image)
              application/json     → field 'url'   (public image URL)
    Returns:  JSON with prediction, disease_name, symptoms, treatment, prevention

    Authentication: Optional JWT. If authenticated, scan is saved to user account.
    If not authenticated, scan is saved as anonymous (legacy) record.
    """
    image_bytes = None
    try:
        if request.is_json:
            data = request.get_json()
            if "url" not in data:
                return jsonify({"error": "No 'url' key in JSON payload"}), 400
            resp = requests.get(data["url"], headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
            resp.raise_for_status()
            image_bytes = resp.content

        elif "file" in request.files:
            image_bytes = request.files["file"].read()

        else:
            return jsonify({"error": "No image file or URL provided"}), 400

        prediction_key, confidence = model_prediction(image_bytes)
        logger.info(f"Prediction result: {prediction_key} | Confidence: {confidence}%")

        info       = disease_info_db.get(prediction_key, {})
        symptoms   = info.get("symptoms",   "Information not available in database.")
        treatment  = info.get("treatment",  "Information not available in database.")
        prevention = info.get("prevention", "Information not available in database.")

        if "healthy" in prediction_key.lower():
            symptoms  = "✅ No disease symptoms detected. The plant appears healthy."
            treatment = "No treatment necessary. Maintain your current care routine."

        disease_name = (
            prediction_key.split("___")[-1]
            .replace("_", " ")
            .replace("(including sour)", "")
            .strip()
        )

        is_healthy = "healthy" in prediction_key.lower()
        if is_healthy:
            severity = "LOW"
        elif confidence >= 90:
            severity = "HIGH"
        elif confidence >= 70:
            severity = "MEDIUM"
        else:
            severity = "LOW"

        image_data_url = create_thumbnail_base64(image_bytes)

        filename = None
        if "file" in request.files:
            filename = request.files["file"].filename
        elif request.is_json and request.get_json().get("url"):
            url_path = request.get_json()["url"].split("/")[-1]
            filename = url_path.split("?")[0] if url_path else "url_upload.jpg"

        # ── Extract user_id from JWT if present (optional auth) ───────────
        user_id = get_optional_user_id()
        logger.info(f"[/predict] user_id from JWT: {user_id!r}  (None = guest scan)")

        # ── Auto risk analysis (uses simulated weather as default) ─────────
        weather_snap = None
        risk_level   = None
        risk_score   = None
        try:
            default_weather = get_weather("Bhopal")
            weather_snap = {
                "temperature": default_weather.get("temperature"),
                "humidity":    default_weather.get("humidity"),
                "rainChance":  default_weather.get("rainChance"),
                "windSpeed":   default_weather.get("windSpeed"),
                "condition":   default_weather.get("condition"),
                "city":        default_weather.get("city", "Bhopal"),
            }
            risk_result = risk_engine.analyse_risk(
                disease     = disease_name,
                humidity    = default_weather.get("humidity", 60),
                temperature = default_weather.get("temperature", 28),
            )
            risk_level = risk_result.get("risk")
            risk_score = risk_result.get("score")
        except Exception as re:
            logger.warning(f"Risk analysis skipped: {re}")

        # Save scan to MongoDB (with weather + risk)
        scan_id = database.save_scan(
            disease          = disease_name,
            confidence       = confidence,
            severity         = severity,
            is_healthy       = is_healthy,
            image_data_url   = image_data_url,
            filename         = filename,
            user_id          = user_id,
            weather_snapshot = weather_snap,
            risk_level       = risk_level,
            risk_score       = risk_score,
        )
        logger.info(f"[/predict] MongoDB insert → scan_id: {scan_id!r}  user_id: {user_id!r}")

        return jsonify({
            "id":           scan_id,
            "prediction":   prediction_key,
            "disease_name": disease_name,
            "symptoms":     symptoms,
            "treatment":    treatment,
            "prevention":   prevention,
            "confidence":   confidence,
            "riskLevel":    risk_level,
            "riskScore":    risk_score,
            "riskReason":   (risk_result or {}).get("reason"),
        })

    except requests.exceptions.RequestException as e:
        logger.error(f"Image download error: {e}")
        return jsonify({"error": f"Could not download image: {str(e)}"}), 500
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


# ---------------------------------------------------------------------------
# MongoDB API Routes for Scan History (user-scoped)
# ---------------------------------------------------------------------------

@app.route("/api/scans", methods=["GET"])
def get_scans_api():
    """
    GET /api/scans
    Authenticated: returns only the current user's scans.
    Guest: returns legacy anonymous scans only.
    """
    try:
        user_id = get_optional_user_id()
        logger.info(f"[GET /api/scans] user_id={user_id!r}")
        scans = database.get_scans(user_id=user_id)
        logger.info(f"[GET /api/scans] Returning {len(scans)} scans for user_id={user_id!r}")
        return jsonify(scans)
    except Exception as e:
        logger.error(f"Error fetching scans in API: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/scans/<scan_id>", methods=["DELETE"])
def delete_scan_api(scan_id):
    """DELETE /api/scans/<scan_id> — Delete a specific scan (scoped to user if authenticated)."""
    try:
        user_id = get_optional_user_id()
        success = database.delete_scan(scan_id, user_id=user_id)
        if success:
            return jsonify({"success": True, "message": f"Scan {scan_id} deleted successfully"})
        else:
            return jsonify({"success": False, "message": "Scan not found or database offline"}), 404
    except Exception as e:
        logger.error(f"Error deleting scan in API: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/scans", methods=["DELETE"])
def clear_all_scans_api():
    """DELETE /api/scans — Clear all scans for the current user (or legacy if guest)."""
    try:
        user_id = get_optional_user_id()
        success = database.clear_all(user_id=user_id)
        if success:
            return jsonify({"success": True, "message": "All scans cleared successfully"})
        else:
            return jsonify({"success": False, "message": "Failed to clear scans or database offline"}), 500
    except Exception as e:
        logger.error(f"Error clearing scans in API: {e}")
        return jsonify({"error": str(e)}), 500



# ---------------------------------------------------------------------------
# Phase 1B — Personalised Dashboard
# ---------------------------------------------------------------------------

@app.route("/api/dashboard", methods=["GET"])
@jwt_required()
def dashboard_api():
    """
    GET /api/dashboard  (JWT required)
    Returns personalised KPIs and activity data for the logged-in user.
    """
    try:
        user_id = get_jwt_identity()
        stats   = database.get_dashboard_stats(user_id)
        return jsonify({"success": True, **stats})
    except Exception as e:
        logger.error(f"Dashboard API error: {e}")
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Phase 2B — MongoDB Feedback API
# ---------------------------------------------------------------------------

@app.route("/api/feedback", methods=["POST"])
@jwt_required()
def submit_feedback():
    """
    POST /api/feedback  (JWT required)
    Body: { rating: int(1-5), message: str(10-1000 chars) }
    Returns: { success, id }
    """
    try:
        user_id = get_jwt_identity()
        user    = database.get_user_by_id(user_id)
        if not user:
            return jsonify({"error": "User not found"}), 404

        data    = request.get_json() or {}
        rating  = data.get("rating")
        message = (data.get("message") or "").strip()

        # Validate rating
        if rating is None or not isinstance(rating, int) or not (1 <= rating <= 5):
            return jsonify({"error": "Rating must be an integer between 1 and 5."}), 400

        # Validate message
        if len(message) < 10:
            return jsonify({"error": "Message must be at least 10 characters."}), 400
        if len(message) > 1000:
            return jsonify({"error": "Message cannot exceed 1000 characters."}), 400

        feedback_id = database.save_feedback(
            user_id = user_id,
            name    = user.get("name", "Anonymous"),
            email   = user.get("email", ""),
            rating  = rating,
            message = message,
        )

        if not feedback_id:
            return jsonify({"error": "Failed to save feedback. Database may be offline."}), 500

        logger.info(f"Feedback submitted | user: {user_id} | rating: {rating}")
        return jsonify({"success": True, "id": feedback_id})

    except Exception as e:
        logger.error(f"Submit feedback error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/feedback/my", methods=["GET"])
@jwt_required()
def get_my_feedback():
    """
    GET /api/feedback/my  (JWT required)
    Returns the current user's feedback submissions, newest first.
    """
    try:
        user_id   = get_jwt_identity()
        feedbacks = database.get_my_feedbacks(user_id)
        return jsonify({"success": True, "feedbacks": feedbacks})
    except Exception as e:
        logger.error(f"Get my feedback error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/feedback/stats", methods=["GET"])
def get_feedback_stats():
    """
    GET /api/feedback/stats  (public, no auth required)
    Returns aggregate feedback statistics: total, avg_rating, distribution.
    """
    try:
        stats = database.get_feedback_stats()
        return jsonify({"success": True, **stats})
    except Exception as e:
        logger.error(f"Get feedback stats error: {e}")
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Phase 1C — Real-Time Weather
# ---------------------------------------------------------------------------

@app.route("/api/weather", methods=["GET"])
def weather_api():
    """
    GET /api/weather?city=Bhopal
    GET /api/weather?lat=23.25&lon=77.41   ← Bug #2 fix: GPS coords support
    Returns agricultural weather data. Caches responses for 30 min.
    Authentication: Optional.
    """
    import time as _time

    lat_str = request.args.get("lat", "").strip()
    lon_str = request.args.get("lon", "").strip()

    # ── Geolocation lookup (Bug #2 fix) ──────────────────────────────────────
    if lat_str and lon_str:
        try:
            lat = float(lat_str)
            lon = float(lon_str)
        except ValueError:
            return jsonify({"error": "Invalid lat/lon values"}), 400

        coord_key = f"coords_{lat:.2f}_{lon:.2f}"
        cached = _weather_cache.get(coord_key)
        if cached and cached["expires_at"] > _time.time():
            logger.info(f"Weather cache hit (coords): {coord_key}")
            return jsonify({**cached["data"], "cached": True})

        try:
            weather  = get_weather_by_coords(lat, lon)
            insights = generate_farming_insights(weather)
            payload  = {**weather, "insights": insights}
            _weather_cache[coord_key] = {"data": payload, "expires_at": _time.time() + 1800}
            return jsonify(payload)
        except Exception as e:
            logger.error(f"Weather coords API error: {e}")
            return jsonify({"error": str(e)}), 500

    # ── City lookup (original) ────────────────────────────────────────────────
    city     = request.args.get("city", "Bhopal").strip()
    city_key = city.lower()

    cached = _weather_cache.get(city_key)
    if cached and cached["expires_at"] > _time.time():
        logger.info(f"Weather cache hit: {city_key}")
        return jsonify({**cached["data"], "cached": True})

    try:
        weather  = get_weather(city)
        insights = generate_farming_insights(weather)
        payload  = {**weather, "insights": insights}
        _weather_cache[city_key] = {"data": payload, "expires_at": _time.time() + 1800}
        return jsonify(payload)
    except Exception as e:
        logger.error(f"Weather API error: {e}")
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Phase 1E — Disease Spread Risk Analysis
# ---------------------------------------------------------------------------

@app.route("/api/risk-analysis", methods=["POST"])
def risk_analysis_api():
    """
    POST /api/risk-analysis
    Body: { disease, humidity, temperature }
    Returns: { risk, score, reason, category }
    Authentication: Optional.
    """
    try:
        data        = request.get_json() or {}
        disease     = data.get("disease", "Unknown")
        humidity    = float(data.get("humidity", 60))
        temperature = float(data.get("temperature", 28))

        result = risk_engine.analyse_risk(disease, humidity, temperature)
        return jsonify({**result, "success": True})

    except Exception as e:
        logger.error(f"Risk analysis API error: {e}")
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# PDF Report Generation Route
# ---------------------------------------------------------------------------

@app.route("/generate-report", methods=["POST"])
def generate_report():
    """
    POST /generate-report
    Generates a premium professional PDF report for a given scan.
    Authentication: Optional JWT (used for future report personalization).

    Accepts multipart/form-data:
      - file         (optional)  : re-uploaded image, OR
      - data         (required)  : JSON string with scan fields
      - image_data_url (optional): base64 data URL from scan result

    Returns: application/pdf binary stream
    """
    try:
        if request.is_json:
            data = request.get_json() or {}
            image_bytes = None
        else:
            raw_data = request.form.get("data", "{}")
            try:
                data = json.loads(raw_data)
            except json.JSONDecodeError:
                data = {}

            image_bytes = None
            if "file" in request.files:
                image_bytes = request.files["file"].read()

        disease_name    = data.get("disease_name",  "Unknown Disease")
        confidence      = float(data.get("confidence", 0))
        symptoms        = data.get("symptoms",     "No symptom data available.")
        treatment       = data.get("treatment",    "No treatment data available.")
        prevention      = data.get("prevention",   "No prevention data available.")
        severity        = data.get("severity",     "LOW")
        is_healthy      = bool(data.get("is_healthy", False))
        scan_id         = data.get("scan_id")  or data.get("id")
        db_id           = data.get("db_id")    or data.get("id")
        filename        = data.get("filename")
        plant_type      = data.get("plant_type")
        image_data_url  = data.get("image_data_url") or data.get("imageDataUrl")

        logger.info(f"Generating PDF report for: {disease_name} | confidence: {confidence}")

        pdf_bytes = pdf_generator.generate_pdf_report(
            disease_name    = disease_name,
            confidence      = confidence,
            symptoms        = symptoms,
            treatment       = treatment,
            prevention      = prevention,
            severity        = severity,
            is_healthy      = is_healthy,
            scan_id         = scan_id,
            db_id           = db_id,
            image_bytes     = image_bytes,
            image_data_url  = image_data_url,
            filename        = filename,
            plant_type      = plant_type,
        )

        safe_name = (disease_name or "Report").replace(" ", "_")[:40]
        pdf_filename = f"AgroAI_Report_{safe_name}.pdf"

        return send_file(
            BytesIO(pdf_bytes),
            mimetype="application/pdf",
            as_attachment=True,
            download_name=pdf_filename,
        )

    except Exception as e:
        logger.error(f"PDF generation error: {e}")
        return jsonify({"error": f"PDF generation failed: {str(e)}"}), 500


@app.route("/chat", methods=["POST"])
def chat():
    """
    POST /chat
    Intelligent chatbot with three-layer response strategy:
      Layer 1 – Local DB (disease_info.json) for factual keyword queries → no API call
      Layer 2 – Gemini 1.5 Flash with full chat history + retry logic
      Layer 3 – Fallback to local DB (or friendly error) if Gemini fails
    """
    data = request.json or {}

    user_message       = (data.get("message") or "").strip()
    session_id         = data.get("session_id", "default_session")
    disease_context    = data.get("disease",    "a plant disease")
    symptoms_context   = data.get("symptoms",   "N/A")
    treatment_context  = data.get("treatment",  "N/A")
    prevention_context = data.get("prevention", "N/A")

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    logger.info(f"[{session_id}] User: {user_message!r} | Disease: {disease_context!r}")

    # Layer 1: Answer from local disease_info.json (zero API cost)
    matching_key = next(
        (k for k in disease_info_db
         if disease_context.lower() in k.lower() or k.lower() in disease_context.lower()),
        None
    )

    local_answer = build_local_response(matching_key, user_message) if matching_key else None

    FACTUAL_KEYWORDS = [
        "symptom", "sign", "treat", "cure", "prevent", "avoid",
        "cause", "why", "medicine", "spray", "fungicide", "what is", "how to"
    ]
    is_factual = any(kw in user_message.lower() for kw in FACTUAL_KEYWORDS)

    if local_answer and is_factual:
        logger.info(f"[{session_id}] Served from local DB (no API call).")
        return jsonify({"response": local_answer})

    # Layer 2: Gemini 1.5 Flash
    # Phase 2A: Inject language instruction if Hindi mode
    lang = (data.get("language") or "en").lower()
    hindi_prefix = (
        "CRITICAL INSTRUCTION: You MUST respond ENTIRELY in Hindi (हिंदी). "
        "Use simple, farmer-friendly Hindi language that rural farmers can understand. "
        "Do NOT use any English except for scientific/technical terms that have no Hindi equivalent.\n\n"
    ) if lang == "hi" else ""

    system_instruction = (
        hindi_prefix +
        "You are an intelligent and empathetic agriculture assistant specializing in plant diseases. "
        "Your mission: help farmers, gardeners, and students understand plant health, "
        "provide actionable treatment and prevention advice, and share general farming tips. "
        "Be friendly, concise, and use simple language. Format lists clearly with numbers or bullets.\n\n"
        "Current diagnosis context:\n"
        f"  • Disease detected : {disease_context}\n"
        f"  • Symptoms         : {symptoms_context}\n"
        f"  • Treatment        : {treatment_context}\n"
        f"  • Prevention       : {prevention_context}\n\n"
        "Build naturally on this context. Ask follow-up questions when appropriate. "
        "Do NOT repeat the context verbatim."
    )

    if session_id not in chat_sessions:
        chat_sessions[session_id] = []

    history = chat_sessions[session_id]

    try:
        reply_text, updated_history = call_gemini_with_retry(
            system_instruction=system_instruction,
            history=history,
            user_message=user_message,
        )

        chat_sessions[session_id] = updated_history

        logger.info(f"[{session_id}] Gemini responded OK.")
        return jsonify({"response": reply_text})

    except Exception as exc:
        err_str = str(exc)
        logger.error(f"[{session_id}] Gemini failed: {err_str}")

        # Layer 3: Graceful fallback
        if local_answer:
            fallback = (
                f"🌱 *Powered by Local Plant Expert Mode*\n\n"
                f"{local_answer.replace('🌱 *Powered by Local Plant Expert Mode*', '').strip()}\n\n"
                f"_You can also ask me about symptoms, treatment, or prevention and I'll answer instantly!_"
            )
            logger.info(f"[{session_id}] Falling back to local DB.")
            return jsonify({"response": fallback, "mode": "local"})

        disease_display = disease_context.replace("_", " ").strip() if disease_context else "your plant"
        msg = (
            f"🌱 *Powered by Local Plant Expert Mode*\n\n"
            f"I'm currently using my offline plant disease knowledge base.\n\n"
            f"For **{disease_display}**, here's what you can do:\n"
            f"• Ask me about **symptoms** — I'll tell you what to look for\n"
            f"• Ask about **treatment** — I'll suggest the best remedies\n"
            f"• Ask about **prevention** — I'll help you protect your crops\n"
            f"• Ask about **causes** — I'll explain why this happens\n\n"
            f"_Just type your question and I'll answer right away!_"
        )
        return jsonify({"response": msg, "mode": "local"})


# ===========================================================================
# Phase 3A — RAG Agriculture Expert Routes
# ===========================================================================

@app.route("/api/rag-chat", methods=["POST"])
def rag_chat():
    """
    POST /api/rag-chat
    Body: { "question": str, "language": "en"|"hi" }

    Flow:
      1. FAISS vector search → top-5 relevant agriculture document chunks
      2. Build grounded Gemini prompt with retrieved context
      3. Return answer + source citations
      4. Persist to chat_history (user-scoped if JWT present)
      5. Fallback to direct Gemini if no relevant chunks found

    Authentication: Optional JWT (chat history saved only when authenticated)
    """
    data     = request.get_json() or {}
    question = (data.get("question") or "").strip()
    language = (data.get("language") or "en").lower()

    if not question:
        return jsonify({"error": "question is required"}), 400

    if language not in ("en", "hi"):
        language = "en"

    user_id = get_optional_user_id()
    logger.info(f"[RAG] question={question!r:.60} | lang={language} | user={user_id!r}")

    # ── Call RAG service ──────────────────────────────────────────────────
    rag = _rag or get_rag_service()
    if rag is None:
        return jsonify({"error": "RAG service unavailable"}), 503

    try:
        result = rag.answer_agriculture_query(question=question, language=language)
    except Exception as e:
        logger.error(f"[RAG] Query failed: {e}")
        return jsonify({"error": "RAG query failed", "detail": str(e)}), 500

    answer  = result.get("answer", "")
    sources = result.get("sources", [])
    mode    = result.get("mode", "fallback")

    # ── Persist to MongoDB (best-effort) ─────────────────────────────────
    try:
        database.save_chat_message(
            user_id=user_id,
            question=question,
            answer=answer,
            sources=sources,
            mode=mode,
        )
    except Exception as db_err:
        logger.warning(f"[RAG] chat_history save failed (non-fatal): {db_err}")

    logger.info(f"[RAG] mode={mode} | sources={len(sources)}")
    return jsonify({"answer": answer, "sources": sources, "mode": mode})


@app.route("/api/rag-chat/history", methods=["GET"])
def rag_chat_history():
    """
    GET /api/rag-chat/history
    Returns last 10 RAG chat messages for the current user.
    Authentication: Optional JWT.
    """
    try:
        user_id = get_optional_user_id()
        limit   = min(int(request.args.get("limit", 10)), 50)
        history = database.get_chat_history(user_id=user_id, limit=limit)
        return jsonify({"success": True, "history": history})
    except Exception as e:
        logger.error(f"[RAG] history fetch error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/admin/rebuild-index", methods=["POST"])
@jwt_required()
def admin_rebuild_index():
    """
    POST /api/admin/rebuild-index  (JWT required)
    Hot-rebuilds the FAISS vector index from knowledge_base/ documents.
    Reloads the singleton so new documents are searchable immediately.
    No server restart required after this endpoint is called.
    """
    global _rag
    try:
        from ingest import run_ingestion
        logger.info("[ADMIN] Rebuilding FAISS index …")
        chunk_count = run_ingestion()          # re-ingests all documents
        if _rag:
            _rag.reload()                      # hot-reload vectorstore into singleton
        else:
            _rag = get_rag_service()
        logger.info(f"[ADMIN] Rebuild complete — {chunk_count} chunks indexed.")
        return jsonify({
            "success":        True,
            "chunks_indexed": chunk_count,
            "message":        f"Knowledge base rebuilt. {chunk_count} chunks now searchable.",
        })
    except Exception as e:
        logger.error(f"[ADMIN] Rebuild failed: {e}")
        return jsonify({"error": str(e)}), 500




# ===========================================================================
# Phase 4A — Admin Intelligence Dashboard Routes
# ===========================================================================

from functools import wraps

def admin_required(fn):
    """
    Decorator: requires a valid JWT where the user's role == 'admin'.

    FIX (Flask-JWT-Extended v4+):
        Do NOT stack @jwt_required() on the inner `wrapper` — that registers
        all admin endpoints under the same name 'wrapper', causing Flask's
        view-function registry to collide and the first registered route to
        shadow all others (manifests as 401 / 405 on later-registered routes).

        Correct pattern: call verify_jwt_in_request() inside the function body.
    """
    @wraps(fn)
    def wrapper(*args, **kwargs):
        from flask import request as _req
        from flask_jwt_extended import verify_jwt_in_request, get_jwt, get_jwt_identity

        # ── Verify JWT (raises exception → flask-jwt returns 401 if invalid) ─
        try:
            verify_jwt_in_request()
        except Exception as jwt_err:
            logger.error(f"[ADMIN] JWT verification failed for {_req.path}: {jwt_err}")
            return jsonify({"error": "Invalid or expired token. Please log in again."}), 401

        # ── [ADMIN DEBUG] Full auth trace ─────────────────────────────────
        auth_header = _req.headers.get("Authorization", "<missing>")
        decoded_jwt = get_jwt()            # full decoded payload dict
        user_id     = get_jwt_identity()   # the 'sub' claim (user_id_str)

        logger.warning(
            "\n"
            "=" * 60 + "\n"
            "[ADMIN DEBUG] Incoming admin request\n"
            f"  Endpoint         : {_req.path}\n"
            f"  Authorization    : {auth_header[:60]}...\n"
            f"  Decoded JWT sub  : {user_id!r}\n"
            f"  Full JWT payload : {decoded_jwt}\n"
        )

        # ── DB lookup ────────────────────────────────────────────────────
        user    = database.get_user_by_id(user_id)
        db_role = user.get("role") if user else "<user not found in DB>"
        allowed = bool(user and user.get("role") == "admin")

        logger.warning(
            f"  DB user found    : {bool(user)}\n"
            f"  DB role          : {db_role!r}\n"
            f"  Admin allowed    : {allowed}\n"
            "=" * 60
        )
        # ── End debug ──────────────────────────────────────────────────────

        if not user or user.get("role") != "admin":
            return jsonify({"error": "Admin access required. You do not have permission to view this resource."}), 403
        return fn(*args, **kwargs)
    return wrapper



@app.route("/api/admin/overview", methods=["GET"])
@admin_required
def admin_overview():
    """
    GET /api/admin/overview  (admin JWT required)
    Returns platform KPIs: total users, active users, total scans,
    total feedback, total RAG queries, avg scans per user.
    """
    try:
        data = database.get_admin_overview()
        return jsonify({"success": True, **data})
    except Exception as e:
        logger.error(f"[Admin] overview error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/admin/agriculture", methods=["GET"])
@admin_required
def admin_agriculture():
    """
    GET /api/admin/agriculture  (admin JWT required)
    Returns disease analytics, risk summaries, weather impact,
    crop issue distribution, severity breakdown.
    """
    try:
        data = database.get_admin_agriculture()
        return jsonify({"success": True, **data})
    except Exception as e:
        logger.error(f"[Admin] agriculture error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/admin/rag", methods=["GET"])
@admin_required
def admin_rag():
    """
    GET /api/admin/rag  (admin JWT required)
    Returns RAG analytics: total queries, top questions, top retrieved
    documents, category distribution, success rate, fallback %, trend.
    """
    try:
        data = database.get_admin_rag()
        return jsonify({"success": True, **data})
    except Exception as e:
        logger.error(f"[Admin] rag error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/admin/feedback", methods=["GET"])
@admin_required
def admin_feedback():
    """
    GET /api/admin/feedback  (admin JWT required)
    Returns feedback analytics: avg rating, distribution, latest 10,
    keyword frequency, 30-day trend.
    """
    try:
        data = database.get_admin_feedback()
        return jsonify({"success": True, **data})
    except Exception as e:
        logger.error(f"[Admin] feedback error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/admin/languages", methods=["GET"])
@admin_required
def admin_languages():
    """
    GET /api/admin/languages  (admin JWT required)
    Returns language analytics: EN vs HI usage %, most used language,
    30-day trend derived from chat_history Devanagari detection.
    """
    try:
        data = database.get_admin_languages()
        return jsonify({"success": True, **data})
    except Exception as e:
        logger.error(f"[Admin] languages error: {e}")
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Phase 4C — Admin User Management
# ---------------------------------------------------------------------------

@app.route("/api/admin/users", methods=["GET"])
@admin_required
def admin_users():
    """
    GET /api/admin/users  (admin JWT required)
    Returns all registered users enriched with platform usage statistics.

    Each user entry contains:
        _id, name, email, role, createdAt,
        totalScans, trackedPlants, feedbackCount, ragQueries,
        lastActivity, status (active/inactive)

    SECURITY: Passwords and tokens are NEVER returned (excluded in database layer).
    """
    try:
        users = database.get_admin_users()
        return jsonify({"success": True, "users": users, "total": len(users)})
    except Exception as e:
        logger.error(f"[Admin] users error: {e}")
        return jsonify({"error": str(e)}), 500




# ===========================================================================
# Phase 4B — Plant Progress Tracking API Routes
# ===========================================================================

@app.route("/api/plants/track", methods=["POST"])
@jwt_required()
def create_plant_track():
    """
    POST /api/plants/track  (JWT required)
    Body: { plantName: str }
    Returns: { success, plantId, plantName }
    Creates a new tracked plant record for the authenticated user.
    """
    try:
        user_id = get_jwt_identity()
        data    = request.get_json() or {}
        plant_name = (data.get("plantName") or "").strip()

        if not plant_name:
            return jsonify({"error": "plantName is required"}), 400
        if len(plant_name) > 80:
            return jsonify({"error": "plantName must be 80 characters or fewer"}), 400

        plant_id = database.create_plant_track(user_id=user_id, plant_name=plant_name)
        if not plant_id:
            return jsonify({"error": "Failed to create plant track. Database may be offline."}), 500

        logger.info(f"[Phase4B] /api/plants/track | user={user_id} | name={plant_name!r}")
        return jsonify({"success": True, "plantId": plant_id, "plantName": plant_name})

    except Exception as e:
        logger.error(f"[Phase4B] create_plant_track error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/plants/<plant_id>/scan", methods=["POST"])
@jwt_required()
def add_plant_scan(plant_id):
    """
    POST /api/plants/<plant_id>/scan  (JWT required)
    Body: {
        disease: str, confidence: float, riskScore: int,
        weatherSnapshot: dict, imageUrl: str
    }
    Appends a new scan snapshot to a tracked plant's history.
    Returns: { success, scanId }
    """
    try:
        user_id = get_jwt_identity()
        data    = request.get_json() or {}

        disease          = (data.get("disease") or "Unknown").strip()
        confidence       = float(data.get("confidence", 0))
        risk_score       = int(data.get("riskScore", 0)) if data.get("riskScore") is not None else 0
        weather_snapshot = data.get("weatherSnapshot") or {}
        image_url        = (data.get("imageUrl") or "").strip()

        scan_id = database.add_tracked_scan(
            plant_id         = plant_id,
            user_id          = user_id,
            disease          = disease,
            confidence       = confidence,
            risk_score       = risk_score,
            weather_snapshot = weather_snapshot,
            image_url        = image_url,
        )

        if not scan_id:
            return jsonify({"error": "Failed to save scan. Plant not found or DB offline."}), 404

        logger.info(f"[Phase4B] /api/plants/{plant_id}/scan | user={user_id} | disease={disease!r}")
        return jsonify({"success": True, "scanId": scan_id})

    except Exception as e:
        logger.error(f"[Phase4B] add_plant_scan error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/plants", methods=["GET"])
@jwt_required()
def get_plants():
    """
    GET /api/plants  (JWT required)
    Returns all tracked plants for the authenticated user, including
    latest disease, scan count, trend, and aggregate analytics.
    """
    try:
        user_id  = get_jwt_identity()
        plants   = database.get_user_plants(user_id)
        analytics = database.get_plant_analytics(user_id)
        logger.info(f"[Phase4B] /api/plants | user={user_id} | count={len(plants)}")
        return jsonify({"success": True, "plants": plants, "analytics": analytics})

    except Exception as e:
        logger.error(f"[Phase4B] get_plants error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/plants/<plant_id>/history", methods=["GET"])
@jwt_required()
def get_plant_history(plant_id):
    """
    GET /api/plants/<plant_id>/history  (JWT required)
    Returns the full scan history + chart-ready analytics for a single plant.
    Verifies ownership — 404 if plant does not belong to the requesting user.
    """
    try:
        user_id = get_jwt_identity()
        result  = database.get_plant_history(plant_id=plant_id, user_id=user_id)
        if not result.get("plant"):
            return jsonify({"error": "Plant not found or access denied"}), 404
        logger.info(f"[Phase4B] /api/plants/{plant_id}/history | user={user_id} | scans={len(result['scans'])}")
        return jsonify({"success": True, **result})

    except Exception as e:
        logger.error(f"[Phase4B] get_plant_history error: {e}")
        return jsonify({"error": str(e)}), 500


# ===========================================================================
# Start server
# ===========================================================================
# if __name__ == "__main__":
#     logger.info("Plant Disease API starting on http://127.0.0.1:5000")
#     app.run(debug=True, port=5000)

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )