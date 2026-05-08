"""
Plant Disease Detection System - Production-Ready Flask Backend
================================================================
Author notes:
  - Uses google-genai (new official SDK) with gemini-1.5-flash
  - disease_info.json is queried FIRST for factual questions (no API call)
  - Gemini is called only for open-ended / conversational queries
  - 1 retry after 5 s on quota / server errors
  - Full graceful fallback to local DB if Gemini is unavailable
  - python-dotenv for safe API key loading
  - Structured logging for easy demo/viva monitoring
"""

import os
import json
import time
import logging
import numpy as np
import tensorflow as tf
from PIL import Image
from io import BytesIO
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
import requests

# ---------------------------------------------------------------------------
# New official Gemini SDK  (pip install google-genai)
# ---------------------------------------------------------------------------
from google import genai
from google.genai import types

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
# Load .env and configure Gemini client
# ---------------------------------------------------------------------------
load_dotenv()  # reads GOOGLE_API_KEY from the .env file in the project root

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    logger.warning("GOOGLE_API_KEY not set in .env – chatbot will run in offline mode only.")

# Stable free-tier model for the new google-genai SDK.
# gemini-2.0-flash-lite has the highest free quota (generous RPM/RPD limits).
# The local-DB-first strategy in the chat route minimises API calls further.
GEMINI_MODEL = "gemini-1.5-flash"


# Create client once at startup (safe if key is None – handled in chat route)
gemini_client = genai.Client(api_key=GOOGLE_API_KEY) if GOOGLE_API_KEY else None
if gemini_client:
    logger.info(f"Gemini client ready | model: {GEMINI_MODEL}")

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder=BASE_DIR, template_folder=BASE_DIR)
CORS(app)

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

# Class names order must match what the model was trained on
CLASS_NAMES = list(disease_info_db.keys())

# In-memory conversational session store  { session_id -> list[Content] }
chat_sessions: dict = {}

# ===========================================================================
# Helper utilities
# ===========================================================================

def model_prediction(image_bytes: bytes) -> str:
    """Run Keras model and return the predicted class name."""
    img = Image.open(BytesIO(image_bytes))
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize((128, 128))
    arr = tf.keras.preprocessing.image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    preds = model.predict(arr)
    idx = int(np.argmax(preds))
    return CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else "Unknown_Class"


def _fmt_bullets(text: str) -> str:
    """Convert numbered list lines into clean bullet lines."""
    lines = text.strip().split("\n")
    out = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Already starts with a number "1. ..."
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

    # --- Intent detection ---
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
        # Generic full overview
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
    Retries once (after retry_delay seconds) on quota / server errors.
    Returns (reply_text, updated_history_list) or raises on failure.
    """
    if not gemini_client:
        raise RuntimeError("Gemini client not initialised (no API key).")

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            # Reconstruct the chat with full history on each call
            # (google-genai SDK creates stateless chats via Client)
            chat = gemini_client.chats.create(
                model=GEMINI_MODEL,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    # Keep responses focused and fast
                    max_output_tokens=1024,
                    temperature=0.7,
                ),
                history=history  # list of Content objects from previous turns
            )

            response = chat.send_message(user_message)

            # Guard: empty / blocked response
            reply = getattr(response, "text", None)
            if not reply:
                raise ValueError("Gemini returned an empty response.")

            return reply, list(chat.get_history())

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

        prediction_key = model_prediction(image_bytes)
        logger.info(f"Prediction result: {prediction_key}")

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

        return jsonify({
            "prediction":   prediction_key,
            "disease_name": disease_name,
            "symptoms":     symptoms,
            "treatment":    treatment,
            "prevention":   prevention,
        })

    except requests.exceptions.RequestException as e:
        logger.error(f"Image download error: {e}")
        return jsonify({"error": f"Could not download image: {str(e)}"}), 500
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


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

    # ------------------------------------------------------------------
    # Layer 1: Answer from local disease_info.json (zero API cost)
    # ------------------------------------------------------------------
    # Match the disease context to a DB key
    matching_key = next(
        (k for k in disease_info_db
         if disease_context.lower() in k.lower() or k.lower() in disease_context.lower()),
        None
    )

    local_answer = build_local_response(matching_key, user_message) if matching_key else None

    # Serve locally for clearly factual queries – saves API quota
    FACTUAL_KEYWORDS = [
        "symptom", "sign", "treat", "cure", "prevent", "avoid",
        "cause", "why", "medicine", "spray", "fungicide", "what is", "how to"
    ]
    is_factual = any(kw in user_message.lower() for kw in FACTUAL_KEYWORDS)

    if local_answer and is_factual:
        logger.info(f"[{session_id}] Served from local DB (no API call).")
        return jsonify({"response": local_answer})

    # ------------------------------------------------------------------
    # Layer 2: Gemini 1.5 Flash – open-ended / conversational queries
    # ------------------------------------------------------------------
    # Rich system prompt (prompt engineering for agri-assistant persona)
    system_instruction = (
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

    # Retrieve or initialise session history (list of Content objects)
    if session_id not in chat_sessions:
        chat_sessions[session_id] = []

    history = chat_sessions[session_id]  # list[types.Content] or list[dict]

    try:
        reply_text, updated_history = call_gemini_with_retry(
            system_instruction=system_instruction,
            history=history,
            user_message=user_message,
        )

        # Persist history as plain dicts (JSON-serialisable) for future calls
        chat_sessions[session_id] = [
            {"role": m.role, "parts": [p.text for p in m.parts]}
            for m in updated_history
        ]

        logger.info(f"[{session_id}] Gemini responded OK.")
        return jsonify({"response": reply_text})

    except Exception as exc:
        err_str = str(exc)
        logger.error(f"[{session_id}] Gemini failed: {err_str}")

        # ------------------------------------------------------------------
        # Layer 3: Graceful fallback — NEVER expose API errors to the user
        # ------------------------------------------------------------------
        if local_answer:
            fallback = (
                f"🌱 *Powered by Local Plant Expert Mode*\n\n"
                f"{local_answer.replace('🌱 *Powered by Local Plant Expert Mode*', '').strip()}\n\n"
                f"_You can also ask me about symptoms, treatment, or prevention and I'll answer instantly!_"
            )
            logger.info(f"[{session_id}] Falling back to local DB.")
            return jsonify({"response": fallback, "mode": "local"})

        # No local answer either — give a warm, helpful generic response
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
# Start server
# ===========================================================================
if __name__ == "__main__":
    logger.info("Plant Disease API starting on http://127.0.0.1:5000")
    app.run(debug=True, port=5000)