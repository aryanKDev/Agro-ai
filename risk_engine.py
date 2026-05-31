"""
AgroAI — Disease Spread Risk Engine (Phase 1E)
===============================================
Predicts disease spread likelihood using weather conditions + disease category.
Loads disease_rules.json from the project root on first call.
"""

import os
import json
import logging

logger = logging.getLogger(__name__)

_BASE = os.path.dirname(os.path.abspath(__file__))
_RULES_PATH = os.path.join(_BASE, "disease_rules.json")
_disease_rules: dict = {}


def _load_rules():
    global _disease_rules
    if _disease_rules:
        return
    try:
        with open(_RULES_PATH, "r", encoding="utf-8") as f:
            _disease_rules = json.load(f)
        logger.info(f"Loaded {len(_disease_rules)} disease rules from disease_rules.json")
    except Exception as e:
        logger.error(f"Failed to load disease_rules.json: {e}")
        _disease_rules = {}


def _fuzzy_match(disease_name: str) -> dict | None:
    """
    Find a matching disease rule by fuzzy substring matching.
    Handles partial names like "Tomato Early Blight" matching
    "Tomato___Early_blight" in the rules file.
    """
    if not disease_name:
        return None

    normalised = disease_name.lower().replace(" ", "_").replace("-", "_")

    # Exact key match first
    if disease_name in _disease_rules:
        return _disease_rules[disease_name]

    # Substring match in both directions
    for key, val in _disease_rules.items():
        key_norm = key.lower().replace(" ", "_")
        if normalised in key_norm or key_norm in normalised:
            return val

    # Word-overlap fallback
    words = set(normalised.split("_"))
    best_match, best_score = None, 0
    for key, val in _disease_rules.items():
        key_words = set(key.lower().replace(" ", "_").split("_"))
        score = len(words & key_words)
        if score > best_score:
            best_score, best_match = score, val

    return best_match if best_score >= 2 else None


def analyse_risk(disease: str, humidity: float, temperature: float) -> dict:
    """
    Assess disease spread risk based on disease category and weather.

    Args:
        disease    : Disease display name (e.g. "Tomato Early Blight")
        humidity   : Current humidity %
        temperature: Current temperature °C

    Returns:
        {risk: HIGH|MEDIUM|LOW, score: 0-100, reason: str, category: str}
    """
    _load_rules()

    # Healthy plants → always LOW
    if "healthy" in disease.lower():
        return {
            "risk":     "LOW",
            "score":    5,
            "reason":   "Plant is healthy. No disease spread risk.",
            "category": "healthy",
        }

    rule = _fuzzy_match(disease)
    category = (rule or {}).get("category", "fungal")

    humidity     = float(humidity)
    temperature  = float(temperature)

    # ── FUNGAL ──────────────────────────────────────────────────────────────
    if category == "fungal":
        if humidity > 85 and 15 <= temperature <= 32:
            return {
                "risk":     "HIGH",
                "score":    min(95, int(60 + (humidity - 85) * 2.5 + max(0, 30 - abs(temperature - 23)) * 0.5)),
                "reason":   "High humidity and ideal temperature strongly favour fungal disease spread.",
                "category": category,
            }
        if humidity > 85:
            return {
                "risk":     "HIGH",
                "score":    80,
                "reason":   "Very high humidity creates optimal conditions for fungal growth.",
                "category": category,
            }
        if humidity > 70:
            score = int(45 + (humidity - 70) * 1.5)
            return {
                "risk":     "MEDIUM",
                "score":    score,
                "reason":   "Elevated humidity may accelerate fungal disease progression.",
                "category": category,
            }
        return {
            "risk":     "LOW",
            "score":    int(max(10, humidity * 0.3)),
            "reason":   "Current humidity is within safe range for fungal diseases.",
            "category": category,
        }

    # ── BACTERIAL ────────────────────────────────────────────────────────────
    if category == "bacterial":
        if humidity > 80 and temperature > 25:
            return {
                "risk":     "HIGH",
                "score":    min(90, int(55 + (humidity - 80) + (temperature - 25) * 1.5)),
                "reason":   "Warm, wet conditions strongly promote bacterial infection spread.",
                "category": category,
            }
        if humidity > 80:
            return {
                "risk":     "MEDIUM",
                "score":    65,
                "reason":   "High humidity can facilitate bacterial disease transmission via water splash.",
                "category": category,
            }
        if temperature > 30:
            return {
                "risk":     "MEDIUM",
                "score":    55,
                "reason":   "High temperatures can stress plants and increase bacterial susceptibility.",
                "category": category,
            }
        return {
            "risk":     "LOW",
            "score":    20,
            "reason":   "Conditions are not particularly favourable for bacterial spread.",
            "category": category,
        }

    # ── VIRAL ────────────────────────────────────────────────────────────────
    if category == "viral":
        if temperature > 30:
            return {
                "risk":     "HIGH",
                "score":    75,
                "reason":   "High temperatures increase insect vector activity, accelerating viral spread.",
                "category": category,
            }
        if temperature > 22:
            return {
                "risk":     "MEDIUM",
                "score":    50,
                "reason":   "Warm conditions may support insect vectors that transmit viral diseases.",
                "category": category,
            }
        return {
            "risk":     "LOW",
            "score":    25,
            "reason":   "Cooler temperatures limit insect vector activity.",
            "category": category,
        }

    # ── PEST ─────────────────────────────────────────────────────────────────
    if category == "pest":
        if temperature > 28 and humidity < 50:
            return {
                "risk":     "HIGH",
                "score":    82,
                "reason":   "Hot, dry conditions strongly favour pest population growth.",
                "category": category,
            }
        if temperature > 22:
            return {
                "risk":     "MEDIUM",
                "score":    48,
                "reason":   "Warm conditions may accelerate pest reproduction cycles.",
                "category": category,
            }
        return {
            "risk":     "LOW",
            "score":    20,
            "reason":   "Current conditions are not particularly favourable for pest outbreaks.",
            "category": category,
        }

    # ── UNKNOWN fallback ──────────────────────────────────────────────────────
    return {
        "risk":     "MEDIUM",
        "score":    50,
        "reason":   "Unable to classify disease. Monitor crops closely.",
        "category": "unknown",
    }
