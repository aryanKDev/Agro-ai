"""
AgroAI — Farming Insights Engine (Phase 1D)
============================================
Generates contextual smart farming recommendations based on current weather.
All rules are threshold-based — no external API needed.
"""

from typing import List, Dict


# ---------------------------------------------------------------------------
# Rule definitions
# ---------------------------------------------------------------------------
_RULES = [
    {
        "condition": lambda w: w.get("humidity", 0) > 85,
        "message":   "High humidity detected — fungal diseases may spread rapidly. Inspect crops frequently.",
        "level":     "danger",
        "icon":      "🍄",
    },
    {
        "condition": lambda w: 70 < w.get("humidity", 0) <= 85,
        "message":   "Moderate humidity — monitor for early signs of mildew or leaf spot.",
        "level":     "warning",
        "icon":      "💧",
    },
    {
        "condition": lambda w: w.get("rainChance", 0) > 70,
        "message":   "Heavy rainfall expected — avoid applying pesticides or fungicides today.",
        "level":     "danger",
        "icon":      "🌧️",
    },
    {
        "condition": lambda w: 40 < w.get("rainChance", 0) <= 70,
        "message":   "Moderate rain chance — wait for dry weather before foliar spraying.",
        "level":     "warning",
        "icon":      "🌦️",
    },
    {
        "condition": lambda w: w.get("temperature", 0) > 35,
        "message":   "High temperature alert — increase irrigation frequency to prevent heat stress.",
        "level":     "danger",
        "icon":      "🌡️",
    },
    {
        "condition": lambda w: 28 < w.get("temperature", 0) <= 35,
        "message":   "Warm conditions — ideal for bacterial growth. Ensure proper spacing for airflow.",
        "level":     "warning",
        "icon":      "☀️",
    },
    {
        "condition": lambda w: w.get("windSpeed", 0) > 25,
        "message":   "Strong winds detected — delay all spraying operations to prevent drift.",
        "level":     "danger",
        "icon":      "🌬️",
    },
    {
        "condition": lambda w: 15 < w.get("windSpeed", 0) <= 25,
        "message":   "Moderate wind — spray early morning or late evening for best efficacy.",
        "level":     "warning",
        "icon":      "💨",
    },
    {
        "condition": lambda w: (
            w.get("humidity", 0) <= 70
            and w.get("rainChance", 0) <= 40
            and w.get("temperature", 0) <= 30
            and w.get("windSpeed", 0) <= 15
        ),
        "message":   "Ideal farming conditions — good day for spraying, harvesting, and field operations.",
        "level":     "success",
        "icon":      "✅",
    },
]


def generate_farming_insights(weather: dict) -> List[Dict]:
    """
    Evaluate weather dict against rule set and return matching insights.

    Args:
        weather: dict with keys temperature, humidity, rainChance, windSpeed

    Returns:
        List of insight dicts: [{message, level, icon}, ...]
        level ∈ {success, warning, danger}
    """
    insights = []
    for rule in _RULES:
        try:
            if rule["condition"](weather):
                insights.append({
                    "message": rule["message"],
                    "level":   rule["level"],
                    "icon":    rule["icon"],
                })
        except Exception:
            continue

    # Always return at least one insight
    if not insights:
        insights.append({
            "message": "Weather data received. Monitor crops regularly.",
            "level":   "success",
            "icon":    "🌿",
        })

    return insights
