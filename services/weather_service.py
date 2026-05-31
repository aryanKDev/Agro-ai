"""
AgroAI — Weather Service (Phase 1C + Bug Fix)
==============================================
Fetches real-time agricultural weather data from OpenWeatherMap API.
Falls back to a simulated dataset if the API key is missing or quota exceeded.

Supports:
  - get_weather(city)           — city name lookup
  - get_weather_by_coords(lat, lon) — GPS coordinate lookup (Bug #2 fix)
"""

import os
import logging
import requests as http_requests
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")
OWM_BASE = "https://api.openweathermap.org/data/2.5/weather"

# ---------------------------------------------------------------------------
# Fallback simulation data  (used when API key absent or network fails)
# ---------------------------------------------------------------------------
_FALLBACK_CITIES = {
    "bhopal":     {"temperature": 31, "humidity": 78, "rainChance": 55, "windSpeed": 12, "condition": "Partly Cloudy"},
    "delhi":      {"temperature": 36, "humidity": 55, "rainChance": 10, "windSpeed": 18, "condition": "Hazy"},
    "mumbai":     {"temperature": 30, "humidity": 88, "rainChance": 70, "windSpeed": 22, "condition": "Cloudy"},
    "bangalore":  {"temperature": 25, "humidity": 68, "rainChance": 30, "windSpeed": 10, "condition": "Clear"},
    "chennai":    {"temperature": 33, "humidity": 82, "rainChance": 40, "windSpeed": 14, "condition": "Humid"},
    "kolkata":    {"temperature": 32, "humidity": 85, "rainChance": 65, "windSpeed": 16, "condition": "Cloudy"},
    "hyderabad":  {"temperature": 28, "humidity": 70, "rainChance": 20, "windSpeed":  9, "condition": "Clear"},
    "pune":       {"temperature": 29, "humidity": 72, "rainChance": 35, "windSpeed": 11, "condition": "Partly Cloudy"},
    "jaipur":     {"temperature": 38, "humidity": 40, "rainChance":  5, "windSpeed": 20, "condition": "Sunny"},
    "lucknow":    {"temperature": 34, "humidity": 65, "rainChance": 25, "windSpeed": 13, "condition": "Hazy"},
}

_DEFAULT_FALLBACK = {"temperature": 30, "humidity": 75, "rainChance": 40, "windSpeed": 15, "condition": "Partly Cloudy"}


def _rain_chance_from_owm(data: dict) -> int:
    """Estimate rain probability from OWM response (pop is in forecast, not current)."""
    weather_id = data.get("weather", [{}])[0].get("id", 800)
    if weather_id < 300:   return 95  # thunderstorm
    if weather_id < 500:   return 80  # drizzle
    if weather_id < 600:   return 85  # rain
    if weather_id < 700:   return 30  # snow
    if weather_id < 800:   return 20  # atmosphere
    if weather_id == 800:  return 5   # clear
    if weather_id < 804:   return 25  # few/scattered clouds
    return 40  # broken/overcast


def _normalise_owm(d: dict, coord_based: bool = False) -> dict:
    """Convert a raw OWM current-weather response to AgroAI's normalised format."""
    return {
        "city":        d.get("name", "Unknown"),
        "temperature": round(d["main"]["temp"]),
        "humidity":    d["main"]["humidity"],
        "rainChance":  _rain_chance_from_owm(d),
        "windSpeed":   round(d["wind"]["speed"] * 3.6),   # m/s → km/h
        "condition":   d["weather"][0]["description"].title(),
        "source":      "live",
        "coordBased":  coord_based,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_weather(city: str = "Bhopal") -> dict:
    """
    Fetch weather for a city name.
    Returns a normalised dict always containing:
      city, temperature(°C), humidity(%), rainChance(%), windSpeed(km/h), condition, source
    """
    city_clean = city.strip()

    # ── Try live OpenWeatherMap API ──────────────────────────────────────────
    if OPENWEATHER_API_KEY:
        try:
            resp = http_requests.get(OWM_BASE, params={
                "q":     city_clean,
                "appid": OPENWEATHER_API_KEY,
                "units": "metric",
            }, timeout=8)

            if resp.status_code == 200:
                result = _normalise_owm(resp.json(), coord_based=False)
                logger.info(f"OWM live weather for '{city_clean}': {result['temperature']}°C")
                return result
            elif resp.status_code == 404:
                logger.warning(f"City not found in OWM: {city_clean}")
            else:
                logger.warning(f"OWM returned {resp.status_code} for {city_clean}")

        except Exception as exc:
            logger.warning(f"OWM API request failed: {exc}")

    # ── Fallback to simulation ───────────────────────────────────────────────
    logger.info(f"Using simulated weather for: {city_clean}")
    fallback = _FALLBACK_CITIES.get(city_clean.lower(), _DEFAULT_FALLBACK).copy()
    fallback["city"]       = city_clean.title()
    fallback["source"]     = "simulated"
    fallback["coordBased"] = False
    return fallback


def get_weather_by_coords(lat: float, lon: float) -> dict:
    """
    Fetch weather by GPS coordinates (Bug #2 — browser geolocation support).
    Returns same normalised dict as get_weather().
    Falls back to Bhopal simulation if API key is missing or network fails.
    """
    # ── Try live OpenWeatherMap API ──────────────────────────────────────────
    if OPENWEATHER_API_KEY:
        try:
            resp = http_requests.get(OWM_BASE, params={
                "lat":   lat,
                "lon":   lon,
                "appid": OPENWEATHER_API_KEY,
                "units": "metric",
            }, timeout=8)

            if resp.status_code == 200:
                result = _normalise_owm(resp.json(), coord_based=True)
                logger.info(f"OWM live weather for coords {lat},{lon}: {result['city']} {result['temperature']}°C")
                return result
            else:
                logger.warning(f"OWM coords lookup returned {resp.status_code}")

        except Exception as exc:
            logger.warning(f"OWM coords API request failed: {exc}")

    # ── Fallback ──────────────────────────────────────────────────────────────
    logger.info(f"Using simulated weather fallback for coords: {lat},{lon}")
    fallback = _FALLBACK_CITIES["bhopal"].copy()
    fallback["city"]       = "Your Location"
    fallback["source"]     = "simulated"
    fallback["coordBased"] = True
    return fallback
