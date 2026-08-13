import os
import requests
import json

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

API_KEY = os.getenv("OPENWEATHER_API_KEY", "")


def get_weather(city):
    """Fetch weather data for a given city"""
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()

        return {
            "temperature": data["main"]["temp"],
            "humidity": data["main"]["humidity"],
            "pressure": data["main"]["pressure"],
            "feels_like": data["main"]["feels_like"],
            "weather": data["weather"][0]["description"],
            "wind_speed": data["wind"]["speed"],
            "clouds": data["clouds"]["all"]
        }
    except Exception as e:
        print(f"Weather API error: {e}")
        return {
            "temperature": "N/A",
            "humidity": "N/A",
            "pressure": "N/A",
            "feels_like": "N/A",
            "weather": "Unavailable",
            "wind_speed": "N/A",
            "clouds": "N/A"
        }


def get_weather_by_coordinates(lat, lon):
    """Fetch weather data for specific latitude/longitude"""
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()

        return {
            "city": data.get("name", "Unknown"),
            "temperature": round(data["main"]["temp"], 2),
            "humidity": data["main"]["humidity"],
            "pressure": data["main"]["pressure"],
            "feels_like": round(data["main"]["feels_like"], 2),
            "weather": data["weather"][0]["description"],
            "wind_speed": round(data["wind"]["speed"], 2),
            "clouds": data["clouds"]["all"],
            "rainfall": data.get("rain", {}).get("1h", 0)
        }
    except Exception as e:
        print(f"Weather API error for coordinates: {e}")
        return {
            "city": "Unknown",
            "temperature": 0,
            "humidity": 0,
            "pressure": 0,
            "feels_like": 0,
            "weather": "Unavailable",
            "wind_speed": 0,
            "clouds": 0,
            "rainfall": 0
        }


def get_weather_estimate_for_prediction(lat, lon, month_name=None):
    """
    Combine live current temperature with a seasonal rainfall estimate
    (derived from the monsoon forecast table) into the inputs
    predict_crop_yield() / predict_crop_yield_ml() need, instead of
    requiring the user to guess and type them in manually.

    Falls back to sensible Telangana seasonal averages if the live API
    call fails (e.g. no internet or missing OPENWEATHER_API_KEY).
    """
    import datetime

    if month_name is None:
        month_name = datetime.datetime.now().strftime("%B")

    live = get_weather_by_coordinates(lat, lon)
    monsoon = get_monsoon_forecast(month_name)

    # get_weather_by_coordinates() returns "Unavailable" as its weather
    # description (and temperature=0) when the live call fails -- use that
    # as the actual signal, since temperature=0 alone isn't reliably
    # distinguishable from a real freezing reading.
    live_available = live.get("weather") not in (None, "Unavailable") and live.get("city") != "Unknown"

    rainfall_category_mm = {
        "Very High": 1400,
        "High": 900,
        "Medium": 500,
        "Low": 250,
    }
    rainfall_estimate = rainfall_category_mm.get(monsoon["rainfall"], 500)

    temperature = live.get("temperature") if live_available else 28.0  # Telangana seasonal average fallback

    return {
        "temperature": round(float(temperature), 1),
        "rainfall_estimate": rainfall_estimate,
        "rainfall_category": monsoon["rainfall"],
        "humidity": live.get("humidity", "N/A"),
        "weather_description": live.get("weather", "Unavailable"),
        "month": month_name,
        "source": "live" if live_available else "fallback",
    }


def get_monsoon_forecast(month):
    """Return monsoon forecast based on month"""
    monsoon_data = {
        "June": {"rainfall": "High", "humidity": "Very High", "crops": ["Rice", "Sugarcane"]},
        "July": {"rainfall": "Very High", "humidity": "Very High", "crops": ["Rice", "Maize"]},
        "August": {"rainfall": "High", "humidity": "High", "crops": ["Rice", "Cotton"]},
        "September": {"rainfall": "Medium", "humidity": "High", "crops": ["Rice", "Pulses"]},
    }
    return monsoon_data.get(month, {"rainfall": "Low", "humidity": "Medium", "crops": ["Cotton", "Groundnut"]})