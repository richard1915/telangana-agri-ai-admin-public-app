"""
Geocoding -- address to coordinates
=====================================

Uses OpenStreetMap's Nominatim API (https://nominatim.org) to convert a
farmer's village/district address into latitude/longitude, so the map
can be used without requiring anyone to already know exact coordinates.

Free, no API key required. Nominatim's usage policy requires a
descriptive User-Agent header and asks for at most ~1 request/second for
casual use -- both are respected here. For anything beyond light demo
use, consider self-hosting Nominatim or using a paid geocoding provider.
"""

import requests

NOMINATIM_ENDPOINT = "https://nominatim.openstreetmap.org/search"
USER_AGENT = "TelanganaAgricultureAI/1.0 (student project)"


def geocode_address(village="", district="", state="Telangana", country="India"):
    """
    Convert an address into (latitude, longitude). Returns
    {"available": True, "latitude": ..., "longitude": ..., "display_name": ...}
    on success, or {"available": False, "error": ...} if nothing matched
    or the request failed.
    """
    parts = [p.strip() for p in [village, district, state, country] if p and p.strip()]
    if not parts:
        return {"available": False, "error": "No address provided."}

    query = ", ".join(parts)
    try:
        response = requests.get(
            NOMINATIM_ENDPOINT,
            params={"q": query, "format": "json", "limit": 1, "countrycodes": "in"},
            headers={"User-Agent": USER_AGENT},
            timeout=10,
        )
        response.raise_for_status()
        results = response.json()
        if not results:
            return {"available": False, "error": f"No location found for '{query}'. Try a broader address (e.g. just district + state)."}

        result = results[0]
        return {
            "available": True,
            "latitude": float(result["lat"]),
            "longitude": float(result["lon"]),
            "display_name": result.get("display_name", query),
            "query": query,
        }
    except Exception as e:
        print(f"Geocoding error: {e}")
        return {"available": False, "error": f"Could not geocode '{query}': {e}"}
