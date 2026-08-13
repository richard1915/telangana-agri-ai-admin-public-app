import os
import requests
import numpy as np
from datetime import datetime, timedelta

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ------------------------------------------------------------------
# Agromonitoring (by OpenWeatherMap) -- simpler alternative to Sentinel
# Hub: a single API key, no OAuth client, uses `requests` which is
# already a core dependency (no extra pip install needed). Free tier
# has a small polygon cap, so this is meant as an easy way to see live
# data working, not necessarily the final production data source -- see
# README "Live satellite data" section for the tradeoffs vs Sentinel Hub.
# ------------------------------------------------------------------
AGRO_API_KEY = os.getenv("AGRO_API_KEY", "")
AGRO_BASE_URL = "http://api.agromonitoring.com/agro/1.0"
AGRO_AUTHENTICATED = bool(AGRO_API_KEY)
_agro_polygon_cache = {}  # (lat_rounded, lon_rounded) -> polyid, so we don't recreate a polygon on every call


def _agro_get_or_create_polygon(latitude, longitude, buffer_deg=0.0015, timeout=15):
    """
    Agromonitoring needs a saved polygon (not just a point) to query.
    Creates a small square around the given point on first use and
    caches the resulting polyid for that rounded coordinate so repeat
    lookups (e.g. the precision-spraying grid) don't create a new
    polygon per call and eat into the free tier's polygon quota.
    """
    key = (round(latitude, 5), round(longitude, 5))
    if key in _agro_polygon_cache:
        return _agro_polygon_cache[key]

    coords = [[
        [longitude - buffer_deg, latitude - buffer_deg],
        [longitude + buffer_deg, latitude - buffer_deg],
        [longitude + buffer_deg, latitude + buffer_deg],
        [longitude - buffer_deg, latitude + buffer_deg],
        [longitude - buffer_deg, latitude - buffer_deg],
    ]]
    body = {
        "name": f"tg-agricol-{latitude:.5f}_{longitude:.5f}",
        "geo_json": {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": coords}},
    }
    resp = requests.post(f"{AGRO_BASE_URL}/polygons", params={"appid": AGRO_API_KEY}, json=body, timeout=timeout)
    resp.raise_for_status()
    polyid = resp.json()["id"]
    _agro_polygon_cache[key] = polyid
    return polyid


def get_agromonitoring_indices(latitude, longitude, lookback_days=30, timeout=15):
    """
    NDVI + NDWI (moisture proxy) for a point via Agromonitoring's
    Satellite Imagery Search API, using the most recent available scene
    in the lookback window. Agromonitoring doesn't expose a bare-soil
    index (BSI) the way Sentinel Hub's evalscript does, so callers
    should treat bsi=0.0 (neutral) from this source rather than a real
    measurement -- soil-type classification from this source leans more
    on NDVI/NDWI than BSI as a result.
    """
    if not AGRO_AUTHENTICATED:
        return None
    try:
        polyid = _agro_get_or_create_polygon(latitude, longitude, timeout=timeout)
        end = int(datetime.now().timestamp())
        start = int((datetime.now() - timedelta(days=lookback_days)).timestamp())
        resp = requests.get(
            f"{AGRO_BASE_URL}/image/search",
            params={"start": start, "end": end, "polyid": polyid, "appid": AGRO_API_KEY},
            timeout=timeout,
        )
        resp.raise_for_status()
        scenes = resp.json()
        if not scenes:
            return None
        # Most recent scene with usable stats.
        latest = sorted(scenes, key=lambda s: s.get("dt", 0), reverse=True)[0]
        stats_urls = latest.get("stats", {})

        def _mean_of(index_name):
            url = stats_urls.get(index_name)
            if not url:
                return None
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            return float(r.json().get("mean"))

        ndvi = _mean_of("ndvi")
        ndwi = _mean_of("ndwi")
        if ndvi is None:
            return None
        return {
            "ndvi": round(ndvi, 3),
            "ndmi": round(ndwi, 3) if ndwi is not None else round(ndvi * 0.6, 3),  # NDWI as an NDMI-equivalent moisture proxy
            "bsi": 0.0,  # not available from this provider -- neutral placeholder, not a measurement
            "source": "live",
            "provider": "agromonitoring",
        }
    except Exception as e:
        print(f"Agromonitoring lookup failed: {e}")
        return None


try:
    from sentinelhub import SHConfig, SentinelHubRequest, DataCollection, MimeType, CRS, BBox
    SENTINELHUB_AVAILABLE = True
    config = SHConfig()
    config.instance_id = os.getenv("SENTINEL_HUB_INSTANCE_ID", "")
    # The Process API (SentinelHubRequest, used below) authenticates via
    # OAuth client_id/client_secret, NOT instance_id -- instance_id alone
    # is a legacy field for older WMS/WCS-style requests and will not
    # authenticate the evalscript-based calls this module makes.
    #
    # As of the Copernicus Data Space Ecosystem (CDSE) migration, Sentinel
    # Hub's FREE tier is accessed through dataspace.copernicus.eu, not the
    # old apps.sentinel-hub.com. That also means the base URL and token
    # URL below are mandatory, not optional -- SHConfig defaults to the
    # legacy sentinel-hub.com endpoints, which a CDSE OAuth client cannot
    # authenticate against, so requests would fail even with valid
    # credentials if these two lines were left out.
    #
    # Setup (free): create an account at https://dataspace.copernicus.eu,
    # then from your profile icon open "Sentinel Hub" -> User Settings ->
    # OAuth Clients -> create a new client, and put its ID/secret below.
    config.sh_client_id = os.getenv("SENTINEL_HUB_CLIENT_ID", "")
    config.sh_client_secret = os.getenv("SENTINEL_HUB_CLIENT_SECRET", "")
    config.sh_base_url = os.getenv("SENTINEL_HUB_BASE_URL", "https://sh.dataspace.copernicus.eu")
    config.sh_token_url = os.getenv(
        "SENTINEL_HUB_TOKEN_URL",
        "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
    )
    SENTINELHUB_AUTHENTICATED = bool(config.sh_client_id and config.sh_client_secret)
except ImportError:
    # sentinelhub is an optional dependency (see requirements.txt). Without
    # it, every function below falls back to synthetic/placeholder data
    # instead of crashing the whole app on import.
    SENTINELHUB_AVAILABLE = False
    SENTINELHUB_AUTHENTICATED = False
    config = None


def get_bounding_box(latitude, longitude, buffer_km=2):
    """Convert lat/lon to bounding box with buffer"""
    # Approximate km to degrees (1 degree ≈ 111 km)
    buffer_deg = buffer_km / 111.0
    bbox = BBox(
        [longitude - buffer_deg, latitude - buffer_deg, 
         longitude + buffer_deg, latitude + buffer_deg],
        crs=CRS.WGS84
    )
    return bbox


def get_satellite_imagery(latitude, longitude, width=256, height=256):
    """Fetch true color and multispectral data from Sentinel-2"""
    try:
        bbox = get_bounding_box(latitude, longitude)
        
        # True color request (RGB for visualization)
        evalscript_rgb = """
        return [B04, B03, B02];
        """
        
        request_rgb = SentinelHubRequest(
            evalscript=evalscript_rgb,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=DataCollection.SENTINEL2_L2A,
                    time_interval=(datetime.now() - timedelta(days=30), datetime.now())
                )
            ],
            responses=[
                SentinelHubRequest.output_response("default", MimeType.PNG)
            ],
            bbox=bbox,
            size=[width, height],
            config=config
        )
        
        rgb_data = request_rgb.get_data()
        return rgb_data
    except Exception as e:
        print(f"Error fetching satellite imagery: {e}")
        return None


def calculate_soil_indices(latitude, longitude):
    """Calculate spectral indices for soil classification"""
    if not (SENTINELHUB_AVAILABLE and SENTINELHUB_AUTHENTICATED):
        # Try the simpler Agromonitoring source next, if configured.
        if AGRO_AUTHENTICATED:
            agro_result = get_agromonitoring_indices(latitude, longitude)
            if agro_result:
                return agro_result
        # No point attempting a Sentinel Hub request we know will fail
        # auth, and Agromonitoring is either unset or failed -- go
        # straight to the synthetic fallback and say so honestly.
        return generate_synthetic_indices(latitude, longitude)

    try:
        bbox = get_bounding_box(latitude, longitude)
        
        # Calculate NDVI, NDMI, and other indices
        evalscript = """
        var ndvi = (B08 - B04) / (B08 + B04);
        var ndmi = (B8A - B11) / (B8A + B11);
        var bsi = ((B11 + B04) - (B08 + B02)) / ((B11 + B04) + (B08 + B02));
        return [ndvi, ndmi, bsi];
        """
        
        request = SentinelHubRequest(
            evalscript=evalscript,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=DataCollection.SENTINEL2_L2A,
                    time_interval=(datetime.now() - timedelta(days=30), datetime.now())
                )
            ],
            responses=[
                SentinelHubRequest.output_response("default", MimeType.TIFF)
            ],
            bbox=bbox,
            size=[256, 256],
            config=config
        )
        
        data = request.get_data()
        
        # Calculate mean indices
        ndvi = float(np.mean(data[0]))
        ndmi = float(np.mean(data[1]))
        bsi = float(np.mean(data[2]))
        
        return {"ndvi": ndvi, "ndmi": ndmi, "bsi": bsi, "source": "live", "provider": "sentinel_hub"}
    except Exception as e:
        print(f"Error calculating indices: {e}")
        if AGRO_AUTHENTICATED:
            agro_result = get_agromonitoring_indices(latitude, longitude)
            if agro_result:
                return agro_result
        # Return default synthetic values based on location for demo
        return generate_synthetic_indices(latitude, longitude)


def generate_synthetic_indices(latitude, longitude):
    """Generate synthetic indices for demonstration when API fails"""
    import random
    # Use high-precision coordinates in the seed so nearby grid points
    # (e.g. Stage 2's precision-spraying grid, ~0.001 degrees apart) get
    # distinct values instead of collapsing onto the same seed. Truncating
    # to int(lat*100 + lon*100) loses everything below ~1km of precision.
    random.seed(int(round(latitude * 1_000_000)) * 2_000_003 + int(round(longitude * 1_000_000)))
    
    return {
        "ndvi": round(random.uniform(0.2, 0.7), 3),
        "ndmi": round(random.uniform(0.1, 0.6), 3),
        "bsi": round(random.uniform(-0.2, 0.4), 3),
        "source": "synthetic",
        "provider": "synthetic",
    }


def classify_soil_type(ndvi, ndmi, bsi):
    """Classify soil type based on spectral indices"""
    
    if ndvi < -0.1:
        soil_type = "Water/Barren"
        confidence = "High"
    elif ndvi < 0.2:
        soil_type = "Barren/Built-up"
        confidence = "High"
    elif ndvi < 0.4:
        if bsi > 0.3:
            soil_type = "Red Soil"
            confidence = "Medium"
        elif bsi < -0.1:
            soil_type = "Black Soil"
            confidence = "Medium"
        else:
            soil_type = "Laterite Soil"
            confidence = "Low"
    else:
        if ndmi > 0.4:
            soil_type = "Alluvial (High moisture)"
            confidence = "Medium"
        else:
            soil_type = "Red Soil (Cultivated)"
            confidence = "Medium"
    
    return soil_type, confidence


def get_soil_type_for_location(latitude, longitude):
    """Get soil type for a specific location using Sentinel Hub"""
    try:
        indices = calculate_soil_indices(latitude, longitude)
        if indices:
            soil_type, confidence = classify_soil_type(
                indices["ndvi"], 
                indices["ndmi"], 
                indices["bsi"]
            )
            return {
                "soil_type": soil_type,
                "confidence": confidence,
                "ndvi": round(indices["ndvi"], 3),
                "ndmi": round(indices["ndmi"], 3),
                "bsi": round(indices["bsi"], 3),
                "source": indices.get("source", "synthetic"),
                "provider": indices.get("provider", "synthetic"),
            }
    except Exception as e:
        print(f"Error determining soil type: {e}")
    
    return None


def get_soil_moisture():
    """Get NDVI for Telangana region"""
    try:
        bbox = BBox([78.0, 17.0, 79.0, 18.0], crs=CRS.WGS84)
        
        evalscript = """
        return [ (B08 - B04) / (B08 + B04) ];
        """
        
        request = SentinelHubRequest(
            evalscript=evalscript,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=DataCollection.SENTINEL2_L2A
                )
            ],
            responses=[
                SentinelHubRequest.output_response("default", MimeType.TIFF)
            ],
            bbox=bbox,
            size=[512, 512],
            config=config
        )
        
        data = request.get_data()
        avg_ndvi = float(np.mean(data))
        
        return avg_ndvi
    except Exception as e:
        print(f"Error getting soil moisture: {e}")
        return 0.45