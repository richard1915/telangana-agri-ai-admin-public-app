"""
Field Health Imaging
=====================

Turns the same satellite grid used in Stage 2 into six distinct
diagnostic layers, each renderable as a heatmap overlay on the satellite
basemap:

  1. Crop stress            - from NDVI (vegetation vigor)
  2. Heat signature          - simulated canopy thermal anomaly
  3. Water stress            - from NDMI (moisture index)
  4. Nitrogen deficiency     - proxy from NDVI + bare-soil index
  5. Areas of over-application - zones with low stress but a farm
                                  already applying a high chemical dose
  6. Areas needing additional nutrients - zones with high stress and
                                  measured NPK below the crop's typical need

Layers 1-4 are per-zone (derived from satellite/drone-style readings).
Layers 5-6 combine per-zone stress with the farmer's actual entered soil
data (nitrogen/phosphorus/potassium, already-applied chemical dose) so
they reflect this specific farm rather than being purely synthetic.
"""

import random

from .satellite_api import get_soil_type_for_location
from .precision_spray import generate_field_grid

# Typical NPK a healthy field needs (kg/acre) -- used only as a rough
# reference band for the nutrient-deficiency layer, not a precise agronomic
# standard.
REFERENCE_NPK = {"nitrogen": 200.0, "phosphorus": 50.0, "potassium": 200.0}


def _clip01(x):
    return max(0.0, min(1.0, x))


def _thermal_anomaly(lat, lon, ndvi):
    rng = random.Random(int(lat * 1_000_000) ^ int(lon * 1_000_000))
    base = (1 - ndvi) * 6.0
    return round(base + rng.uniform(-0.5, 0.5), 2)


def generate_field_health_layers(shape_points, center_lat=None, center_lon=None,
                                  grid_resolution=6, soil_info=None):
    """
    Build the six diagnostic layers across a grid covering the field.

    soil_info: the farmer's Soil Information dict (nitrogen, phosphorus,
    potassium, soil_chemical_dose) if available, used to make the
    over-application / nutrient-deficiency layers farm-specific rather
    than purely synthetic.
    """
    grid_points = generate_field_grid(
        shape_points, center_lat=center_lat, center_lon=center_lon,
        grid_resolution=grid_resolution,
    )

    soil_info = soil_info or {}
    applied_dose = float(soil_info.get("soil_chemical_dose", 0.0))
    nitrogen = float(soil_info.get("nitrogen", REFERENCE_NPK["nitrogen"]))
    phosphorus = float(soil_info.get("phosphorus", REFERENCE_NPK["phosphorus"]))
    potassium = float(soil_info.get("potassium", REFERENCE_NPK["potassium"]))
    npk_ratio = (
        (nitrogen / REFERENCE_NPK["nitrogen"]) * 0.5
        + (phosphorus / REFERENCE_NPK["phosphorus"]) * 0.25
        + (potassium / REFERENCE_NPK["potassium"]) * 0.25
    )
    npk_ratio = _clip01(npk_ratio)
    # High applied dose (relative to a "generous" 60 kg/acre reference)
    # raises the chance any low-stress zone is flagged as over-applied.
    dose_pressure = _clip01(applied_dose / 60.0)

    cells = []
    for zone_id, (lat, lon) in enumerate(grid_points, start=1):
        info = get_soil_type_for_location(lat, lon)
        if not info:
            continue

        ndvi = info["ndvi"]
        ndmi = info["ndmi"]
        bsi = info["bsi"]

        crop_stress = _clip01(1 - (ndvi + 0.2) / 1.0)          # low NDVI -> high stress
        heat_signature = _clip01(_thermal_anomaly(lat, lon, ndvi) / 6.0)
        water_stress = _clip01(1 - (ndmi + 0.2) / 0.9)          # low NDMI -> high water stress
        nitrogen_deficiency = _clip01(0.6 * crop_stress + 0.4 * _clip01((bsi + 0.2) / 0.6))

        # Over-application: this zone looks healthy (low stress) but the
        # farm as a whole is already applying a high chemical dose.
        over_application = _clip01((1 - crop_stress) * dose_pressure)

        # Needs additional nutrients: zone is stressed AND the farm's
        # measured NPK is below the reference band.
        nutrient_need = _clip01(crop_stress * (1 - npk_ratio))

        cells.append({
            "zone_id": zone_id,
            "lat": lat,
            "lon": lon,
            "source": info.get("source", "synthetic"),
            "crop_stress": round(crop_stress, 3),
            "heat_signature": round(heat_signature, 3),
            "water_stress": round(water_stress, 3),
            "nitrogen_deficiency": round(nitrogen_deficiency, 3),
            "over_application": round(over_application, 3),
            "nutrient_need": round(nutrient_need, 3),
        })

    return cells


LAYER_DEFINITIONS = {
    "Crop stress": {
        "key": "crop_stress",
        "gradient": {0.2: "#2ecc71", 0.5: "#f1c40f", 0.8: "#e67e22", 1.0: "#e74c3c"},
        "description": "Derived from NDVI. Red = likely stressed vegetation.",
    },
    "Heat signatures": {
        "key": "heat_signature",
        "gradient": {0.2: "#3498db", 0.5: "#f1c40f", 0.8: "#e67e22", 1.0: "#c0392b"},
        "description": "Simulated canopy thermal anomaly. Warmer colors = hotter canopy (often linked to water stress).",
    },
    "Water stress": {
        "key": "water_stress",
        "gradient": {0.2: "#2980b9", 0.5: "#1abc9c", 0.8: "#f39c12", 1.0: "#8e44ad"},
        "description": "Derived from NDMI (moisture index). Purple = driest zones.",
    },
    "Nitrogen deficiency": {
        "key": "nitrogen_deficiency",
        "gradient": {0.2: "#27ae60", 0.5: "#f1c40f", 0.8: "#e67e22", 1.0: "#d35400"},
        "description": "Proxy from NDVI + bare-soil index. Orange/brown = likely nitrogen-short.",
    },
    "Areas of over-application": {
        "key": "over_application",
        "gradient": {0.2: "#ecf0f1", 0.5: "#9b59b6", 1.0: "#2c3e50"},
        "description": "Healthy zones on a farm that's already applying a high chemical dose -- candidates for cutting back.",
    },
    "Areas needing additional nutrients": {
        "key": "nutrient_need",
        "gradient": {0.2: "#ecf0f1", 0.5: "#3498db", 1.0: "#1b4f72"},
        "description": "Stressed zones where measured NPK is below the reference band -- candidates for added nutrients.",
    },
}
