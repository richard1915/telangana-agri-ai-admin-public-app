"""
Stage 2 - Precision Spraying
============================

Stage 1 (satellite_api.py) answers "what is the soil/crop condition here?"
for a single point or a set of plotted boundary points.

Stage 2 answers "WHERE should the chemical actually be applied?" by:
  1. Sampling a grid across the field and reusing Stage 1's satellite
     function to flag candidate stress zones (low NDVI = possible stress).
  2. Simulating a drone RGB + thermal inspection, but ONLY over the
     candidate zones flagged in step 1 (not the whole field) -- this
     mirrors the real workflow where a drone doesn't need to fly a
     detailed pass over healthy areas.
  3. Turning the refined per-zone severity into a prescription map: how
     much of the Stage 1 / MOA-optimized dose (kg/acre) each zone
     actually needs, with healthy zones getting zero.
  4. Summarizing precision-sprayed chemical use vs a blanket application
     of the same optimized dose across the whole field, to show the
     additional savings Stage 2 contributes on top of Stage 1's MOA
     reduction.
"""

import random

from .satellite_api import get_soil_type_for_location

# NDVI thresholds used to flag a grid cell as a stress candidate during the
# satellite screening pass (step 1).
NDVI_HEALTHY_THRESHOLD = 0.5
NDVI_MILD_THRESHOLD = 0.35
NDVI_MODERATE_THRESHOLD = 0.2

# Dose multipliers applied to the Stage 1 / MOA-optimized per-acre dose,
# based on final severity after drone inspection (step 3). Healthy zones
# are skipped entirely -- this is the core of "precision" spraying.
SEVERITY_DOSE_MULTIPLIER = {
    "Healthy": 0.0,
    "Mild stress": 0.5,
    "Moderate stress": 0.8,
    "Severe stress": 1.0,
}


def _point_in_polygon(lat, lon, polygon):
    """
    Standard ray-casting point-in-polygon test. polygon is a list of
    (lat, lon) tuples. Used so grid points outside an irregularly-shaped
    plotted field are excluded from analysis, instead of analyzing the
    whole rectangular bounding box (which can include area well outside
    the field you actually selected).
    """
    inside = False
    n = len(polygon)
    j = n - 1
    for i in range(n):
        lat_i, lon_i = polygon[i]
        lat_j, lon_j = polygon[j]
        intersects = ((lon_i > lon) != (lon_j > lon)) and (
            lat < (lat_j - lat_i) * (lon - lon_i) / (lon_j - lon_i + 1e-15) + lat_i
        )
        if intersects:
            inside = not inside
        j = i
    return inside


def generate_field_grid(shape_points, center_lat=None, center_lon=None, grid_resolution=4):
    """
    Build a grid of sample points covering the field.

    If a plotted farm boundary (>=3 points, from the Maps page) is
    available, the grid is generated over its bounding box and then
    filtered to only the points that actually fall inside the plotted
    polygon (not just its bounding rectangle) -- so analysis reflects the
    field you actually selected, not a larger rectangular area around it.
    Otherwise falls back to a small default extent around a center point
    (e.g. the farmer's saved latitude/longitude).
    """
    is_polygon = shape_points and len(shape_points) >= 3

    if is_polygon:
        lats = [p[0] for p in shape_points]
        lons = [p[1] for p in shape_points]
        lat_min, lat_max = min(lats), max(lats)
        lon_min, lon_max = min(lons), max(lons)
    else:
        if center_lat is None or center_lon is None:
            raise ValueError("Need either plotted shape_points or a center_lat/center_lon.")
        # ~300m default half-extent when no boundary has been plotted.
        half_extent_deg = 0.0027
        lat_min, lat_max = center_lat - half_extent_deg, center_lat + half_extent_deg
        lon_min, lon_max = center_lon - half_extent_deg, center_lon + half_extent_deg

    # Use a finer bounding-box grid when filtering to a polygon, since a
    # coarse grid intersected with an irregular shape can otherwise leave
    # very few (or zero) points inside a narrow/angled field.
    raw_resolution = max(2, min(6, grid_resolution))
    sample_resolution = raw_resolution * 2 if is_polygon else raw_resolution

    lat_step = (lat_max - lat_min) / max(1, sample_resolution - 1)
    lon_step = (lon_max - lon_min) / max(1, sample_resolution - 1)

    grid_points = []
    for i in range(sample_resolution):
        for j in range(sample_resolution):
            grid_points.append((lat_min + i * lat_step, lon_min + j * lon_step))

    if is_polygon:
        filtered = [pt for pt in grid_points if _point_in_polygon(pt[0], pt[1], shape_points)]
        # Safety fallback: a very narrow/thin field can leave too few (or
        # zero) points inside after filtering -- rather than return an
        # empty grid, fall back to the unfiltered bounding-box points so
        # Stage 2 / Field Health Imaging still have something to analyze.
        if len(filtered) >= 4:
            return filtered
        return grid_points

    return grid_points


def _classify_ndvi_stress(ndvi):
    if ndvi >= NDVI_HEALTHY_THRESHOLD:
        return "Healthy"
    elif ndvi >= NDVI_MILD_THRESHOLD:
        return "Mild stress"
    elif ndvi >= NDVI_MODERATE_THRESHOLD:
        return "Moderate stress"
    else:
        return "Severe stress"


def satellite_screen_field(grid_points):
    """
    Step 1: Satellite screening pass over the whole field.
    Reuses Stage 1's get_soil_type_for_location() for every grid cell to
    flag candidate stress zones by NDVI. This is deliberately the same
    function Stage 1 uses on the Maps page -- Stage 2 builds on it rather
    than re-implementing satellite analysis.
    """
    results = []
    for zone_id, (lat, lon) in enumerate(grid_points, start=1):
        info = get_soil_type_for_location(lat, lon)
        if not info:
            continue
        stress_class = _classify_ndvi_stress(info["ndvi"])
        results.append({
            "zone_id": zone_id,
            "lat": lat,
            "lon": lon,
            "ndvi": info["ndvi"],
            "soil_type": info["soil_type"],
            "source": info.get("source", "synthetic"),
            "satellite_stress_class": stress_class,
            "needs_drone_inspection": stress_class != "Healthy",
        })
    return results


def simulate_drone_thermal_scan(screened_zones):
    """
    Step 2: Drone RGB + thermal inspection.
    Only runs on zones flagged by the satellite pass (needs_drone_inspection
    == True) -- healthy zones are never visited, mirroring the real
    workflow's efficiency gain. Simulates a canopy-temperature anomaly
    (deterministically seeded by location, so results are reproducible)
    that's inversely related to NDVI: more stressed vegetation tends to
    run warmer due to reduced transpiration.
    """
    inspected = []
    for zone in screened_zones:
        if not zone["needs_drone_inspection"]:
            inspected.append({
                **zone,
                "drone_inspected": False,
                "thermal_anomaly_c": 0.0,
                "final_stress_class": "Healthy",
            })
            continue

        rng = random.Random(int(zone["lat"] * 1e5) ^ int(zone["lon"] * 1e5))
        # Lower NDVI -> larger simulated thermal anomaly above canopy baseline.
        base_anomaly = (1 - zone["ndvi"]) * 6.0
        thermal_anomaly = round(base_anomaly + rng.uniform(-0.5, 0.5), 2)

        if thermal_anomaly >= 3.5:
            final_class = "Severe stress"
        elif thermal_anomaly >= 2.0:
            final_class = "Moderate stress"
        elif thermal_anomaly > 0.5:
            final_class = "Mild stress"
        else:
            final_class = "Healthy"

        inspected.append({
            **zone,
            "drone_inspected": True,
            "thermal_anomaly_c": thermal_anomaly,
            "final_stress_class": final_class,
        })
    return inspected


def generate_prescription_map(inspected_zones, optimized_dose_per_acre, zone_area_acres):
    """
    Step 3 + 4: Turn per-zone severity into an actual variable-rate
    prescription (how much of the Stage 1 / MOA-optimized dose each zone
    gets), then summarize precision-sprayed volume vs a blanket
    application of the same optimized dose across the whole field.
    """
    prescription = []
    for zone in inspected_zones:
        severity = zone["final_stress_class"]
        multiplier = SEVERITY_DOSE_MULTIPLIER.get(severity, 0.0)
        zone_dose_kg = round(optimized_dose_per_acre * multiplier * zone_area_acres, 4)
        prescription.append({
            **zone,
            "prescribed_dose_kg": zone_dose_kg,
            "will_spray": zone_dose_kg > 0,
        })

    total_zones = len(prescription)
    sprayed_zones = sum(1 for z in prescription if z["will_spray"])
    total_area = total_zones * zone_area_acres
    sprayed_area = sprayed_zones * zone_area_acres

    total_precision_chemical = round(sum(z["prescribed_dose_kg"] for z in prescription), 3)
    total_blanket_chemical = round(optimized_dose_per_acre * total_area, 3)
    stage2_savings_kg = round(total_blanket_chemical - total_precision_chemical, 3)
    stage2_savings_pct = round((stage2_savings_kg / total_blanket_chemical) * 100, 1) if total_blanket_chemical > 0 else 0.0

    summary = {
        "total_zones": total_zones,
        "sprayed_zones": sprayed_zones,
        "skipped_zones": total_zones - sprayed_zones,
        "total_area_acres": round(total_area, 3),
        "sprayed_area_acres": round(sprayed_area, 3),
        "area_sprayed_pct": round((sprayed_area / total_area) * 100, 1) if total_area > 0 else 0.0,
        "total_precision_chemical_kg": total_precision_chemical,
        "total_blanket_chemical_kg": total_blanket_chemical,
        "stage2_additional_savings_kg": stage2_savings_kg,
        "stage2_additional_savings_pct": stage2_savings_pct,
    }
    return prescription, summary
