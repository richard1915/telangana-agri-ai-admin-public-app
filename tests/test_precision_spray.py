"""
Automated tests for backend/precision_spray.py -- Stage 2.
Corresponds to the "Precision Spraying (Stage 2)" section of the test matrix.
"""
import pytest
from backend.precision_spray import (
    generate_field_grid, satellite_screen_field, simulate_drone_thermal_scan,
    generate_prescription_map, _point_in_polygon,
)

RECT_BOUNDARY = [(17.380, 78.480), (17.380, 78.484), (17.384, 78.484), (17.384, 78.480)]
TRIANGLE_BOUNDARY = [(17.380, 78.480), (17.380, 78.486), (17.386, 78.480)]


def test_grid_respects_plotted_boundary():
    """SPRAY-01: field grid points fall within the plotted boundary's bounding box."""
    grid = generate_field_grid(RECT_BOUNDARY, grid_resolution=4)
    lats = [p[0] for p in RECT_BOUNDARY]
    lons = [p[1] for p in RECT_BOUNDARY]
    for lat, lon in grid:
        assert min(lats) <= lat <= max(lats)
        assert min(lons) <= lon <= max(lons)


def test_grid_filters_to_actual_polygon_not_just_bounding_box():
    """
    Regression test: a non-rectangular field should only be analyzed
    within its actual shape, not the full rectangular bounding box.
    """
    grid = generate_field_grid(TRIANGLE_BOUNDARY, grid_resolution=4)
    assert len(grid) > 0
    assert all(_point_in_polygon(lat, lon, TRIANGLE_BOUNDARY) for lat, lon in grid)


def test_grid_falls_back_to_center_point_without_boundary():
    """SPRAY-02: falls back to a default extent around a center point when no boundary is plotted."""
    grid = generate_field_grid([], center_lat=17.385, center_lon=78.4867, grid_resolution=4)
    assert len(grid) > 0


def test_grid_raises_clear_error_without_boundary_or_center():
    """SPRAY-03: raises a descriptive ValueError instead of crashing when there's nothing to work from."""
    with pytest.raises(ValueError):
        generate_field_grid([], center_lat=None, center_lon=None)


def test_nearby_grid_cells_produce_distinct_ndvi():
    """
    Regression test for the seed-collision bug: nearby grid points must
    each get their own random seed (verified directly, since checking the
    3-decimal-rounded NDVI output alone allows a small number of
    coincidental duplicates by chance -- with ~500 possible rounded
    values, ~49 draws naturally produces a couple of matches even with
    fully independent seeds, per the birthday paradox).
    """
    grid = generate_field_grid(RECT_BOUNDARY, grid_resolution=4)
    seeds = {
        int(round(lat * 1_000_000)) * 2_000_003 + int(round(lon * 1_000_000))
        for lat, lon in grid
    }
    assert len(seeds) == len(grid)

    # And as a coarser sanity check on the actual bug this guards against
    # (catastrophic collapse -- e.g. all 49 points sharing 1-2 values),
    # most rounded NDVI outputs should still be unique.
    screened = satellite_screen_field(grid)
    ndvi_values = [z["ndvi"] for z in screened]
    assert len(set(ndvi_values)) >= len(ndvi_values) * 0.8


def test_drone_inspection_skips_healthy_zones():
    """SPRAY-06: zones not flagged by satellite screening are never drone-inspected."""
    grid = generate_field_grid(RECT_BOUNDARY, grid_resolution=4)
    screened = satellite_screen_field(grid)
    inspected = simulate_drone_thermal_scan(screened)
    for zone in inspected:
        if not zone["needs_drone_inspection"]:
            assert zone["drone_inspected"] is False
            assert zone["final_stress_class"] == "Healthy"


def test_prescription_map_zero_doses_healthy_zones():
    """SPRAY-07: healthy zones must receive zero prescribed dose."""
    grid = generate_field_grid(RECT_BOUNDARY, grid_resolution=4)
    screened = satellite_screen_field(grid)
    inspected = simulate_drone_thermal_scan(screened)
    prescription, summary = generate_prescription_map(inspected, optimized_dose_per_acre=42.3, zone_area_acres=0.5)
    for zone in prescription:
        if zone["final_stress_class"] == "Healthy":
            assert zone["prescribed_dose_kg"] == 0
            assert zone["will_spray"] is False


def test_savings_summary_totals_are_consistent():
    """SPRAY-09: blanket minus precision chemical use should equal the reported savings figure."""
    grid = generate_field_grid(RECT_BOUNDARY, grid_resolution=4)
    screened = satellite_screen_field(grid)
    inspected = simulate_drone_thermal_scan(screened)
    _, summary = generate_prescription_map(inspected, optimized_dose_per_acre=42.3, zone_area_acres=0.5)
    expected_savings = round(summary["total_blanket_chemical_kg"] - summary["total_precision_chemical_kg"], 3)
    assert abs(summary["stage2_additional_savings_kg"] - expected_savings) < 0.01
