"""
Automated tests for chemical composition, satellite fallback, weather
fallback, and the database layer.
"""
from backend.chemical_composition import (
    compute_nutrients_delivered, dose_to_close_nutrient_gap,
    infer_chemical_category, get_chemical_composition,
)
from backend.satellite_api import get_soil_type_for_location, SENTINELHUB_AVAILABLE, SENTINELHUB_AUTHENTICATED
from backend.model import recommend_chemical_by_soil_type
from backend.database import (
    create_db, insert_moa_result, fetch_all, fetch_farmer_moa_history,
    insert_harvest_outcome, fetch_harvest_accuracy_summary,
)


# ---- Chemical composition ----

def test_same_dose_different_chemicals_delivers_different_nutrients():
    """CHEM-01: 50kg of different chemicals must deliver different actual nitrogen amounts."""
    urea = compute_nutrients_delivered("Urea", 50)
    dap = compute_nutrients_delivered("DAP", 50)
    assert urea["nutrients_delivered_kg_per_acre"]["Nitrogen"] != dap["nutrients_delivered_kg_per_acre"]["Nitrogen"]


def test_npk_grade_auto_parsed_from_branded_name():
    """CHEM-02: a branded product name containing an N-P-K grade should resolve without an exact dictionary entry."""
    result = compute_nutrients_delivered("Tata Gromor 28-28-0", 50)
    assert result["available"] is True


def test_manual_composition_override_when_unrecognized():
    """CHEM-03: an unrecognized chemical with no grade in its name should work via manual override."""
    result = compute_nutrients_delivered(
        "Some Unknown Branded Product", 50,
        manual_composition={"Nitrogen": 28, "Phosphorus": 28, "Potassium": 0},
    )
    assert result["available"] is True


def test_pest_control_chemicals_are_categorized_correctly():
    """CHEM-04: pesticide/fungicide/herbicide must map to pest_control, fertilizers to fertilizer."""
    assert infer_chemical_category("Pesticide") == "pest_control"
    assert infer_chemical_category("Fungicide") == "pest_control"
    assert infer_chemical_category("Urea") == "fertilizer"
    assert infer_chemical_category("NPK") == "fertilizer"


def test_all_seven_telangana_soil_types_resolve():
    """SOIL-01: every one of Telangana's 7 official soil types must return a valid chemical recommendation."""
    for soil in [
        "Chalka (Red Sandy Loam)", "Dubba (Red Loamy Sand)", "Lateritic Soil",
        "Shallow-Medium Black Soil", "Deep Black Soil (Black Cotton)",
        "Salt-affected Soil", "Alluvial Soil",
    ]:
        result = recommend_chemical_by_soil_type(soil, ph=6.8)
        assert result["primary_chemical"]


def test_legacy_soil_type_aliases_still_resolve():
    """Backward compatibility: old generic soil type names must still work after the 7-type expansion."""
    for legacy_name in ("Red Soil", "Black Soil", "Laterite", "Alluvial"):
        result = recommend_chemical_by_soil_type(legacy_name, ph=6.8)
        assert result["primary_chemical"]


# ---- Satellite fallback ----

def test_satellite_falls_back_gracefully_without_sentinelhub_auth():
    """SAT-01/02: without OAuth credentials, satellite calls should return synthetic data, not crash."""
    result = get_soil_type_for_location(17.4, 78.5)
    assert result is not None
    assert result["source"] in ("live", "synthetic")
    if not SENTINELHUB_AUTHENTICATED:
        assert result["source"] == "synthetic"


# ---- Database ----

def test_create_db_is_idempotent():
    """DB-01: calling create_db() again must not error or wipe existing tables."""
    create_db()
    create_db()  # should not raise


def test_moa_result_insert_and_fetch_roundtrip():
    """DB-02/03: an inserted MOA result should be retrievable via fetch_farmer_moa_history."""
    ok = insert_moa_result(
        student_name="Test Student", farmer_name="Test Farmer QA",
        next_crop="Rice", initial_dose=100.0, optimized_dose=80.0,
        reduction_percentage=20.0, farm_area=2.0, total_initial=200.0, total_optimized=160.0,
    )
    assert ok is True
    history = fetch_farmer_moa_history("Test Farmer QA")
    assert len(history) >= 1
    assert history[0]["farmer_name"] == "Test Farmer QA"


def test_fetch_all_rejects_unknown_table_name():
    """DB-04: fetch_all must not execute an arbitrary/unknown table name (SQL injection guard)."""
    result = fetch_all("users_secret_table; DROP TABLE moa_results;--")
    assert result == []


# ---- Harvest outcome tracking (real-data validation feedback loop) ----

def test_harvest_outcome_insert_and_summary():
    """A recorded harvest outcome should appear in the accuracy summary with a computed MAE."""
    ok = insert_harvest_outcome(
        student_name="Test Student", farmer_name="Harvest QA Farmer", crop="Rice",
        predicted_yield_kg_per_acre=4000.0, actual_yield_kg_per_acre=3800.0,
        harvest_date="2026-11-01", notes="test row",
    )
    assert ok is True
    summary = fetch_harvest_accuracy_summary()
    assert summary["count"] >= 1
    assert summary["mae_kg_per_acre"] is not None


def test_harvest_accuracy_summary_handles_no_data():
    """Before any outcomes are recorded, the summary should return safe defaults, not crash."""
    from backend.database import harvest_outcomes, engine
    with engine.begin() as conn:
        conn.execute(harvest_outcomes.delete())
    summary = fetch_harvest_accuracy_summary()
    assert summary["count"] == 0
    assert summary["mae_kg_per_acre"] is None
