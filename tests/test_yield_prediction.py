"""
Automated tests for backend/meerkat_optimizer.py -- yield prediction.
Corresponds to the "Yield Prediction" section of the test case matrix.
"""
from backend.meerkat_optimizer import predict_crop_yield, predict_crop_yield_ml


def test_rule_based_prediction_returns_expected_fields():
    """YIELD-01: rule-based prediction returns all documented fields."""
    result = predict_crop_yield(
        "Rice", ph=6.5, moisture=45, temperature=27,
        nitrogen=100, phosphorus=50, potassium=40, rainfall=1300,
    )
    for field in ("predicted_yield", "confidence_score", "limiting_factor", "factors", "base_yield_potential"):
        assert field in result


def test_low_moisture_reduces_predicted_yield_below_base():
    """Regression test for the 'why is yield low' behavior: a limiting factor should pull yield down."""
    result = predict_crop_yield(
        "Rice", ph=6.5, moisture=18, temperature=27,
        nitrogen=100, phosphorus=50, potassium=40, rainfall=1300,
    )
    assert result["predicted_yield"] < result["base_yield_potential"]
    assert result["limiting_factor"] == "Moisture"


def test_new_telangana_crops_are_supported():
    """Crop-expansion regression test: newly added Telangana crops must not fall back to defaults silently."""
    for crop in ("Turmeric", "Red Gram", "Green Gram", "Bajra", "Soybean", "Paddy"):
        result = predict_crop_yield(
            crop, ph=6.5, moisture=25, temperature=27,
            nitrogen=100, phosphorus=40, potassium=60, rainfall=700,
        )
        assert result["base_yield_potential"] > 0
        assert result["predicted_yield"] > 0


def test_trained_ml_model_predicts_successfully():
    """YIELD-02/03: trained ML model self-trains on first use and returns a prediction."""
    result = predict_crop_yield_ml(
        "Rice", "Chalka (Red Sandy Loam)", ph=6.5, nitrogen=100, phosphorus=50,
        potassium=40, organic_carbon=0.8, moisture=45, electrical_conductivity=1.0,
    )
    assert result["available"] is True
    assert result["predicted_yield"] > 0
    assert 0 <= result["confidence_score"] <= 100
