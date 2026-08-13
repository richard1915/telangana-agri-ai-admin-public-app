"""
Automated tests for the remaining new features: explainability (XAI),
PDF report generation, and cross-season farmer history.
"""
import os
import tempfile
from backend.explainability import get_global_feature_importance, explain_prediction
from backend.report_generator import generate_pdf_report
from backend.database import create_db, insert_moa_result, fetch_farmer_season_summary


def test_global_feature_importance_available_and_sums_reasonably():
    result = get_global_feature_importance()
    assert result["available"] is True
    assert len(result["features"]) > 0
    total_pct = sum(f["importance_pct"] for f in result["features"])
    assert 0 < total_pct <= 100.5  # allow small rounding slack


def test_explain_prediction_returns_directional_contributions():
    result = explain_prediction(
        "Rice", "Chalka (Red Sandy Loam)", ph=6.5, nitrogen=100, phosphorus=50,
        potassium=40, organic_carbon=0.8, moisture=45, electrical_conductivity=1.0,
    )
    assert result["available"] is True
    assert len(result["contributions"]) > 0
    for c in result["contributions"]:
        assert c["direction"] in ("increases yield", "decreases yield")


def test_pdf_report_generates_a_real_file():
    """
    Regression test using the exact context shape the UI actually sends
    (a plain string here instead of the nested dict would previously
    crash with AttributeError -- this locks in the correct schema).
    """
    context = {
        "farmer": {"farmer_name": "PDF QA Farmer", "district": "Rangareddy", "village": "Shamshabad",
                   "farm_size": 3.0, "latitude": 17.2403, "longitude": 78.4294},
        "soil": {"ph": 6.6, "nitrogen": 220, "phosphorus": 55, "potassium": 210, "organic_carbon": 0.85,
                 "moisture": 32, "electrical_conductivity": 1.1},
        "weather": {"temperature_c": 29, "rainfall_mm": 900},
        "crop_recommendation": {"next_crop": "Cotton", "recommended_chemical": "NPK 10-26-26",
                                  "predicted_yield": 1750, "confidence_score": 88, "limiting_factor": "Moisture"},
        "moa_recommendation": {"method": "MOA", "initial_dose": 100, "optimized_dose": 81.2,
                                 "reduction_percentage": 18.8, "farm_area": 3.0},
        "cost_savings": {"rupees_saved_per_acre": 124, "kg_saved_per_acre": 19, "percentage_reduction": 19},
        "environmental_impact": {"chemical_reduction_pct": 18.8, "estimated_residue_reduction_pct": 18.8,
                                   "soil_health_improvement_indicator": "Moderate"},
        "latitude": 17.2403, "longitude": 78.4294, "generated_by": "QA Test",
    }
    output_path = os.path.join(tempfile.gettempdir(), "pytest_report.pdf")
    result_path = generate_pdf_report(output_path, context)
    assert os.path.exists(result_path)
    with open(result_path, "rb") as f:
        header = f.read(5)
    assert header == b"%PDF-"  # a genuine PDF, not an empty/broken file
    os.remove(result_path)


def test_farmer_season_summary_accumulates_across_multiple_entries():
    create_db()
    farmer = "Season Accumulation QA"
    insert_moa_result(student_name="T", farmer_name=farmer, next_crop="Rice",
                       initial_dose=100, optimized_dose=80, reduction_percentage=20,
                       farm_area=2, total_initial=200, total_optimized=160)
    insert_moa_result(student_name="T", farmer_name=farmer, next_crop="Cotton",
                       initial_dose=60, optimized_dose=50, reduction_percentage=16.7,
                       farm_area=2, total_initial=120, total_optimized=100)
    summary = fetch_farmer_season_summary(farmer)
    assert summary["season_count"] == 2
    assert summary["cumulative_kg_saved"] > 0
