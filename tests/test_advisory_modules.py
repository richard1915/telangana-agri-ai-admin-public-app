"""
Automated tests for the new advisory-driven modules:
backend/optimizer_comparison.py, backend/cost_savings.py,
backend/environmental_impact.py.
"""
from backend.optimizer_comparison import compare_optimizers, run_moa, run_pso, run_ga, run_rule_based
from backend.cost_savings import estimate_cost_savings
from backend.environmental_impact import estimate_environmental_impact


def test_compare_optimizers_returns_all_four_methods():
    result = compare_optimizers(100, target_yield_percentage=0.95, chemical_category="fertilizer")
    methods = {row["method"] for row in result["rows"]}
    assert len(result["rows"]) == 4
    assert any("MOA" in m for m in methods)
    assert any("PSO" in m for m in methods)
    assert any("GA" in m for m in methods)
    assert any("Rule-based" in m for m in methods)
    assert result["recommended"] is not None


def test_compare_optimizers_rejects_zero_dose():
    result = compare_optimizers(0)
    assert result["rows"] == []
    assert result["recommended"] is None


def test_rule_based_is_the_fastest_method():
    """The rule-based baseline does no search, so it should be the cheapest to run."""
    moa = run_moa(100)
    pso = run_pso(100)
    ga = run_ga(100)
    rule = run_rule_based(100)
    assert rule["runtime_ms"] <= moa["runtime_ms"]
    assert rule["runtime_ms"] <= pso["runtime_ms"]
    assert rule["runtime_ms"] <= ga["runtime_ms"]


def test_cost_savings_zero_when_no_reduction():
    result = estimate_cost_savings("Urea", 100, 100, 2.0)
    assert result["kg_saved_total"] == 0.0
    assert result["rupees_saved_total"] == 0.0
    assert result["percentage_reduction"] == 0.0


def test_cost_savings_scales_with_farm_area():
    small = estimate_cost_savings("DAP (18-46-0)", 100, 80, 1.0)
    large = estimate_cost_savings("DAP (18-46-0)", 100, 80, 4.0)
    assert large["kg_saved_total"] == small["kg_saved_total"] * 4
    assert large["rupees_saved_total"] == small["rupees_saved_total"] * 4


def test_cost_savings_flags_unknown_chemical_as_indicative():
    result = estimate_cost_savings("Some Unlisted Chemical", 100, 80, 1.0)
    assert result["price_is_indicative"] is True


def test_environmental_impact_reduction_matches_dose_reduction():
    result = estimate_environmental_impact("Urea", 100, 80, organic_carbon_pct=0.6)
    assert result["chemical_reduction_pct"] == 20.0
    assert result["soil_health_improvement_indicator"] in ("Low", "Moderate", "High")


def test_environmental_impact_no_reduction_gives_low_indicator():
    result = estimate_environmental_impact("Urea", 100, 100, organic_carbon_pct=0.2)
    assert result["chemical_reduction_pct"] == 0.0
    assert result["soil_health_improvement_indicator"] == "Low"
