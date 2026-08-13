"""
Environmental impact estimation.

Turns an optimizer's chemical-dose reduction into the three
farmer/agronomist-facing indicators requested for the environmental
dashboard: chemical reduction, estimated residue reduction, and a soil
health improvement indicator. All figures here are directional estimates
derived from the same dose-response and degradation models already used
elsewhere in the backend (meerkat_optimizer, model.estimate_residue_level)
-- not a substitute for lab soil testing or residue analysis.
"""

from backend.model import estimate_residue_level


def estimate_environmental_impact(chemical_name, baseline_dose_per_acre, optimized_dose_per_acre,
                                   organic_carbon_pct=None, days_since_application=30):
    """
    Args:
        chemical_name: used to look up a degradation half-life
        baseline_dose_per_acre / optimized_dose_per_acre: kg/acre
        organic_carbon_pct: measured soil organic carbon %, if available
            (used only for the directional soil-health indicator)
        days_since_application: horizon used for the residue comparison

    Returns chemical reduction %, estimated residue-load reduction %, and
    a simple soil health improvement indicator (Low/Moderate/High) based
    on how much less chemical load the soil is carrying plus its organic
    carbon trend.
    """
    baseline_dose_per_acre = max(0.0, float(baseline_dose_per_acre))
    optimized_dose_per_acre = max(0.0, float(optimized_dose_per_acre))

    chemical_reduction_pct = (
        (baseline_dose_per_acre - optimized_dose_per_acre) / baseline_dose_per_acre * 100.0
        if baseline_dose_per_acre > 0 else 0.0
    )

    # Residue at the same time horizon is proportional to applied dose
    # under the same first-order decay model used in model.py -- so the
    # residue-load reduction tracks the dose reduction 1:1 at any fixed
    # number of days since application.
    baseline_residue = estimate_residue_level(chemical_name, baseline_dose_per_acre, days_since_application)
    optimized_residue = estimate_residue_level(chemical_name, optimized_dose_per_acre, days_since_application)
    residue_reduction_pct = (
        (baseline_residue["remaining_amount"] - optimized_residue["remaining_amount"])
        / baseline_residue["remaining_amount"] * 100.0
        if baseline_residue["remaining_amount"] > 0 else 0.0
    )

    # Directional soil-health indicator: combines chemical-load reduction
    # (less residue build-up over repeated seasons) with organic carbon
    # level (a standard soil-health proxy). This is a heuristic label for
    # the dashboard, not a lab-verified soil health score.
    carbon_bonus = 0
    if organic_carbon_pct is not None:
        if organic_carbon_pct >= 0.75:
            carbon_bonus = 2
        elif organic_carbon_pct >= 0.5:
            carbon_bonus = 1

    score = 0
    if chemical_reduction_pct >= 20:
        score = 2
    elif chemical_reduction_pct >= 8:
        score = 1
    score += carbon_bonus

    if score >= 3:
        indicator = "High"
    elif score >= 1:
        indicator = "Moderate"
    else:
        indicator = "Low"

    return {
        "chemical_reduction_pct": round(chemical_reduction_pct, 1),
        "estimated_residue_reduction_pct": round(residue_reduction_pct, 1),
        "soil_health_improvement_indicator": indicator,
        "note": (
            "Directional estimates derived from the dose-response and residue-decay models "
            "used elsewhere in this app -- not a substitute for lab soil testing or residue analysis."
        ),
    }
