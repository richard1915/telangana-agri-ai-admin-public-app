"""
Cost savings estimation.

Converts a chemical dose reduction (from MOA or any other optimizer in
optimizer_comparison.py) into farmer-facing rupee and kilogram savings.
Prices are indicative Telangana retail-range figures for common inputs
(2025-26 season) meant for a directional estimate, NOT a market quote --
the UI should always label this as "estimated savings."
"""

# ₹ per kg, indicative retail price ranges for common inputs used
# elsewhere in this project (backend/model.py chemical_database,
# backend/chemical_composition.py). Kept as a simple flat table so it is
# easy for a student/agronomist to update without touching UI code.
INDICATIVE_PRICE_PER_KG = {
    "Urea": 6.5,
    "DAP (18-46-0)": 27.0,
    "DAP": 27.0,
    "NPK 10-26-26": 24.0,
    "NPK 20-20-20": 26.0,
    "NPK 10-10-10": 22.0,
    "MOP": 17.5,
    "Gypsum": 6.0,
    "Lime": 8.0,
    "Dolomite": 9.0,
    "Limestone": 7.5,
    "Calcium Carbonate": 8.5,
    "Calcium Nitrate": 32.0,
    "Calcium Ammonium Nitrate": 20.0,
    "Zinc Sulfate": 55.0,
    "Iron Sulfate": 45.0,
    "Iron Chelate": 320.0,
    "Sulfur": 15.0,
    "Boron": 130.0,
    "Copper Sulfate": 210.0,
    "Aluminum Sulfate": 40.0,
    "Potassium Sulfate": 65.0,
    "Phosphate Rock": 18.0,
    "Wood Ash": 4.0,
    "Micronutrients": 150.0,
    "Pesticide": 850.0,
    "Fungicide": 700.0,
    "Herbicide": 500.0,
}

DEFAULT_PRICE_PER_KG = 25.0  # fallback for chemicals not in the table above


def estimate_cost_savings(chemical_name, baseline_dose_per_acre, optimized_dose_per_acre, farm_area_acres):
    """
    Args:
        chemical_name: e.g. "DAP (18-46-0)" -- looked up in the price table
        baseline_dose_per_acre / optimized_dose_per_acre: kg/acre
        farm_area_acres: total farm size

    Returns rupee and kg savings at both per-acre and farm-total scale.
    """
    price = INDICATIVE_PRICE_PER_KG.get(chemical_name, DEFAULT_PRICE_PER_KG)
    baseline_dose_per_acre = max(0.0, float(baseline_dose_per_acre))
    optimized_dose_per_acre = max(0.0, float(optimized_dose_per_acre))
    farm_area_acres = max(0.0, float(farm_area_acres))

    kg_saved_per_acre = max(0.0, baseline_dose_per_acre - optimized_dose_per_acre)
    kg_saved_total = kg_saved_per_acre * farm_area_acres
    rupees_saved_per_acre = kg_saved_per_acre * price
    rupees_saved_total = kg_saved_total * price
    pct_reduction = (
        (kg_saved_per_acre / baseline_dose_per_acre * 100.0) if baseline_dose_per_acre > 0 else 0.0
    )

    return {
        "chemical_name": chemical_name,
        "price_per_kg_inr": price,
        "price_is_indicative": chemical_name not in INDICATIVE_PRICE_PER_KG,
        "kg_saved_per_acre": round(kg_saved_per_acre, 2),
        "kg_saved_total": round(kg_saved_total, 2),
        "rupees_saved_per_acre": round(rupees_saved_per_acre, 0),
        "rupees_saved_total": round(rupees_saved_total, 0),
        "percentage_reduction": round(pct_reduction, 1),
    }
