"""
Chemical Composition
=====================

Every chemical name used elsewhere in this project (model.py's soil-type
recommendations, the "Already Used Chemical" dropdown, degradation rates)
has been treated as an interchangeable "kg/acre" quantity so far. That's
not agronomically accurate: 50 kg of Urea delivers a very different
amount of nitrogen than 50 kg of NPK 10-26-26.

This module gives each chemical its real nutrient composition (percent by
weight, standard industry fertilizer grades), so the rest of the app can
answer two concrete questions instead of a generic dose number:

  1. "How much actual N / P / K / S / Ca / micronutrient does X kg of
     THIS chemical deliver?"          -> compute_nutrients_delivered()
  2. "I have a Y kg/acre nutrient gap from my soil test -- how many kg
     of THIS chemical do I need to close it?"  -> dose_to_close_nutrient_gap()

Figures below are standard/typical fertilizer-grade percentages (N-P2O5-K2O
convention where applicable). Real products vary slightly by brand/region
-- treat these as reference values, not a substitute for checking a
specific product's label.
"""

# Percent by weight. Keys match nutrient columns used elsewhere in the
# project (Nitrogen, Phosphorus, Potassium) plus secondary/micronutrients
# where relevant. Phosphorus/Potassium here are expressed as elemental
# equivalents converted from P2O5 (x 0.436) / K2O (x 0.830) for
# consistency with the kg/acre "Phosphorus"/"Potassium" fields already
# used in Soil Information.
CHEMICAL_COMPOSITION = {
    # NPK blends
    "NPK": {"Nitrogen": 19.0, "Phosphorus": 19.0 * 0.436, "Potassium": 19.0 * 0.830},  # generic complex fertilizer (19-19-19) -- used when no specific grade is selected
    "NPK 10-26-26": {"Nitrogen": 10.0, "Phosphorus": 26.0 * 0.436, "Potassium": 26.0 * 0.830},
    "NPK 20-20-20": {"Nitrogen": 20.0, "Phosphorus": 20.0 * 0.436, "Potassium": 20.0 * 0.830},
    "NPK 10-10-10": {"Nitrogen": 10.0, "Phosphorus": 10.0 * 0.436, "Potassium": 10.0 * 0.830},
    "DAP": {"Nitrogen": 18.0, "Phosphorus": 46.0 * 0.436, "Potassium": 0.0},
    "DAP (18-46-0)": {"Nitrogen": 18.0, "Phosphorus": 46.0 * 0.436, "Potassium": 0.0},
    "Urea": {"Nitrogen": 46.0, "Phosphorus": 0.0, "Potassium": 0.0},
    "MOP": {"Nitrogen": 0.0, "Phosphorus": 0.0, "Potassium": 60.0 * 0.830},

    # Single-nutrient / secondary-nutrient sources
    "Calcium Nitrate": {"Nitrogen": 15.5, "Phosphorus": 0.0, "Potassium": 0.0, "Calcium": 19.0},
    "Calcium Ammonium Nitrate": {"Nitrogen": 26.0, "Phosphorus": 0.0, "Potassium": 0.0, "Calcium": 8.0},
    "Potassium Sulfate": {"Nitrogen": 0.0, "Phosphorus": 0.0, "Potassium": 50.0 * 0.830, "Sulfur": 18.0},
    "Phosphate Rock": {"Nitrogen": 0.0, "Phosphorus": 25.0 * 0.436, "Potassium": 0.0},
    "Gypsum": {"Nitrogen": 0.0, "Phosphorus": 0.0, "Potassium": 0.0, "Calcium": 23.0, "Sulfur": 18.0},

    # Micronutrients
    "Zinc Sulfate": {"Nitrogen": 0.0, "Phosphorus": 0.0, "Potassium": 0.0, "Zinc": 21.0},
    "Iron Sulfate": {"Nitrogen": 0.0, "Phosphorus": 0.0, "Potassium": 0.0, "Iron": 19.0},
    "Copper Sulfate": {"Nitrogen": 0.0, "Phosphorus": 0.0, "Potassium": 0.0, "Copper": 25.0},
    "Boron": {"Nitrogen": 0.0, "Phosphorus": 0.0, "Potassium": 0.0, "Boron_element": 17.0},
    "Micronutrients": {"Nitrogen": 0.0, "Phosphorus": 0.0, "Potassium": 0.0},

    # pH correctors (no significant N-P-K content; effect is on soil pH, not nutrient supply)
    "Lime": {"Nitrogen": 0.0, "Phosphorus": 0.0, "Potassium": 0.0, "Calcium": 38.0, "ph_effect": "raises"},
    "Dolomite": {"Nitrogen": 0.0, "Phosphorus": 0.0, "Potassium": 0.0, "Calcium": 22.0, "Magnesium": 13.0, "ph_effect": "raises"},
    "Limestone": {"Nitrogen": 0.0, "Phosphorus": 0.0, "Potassium": 0.0, "Calcium": 38.0, "ph_effect": "raises"},
    "Calcium Carbonate": {"Nitrogen": 0.0, "Phosphorus": 0.0, "Potassium": 0.0, "Calcium": 40.0, "ph_effect": "raises"},
    "Wood Ash": {"Nitrogen": 0.0, "Phosphorus": 1.0 * 0.436, "Potassium": 5.0 * 0.830, "ph_effect": "raises"},
    "Sulfur": {"Nitrogen": 0.0, "Phosphorus": 0.0, "Potassium": 0.0, "Sulfur": 90.0, "ph_effect": "lowers"},
    "Aluminum Sulfate": {"Nitrogen": 0.0, "Phosphorus": 0.0, "Potassium": 0.0, "Sulfur": 14.0, "ph_effect": "lowers"},
    "Iron Chelate": {"Nitrogen": 0.0, "Phosphorus": 0.0, "Potassium": 0.0, "Iron": 6.0, "ph_effect": "lowers"},
    "Acid fortified fertilizers": {"Nitrogen": 0.0, "Phosphorus": 0.0, "Potassium": 0.0, "ph_effect": "lowers"},
    "Acidifying fertilizers": {"Nitrogen": 0.0, "Phosphorus": 0.0, "Potassium": 0.0, "ph_effect": "lowers"},

    # Pest/disease control -- not nutrient carriers; composition is not
    # meaningfully expressible as N-P-K, kept for completeness.
    "Pesticide": {"Nitrogen": 0.0, "Phosphorus": 0.0, "Potassium": 0.0, "carrier": True},
    "Fungicide": {"Nitrogen": 0.0, "Phosphorus": 0.0, "Potassium": 0.0, "carrier": True},
    "Herbicide": {"Nitrogen": 0.0, "Phosphorus": 0.0, "Potassium": 0.0, "carrier": True},
}


import re

PEST_CONTROL_CHEMICALS = {"Pesticide", "Fungicide", "Herbicide"}


def infer_chemical_category(chemical_name):
    """
    Map a chemical name to the category meerkat_chemical_reduction() needs
    ("fertilizer" or "pest_control") to pick the right dose-response curve.
    Defaults to "fertilizer" for anything not explicitly a pest-control
    product (covers NPK blends, single-nutrient sources, and pH correctors,
    which all share the smoother diminishing-returns behavior).
    """
    if chemical_name in PEST_CONTROL_CHEMICALS:
        return "pest_control"
    composition = CHEMICAL_COMPOSITION.get(chemical_name)
    if composition and composition.get("carrier"):
        return "pest_control"
    return "fertilizer"


# Matches N-P-K grades written as e.g. "12-32-16", "12:32:16", "NPK 20-20-0",
# "Tata Gromor 28-28-0" -- most Indian fertilizer bags print exactly this
# pattern regardless of brand/product name, so this handles any grade we
# didn't explicitly hardcode, not just the handful of named entries above.
_NPK_GRADE_PATTERN = re.compile(r"(\d{1,2})\s*[-:]\s*(\d{1,2})\s*[-:]\s*(\d{1,2})")


def _parse_npk_grade(chemical_name):
    """
    Try to extract an N-P2O5-K2O grade directly from the chemical name
    string. Returns a composition dict if found, else None.
    """
    match = _NPK_GRADE_PATTERN.search(chemical_name)
    if not match:
        return None
    n, p2o5, k2o = (float(x) for x in match.groups())
    if n > 60 or p2o5 > 60 or k2o > 60:
        return None  # implausible as a fertilizer grade -- likely a false match
    return {
        "Nitrogen": n,
        "Phosphorus": p2o5 * 0.436,
        "Potassium": k2o * 0.830,
    }


def get_chemical_composition(chemical_name):
    """
    Return the nutrient composition dict for a chemical, or None if unknown.
    Tries, in order: (1) exact match against the known-chemical table,
    (2) parsing an N-P-K grade directly out of the name (handles branded
    products and grades we didn't explicitly hardcode).
    """
    exact = CHEMICAL_COMPOSITION.get(chemical_name)
    if exact is not None:
        return exact
    return _parse_npk_grade(chemical_name)


def compute_nutrients_delivered(chemical_name, dose_kg_per_acre, manual_composition=None):
    """
    Convert a raw chemical dose (kg/acre) into the actual nutrients it
    delivers (kg/acre of each), using real fertilizer-grade composition
    instead of treating all chemicals as interchangeable.

    If the chemical isn't recognized (unknown brand name, no grade in the
    name) and manual_composition is provided (e.g. {"Nitrogen": 20,
    "Phosphorus": 20, "Potassium": 20} read off the product label by the
    farmer/student), that's used instead of failing outright.
    """
    composition = get_chemical_composition(chemical_name)
    if composition is None and manual_composition is not None:
        composition = {
            "Nitrogen": manual_composition.get("Nitrogen", 0.0),
            "Phosphorus": manual_composition.get("Phosphorus", 0.0) * 0.436,
            "Potassium": manual_composition.get("Potassium", 0.0) * 0.830,
        }
    if composition is None:
        return {
            "available": False,
            "error": (
                f"No composition data for '{chemical_name}' and no auto-detectable "
                f"N-P-K grade in the name. Enter the grade printed on the product "
                f"label manually (most Indian fertilizer bags print it, e.g. '19-19-19')."
            ),
        }

    delivered = {}
    for nutrient, pct in composition.items():
        if nutrient in ("ph_effect", "carrier"):
            continue
        delivered[nutrient] = round(dose_kg_per_acre * (pct / 100.0), 3)

    return {
        "available": True,
        "chemical": chemical_name,
        "dose_kg_per_acre": dose_kg_per_acre,
        "nutrients_delivered_kg_per_acre": delivered,
        "ph_effect": composition.get("ph_effect"),
        "is_pest_control_only": composition.get("carrier", False),
    }


def dose_to_close_nutrient_gap(chemical_name, nutrient_type, nutrient_gap_kg_per_acre):
    """
    Given a measured nutrient shortfall (e.g. from a soil test: "need 50
    kg/acre more Nitrogen") and a chosen chemical, compute how many kg/acre
    of that chemical are needed to close the gap.

    nutrient_type must be one of the keys used in Soil Information:
    'Nitrogen', 'Phosphorus', 'Potassium' (or a secondary nutrient key
    present in that chemical's composition, e.g. 'Sulfur', 'Zinc').
    """
    composition = get_chemical_composition(chemical_name)
    if composition is None:
        return {"available": False, "error": f"No composition data for '{chemical_name}'."}

    pct = composition.get(nutrient_type, 0.0)
    if not isinstance(pct, (int, float)) or pct <= 0:
        return {
            "available": False,
            "error": f"'{chemical_name}' does not supply meaningful {nutrient_type} "
                     f"(0% content) -- choose a different chemical for this nutrient.",
        }

    required_dose = round(nutrient_gap_kg_per_acre / (pct / 100.0), 3)
    return {
        "available": True,
        "chemical": chemical_name,
        "nutrient_type": nutrient_type,
        "nutrient_gap_kg_per_acre": nutrient_gap_kg_per_acre,
        "required_dose_kg_per_acre": required_dose,
    }


def compare_chemicals_for_nutrient_gap(nutrient_type, nutrient_gap_kg_per_acre, candidates=None):
    """
    Compare several chemicals side by side for closing the same nutrient
    gap -- useful for showing a farmer/student that different chemicals
    require very different application quantities to deliver the same
    actual nutrient amount.
    """
    if candidates is None:
        candidates = [
            name for name, comp in CHEMICAL_COMPOSITION.items()
            if isinstance(comp.get(nutrient_type), (int, float)) and comp.get(nutrient_type, 0) > 0
        ]

    results = []
    for chem in candidates:
        r = dose_to_close_nutrient_gap(chem, nutrient_type, nutrient_gap_kg_per_acre)
        if r.get("available"):
            results.append(r)

    results.sort(key=lambda r: r["required_dose_kg_per_acre"])
    return results
