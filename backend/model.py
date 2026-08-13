def recommend_crop(data):
    """
    Recommend crop based on soil parameters
    """
    try:
        ph = float(data.get("ph", 7))
        carbon = float(data.get("carbon", 1))
        moisture = float(data.get("moisture", 30))
        
        if ph < 6:
            crop = "Rice"
        elif carbon > 1.5:
            crop = "Cotton"
        elif moisture > 40:
            crop = "Sugarcane"
        else:
            crop = "Maize"
        
        return crop
    except (ValueError, TypeError) as e:
        return "Maize"  # Default crop


def recommend_chemical_by_soil_type(soil_type, ph):
    """
    Recommend safe chemicals based on soil type and pH.

    Soil types follow Telangana's official 7-type classification
    ("Soils of Andhra Pradesh", 1976 -- still the standard reference used
    by the state agriculture department): Chalka, Dubba, Lateritic,
    Shallow-Medium Black, Deep Black, Salt-affected, and Alluvial. Older
    generic names (Red Soil/Black Soil/Laterite Soil/Alluvial Soil) are
    kept as aliases so existing data/callers still resolve correctly.
    """
    chemical_database = {
        "Chalka (Red Sandy Loam)": {
            # Sandy, low water/nutrient retention -- most widespread soil
            # in Telangana (~32 of 33 districts). Needs organic matter and
            # split/slow-release fertilizer application more than most.
            "low_ph": ["Lime", "Dolomite", "Wood Ash"],
            "optimal_ph": ["NPK 10-26-26", "Zinc Sulfate", "Iron Sulfate"],
            "high_ph": ["Sulfur", "Iron Chelate"]
        },
        "Dubba (Red Loamy Sand)": {
            # Higher clay content than Chalka -- better water retention,
            # still part of the red soil family.
            "low_ph": ["Lime", "Dolomite"],
            "optimal_ph": ["NPK 10-26-26", "Zinc Sulfate", "Iron Sulfate"],
            "high_ph": ["Sulfur", "Iron Chelate"]
        },
        "Lateritic Soil": {
            "low_ph": ["Lime", "Limestone", "Calcium Carbonate"],
            "optimal_ph": ["NPK 20-20-20", "Boron", "Copper Sulfate"],
            "high_ph": ["Sulfur", "Acid fortified fertilizers"]
        },
        "Shallow-Medium Black Soil": {
            "low_ph": ["Gypsum", "Calcium Nitrate"],
            "optimal_ph": ["DAP (18-46-0)", "MOP", "Micronutrients"],
            "high_ph": ["Sulfur", "Aluminum Sulfate"]
        },
        "Deep Black Soil (Black Cotton)": {
            # High clay, cracks when dry, good nutrient retention but poor
            # drainage. Runs alkaline more often than the other types --
            # Gypsum is the primary structure/reclamation agent here.
            "low_ph": ["Gypsum", "Calcium Nitrate"],
            "optimal_ph": ["Gypsum", "DAP (18-46-0)", "MOP"],
            "high_ph": ["Gypsum", "Sulfur", "Aluminum Sulfate"]
        },
        "Salt-affected Soil": {
            # Saline/sodic patches -- Gypsum-based reclamation takes
            # priority over routine fertilization until salinity is
            # brought down; avoid high-salt-index chemicals early on.
            "low_ph": ["Gypsum"],
            "optimal_ph": ["Gypsum", "Calcium Nitrate", "Micronutrients"],
            "high_ph": ["Gypsum", "Sulfur"]
        },
        "Alluvial Soil": {
            "low_ph": ["Calcium Ammonium Nitrate"],
            "optimal_ph": ["NPK 10-10-10", "Potassium Sulfate", "Phosphate Rock"],
            "high_ph": ["Sulfur", "Acidifying fertilizers"]
        },
    }

    # Backward-compatible aliases -- older generic soil type names (still
    # used in some existing records/dropdowns) map onto the closest of
    # the 7 real types above.
    aliases = {
        "Red Soil": "Chalka (Red Sandy Loam)",
        "Black Soil": "Shallow-Medium Black Soil",
        "Laterite": "Lateritic Soil",
        "Alluvial": "Alluvial Soil",
    }

    soil_type = soil_type.strip()
    soil_type = aliases.get(soil_type, soil_type)
    if soil_type not in chemical_database:
        soil_type = "Chalka (Red Sandy Loam)"  # Default -- most common soil type in Telangana
    
    if ph < 6.0:
        category = "low_ph"
    elif ph > 7.5:
        category = "high_ph"
    else:
        category = "optimal_ph"
    
    recommendations = chemical_database[soil_type][category]
    
    return {
        "soil_type": soil_type,
        "ph_category": category,
        "primary_chemical": recommendations[0],
        "all_recommendations": recommendations,
        "confidence": 0.85
    }


def check_chemical_safety(previous_chemical, current_recommendation, soil_type):
    """
    Check if switching chemicals is safe based on soil type
    Returns safety assessment
    """
    # Define incompatible chemical pairs
    incompatible_pairs = [
        ("Lime", "Sulfur"),
        ("Sulfur", "Lime"),
        ("Calcium Nitrate", "Potassium Sulfate"),  # May cause precipitation
    ]
    
    is_compatible = True
    for chem1, chem2 in incompatible_pairs:
        if (previous_chemical.strip() == chem1 and current_recommendation == chem2) or \
           (previous_chemical.strip() == chem2 and current_recommendation == chem1):
            is_compatible = False
            break
    
    if is_compatible:
        return {
            "safe": True,
            "message": f"Safe to use {current_recommendation} after {previous_chemical}",
            "wait_days": 0
        }
    else:
        return {
            "safe": False,
            "message": f"{current_recommendation} may react with residual {previous_chemical}. Recommendation: Wait 7-10 days.",
            "wait_days": 10
        }


def estimate_residue_level(chemical_name, application_amount, days_since_application):
    """
    Estimate remaining chemical residue in soil
    Based on degradation rates
    """
    # Degradation rates (half-life in days)
    degradation_rates = {
        "Nitrogen": 30,
        "Phosphorus": 180,
        "Potassium": 150,
        "Sulfur": 60,
        "Lime": 240,
        "Pesticide": 14,
        "Fungicide": 21,
        "Herbicide": 30,
        "NPK": 45,
        "DAP": 90,
        "Gypsum": 180,
        "Zinc Sulfate": 60,
        "Iron Sulfate": 90,
        "Boron": 120,
        "Copper Sulfate": 180
    }
    
    half_life = degradation_rates.get(chemical_name, 60)  # Default 60 days
    
    # Calculate remaining amount using exponential decay
    import math
    remaining_ratio = 0.5 ** (days_since_application / half_life)
    remaining_amount = application_amount * remaining_ratio
    
    return {
        "chemical": chemical_name,
        "remaining_amount": round(remaining_amount, 2),
        "remaining_percentage": round(remaining_ratio * 100, 1),
        "half_life_days": half_life,
        "days_since_application": days_since_application
    }

def agronomic_rule_check(ph, moisture, carbon, confidence_score=None):
    """
    Version-controlled agronomic safety layer, applied BEFORE a recommendation
    is shown to the farmer. This is deliberately separate from and upstream
    of the ML prediction / MOA optimization -- it encodes hard, hand-authored
    agronomic boundaries that no model output is allowed to override.

    Returns a dict with any flags raised and whether the recommendation
    must be routed to an agronomist/AEO for review before use.
    """
    flags = []

    if ph < 4.5 or ph > 9.0:
        flags.append(f"pH {ph} is outside the safe cultivable range (4.5-9.0)")
    if moisture < 5:
        flags.append(f"Moisture {moisture}% is critically low -- verify sensor/reading before recommending")
    if moisture > 90:
        flags.append(f"Moisture {moisture}% suggests waterlogging -- verify before recommending")
    if carbon is not None and carbon < 0:
        flags.append("Organic carbon reading is invalid (negative)")

    low_confidence = confidence_score is not None and confidence_score < 60

    return {
        "flags": flags,
        "low_confidence": low_confidence,
        "requires_agronomist_review": bool(flags) or low_confidence,
    }


def needs_agronomist_review(confidence_score, rule_flags):
    """
    Single gate used by the UI to decide whether a recommendation can go
    straight to the farmer, or must wait for agronomist/AEO sign-off.
    Low-confidence ML predictions and any agronomic rule-check flag both
    route to a human rather than auto-publishing the recommendation.
    """
    if rule_flags:
        return True
    if confidence_score is not None and confidence_score < 60:
        return True
    return False
