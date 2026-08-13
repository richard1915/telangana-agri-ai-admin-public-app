"""
Soil Biological Activity Indicator -- Waksman-inspired extension.

Adds a biological lens alongside the project's existing chemical-only
reasoning (pH, N, P, K, EC -- see backend/model.py). Soil fertility is a
biological process as much as a chemical one: the microbial community in
soil is what actually makes nutrients plant-available, and organic
carbon functions as that community's food supply rather than just a
nutrient figure in isolation. This traces to Selman Waksman's
foundational work in soil microbiology.

Scope and limits (read before using this module's output):
  - This project has no way to measure microbial biomass, species
    diversity, or enzyme activity -- there is no lab assay in the
    pipeline. Nothing here is a substitute for one.
  - What this module DOES do is combine data already collected
    elsewhere in the app -- organic carbon (microbial food supply),
    moisture and EC (both independently suppress microbial activity
    even when carbon is adequate), and the farmer's organic-carbon
    trend across seasons -- into one directional Low/Moderate/High
    label. This is deliberately more than a carbon-only re-label of
    the existing soil-health indicator in environmental_impact.py:
    two fields with identical carbon but very different moisture/EC
    will get different levels here.
  - Available N is reported as its own Low/Medium/High note, NOT
    folded into a C:N ratio. A real C:N ratio needs Total N % on the
    same basis as organic carbon %; this app collects Available N
    (kg/acre, Alkaline KMnO4 method) -- a smaller, different quantity
    that isn't convertible to Total N without regional lab calibration
    this project doesn't have. Reporting it standalone avoids
    presenting an invalid number as if it were meaningful.
  - This is presented as a future-work / decision-support signal for
    agronomist review (Project Review Report, Section 10) -- it does
    NOT currently feed into agronomic_rule_check(), the dose optimizer,
    or any recommendation shown to a farmer. It is display-only until
    an agronomist confirms the thresholds used here.
"""


def estimate_biological_activity(organic_carbon_pct, carbon_history=None, nitrogen=None,
                                  moisture=None, electrical_conductivity=None,
                                  total_nitrogen_pct=None):
    """
    Args:
        organic_carbon_pct: current soil organic carbon reading, %.
        carbon_history: optional list of this farmer/field's past organic
            carbon readings, oldest first (e.g. from
            database.fetch_farmer_soil_carbon_history()). Needs at least
            2 readings to report a trend; otherwise trend is "insufficient
            data".
        nitrogen: optional current Available N reading, kg/acre (as
            collected elsewhere in this app). Used only to report a
            Low/Medium/High available-N sufficiency flag alongside the
            carbon-based level -- see note below on why this is NOT
            combined into a C:N ratio.
        moisture: optional current soil moisture, %. Microbial activity
            drops off outside a workable moisture range even when carbon
            is adequate -- used to adjust the activity level down if
            moisture is very low or waterlogged.
        electrical_conductivity: optional EC, dS/m. High salinity
            suppresses microbial activity independently of carbon --
            used the same way as moisture, to adjust the level down.
        total_nitrogen_pct: optional LAB-VERIFIED Total Nitrogen, in %
            (same basis as organic_carbon_pct) -- e.g. from a Kjeldahl or
            CHNS lab test, NOT the Available N (kg/acre) field used
            elsewhere in this app. Only when this is supplied does a real
            C:N ratio get computed; without it, nitrogen sufficiency is
            reported as the Available-N band instead (see available_n_note).

    Returns a dict: biological_activity_level (Low/Moderate/High),
    level_note, organic_carbon_trend, available_n_note (or None),
    c_n_ratio / c_n_ratio_note (present only when total_nitrogen_pct is
    given), and a disclaimer that must be shown alongside the result
    wherever this is displayed.
    """
    if organic_carbon_pct is None:
        return {"available": False, "reason": "No organic carbon reading provided."}

    try:
        oc = float(organic_carbon_pct)
    except (TypeError, ValueError):
        return {"available": False, "reason": "Organic carbon reading is not a valid number."}

    if oc < 0:
        return {"available": False, "reason": "Organic carbon reading is invalid (negative)."}

    # Base score from organic carbon (the microbial food-supply proxy).
    # Thresholds kept consistent with the existing organic-carbon bonus in
    # backend/environmental_impact.py (>=0.75 / >=0.5) so the two don't
    # silently disagree with each other in front of the same user.
    if oc >= 0.75:
        score = 2
    elif oc >= 0.5:
        score = 1
    else:
        score = 0

    constraint_notes = []

    # Moisture and salinity don't feed the microbial community the way
    # carbon does, but both independently suppress activity even when
    # carbon is adequate -- so they adjust the score rather than replace
    # it. This is what differentiates this indicator from the OC-only
    # soil_health_improvement_indicator in environmental_impact.py.
    if moisture is not None:
        try:
            m = float(moisture)
            if m < 15:
                score -= 1
                constraint_notes.append(f"Moisture is low ({m}%) -- likely suppressing microbial activity even with adequate carbon.")
            elif m > 85:
                score -= 1
                constraint_notes.append(f"Moisture is very high ({m}%) -- waterlogging risk suppresses aerobic microbial activity.")
        except (TypeError, ValueError):
            pass

    if electrical_conductivity is not None:
        try:
            ec = float(electrical_conductivity)
            if ec > 4.0:
                score -= 1
                constraint_notes.append(f"EC is high ({ec} dS/m) -- salinity at this level typically suppresses microbial activity.")
        except (TypeError, ValueError):
            pass

    score = max(0, min(2, score))
    if score >= 2:
        level = "High"
        level_note = "Organic carbon is adequate and no moisture/salinity constraint was detected -- conditions typically support active microbial cycling."
    elif score == 1:
        level = "Moderate"
        level_note = "Microbial activity is likely constrained -- see notes below." if constraint_notes else "Organic carbon is adequate but not abundant -- microbial activity is likely constrained."
    else:
        level = "Low"
        level_note = "Conditions likely limit the soil's microbial community and its nutrient-cycling capacity -- see notes below." if constraint_notes else "Organic carbon this low likely limits the soil's microbial community and its nutrient-cycling capacity."

    trend = "insufficient data"
    if carbon_history:
        readings = [float(v) for v in carbon_history if v is not None]
        if len(readings) >= 2:
            delta = readings[-1] - readings[0]
            if delta > 0.05:
                trend = "improving"
            elif delta < -0.05:
                trend = "declining"
            else:
                trend = "stable"

    # Available-N sufficiency, reported standalone rather than combined
    # into a C:N ratio. Real C:N needs Total N % on the same basis as
    # organic carbon %; what this app collects by default is Available N
    # (kg/acre, Alkaline KMnO4 method) -- a different, much smaller
    # quantity that is not convertible to Total N without regional
    # calibration data this project doesn't have. Bands below follow the
    # ranges used on Indian Soil Health Cards for Available N.
    available_n_note = None
    if nitrogen is not None:
        try:
            n_val = float(nitrogen)
            if n_val < 280:
                n_band = "Low"
            elif n_val <= 560:
                n_band = "Medium"
            else:
                n_band = "High"
            available_n_note = (
                f"Available N: {n_band} ({n_val:.0f} kg/acre). Reported separately, not as a "
                f"C:N ratio -- this app measures Available N, not Total N, and the two are not "
                f"interchangeable without lab conversion."
            )
        except (TypeError, ValueError):
            pass

    # Real C:N ratio -- computed ONLY when a lab-verified Total N % is
    # supplied, on the same % basis as organic_carbon_pct. This is the
    # standard agronomic formula (OC% / Total N%); typical fertile
    # cultivated soils fall around 10-12. Deliberately kept separate from
    # available_n_note above so a caller can never accidentally get a
    # ratio built from the wrong nitrogen measurement.
    c_n_ratio = None
    c_n_ratio_note = None
    if total_nitrogen_pct is not None:
        try:
            tn = float(total_nitrogen_pct)
        except (TypeError, ValueError):
            tn = None
        if tn is not None and tn <= 0:
            c_n_ratio_note = "Lab Total Nitrogen reading is invalid (must be greater than 0) -- C:N ratio not computed."
        elif tn is not None:
            c_n_ratio = round(oc / tn, 1)
            if c_n_ratio < 8:
                band_note = "narrow -- nitrogen may mineralize/leach faster than crops can use it"
            elif c_n_ratio <= 12:
                band_note = "within the typical range for fertile cultivated soil (~10-12)"
            else:
                band_note = "wide -- decomposer microbes may tie up (immobilize) available nitrogen, temporarily limiting crop uptake"
            c_n_ratio_note = (
                f"C:N ratio {c_n_ratio}:1, based on lab-verified Total Nitrogen -- {band_note}."
            )

    result = {
        "available": True,
        "biological_activity_level": level,
        "level_note": level_note,
        "constraint_notes": constraint_notes,
        "organic_carbon_trend": trend,
        "available_n_note": available_n_note,
        "disclaimer": (
            "Proxy indicator derived from organic carbon, moisture, and EC already collected "
            "elsewhere in this app. This project does not measure microbial biomass, species "
            "diversity, or enzyme activity. Shown as a future-work / decision-support signal for "
            "agronomist review -- it does not influence any recommendation or dose in this app "
            "(Project Review Report, Section 10)."
        ),
    }
    if total_nitrogen_pct is not None:
        result["c_n_ratio"] = c_n_ratio
        result["c_n_ratio_note"] = c_n_ratio_note
    return result
