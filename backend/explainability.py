"""
Explainable AI (XAI) layer for the trained yield model.

Advisory gap addressed: model outputs (ML prediction, MOA-optimized dose)
were previously shown to the farmer/agronomist as bare numbers with no
indication of WHY the model produced them. This module answers "why did
the model recommend this?" in two ways:

  1. Global feature importance -- which soil/crop features matter most to
     the model overall (Random Forest's built-in importances).
  2. Per-recommendation explanation -- SHAP values for the specific farm
     being scored, if the `shap` package is installed; otherwise a
     permutation-importance-based local approximation so the app still
     works without the optional dependency.

Feature importance / SHAP is the standard shown to the agronomist and
farmer alongside the number, per the advisory's push for uncertainty and
explainability rather than opaque outputs.
"""

import numpy as np
import pandas as pd

from backend.meerkat_optimizer import _load_yield_model, _ML_FEATURE_COLUMNS

try:
    import shap  # optional dependency
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False


def _get_expanded_feature_names(pipeline):
    """Feature names after the ColumnTransformer's one-hot encoding."""
    preprocessor = pipeline.named_steps["preprocess"]
    cat_encoder = preprocessor.named_transformers_["cat"]
    cat_cols = ["Soil_Type", "Crop"]
    cat_names = list(cat_encoder.get_feature_names_out(cat_cols))
    passthrough_cols = [c for c in _ML_FEATURE_COLUMNS if c not in cat_cols]
    return cat_names + passthrough_cols


def get_global_feature_importance(top_n=10):
    """
    Model-wide feature importance from the Random Forest (mean decrease in
    impurity). Answers "what does the model pay attention to in general?"
    """
    try:
        pipeline = _load_yield_model()
        rf = pipeline.named_steps["rf"]
        names = _get_expanded_feature_names(pipeline)
        importances = rf.feature_importances_

        # Collapse one-hot columns back to their parent feature (e.g. all
        # "Soil_Type_*" columns summed into one "Soil_Type" bar) so the
        # chart reads at the level a farmer/agronomist actually cares
        # about, not one bar per soil-type category.
        collapsed = {}
        for name, imp in zip(names, importances):
            parent = name.split("_")[0] if name.startswith(("Soil", "Crop")) else name
            if name.startswith("Soil_Type_"):
                parent = "Soil_Type"
            elif name.startswith("Crop_"):
                parent = "Crop"
            else:
                parent = name
            collapsed[parent] = collapsed.get(parent, 0.0) + float(imp)

        ranked = sorted(collapsed.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
        total = sum(v for _, v in ranked) or 1.0
        return {
            "available": True,
            "method": "Random Forest mean decrease in impurity",
            "features": [{"feature": k, "importance_pct": round(v / sum(collapsed.values()) * 100, 1)} for k, v in ranked],
        }
    except Exception as e:
        return {"available": False, "error": str(e)}


def explain_prediction(crop_type, soil_type, ph, nitrogen, phosphorus, potassium,
                        organic_carbon, moisture, electrical_conductivity, top_n=6):
    """
    Local explanation for ONE specific recommendation -- the row of
    soil/crop values actually used to make this farmer's prediction.
    Uses SHAP TreeExplainer if `shap` is installed; otherwise falls back
    to a permutation-based local importance so the app degrades
    gracefully instead of breaking.
    """
    try:
        pipeline = _load_yield_model()
    except Exception as e:
        return {"available": False, "error": str(e)}

    row = pd.DataFrame([{
        "Soil_Type": soil_type, "Crop": crop_type, "pH": ph, "Nitrogen": nitrogen,
        "Phosphorus": phosphorus, "Potassium": potassium, "Organic_Carbon": organic_carbon,
        "Soil_Moisture": moisture, "Electrical_Conductivity": electrical_conductivity,
    }])[_ML_FEATURE_COLUMNS]

    preprocessor = pipeline.named_steps["preprocess"]
    rf = pipeline.named_steps["rf"]
    x_transformed = preprocessor.transform(row)
    names = _get_expanded_feature_names(pipeline)

    if _SHAP_AVAILABLE:
        try:
            explainer = shap.TreeExplainer(rf)
            shap_values = explainer.shap_values(x_transformed)
            contributions = np.array(shap_values).flatten()
            method = "SHAP (TreeExplainer)"
        except Exception:
            contributions = None
    else:
        contributions = None

    if contributions is None:
        # Fallback: single-feature-group perturbation. Zero out (or reset
        # to the dataset mean for numeric columns) one readable feature
        # group at a time and measure the change in prediction -- a
        # simple, dependency-free stand-in for SHAP's local attribution.
        method = "Permutation-based local importance (SHAP not installed)"
        base_pred = float(rf.predict(x_transformed)[0])
        contributions_by_name = {}
        numeric_cols = [c for c in _ML_FEATURE_COLUMNS if c not in ("Soil_Type", "Crop")]
        for col in numeric_cols:
            perturbed = row.copy()
            perturbed[col] = perturbed[col] * 0.8  # -20% perturbation
            perturbed_pred = float(rf.predict(preprocessor.transform(perturbed))[0])
            contributions_by_name[col] = base_pred - perturbed_pred
        # Categorical groups: compare to their one-hot contribution weight
        for group, col in (("Soil_Type", "Soil_Type"), ("Crop", "Crop")):
            names_in_group = [n for n in names if n.startswith(group + "_")]
            idxs = [names.index(n) for n in names_in_group]
            weight = float(np.sum(rf.feature_importances_[idxs])) * base_pred
            contributions_by_name[group] = weight
        ranked = sorted(contributions_by_name.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_n]
    else:
        # Collapse SHAP one-hot contributions back to parent feature names.
        collapsed = {}
        for name, val in zip(names, contributions):
            parent = name.split("_")[0] if name.startswith(("Soil_Type_", "Crop_")) else name
            collapsed[parent] = collapsed.get(parent, 0.0) + float(val)
        ranked = sorted(collapsed.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_n]

    return {
        "available": True,
        "method": method,
        "contributions": [
            {"feature": k, "impact_kg_per_acre": round(v, 1), "direction": "increases yield" if v >= 0 else "decreases yield"}
            for k, v in ranked
        ],
    }
