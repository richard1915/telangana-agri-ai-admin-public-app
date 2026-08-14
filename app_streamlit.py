import os
import tempfile
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import folium
from folium.plugins import MiniMap, Fullscreen, MeasureControl, MousePosition, LocateControl, HeatMap, Geocoder, Draw
from streamlit_folium import st_folium
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from backend.database import create_db, insert_soil_entry
from backend.database import (
    insert_student_data,
    insert_farmer_data,
    insert_ml_result,
    insert_moa_result,
    fetch_all,
    fetch_farmer_moa_history,
    fetch_farmer_season_summary,
    insert_harvest_outcome,
    fetch_harvest_accuracy_summary,
    insert_agronomist_review,
    fetch_farmer_review_history,
    fetch_farmer_soil_carbon_history,
    insert_soil_submission,
    fetch_pending_submissions,
    mark_submission_processed,
    fetch_farmer_by_name,
)
from backend.model import (
    recommend_crop,
    recommend_chemical_by_soil_type,
    estimate_residue_level,
    check_chemical_safety,
    agronomic_rule_check,
    needs_agronomist_review,
)
from backend.meerkat_optimizer import (
    meerkat_chemical_reduction, predict_crop_yield, predict_crop_yield_ml,
    train_yield_model, get_model_metrics,
)
from backend.optimizer_comparison import compare_optimizers
from backend.explainability import get_global_feature_importance, explain_prediction
from backend.cost_savings import estimate_cost_savings
from backend.environmental_impact import estimate_environmental_impact
from backend.biological_health import estimate_biological_activity
from backend.report_generator import generate_pdf_report
from backend.satellite_api import get_soil_type_for_location
from backend.weather_api import get_weather_estimate_for_prediction
from backend.precision_spray import generate_field_grid, satellite_screen_field, simulate_drone_thermal_scan, generate_prescription_map
from backend.crop_health_imaging import generate_field_health_layers, LAYER_DEFINITIONS
from backend.chemical_composition import compute_nutrients_delivered, dose_to_close_nutrient_gap, compare_chemicals_for_nutrient_gap, infer_chemical_category
from backend.awareness_bot import ask_awareness_bot, is_gemini_configured, is_groq_configured, STATIC_FAQ
from backend.geocoding import geocode_address

# --- Admin (laptop, full pipeline) vs Public (Render, students) mode ---
# APP_MODE=admin (default, e.g. local .env) -> full app behind a login,
# including Run ML / Run MOA / Save results -- this is the only mode
# that ever calls the ML/optimizer backend.
# APP_MODE=public (set in render.yaml for the deployed service) ->
# students can only submit data and view already-saved results; no
# login, and the pages that run the model are not in the nav at all.
APP_MODE = (os.getenv("APP_MODE") or "admin").strip().lower()
IS_ADMIN_MODE = APP_MODE != "public"
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "")

st.set_page_config(page_title="Telangana Agriculture AI", layout="wide", page_icon="🌾")


def require_admin_login():
    """
    Simple shared-password gate for admin mode. Not meant to be
    bank-grade security -- this app is only ever reachable on the
    admin's own laptop (APP_MODE=admin is never set on the deployed
    Render service), so the gate exists to stop someone who's borrowed
    the laptop from casually opening it, not to resist a real attacker.
    """
    if st.session_state.get("admin_authed"):
        return
    st.markdown(
        """
        <div class="app-hero">
          <h1>🌾 Telangana Smart Agriculture AI -- Admin</h1>
          <p>Local admin mode. Log in to run ML/MOA analysis and save results.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if not ADMIN_PASSWORD:
        st.error(
            "ADMIN_PASSWORD is not set. Add ADMIN_PASSWORD=<your password> to your local "
            ".env file, then restart the app."
        )
        st.stop()
    with st.form("admin_login_form"):
        entered = st.text_input("Admin password", type="password")
        submitted = st.form_submit_button("Log in", use_container_width=True)
    if submitted:
        if entered == ADMIN_PASSWORD:
            st.session_state["admin_authed"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()


@st.cache_resource
def _init_db():
    """
    Streamlit re-executes this whole script top-to-bottom on every widget
    interaction, so calling create_db() unconditionally at module level
    ran it dozens of times per session. metadata.create_all() is meant to
    be idempotent, but two of those calls racing against each other (e.g.
    while the DB file is still being created) could both try to CREATE
    TABLE at once -- one wins, the other gets an "already exists" error.
    @st.cache_resource makes this run exactly once for the life of the
    app process, which removes the race instead of just tolerating it.
    """
    create_db()
    return True


_init_db()

st.markdown(
    """
    <style>
    :root {
        --tg-bg: #ddd0b0;
        --tg-bg-alt: #cbb98f;
        --tg-card: #e8dcc0;
        --tg-green: #2e7d32;
        --tg-green-dark: #1b4d1e;
        --tg-brown: #6d4c30;
        --tg-brown-dark: #4a3728;
        --tg-border: #b7a274;
        --tg-text: #2a2318;
    }

    .stApp, body, [data-testid="stAppViewContainer"] { background-color: var(--tg-bg); }
    .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1200px; }

    [data-testid="stHeader"] { background-color: var(--tg-bg); }

    [data-testid="stSidebar"] { background-color: var(--tg-bg-alt); border-right: 1px solid var(--tg-border); }
    [data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] p { color: var(--tg-brown-dark); }

    .app-hero {
        background: linear-gradient(135deg, var(--tg-green-dark) 0%, var(--tg-green) 55%, var(--tg-brown) 100%);
        color: #ffffff;
        padding: 1.5rem 1.75rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        border: 1px solid var(--tg-brown-dark);
        box-shadow: 0 2px 10px rgba(27, 77, 30, 0.18);
    }
    .app-hero h1 { color: #ffffff; margin: 0; font-size: 1.6rem; }
    .app-hero p { color: #ead9c3; margin: 0.35rem 0 0 0; font-size: 0.95rem; }

    .page-header {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        padding-bottom: 0.4rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid var(--tg-brown);
    }
    .page-header .icon { font-size: 1.6rem; }
    .page-header h2 { margin: 0; padding: 0; font-size: 1.4rem; color: var(--tg-green-dark); }

    /* Body text and headings on the warm background */
    p, li, span, label, .stMarkdown { color: var(--tg-text); }
    h1, h2, h3, .streamlit-expanderHeader { color: var(--tg-green-dark); }

    div[data-testid="stMetric"] {
        background-color: var(--tg-card);
        border: 1px solid var(--tg-border);
        border-left: 4px solid var(--tg-green);
        border-radius: 10px;
        padding: 0.85rem 1rem;
        box-shadow: 0 1px 3px rgba(74, 55, 40, 0.08);
    }
    div[data-testid="stMetricLabel"] { color: var(--tg-brown); }
    div[data-testid="stMetricValue"] { color: var(--tg-green-dark); }

    div.stButton > button, div.stFormSubmitButton > button, div.stDownloadButton > button {
        border-radius: 8px;
        border: 1px solid var(--tg-green);
        color: var(--tg-green-dark);
        background-color: var(--tg-card);
    }
    div.stButton > button[kind="primary"] { background-color: var(--tg-green); border-color: var(--tg-green-dark); color: #ffffff; }
    div.stButton > button:hover, div.stFormSubmitButton > button:hover { border-color: var(--tg-brown); color: var(--tg-brown); }

    [data-testid="stExpander"] { background-color: var(--tg-card); border: 1px solid var(--tg-border); border-radius: 10px; }
    [data-testid="stExpander"] summary { color: var(--tg-brown); }

    /* Forms, text inputs, selects -- replace default white fields */
    [data-testid="stForm"] { background-color: var(--tg-card); border: 1px solid var(--tg-border); border-radius: 12px; padding: 1rem 1.2rem; }
    div[data-baseweb="input"], div[data-baseweb="select"], div[data-baseweb="textarea"], textarea {
        background-color: var(--tg-card) !important;
        border-color: var(--tg-border) !important;
    }

    /* Tables / dataframes */
    [data-testid="stDataFrame"], [data-testid="stTable"] { background-color: var(--tg-card); border: 1px solid var(--tg-border); border-radius: 8px; }

    /* Tabs -- brown underline for the earthy accent */
    button[data-baseweb="tab"] { color: var(--tg-brown); }
    button[data-baseweb="tab"][aria-selected="true"] { color: var(--tg-green); border-bottom-color: var(--tg-brown) !important; }

    /* Info / success / warning boxes tinted to match instead of default flat colors */
    div[data-testid="stAlert"] { border-radius: 10px; border: 1px solid var(--tg-border); }

    hr { margin: 1.5rem 0; border-color: var(--tg-border); }
    </style>
    """,
    unsafe_allow_html=True,
)

PAGE_ICONS = {
    "Area": "📍",
    "Dashboard": "🏠",
    "Student Registration": "🎓",
    "Farmer Details": "👨‍🌾",
    "Soil Information": "🧪",
    "ML & MOA Analysis": "🤖",
    "Maps": "🗺️",
    "Precision Spraying (Stage 2)": "🚁",
    "Farmer Awareness": "💬",
    "Harvest Outcomes": "🌾",
    "Database": "🗄️",
    # Public (student) read-only results pages -- icons match what was
    # asked for in the public-mode nav spec.
    "Crop Results": "🌱",
    "Soil Results": "🧪",
    "ML/MOA Results": "🤖",
    "Charts": "📊",
    "Project Results": "📈",
}


def page_header(title: str, subtitle: str = None, icon: str = None):
    """Consistent icon + title header used at the top of every page."""
    icon = icon or PAGE_ICONS.get(title, "•")
    st.markdown(
        f'<div class="page-header"><span class="icon">{icon}</span><h2>{title}</h2></div>',
        unsafe_allow_html=True,
    )
    if subtitle:
        st.caption(subtitle)


# Shared chart palette -- dark green / brown to match the app theme instead
# of Streamlit's default flat blue, with bold dark labels for readability
# on the ivory/tan background.
CHART_GREEN = "#2e7d32"
CHART_BROWN = "#6d4c30"
CHART_GOLD = "#c8963e"
CHART_RED = "#a5432f"
CHART_LABEL_COLOR = "#241f1a"
CHART_CATEGORY_RANGE = [CHART_GREEN, CHART_BROWN, CHART_GOLD, "#4a7c59", "#8a6d3b"]


def themed_bar_chart(df: pd.DataFrame, category_col: str, value_col: str,
                      color: str = CHART_GREEN, diverging: bool = False, horizontal: bool = False):
    """
    Bar chart styled to match the app's dark green / brown theme with bold,
    high-contrast axis labels -- replaces st.bar_chart's default thin grey
    text and flat blue bars, which were hard to read on the ivory background.
    If diverging=True, bars are colored green (>=0) / brown (<0) -- useful
    for feature-contribution charts where sign matters.
    """
    base = alt.Chart(df)
    label_font = alt.Axis(
        labelColor=CHART_LABEL_COLOR, titleColor=CHART_LABEL_COLOR,
        labelFontSize=12, titleFontSize=12, labelFontWeight="bold", titleFontWeight="bold",
        gridColor="#b7a274",
    )
    color_enc = (
        alt.condition(f"datum.{value_col} >= 0", alt.value(CHART_GREEN), alt.value(CHART_BROWN))
        if diverging else alt.value(color)
    )
    if horizontal:
        chart = base.mark_bar(cornerRadiusTopRight=3, cornerRadiusBottomRight=3).encode(
            x=alt.X(f"{value_col}:Q", axis=label_font),
            y=alt.Y(f"{category_col}:N", sort="-x", axis=label_font),
            color=color_enc,
            tooltip=[category_col, value_col],
        )
    else:
        chart = base.mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
            x=alt.X(f"{category_col}:N", axis=label_font, sort=None),
            y=alt.Y(f"{value_col}:Q", axis=label_font),
            color=color_enc,
            tooltip=[category_col, value_col],
        )
    st.altair_chart(
        chart.configure_view(strokeWidth=0).properties(height=320),
        use_container_width=True,
    )


def themed_donut_chart(df: pd.DataFrame, category_col: str, value_col: str,
                        colors: list = None):
    """Donut/pie chart in the theme palette with bold legend/tooltip text."""
    colors = colors or CHART_CATEGORY_RANGE
    chart = alt.Chart(df).mark_arc(innerRadius=60, stroke="#fbf8f2", strokeWidth=2).encode(
        theta=alt.Theta(f"{value_col}:Q"),
        color=alt.Color(
            f"{category_col}:N",
            scale=alt.Scale(range=colors),
            legend=alt.Legend(labelColor=CHART_LABEL_COLOR, titleColor=CHART_LABEL_COLOR,
                               labelFontSize=12, labelFontWeight="bold"),
        ),
        tooltip=[category_col, value_col],
    )
    st.altair_chart(chart.properties(height=320), use_container_width=True)


TELANGANA_BOUNDS = {
    "lat_min": 15.75,
    "lat_max": 19.95,
    "lon_min": 77.20,
    "lon_max": 81.10,
}

BASELINE_DOSE_BY_CROP = {
    "Rice": 120.0,
    "Cotton": 60.0,
    "Groundnut": 40.0,
    "Sugarcane": 140.0,
    "Maize": 100.0,
    "Sorghum": 60.0,
    "Pulses": 40.0,
    "Tobacco": 80.0,
    "Tomato": 90.0,
    "Brinjal": 95.0,
    "Onion": 85.0,
    "Chilli": 100.0,
    "Bhindi": 75.0,
    "Cabbage": 90.0,
    "Cauliflower": 90.0,
    # Telangana priority crops
    "Paddy": 120.0,
    "Turmeric": 90.0,
    "Red Gram": 35.0,
    "Green Gram": 30.0,
    "Black Gram": 30.0,
    "Bengal Gram": 35.0,
    "Bajra": 55.0,
    "Sesame": 35.0,
    "Castor": 55.0,
    "Soybean": 45.0,
    "Sunflower": 55.0,
}


def init_state():
    defaults = {
        "student_registration": {},
        "farmer_details": {},
        "soil_information": {},
        "map_lat": 17.3850,
        "map_lon": 78.4867,
        "shape_points": [],
        "last_click_sig": "",
        "last_moa_result": {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def is_within_telangana(lat: float, lon: float) -> bool:
    return (
        TELANGANA_BOUNDS["lat_min"] <= lat <= TELANGANA_BOUNDS["lat_max"]
        and TELANGANA_BOUNDS["lon_min"] <= lon <= TELANGANA_BOUNDS["lon_max"]
    )


def normalize_soil_type(soil_type: str) -> str:
    """
    Maps a Soil Information dropdown selection to the keys used in
    backend/model.py's chemical_database. Telangana's 7 official soil
    types (Chalka, Dubba, Lateritic, Shallow-Medium Black, Deep Black,
    Salt-affected, Alluvial) pass through unchanged; older generic names
    are kept as aliases for backward compatibility with existing records.
    """
    mapping = {
        "Chalka (Red Sandy Loam)": "Chalka (Red Sandy Loam)",
        "Dubba (Red Loamy Sand)": "Dubba (Red Loamy Sand)",
        "Lateritic Soil": "Lateritic Soil",
        "Shallow-Medium Black Soil": "Shallow-Medium Black Soil",
        "Deep Black Soil (Black Cotton)": "Deep Black Soil (Black Cotton)",
        "Salt-affected Soil": "Salt-affected Soil",
        "Alluvial Soil": "Alluvial Soil",
        # Legacy generic names, kept for backward compatibility
        "Red Soil": "Chalka (Red Sandy Loam)",
        "Black Soil": "Shallow-Medium Black Soil",
        "Laterite Soil": "Lateritic Soil",
        "Laterite": "Lateritic Soil",
        "Alluvial": "Alluvial Soil",
    }
    return mapping.get(soil_type, "Chalka (Red Sandy Loam)")


def normalize_prev_chemical(chemical: str) -> str:
    mapping = {
        "Urea": "Nitrogen",
        "NPK": "NPK",
        "DAP": "DAP",
        "Lime": "Lime",
        "Sulfur": "Sulfur",
        "Pesticide": "Pesticide",
        "Fungicide": "Fungicide",
        "Herbicide": "Herbicide",
        "Gypsum": "Gypsum",
        "Zinc Sulfate": "Zinc Sulfate",
        "Iron Sulfate": "Iron Sulfate",
    }
    return mapping.get(chemical, "NPK")


def load_telangana_districts() -> list[str]:
    fallback = [
        "Adilabad",
        "Bhadradri Kothagudem",
        "Hanumakonda",
        "Hyderabad",
        "Jagtial",
        "Jangaon",
        "Jayashankar Bhupalpally",
        "Jogulamba Gadwal",
        "Kamareddy",
        "Karimnagar",
        "Khammam",
        "Komaram Bheem Asifabad",
        "Mahabubabad",
        "Mahabubnagar",
        "Mancherial",
        "Medak",
        "Medchal Malkajgiri",
        "Mulugu",
        "Nagarkurnool",
        "Nalgonda",
        "Narayanpet",
        "Nirmal",
        "Nizamabad",
        "Peddapalli",
        "Rajanna Sircilla",
        "Ranga Reddy",
        "Sangareddy",
        "Siddipet",
        "Suryapet",
        "Vikarabad",
        "Wanaparthy",
        "Warangal",
        "Yadadri Bhuvanagiri",
    ]
    try:
        df = pd.read_csv("dataset/telangana_soil_data.csv")
        if "District" in df.columns:
            districts = sorted(df["District"].dropna().astype(str).str.strip().unique().tolist())
            if len(districts) >= 33:
                return districts
    except Exception:
        pass
    return fallback


def load_soil_dataset() -> pd.DataFrame:
    try:
        return pd.read_csv("dataset/telangana_soil_data.csv")
    except Exception:
        return pd.DataFrame()


def classify_land_and_water(soil_info: dict) -> tuple[str, str]:
    ndvi = soil_info.get("ndvi", 0)
    ndmi = soil_info.get("ndmi", 0)
    bsi = soil_info.get("bsi", 0)

    if ndvi < -0.05:
        land = "Water body / flooded patch"
    elif ndvi < 0.2:
        land = "Barren or built-up land"
    elif bsi > 0.25:
        land = "Bare agricultural land"
    else:
        land = "Cultivated vegetation land"

    if ndmi > 0.4:
        water = "High water/moisture presence"
    elif ndmi > 0.2:
        water = "Moderate water/moisture presence"
    else:
        water = "Low water/moisture presence"

    return land, water


def _interpolate_gradient_color(value: float, gradient: dict) -> str:
    """
    Linear interpolation between a folium HeatMap-style gradient dict
    (e.g. {0.2: '#2ecc71', 0.5: '#f1c40f', ...}) so grid-cell rectangles
    can use the exact same color scale as the smooth heatmap, instead of
    a separate hardcoded palette that could drift out of sync with it.
    """
    stops = sorted(gradient.items())
    value = max(stops[0][0], min(stops[-1][0], value))
    for (v0, c0), (v1, c1) in zip(stops, stops[1:]):
        if v0 <= value <= v1:
            t = 0.0 if v1 == v0 else (value - v0) / (v1 - v0)
            r0, g0, b0 = int(c0[1:3], 16), int(c0[3:5], 16), int(c0[5:7], 16)
            r1, g1, b1 = int(c1[1:3], 16), int(c1[3:5], 16), int(c1[5:7], 16)
            r = round(r0 + (r1 - r0) * t)
            g = round(g0 + (g1 - g0) * t)
            b = round(b0 + (b1 - b0) * t)
            return f"#{r:02x}{g:02x}{b:02x}"
    return stops[-1][1]


def _add_map_legend(fmap, title: str, gradient: dict):
    """Small fixed-position HTML legend on the map, matching the gradient in use."""
    stops = sorted(gradient.items())
    swatches = "".join(
        f'<div style="display:flex;align-items:center;margin:2px 0;">'
        f'<span style="display:inline-block;width:14px;height:14px;background:{color};'
        f'border-radius:3px;margin-right:6px;border:1px solid #4a3728;"></span>'
        f'<span style="font-size:12px;color:#2a2318;">{value:.1f}</span></div>'
        for value, color in stops
    )
    legend_html = f"""
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999;
                background: #e8dcc0; padding: 10px 12px; border-radius: 8px;
                border: 1px solid #6d4c30; box-shadow: 0 2px 6px rgba(0,0,0,0.3);
                font-family: sans-serif;">
        <div style="font-weight:bold; font-size:12px; color:#1b4d1e; margin-bottom:4px;">{title}</div>
        {swatches}
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(legend_html))


def classify_cultivation_suitability(soil_info: dict, land_type: str, water_status: str) -> tuple[str, str]:
    """
    A direct, explicit "is this land good for cultivation" verdict --
    built from the same NDVI/NDMI/BSI already computed for land type and
    water presence, but stated as a plain suitability call instead of
    making the farmer/agronomist infer it from the raw index labels.
    Returns (verdict, reason).
    """
    ndvi = soil_info.get("ndvi", 0)
    ndmi = soil_info.get("ndmi", 0)

    if "Water body" in land_type:
        return "Not suitable", "This point looks like standing water or a flooded patch, not land."
    if "Barren or built-up" in land_type:
        return "Not suitable", "Very low vegetation signal -- likely bare/built-up ground, not cropland."
    if ndmi < 0.1:
        return "Marginal", "Land looks cultivable, but moisture is very low -- irrigation access should be confirmed before planting."
    if "Bare agricultural" in land_type and ndmi < 0.2:
        return "Marginal", "Bare/fallow field with low moisture -- workable, but check soil moisture and irrigation before committing to a crop."
    if ndvi >= 0.2:
        return "Suitable", "Vegetation and moisture signals both look workable for cultivation."
    return "Marginal", "Mixed signal -- worth a field visit before treating this as confirmed cropland."


def render_residue_graph(initial_amount: float, half_life_days: int, current_day: int):
    days = np.linspace(0, max(10, half_life_days * 3), 120)
    residue_curve = initial_amount * (0.5 ** (days / max(1, half_life_days)))
    current_value = initial_amount * (0.5 ** (current_day / max(1, half_life_days)))
    if plt is not None:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(days, residue_curve, linewidth=2, label="Residue")
        ax.axvline(current_day, linestyle="--", color="red", label=f"Current day: {current_day}")
        ax.axhline(current_value, linestyle=":", color="orange")
        ax.set_title("ML Residue Decay Curve")
        ax.set_xlabel("Days Since Application")
        ax.set_ylabel("Estimated Residue (kg/acre)")
        ax.grid(alpha=0.3)
        ax.legend()
        st.pyplot(fig)
    else:
        chart_df = pd.DataFrame({"day": days, "residue_kg_per_acre": residue_curve}).set_index("day")
        st.line_chart(chart_df, use_container_width=True)
        st.caption("Matplotlib not installed, showing Streamlit chart fallback.")

    st.caption(f"Current day: {current_day} | Estimated residue now: {current_value:.2f} kg/acre")


def soil_fertility_score(soil: dict) -> float:
    ph = float(soil.get("ph", 6.5))
    oc = float(soil.get("organic_carbon", 0.8))
    ec = float(soil.get("electrical_conductivity", 0.3))
    moisture = float(soil.get("moisture", 35.0))

    ph_score = max(0.0, 100.0 - abs(ph - 6.8) * 25.0)
    oc_score = min(100.0, (oc / 1.0) * 100.0)
    ec_score = max(0.0, 100.0 - max(0.0, ec - 0.8) * 80.0)
    moisture_score = max(0.0, 100.0 - abs(moisture - 35.0) * 3.0)
    return round((ph_score * 0.3) + (oc_score * 0.35) + (ec_score * 0.2) + (moisture_score * 0.15), 1)


init_state()

if IS_ADMIN_MODE:
    require_admin_login()

st.markdown(
    """
    <div class="app-hero">
      <h1>🌾 Telangana Smart Agriculture AI</h1>
      <p>Project scope is restricted to Telangana only.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Admin (laptop) gets the full application.
# Public (Render) gets ONLY the read-only dashboard/results.
ADMIN_ONLY_PAGES = {
    "ML & MOA Analysis",
    "Precision Spraying (Stage 2)",
    "Database",
    "Harvest Outcomes",
}

PUBLIC_RESULT_PAGES = [
    "Crop Results",
    "Soil Results",
    "ML/MOA Results",
    "Charts",
    "Project Results",
]

if IS_ADMIN_MODE:
    nav_labels = list(PAGE_ICONS.keys())
    if "admin_authed" in st.session_state:
        st.sidebar.success("Admin mode")
        if st.sidebar.button("Log out"):
            st.session_state["admin_authed"] = False
            st.rerun()

else:
    # PUBLIC/STUDENT MODE:
    # Students can register themselves and enter farmer + soil data.
    # ML/MOA processing remains ADMIN ONLY.
    nav_labels = [
        "Area",
        "Student Registration",
        "Farmer Details",
        "Soil Information",
        "Dashboard",
        "Crop Results",
        "Soil Results",
        "ML/MOA Results",
        "Charts",
        "Project Results",
    ]

    st.sidebar.info("📊 Student / Public Mode")

page = st.sidebar.radio(
    "Navigation",
    nav_labels,
    format_func=lambda p: f"{PAGE_ICONS.get(p, '•')}  {p}",
)


if page == "Area":
    page_header(
        "Service Area",
        subtitle="This project covers Telangana state only. Start here, then move through the workflow below.",
        icon=PAGE_ICONS["Area"],
    )

    st.markdown(
        """
        <div class="app-hero">
          <h1>🌾 Telangana Smart Agriculture AI</h1>
          <p>Coverage: all 33 districts of Telangana. Data entered outside this area is not supported.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    area_col1, area_col2 = st.columns([2, 1])
    with area_col1:
        area_map = folium.Map(
            location=[
                (TELANGANA_BOUNDS["lat_min"] + TELANGANA_BOUNDS["lat_max"]) / 2,
                (TELANGANA_BOUNDS["lon_min"] + TELANGANA_BOUNDS["lon_max"]) / 2,
            ],
            zoom_start=7,
        )
        folium.Rectangle(
            bounds=[
                [TELANGANA_BOUNDS["lat_min"], TELANGANA_BOUNDS["lon_min"]],
                [TELANGANA_BOUNDS["lat_max"], TELANGANA_BOUNDS["lon_max"]],
            ],
            color="#2e7d32",
            weight=2,
            fill=True,
            fill_color="#2e7d32",
            fill_opacity=0.08,
            tooltip="Telangana service area",
        ).add_to(area_map)
        folium.Marker(
            [17.3850, 78.4867],
            tooltip="Hyderabad (state capital)",
            icon=folium.Icon(color="green", icon="home"),
        ).add_to(area_map)
        st_folium(area_map, height=420, width=None, returned_objects=[], key="area_overview_map")

    with area_col2:
        st.subheader("Districts Covered")
        district_list = load_telangana_districts()
        st.caption(f"{len(district_list)} districts")
        st.dataframe(pd.DataFrame({"District": district_list}), use_container_width=True, height=380, hide_index=True)

    st.divider()
    st.subheader("How this app works")
    st.write(
        "This is a step-by-step tool for agriculture students to collect and analyze real farm data "
        "in Telangana. Follow the steps in order using the sidebar:"
    )
    workflow_steps = [
        ("📍 Area", "You're here. Confirms this project only covers Telangana."),
        ("🎓 Student Registration", "The student using the app registers themselves once."),
        ("👨‍🌾 Farmer Details", "The student enters the farmer's details, farm location and size."),
        ("💾 Saved automatically", "Farmer details are saved to the database as soon as the form is submitted."),
        ("🧪 Soil Information", "The student enters (or uploads) the farmer's soil test values."),
        ("🤖 ML & MOA Analysis onward", "Crop recommendation, dose optimization, maps and reports."),
    ]
    for step_title, step_desc in workflow_steps:
        st.markdown(f"**{step_title}** — {step_desc}")

    st.info("Use the sidebar to go to **Student Registration** next.")


elif page == "Dashboard":
    page_header(
        "Dashboard",
        subtitle="Read-only project results from completed analyses.",
        icon="📊",
    )

    # PUBLIC DASHBOARD
    if not IS_ADMIN_MODE:
        ml_records = fetch_all("ml_results")
        moa_records = fetch_all("moa_results")
        soil_records = fetch_all("soil_entries")
        submissions = fetch_all("soil_submissions")

        # Summary metrics
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric("Analyses Completed", len(moa_records))

        with c2:
            st.metric("Crop Predictions", len(ml_records))

        with c3:
            st.metric("Soil Records", len(soil_records))

        with c4:
            pending = sum(
                1 for row in submissions
                if row.get("status") == "pending"
            )
            st.metric("Pending Submissions", pending)

        st.divider()

        # Latest crop/ML results
        st.subheader("🌱 Latest Crop & Yield Results")

        if ml_records:
            ml_df = pd.DataFrame(ml_records)

            columns = [
                c for c in [
                    "created_at",
                    "farmer_name",
                    "next_crop",
                    "recommended_crop",
                    "predicted_yield",
                    "confidence_score",
                    "limiting_factor",
                ]
                if c in ml_df.columns
            ]

            st.dataframe(
                ml_df[columns].tail(20),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No completed ML results available yet.")

        st.divider()

        # Latest MOA results
        st.subheader("🤖 Latest MOA Chemical Optimization Results")

        if moa_records:
            moa_df = pd.DataFrame(moa_records)

            columns = [
                c for c in [
                    "created_at",
                    "farmer_name",
                    "next_crop",
                    "initial_dose",
                    "optimized_dose",
                    "reduction_percentage",
                ]
                if c in moa_df.columns
            ]

            st.dataframe(
                moa_df[columns].tail(20),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No completed MOA results available yet.")

        st.divider()

        st.success(
            "This is a read-only student dashboard. "
            "ML and MOA processing is performed on the administrator's local system."
        )

    # ADMIN DASHBOARD
    else:
        pending_count = len(fetch_pending_submissions())

        if pending_count:
            st.warning(
                f"📥 {pending_count} student submission(s) waiting "
                "for ML & MOA analysis."
            )
        else:
            st.success("No pending student submissions.")

        st.write(
            "Workflow: Area → Student Registration → Farmer Details → "
            "Soil Information → ML & MOA Analysis → Maps → Precision Spraying"
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Student Registered",
                "Yes" if st.session_state.student_registration else "No"
            )

        with col2:
            st.metric(
                "Farmer Details Saved",
                "Yes" if st.session_state.farmer_details else "No"
            )

        with col3:
            st.metric(
                "Soil Information Saved",
                "Yes" if st.session_state.soil_information else "No"
            )

        st.divider()

        st.subheader("Telangana Soil Dataset")

        dataset_df = load_soil_dataset()

        if not dataset_df.empty:
            st.dataframe(
                dataset_df,
                use_container_width=True,
                height=280
            )
        else:
            st.warning("Could not load Telangana soil dataset.")

elif page == "Student Registration":
    page_header("Student Registration")
    with st.form("student_registration_form"):
        col1, col2 = st.columns(2)
        with col1:
            student_name = st.text_input("Student Name *", value=st.session_state.student_registration.get("student_name", ""))
            student_id = st.text_input("Student ID / Roll No *", value=st.session_state.student_registration.get("student_id", ""))
        with col2:
            college = st.text_input("College *", value=st.session_state.student_registration.get("college", ""))
            department = st.text_input("Department", value=st.session_state.student_registration.get("department", "Agriculture"))
        save_student = st.form_submit_button("Save Student Registration", use_container_width=True)

    if save_student:
        if student_name and student_id and college:
            st.session_state.student_registration = {
                "student_name": student_name.strip(),
                "student_id": student_id.strip(),
                "college": college.strip(),
                "department": department.strip(),
            }
            insert_student_data(
                student_name.strip(),
                student_id.strip(),
                college.strip(),
                department.strip(),
            )
            st.success("Student registration saved.")
        else:
            st.error("Please fill all required fields.")


elif page == "Farmer Details":
    page_header("Farmer Details")
    district_options = load_telangana_districts()
    saved_district = st.session_state.farmer_details.get("district", district_options[0])
    district_index = district_options.index(saved_district) if saved_district in district_options else 0

    with st.expander("Find coordinates from an address (instead of entering lat/lon manually)"):
        st.caption("Uses OpenStreetMap to look up coordinates from a village/district name -- works like typing an address into a map app.")
        geo_col1, geo_col2 = st.columns(2)
        with geo_col1:
            geo_village = st.text_input("Village", key="geo_village")
        with geo_col2:
            geo_district = st.selectbox("District", district_options, key="geo_district")
        if st.button("Find on map", key="geocode_button"):
            geo_result = geocode_address(village=geo_village, district=geo_district)
            if geo_result.get("available"):
                st.session_state.farmer_details["latitude"] = geo_result["latitude"]
                st.session_state.farmer_details["longitude"] = geo_result["longitude"]
                st.success(f"Found: {geo_result['display_name']} -> {geo_result['latitude']:.5f}, {geo_result['longitude']:.5f}")
                st.rerun()
            else:
                st.error(geo_result.get("error", "Could not find that address."))

    with st.form("farmer_details_form"):
        col1, col2 = st.columns(2)
        with col1:
            farmer_name = st.text_input("Farmer Name *", value=st.session_state.farmer_details.get("farmer_name", ""))
            district = st.selectbox("District *", district_options, index=district_index)
            village = st.text_input("Village *", value=st.session_state.farmer_details.get("village", ""))
            farm_size = st.number_input(
                "Farm Size (Acres) *",
                min_value=0.1,
                value=float(st.session_state.farmer_details.get("farm_size", 1.0)),
                step=0.1,
            )
        with col2:
            phone = st.text_input("Phone Number", value=st.session_state.farmer_details.get("phone", ""))
            soil_testing_done = st.radio("Soil Testing Done? *", ["Yes", "No"], horizontal=True)
            soil_health_card = st.radio("Government Soil Health Card Available? *", ["Yes", "No"], horizontal=True)
            soil_health_card_id = st.text_input("Soil Health Card Number (if available)")

        st.subheader("Location (Telangana only)")
        col3, col4 = st.columns(2)
        with col3:
            latitude = st.number_input(
                "Latitude *",
                min_value=TELANGANA_BOUNDS["lat_min"],
                max_value=TELANGANA_BOUNDS["lat_max"],
                value=float(st.session_state.farmer_details.get("latitude", 17.3850)),
                step=0.0001,
            )
        with col4:
            longitude = st.number_input(
                "Longitude *",
                min_value=TELANGANA_BOUNDS["lon_min"],
                max_value=TELANGANA_BOUNDS["lon_max"],
                value=float(st.session_state.farmer_details.get("longitude", 78.4867)),
                step=0.0001,
            )

        st.subheader("Consent")
        consent_checkbox = st.checkbox(
            "The farmer has been informed what data is collected, how it will be used to "
            "generate recommendations, and agrees to it being stored *",
        )

        save_farmer = st.form_submit_button("Save Farmer Details", use_container_width=True)

    if save_farmer:
        if not consent_checkbox:
            st.error("Farmer consent is required before any details can be saved.")
        elif farmer_name and district and village:
            consent_date = str(pd.Timestamp.now().date())
            st.session_state.farmer_details = {
                "farmer_name": farmer_name.strip(),
                "district": district.strip(),
                "village": village.strip(),
                "farm_size": float(farm_size),
                "phone": phone.strip(),
                "soil_testing_done": soil_testing_done,
                "soil_health_card": soil_health_card,
                "soil_health_card_id": soil_health_card_id.strip(),
                "latitude": float(latitude),
                "longitude": float(longitude),
                "consent_given": "Yes",
                "consent_date": consent_date,
            }
            insert_farmer_data(
                farmer_name.strip(),
                district.strip(),
                village.strip(),
                phone.strip(),
                float(farm_size),
                soil_testing_done,
                soil_health_card,
                soil_health_card_id.strip(),
                float(latitude),
                float(longitude),
                consent_given="Yes",
                consent_date=consent_date,
            )
            if soil_testing_done == "No":
                st.warning("Soil testing is not done. ML confidence may be lower.")
            if soil_health_card == "No":
                st.warning("Government Soil Health Card not available.")
            st.success("Farmer details saved.")
        else:
            st.error("Please fill all required fields.")


elif page == "Soil Information":
    page_header("Soil Information")

    farmer_coords = st.session_state.farmer_details
    if farmer_coords.get("latitude") and farmer_coords.get("longitude"):
        if st.button("Auto-fetch current weather for this farm location"):
            weather_est = get_weather_estimate_for_prediction(
                farmer_coords["latitude"], farmer_coords["longitude"]
            )
            st.session_state.soil_information["temperature"] = weather_est["temperature"]
            st.session_state.soil_information["rainfall"] = weather_est["rainfall_estimate"]
            st.info(
                f"{weather_est['weather_description'].title()}, {weather_est['temperature']}°C, "
                f"~{weather_est['rainfall_estimate']}mm seasonal rainfall "
                f"({weather_est['rainfall_category']} for {weather_est['month']}). "
                f"Source: {weather_est['source']}."
            )
            st.rerun()
    else:
        st.caption("Enter latitude/longitude in Farmer Details first to enable weather auto-fetch.")

    if farmer_coords.get("soil_health_card") == "Yes":
        card_id = farmer_coords.get("soil_health_card_id", "").strip()
        st.info(
            f"You indicated a Government Soil Health Card is available"
            f"{f' (ID: {card_id})' if card_id else ''}. There's no public API to fetch it "
            f"automatically (India's Soil Health Card scheme doesn't offer third-party "
            f"integration) -- but if you have the physical/digital card, enter the exact "
            f"lab-tested pH, N, P, K, EC, and Organic Carbon values printed on it below "
            f"instead of estimating. Lab-measured values are always more accurate than a guess."
        )

    st.subheader("Chemical Usage Option")
    st.caption("Choose this first -- it controls which fields appear below (updates immediately, no need to submit).")
    chemical_usage_mode = st.radio(
        "Select one option for previous crop",
        [
            "Already used chemicals in previous crop",
            "Not used chemicals in previous crop",
        ],
        key="chemical_usage_mode_radio",
    )

    with st.form("soil_information_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            soil_type = st.selectbox(
                "Soil Type *",
                [
                    "Chalka (Red Sandy Loam)",
                    "Dubba (Red Loamy Sand)",
                    "Lateritic Soil",
                    "Shallow-Medium Black Soil",
                    "Deep Black Soil (Black Cotton)",
                    "Salt-affected Soil",
                    "Alluvial Soil",
                ],
                help="Telangana's 7 official soil types (Soils of Andhra Pradesh, 1976 classification). Not sure which? Chalka is the most common, covering most of the state.",
            )
            ph = st.number_input("Soil pH *", min_value=3.5, max_value=10.0, value=float(st.session_state.soil_information.get("ph", 6.5)), step=0.1)
            organic_carbon = st.number_input("Organic Carbon (%) *", min_value=0.0, value=float(st.session_state.soil_information.get("organic_carbon", 0.8)), step=0.1)
        with col2:
            moisture = st.number_input("Soil Moisture (%) *", min_value=0.0, max_value=100.0, value=float(st.session_state.soil_information.get("moisture", 35.0)), step=1.0)
            nitrogen = st.number_input("Nitrogen (kg/acre)", min_value=0.0, value=float(st.session_state.soil_information.get("nitrogen", 100.0)), step=5.0)
            phosphorus = st.number_input("Phosphorus (kg/acre)", min_value=0.0, value=float(st.session_state.soil_information.get("phosphorus", 50.0)), step=5.0)
        with col3:
            potassium = st.number_input("Potassium (kg/acre)", min_value=0.0, value=float(st.session_state.soil_information.get("potassium", 50.0)), step=5.0)
            rainfall = st.number_input("Rainfall (mm)", min_value=0.0, value=float(st.session_state.soil_information.get("rainfall", 700.0)), step=25.0)
            temperature = st.number_input("Temperature (deg C)", min_value=5.0, max_value=50.0, value=float(st.session_state.soil_information.get("temperature", 28.0)), step=0.5)
            electrical_conductivity = st.number_input(
                "Electrical Conductivity (dS/m)",
                min_value=0.0,
                value=float(st.session_state.soil_information.get("electrical_conductivity", 0.3)),
                step=0.1,
            )

        if chemical_usage_mode == "Already used chemicals in previous crop":
            st.subheader("Chemical Use for Soil (Optional)")
            col_chem1, col_chem2 = st.columns(2)
            with col_chem1:
                soil_chemical_type = st.selectbox(
                    "Soil Chemical Type",
                    ["Fertilizer", "Micronutrient", "Soil Conditioner", "Pesticide", "Bio-input"],
                )
                soil_chemical_name = st.text_input(
                    "Soil Chemical Name",
                    value=st.session_state.soil_information.get("soil_chemical_name", ""),
                )
            with col_chem2:
                soil_chemical_dose = st.number_input(
                    "Soil Chemical Dose (kg/acre)",
                    min_value=0.0,
                    value=float(st.session_state.soil_information.get("soil_chemical_dose", 0.0)),
                    step=1.0,
                )
                soil_chemical_note = st.text_input(
                    "Chemical Use Note",
                    value=st.session_state.soil_information.get("soil_chemical_note", ""),
                )

            st.subheader("Already Used Chemical Details")
            col4, col5, col6 = st.columns(3)
            with col4:
                previous_chemical = st.selectbox(
                    "Already Used Chemical",
                    ["NPK", "DAP", "Urea", "Lime", "Sulfur", "Pesticide", "Fungicide", "Herbicide", "Gypsum", "Zinc Sulfate", "Iron Sulfate"],
                )
            with col5:
                previous_chem_amount = st.number_input("Applied Amount (kg/acre)", min_value=0.1, value=100.0, step=5.0)
            with col6:
                days_since_application = st.number_input("Days Since Application", min_value=0, value=30, step=1)
        else:
            st.info(
                "No previous chemical use recorded -- MOA will optimize starting from the "
                "standard baseline dose for the crop you select on the ML & MOA Analysis page, "
                "with no residue carried over from a prior application."
            )
            soil_chemical_type = "None"
            soil_chemical_name = ""
            soil_chemical_dose = 0.0
            soil_chemical_note = ""
            previous_chemical = "None"
            previous_chem_amount = 0.0
            days_since_application = 0

        save_soil = st.form_submit_button("Save Soil Information", use_container_width=True)

    if save_soil:
        st.session_state.soil_information = {
            "soil_type": soil_type,
            "ph": float(ph),
            "organic_carbon": float(organic_carbon),
            "moisture": float(moisture),
            "nitrogen": float(nitrogen),
            "phosphorus": float(phosphorus),
            "potassium": float(potassium),
            "rainfall": float(rainfall),
            "temperature": float(temperature),
            "electrical_conductivity": float(electrical_conductivity),
            "chemical_usage_mode": chemical_usage_mode,
            "previous_chemical": previous_chemical,
            "previous_chem_amount": float(previous_chem_amount),
            "days_since_application": int(days_since_application),
            "soil_chemical_type": str(soil_chemical_type),
            "soil_chemical_name": str(soil_chemical_name),
            "soil_chemical_dose": float(soil_chemical_dose),
            "soil_chemical_note": str(soil_chemical_note),
        }
        st.success("Soil information saved.")

        # Persist immediately as a pending submission, independent of the
        # ML & MOA Analysis page. This is what lets a student on Render
        # submit data and be done -- the admin picks it up later from
        # their laptop's Dashboard, without needing the same session.
        submission_id = insert_soil_submission(
            student_name=st.session_state.student_registration.get("student_name", ""),
            farmer_name=st.session_state.farmer_details.get("farmer_name", ""),
            district=st.session_state.farmer_details.get("district", ""),
            village=st.session_state.farmer_details.get("village", ""),
            soil_type=soil_type,
            ph=float(ph),
            organic_carbon=float(organic_carbon),
            moisture=float(moisture),
            nitrogen=float(nitrogen),
            phosphorus=float(phosphorus),
            potassium=float(potassium),
            rainfall=float(rainfall),
            temperature=float(temperature),
            electrical_conductivity=float(electrical_conductivity),
            chemical_usage_mode=chemical_usage_mode,
            previous_chemical=previous_chemical,
            previous_chem_amount=float(previous_chem_amount),
            days_since_application=int(days_since_application),
            soil_chemical_type=str(soil_chemical_type),
            soil_chemical_name=str(soil_chemical_name),
            soil_chemical_dose=float(soil_chemical_dose),
            soil_chemical_note=str(soil_chemical_note),
        )
        if submission_id:
           st.session_state["soil_information"]["submission_id"] = submission_id

           st.success(
               "✅ Submission received successfully. "
               "Farmer and soil information has been saved. "
               "The administrator will process the ML & MOA analysis."
          )
        else:
           st.warning("Saved locally, but could not queue this submission in the database.")

        # EC-based salinity sanity check -- high electrical conductivity is
        # a direct lab indicator of salinity, independent of which soil
        # type was manually selected above. 4 dS/m is the standard
        # threshold used to classify soil as saline (FAO/ICAR guidelines).
        if float(electrical_conductivity) > 4.0 and soil_type != "Salt-affected Soil":
            st.warning(
                f"Your measured Electrical Conductivity ({electrical_conductivity} dS/m) is above "
                f"4 dS/m, which typically indicates salt-affected soil regardless of the soil type "
                f"selected above. Consider re-selecting 'Salt-affected Soil' for a more accurate "
                f"chemical recommendation -- Gypsum-based reclamation usually takes priority over "
                f"routine fertilization in this case."
            )


elif page == "ML & MOA Analysis":
    page_header("ML and MOA Analysis for Next Crop", icon=PAGE_ICONS["ML & MOA Analysis"])

    with st.expander("📥 Load a pending student submission", expanded=not st.session_state.soil_information):
        st.caption(
            "Submissions students saved from the public app (phone/Render) show up here, "
            "oldest first. Loading one fills in Student/Farmer/Soil below so you can run the "
            "pipeline against it -- nothing here calls the model, it just loads data."
        )
        pending = fetch_pending_submissions()
        if not pending:
            st.info("No pending submissions.")
        else:
            options = {
                f"#{row['id']} -- {row.get('farmer_name') or 'unnamed farmer'} "
                f"({row.get('student_name') or 'unknown student'}, {row.get('created_at')})": row
                for row in pending
            }
            chosen_label = st.selectbox("Pending submissions", list(options.keys()), key="pending_submission_pick")
            if st.button("Load this submission", key="load_pending_submission"):
                row = options[chosen_label]
                farmer_row = fetch_farmer_by_name(row.get("farmer_name", "")) or {}
                st.session_state.student_registration = {
                    "student_name": row.get("student_name", ""),
                    "student_id": st.session_state.student_registration.get("student_id", ""),
                    "college": st.session_state.student_registration.get("college", ""),
                    "department": st.session_state.student_registration.get("department", ""),
                }
                st.session_state.farmer_details = {
                    "farmer_name": row.get("farmer_name", ""),
                    "district": farmer_row.get("district", row.get("district", "")),
                    "village": farmer_row.get("village", row.get("village", "")),
                    "farm_size": farmer_row.get("farm_size", 1.0),
                    "phone": farmer_row.get("phone", ""),
                    "soil_testing_done": farmer_row.get("soil_testing_done", "No"),
                    "soil_health_card": farmer_row.get("soil_health_card", "No"),
                    "soil_health_card_id": farmer_row.get("soil_health_card_id", ""),
                    "latitude": farmer_row.get("latitude", 17.3850),
                    "longitude": farmer_row.get("longitude", 78.4867),
                    "consent_given": farmer_row.get("consent_given", "No"),
                    "consent_date": farmer_row.get("consent_date", ""),
                }
                st.session_state.soil_information = {
                    "submission_id": row["id"],
                    "soil_type": row.get("soil_type", "Chalka (Red Sandy Loam)"),
                    "ph": row.get("ph", 6.5),
                    "organic_carbon": row.get("organic_carbon", 0.8),
                    "moisture": row.get("moisture", 35.0),
                    "nitrogen": row.get("nitrogen", 100.0),
                    "phosphorus": row.get("phosphorus", 50.0),
                    "potassium": row.get("potassium", 50.0),
                    "rainfall": row.get("rainfall", 700.0),
                    "temperature": row.get("temperature", 28.0),
                    "electrical_conductivity": row.get("electrical_conductivity", 0.3),
                    "chemical_usage_mode": row.get("chemical_usage_mode", "Not used chemicals in previous crop"),
                    "previous_chemical": row.get("previous_chemical", "None"),
                    "previous_chem_amount": row.get("previous_chem_amount", 0.0),
                    "days_since_application": row.get("days_since_application", 0),
                    "soil_chemical_type": row.get("soil_chemical_type", "None"),
                    "soil_chemical_name": row.get("soil_chemical_name", ""),
                    "soil_chemical_dose": row.get("soil_chemical_dose", 0.0),
                    "soil_chemical_note": row.get("soil_chemical_note", ""),
                }
                st.success(f"Loaded submission #{row['id']} for {row.get('farmer_name')}.")
                st.rerun()

    if not st.session_state.student_registration or not st.session_state.farmer_details or not st.session_state.soil_information:
        st.warning("Please complete Student Registration, Farmer Details, and Soil Information first.")
    else:
        student = st.session_state.student_registration
        farmer = st.session_state.farmer_details
        soil = st.session_state.soil_information

        st.subheader("Verification Status")
        colv1, colv2 = st.columns(2)
        with colv1:
            st.metric("Soil Testing", farmer.get("soil_testing_done", "No"))
        with colv2:
            st.metric("Govt Soil Health Card", farmer.get("soil_health_card", "No"))
        if farmer.get("soil_health_card", "No") == "Yes":
            st.caption(f"Card Number: {farmer.get('soil_health_card_id', 'Not entered')}")

        next_crop = st.selectbox("Select Next Crop", list(BASELINE_DOSE_BY_CROP.keys()))
        model_soil_type = normalize_soil_type(soil["soil_type"])

        ml_crop = recommend_crop({"ph": soil["ph"], "carbon": soil["organic_carbon"], "moisture": soil["moisture"]})
        chemical_rec = recommend_chemical_by_soil_type(model_soil_type, soil["ph"])
        baseline_per_acre = BASELINE_DOSE_BY_CROP[next_crop]
        if float(soil.get("soil_chemical_dose", 0.0)) > 0:
            baseline_per_acre = float(soil["soil_chemical_dose"])

        farmer_history = fetch_farmer_moa_history(farmer.get("farmer_name", ""))
        previous_cycles = len(farmer_history)
        if previous_cycles > 0:
            prev_optimized = float(farmer_history[0].get("optimized_dose", baseline_per_acre))
            # Continue from previously optimized farmer dose so reduction is progressive by cycle.
            baseline_per_acre = min(baseline_per_acre, prev_optimized)

        fertility = soil_fertility_score(soil)
        if previous_cycles == 0:
            cycle_cap_pct = 30.0
            cycle_label = "Cycle 1 (first optimization)"
        else:
            if fertility >= 80:
                cycle_cap_pct = 15.0
            elif fertility >= 60:
                cycle_cap_pct = 10.0
            elif fertility >= 40:
                cycle_cap_pct = 7.0
            else:
                cycle_cap_pct = 5.0
            cycle_label = f"Cycle {previous_cycles + 1} (fertility-based progressive optimization)"

        use_moa_for_next_crop = True

        st.subheader("Soil Chemical Use Summary")
        st.write(
            f"Type: **{soil.get('soil_chemical_type', 'N/A')}** | "
            f"Name: **{soil.get('soil_chemical_name', 'N/A') or 'N/A'}** | "
            f"Dose: **{float(soil.get('soil_chemical_dose', 0.0)):.1f} kg/acre**"
        )
        if soil.get("soil_chemical_note"):
            st.caption(f"Note: {soil.get('soil_chemical_note')}")
        st.caption("If dose is provided above, MOA uses it as baseline for optimization.")

        if soil["chemical_usage_mode"] == "Already used chemicals in previous crop":
            st.subheader("Option 1: Already Used Chemicals -> ML Residue + MOA")
            prev_chem_for_model = normalize_prev_chemical(soil["previous_chemical"])
            residue = estimate_residue_level(
                prev_chem_for_model,
                soil["previous_chem_amount"],
                soil["days_since_application"],
            )

            residue_status = "Present" if residue["remaining_percentage"] > 10 else "Low/Trace"
            residue_percentage = residue["remaining_percentage"]
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("ML Residue Status", residue_status)
            with col2:
                st.metric("Residue Remaining", f"{residue['remaining_percentage']}%")
            with col3:
                st.metric("Remaining Amount", f"{residue['remaining_amount']} kg/acre")

            if residue_status == "Present":
                st.warning("ML indicates chemical residue is present in soil.")
            else:
                st.info("ML indicates only low/trace residue in soil.")

            safety = check_chemical_safety(
                previous_chemical=soil["previous_chemical"],
                current_recommendation=chemical_rec["primary_chemical"],
                soil_type=model_soil_type,
            )
            if safety["safe"]:
                st.success(safety["message"])
            else:
                st.warning(safety["message"])

            render_residue_graph(soil["previous_chem_amount"], residue["half_life_days"], soil["days_since_application"])

            st.subheader("Chemical Formula Analysis")
            st.caption(
                "Different chemicals deliver very different actual nutrient amounts per kg -- "
                "this converts the raw dose into real N/P/K (and secondary nutrients). Works "
                "automatically for known chemical names and for any name containing an N-P-K "
                "grade (e.g. '12-32-16', as printed on most Indian fertilizer bags)."
            )

            def render_composition(chemical_name, dose, manual_key):
                composition = compute_nutrients_delivered(chemical_name, dose)
                if not composition.get("available"):
                    st.warning(composition["error"])
                    with st.expander(f"Enter the grade for '{chemical_name}' manually"):
                        st.caption("Check the product label -- most Indian fertilizer bags print an N-P2O5-K2O grade even for branded products.")
                        mc1, mc2, mc3 = st.columns(3)
                        with mc1:
                            man_n = st.number_input("Nitrogen %", min_value=0.0, max_value=60.0, value=0.0, key=f"{manual_key}_n")
                        with mc2:
                            man_p = st.number_input("Phosphorus (P2O5) %", min_value=0.0, max_value=60.0, value=0.0, key=f"{manual_key}_p")
                        with mc3:
                            man_k = st.number_input("Potassium (K2O) %", min_value=0.0, max_value=60.0, value=0.0, key=f"{manual_key}_k")
                        if man_n or man_p or man_k:
                            composition = compute_nutrients_delivered(
                                chemical_name, dose,
                                manual_composition={"Nitrogen": man_n, "Phosphorus": man_p, "Potassium": man_k},
                            )
                        else:
                            return

                if composition.get("available"):
                    if composition.get("is_pest_control_only"):
                        st.info(f"{chemical_name} is a pest/disease control product, not a nutrient carrier -- no N/P/K breakdown applies.")
                    else:
                        delivered = composition["nutrients_delivered_kg_per_acre"]
                        nutrient_cols = st.columns(min(4, max(1, len(delivered))))
                        for i, (nutrient, amount) in enumerate(delivered.items()):
                            with nutrient_cols[i % len(nutrient_cols)]:
                                st.metric(nutrient, f"{amount} kg/acre")
                        if composition.get("ph_effect"):
                            st.caption(f"Also affects soil pH: {composition['ph_effect']}")

            render_composition(prev_chem_for_model, soil["previous_chem_amount"], "prev_chem")

            n_gap = max(0.0, 200.0 - soil["nitrogen"])
            if n_gap > 0:
                st.write(f"Your measured Nitrogen is {n_gap:.0f} kg/acre below the {200:.0f} kg/acre reference band. Chemicals compared for closing this gap:")
                comparison = compare_chemicals_for_nutrient_gap("Nitrogen", n_gap)
                if comparison:
                    comp_df = pd.DataFrame([
                        {"Chemical": c["chemical"], "Dose needed to close gap (kg/acre)": c["required_dose_kg_per_acre"]}
                        for c in comparison
                    ])
                    st.dataframe(comp_df, use_container_width=True, hide_index=True)
            else:
                st.caption("Measured Nitrogen is already at or above the reference band -- no gap to close.")

            custom_chem_name = soil.get("soil_chemical_name", "").strip()
            custom_chem_dose = soil.get("soil_chemical_dose", 0.0)
            if custom_chem_name and custom_chem_dose:
                st.write(f"**Also analyzing the custom chemical entered on Soil Information: '{custom_chem_name}'**")
                render_composition(custom_chem_name, custom_chem_dose, "custom_chem")

            detected_category = infer_chemical_category(prev_chem_for_model)
            if detected_category == "pest_control":
                st.caption(
                    f"'{prev_chem_for_model}' is treated as a pest-control product: MOA uses a "
                    "stricter, threshold-aware safety margin rather than the smoother fertilizer curve, "
                    "since under-dosing pest control risks an outbreak rather than a gradual yield dip."
                )
            optimization = meerkat_chemical_reduction(
                baseline_per_acre, target_yield_percentage=0.95, chemical_category=detected_category
            )

        else:
            st.subheader("Option 2: No Previous Chemicals -> MOA Minimal Suggestion")
            use_moa_for_next_crop = st.checkbox(
                "Use MOA to optimize chemical usage for next crop",
                value=True,
            )
            if use_moa_for_next_crop:
                optimization = meerkat_chemical_reduction(baseline_per_acre, target_yield_percentage=0.95)
                st.info("MOA enabled: optimizing chemical usage and predicting next-crop yield with optimized dose.")
            else:
                optimization = {
                    "optimal_chemical_dose": baseline_per_acre,
                    "reduction_percentage": 0.0,
                }
                st.warning("MOA not selected. Yield prediction will use baseline chemical dose.")
            residue_status = "Not applicable"
            residue_percentage = 0.0

        # Apply cycle reduction cap:
        # cycle 1 -> up to 30%; later cycles -> additional reduction by fertility score.
        cap_floor_dose = baseline_per_acre * (1.0 - (cycle_cap_pct / 100.0))
        optimized_per_acre = max(cap_floor_dose, float(optimization["optimal_chemical_dose"]))
        reduction_pct_cycle = ((baseline_per_acre - optimized_per_acre) / baseline_per_acre * 100.0) if baseline_per_acre > 0 else 0.0
        optimization["optimal_chemical_dose"] = round(optimized_per_acre, 3)
        optimization["reduction_percentage"] = round(reduction_pct_cycle, 1)
        farm_area = float(farmer["farm_size"])
        total_initial = baseline_per_acre * farm_area
        total_optimized = optimized_per_acre * farm_area

        st.session_state["last_moa_result"] = {
            "next_crop": next_crop,
            "chemical_name": chemical_rec["primary_chemical"],
            "optimized_dose_per_acre": round(optimized_per_acre, 3),
            "baseline_dose_per_acre": round(baseline_per_acre, 3),
            "farm_area_acres": farm_area,
        }


        st.subheader("MOA Chemical Optimization for Next Crop")
        category_used = optimization.get("chemical_category", "fertilizer")
        st.caption(
            f"{cycle_label} | Soil fertility score: {fertility}/100 | Max reduction this cycle: {cycle_cap_pct:.0f}% | "
            f"Dose-response model: {'pest control (conservative)' if category_used == 'pest_control' else 'fertilizer'}"
        )
        col4, col5, col6, col7 = st.columns(4)
        with col4:
            st.metric("Next Crop", next_crop)
        with col5:
            st.metric("Initial Dose (kg/acre)", f"{baseline_per_acre:.1f}")
        with col6:
            st.metric("MOA Optimized (kg/acre)", f"{optimized_per_acre:.1f}")
        with col7:
            st.metric("Reduction", f"{optimization['reduction_percentage']:.1f}%")

        st.write(
            f"For {farm_area:.1f} acres: initial total = {total_initial:.1f} kg, "
            f"optimized total = {total_optimized:.1f} kg, savings = {total_initial - total_optimized:.1f} kg."
        )
        st.write(f"Recommended chemical for soil condition: **{chemical_rec['primary_chemical']}**")

        st.subheader("How to Use Optimized Chemical (MOA Plan)")
        saved_per_acre = max(0.0, baseline_per_acre - optimized_per_acre)
        stage_1 = optimized_per_acre * 0.5
        stage_2 = optimized_per_acre * 0.3
        stage_3 = optimized_per_acre * 0.2
        usage_plan_df = pd.DataFrame(
            {
                "Stage": ["Basal application", "Vegetative stage", "Flowering/fruiting stage"],
                "Dose (kg/acre)": [round(stage_1, 2), round(stage_2, 2), round(stage_3, 2)],
                "When to apply": [
                    "At sowing/transplanting",
                    "20-30 days after sowing",
                    "40-55 days after sowing",
                ],
            }
        )
        st.dataframe(usage_plan_df, use_container_width=True)
        st.caption(
            f"Per acre use: {optimized_per_acre:.2f} kg. "
            f"Farm total use: {total_optimized:.2f} kg. "
            f"Reduced/avoided chemical: {saved_per_acre * farm_area:.2f} kg."
        )

        st.subheader("Optimizer Comparison: MOA vs PSO vs GA vs Rule-based")
        st.caption(
            "MOA is one candidate optimizer, not the assumed answer -- benchmarked here against "
            "Particle Swarm Optimization, a Genetic Algorithm, and a fixed-percentage rule-based "
            "baseline on the identical dose-reduction problem, per the technical advisory."
        )
        comparison = compare_optimizers(
            baseline_per_acre, target_yield_percentage=0.95, chemical_category=category_used,
        )
        if comparison["rows"]:
            comp_df = pd.DataFrame(comparison["rows"]).rename(columns={
                "method": "Method",
                "optimal_chemical_dose": "Optimized dose (kg/acre)",
                "reduction_percentage": "Chemical reduction (%)",
                "achieved_yield_pct": "Achieved yield (%)",
                "runtime_ms": "Runtime (ms)",
            })
            st.dataframe(comp_df, use_container_width=True, hide_index=True)
            chart_col1, chart_col2 = st.columns(2)
            with chart_col1:
                themed_bar_chart(comp_df, "Method", "Chemical reduction (%)", color=CHART_GREEN)
            with chart_col2:
                themed_bar_chart(comp_df, "Method", "Runtime (ms)", color=CHART_BROWN)
            st.info(f"Suggested candidate on this run: **{comparison['recommended']}**. {comparison['note']}")

        st.subheader("MOA Chemical Usage Pie Chart")
        pie_df = pd.DataFrame(
            {
                "Category": ["Optimized use", "Reduced / avoided"],
                "kg_per_acre": [round(optimized_per_acre, 3), round(saved_per_acre, 3)],
            }
        )
        themed_donut_chart(pie_df, "Category", "kg_per_acre", colors=[CHART_GREEN, CHART_BROWN])

        reduction_scale = max(0.1, optimized_per_acre / baseline_per_acre)
        yield_prediction = predict_crop_yield(
            crop_type=next_crop,
            ph=soil["ph"],
            moisture=soil["moisture"],
            temperature=soil["temperature"],
            nitrogen=soil["nitrogen"] * reduction_scale,
            phosphorus=soil["phosphorus"] * reduction_scale,
            potassium=soil["potassium"] * reduction_scale,
            rainfall=soil["rainfall"],
        )

        st.subheader("ML Yield Prediction for Next Crop")
        st.caption("Rule-based estimate (hand-authored formula, not trained on data)")
        col8, col9, col10 = st.columns(3)
        with col8:
            st.metric("Predicted Yield", f"{yield_prediction['predicted_yield']:.0f} kg/acre")
        with col9:
            st.metric("Confidence", f"{yield_prediction['confidence_score']}%")
        with col10:
            st.metric("Limiting Factor", yield_prediction["limiting_factor"])

        conf = yield_prediction["confidence_score"]
        if conf < 60:
            st.error(
                f"Confidence is {conf}% -- below the 60% threshold. This recommendation is routed "
                f"to mandatory agronomist review below and should not be used as-is."
            )
        elif conf < 80:
            st.warning(
                f"Confidence is {conf}% -- usable, but on the lower side. Worth a quick agronomist "
                f"double-check before applying at scale, especially if the limiting factor "
                f"('{yield_prediction['limiting_factor']}') is something you can still verify or correct "
                f"(e.g. re-check rainfall/moisture data) before committing to this dose."
            )
        else:
            st.success(f"Confidence is {conf}% -- high. No additional review required on confidence grounds alone.")

        factors_df = pd.DataFrame(
            {
                "Factor": list(yield_prediction["factors"].keys()),
                "Effectiveness (%)": list(yield_prediction["factors"].values()),
            }
        )
        st.dataframe(factors_df, use_container_width=True)
        themed_bar_chart(factors_df, "Factor", "Effectiveness (%)", color=CHART_GREEN)

        st.subheader("Trained ML Model Prediction (Random Forest)")
        st.caption("Actual scikit-learn model trained on dataset/telangana_soil_data.csv")
        ml_prediction = predict_crop_yield_ml(
            crop_type=next_crop,
            soil_type=model_soil_type,
            ph=soil["ph"],
            nitrogen=soil["nitrogen"] * reduction_scale,
            phosphorus=soil["phosphorus"] * reduction_scale,
            potassium=soil["potassium"] * reduction_scale,
            organic_carbon=soil["organic_carbon"],
            moisture=soil["moisture"],
            electrical_conductivity=soil["electrical_conductivity"],
        )
        if ml_prediction.get("available"):
            col11, col12, col13 = st.columns(3)
            with col11:
                st.metric("Predicted Yield", f"{ml_prediction['predicted_yield']:.0f} kg/acre")
            with col12:
                st.metric("Confidence", f"{ml_prediction['confidence_score']}%")
            with col13:
                delta = ml_prediction["predicted_yield"] - yield_prediction["predicted_yield"]
                st.metric("vs Rule-based", f"{delta:+.0f} kg/acre")
            st.caption(f"Model: {ml_prediction['model_type']} | tree agreement std: {ml_prediction['tree_prediction_std_kg_acre']} kg/acre")
        else:
            st.warning(f"Trained ML model unavailable: {ml_prediction.get('error', 'unknown error')}. Falling back to the rule-based estimate above.")

        with st.expander("Why did the model predict this? (Explainable AI / XAI)", expanded=False):
            st.caption(
                "Shows why the trained model made this recommendation -- SHAP values if the "
                "`shap` package is installed, otherwise a permutation-based local importance "
                "so this still works without the optional dependency."
            )
            explanation = explain_prediction(
                crop_type=next_crop, soil_type=model_soil_type, ph=soil["ph"],
                nitrogen=soil["nitrogen"] * reduction_scale, phosphorus=soil["phosphorus"] * reduction_scale,
                potassium=soil["potassium"] * reduction_scale, organic_carbon=soil["organic_carbon"],
                moisture=soil["moisture"], electrical_conductivity=soil["electrical_conductivity"],
            )
            if explanation.get("available"):
                st.caption(f"Method: {explanation['method']}")
                exp_df = pd.DataFrame(explanation["contributions"])
                st.dataframe(exp_df, use_container_width=True, hide_index=True)
                themed_bar_chart(exp_df, "feature", "impact_kg_per_acre", diverging=True, horizontal=True)
            else:
                st.warning(f"Explanation unavailable: {explanation.get('error', 'unknown error')}")

            st.caption("Model-wide feature importance (what the model relies on in general, not just this farm):")
            global_importance = get_global_feature_importance()
            if global_importance.get("available"):
                gi_df = pd.DataFrame(global_importance["features"])
                themed_bar_chart(gi_df, "feature", "importance_pct", color=CHART_BROWN, horizontal=True)
            else:
                st.warning(f"Feature importance unavailable: {global_importance.get('error', 'unknown error')}")

        with st.expander("Model Metrics (MAE, RMSE, R², Cross-validation)", expanded=False):
            st.caption(
                "How trustworthy the trained yield model actually is, evaluated on held-out "
                "Telangana soil data -- addresses the advisory gap that uncertainty and model "
                "failure were not previously surfaced to the user."
            )
            if st.button("Compute model metrics", key="compute_model_metrics"):
                with st.spinner("Evaluating model on held-out data and running cross-validation..."):
                    metrics = get_model_metrics()
                st.session_state["last_model_metrics"] = metrics
            metrics = st.session_state.get("last_model_metrics")
            if metrics:
                mcol1, mcol2, mcol3, mcol4 = st.columns(4)
                with mcol1:
                    st.metric("MAE (kg/acre)", metrics["mae_kg_per_acre"])
                with mcol2:
                    st.metric("RMSE (kg/acre)", metrics["rmse_kg_per_acre"])
                with mcol3:
                    st.metric("R²", metrics["r2_score"])
                with mcol4:
                    st.metric(f"CV R² ({metrics['cv_folds']}-fold)", f"{metrics['cv_r2_mean']} ± {metrics['cv_r2_std']}")
                st.caption(f"{metrics['model_type']} | trained on {metrics['training_rows']} rows, tested on {metrics['test_rows']} rows.")

        # --- Agronomic safety layer + mandatory human validation gate ---
        # Runs before any recommendation is treated as final. Hand-authored
        # boundaries live in backend/model.py::agronomic_rule_check and can't
        # be overridden by a model output; low-confidence predictions are
        # routed here too instead of being auto-published to the farmer.
        st.subheader("Agronomic Rule Check")
        rule_check = agronomic_rule_check(
            ph=soil["ph"], moisture=soil["moisture"], carbon=soil["organic_carbon"],
            confidence_score=yield_prediction["confidence_score"],
        )
        review_required = needs_agronomist_review(yield_prediction["confidence_score"], rule_check["flags"])
        if review_required:
            st.warning("This recommendation requires agronomist/AEO review before it is used in the field.")
            for flag in rule_check["flags"]:
                st.write(f"- {flag}")
            if rule_check["low_confidence"]:
                st.write(f"- ML confidence ({yield_prediction['confidence_score']}%) is below the 60% review threshold")
        else:
            st.success("No agronomic rule flags. Confidence is above the review threshold.")

        with st.expander("Agronomist / AEO sign-off"):
            reviewer_name = st.text_input("Reviewer name", key="reviewer_name")
            decision = st.selectbox("Decision", ["Approved", "Modified", "Rejected"], key="review_decision")
            review_notes = st.text_area("Notes (required if Modified or Rejected)", key="review_notes")
            if st.button("Record agronomist decision", use_container_width=True):
                if not reviewer_name:
                    st.error("Reviewer name is required.")
                elif decision != "Approved" and not review_notes:
                    st.error("Notes are required when modifying or rejecting a recommendation.")
                else:
                    review_ok = insert_agronomist_review(
                        student_name=student["student_name"],
                        farmer_name=farmer["farmer_name"],
                        next_crop=next_crop,
                        confidence_score=float(yield_prediction["confidence_score"]),
                        rule_flags="; ".join(rule_check["flags"]) if rule_check["flags"] else "",
                        requires_review="Yes" if review_required else "No",
                        reviewer_name=reviewer_name.strip(),
                        decision=decision,
                        notes=review_notes.strip(),
                    )
                    if review_ok:
                        st.success(f"Review recorded: {decision} by {reviewer_name.strip()}.")
                    else:
                        st.error("Could not save the review -- check the Database page / logs.")

        st.subheader("Estimated Cost Savings")
        cost_savings = estimate_cost_savings(
            chemical_name=chemical_rec["primary_chemical"],
            baseline_dose_per_acre=baseline_per_acre,
            optimized_dose_per_acre=optimized_per_acre,
            farm_area_acres=farm_area,
        )
        ccol1, ccol2, ccol3 = st.columns(3)
        with ccol1:
            st.metric("₹ saved / acre", f"₹{cost_savings['rupees_saved_per_acre']:.0f}")
        with ccol2:
            st.metric("Chemical saved (kg)", f"{cost_savings['kg_saved_total']:.1f} kg")
        with ccol3:
            st.metric("Reduction", f"{cost_savings['percentage_reduction']:.1f}%")
        st.caption(f"₹{cost_savings['rupees_saved_total']:.0f} saved across the whole {farm_area:.1f}-acre farm this season.")
        if cost_savings["price_is_indicative"]:
            st.caption("Price used is an indicative default -- exact market price for this chemical was not on file.")

        st.subheader("Environmental Dashboard")
        env_impact = estimate_environmental_impact(
            chemical_name=chemical_rec["primary_chemical"],
            baseline_dose_per_acre=baseline_per_acre,
            optimized_dose_per_acre=optimized_per_acre,
            organic_carbon_pct=soil.get("organic_carbon"),
        )
        ecol1, ecol2, ecol3 = st.columns(3)
        with ecol1:
            st.metric("Chemical reduction", f"{env_impact['chemical_reduction_pct']:.1f}%")
        with ecol2:
            st.metric("Est. residue reduction", f"{env_impact['estimated_residue_reduction_pct']:.1f}%")
        with ecol3:
            st.metric("Soil health indicator", env_impact["soil_health_improvement_indicator"])
        st.caption(env_impact["note"])

        with st.expander("Soil Biological Activity Indicator (experimental)", expanded=False):
            st.caption(
                "Waksman-inspired extension -- reasons about soil biology (microbial "
                "activity) alongside the chemistry above, using organic carbon as a proxy "
                "for the soil's microbial food supply. Display-only: does not affect the "
                "crop, chemical, or dose recommendation above. See Project Review Report, "
                "Section 10, for the full framing and the questions this raises for "
                "agronomist review."
            )
            st.caption(
                "By default, nitrogen sufficiency is shown using the Available N (kg/acre) "
                "this app already collects -- not a true C:N ratio, since that needs Total N% "
                "on a lab report. If a farmer has had a lab test done (Kjeldahl/CHNS Total "
                "Nitrogen %), enter it below to unlock the real C:N ratio."
            )
            lab_total_n = st.number_input(
                "Lab-verified Total Nitrogen % (optional -- leave 0 if not available)",
                min_value=0.0, max_value=5.0, value=0.0, step=0.01, format="%.2f",
                key="lab_total_nitrogen_pct",
            )
            carbon_history = fetch_farmer_soil_carbon_history(farmer.get("farmer_name", ""))
            bio = estimate_biological_activity(
                organic_carbon_pct=soil.get("organic_carbon"),
                carbon_history=carbon_history,
                nitrogen=soil.get("nitrogen"),
                moisture=soil.get("moisture"),
                electrical_conductivity=soil.get("electrical_conductivity"),
                total_nitrogen_pct=lab_total_n if lab_total_n > 0 else None,
            )
            if bio.get("available"):
                bcol1, bcol2 = st.columns(2)
                with bcol1:
                    st.metric("Biological activity (proxy)", bio["biological_activity_level"])
                with bcol2:
                    st.metric("Organic carbon trend", bio["organic_carbon_trend"].title())
                st.write(bio["level_note"])
                for note in bio["constraint_notes"]:
                    st.warning(note)
                if bio.get("c_n_ratio") is not None:
                    st.metric("C:N ratio (lab-verified)", f"{bio['c_n_ratio']}:1")
                    st.success(bio["c_n_ratio_note"])
                elif bio["available_n_note"]:
                    st.caption(bio["available_n_note"])
                st.caption(bio["disclaimer"])
            else:
                st.info(bio.get("reason", "Biological activity indicator unavailable."))

        st.subheader("Farmer Season History (recurring, not one-time)")
        st.caption(
            "This farmer is looked up by name every visit -- each new crop/season adds a row "
            "here rather than overwriting the last one, so savings accumulate across seasons."
        )
        season_summary = fetch_farmer_season_summary(farmer.get("farmer_name", ""))
        if season_summary["season_count"] > 0:
            season_df = pd.DataFrame(season_summary["seasons"])
            st.dataframe(season_df, use_container_width=True, hide_index=True)
            st.metric("Cumulative chemical saved across all seasons", f"{season_summary['cumulative_kg_saved']:.1f} kg")
        else:
            st.info("No prior seasons on record yet for this farmer -- this will be Season 1 once saved below.")

        with st.expander("Download PDF Report", expanded=False):
            st.caption("Bundles soil info, weather, crop/MOA recommendation, savings, environmental impact and a location map into one PDF.")
            if st.button("Generate PDF report", use_container_width=True):
                weather_ctx = {
                    "temperature_c": soil.get("temperature"),
                    "rainfall_mm": soil.get("rainfall"),
                }
                report_context = {
                    "farmer": farmer,
                    "soil": soil,
                    "weather": weather_ctx,
                    "crop_recommendation": {
                        "next_crop": next_crop,
                        "recommended_chemical": chemical_rec["primary_chemical"],
                        "predicted_yield": yield_prediction["predicted_yield"],
                        "confidence_score": yield_prediction["confidence_score"],
                        "limiting_factor": yield_prediction["limiting_factor"],
                    },
                    "moa_recommendation": {
                        "method": "MOA (Meerkat Optimization Algorithm)",
                        "initial_dose": baseline_per_acre,
                        "optimized_dose": optimized_per_acre,
                        "reduction_percentage": optimization["reduction_percentage"],
                        "farm_area": farm_area,
                    },
                    "cost_savings": cost_savings,
                    "environmental_impact": env_impact,
                    "latitude": farmer.get("latitude"),
                    "longitude": farmer.get("longitude"),
                    "generated_by": student["student_name"],
                }
                safe_name = "".join(
                    c if c.isalnum() or c in (" ", "_", "-") else "_"
                    for c in farmer.get("farmer_name", "farmer")
                ).strip().replace(" ", "_") or "farmer"
                report_path = os.path.join(tempfile.gettempdir(), f"farm_report_{safe_name}.pdf")
                generate_pdf_report(report_path, report_context)
                with open(report_path, "rb") as f:
                    st.download_button(
                        "Download report PDF", data=f.read(),
                        file_name=os.path.basename(report_path), mime="application/pdf",
                        use_container_width=True,
                    )

        if st.button("Save Analysis Record", use_container_width=True):
            ok = insert_soil_entry(
                student_name=student["student_name"],
                college=student["college"],
                farmer_name=farmer["farmer_name"],
                district=farmer["district"],
                village=farmer["village"],
                ph=soil["ph"],
                carbon=soil["organic_carbon"],
                moisture=soil["moisture"],
                chemical=chemical_rec["primary_chemical"],
            )
            ml_ok = insert_ml_result(
                student_name=student["student_name"],
                farmer_name=farmer["farmer_name"],
                next_crop=next_crop,
                recommended_crop=ml_crop,
                recommended_chemical=chemical_rec["primary_chemical"],
                residue_status=residue_status,
                residue_percentage=float(residue_percentage),
                predicted_yield=float(yield_prediction["predicted_yield"]),
                confidence_score=float(yield_prediction["confidence_score"]),
                limiting_factor=yield_prediction["limiting_factor"],
            )
            moa_ok = insert_moa_result(
                student_name=student["student_name"],
                farmer_name=farmer["farmer_name"],
                next_crop=next_crop,
                initial_dose=float(baseline_per_acre),
                optimized_dose=float(optimized_per_acre),
                reduction_percentage=float(optimization["reduction_percentage"]),
                farm_area=float(farm_area),
                total_initial=float(total_initial),
                total_optimized=float(total_optimized),
                season_number=previous_cycles + 1,
                optimizer_method="MOA",
            )
            if ok:
                if ml_ok and moa_ok:
                    st.success("Soil entry, ML result, and MOA result saved to database.")
                else:
                    st.warning("Soil entry saved, but ML/MOA save had an issue.")
            else:
                st.error("Could not save analysis record.")

            pending_submission_id = soil.get("submission_id")
            if pending_submission_id:
                if mark_submission_processed(pending_submission_id):
                    st.caption(f"Submission #{pending_submission_id} marked processed -- students will now see these results.")
                else:
                    st.caption(f"Saved, but couldn't mark submission #{pending_submission_id} as processed.")

        st.caption(f"ML suggested crop from soil parameters: {ml_crop}")


elif page == "Maps":
    page_header("Telangana Map: Cursor Location Analysis (Soil + Land + Water)", icon=PAGE_ICONS["Maps"])
    st.caption("Telangana-only. Move cursor by clicking map or by precise latitude/longitude inputs.")

    if st.session_state.farmer_details:
        st.session_state.map_lat = float(st.session_state.farmer_details.get("latitude", st.session_state.map_lat))
        st.session_state.map_lon = float(st.session_state.farmer_details.get("longitude", st.session_state.map_lon))

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.session_state.map_lat = st.number_input(
            "Cursor Latitude",
            min_value=TELANGANA_BOUNDS["lat_min"],
            max_value=TELANGANA_BOUNDS["lat_max"],
            value=float(st.session_state.map_lat),
            step=0.0001,
        )
    with col_b:
        st.session_state.map_lon = st.number_input(
            "Cursor Longitude",
            min_value=TELANGANA_BOUNDS["lon_min"],
            max_value=TELANGANA_BOUNDS["lon_max"],
            value=float(st.session_state.map_lon),
            step=0.0001,
        )
    with col_c:
        zoom_level = st.slider("Map Zoom", min_value=7, max_value=18, value=10)

    st.session_state.map_lat = min(max(st.session_state.map_lat, TELANGANA_BOUNDS["lat_min"]), TELANGANA_BOUNDS["lat_max"])
    st.session_state.map_lon = min(max(st.session_state.map_lon, TELANGANA_BOUNDS["lon_min"]), TELANGANA_BOUNDS["lon_max"])

    st.caption("Nudge cursor (fine adjustment without dragging):")
    nudge_step = st.select_slider(
        "Nudge step size", options=[0.0001, 0.001, 0.005, 0.01], value=0.001,
        key="nudge_step_size", label_visibility="collapsed",
    )
    ncol1, ncol2, ncol3, ncol4 = st.columns(4)
    with ncol1:
        if st.button("\u2190 West", use_container_width=True):
            st.session_state.map_lon -= nudge_step
    with ncol2:
        if st.button("\u2191 North", use_container_width=True):
            st.session_state.map_lat += nudge_step
    with ncol3:
        if st.button("\u2193 South", use_container_width=True):
            st.session_state.map_lat -= nudge_step
    with ncol4:
        if st.button("\u2192 East", use_container_width=True):
            st.session_state.map_lon += nudge_step

    st.session_state.map_lat = min(max(st.session_state.map_lat, TELANGANA_BOUNDS["lat_min"]), TELANGANA_BOUNDS["lat_max"])
    st.session_state.map_lon = min(max(st.session_state.map_lon, TELANGANA_BOUNDS["lon_min"]), TELANGANA_BOUNDS["lon_max"])

    pcol1, pcol2 = st.columns(2)
    with pcol1:
        st.caption(f"Plotted points (auto-added by map clicks): {len(st.session_state.shape_points)}")
    with pcol2:
        if st.button("Clear All Plotted Points"):
            st.session_state.shape_points = []
    if st.button("Add Current Cursor As Point (if map click not captured)"):
        st.session_state.shape_points.append((st.session_state.map_lat, st.session_state.map_lon))
        st.success("Current cursor added as plotted point.")

    m = folium.Map(location=[st.session_state.map_lat, st.session_state.map_lon], zoom_start=zoom_level, tiles=None)
    folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        attr="Google Satellite",
        name="Google Satellite",
        overlay=False,
        control=True,
    ).add_to(m)
    Fullscreen().add_to(m)
    MiniMap(toggle_display=True).add_to(m)
    LocateControl(auto_start=False).add_to(m)
    MousePosition().add_to(m)
    MeasureControl(primary_length_unit="kilometers").add_to(m)
    Geocoder(collapsed=False, placeholder="Search a place name (e.g. Warangal, Shamshabad)...").add_to(m)
    Draw(
        export=False,
        draw_options={
            "polyline": False, "circle": False, "circlemarker": False,
            "marker": True, "polygon": True, "rectangle": True,
        },
        edit_options={"edit": True, "remove": True},
    ).add_to(m)
    folium.Marker(
        [st.session_state.map_lat, st.session_state.map_lon],
        tooltip="Current Cursor",
        popup=f"{st.session_state.map_lat:.4f}, {st.session_state.map_lon:.4f}",
    ).add_to(m)

    for idx, (pt_lat, pt_lon) in enumerate(st.session_state.shape_points, start=1):
        folium.CircleMarker(
            [pt_lat, pt_lon],
            radius=4,
            color="yellow",
            fill=True,
            fill_opacity=0.9,
            tooltip=f"Point {idx}",
        ).add_to(m)
    if len(st.session_state.shape_points) >= 3:
        folium.Polygon(
            locations=st.session_state.shape_points,
            color="cyan",
            fill=True,
            fill_opacity=0.2,
            tooltip="Point-plotted shape",
        ).add_to(m)

    map_data = st_folium(
        m,
        height=520,
        width=None,
        returned_objects=["last_clicked", "last_active_drawing", "all_drawings"],
        key=f"main_map_{st.session_state.map_lat:.4f}_{st.session_state.map_lon:.4f}_{zoom_level}",
    )
    clicked = map_data.get("last_clicked") if map_data else None
    if clicked:
        clicked_lat = float(clicked["lat"])
        clicked_lon = float(clicked["lng"])
        if is_within_telangana(clicked_lat, clicked_lon):
            st.session_state.map_lat = clicked_lat
            st.session_state.map_lon = clicked_lon
            click_sig = f"{clicked_lat:.6f},{clicked_lon:.6f}"
            if st.session_state.last_click_sig != click_sig:
                st.session_state.shape_points.append((clicked_lat, clicked_lon))
                st.session_state.last_click_sig = click_sig
            st.success(f"Cursor moved to: {clicked_lat:.5f}, {clicked_lon:.5f} and point plotted.")
        else:
            st.error("Selected point is outside Telangana. Please click within Telangana.")

    all_drawings = map_data.get("all_drawings") if map_data else None
    if all_drawings:
        drawn_points = []
        for feature in all_drawings:
            geom = feature.get("geometry", {})
            coords = geom.get("coordinates")
            if not coords:
                continue
            if geom.get("type") == "Polygon":
                for lon_c, lat_c in coords[0]:
                    drawn_points.append((lat_c, lon_c))
            elif geom.get("type") == "Point":
                lon_c, lat_c = coords
                drawn_points.append((lat_c, lon_c))
        if drawn_points and st.button(f"Use {len(drawn_points)} drawn point(s) as the plotted boundary", use_container_width=True):
            st.session_state.shape_points = drawn_points
            st.success("Drawn shape saved as the plotted boundary -- used by Stage 2 and Field Health Imaging.")

    lat = st.session_state.map_lat
    lon = st.session_state.map_lon
    st.write(f"Current cursor: lat={lat:.5f}, lon={lon:.5f}")

    st.caption("Google Satellite only. Click map to move cursor and auto-plot points.")

    if st.button("Analyze Current Cursor Location", use_container_width=True):
        if not is_within_telangana(lat, lon):
            st.error("Location is outside Telangana.")
        else:
            soil_info = get_soil_type_for_location(lat, lon)
            if soil_info:
                land_type, water_status = classify_land_and_water(soil_info)
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Soil Type", soil_info["soil_type"])
                with col2:
                    st.metric("Land Type", land_type)
                with col3:
                    st.metric("Water Presence", water_status)

                st.write(
                    f"Indices: NDVI={soil_info['ndvi']}, NDMI={soil_info['ndmi']}, "
                    f"BSI={soil_info['bsi']} | Confidence={soil_info['confidence']}"
                )
                if soil_info.get("source") == "live":
                    st.success("Source: live Sentinel Hub data")
                else:
                    st.caption("Source: synthetic fallback (Sentinel Hub not authenticated — see README for OAuth setup)")
            else:
                st.warning("Could not analyze satellite indices for this point.")

    st.divider()
    st.subheader("Plotted Area Analysis (from points)")
    if len(st.session_state.shape_points) == 0:
        st.info("Plot points by clicking on map to analyze area soil and water.")
    else:
        sample_points = st.session_state.shape_points[: min(12, len(st.session_state.shape_points))]
        analyses = []
        for p_lat, p_lon in sample_points:
            info = get_soil_type_for_location(p_lat, p_lon)
            if info:
                analyses.append(info)

        if analyses:
            soil_counts = pd.Series([a["soil_type"] for a in analyses]).value_counts()
            dominant_soil = soil_counts.index[0]
            avg_ndmi = float(np.mean([a["ndmi"] for a in analyses]))
            avg_ndvi = float(np.mean([a["ndvi"] for a in analyses]))
            avg_bsi = float(np.mean([a["bsi"] for a in analyses]))

            agg = {
                "soil_type": dominant_soil,
                "ndvi": round(avg_ndvi, 3),
                "ndmi": round(avg_ndmi, 3),
                "bsi": round(avg_bsi, 3),
                "confidence": "Aggregated",
            }
            land_type, water_status = classify_land_and_water(agg)
            suitability, suitability_reason = classify_cultivation_suitability(agg, land_type, water_status)

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Dominant Soil Type", dominant_soil)
            with c2:
                st.metric("Land Type", land_type)
            with c3:
                st.metric("Water Presence", water_status)
            with c4:
                st.metric("Cultivation Suitability", suitability)
            if suitability == "Suitable":
                st.success(suitability_reason)
            elif suitability == "Marginal":
                st.warning(suitability_reason)
            else:
                st.error(suitability_reason)

            st.write(
                f"Plotted points used: {len(analyses)} | "
                f"Avg NDVI={agg['ndvi']}, NDMI={agg['ndmi']}, BSI={agg['bsi']}"
            )
            live_count = sum(1 for a in analyses if a.get("source") == "live")
            if live_count == len(analyses):
                st.success("Source: live Sentinel Hub data for all points")
            elif live_count == 0:
                st.caption("Source: synthetic fallback for all points (Sentinel Hub not authenticated)")
            else:
                st.caption(f"Source: live for {live_count}/{len(analyses)} points, synthetic fallback for the rest")
        else:
            st.warning("Could not analyze plotted points.")

    st.divider()
    st.subheader("Field Health Imaging")
    st.caption("Satellite-derived overlays on the basemap for six diagnostic layers, scoped to this one field.")

    hcol1, hcol2, hcol3 = st.columns(3)
    with hcol1:
        layer_name = st.selectbox("Layer", list(LAYER_DEFINITIONS.keys()))
    with hcol2:
        imaging_resolution = st.slider("Imaging grid resolution", min_value=3, max_value=10, value=6, key="imaging_res")
    with hcol3:
        map_style = st.radio("Map style", ["Grid cells (sharp)", "Smooth heatmap"], key="imaging_style")

    if st.button("Generate Field Health Imagery", use_container_width=True):
        try:
            cells = generate_field_health_layers(
                st.session_state.shape_points,
                center_lat=st.session_state.farmer_details.get("latitude") or st.session_state.map_lat,
                center_lon=st.session_state.farmer_details.get("longitude") or st.session_state.map_lon,
                grid_resolution=imaging_resolution,
                soil_info=st.session_state.soil_information,
            )
            st.session_state["health_imaging_cells"] = cells
        except ValueError as e:
            st.error(str(e))

    cells = st.session_state.get("health_imaging_cells")
    if cells:
        layer_def = LAYER_DEFINITIONS[layer_name]
        value_key = layer_def["key"]
        st.caption(layer_def["description"])

        live_count = sum(1 for c in cells if c.get("source") == "live")
        if live_count == len(cells):
            st.success("Live satellite data (Sentinel Hub) for all zones.")
        elif live_count == 0:
            st.warning(
                "Showing estimated/synthetic zones -- no live satellite connection is configured. "
                "Sentinel Hub via the Copernicus Data Space Ecosystem offers a free tier; see "
                "'Get live satellite data' below the map for setup steps."
            )
        else:
            st.caption(f"Source: live for {live_count}/{len(cells)} zones, synthetic fallback for the rest.")

        center_lat = sum(c["lat"] for c in cells) / len(cells)
        center_lon = sum(c["lon"] for c in cells) / len(cells)

        # Cell footprint in degrees, so grid-cell mode draws contiguous
        # tiles instead of floating dots -- derived from the spacing
        # between this field's own grid points, single-farm scale only.
        lats = sorted(set(round(c["lat"], 8) for c in cells))
        lons = sorted(set(round(c["lon"], 8) for c in cells))
        lat_step = min((b - a for a, b in zip(lats, lats[1:])), default=0.0006) or 0.0006
        lon_step = min((b - a for a, b in zip(lons, lons[1:])), default=0.0006) or 0.0006

        hm = folium.Map(location=[center_lat, center_lon], zoom_start=17, tiles=None)
        folium.TileLayer(
            tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
            attr="Google Satellite",
            name="Google Satellite",
            overlay=False,
            control=True,
        ).add_to(hm)
        Fullscreen().add_to(hm)
        MiniMap(toggle_display=True).add_to(hm)
        LocateControl(auto_start=False).add_to(hm)
        MousePosition().add_to(hm)
        MeasureControl(primary_length_unit="kilometers").add_to(hm)

        if map_style == "Smooth heatmap":
            heat_data = [[c["lat"], c["lon"], c[value_key]] for c in cells]
            HeatMap(
                heat_data,
                radius=35,
                blur=25,
                max_zoom=18,
                gradient=layer_def["gradient"],
                min_opacity=0.4,
            ).add_to(hm)
            for c in cells:
                folium.CircleMarker(
                    [c["lat"], c["lon"]],
                    radius=2,
                    color="white",
                    fill=True,
                    fill_opacity=0.9,
                    tooltip=f"Zone {c['zone_id']} | {layer_name}: {c[value_key]}",
                ).add_to(hm)
        else:
            # Sharp, tiled grid-cell rectangles -- crisper "vegetation
            # index grid" look, closer to standard NDVI zone maps than a
            # blurred heatmap.
            for c in cells:
                color = _interpolate_gradient_color(c[value_key], layer_def["gradient"])
                folium.Rectangle(
                    bounds=[
                        [c["lat"] - lat_step / 2, c["lon"] - lon_step / 2],
                        [c["lat"] + lat_step / 2, c["lon"] + lon_step / 2],
                    ],
                    color="#ffffff",
                    weight=1,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.65,
                    tooltip=f"Zone {c['zone_id']} | {layer_name}: {c[value_key]} | {c.get('source', 'synthetic')}",
                ).add_to(hm)

        _add_map_legend(hm, f"{layer_name} (0=low, 1=high)", layer_def["gradient"])

        st_folium(
            hm, height=520, width=None, returned_objects=[],
            key=f"health_imaging_{center_lat:.5f}_{center_lon:.5f}_{layer_name.replace(' ', '_')}_{map_style}",
        )

        values = [c[value_key] for c in cells]
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Field average", round(sum(values) / len(values), 3))
        with col2:
            st.metric("Highest zone", round(max(values), 3))
        with col3:
            st.metric("Zones above 0.6 (high)", sum(1 for v in values if v > 0.6))

        with st.expander("Get live satellite data (free)", expanded=False):
            st.markdown(
                "Two free options -- Sentinel Hub is tried first, Agromonitoring is used "
                "automatically if Sentinel Hub isn't configured:\n\n"
                "**Option A -- Sentinel Hub (more setup, best coverage)**\n"
                "1. Create a free account at dataspace.copernicus.eu\n"
                "2. Profile icon -> Sentinel Hub -> User Settings -> OAuth Clients -> create one\n"
                "3. Add to `.env`: `SENTINEL_HUB_CLIENT_ID=...` and `SENTINEL_HUB_CLIENT_SECRET=...`\n"
                "4. Install: `pip install -r requirements-optional.txt`\n\n"
                "**Option B -- Agromonitoring (simpler, one flat key, no OAuth)**\n"
                "1. Create a free account at agromonitoring.com\n"
                "2. Add to `.env`: `AGRO_API_KEY=...`\n"
                "3. No extra install needed -- uses `requests`, already installed\n\n"
                "Either way: restart the app and zones will switch from 'synthetic fallback' "
                "to 'live' automatically, no code changes needed."
            )
    else:
        st.info("Click 'Generate Field Health Imagery' to render the selected layer for this field.")


elif page == "Precision Spraying (Stage 2)":
    page_header("Stage 2: Precision Spraying", icon=PAGE_ICONS["Precision Spraying (Stage 2)"])
    st.caption("Stage 2 answers: 'Where should the chemical be applied?' — it builds on Stage 1's satellite analysis.")
    st.write(
        "Satellite imagery first screens the whole field for possible crop stress. "
        "A simulated drone with RGB and thermal cameras then inspects only the flagged areas. "
        "The result is a prescription map showing exactly where to spray, so the drone applies "
        "chemical only to the locations that actually need it."
    )
    st.divider()

    moa_result = st.session_state.get("last_moa_result", {})
    if not moa_result:
        st.warning(
            "Run ML & MOA Analysis first — Stage 2 uses the MOA-optimized dose from Stage 1 "
            "as the maximum rate for the prescription map."
        )
    else:
        st.info(
            f"Using Stage 1 result: **{moa_result['next_crop']}** with "
            f"**{moa_result['chemical_name']}** at **{moa_result['optimized_dose_per_acre']} kg/acre** "
            f"(MOA-optimized) across {moa_result['farm_area_acres']:.1f} acres."
        )

        grid_resolution = st.slider(
            "Grid resolution (higher = more zones, slower)", min_value=2, max_value=6, value=4
        )
        has_boundary = len(st.session_state.shape_points) >= 3
        if has_boundary:
            st.caption(f"Using the {len(st.session_state.shape_points)}-point farm boundary plotted on the Maps page.")
        else:
            st.caption("No farm boundary plotted on the Maps page yet — using a small default area around the farmer's saved location.")

        if st.button("Step 1: Run Satellite Screening", use_container_width=True):
            try:
                grid_points = generate_field_grid(
                    st.session_state.shape_points,
                    center_lat=st.session_state.farmer_details.get("latitude"),
                    center_lon=st.session_state.farmer_details.get("longitude"),
                    grid_resolution=grid_resolution,
                )
                st.session_state["stage2_screened"] = satellite_screen_field(grid_points)
                st.session_state.pop("stage2_inspected", None)
                st.session_state.pop("stage2_prescription", None)
            except ValueError as e:
                st.error(str(e))

        screened = st.session_state.get("stage2_screened")
        if screened:
            screened_df = pd.DataFrame(screened)
            candidate_count = int(screened_df["needs_drone_inspection"].sum())
            live_count = int((screened_df["source"] == "live").sum())
            st.write(f"Satellite screening complete: {len(screened_df)} zones scanned, "
                     f"**{candidate_count} flagged** as possible stress candidates for drone inspection.")
            if live_count == len(screened_df):
                st.success("Source: live Sentinel Hub data for all zones")
            elif live_count == 0:
                st.caption("Source: synthetic fallback for all zones (Sentinel Hub not authenticated — see README)")
            else:
                st.caption(f"Source: live for {live_count}/{len(screened_df)} zones, synthetic fallback for the rest")
            st.dataframe(
                screened_df[["zone_id", "ndvi", "soil_type", "source", "satellite_stress_class", "needs_drone_inspection"]],
                use_container_width=True,
            )

            if st.button("Step 2: Deploy Drone (RGB + Thermal) on Flagged Zones", use_container_width=True):
                st.session_state["stage2_inspected"] = simulate_drone_thermal_scan(screened)
                st.session_state.pop("stage2_prescription", None)

        inspected = st.session_state.get("stage2_inspected")
        if inspected:
            inspected_df = pd.DataFrame(inspected)
            st.write("Drone inspection complete. Final severity per zone (healthy zones were never visited by the drone):")
            st.dataframe(
                inspected_df[["zone_id", "drone_inspected", "thermal_anomaly_c", "final_stress_class"]],
                use_container_width=True,
            )

            zone_area = moa_result["farm_area_acres"] / max(1, len(inspected))
            if st.button("Step 3: Generate Prescription Map", use_container_width=True):
                prescription, summary = generate_prescription_map(
                    inspected, moa_result["optimized_dose_per_acre"], zone_area
                )
                st.session_state["stage2_prescription"] = prescription
                st.session_state["stage2_summary"] = summary

        prescription = st.session_state.get("stage2_prescription")
        summary = st.session_state.get("stage2_summary")
        if prescription and summary:
            st.subheader("Prescription Map")

            severity_color = {
                "Healthy": "green",
                "Mild stress": "yellow",
                "Moderate stress": "orange",
                "Severe stress": "red",
            }
            center_lat = sum(z["lat"] for z in prescription) / len(prescription)
            center_lon = sum(z["lon"] for z in prescription) / len(prescription)
            pm = folium.Map(location=[center_lat, center_lon], zoom_start=16, tiles=None)
            folium.TileLayer(
                tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
                attr="Google Satellite",
                name="Google Satellite",
                overlay=False,
                control=True,
            ).add_to(pm)
            Fullscreen().add_to(pm)
            MiniMap(toggle_display=True).add_to(pm)
            LocateControl(auto_start=False).add_to(pm)
            MousePosition().add_to(pm)
            MeasureControl(primary_length_unit="kilometers").add_to(pm)
            for zone in prescription:
                folium.CircleMarker(
                    [zone["lat"], zone["lon"]],
                    radius=8 if zone["will_spray"] else 4,
                    color=severity_color.get(zone["final_stress_class"], "gray"),
                    fill=True,
                    fill_opacity=0.8,
                    tooltip=(
                        f"Zone {zone['zone_id']} | {zone['final_stress_class']} | "
                        f"{zone['prescribed_dose_kg']} kg prescribed"
                    ),
                ).add_to(pm)
            st_folium(
                pm, height=480, width=None, returned_objects=[],
                key=f"prescription_map_{center_lat:.5f}_{center_lon:.5f}",
            )
            st.caption("Green = healthy (skipped) · Yellow = mild · Orange = moderate · Red = severe (full dose)")

            st.subheader("Spray Summary")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Zones sprayed", f"{summary['sprayed_zones']} / {summary['total_zones']}")
            with col2:
                st.metric("Area sprayed", f"{summary['area_sprayed_pct']}%")
            with col3:
                st.metric("Precision chemical use", f"{summary['total_precision_chemical_kg']} kg")
            with col4:
                st.metric("Stage 2 additional savings", f"{summary['stage2_additional_savings_pct']}%")

            st.write(
                f"Blanket spraying the MOA-optimized dose over the whole field would use "
                f"**{summary['total_blanket_chemical_kg']} kg**. Spraying only the "
                f"{summary['sprayed_zones']} flagged zones at severity-scaled rates uses "
                f"**{summary['total_precision_chemical_kg']} kg** — an additional "
                f"**{summary['stage2_additional_savings_kg']} kg saved** on top of Stage 1's MOA reduction."
            )


elif page in ("Farmer Awareness", "Agriculture Information"):
    page_header("Farmer Awareness Assistant", icon=PAGE_ICONS["Farmer Awareness"])
    st.caption(
        "Ask questions about safe and effective chemical use in plain language. "
        "This is separate from the MOA/ML dose optimization -- it's here to explain "
        "the 'why' behind safe practices, not to replace a doctor or agronomist."
    )

    if is_groq_configured() and is_gemini_configured():
        st.success("Groq (primary, free) and Gemini (backup) are both configured.")
    elif is_groq_configured():
        st.success("Groq is configured (free tier, primary). (Tip: add GEMINI_API_KEY too for a backup provider.)")
    elif is_gemini_configured():
        st.success("Gemini is configured. (Tip: add GROQ_API_KEY too for a free primary backup.)")
    else:
        st.info("No AI provider configured -- set GROQ_API_KEY (free, recommended) and/or GEMINI_API_KEY in .env. Showing a static awareness FAQ below instead of a live chatbot.")

    if "awareness_chat" not in st.session_state:
        st.session_state["awareness_chat"] = []

    for turn in st.session_state["awareness_chat"]:
        with st.chat_message(turn["role"]):
            st.write(turn["text"])

    user_question = st.chat_input("Ask a question about safe chemical use...")
    if user_question:
        st.session_state["awareness_chat"].append({"role": "user", "text": user_question})
        with st.chat_message("user"):
            st.write(user_question)

        result = ask_awareness_bot(user_question, st.session_state["awareness_chat"][:-1])
        with st.chat_message("model"):
            if result.get("available"):
                st.write(result["answer"])
                if result.get("provider"):
                    st.caption(f"Answered by: {result['provider']}")
                st.session_state["awareness_chat"].append({"role": "model", "text": result["answer"]})
            else:
                st.warning(result.get("error", "Assistant unavailable."))
                st.write("Here are some common questions and answers instead:")
                for item in result.get("fallback_faq", STATIC_FAQ):
                    with st.expander(item["question"]):
                        st.write(item["answer"])

    st.divider()
    st.subheader("Common Questions")
    st.caption("Always available, whether or not the live assistant is configured.")
    for item in STATIC_FAQ:
        with st.expander(item["question"]):
            st.write(item["answer"])

    st.warning(
        "This assistant does not replace medical advice, a licensed agronomist, "
        "or your local agriculture extension officer -- especially for poisoning "
        "symptoms or regulatory questions."
    )


elif page == "Harvest Outcomes":
    page_header("Harvest Outcomes")
    st.caption(
        "This is how the yield model stops being purely synthetic: log what a field actually "
        "yielded at harvest, tied back to the prediction made earlier in the season. Every row here "
        "is one real (predicted, actual) pair -- the more of these get logged, the more the accuracy "
        "numbers below mean something."
    )

    with st.form("harvest_outcome_form"):
        col1, col2 = st.columns(2)
        with col1:
            ho_student = st.text_input("Student Name", value=st.session_state.student_registration.get("student_name", ""))
            ho_farmer = st.text_input("Farmer Name", value=st.session_state.farmer_details.get("farmer_name", ""))
            ho_crop = st.text_input("Crop")
        with col2:
            ho_predicted = st.number_input("Predicted Yield (kg/acre)", min_value=0.0, step=10.0,
                                             help="The number shown on ML & MOA Analysis at the time of prediction")
            ho_actual = st.number_input("Actual Yield at Harvest (kg/acre)", min_value=0.0, step=10.0)
            ho_date = st.date_input("Harvest Date")
        ho_notes = st.text_area("Notes (optional)", placeholder="Anything unusual this season -- drought, pest outbreak, late planting, etc.")

        if st.form_submit_button("Record Harvest Outcome", use_container_width=True):
            if not ho_farmer or not ho_crop or ho_actual <= 0:
                st.error("Farmer name, crop, and actual yield are required.")
            else:
                ok = insert_harvest_outcome(
                    student_name=ho_student, farmer_name=ho_farmer, crop=ho_crop,
                    predicted_yield_kg_per_acre=float(ho_predicted),
                    actual_yield_kg_per_acre=float(ho_actual),
                    harvest_date=str(ho_date), notes=ho_notes,
                )
                if ok:
                    st.success("Harvest outcome recorded.")
                else:
                    st.error("Could not save -- check the Database page / logs.")

    st.divider()
    st.subheader("Model Accuracy So Far")
    summary = fetch_harvest_accuracy_summary()
    if summary["count"] == 0:
        st.info("No harvest outcomes recorded yet. Once a few are logged, real accuracy numbers will appear here instead of the synthetic-data metrics reported during training.")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Outcomes recorded", summary["count"])
        with col2:
            st.metric("Mean Absolute Error", f"{summary['mae_kg_per_acre']} kg/acre")
        with col3:
            st.metric("Mean Error %", f"{summary['mean_error_pct']}%" if summary["mean_error_pct"] is not None else "N/A")

        outcomes_df = pd.DataFrame(summary["rows"])
        if not outcomes_df.empty:
            st.dataframe(
                outcomes_df[["farmer_name", "crop", "predicted_yield_kg_per_acre", "actual_yield_kg_per_acre", "harvest_date", "notes"]],
                use_container_width=True,
            )
            if len(outcomes_df) >= 2:
                chart_df = outcomes_df[["farmer_name", "predicted_yield_kg_per_acre", "actual_yield_kg_per_acre"]].rename(
                    columns={"predicted_yield_kg_per_acre": "Predicted", "actual_yield_kg_per_acre": "Actual"}
                )
                melted = chart_df.melt(id_vars="farmer_name", var_name="Type", value_name="Yield (kg/acre)")
                grouped_chart = alt.Chart(melted).mark_bar().encode(
                    x=alt.X("farmer_name:N", axis=alt.Axis(labelColor=CHART_LABEL_COLOR, titleColor=CHART_LABEL_COLOR,
                                                            labelFontSize=12, titleFontSize=12, labelFontWeight="bold",
                                                            titleFontWeight="bold", title="Farmer")),
                    y=alt.Y("Yield (kg/acre):Q", axis=alt.Axis(labelColor=CHART_LABEL_COLOR, titleColor=CHART_LABEL_COLOR,
                                                                labelFontSize=12, titleFontSize=12, labelFontWeight="bold",
                                                                titleFontWeight="bold", gridColor="#b7a274")),
                    color=alt.Color("Type:N", scale=alt.Scale(range=[CHART_GREEN, CHART_BROWN]),
                                     legend=alt.Legend(labelColor=CHART_LABEL_COLOR, titleColor=CHART_LABEL_COLOR,
                                                        labelFontWeight="bold")),
                    xOffset="Type:N",
                    tooltip=["farmer_name", "Type", "Yield (kg/acre)"],
                )
                st.altair_chart(grouped_chart.configure_view(strokeWidth=0).properties(height=320), use_container_width=True)

        if summary["count"] < 10:
            st.caption(f"Only {summary['count']} outcome(s) so far -- treat this accuracy number as an early signal, not a validated result. Statistical confidence improves substantially past ~20-30 recorded outcomes.")

    st.divider()
    st.subheader("Model Retraining")
    st.caption(
        "Retrain the yield model on the current dataset (dataset/telangana_soil_data.csv). "
        "As harvest outcomes accumulate you can fold them into that CSV and retrain here, "
        "closing the loop back to the ML & MOA Analysis step."
    )
    if st.button("Retrain model now"):
        with st.spinner("Retraining..."):
            _, metrics = train_yield_model(save=True)
        st.success(f"Model retrained and saved. Test MAE: {metrics.get('mae_kg_per_acre', 'n/a')} kg/acre")

    st.divider()
    st.subheader("Start Next Season")
    st.caption(
        "This is not a one-time process -- the pipeline repeats every season for the same "
        "farmer. This keeps the farmer's identity and field details and sends you back to "
        "Soil Information to begin the next cycle."
    )
    if st.session_state.farmer_details:
        if st.button(f"Start next season for {st.session_state.farmer_details.get('farmer_name', 'this farmer')}"):
            st.session_state.soil_information = {}
            st.success("Ready for the next season. Go to Soil Information to continue -- Farmer Details are kept as-is.")
    else:
        st.info("No farmer currently loaded -- complete Farmer Details first.")


elif page == "Crop Results":
    page_header("Crop Results", icon=PAGE_ICONS["Crop Results"])
    st.caption("Recommended crop and predicted yield from completed ML analyses -- read-only.")
    records = fetch_all("ml_results")
    if records:
        df = pd.DataFrame(records)[[
            "created_at", "farmer_name", "next_crop", "recommended_crop",
            "predicted_yield", "confidence_score", "limiting_factor",
        ]]
        search = st.text_input("Filter by farmer name")
        if search:
            df = df[df["farmer_name"].str.contains(search, case=False, na=False)]
        st.dataframe(df, use_container_width=True, height=400)
    else:
        st.info("No crop results yet -- check back once your submission has been analyzed.")


elif page == "Soil Results":
    page_header("Soil Results", icon=PAGE_ICONS["Soil Results"])
    st.caption("Soil values recorded per analyzed farm and your submission status -- read-only.")

    st.subheader("Your submission status")
    sub_search = st.text_input("Search your submission by farmer name", key="soil_status_search")
    submissions = fetch_all("soil_submissions")
    if submissions:
        sdf = pd.DataFrame(submissions)[["created_at", "farmer_name", "soil_type", "ph", "organic_carbon", "moisture", "status"]]
        if sub_search:
            sdf = sdf[sdf["farmer_name"].str.contains(sub_search, case=False, na=False)]
        st.dataframe(sdf, use_container_width=True, height=260)
    else:
        st.info("No soil submissions yet.")

    st.subheader("Analyzed soil results")
    records = fetch_all("soil_entries")
    if records:
        df = pd.DataFrame(records)[["farmer_name", "district", "village", "ph", "carbon", "moisture", "chemical"]]
        st.dataframe(df, use_container_width=True, height=300)
    else:
        st.info("No analyzed soil results yet.")


elif page == "ML/MOA Results":
    page_header("ML/MOA Results", icon=PAGE_ICONS["ML/MOA Results"])
    st.caption("Yield prediction and MOA chemical-dose optimization results -- read-only.")

    st.subheader("ML: Yield & Crop Recommendation")
    ml_records = fetch_all("ml_results")
    if ml_records:
        st.dataframe(pd.DataFrame(ml_records).drop(columns=["id"], errors="ignore"), use_container_width=True, height=280)
    else:
        st.info("No ML results yet.")

    st.subheader("MOA: Chemical Dose Optimization")
    moa_records = fetch_all("moa_results")
    if moa_records:
        st.dataframe(pd.DataFrame(moa_records).drop(columns=["id"], errors="ignore"), use_container_width=True, height=280)
    else:
        st.info("No MOA results yet.")


elif page == "Charts":
    page_header("Charts", icon=PAGE_ICONS["Charts"])
    st.caption("Summary charts over saved results -- these only chart already-computed data, they don't run the model.")

    moa_records = fetch_all("moa_results")
    if moa_records:
        moa_df = pd.DataFrame(moa_records)
        st.subheader("Chemical Dose: Initial vs Optimized (kg/acre)")
        by_crop = moa_df.groupby("next_crop")[["initial_dose", "optimized_dose"]].mean().reset_index()
        melted = by_crop.melt("next_crop", var_name="dose_type", value_name="kg_per_acre")
        chart = alt.Chart(melted).mark_bar().encode(
            x="next_crop:N", y="kg_per_acre:Q", color="dose_type:N", xOffset="dose_type:N",
        )
        st.altair_chart(chart, use_container_width=True)

        st.subheader("Average Dose Reduction % by Crop")
        reduction_chart = alt.Chart(moa_df).mark_bar().encode(
            x="next_crop:N", y="mean(reduction_percentage):Q",
        )
        st.altair_chart(reduction_chart, use_container_width=True)
    else:
        st.info("No MOA results yet to chart.")

    ml_records = fetch_all("ml_results")
    if ml_records:
        ml_df = pd.DataFrame(ml_records)
        st.subheader("Predicted Yield Distribution")
        yield_chart = alt.Chart(ml_df).mark_bar().encode(
            x=alt.X("predicted_yield:Q", bin=True), y="count()",
        )
        st.altair_chart(yield_chart, use_container_width=True)
    else:
        st.info("No ML results yet to chart.")


elif page == "Project Results":
    page_header("Project Results", icon=PAGE_ICONS["Project Results"])
    st.caption("Overall project stats -- read-only.")

    students = fetch_all("student_data")
    farmers = fetch_all("farmer_data")
    submissions = fetch_all("soil_submissions")
    moa_records = fetch_all("moa_results")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Students Registered", len(students))
    with c2:
        st.metric("Farmers Covered", len(farmers))
    with c3:
        pending_n = sum(1 for s in submissions if s.get("status") == "pending")
        st.metric("Submissions Pending", pending_n)
    with c4:
        st.metric("Analyses Completed", len(moa_records))

    if moa_records:
        avg_reduction = sum(r.get("reduction_percentage") or 0 for r in moa_records) / len(moa_records)
        st.metric("Average Chemical Dose Reduction", f"{avg_reduction:.1f}%")

    accuracy = fetch_harvest_accuracy_summary()
    if accuracy:
        st.subheader("Predicted vs Actual Harvest Accuracy")
        st.json(accuracy)


elif page == "Database":
    page_header("Database Records", icon=PAGE_ICONS["Database"])
    st.caption("View saved student, farmer, soil, ML and MOA records.")

    tables = [
        "student_data",
        "farmer_data",
        "soil_entries",
        "ml_results",
        "moa_results",
        "harvest_outcomes",
    ]

    for tname in tables:
        st.subheader(tname)
        records = fetch_all(tname)
        if records:
            df = pd.DataFrame(records)
            st.dataframe(df, use_container_width=True, height=260)
            st.download_button(
                f"Download {tname} CSV",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name=f"{tname}.csv",
                mime="text/csv",
                key=f"dl_{tname}",
            )
        else:
            st.info(f"No records in {tname}.")
