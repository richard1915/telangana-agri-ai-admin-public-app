# Telangana Smart Agriculture AI

ML-based crop/chemical recommendations + a Meerkat Optimization Algorithm (MOA)
to minimize chemical dose while maintaining target crop yield, for Telangana
farms.

## Project structure

```
.
├── backend/
│   ├── database.py            # SQLite persistence (students, farmers, soil, ML/MOA results)
│   ├── model.py                # Crop recommendation, chemical lookup, residue decay, safety checks
│   ├── meerkat_optimizer.py    # Meerkat Optimization Algorithm + rule-based & trained ML yield prediction
│   ├── satellite_api.py        # Sentinel Hub NDVI / soil-index lookup (falls back to synthetic data)
│   ├── weather_api.py          # OpenWeatherMap lookup
│   ├── precision_spray.py      # Stage 2: satellite screening -> drone inspection -> prescription map
│   ├── crop_health_imaging.py  # Six diagnostic heatmap layers (crop stress, water stress, etc.)
│   ├── chemical_composition.py # Real N-P-K composition per chemical, grade auto-parsing
│   └── awareness_bot.py        # Gemini/Groq-backed farmer awareness chat, with static FAQ fallback
├── app_streamlit.py            # The app — full multi-page UI
├── run_streamlit.py            # Launches the app
├── dataset/telangana_soil_data.csv   # Sample soil dataset (District, Soil_Type, pH, NPK, Yield, etc.)
├── requirements.txt             # Core dependencies -- always installs cleanly
├── requirements-optional.txt    # Optional: sentinelhub (real satellite data)
└── .env.example                 # Copy to .env and add your own API keys
```

## Pages

Dashboard → Student Registration → Farmer Details → Soil Information →
ML & MOA Analysis → Maps (incl. Field Health Imaging) →
Precision Spraying (Stage 2) → Farmer Awareness → Harvest Outcomes → Database

## End-to-end workflow

This is a repeating cycle per farmer, not a one-time pipeline -- Harvest
Outcomes feeds Model Retraining, which feeds the same farmer's next season:

```
Farmer Registration  (+ consent capture -- required before saving)
        |
Field Registration
        |
GPS Location
        |
Soil Health Card Upload
        |
Weather API
        |
Satellite Data (NDVI, Moisture)  -> flags candidate zones for inspection,
        |                            not proof spraying is required
Agronomic Rule Check  (hand-authored safety boundaries, model-independent)
        |
Machine Learning Prediction  (+ confidence score)
        |
Crop Recommendation
        |
MOA Optimization  (benchmarked against the rule-based estimate and the
        |           trained ML model, not assumed to be the best method)
Agronomist / AEO Validation  (mandatory review gate; low confidence or
        |                      any rule flag routes here automatically)
Precision Spraying Map
        |
Farmer Awareness Assistant
        |
Harvest Outcome Recording
        |
Model Performance Evaluation
        |
Database
        |
Model Retraining
        |
        +--> back to Field Registration / Soil Information for the same
             farmer's next season ("Start Next Season" button on the
             Harvest Outcomes page)
```

Implementation notes:
- **Consent** is captured on the Farmer Details page (`consent_given`,
  `consent_date` in `farmer_data`) and is required before the form saves.
- **Agronomic Rule Check** (`backend/model.py::agronomic_rule_check`) is a
  hand-authored safety layer, separate from and upstream of the ML/MOA
  output, so no model prediction can override it.
- **Agronomist/AEO Validation** is a mandatory sign-off step on the ML &
  MOA Analysis page. Any rule-check flag, or ML confidence below 60%,
  automatically requires review before the recommendation is used;
  decisions are logged to the new `agronomist_reviews` table.
- **Model Retraining** and **Start Next Season** are both on the Harvest
  Outcomes page -- retraining re-runs `train_yield_model()` on the current
  dataset, and "Start Next Season" clears Soil Information while keeping
  the farmer's saved details, so the same farmer can be run through
  another cycle.

## Deploy for free (Render + Supabase) — no Azure needed

This app is Streamlit, so it deploys as a normal web service. Render's free
tier needs no credit card and works well for a student project. Two pieces:

**1. Push this project to a GitHub repo** (if it isn't already). Make sure
`.env` is *not* committed — it's already in `.gitignore`.

**2. Database: create a free Supabase Postgres project first.**
Render's own free Postgres expires 30 days after creation, which will wipe
your saved farmer/soil/student records — not good for something students
will keep using. Supabase's free Postgres project does **not** expire, so
use it as `DATABASE_URL` instead:
1. Sign up free at https://supabase.com, create a new project.
2. Project Settings -> Database -> copy the "Connection string" (URI form).
3. Keep it handy for step 4 below.

**3. Deploy the web service on Render:**
1. Sign up free at https://render.com (no card required) and connect your
   GitHub repo.
2. Render will detect `render.yaml` in this repo and offer to create the
   service from it ("New +" -> "Blueprint"). If it doesn't, create a
   **Web Service** manually with:
   - Build command: `pip install -r requirements.txt psycopg2-binary`
   - Start command: `streamlit run app_streamlit.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true`
3. In the service's **Environment** tab, add `DATABASE_URL` with the
   Supabase connection string from step 2. Add any of the other optional
   keys from `.env.example` (weather, satellite, Gemini/Groq) the same way
   — every one is optional and the app falls back to synthetic/static data
   without it.
4. Deploy. Render gives you a public `https://<your-app>.onrender.com` URL
   others (students, farmers, mentors) can open directly — this is your
   "mobile app" too, since a Streamlit page like this works fine as a
   mobile web app in any phone browser; no app-store build needed.

**Free-tier limits worth knowing:**
- The free web service **spins down after 15 minutes with no visitors**
  and takes ~30-60 seconds to wake back up on the next visit. Fine for a
  student project, not for something that needs to feel instant 24/7.
- Render's free web services have an **ephemeral filesystem** — anything
  written to local disk (including a local SQLite file) is wiped on every
  restart/redeploy/spin-down. This is exactly why `DATABASE_URL` (Supabase)
  matters above: without it, every farmer/soil record students enter would
  disappear the next time the service spins down.
- 750 free instance-hours/month per Render workspace — plenty for a
  single low-traffic student project.

## Setup

```bash
# 1. Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt
# Optional -- real satellite data instead of synthetic (may fail to build
# on some systems, especially Windows without build tools; safe to skip):
pip install -r requirements-optional.txt

# 3. Add your API keys
copy .env.example .env       # Windows
cp .env.example .env         # macOS/Linux
# then edit .env -- every key is optional, the app falls back to
# synthetic/static data for anything not configured
```

Every external API (OpenWeatherMap, Sentinel Hub, Gemini, Groq) has a
safe fallback — the app runs fully without any of them configured.

## Run

```bash
python run_streamlit.py
```
Opens at http://localhost:8501

## Notes on the MOA implementation

`backend/meerkat_optimizer.py` implements a population-based metaheuristic
with:
- **Greedy per-individual selection** — a candidate dose only replaces the
  current one if it scores better (real hill-climbing, not random drift).
- **Adaptive step size** — wide exploration early, fine-tuning later.
- **Sentry-pull behavior** — a portion of moves each iteration pull toward
  the current best-known dose, loosely modeling meerkat sentry/alarm-call
  coordination.
- **Elitism** — the best solution found so far always survives into the
  next generation.
- **Early stopping** — the loop exits once the best score hasn't improved
  for 10 iterations, which also demonstrates convergence for your report.

## Metaheuristic vs machine learning — both are in `meerkat_optimizer.py`

The file is organized into three clearly labeled sections so you can point
to each independently in your report/viva:

1. **Metaheuristic** — `meerkat_chemical_reduction()`. A population-based
   search that optimizes a hand-defined fitness function at run time. No
   training data involved.
2. **Rule-based estimate** — `predict_crop_yield()`. Hand-authored optimal
   ranges per crop combined with a fixed weighted formula. Also not
   trained — kept for transparency and comparison.
3. **Trained ML model** — `train_yield_model()` / `predict_crop_yield_ml()`.
   An actual `scikit-learn` `RandomForestRegressor`, trained on
   `dataset/telangana_soil_data.csv` (which includes a `Yield_kg_per_acre`
   column for supervised learning). This is genuine ML — it learns its own
   parameters from data rather than having them hand-specified. It trains
   itself automatically on first use and caches the model to
   `backend/yield_rf_model.pkl` (gitignored — delete it to force retraining
   after you change the dataset).

The Streamlit "ML & MOA Analysis" page shows the rule-based estimate and
the trained-model prediction side by side for direct comparison.

## Weather auto-fetch

On the Soil Information page, if a farmer's latitude/longitude were saved
in Farmer Details, an "Auto-fetch current weather" button calls
`get_weather_estimate_for_prediction()` in `backend/weather_api.py`, which
combines a live OpenWeatherMap lookup with a seasonal rainfall estimate
(derived from `get_monsoon_forecast()`) and pre-fills the temperature and
rainfall fields. If the live call fails (no key, no internet), it falls
back to Telangana seasonal averages and reports `source: "fallback"` so
you always know which you're looking at.

## Security note

API keys were previously hardcoded in source. They've been moved to
environment variables (`.env`, gitignored) — **do not commit `.env` or your
real keys to GitHub.**

## Stage 2 — Precision Spraying

`backend/precision_spray.py` adds a second stage that answers *"where*
should the chemical be applied?"* (Stage 1 / the ML & MOA Analysis page
answers *"how much"*). It deliberately reuses Stage 1's
`get_soil_type_for_location()` rather than re-implementing satellite
analysis:

1. **Satellite screening** — samples a grid across the field (using the
   plotted farm boundary from the Maps page, or a default extent around
   the farmer's location) and flags low-NDVI cells as stress candidates.
2. **Drone inspection** — a simulated RGB + thermal pass, but *only* over
   the zones flagged in step 1. A thermal anomaly (inversely related to
   NDVI) refines each candidate's severity into Healthy / Mild / Moderate
   / Severe.
3. **Prescription map** — each zone gets a dose that's a severity-scaled
   fraction of the Stage 1 MOA-optimized rate; healthy zones get zero.
   Rendered as a color-coded Leaflet map (green = skipped, red = full
   dose).
4. **Savings summary** — compares total precision-sprayed chemical
   against blanket-spraying the same optimized rate over the whole field,
   showing Stage 2's *additional* savings on top of Stage 1's MOA
   reduction.

Run "ML & MOA Analysis" first so a Stage 1 optimized dose exists in
session state, then visit "Precision Spraying (Stage 2)".

## Field Health Imaging

On the Maps page, "Field Health Imaging" renders six overlays on the
satellite basemap for the single field you've drawn
(`backend/crop_health_imaging.py`):

- **Crop stress** — from NDVI
- **Heat signatures** — simulated canopy thermal anomaly (same model as
  Stage 2's drone inspection)
- **Water stress** — from NDMI
- **Nitrogen deficiency** — proxy from NDVI + bare-soil index
- **Areas of over-application** — healthy zones on a farm that's already
  applying a high chemical dose (uses the farmer's actual entered dose)
- **Areas needing additional nutrients** — stressed zones where the
  farmer's measured NPK is below a reference band

Pick a layer, resolution, and map style (sharp grid cells or a blurred
heatmap) and click "Generate Field Health Imagery". Grid-cell mode tiles
the field with colored rectangles sized to the field's own grid spacing
(single-farm scale, not a regional mosaic) and adds an on-map color
legend. Each layer reports whether it came from live Sentinel Hub data
or the synthetic fallback.

**Getting live data (free):** two options, tried automatically in this
order:

1. **Sentinel Hub** (via the Copernicus Data Space Ecosystem) -- create
   an account at dataspace.copernicus.eu, open the Sentinel Hub
   dashboard from your profile icon, create an OAuth client under User
   Settings, and put the Client ID/Secret in `.env` (see `.env.example`).
   `backend/satellite_api.py` already points at the correct CDSE
   endpoints (`sh_base_url` / `sh_token_url`) -- a valid CDSE OAuth
   client is all that's needed; no code changes. Best NDVI/NDMI/BSI
   coverage, but needs `pip install -r requirements-optional.txt` and
   an OAuth client (a bit more setup).
2. **Agromonitoring** (by OpenWeatherMap) -- simpler: one flat API key
   from agromonitoring.com, no extra package (`requests` is already a
   core dependency), no OAuth. Free tier has a small polygon cap. Only
   provides NDVI/NDWI (used here as an NDMI stand-in) -- no bare-soil
   index, so soil-type classification from this source leans more on
   vegetation/moisture than bare soil. Set `AGRO_API_KEY` in `.env` and
   it's used automatically whenever Sentinel Hub isn't configured.

Without either configured, every layer uses a seeded synthetic fallback
so the app still runs end-to-end. Each result reports both `source`
("live"/"synthetic") and `provider` ("sentinel_hub"/"agromonitoring"/"synthetic").

## Chemical formula analysis (fertilizer vs pest control)

Two real gaps got fixed based on testing feedback:

1. **`backend/chemical_composition.py`** -- chemicals were previously
   treated as interchangeable kg/acre quantities. This module gives each
   one its real nutrient composition (e.g. Urea 46% N vs DAP 18% N + 46%
   P), so `compute_nutrients_delivered()` converts a raw dose into actual
   nutrients delivered, and `compare_chemicals_for_nutrient_gap()` shows
   how much of different chemicals is needed to close the same nutrient
   shortfall. Surfaced on the ML & MOA Analysis page as "Chemical Formula
   Analysis".

2. **Category-aware MOA dose-response curve** -- `meerkat_chemical_reduction()`
   now takes a `chemical_category` ("fertilizer" or "pest_control").
   Fertilizers keep the original smooth diminishing-returns curve;
   pest-control chemicals (pesticide/fungicide/herbicide) use a steeper,
   threshold-like curve with a more conservative safety floor, since
   under-dosing pest control risks an outbreak rather than a gradual
   yield dip. `chemical_composition.infer_chemical_category()` picks the
   right one automatically from the chemical name.

## Farmer Awareness Assistant

New "Farmer Awareness" page (`backend/awareness_bot.py`) -- a chat
assistant answering plain-language safety/usage questions, backed by the
Gemini API when `GEMINI_API_KEY` is set in `.env` (get one at
https://aistudio.google.com/apikey). Without a key, it falls back to a
curated static FAQ automatically, so the feature has value either way.
Explicitly designed to redirect medical/regulatory questions to a real
professional rather than answer them itself.

## Crop coverage — expanded to Telangana's actual priority crops

Previously capped at 15 generic crops. Added 11 more based on real
Telangana agriculture data (PJTSAU / district agriculture statistics),
covering the crops that actually dominate the state's cropped area:

- **Paddy** (alias of Rice — same crop, local name)
- **Turmeric** — Nizamabad is India's largest turmeric market
- **Red Gram / Green Gram / Black Gram / Bengal Gram** — the major pulses
  (previously lumped into one generic "Pulses" entry; legumes get lower
  optimal-N targets since they fix their own nitrogen)
- **Bajra, Sesame, Castor, Soybean, Sunflower**

All 6 factor tables in `predict_crop_yield()` (pH, moisture, temperature,
NPK, rainfall, base yield potential) were extended for every new crop, and
`dataset/telangana_soil_data.csv` was regenerated to match — fixing an
earlier inconsistency where "Turmeric"/"Paddy" existed in the training
data but had no corresponding entries in the yield formula. The trained
ML model (`train_yield_model()`) picks up the new crops automatically on
its next retrain (delete `backend/yield_rf_model.pkl` to force this, or
it happens automatically once the cached model's dataset no longer
matches).

Figures used (base yield, optimal NPK, etc.) are reference approximations
consistent with the rest of this table, not lab-validated numbers — treat
them the same way as the original 15-crop data: a reasonable starting
point, not ground truth.

## Chemical name robustness — branded products, different names, custom grades

Tested and fixed a real gap: the system previously only worked for 11
fixed dropdown chemical names. Now it handles three layers, in order:

1. **Exact match** against the known-chemical table (Urea, DAP, NPK
   blends, etc.)
2. **Auto-parsed N-P-K grade** -- if the chemical name (including a
   branded product name, e.g. "Tata Gromor 28-28-0") contains a
   `N-P-K` or `N:P:K` pattern, it's extracted and used directly. This
   covers most Indian fertilizer products, since the grade is legally
   required on the packaging regardless of brand.
3. **Manual entry fallback** -- if neither of the above resolves (e.g. a
   branded pesticide with no grade in its name), the UI shows an
   expandable form to type in the N-P2O5-K2O percentages straight off the
   product label.

The free-text "Soil Chemical Name" field from Soil Information is now
also analyzed (previously it was stored but never used for composition
analysis).

## Gemini model note

`gemini-2.5-flash` started returning unexpected 404s across many
developers in mid-2026, ahead of its official shutdown date. Default
model is now `gemini-3.5-flash` (current stable GA model). If this
happens again, update `GEMINI_MODEL` in `.env` -- check
https://ai.google.dev/gemini-api/docs/models for the current model list.
Alternatives to Gemini worth considering: Claude API, OpenAI API, or a
locally-run model via Ollama (works offline, no API key, better fit for
areas with unreliable internet -- tradeoff is lower answer quality and
needs a reasonably capable device to run on).

## Maps page fixes (tested end-to-end)

Found and fixed a real bug: typing new coordinates updated the Python
variables, but `st_folium` preserves its own client-side pan/zoom state
across reruns (a known streamlit-folium behavior meant to avoid disrupting
manual map interaction) -- so the visual map never actually recentered to
show the new location's satellite imagery. Fixed by giving `st_folium` a
`key` that changes with the coordinates/zoom, forcing a genuine remount
when you type new coordinates, click the map, or use a nudge button.

Also added, since zoom in/out was the only interaction available before:

- **Nudge buttons** (N/S/E/W with adjustable step size) for fine cursor
  movement without dragging.
- **Geocoder search box** on the map -- type a place name (e.g.
  "Warangal") instead of raw lat/lon.
- **Draw tool** -- draw a polygon/rectangle/marker directly on the map
  with proper drag handles, instead of only click-to-add individual
  points. Drawn shapes can be saved as the plotted boundary used by
  Stage 2 and Field Health Imaging.

## Farmer Awareness Assistant -- now has automatic fallback

`backend/awareness_bot.py` now tries several Gemini model IDs in sequence
(`gemini-3.5-flash` -> `gemini-flash-latest` -> `gemini-2.5-flash` ->
`gemini-2.5-flash-lite`) instead of a single hardcoded one, so a single
model deprecation (which has been happening frequently) doesn't break the
feature outright. If Gemini fails entirely or isn't configured, it
automatically tries **Groq** (`GROQ_API_KEY` in `.env` -- free tier, no
cost, get a key at https://console.groq.com/keys) before finally falling
back to the static FAQ. The chat UI shows which provider actually
answered each message.

## Install fix — "pandas/sklearn/flask/dotenv not working"

Root cause found: `pip install -r requirements.txt` resolves the entire
file as one dependency graph before installing anything. `sentinelhub`
(needs rasterio/GDAL) frequently fails to build on Windows without extra
system libraries -- when it does, **pip aborts the whole install with
nothing installed at all**, not just sentinelhub. That's why pandas,
scikit-learn, flask, and python-dotenv could all appear "not working"
simultaneously even though they're straightforward, reliable packages on
their own.

Fixed by moving `sentinelhub` out of `requirements.txt` into a separate
`requirements-optional.txt`. Verified in a completely fresh virtual
environment that `pip install -r requirements.txt` now installs cleanly
and every previously-failing import (pandas, sklearn.*, flask, jinja2,
dotenv) succeeds.

```bash
pip install -r requirements.txt              # always works, core app
pip install -r requirements-optional.txt      # optional, real satellite data
```

If you'd already tried installing and hit errors, delete your `venv`
folder and recreate it fresh, then reinstall -- a partially-failed
install can leave things in a broken half-installed state.

## Field-selection scoping — analysis now respects your actual field shape

`generate_field_grid()` (used by both Stage 2 and Field Health Imaging)
previously sampled the full rectangular bounding box of your plotted
points -- for a non-rectangular field (an angled or irregular boundary),
this meant analyzing area outside your actual field too. Now filters grid
points to only those genuinely inside your plotted polygon (point-in-
polygon test), with a denser initial sample so filtering still leaves
enough points to analyze. Falls back to the unfiltered grid only if your
field is so narrow/thin that fewer than 4 points would otherwise survive.

## Awareness assistant — Groq now primary, Gemini secondary backup

Switched the default order: **Groq first** (free, no cost, and has been
more reliable than Gemini's rapidly-changing model lineup), Gemini as a
secondary backup, static FAQ as the final fallback.

Also fixed a second real deprecation issue: Groq retired
`llama-3.3-70b-versatile` and `llama-3.1-8b-instant` on June 17, 2026 --
the previous default would have failed the same way Gemini did. Default
is now `openai/gpt-oss-120b` (Groq's current recommended replacement),
with `openai/gpt-oss-20b` as an automatic fallback. Both Groq and Gemini
now try multiple model IDs in sequence rather than a single hardcoded
one, so a single future deprecation doesn't break the feature outright.

## Data storage — SQLite by default, Postgres-ready

`backend/database.py` uses SQLAlchemy so the storage backend is a config
choice, not a code change:

- **No `DATABASE_URL` set** (default) -- SQLite file at `agriculture.db`.
  Zero setup, fine for a single-user demo. This is what you get out of
  the box.
- **`DATABASE_URL` set** in `.env` -- same code, same tables, now backed
  by a real database. To move to Postgres:
  1. Create a free project at https://supabase.com (or any Postgres host)
  2. Copy its connection string into `.env` as `DATABASE_URL`
  3. `pip install psycopg2-binary` (see `requirements-optional.txt`)
  4. Run the app -- `create_db()` creates the tables there instead

SQLite's real limitations for anything beyond a demo: single-writer
locking (two people entering data at once can conflict), lives on
whichever machine runs it (no sync, no backup), and no access control
(anyone with file access sees every farmer's phone number and location).
Worth raising as an open question in the mentorship conversation --
Postgres/Supabase solves the concurrency and backup problems, but access
control and India's DPDP Act compliance (this project collects real names,
phone numbers, and farm locations) are policy decisions, not just a
database swap.

## API layer (FastAPI) -- built, then removed

A `backend/api.py` (FastAPI) + `run_api.py` was built at one point to
expose the MOA/ML/chemical-composition logic as a JSON API for external
integration (e.g. a partner's own backend calling into this project).
Decision was made to remove it and stay Streamlit-only for now --
simpler story for the current demo/mentorship-pitch stage, revisit if a
partner actually needs programmatic integration later.

## Maps — fixed everywhere, not just the main page

The recenter bug (found and fixed earlier on the main Maps page) was
still present on two other maps -- Field Health Imaging and Stage 2's
prescription map -- since each `st_folium()` call needs its own dynamic
`key` to force a remount. Fixed all three. Every map on the site now
recenters correctly when coordinates change.

## Address → coordinates (geocoding)

New `backend/geocoding.py` uses OpenStreetMap's free Nominatim API (no
API key needed) to convert a village/district into latitude/longitude.
On Farmer Details, expand "Find coordinates from an address" above the
form, pick a village/district, and click "Find on map" -- no need to
already know exact coordinates.

## Soil types — Telangana's real 7-type classification

Replaced the generic 4-type list (Red/Black/Laterite/Alluvial) with
Telangana's official classification (Soils of Andhra Pradesh, 1976 --
still the standard reference): **Chalka** (Red Sandy Loam, most common
statewide), **Dubba** (Red Loamy Sand), **Lateritic**, **Shallow-Medium
Black**, **Deep Black (Black Cotton)**, **Salt-affected**, and
**Alluvial**. Each has its own chemical recommendation profile in
`backend/model.py` (e.g. Gypsum-first reclamation for Deep Black and
Salt-affected soils, which run alkaline more often than the others).
Old generic names are kept as aliases so existing records still resolve
correctly.

Also added an EC-based sanity check: if measured Electrical Conductivity
exceeds 4 dS/m (the standard FAO/ICAR salinity threshold) but a different
soil type was selected, the app now flags that Salt-affected Soil is
likely the more accurate classification -- lab data should override a
manual guess.

## Soil Health Card

There's no public API for India's Soil Health Card scheme -- it's not
built for third-party integration. Instead, the app now explains this
plainly when a farmer has indicated a card is available, and reframes
the existing pH/N/P/K/EC/Organic Carbon fields as "enter your card's
real lab-tested values here" rather than generic estimates, since lab
data is always more accurate than the app's own guesses.

## Engineering hardening -- tests, CI, concurrency, real-data feedback loop

Four concrete fixes to close gaps from an honest engineering review:

1. **Automated test suite** (`tests/`, 30 tests, pytest) -- the 49
   documented test cases were a plan, not proof. Converted the highest-
   value ones into real, runnable tests covering the MOA, yield
   prediction, Stage 2, chemical composition, satellite fallback, and
   the database layer. Run with:
   ```bash
   pip install -r requirements-dev.txt
   pytest tests/ -v
   ```
   One of the first runs caught a real edge case in the NDVI-uniqueness
   regression test itself (an overly strict assertion, not a code bug --
   documented in `tests/test_precision_spray.py`) -- a good example of
   why actually running tests matters more than writing them.

2. **CI** (`.github/workflows/tests.yml`) -- runs the full suite plus a
   compile check on every push/PR, across Python 3.11 and 3.12, once this
   is pushed to GitHub.

3. **SQLite concurrency** (`backend/database.py`) -- enabled WAL mode
   and a busy-timeout, which lets reads and a write proceed concurrently
   instead of SQLite's default full-lock behavior. This doesn't turn
   SQLite into a real multi-user database (still one writer at a time,
   still no access control) -- it meaningfully reduces lock contention
   for a handful of concurrent users short of migrating to Postgres, not
   a substitute for it at real scale.

4. **Harvest Outcomes** (new page + `harvest_outcomes` table) -- the
   concrete first step toward validating the yield model on real data
   instead of only synthetic training data. Log what a field actually
   yielded at harvest, tied back to the earlier prediction; an accuracy
   dashboard (MAE, mean error %) builds up automatically as real
   (predicted, actual) pairs accumulate. With zero outcomes logged it
   honestly says so rather than faking a number -- and even once real
   outcomes exist, treat anything under ~20-30 recorded pairs as an
   early signal, not a validated result.

## Advisory-driven additions (technical + strategic recommendations)

The following were added in response to the "Jyothi Agritech Company
Advisory" review document. Each maps to a specific gap the document
raised.

1. **MOA is now benchmarked, not assumed** (`backend/optimizer_comparison.py`)
   -- addresses "The Meerkat Optimization Algorithm is assumed to be
   effective." MOA is run alongside Particle Swarm Optimization, a
   Genetic Algorithm, and a fixed-percentage rule-based baseline on the
   identical dose-reduction problem, and compared on chemical reduction
   %, achieved yield %, and runtime. See the "Optimizer Comparison"
   section on the ML & MOA Analysis page.

2. **Explainable AI (XAI)** (`backend/explainability.py`) -- shows why
   the trained yield model produced a given prediction: SHAP values if
   the optional `shap` package is installed, otherwise a
   permutation-based local importance so it still works without it.
   Also shows model-wide feature importance. See "Why did the model
   predict this?" on the ML & MOA Analysis page.

3. **Model metrics** (`backend/meerkat_optimizer.get_model_metrics()`)
   -- MAE, RMSE, R² on a held-out split, plus k-fold cross-validation
   R² (mean/std), addressing "Uncertainty and model failure are not
   addressed."

4. **PDF reports** (`backend/report_generator.py`) -- downloadable PDF
   bundling soil info, weather, crop recommendation, MOA recommendation,
   estimated savings, environmental impact and a location marker. Built
   with reportlab; the "map" is a vector location marker (no network
   tile fetch required), styled in the app's dark green / brown theme.

5. **Cost savings** (`backend/cost_savings.py`) -- ₹ saved per acre, kg
   chemical saved, and percentage reduction, using an indicative
   price-per-kg table for common Telangana inputs.

6. **Environmental dashboard** (`backend/environmental_impact.py`) --
   chemical reduction %, estimated residue-load reduction % (reuses the
   existing degradation-half-life model), and a directional soil-health
   improvement indicator.

7. **Recurring, per-farmer tracking, not one-time** -- `moa_results` now
   has `season_number` / `optimizer_method` columns, and
   `fetch_farmer_season_summary()` returns every season on record for a
   farmer with cumulative savings. This was already partially true
   (farmer_name-keyed history + progressive per-cycle reduction caps
   existed before this change) -- this adds an explicit season index and
   a dedicated "Farmer Season History" view so it's visible in the UI,
   not just implicit in the database.

8. **Agronomist supervision + cloud deployment note** (Dashboard page) --
   a short panel restating the advisory's suggested workflow (validated
   data -> rule checks -> ML -> optimization -> **agronomist review** ->
   farmer recommendation -> outcome capture) and the suggested cloud/AI
   startup programs (AWS Activate, Microsoft for Startups, Google Cloud
   for Startups, NVIDIA Inception, Esri), consistent with the existing
   `database.py` design that already runs on SQLite locally or
   Postgres/Supabase via `DATABASE_URL` with no code changes. The
   agronomist rule-check / review gate and per-farmer review history
   already existed in this codebase before this change
   (`backend/model.py::agronomic_rule_check`, `agronomist_reviews`
   table) -- this section makes that operating model explicit and
   visible on the Dashboard rather than only enforced in code.

9. **Theme** -- updated to an explicit dark green + brown palette
   (`.streamlit/config.toml`, CSS in `app_streamlit.py`) instead of the
   previous green-only theme.

Not carried over from the advisory (out of scope for this codebase):
organization/partner outreach, commercial model, and the 100-day
action plan are business/operational recommendations with no
corresponding code change -- see the advisory document itself for those.

## Soil Biological Activity Indicator (experimental, Waksman-inspired)

Adds a biological lens next to the app's existing chemical-only
reasoning (pH, N, P, K, EC). Soil fertility is a biological process as
much as a chemical one -- the microbial community in soil is what makes
nutrients plant-available, and organic carbon acts as its food supply
rather than just a nutrient figure. This traces to Selman Waksman's
foundational soil-microbiology research; see Project Review Report,
Section 10, for the full framing.

- New module: `backend/biological_health.py` --
  `estimate_biological_activity()` derives a Low/Moderate/High
  biological-activity level from organic carbon (the base signal),
  **adjusted down by moisture and EC readings already collected
  elsewhere in the app** -- since both independently suppress
  microbial activity even when carbon is adequate. This is
  deliberately more than a re-label of the OC-only
  `soil_health_improvement_indicator` in `environmental_impact.py`:
  two fields with identical carbon but different moisture/EC get
  different levels here.
- Also reports an organic-carbon trend (improving/declining/stable)
  from the farmer's recorded season history (new
  `database.fetch_farmer_soil_carbon_history()`), and an Available-N
  sufficiency note (Low/Medium/High, Soil Health Card bands) by
  default. **Deliberately not** combined into a C:N ratio by default:
  this app collects Available N (kg/acre, Alkaline KMnO4 method), not
  Total N % -- the two aren't interchangeable without lab conversion.
- **Optional real C:N ratio**: when a lab-verified Total Nitrogen % is
  supplied (e.g. Kjeldahl/CHNS test) via `total_nitrogen_pct`, the
  module computes the standard C:N ratio (organic carbon % ÷ Total N %,
  both on the same basis) and bands it narrow / typical (~10-12) / wide.
  Exposed in the app as an optional "Lab-verified Total Nitrogen %"
  input in the same expander -- left at 0 (i.e. not supplied) it has no
  effect and the Available-N band above is shown instead.
- Surfaced in the app as a new **"Soil Biological Activity Indicator
  (experimental)"** expander directly under the Environmental
  Dashboard.
- **Display-only.** It does not feed into `agronomic_rule_check()`, the
  Meerkat Optimizer, or any recommendation/dose shown to the farmer --
  by design, until an agronomist has reviewed the thresholds used here
  (the open questions raised in Section 10 of the report). Every
  result carries an explicit disclaimer to that effect in the UI.
- 27 unit tests in `tests/test_biological_health.py` covering
  availability handling, all three activity levels, the
  moisture/EC downgrade logic, all three trend directions, the
  Available-N bands, and the optional lab-verified C:N ratio (all
  three bands, invalid input, and coexistence with the Available-N note).


