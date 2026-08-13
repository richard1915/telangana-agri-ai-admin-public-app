"""
Database layer -- SQLite by default, Postgres/Supabase-ready.

Same function signatures as before (insert_soil_entry, insert_moa_result,
fetch_all, etc.), so app_streamlit.py doesn't need to change when the
backend changes. What changed is *how* it connects:

  - No DATABASE_URL set -> SQLite file at ../agriculture.db (zero setup,
    same as before). Fine for a single-user demo.
  - DATABASE_URL set (e.g. a Supabase/Postgres connection string) -> same
    code, same tables, now backed by a real multi-user database.

To move to Postgres later: create a Supabase project (or any Postgres
host), put its connection string in .env as DATABASE_URL, install
psycopg2-binary (see requirements-optional.txt), and run the app --
create_db() will create the tables there instead of in SQLite. No other
code changes needed.
"""

import os
from sqlalchemy import (
    create_engine, MetaData, Table, Column, Integer, Float, Text, DateTime, select
)
from sqlalchemy.exc import OperationalError
from sqlalchemy.sql import func
from typing import Dict, List

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "agriculture.db")
# `or` (not just getenv's default arg) so an empty DATABASE_URL= line in
# .env -- which is present but blank, not unset -- still falls back to
# SQLite instead of trying to parse "" as a connection string.
DATABASE_URL = os.getenv("DATABASE_URL") or f"sqlite:///{DB_PATH}"

_connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, pool_pre_ping=True, connect_args=_connect_args)

if DATABASE_URL.startswith("sqlite"):
    # WAL (Write-Ahead Logging) mode lets readers and a writer proceed
    # concurrently instead of SQLite's default behavior where any write
    # blocks all other connections. This doesn't make SQLite a real
    # multi-user database (still one writer at a time, still one file,
    # still no access control), but it meaningfully reduces the "one
    # student's save locks out everyone else" problem for a handful of
    # concurrent users -- a real improvement short of migrating to
    # Postgres, not a substitute for it at real scale.
    from sqlalchemy import event

    @event.listens_for(engine, "connect")
    def _enable_wal(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA busy_timeout=5000")  # wait up to 5s on a lock instead of failing immediately
        cursor.close()

metadata = MetaData()

soil_entries = Table(
    "soil_entries", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("student_name", Text),
    Column("college", Text),
    Column("farmer_name", Text),
    Column("district", Text),
    Column("village", Text),
    Column("ph", Float),
    Column("carbon", Float),
    Column("moisture", Float),
    Column("chemical", Text),
)

student_data = Table(
    "student_data", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("student_name", Text),
    Column("student_id", Text),
    Column("college", Text),
    Column("department", Text),
    Column("created_at", DateTime(timezone=True), server_default=func.now()),
)

farmer_data = Table(
    "farmer_data", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("farmer_name", Text),
    Column("district", Text),
    Column("village", Text),
    Column("phone", Text),
    Column("farm_size", Float),
    Column("soil_testing_done", Text),
    Column("soil_health_card", Text),
    Column("soil_health_card_id", Text),
    Column("latitude", Float),
    Column("longitude", Float),
    # Consent / data-terms capture -- required before any data collected for
    # this farmer is used to generate a recommendation. See advisory gap:
    # "Regulation and data privacy are absent."
    Column("consent_given", Text),
    Column("consent_date", Text),
    Column("created_at", DateTime(timezone=True), server_default=func.now()),
)

ml_results = Table(
    "ml_results", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("student_name", Text),
    Column("farmer_name", Text),
    Column("next_crop", Text),
    Column("recommended_crop", Text),
    Column("recommended_chemical", Text),
    Column("residue_status", Text),
    Column("residue_percentage", Float),
    Column("predicted_yield", Float),
    Column("confidence_score", Float),
    Column("limiting_factor", Text),
    Column("created_at", DateTime(timezone=True), server_default=func.now()),
)

moa_results = Table(
    "moa_results", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("student_name", Text),
    Column("farmer_name", Text),
    Column("next_crop", Text),
    Column("initial_dose", Float),
    Column("optimized_dose", Float),
    Column("reduction_percentage", Float),
    Column("farm_area", Float),
    Column("total_initial", Float),
    Column("total_optimized", Float),
    # Which recommendation cycle this is for the farmer (1 = first-ever
    # optimization, 2 = next crop after that, ...). This is what makes the
    # system a recurring, per-farmer advisory tool rather than a one-time
    # calculator -- each new crop/season for the same farmer_name gets its
    # own row here instead of overwriting the last one. See
    # fetch_farmer_season_summary() below for the cross-season view.
    Column("season_number", Integer),
    Column("optimizer_method", Text),  # e.g. "MOA", "PSO" -- which candidate method was used
    Column("created_at", DateTime(timezone=True), server_default=func.now()),
)

agronomist_reviews = Table(
    "agronomist_reviews", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("student_name", Text),
    Column("farmer_name", Text),
    Column("next_crop", Text),
    Column("confidence_score", Float),
    Column("rule_flags", Text),          # comma-separated agronomic rule-check flags, if any
    Column("requires_review", Text),     # "Yes" / "No" -- set by the rule/confidence gate
    Column("reviewer_name", Text),
    Column("decision", Text),            # Approved / Modified / Rejected
    Column("notes", Text),
    Column("created_at", DateTime(timezone=True), server_default=func.now()),
)

harvest_outcomes = Table(
    "harvest_outcomes", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("student_name", Text),
    Column("farmer_name", Text),
    Column("crop", Text),
    Column("predicted_yield_kg_per_acre", Float),
    Column("actual_yield_kg_per_acre", Float),
    Column("harvest_date", Text),
    Column("notes", Text),
    Column("created_at", DateTime(timezone=True), server_default=func.now()),
)

soil_submissions = Table(
    "soil_submissions", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    # Who submitted it and which farmer it's for -- links back to
    # student_data / farmer_data by name (this project doesn't have
    # foreign keys between those tables, so name is the join key
    # everywhere, same as the rest of this file).
    Column("student_name", Text),
    Column("farmer_name", Text),
    Column("district", Text),
    Column("village", Text),
    Column("soil_type", Text),
    Column("ph", Float),
    Column("organic_carbon", Float),
    Column("moisture", Float),
    Column("nitrogen", Float),
    Column("phosphorus", Float),
    Column("potassium", Float),
    Column("rainfall", Float),
    Column("temperature", Float),
    Column("electrical_conductivity", Float),
    Column("chemical_usage_mode", Text),
    Column("previous_chemical", Text),
    Column("previous_chem_amount", Float),
    Column("days_since_application", Integer),
    Column("soil_chemical_type", Text),
    Column("soil_chemical_name", Text),
    Column("soil_chemical_dose", Float),
    Column("soil_chemical_note", Text),
    # "pending" -> submitted by a student, not yet run through ML/MOA by
    # the admin. "processed" -> admin loaded it, ran the pipeline, and
    # saved ml_results/moa_results for it. This is what makes the
    # public (student) app and the admin (laptop-only) app work as a
    # queue instead of needing to share a live session.
    Column("status", Text),
    Column("created_at", DateTime(timezone=True), server_default=func.now()),
    Column("processed_at", DateTime(timezone=True)),
)

TABLES = {
    "soil_entries": soil_entries,
    "student_data": student_data,
    "farmer_data": farmer_data,
    "ml_results": ml_results,
    "moa_results": moa_results,
    "harvest_outcomes": harvest_outcomes,
    "agronomist_reviews": agronomist_reviews,
    "soil_submissions": soil_submissions,
}


def create_db():
    """
    Creates any tables that don't exist yet. Safe to call more than once:
    metadata.create_all() already skips tables it finds present, but on
    some setups (e.g. the app re-running create_db() on every Streamlit
    script rerun, especially over a synced/network drive) two connections
    can race and both attempt "CREATE TABLE" on the same table -- the
    loser gets an "already exists" OperationalError even though nothing
    is actually wrong. That specific error is caught and ignored here
    instead of crashing the app; any other database error still raises.
    """
    try:
        metadata.create_all(engine)
    except OperationalError as e:
        if "already exists" not in str(e).lower():
            raise
    _migrate_add_missing_columns()


def _migrate_add_missing_columns():
    """
    Lightweight migration for columns added after this database was first
    deployed (season_number, optimizer_method on moa_results).
    metadata.create_all() only creates tables that don't exist yet -- it
    won't add a new column to a table that's already there, so an
    existing agriculture.db from before this change needs an explicit
    ALTER TABLE. Safe to call every startup: each statement is wrapped so
    an "already exists" error is silently ignored.
    """
    migrations = [
        "ALTER TABLE moa_results ADD COLUMN season_number INTEGER",
        "ALTER TABLE moa_results ADD COLUMN optimizer_method TEXT",
    ]
    with engine.begin() as conn:
        for stmt in migrations:
            try:
                conn.exec_driver_sql(stmt)
            except Exception:
                pass  # column already exists -- nothing to do


def insert_soil_entry(student_name, college, farmer_name, district, village, ph, carbon, moisture, chemical):
    try:
        with engine.begin() as conn:
            conn.execute(soil_entries.insert().values(
                student_name=student_name, college=college, farmer_name=farmer_name,
                district=district, village=village, ph=ph, carbon=carbon,
                moisture=moisture, chemical=chemical,
            ))
        return True
    except Exception as e:
        print(f"Error inserting soil entry: {e}")
        return False


def insert_soil_submission(**fields) -> int | None:
    """
    Saves a student's Soil Information submission immediately (status
    'pending'), independent of whether the ML & MOA pipeline is ever run
    in that same browser session. This is the hand-off point between the
    public (student, phone/Render) app and the admin (laptop-only) app:
    the admin's Dashboard lists pending rows from here, loads one, runs
    the pipeline, and mark_submission_processed() closes it out.
    Returns the new row's id (needed later to mark it processed), or
    None on failure.
    """
    try:
        with engine.begin() as conn:
            result = conn.execute(soil_submissions.insert().values(status="pending", **fields))
            return result.inserted_primary_key[0]
    except Exception as e:
        print(f"Error inserting soil submission: {e}")
        return None


def fetch_pending_submissions() -> List[Dict]:
    with engine.connect() as conn:
        result = conn.execute(
            select(soil_submissions)
            .where(soil_submissions.c.status == "pending")
            .order_by(soil_submissions.c.id.asc())
        )
        return [dict(row._mapping) for row in result]


def mark_submission_processed(submission_id: int) -> bool:
    try:
        with engine.begin() as conn:
            conn.execute(
                soil_submissions.update()
                .where(soil_submissions.c.id == submission_id)
                .values(status="processed", processed_at=func.now())
            )
        return True
    except Exception as e:
        print(f"Error marking submission processed: {e}")
        return False


def fetch_farmer_by_name(farmer_name: str) -> Dict | None:
    """Most recent farmer_data row for this farmer_name, or None."""
    if not farmer_name:
        return None
    with engine.connect() as conn:
        result = conn.execute(
            select(farmer_data)
            .where(farmer_data.c.farmer_name == farmer_name)
            .order_by(farmer_data.c.id.desc())
            .limit(1)
        )
        row = result.fetchone()
        return dict(row._mapping) if row else None


def fetch_farmer_soil_carbon_history(farmer_name: str) -> List[float]:
    """
    Organic carbon readings recorded for this farmer's soil_entries, oldest
    first (ordered by row id, since soil_entries has no timestamp column).
    Used only by backend.biological_health.estimate_biological_activity()
    to detect a rising/falling organic-carbon trend across seasons -- not
    used anywhere else. Returns [] if the farmer has no entries yet.
    """
    try:
        with engine.begin() as conn:
            rows = conn.execute(
                select(soil_entries.c.carbon)
                .where(soil_entries.c.farmer_name == farmer_name)
                .order_by(soil_entries.c.id.asc())
            ).fetchall()
        return [row[0] for row in rows if row[0] is not None]
    except Exception as e:
        print(f"Error fetching soil carbon history: {e}")
        return []


def insert_student_data(student_name, student_id, college, department):
    try:
        with engine.begin() as conn:
            conn.execute(student_data.insert().values(
                student_name=student_name, student_id=student_id,
                college=college, department=department,
            ))
        return True
    except Exception as e:
        print(f"Error inserting student data: {e}")
        return False


def insert_farmer_data(
    farmer_name, district, village, phone, farm_size,
    soil_testing_done, soil_health_card, soil_health_card_id, latitude, longitude,
    consent_given="No", consent_date="",
):
    try:
        with engine.begin() as conn:
            conn.execute(farmer_data.insert().values(
                farmer_name=farmer_name, district=district, village=village, phone=phone,
                farm_size=farm_size, soil_testing_done=soil_testing_done,
                soil_health_card=soil_health_card, soil_health_card_id=soil_health_card_id,
                latitude=latitude, longitude=longitude,
                consent_given=consent_given, consent_date=consent_date,
            ))
        return True
    except Exception as e:
        print(f"Error inserting farmer data: {e}")
        return False


def insert_ml_result(
    student_name, farmer_name, next_crop, recommended_crop, recommended_chemical,
    residue_status, residue_percentage, predicted_yield, confidence_score, limiting_factor,
):
    try:
        with engine.begin() as conn:
            conn.execute(ml_results.insert().values(
                student_name=student_name, farmer_name=farmer_name, next_crop=next_crop,
                recommended_crop=recommended_crop, recommended_chemical=recommended_chemical,
                residue_status=residue_status, residue_percentage=residue_percentage,
                predicted_yield=predicted_yield, confidence_score=confidence_score,
                limiting_factor=limiting_factor,
            ))
        return True
    except Exception as e:
        print(f"Error inserting ML result: {e}")
        return False


def insert_moa_result(
    student_name, farmer_name, next_crop, initial_dose, optimized_dose,
    reduction_percentage, farm_area, total_initial, total_optimized,
    season_number=None, optimizer_method="MOA",
):
    try:
        with engine.begin() as conn:
            conn.execute(moa_results.insert().values(
                student_name=student_name, farmer_name=farmer_name, next_crop=next_crop,
                initial_dose=initial_dose, optimized_dose=optimized_dose,
                reduction_percentage=reduction_percentage, farm_area=farm_area,
                total_initial=total_initial, total_optimized=total_optimized,
                season_number=season_number, optimizer_method=optimizer_method,
            ))
        return True
    except Exception as e:
        print(f"Error inserting MOA result: {e}")
        return False


def insert_agronomist_review(
    student_name, farmer_name, next_crop, confidence_score, rule_flags,
    requires_review, reviewer_name, decision, notes="",
):
    """
    Records the human validation step between an ML/MOA recommendation and
    it being acted on in the field. `requires_review` is set upstream by
    the confidence/rule-check gate (see backend/model.py); `decision` is
    what the agronomist/AEO actually did with it.
    """
    try:
        with engine.begin() as conn:
            conn.execute(agronomist_reviews.insert().values(
                student_name=student_name, farmer_name=farmer_name, next_crop=next_crop,
                confidence_score=confidence_score, rule_flags=rule_flags,
                requires_review=requires_review, reviewer_name=reviewer_name,
                decision=decision, notes=notes,
            ))
        return True
    except Exception as e:
        print(f"Error inserting agronomist review: {e}")
        return False


def fetch_farmer_review_history(farmer_name: str) -> List[Dict]:
    if not farmer_name:
        return []
    with engine.connect() as conn:
        result = conn.execute(
            select(agronomist_reviews)
            .where(agronomist_reviews.c.farmer_name == farmer_name)
            .order_by(agronomist_reviews.c.id.desc())
        )
        return [dict(row._mapping) for row in result]


def fetch_all(table_name: str) -> List[Dict]:
    table = TABLES.get(table_name)
    if table is None:
        return []
    with engine.connect() as conn:
        result = conn.execute(select(table).order_by(table.c.id.desc()))
        return [dict(row._mapping) for row in result]


def fetch_farmer_moa_history(farmer_name: str) -> List[Dict]:
    if not farmer_name:
        return []
    with engine.connect() as conn:
        result = conn.execute(
            select(moa_results)
            .where(moa_results.c.farmer_name == farmer_name)
            .order_by(moa_results.c.id.desc())
        )
        return [dict(row._mapping) for row in result]


def insert_harvest_outcome(
    student_name, farmer_name, crop, predicted_yield_kg_per_acre,
    actual_yield_kg_per_acre, harvest_date, notes="",
):
    """
    Record what actually happened at harvest, tied back to an earlier
    prediction. This is the concrete first step toward validating the
    yield model against real Telangana outcomes instead of only synthetic
    training data -- each row here is one real (predicted, actual) pair.
    """
    try:
        with engine.begin() as conn:
            conn.execute(harvest_outcomes.insert().values(
                student_name=student_name, farmer_name=farmer_name, crop=crop,
                predicted_yield_kg_per_acre=predicted_yield_kg_per_acre,
                actual_yield_kg_per_acre=actual_yield_kg_per_acre,
                harvest_date=harvest_date, notes=notes,
            ))
        return True
    except Exception as e:
        print(f"Error inserting harvest outcome: {e}")
        return False


def fetch_farmer_season_summary(farmer_name: str) -> Dict:
    """
    Cross-season view for ONE farmer: every MOA/optimizer cycle recorded
    for them, oldest first, with cumulative chemical and cost savings
    across all seasons -- not just the latest one-off calculation. This
    is what lets the app be a recurring per-farmer advisory tool: the
    same farmer_name is looked up and updated again for their next crop,
    rather than each visit starting from a blank slate.
    """
    if not farmer_name:
        return {"farmer_name": farmer_name, "seasons": [], "cumulative_kg_saved": 0.0, "season_count": 0}

    with engine.connect() as conn:
        result = conn.execute(
            select(moa_results)
            .where(moa_results.c.farmer_name == farmer_name)
            .order_by(moa_results.c.id.asc())
        )
        rows = [dict(row._mapping) for row in result]

    cumulative_kg_saved = 0.0
    seasons = []
    for i, row in enumerate(rows, start=1):
        kg_saved = max(0.0, (row.get("total_initial") or 0.0) - (row.get("total_optimized") or 0.0))
        cumulative_kg_saved += kg_saved
        seasons.append({
            "season_number": row.get("season_number") or i,
            "next_crop": row.get("next_crop"),
            "reduction_percentage": row.get("reduction_percentage"),
            "kg_saved_this_season": round(kg_saved, 2),
            "cumulative_kg_saved": round(cumulative_kg_saved, 2),
            "optimizer_method": row.get("optimizer_method") or "MOA",
            "created_at": row.get("created_at"),
        })

    return {
        "farmer_name": farmer_name,
        "seasons": seasons,
        "cumulative_kg_saved": round(cumulative_kg_saved, 2),
        "season_count": len(seasons),
    }


def fetch_harvest_accuracy_summary() -> Dict:
    """
    Compare every recorded (predicted, actual) pair to see how the model
    is actually doing in the field. Returns None-safe defaults if there's
    no data yet, so callers can show "not enough data" instead of crashing.
    """
    rows = fetch_all("harvest_outcomes")
    valid = [
        r for r in rows
        if r.get("predicted_yield_kg_per_acre") and r.get("actual_yield_kg_per_acre")
    ]
    if not valid:
        return {"count": 0, "mae_kg_per_acre": None, "mean_error_pct": None, "rows": rows}

    errors = [abs(r["predicted_yield_kg_per_acre"] - r["actual_yield_kg_per_acre"]) for r in valid]
    pct_errors = [
        (abs(r["predicted_yield_kg_per_acre"] - r["actual_yield_kg_per_acre"]) / r["actual_yield_kg_per_acre"]) * 100
        for r in valid if r["actual_yield_kg_per_acre"] > 0
    ]
    return {
        "count": len(valid),
        "mae_kg_per_acre": round(sum(errors) / len(errors), 1),
        "mean_error_pct": round(sum(pct_errors) / len(pct_errors), 1) if pct_errors else None,
        "rows": rows,
    }
