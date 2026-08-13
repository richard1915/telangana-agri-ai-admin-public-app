"""
Shared pytest configuration. Critically, this sets DATABASE_URL to a
temporary SQLite file BEFORE any test imports backend.database, so tests
never touch the real agriculture.db and are safe to run repeatedly.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_TEST_DB_FD, _TEST_DB_PATH = tempfile.mkstemp(suffix=".db")
os.close(_TEST_DB_FD)
os.environ["DATABASE_URL"] = f"sqlite:///{_TEST_DB_PATH}"

import pytest


@pytest.fixture(scope="session", autouse=True)
def _setup_test_db():
    from backend.database import create_db
    create_db()
    yield
    try:
        os.remove(_TEST_DB_PATH)
    except OSError:
        pass
