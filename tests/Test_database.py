"""
test_database.py â€” ØªØ³Øª Ù„Ø§ÛŒÙ‡â€ŒÛŒ Ø¯ÛŒØªØ§Ø¨ÛŒØ³ Ù¾Ø±ÙˆÚ˜Ù‡ CXRPredictor

Ø§Ø² SQLite in-memory Ø§Ø³ØªÙØ§Ø¯Ù‡ Ù…ÛŒÚ©Ù†Ù‡ â†’ Ù†ÛŒØ§Ø²ÛŒ Ø¨Ù‡ PostgreSQL ÙˆØ§Ù‚Ø¹ÛŒ Ù†ÛŒØ³Øª.
Ù‡Ø± ØªØ³Øª Ø¯Ø± ÛŒÙ‡ transaction Ø¬Ø¯Ø§Ú¯Ø§Ù†Ù‡ Ø§Ø¬Ø±Ø§ Ù…ÛŒØ´Ù‡ Ùˆ rollback Ù…ÛŒØ´Ù‡.
"""
import pytest
from unittest.mock import patch, MagicMock
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker


# ---- Engine Ùˆ Base Ø±Ùˆ Ø¨Ø§ patch Ù„ÙˆØ¯ Ù…ÛŒÚ©Ù†ÛŒÙ… ----
# Ú†ÙˆÙ† database.py Ø¯Ø± import Ø²Ù…Ø§Ù† create_engine Ø±Ùˆ ØµØ¯Ø§ Ù…ÛŒØ²Ù†Ù‡ØŒ
# Ø§ÙˆÙ„ env Ù…ØªØºÛŒØ±Ù‡Ø§ Ø±Ùˆ set Ù…ÛŒÚ©Ù†ÛŒÙ… ØªØ§ URL Ø¯Ø±Ø³Øª Ø³Ø§Ø®ØªÙ‡ Ø¨Ø´Ù‡

import os
os.environ.setdefault("DB_USERNAME", "testuser")
os.environ.setdefault("DB_PASSWORD", "testpass")
os.environ.setdefault("DB_HOST", "localhost")
os.environ.setdefault("DB_PORT", "5432")
os.environ.setdefault("DB_NAME", "testdb")

from app.database import Base, get_db
from app.models import PredictionRecord


# ---- Fixtures ----

@pytest.fixture(scope="module")
def sqlite_engine():
    """
    ÛŒÚ© SQLite in-memory engine Ø¨Ø±Ø§ÛŒ Ú©Ù„ Ù…Ø§Ú˜ÙˆÙ„ ØªØ³Øª.
    Ø¬Ø¯ÙˆÙ„â€ŒÙ‡Ø§ ÛŒÙ‡ Ø¨Ø§Ø± Ø³Ø§Ø®ØªÙ‡ Ù…ÛŒØ´Ù† Ùˆ ØªØ§ Ø¢Ø®Ø± Ù‡Ù…Ù‡â€ŒÛŒ ØªØ³Øªâ€ŒÙ‡Ø§ Ø¨Ø§Ù‚ÛŒ Ù…ÛŒÙ…ÙˆÙ†Ù†.
    """
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    yield engine
    engine.dispose()


@pytest.fixture
def db_session(sqlite_engine):
    """
    Ø¨Ø±Ø§ÛŒ Ù‡Ø± ØªØ³Øª ÛŒÙ‡ session Ø¬Ø¯Ø§Ú¯Ø§Ù†Ù‡ Ø¨Ø§ rollback Ø®ÙˆØ¯Ú©Ø§Ø± Ù…ÛŒØ¯Ù‡.
    ØªØºÛŒÛŒØ±Ø§Øª ØªØ³Øªâ€ŒÙ‡Ø§ÛŒ Ù…Ø®ØªÙ„Ù Ø¨Ø§ Ù‡Ù… ØªØ¯Ø§Ø®Ù„ Ù†Ø¯Ø§Ø±Ù†.
    """
    connection = sqlite_engine.connect()
    transaction = connection.begin()
    TestSession = sessionmaker(bind=connection)
    session = TestSession()

    yield session

    session.close()
    transaction.rollback()
    connection.close()


# ---- Helper ----

def _make_record(**overrides):
    defaults = dict(
        original_filename="xray.png",
        stored_filename="abc123def456.png",
        file_path="images/abc123def456.png",
        content_type="image/png",
        file_size_bytes=20480,
        model_name="densenet121-res224-all",
        prediction_count=15,
        max_prediction_score=0.8901,
        predictions={"No Finding": 0.8901, "Pneumonia": 0.1099},
    )
    defaults.update(overrides)
    return PredictionRecord(**defaults)


# ---- Tests: Ø³Ø§Ø®ØªØ§Ø± Ø¬Ø¯ÙˆÙ„ ----

class TestTableStructure:

    def test_prediction_records_table_exists(self, sqlite_engine):
        tables = inspect(sqlite_engine).get_table_names()
        assert "prediction_records" in tables

    def test_table_has_id_column(self, sqlite_engine):
        cols = [c["name"] for c in inspect(sqlite_engine).get_columns("prediction_records")]
        assert "id" in cols

    def test_table_has_original_filename_column(self, sqlite_engine):
        cols = [c["name"] for c in inspect(sqlite_engine).get_columns("prediction_records")]
        assert "original_filename" in cols

    def test_table_has_stored_filename_column(self, sqlite_engine):
        cols = [c["name"] for c in inspect(sqlite_engine).get_columns("prediction_records")]
        assert "stored_filename" in cols

    def test_table_has_predictions_column(self, sqlite_engine):
        cols = [c["name"] for c in inspect(sqlite_engine).get_columns("prediction_records")]
        assert "predictions" in cols

    def test_table_has_model_name_column(self, sqlite_engine):
        cols = [c["name"] for c in inspect(sqlite_engine).get_columns("prediction_records")]
        assert "model_name" in cols

    def test_table_has_max_prediction_score_column(self, sqlite_engine):
        cols = [c["name"] for c in inspect(sqlite_engine).get_columns("prediction_records")]
        assert "max_prediction_score" in cols

    def test_table_has_created_at_column(self, sqlite_engine):
        cols = [c["name"] for c in inspect(sqlite_engine).get_columns("prediction_records")]
        assert "created_at" in cols


# ---- Tests: Ø¹Ù…Ù„ÛŒØ§Øª CRUD ----

class TestCRUDOperations:

    def test_insert_record_assigns_id(self, db_session):
        record = _make_record()
        db_session.add(record)
        db_session.flush()
        assert record.id is not None
        assert isinstance(record.id, int)

    def test_inserted_record_has_correct_original_filename(self, db_session):
        record = _make_record(original_filename="patient_001.png")
        db_session.add(record)
        db_session.flush()
        fetched = db_session.get(PredictionRecord, record.id)
        assert fetched.original_filename == "patient_001.png"

    def test_inserted_record_has_correct_model_name(self, db_session):
        record = _make_record()
        db_session.add(record)
        db_session.flush()
        fetched = db_session.get(PredictionRecord, record.id)
        assert fetched.model_name == "densenet121-res224-all"

    def test_inserted_record_stores_predictions_dict(self, db_session):
        preds = {"Effusion": 0.72, "Pneumonia": 0.28}
        record = _make_record(predictions=preds, stored_filename="unique1.png")
        db_session.add(record)
        db_session.flush()
        fetched = db_session.get(PredictionRecord, record.id)
        assert fetched.predictions == preds

    def test_inserted_record_has_correct_file_size(self, db_session):
        record = _make_record(file_size_bytes=98765, stored_filename="unique2.png")
        db_session.add(record)
        db_session.flush()
        fetched = db_session.get(PredictionRecord, record.id)
        assert fetched.file_size_bytes == 98765

    def test_max_prediction_score_stores_float(self, db_session):
        record = _make_record(max_prediction_score=0.9512, stored_filename="unique3.png")
        db_session.add(record)
        db_session.flush()
        fetched = db_session.get(PredictionRecord, record.id)
        assert abs(fetched.max_prediction_score - 0.9512) < 0.001

    def test_max_prediction_score_can_be_null(self, db_session):
        record = _make_record(max_prediction_score=None, stored_filename="unique4.png")
        db_session.add(record)
        db_session.flush()
        fetched = db_session.get(PredictionRecord, record.id)
        assert fetched.max_prediction_score is None

    def test_delete_record_removes_it(self, db_session):
        record = _make_record(stored_filename="delete_me.png")
        db_session.add(record)
        db_session.flush()
        record_id = record.id
        db_session.delete(record)
        db_session.flush()
        assert db_session.get(PredictionRecord, record_id) is None

    def test_two_records_get_different_ids(self, db_session):
        r1 = _make_record(stored_filename="file_one.png", original_filename="one.png")
        r2 = _make_record(stored_filename="file_two.png", original_filename="two.png")
        db_session.add_all([r1, r2])
        db_session.flush()
        assert r1.id != r2.id

    def test_prediction_count_is_stored_correctly(self, db_session):
        record = _make_record(prediction_count=15, stored_filename="unique5.png")
        db_session.add(record)
        db_session.flush()
        fetched = db_session.get(PredictionRecord, record.id)
        assert fetched.prediction_count == 15


# ---- Tests: get_db dependency ----

class TestGetDbDependency:

    def test_get_db_yields_a_session(self):
        mock_session = MagicMock()
        with patch("app.database.SessionLocal", return_value=mock_session):
            gen = get_db()
            session = next(gen)
        assert session is mock_session

    def test_get_db_closes_session_after_yield(self):
        mock_session = MagicMock()
        with patch("app.database.SessionLocal", return_value=mock_session):
            gen = get_db()
            next(gen)
            try:
                next(gen)
            except StopIteration:
                pass
        mock_session.close.assert_called_once()

    def test_get_db_closes_session_even_on_exception(self):
        """Ø­ØªÛŒ Ø§Ú¯Ù‡ exception Ø¨ÛŒØ§Ø¯ØŒ session Ø¨Ø§ÛŒØ¯ close Ø¨Ø´Ù‡."""
        mock_session = MagicMock()
        with patch("app.database.SessionLocal", return_value=mock_session):
            gen = get_db()
            next(gen)
            gen.throw(RuntimeError("Test error"))
        mock_session.close.assert_called_once()


# ---- Tests: DATABASE_URL ----

class TestDatabaseUrl:

    def test_database_url_starts_with_postgresql_psycopg(self):
        from app.database import DATABASE_URL
        assert DATABASE_URL.startswith("postgresql+psycopg://")

    def test_database_url_contains_host(self):
        from app.database import DATABASE_URL
        assert "localhost" in DATABASE_URL or "@" in DATABASE_URL

    def test_database_url_contains_port(self):
        from app.database import DATABASE_URL
        assert "5432" in DATABASE_URL
