import pytest
from unittest.mock import patch, MagicMock, PropertyMock
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker

from app.database import Base, get_db


# ---- In-memory SQLite engine for isolated tests ----

@pytest.fixture(scope="module")
def sqlite_engine():
    """
    Create a fresh in-memory SQLite database for the whole test module.
    SQLite does NOT need PostgreSQL to be running.
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
    Provide a clean, auto-rollback database session for every single test.
    Changes made inside a test are never committed to the shared engine.
    """
    connection = sqlite_engine.connect()
    transaction = connection.begin()
    TestingSessionLocal = sessionmaker(bind=connection)
    session = TestingSessionLocal()

    yield session

    session.close()
    transaction.rollback()
    connection.close()


# ---- Helpers ----

def _make_record(**kwargs):
    """Build a PredictionRecord with sensible defaults."""
    from app.models import PredictionRecord

    defaults = dict(
        original_filename="test.png",
        stored_filename="abc123.png",
        file_path="images/abc123.png",
        content_type="image/png",
        file_size_bytes=1024,
        model_name="densenet121-res224-all",
        prediction_count=5,
        max_prediction_score=0.89,
        predictions={"No Finding": 0.89, "Pneumonia": 0.11},
    )
    defaults.update(kwargs)
    return PredictionRecord(**defaults)


# ---- Tests: table creation ----

class TestTableCreation:

    def test_prediction_records_table_exists(self, sqlite_engine):
        inspector = inspect(sqlite_engine)
        assert "prediction_records" in inspector.get_table_names()

    def test_table_has_id_column(self, sqlite_engine):
        inspector = inspect(sqlite_engine)
        columns = [c["name"] for c in inspector.get_columns("prediction_records")]
        assert "id" in columns

    def test_table_has_predictions_column(self, sqlite_engine):
        inspector = inspect(sqlite_engine)
        columns = [c["name"] for c in inspector.get_columns("prediction_records")]
        assert "predictions" in columns

    def test_table_has_created_at_column(self, sqlite_engine):
        inspector = inspect(sqlite_engine)
        columns = [c["name"] for c in inspector.get_columns("prediction_records")]
        assert "created_at" in columns


# ---- Tests: CRUD operations ----

class TestCRUDOperations:

    def test_insert_record_succeeds(self, db_session):
        record = _make_record()
        db_session.add(record)
        db_session.flush()
        assert record.id is not None

    def test_inserted_record_has_correct_filename(self, db_session):
        record = _make_record(original_filename="xray.jpg")
        db_session.add(record)
        db_session.flush()
        fetched = db_session.get(type(record), record.id)
        assert fetched.original_filename == "xray.jpg"

    def test_inserted_record_has_correct_model_name(self, db_session):
        record = _make_record()
        db_session.add(record)
        db_session.flush()
        fetched = db_session.get(type(record), record.id)
        assert fetched.model_name == "densenet121-res224-all"

    def test_inserted_record_stores_predictions_dict(self, db_session):
        preds = {"Effusion": 0.72, "Pneumonia": 0.28}
        record = _make_record(predictions=preds)
        db_session.add(record)
        db_session.flush()
        fetched = db_session.get(type(record), record.id)
        assert fetched.predictions == preds

    def test_max_prediction_score_nullable(self, db_session):
        record = _make_record(max_prediction_score=None)
        db_session.add(record)
        db_session.flush()
        fetched = db_session.get(type(record), record.id)
        assert fetched.max_prediction_score is None

    def test_delete_record(self, db_session):
        from app.models import PredictionRecord

        record = _make_record(stored_filename="delete_me.png")
        db_session.add(record)
        db_session.flush()
        record_id = record.id
        db_session.delete(record)
        db_session.flush()
        assert db_session.get(PredictionRecord, record_id) is None

    def test_multiple_records_are_stored_independently(self, db_session):
        r1 = _make_record(stored_filename="file_one.png", original_filename="one.png")
        r2 = _make_record(stored_filename="file_two.png", original_filename="two.png")
        db_session.add_all([r1, r2])
        db_session.flush()
        assert r1.id != r2.id


# ---- Tests: get_db dependency ----

class TestGetDbDependency:

    def test_get_db_yields_a_session(self):
        """
        get_db() must yield exactly one session object and then close it.
        We mock SessionLocal so no real DB connection is needed.
        """
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


# ---- Tests: DATABASE_URL format ----

class TestDatabaseUrl:

    def test_database_url_uses_postgresql_driver(self):
        from app.database import DATABASE_URL

        assert DATABASE_URL.startswith("postgresql+psycopg://")

    def test_database_url_contains_db_name(self):
        from app.database import DATABASE_URL

        assert "cxr" in DATABASE_URL or DATABASE_URL.split("/")[-1] != ""
