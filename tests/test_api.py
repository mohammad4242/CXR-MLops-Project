import io
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


# ---- Fixtures ----

@pytest.fixture
def mock_db():
    """Return a MagicMock that behaves like a SQLAlchemy Session."""
    db = MagicMock()
    db.add = MagicMock()
    db.commit = MagicMock()
    db.rollback = MagicMock()
    db.refresh = MagicMock()
    return db


@pytest.fixture
def mock_predictor():
    """Return a MagicMock that behaves like CXRPredictor."""
    predictor = MagicMock()
    predictor.predict.return_value = {
        "Atelectasis": 0.1234,
        "Cardiomegaly": 0.0567,
        "Effusion": 0.3456,
        "Pneumonia": 0.2345,
        "No Finding": 0.8901,
    }
    return predictor


@pytest.fixture
def client(mock_db, mock_predictor):
    """
    Build a TestClient with all heavy dependencies (DB + ML model) mocked out.
    This way the tests run without PostgreSQL or GPU.
    """
    with (
        patch("app.database.engine"),
        patch("app.database.Base.metadata.create_all"),
        patch("app.inference.CXRPredictor", return_value=mock_predictor),
    ):
        from app.main import app, ml_models, get_db

        # Inject the mock predictor into the running app state
        ml_models["predictor"] = mock_predictor

        # Override the DB dependency so no real PostgreSQL is needed
        app.dependency_overrides[get_db] = lambda: mock_db

        with TestClient(app, raise_server_exceptions=True) as c:
            yield c

        app.dependency_overrides.clear()


# ---- Helper ----

def _make_image_bytes() -> bytes:
    """Return a minimal valid PNG as bytes (1Ã—1 white pixel)."""
    return (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
        b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00"
        b"\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18"
        b"\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
    )


# ---- Tests: /health ----

class TestHealthEndpoint:

    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_has_required_keys(self, client):
        data = client.get("/health").json()
        assert "status" in data
        assert "model_loaded" in data
        assert "model_name" in data
        assert "uptime_seconds" in data

    def test_health_status_is_healthy(self, client):
        data = client.get("/health").json()
        assert data["status"] == "healthy"

    def test_health_model_is_loaded(self, client):
        data = client.get("/health").json()
        assert data["model_loaded"] is True

    def test_health_model_name_is_correct(self, client):
        data = client.get("/health").json()
        assert data["model_name"] == "densenet121-res224-all"


# ---- Tests: /predict ----

class TestPredictEndpoint:

    def test_predict_returns_200_with_valid_image(self, client):
        image_bytes = _make_image_bytes()
        response = client.post(
            "/predict",
            files={"file": ("chest.png", io.BytesIO(image_bytes), "image/png")},
        )
        assert response.status_code == 200

    def test_predict_response_has_predictions_key(self, client):
        image_bytes = _make_image_bytes()
        data = client.post(
            "/predict",
            files={"file": ("chest.png", io.BytesIO(image_bytes), "image/png")},
        ).json()
        assert "predictions" in data

    def test_predict_response_has_success_true(self, client):
        image_bytes = _make_image_bytes()
        data = client.post(
            "/predict",
            files={"file": ("chest.png", io.BytesIO(image_bytes), "image/png")},
        ).json()
        assert data["success"] is True

    def test_predict_response_has_id(self, client):
        image_bytes = _make_image_bytes()
        data = client.post(
            "/predict",
            files={"file": ("chest.png", io.BytesIO(image_bytes), "image/png")},
        ).json()
        assert "id" in data

    def test_predict_response_has_filename(self, client):
        image_bytes = _make_image_bytes()
        data = client.post(
            "/predict",
            files={"file": ("chest.png", io.BytesIO(image_bytes), "image/png")},
        ).json()
        assert data["filename"] == "chest.png"

    def test_predict_rejects_non_image_file(self, client):
        response = client.post(
            "/predict",
            files={"file": ("report.pdf", io.BytesIO(b"%PDF-1.4"), "application/pdf")},
        )
        assert response.status_code == 400

    def test_predict_error_detail_for_non_image(self, client):
        response = client.post(
            "/predict",
            files={"file": ("report.pdf", io.BytesIO(b"%PDF-1.4"), "application/pdf")},
        )
        assert "image" in response.json()["detail"].lower()

    def test_predict_calls_model_predict_once(self, client, mock_predictor):
        image_bytes = _make_image_bytes()
        client.post(
            "/predict",
            files={"file": ("chest.png", io.BytesIO(image_bytes), "image/png")},
        )
        mock_predictor.predict.assert_called_once()

    def test_predict_calls_db_commit(self, client, mock_db):
        image_bytes = _make_image_bytes()
        client.post(
            "/predict",
            files={"file": ("chest.png", io.BytesIO(image_bytes), "image/png")},
        )
        mock_db.commit.assert_called_once()

    def test_predict_returns_prediction_scores_as_floats(self, client):
        image_bytes = _make_image_bytes()
        data = client.post(
            "/predict",
            files={"file": ("chest.png", io.BytesIO(image_bytes), "image/png")},
        ).json()
        for score in data["predictions"].values():
            assert isinstance(score, float)

    def test_predict_rollback_on_model_failure(self, client, mock_predictor, mock_db):
        mock_predictor.predict.side_effect = RuntimeError("Model crashed")
        image_bytes = _make_image_bytes()
        response = client.post(
            "/predict",
            files={"file": ("chest.png", io.BytesIO(image_bytes), "image/png")},
        )
        assert response.status_code == 500
        mock_db.rollback.assert_called_once()
