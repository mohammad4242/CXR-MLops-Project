"""
test_api.py â€” ØªØ³Øª endpoint Ù‡Ø§ÛŒ FastAPI Ù¾Ø±ÙˆÚ˜Ù‡ CXRPredictor

Ù‡ÛŒÚ† PostgreSQL ÛŒØ§ Ù…Ø¯Ù„ ML ÙˆØ§Ù‚Ø¹ÛŒ Ù„Ø§Ø²Ù… Ù†ÛŒØ³Øª.
ØªÙ…Ø§Ù… dependency Ù‡Ø§ Ú©Ø§Ù…Ù„Ø§Ù‹ mock Ø´Ø¯Ù†.
"""
import io
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


# ---- Helpers ----

def _make_image_bytes() -> bytes:
    """ÛŒÚ© PNG Ù…Ø¹ØªØ¨Ø± ÛŒÚ©â€ŒÙ¾ÛŒÚ©Ø³Ù„ÛŒ Ø¨Ø±Ù…ÛŒÚ¯Ø±Ø¯ÙˆÙ†Ù‡."""
    return (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
        b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00"
        b"\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18"
        b"\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
    )


FAKE_PREDICTIONS = {
    "Atelectasis": 0.1234,
    "Cardiomegaly": 0.0567,
    "Effusion": 0.3456,
    "Pneumonia": 0.2345,
    "No Finding": 0.8901,
}


# ---- App + client setup ----
#
# Ø§Ø³ØªØ±Ø§ØªÚ˜ÛŒ:
#  Û±. Ù‚Ø¨Ù„ Ø§Ø² import appØŒ ØªÙ…Ø§Ù… dependency Ù‡Ø§ÛŒ Ø³Ù†Ú¯ÛŒÙ† Ø±Ùˆ patch Ù…ÛŒÚ©Ù†ÛŒÙ…
#  Û². Ø¨Ø¹Ø¯ Ø§Ø² Ø³Ø§Ø®Øª TestClient (Ú©Ù‡ lifespan Ø±Ùˆ Ø§Ø¬Ø±Ø§ Ù…ÛŒÚ©Ù†Ù‡)ØŒ
#     Ù…Ø³ØªÙ‚ÛŒÙ… ml_models["predictor"] Ø±Ùˆ Ø¨Ø§ mock Ø¬Ø§ÛŒÚ¯Ø²ÛŒÙ† Ù…ÛŒÚ©Ù†ÛŒÙ…
#  Û³. get_db Ø±Ùˆ Ø¨Ø§ dependency_overrides Ø¬Ø§ÛŒÚ¯Ø²ÛŒÙ† Ù…ÛŒÚ©Ù†ÛŒÙ…
#
# Ø§ÛŒÙ† Ø±ÙˆØ´ ØªØ¶Ù…ÛŒÙ† Ù…ÛŒÚ©Ù†Ù‡ Ú©Ù‡ app Ù‡Ù…ÛŒØ´Ù‡ Ø§Ø² Ù‡Ù…ÙˆÙ† mock_predictor Ø§Ø³ØªÙØ§Ø¯Ù‡ Ù…ÛŒÚ©Ù†Ù‡.

@pytest.fixture(scope="module")
def app_and_mocks():
    """
    ÛŒÚ© Ø¨Ø§Ø± app Ø±Ùˆ Ø¨Ø§ ØªÙ…Ø§Ù… patch Ù‡Ø§ Ù„ÙˆØ¯ Ù…ÛŒÚ©Ù†Ù‡.
    scope=module ÛŒØ¹Ù†ÛŒ Ø¨Ø±Ø§ÛŒ ØªÙ…Ø§Ù… ØªØ³Øªâ€ŒÙ‡Ø§ÛŒ Ø§ÛŒÙ† ÙØ§ÛŒÙ„ ÙÙ‚Ø· ÛŒÙ‡ Ø¨Ø§Ø± Ø§Ø¬Ø±Ø§ Ù…ÛŒØ´Ù‡.
    """
    mock_db = MagicMock()
    mock_db.add = MagicMock()
    mock_db.commit = MagicMock()
    mock_db.rollback = MagicMock()
    mock_db.refresh = MagicMock()

    mock_predictor = MagicMock()
    mock_predictor.predict.return_value = FAKE_PREDICTIONS.copy()

    with (
        patch("app.database.create_engine"),
        patch("app.database.Base.metadata.create_all"),
        patch("app.inference.CXRPredictor", return_value=mock_predictor),
        patch("app.main.Base.metadata.create_all"),
    ):
        from app.main import app, ml_models, get_db

        # Ù…Ø³ØªÙ‚ÛŒÙ… predictor Ø±Ùˆ replace Ù…ÛŒÚ©Ù†ÛŒÙ…
        ml_models["predictor"] = mock_predictor

        # DB dependency Ø±Ùˆ override Ù…ÛŒÚ©Ù†ÛŒÙ…
        app.dependency_overrides[get_db] = lambda: mock_db

        with TestClient(app, raise_server_exceptions=False) as client:
            yield client, mock_predictor, mock_db

        app.dependency_overrides.clear()


@pytest.fixture(autouse=True)
def reset_mocks(app_and_mocks):
    """
    Ù‚Ø¨Ù„ Ø§Ø² Ù‡Ø± ØªØ³ØªØŒ mock Ù‡Ø§ Ø±Ùˆ reset Ù…ÛŒÚ©Ù†ÛŒÙ… ØªØ§ call countÙ‡Ø§ ØªØ¯Ø§Ø®Ù„ Ù†Ø¯Ø§Ø´ØªÙ‡ Ø¨Ø§Ø´Ù†.
    """
    client, mock_predictor, mock_db = app_and_mocks
    mock_predictor.reset_mock()
    mock_predictor.predict.return_value = FAKE_PREDICTIONS.copy()
    mock_predictor.predict.side_effect = None
    mock_db.reset_mock()


@pytest.fixture
def client(app_and_mocks):
    return app_and_mocks[0]


@pytest.fixture
def mock_predictor(app_and_mocks):
    return app_and_mocks[1]


@pytest.fixture
def mock_db(app_and_mocks):
    return app_and_mocks[2]


# ---- Tests: GET /health ----

class TestHealthEndpoint:

    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_has_status_key(self, client):
        data = client.get("/health").json()
        assert "status" in data

    def test_health_response_has_model_loaded_key(self, client):
        data = client.get("/health").json()
        assert "model_loaded" in data

    def test_health_response_has_model_name_key(self, client):
        data = client.get("/health").json()
        assert "model_name" in data

    def test_health_response_has_uptime_seconds_key(self, client):
        data = client.get("/health").json()
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


# ---- Tests: POST /predict ----

class TestPredictEndpoint:

    def test_predict_returns_200_with_valid_image(self, client):
        response = client.post(
            "/predict",
            files={"file": ("chest.png", io.BytesIO(_make_image_bytes()), "image/png")},
        )
        assert response.status_code == 200

    def test_predict_response_has_predictions_key(self, client):
        data = client.post(
            "/predict",
            files={"file": ("chest.png", io.BytesIO(_make_image_bytes()), "image/png")},
        ).json()
        assert "predictions" in data

    def test_predict_response_success_is_true(self, client):
        data = client.post(
            "/predict",
            files={"file": ("chest.png", io.BytesIO(_make_image_bytes()), "image/png")},
        ).json()
        assert data["success"] is True

    def test_predict_response_has_id(self, client):
        data = client.post(
            "/predict",
            files={"file": ("chest.png", io.BytesIO(_make_image_bytes()), "image/png")},
        ).json()
        assert "id" in data

    def test_predict_response_filename_matches_upload(self, client):
        data = client.post(
            "/predict",
            files={"file": ("chest.png", io.BytesIO(_make_image_bytes()), "image/png")},
        ).json()
        assert data["filename"] == "chest.png"

    def test_predict_prediction_values_are_correct(self, client):
        data = client.post(
            "/predict",
            files={"file": ("chest.png", io.BytesIO(_make_image_bytes()), "image/png")},
        ).json()
        assert data["predictions"]["No Finding"] == 0.8901
        assert data["predictions"]["Pneumonia"] == 0.2345

    def test_predict_rejects_pdf_with_400(self, client):
        response = client.post(
            "/predict",
            files={"file": ("report.pdf", io.BytesIO(b"%PDF-1.4"), "application/pdf")},
        )
        assert response.status_code == 400

    def test_predict_error_detail_mentions_image(self, client):
        response = client.post(
            "/predict",
            files={"file": ("report.pdf", io.BytesIO(b"%PDF-1.4"), "application/pdf")},
        )
        assert "image" in response.json()["detail"].lower()

    def test_predict_calls_model_predict_once(self, client, mock_predictor):
        client.post(
            "/predict",
            files={"file": ("chest.png", io.BytesIO(_make_image_bytes()), "image/png")},
        )
        mock_predictor.predict.assert_called_once()

    def test_predict_calls_db_commit(self, client, mock_db):
        client.post(
            "/predict",
            files={"file": ("chest.png", io.BytesIO(_make_image_bytes()), "image/png")},
        )
        mock_db.commit.assert_called_once()

    def test_predict_returns_500_on_model_failure(self, client, mock_predictor):
        mock_predictor.predict.side_effect = RuntimeError("Model crashed")
        response = client.post(
            "/predict",
            files={"file": ("chest.png", io.BytesIO(_make_image_bytes()), "image/png")},
        )
        assert response.status_code == 500

    def test_predict_rollback_called_on_model_failure(self, client, mock_predictor, mock_db):
        mock_predictor.predict.side_effect = RuntimeError("Model crashed")
        client.post(
            "/predict",
            files={"file": ("chest.png", io.BytesIO(_make_image_bytes()), "image/png")},
        )
        mock_db.rollback.assert_called_once()

    def test_predict_error_detail_on_500(self, client, mock_predictor):
        mock_predictor.predict.side_effect = RuntimeError("Model crashed")
        data = client.post(
            "/predict",
            files={"file": ("chest.png", io.BytesIO(_make_image_bytes()), "image/png")},
        ).json()
        assert "detail" in data
