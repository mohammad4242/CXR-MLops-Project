"""
test_api.py — endpoint tests for the FastAPI app (/health and /predict).

No real PostgreSQL or ML model is needed: the DB session, the predictor, and
the authenticated user are all replaced with lightweight test doubles.
"""
import io
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ---- Helpers ----

def _make_image_bytes() -> bytes:
    """Return a valid one-pixel PNG."""
    return (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
        b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00"
        b"\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18"
        b"\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
    )


# "No Finding" is above the 65% threshold; everything else is below it.
FAKE_PREDICTIONS = {
    "Atelectasis": 0.1234,
    "Cardiomegaly": 0.0567,
    "Effusion": 0.3456,
    "Pneumonia": 0.2345,
    "No Finding": 0.8901,
}

ALL_LOW_PREDICTIONS = {
    "Atelectasis": 0.12,
    "Pneumonia": 0.34,
    "Effusion": 0.5,
}


# ---- App + client setup ----
#
# Strategy:
#  1. Before importing app, patch the heavy dependencies (engine, model).
#  2. After building the TestClient (which runs lifespan), replace
#     ml_models["predictor"] with our mock.
#  3. Override get_db and get_current_user so no real DB or token is required.

@pytest.fixture(scope="module")
def app_and_mocks():
    mock_db = MagicMock()

    mock_predictor = MagicMock()
    mock_predictor.predict.return_value = FAKE_PREDICTIONS.copy()

    fake_user = SimpleNamespace(
        id=1,
        email="tester@example.com",
        full_name="Tester",
        created_at=datetime.now(timezone.utc),
    )

    with (
        patch("app.database.create_engine"),
        patch("app.database.Base.metadata.create_all"),
        patch("app.inference.CXRPredictor", return_value=mock_predictor),
        patch("app.main.Base.metadata.create_all"),
        patch("app.main.engine"),  # keep the startup migration from touching a real DB
    ):
        import app.main as main_module
        from app.main import app, ml_models, get_db
        from app.security import get_current_user

        # Redirect image writes to a throwaway temp dir so the tests stay
        # hermetic and never depend on a writable ./images folder.
        tmp_images = tempfile.mkdtemp(prefix="cxr_test_images_")
        main_module.IMAGES_DIR = Path(tmp_images)

        ml_models["predictor"] = mock_predictor

        app.dependency_overrides[get_db] = lambda: mock_db
        app.dependency_overrides[get_current_user] = lambda: fake_user

        with TestClient(app, raise_server_exceptions=False) as client:
            yield client, mock_predictor, mock_db

        app.dependency_overrides.clear()
        shutil.rmtree(tmp_images, ignore_errors=True)


@pytest.fixture(autouse=True)
def reset_mocks(app_and_mocks):
    """Reset the mocks before each test so call counts don't leak between them."""
    _client, mock_predictor, mock_db = app_and_mocks
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


def _post_image(client, filename="chest.png", content_type="image/png", data=None):
    payload = data if data is not None else _make_image_bytes()
    return client.post(
        "/predict",
        files={"file": (filename, io.BytesIO(payload), content_type)},
    )


# ---- Tests: GET /health ----

class TestHealthEndpoint:

    def test_health_returns_200(self, client):
        assert client.get("/health").status_code == 200

    def test_health_response_has_status_key(self, client):
        assert "status" in client.get("/health").json()

    def test_health_response_has_model_loaded_key(self, client):
        assert "model_loaded" in client.get("/health").json()

    def test_health_response_has_model_name_key(self, client):
        assert "model_name" in client.get("/health").json()

    def test_health_response_has_uptime_seconds_key(self, client):
        assert "uptime_seconds" in client.get("/health").json()

    def test_health_status_is_healthy(self, client):
        assert client.get("/health").json()["status"] == "healthy"

    def test_health_model_is_loaded(self, client):
        assert client.get("/health").json()["model_loaded"] is True

    def test_health_model_name_is_correct(self, client):
        assert client.get("/health").json()["model_name"] == "densenet121-res224-all"


# ---- Tests: POST /predict ----

class TestPredictEndpoint:

    def test_predict_returns_200_with_valid_image(self, client):
        assert _post_image(client).status_code == 200

    def test_predict_response_has_findings_key(self, client):
        assert "findings" in _post_image(client).json()

    def test_predict_response_success_is_true(self, client):
        assert _post_image(client).json()["success"] is True

    def test_predict_response_has_id(self, client):
        assert "id" in _post_image(client).json()

    def test_predict_response_filename_matches_upload(self, client):
        assert _post_image(client).json()["filename"] == "chest.png"

    def test_predict_only_returns_findings_above_threshold(self, client):
        findings = _post_image(client).json()["findings"]
        # Above 0.65 → shown; below → hidden.
        assert findings["No Finding"] == 0.8901
        assert "Pneumonia" not in findings
        assert "Effusion" not in findings

    def test_predict_has_findings_true_when_above_threshold(self, client):
        data = _post_image(client).json()
        assert data["has_findings"] is True
        assert data["message"] is None

    def test_predict_threshold_is_reported(self, client):
        assert _post_image(client).json()["threshold"] == 0.65

    def test_predict_no_findings_when_all_below_threshold(self, client, mock_predictor):
        mock_predictor.predict.return_value = ALL_LOW_PREDICTIONS.copy()
        data = _post_image(client).json()
        assert data["findings"] == {}
        assert data["has_findings"] is False
        assert "65%" in data["message"]

    def test_predict_rejects_pdf_with_400(self, client):
        response = _post_image(
            client, filename="report.pdf", content_type="application/pdf", data=b"%PDF-1.4"
        )
        assert response.status_code == 400

    def test_predict_error_detail_mentions_image(self, client):
        response = _post_image(
            client, filename="report.pdf", content_type="application/pdf", data=b"%PDF-1.4"
        )
        assert "image" in response.json()["detail"].lower()

    def test_predict_calls_model_predict_once(self, client, mock_predictor):
        _post_image(client)
        mock_predictor.predict.assert_called_once()

    def test_predict_calls_db_commit(self, client, mock_db):
        _post_image(client)
        mock_db.commit.assert_called_once()

    def test_predict_returns_500_on_model_failure(self, client, mock_predictor):
        mock_predictor.predict.side_effect = RuntimeError("Model crashed")
        assert _post_image(client).status_code == 500

    def test_predict_rollback_called_on_model_failure(self, client, mock_predictor, mock_db):
        mock_predictor.predict.side_effect = RuntimeError("Model crashed")
        _post_image(client)
        mock_db.rollback.assert_called_once()

    def test_predict_error_detail_on_500(self, client, mock_predictor):
        mock_predictor.predict.side_effect = RuntimeError("Model crashed")
        assert "detail" in _post_image(client).json()
