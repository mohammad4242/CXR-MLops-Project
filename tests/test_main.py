import io
from fastapi.testclient import TestClient

import main
from conftest import DummyPredictor, FakeDB


def test_health_endpoint():
    main.ml_models["predictor"] = DummyPredictor()
    main.app_state["startup_time"] = main.time.time() - 10

    with TestClient(main.app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["model_name"] == "densenet121-res224-all"


def test_predict_success(tmp_path):
    fake_db = FakeDB()

    main.ml_models["predictor"] = DummyPredictor()
    main.IMAGES_DIR = tmp_path
    main.app.dependency_overrides[main.get_db] = lambda: iter([fake_db])

    file_content = b"fake image bytes"

    with TestClient(main.app) as client:
        response = client.post(
            "/predict",
            files={"file": ("chest_xray.png", io.BytesIO(file_content), "image/png")},
        )

    main.app.dependency_overrides.clear()

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["filename"] == "chest_xray.png"
    assert data["predictions"]["Atelectasis"] == 0.91
    assert fake_db.committed is True
    assert len(fake_db.records) == 1

    saved_file = tmp_path / fake_db.records[0].stored_filename
    assert saved_file.exists()
    assert saved_file.read_bytes() == file_content


def test_predict_rejects_non_image():
    fake_db = FakeDB()

    main.ml_models["predictor"] = DummyPredictor()
    main.app.dependency_overrides[main.get_db] = lambda: iter([fake_db])

    with TestClient(main.app) as client:
        response = client.post(
            "/predict",
            files={"file": ("notes.txt", io.BytesIO(b"hello"), "text/plain")},
        )

    main.app.dependency_overrides.clear()

    assert response.status_code == 400
    assert response.json()["detail"] == "File must be an image."
    assert fake_db.committed is False
