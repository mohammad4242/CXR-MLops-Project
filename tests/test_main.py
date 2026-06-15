import io
import sys
import importlib
from datetime import datetime, timezone
from types import ModuleType
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient


class DummyPredictor:
    def predict(self, image_bytes: bytes) -> dict:
        return {"Atelectasis": 0.91, "Effusion": 0.12}


class FakeRecord:
    _id = 1

    def __init__(self, **kwargs):
        self.id = None
        self.created_at = None
        for key, value in kwargs.items():
            setattr(self, key, value)


class FakeDB:
    def __init__(self):
        self.records = []
        self.committed = False
        self.rolled_back = False

    def add(self, record):
        self.records.append(record)

    def commit(self):
        self.committed = True
        if self.records:
            record = self.records[-1]
            record.id = FakeRecord._id
            FakeRecord._id += 1
            record.created_at = datetime.now(timezone.utc)

    def refresh(self, record):
        return record

    def rollback(self):
        self.rolled_back = True


def load_main_with_mocks(tmp_path=None):
    fake_database = ModuleType("database")
    fake_database.engine = MagicMock()
    fake_database.get_db = lambda: iter([])
    fake_database.Base = MagicMock()
    fake_database.Base.metadata = MagicMock()
    fake_database.Base.metadata.create_all = MagicMock()

    fake_models = ModuleType("models")
    fake_models.PredictionRecord = FakeRecord

    fake_inference = ModuleType("inference")
    fake_inference.CXRPredictor = MagicMock(return_value=DummyPredictor())

    fake_torch = MagicMock()
    fake_torchvision = MagicMock()
    fake_torchvision_transforms = MagicMock()
    fake_torchxrayvision = MagicMock()
    fake_skimage = MagicMock()
    fake_skimage_io = MagicMock()

    mocked_modules = {
        "database": fake_database,
        "models": fake_models,
        "inference": fake_inference,
        "torch": fake_torch,
        "torchvision": fake_torchvision,
        "torchvision.transforms": fake_torchvision_transforms,
        "torchxrayvision": fake_torchxrayvision,
        "skimage": fake_skimage,
        "skimage.io": fake_skimage_io,
    }

    with patch.dict(sys.modules, mocked_modules):
        if "main" in sys.modules:
            del sys.modules["main"]
        main = importlib.import_module("main")

    if tmp_path is not None:
        main.IMAGES_DIR = tmp_path

    return main


def test_health_endpoint():
    main = load_main_with_mocks()
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
    main = load_main_with_mocks(tmp_path=tmp_path)
    fake_db = FakeDB()

    main.ml_models["predictor"] = DummyPredictor()
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
    main = load_main_with_mocks()
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
