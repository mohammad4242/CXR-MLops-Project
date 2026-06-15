import sys
from types import ModuleType
from unittest.mock import MagicMock
from datetime import datetime, timezone


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

sys.modules["database"] = fake_database
sys.modules["models"] = fake_models
sys.modules["inference"] = fake_inference
sys.modules["torch"] = fake_torch
sys.modules["torchvision"] = fake_torchvision
sys.modules["torchvision.transforms"] = fake_torchvision_transforms
sys.modules["torchxrayvision"] = fake_torchxrayvision
sys.modules["skimage"] = fake_skimage
sys.modules["skimage.io"] = fake_skimage_io
