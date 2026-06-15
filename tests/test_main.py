import sys
import io
from datetime import datetime, timezone
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

# 1. TRICK PYTHON: Stop heavy ML libraries from loading BEFORE importing main!
sys.modules['torch'] = MagicMock()
sys.modules['torchvision'] = MagicMock()
sys.modules['torchvision.transforms'] = MagicMock()
sys.modules['torchxrayvision'] = MagicMock()
sys.modules['skimage'] = MagicMock()
sys.modules['skimage.io'] = MagicMock()

# NOW we can safely import main without needing PyTorch installed
import main

def test_health_endpoint(monkeypatch):
    mock_predictor = MagicMock()
    monkeypatch.setitem(main.ml_models, "predictor", mock_predictor)
    monkeypatch.setitem(main.app_state, "startup_time", main.time.time() - 10)

    with TestClient(main.app) as client:
        response = client.get('/health')

    assert response.status_code == 200
    assert response.json()['status'] == 'healthy'


def test_predict_success(monkeypatch, tmp_path):
    # 2. Mock Predictor using MagicMock
    mock_predictor = MagicMock()
    mock_predictor.predict.return_value = {"Atelectasis": 0.91}
    monkeypatch.setitem(main.ml_models, 'predictor', mock_predictor)
    
    # 3. Mock Database using MagicMock
    mock_db = MagicMock()
    
    # Fake the behavior of db.refresh() to add an ID and time
    def fake_refresh(record):
        record.id = 1
        record.created_at = datetime.now(timezone.utc)
    mock_db.refresh.side_effect = fake_refresh
    
    main.app.dependency_overrides[main.get_db] = lambda: iter([mock_db])
    monkeypatch.setattr(main, 'IMAGES_DIR', tmp_path)

    file_content = b'fake image bytes'
    with TestClient(main.app) as client:
        response = client.post(
            '/predict',
            files={'file': ('chest_xray.png', io.BytesIO(file_content), 'image/png')}
        )

    main.app.dependency_overrides.clear()

    # 4. Verify the results and check if mocks were called
    assert response.status_code == 200
    assert response.json()['success'] is True
    mock_predictor.predict.assert_called_once()
    mock_db.add.assert_called_once()
    mock_db.commit.assert_called_once()
    mock_db.refresh.assert_called_once()


def test_predict_rejects_non_image():
    with TestClient(main.app) as client:
        response = client.post(
            '/predict',
            files={'file': ('notes.txt', io.BytesIO(b'hello'), 'text/plain')}
        )
    assert response.status_code == 400
    assert response.json()['detail'] == 'File must be an image.'
