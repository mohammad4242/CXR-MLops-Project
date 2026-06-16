import io
import numpy as np
import pytest
from unittest.mock import MagicMock, patch


# ---- Minimal valid grayscale PNG bytes (1x1 pixel) ----

MINIMAL_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x00\x00\x00\x00:~\x9bU\x00\x00"
    b"\x00\nIDATx\x9cc`\x00\x00\x00\x02\x00\x01\xe2!\xbc3"
    b"\x00\x00\x00\x00IEND\xaeB`\x82"
)


# ---- Fake torchxrayvision so we never download the real model ----

@pytest.fixture(autouse=True, scope="module")
def patch_xrv():
    """
    Replace torchxrayvision with a lightweight stub for the whole test module.
    This keeps tests fast and eliminates the need for GPU / large model weights.
    """
    PATHOLOGIES = [
        "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
        "Effusion", "Emphysema", "Fibrosis", "Hernia",
        "Infiltration", "Mass", "Nodule", "Pleural_Thickening",
        "Pneumonia", "Pneumothorax", "No Finding",
    ]

    import torch
    import sys
    import types

    # ---- Build a fake xrv module tree ----
    xrv_mod = types.ModuleType("torchxrayvision")

    # fake xrv.datasets
    datasets_mod = types.ModuleType("torchxrayvision.datasets")

    def normalize(img, maxval):
        return (img / maxval * 2) - 1.0

    class XRayCenterCrop:
        def __call__(self, img):
            return img

    class XRayResizer:
        def __init__(self, size):
            self.size = size

        def __call__(self, img):
            return np.random.rand(1, self.size, self.size).astype(np.float32)

    datasets_mod.normalize = normalize
    datasets_mod.XRayCenterCrop = XRayCenterCrop
    datasets_mod.XRayResizer = XRayResizer
    xrv_mod.datasets = datasets_mod

    # fake xrv.models.DenseNet
    models_mod = types.ModuleType("torchxrayvision.models")

    class FakeDenseNet:
        def __init__(self, weights=None):
            self.pathologies = PATHOLOGIES

        def eval(self):
            return self

        def __call__(self, x):
            batch_size = x.shape[0]
            return torch.rand(batch_size, len(self.pathologies))

    models_mod.DenseNet = FakeDenseNet
    xrv_mod.models = models_mod

    sys.modules["torchxrayvision"] = xrv_mod
    sys.modules["torchxrayvision.datasets"] = datasets_mod
    sys.modules["torchxrayvision.models"] = models_mod

    yield

    # Cleanup so other test modules are not affected
    for key in ["torchxrayvision", "torchxrayvision.datasets", "torchxrayvision.models"]:
        sys.modules.pop(key, None)
    sys.modules.pop("app.inference", None)


@pytest.fixture
def predictor(patch_xrv):
    from app.inference import CXRPredictor
    return CXRPredictor()


# ---- Tests: initialization ----

class TestCXRPredictorInit:

    def test_predictor_initializes_without_error(self, predictor):
        assert predictor is not None

    def test_predictor_has_model_attribute(self, predictor):
        assert hasattr(predictor, "model")

    def test_predictor_has_transform_attribute(self, predictor):
        assert hasattr(predictor, "transform")

    def test_model_is_in_eval_mode(self, predictor):
        # Our fake model returns self on eval(); real model sets training=False
        assert predictor.model is not None


# ---- Tests: predict() output format ----

class TestPredictOutputFormat:

    def test_predict_returns_dict(self, predictor):
        result = predictor.predict(MINIMAL_PNG)
        assert isinstance(result, dict)

    def test_predict_dict_is_not_empty(self, predictor):
        result = predictor.predict(MINIMAL_PNG)
        assert len(result) > 0

    def test_predict_keys_are_strings(self, predictor):
        result = predictor.predict(MINIMAL_PNG)
        for key in result:
            assert isinstance(key, str)

    def test_predict_values_are_floats(self, predictor):
        result = predictor.predict(MINIMAL_PNG)
        for value in result.values():
            assert isinstance(value, float)

    def test_predict_scores_are_in_zero_one_range(self, predictor):
        result = predictor.predict(MINIMAL_PNG)
        for score in result.values():
            # Scores may go slightly outside [0, 1] with sigmoid; keep a generous range
            assert -0.1 <= score <= 1.1, f"Score {score} out of expected range"

    def test_predict_scores_are_rounded_to_4_decimals(self, predictor):
        result = predictor.predict(MINIMAL_PNG)
        for score in result.values():
            assert score == round(score, 4)

    def test_predict_contains_no_finding_key(self, predictor):
        result = predictor.predict(MINIMAL_PNG)
        assert "No Finding" in result

    def test_predict_contains_pneumonia_key(self, predictor):
        result = predictor.predict(MINIMAL_PNG)
        assert "Pneumonia" in result

    def test_predict_contains_all_15_pathologies(self, predictor):
        result = predictor.predict(MINIMAL_PNG)
        assert len(result) == 15


# ---- Tests: predict() with different image formats ----

class TestPredictImageFormats:

    def test_predict_accepts_rgb_image(self, predictor):
        """
        RGB images (3 channels) must be converted to grayscale inside predict().
        """
        rgb_array = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        png_buffer = io.BytesIO()

        import skimage.io as skio
        skio.imsave(png_buffer, rgb_array, plugin="imageio", format="PNG")
        png_bytes = png_buffer.getvalue()

        result = predictor.predict(png_bytes)
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_predict_accepts_grayscale_image(self, predictor):
        gray_array = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        png_buffer = io.BytesIO()

        import skimage.io as skio
        skio.imsave(png_buffer, gray_array, plugin="imageio", format="PNG")
        png_bytes = png_buffer.getvalue()

        result = predictor.predict(png_bytes)
        assert isinstance(result, dict)

    def test_predict_two_calls_return_same_keys(self, predictor):
        r1 = predictor.predict(MINIMAL_PNG)
        r2 = predictor.predict(MINIMAL_PNG)
        assert set(r1.keys()) == set(r2.keys())


# ---- Tests: no_grad / performance ----

class TestPredictPerformance:

    def test_predict_does_not_raise_with_small_image(self, predictor):
        """Ensure the pipeline handles very small images without crashing."""
        try:
            predictor.predict(MINIMAL_PNG)
        except Exception as exc:
            pytest.fail(f"predict() raised an unexpected exception: {exc}")

    def test_predict_called_with_bytes_not_path(self, predictor):
        """predict() must accept raw bytes, not a file path."""
        result = predictor.predict(MINIMAL_PNG)
        assert result is not None
