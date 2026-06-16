"""
test_model.py â€” ØªØ³Øª Ú©Ù„Ø§Ø³ CXRPredictor Ùˆ Ù…ØªØ¯ predict()

Ø§Ø² ÛŒÙ‡ stub Ú©Ø§Ù…Ù„ Ø¨Ø±Ø§ÛŒ torchxrayvision Ùˆ torch Ø§Ø³ØªÙØ§Ø¯Ù‡ Ù…ÛŒÚ©Ù†Ù‡.
Ù†ÛŒØ§Ø²ÛŒ Ø¨Ù‡ Ø¯Ø§Ù†Ù„ÙˆØ¯ Ù…Ø¯Ù„ ÛŒØ§ GPU Ù†ÛŒØ³Øª.
"""
import io
import sys
import types
import numpy as np
import pytest


# ---- Ø³Ø§Ø®Øª stub Ú©Ø§Ù…Ù„ Ù‚Ø¨Ù„ Ø§Ø² Ù‡Ø± import Ø¯ÛŒÚ¯Ù‡â€ŒØ§ÛŒ ----
#
# torchxrayvision.models.DenseNet Ø§Ø² torch.nn.Module Ø§Ø±Ø« Ù…ÛŒØ¨Ø±Ù‡.
# nn.Module ÛŒÙ‡ Ø³Ø±ÛŒ class attribute Ø¯Ø§Ø±Ù‡ Ù…Ø«Ù„ dump_patches Ú©Ù‡
# Ø¨Ø§ÛŒØ¯ Ø¯Ø± stub Ù‡Ù… ÙˆØ¬ÙˆØ¯ Ø¯Ø§Ø´ØªÙ‡ Ø¨Ø§Ø´Ù‡.
# Ø±Ø§Ù‡â€ŒØ­Ù„: FakeDenseNet Ø±Ùˆ Ø§Ø² nn.Module ÙˆØ§Ù‚Ø¹ÛŒ Ø§Ø±Ø« Ù…ÛŒØ¯ÛŒÙ….


def _install_xrv_stub():
    """
    torchxrayvision Ø±Ùˆ Ø¨Ø§ ÛŒÙ‡ stub Ø³Ø¨Ú© Ø¬Ø§ÛŒÚ¯Ø²ÛŒÙ† Ù…ÛŒÚ©Ù†Ù‡.
    FakeDenseNet Ø§Ø² torch.nn.Module Ø§Ø±Ø« Ù…ÛŒØ¨Ø±Ù‡ ØªØ§ Ù‡Ù…Ù‡â€ŒÛŒ attribute Ù‡Ø§ Ø¯Ø±Ø³Øª Ø¨Ø§Ø´Ù†.
    """
    import torch
    import torch.nn as nn

    PATHOLOGIES = [
        "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
        "Effusion", "Emphysema", "Fibrosis", "Hernia",
        "Infiltration", "Mass", "Nodule", "Pleural_Thickening",
        "Pneumonia", "Pneumothorax", "No Finding",
    ]

    # ---- fake datasets ----
    datasets_mod = types.ModuleType("torchxrayvision.datasets")

    def normalize(img, maxval):
        return (img.astype(np.float32) / maxval * 2.0) - 1.0

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

    # ---- fake models ----
    models_mod = types.ModuleType("torchxrayvision.models")

    class FakeDenseNet(nn.Module):
        """
        stub Ú©Ø§Ù…Ù„ Ø§Ø² DenseNet Ú©Ù‡ Ø§Ø² nn.Module Ø§Ø±Ø« Ù…ÛŒØ¨Ø±Ù‡.
        Ù‡Ù…Ù‡â€ŒÛŒ attribute Ù‡Ø§ÛŒ nn.Module Ù…Ø«Ù„ dump_patches Ø§ÛŒÙ†Ø¬Ø§ ÙˆØ¬ÙˆØ¯ Ø¯Ø§Ø±Ù†.
        """
        def __init__(self, weights=None):
            super().__init__()
            self.pathologies = PATHOLOGIES
            # ÛŒÙ‡ linear layer Ø³Ø§Ø¯Ù‡ Ø§Ø¶Ø§ÙÙ‡ Ù…ÛŒÚ©Ù†ÛŒÙ… ØªØ§ model Ù¾Ø§Ø±Ø§Ù…ØªØ± Ø¯Ø§Ø´ØªÙ‡ Ø¨Ø§Ø´Ù‡
            self._dummy = nn.Linear(1, len(PATHOLOGIES), bias=False)

        def forward(self, x):
            batch_size = x.shape[0]
            return torch.sigmoid(torch.rand(batch_size, len(self.pathologies)))

    models_mod.DenseNet = FakeDenseNet

    # ---- torchxrayvision root ----
    xrv_mod = types.ModuleType("torchxrayvision")
    xrv_mod.datasets = datasets_mod
    xrv_mod.models = models_mod

    sys.modules["torchxrayvision"] = xrv_mod
    sys.modules["torchxrayvision.datasets"] = datasets_mod
    sys.modules["torchxrayvision.models"] = models_mod


# stub Ø±Ùˆ Ù‚Ø¨Ù„ Ø§Ø² Ù‡Ø± import Ù†ØµØ¨ Ù…ÛŒÚ©Ù†ÛŒÙ…
_install_xrv_stub()

# Ø­Ø§Ù„Ø§ inference Ø±Ùˆ import Ù…ÛŒÚ©Ù†ÛŒÙ…
from app.inference import CXRPredictor


# ---- PNG Ù‡Ø§ÛŒ ØªØ³Øª ----

MINIMAL_PNG_GRAYSCALE = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x00\x00\x00\x00:~\x9bU\x00\x00"
    b"\x00\nIDATx\x9cc`\x00\x00\x00\x02\x00\x01\xe2!\xbc3"
    b"\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _make_rgb_png(size: int = 32) -> bytes:
    """ÛŒÙ‡ PNG Ø±Ù†Ú¯ÛŒ (RGB) Ø¨Ø§ Ø§Ø¨Ø¹Ø§Ø¯ Ø¯Ù„Ø®ÙˆØ§Ù‡ Ù…ÛŒØ³Ø§Ø²Ù‡."""
    import skimage.io as skio
    arr = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
    buf = io.BytesIO()
    skio.imsave(buf, arr, plugin="imageio", format="PNG")
    return buf.getvalue()


def _make_gray_png(size: int = 32) -> bytes:
    """ÛŒÙ‡ PNG Ø®Ø§Ú©Ø³ØªØ±ÛŒ (grayscale) Ø¨Ø§ Ø§Ø¨Ø¹Ø§Ø¯ Ø¯Ù„Ø®ÙˆØ§Ù‡ Ù…ÛŒØ³Ø§Ø²Ù‡."""
    import skimage.io as skio
    arr = np.random.randint(0, 255, (size, size), dtype=np.uint8)
    buf = io.BytesIO()
    skio.imsave(buf, arr, plugin="imageio", format="PNG")
    return buf.getvalue()


# ---- Fixture ----

@pytest.fixture(scope="module")
def predictor():
    """ÛŒÙ‡ instance Ø§Ø² CXRPredictor Ø¨Ø±Ø§ÛŒ Ú©Ù„ Ù…Ø§Ú˜ÙˆÙ„ ØªØ³Øª."""
    return CXRPredictor()


# ---- Tests: init ----

class TestCXRPredictorInit:

    def test_predictor_initializes_without_error(self, predictor):
        assert predictor is not None

    def test_predictor_has_model_attribute(self, predictor):
        assert hasattr(predictor, "model")

    def test_predictor_has_transform_attribute(self, predictor):
        assert hasattr(predictor, "transform")

    def test_model_is_not_none(self, predictor):
        assert predictor.model is not None

    def test_transform_is_not_none(self, predictor):
        assert predictor.transform is not None


# ---- Tests: ÙØ±Ù…Øª Ø®Ø±ÙˆØ¬ÛŒ predict() ----

class TestPredictOutputFormat:

    def test_predict_returns_dict(self, predictor):
        result = predictor.predict(MINIMAL_PNG_GRAYSCALE)
        assert isinstance(result, dict)

    def test_predict_dict_is_not_empty(self, predictor):
        result = predictor.predict(MINIMAL_PNG_GRAYSCALE)
        assert len(result) > 0

    def test_predict_keys_are_strings(self, predictor):
        result = predictor.predict(MINIMAL_PNG_GRAYSCALE)
        for key in result:
            assert isinstance(key, str), f"Key {key!r} is not a string"

    def test_predict_values_are_floats(self, predictor):
        result = predictor.predict(MINIMAL_PNG_GRAYSCALE)
        for key, value in result.items():
            assert isinstance(value, float), f"Value for {key!r} is not float"

    def test_predict_scores_are_rounded_to_4_decimals(self, predictor):
        result = predictor.predict(MINIMAL_PNG_GRAYSCALE)
        for key, score in result.items():
            assert score == round(score, 4), f"Score for {key!r} not rounded to 4 decimals"

    def test_predict_scores_in_valid_range(self, predictor):
        """sigmoid Ø®Ø±ÙˆØ¬ÛŒ Ø¨ÛŒÙ† Û° Ùˆ Û± Ù‡Ø³Øª."""
        result = predictor.predict(MINIMAL_PNG_GRAYSCALE)
        for key, score in result.items():
            assert 0.0 <= score <= 1.0, f"Score {score} for {key!r} out of [0,1] range"

    def test_predict_contains_no_finding_pathology(self, predictor):
        result = predictor.predict(MINIMAL_PNG_GRAYSCALE)
        assert "No Finding" in result

    def test_predict_contains_pneumonia_pathology(self, predictor):
        result = predictor.predict(MINIMAL_PNG_GRAYSCALE)
        assert "Pneumonia" in result

    def test_predict_contains_atelectasis_pathology(self, predictor):
        result = predictor.predict(MINIMAL_PNG_GRAYSCALE)
        assert "Atelectasis" in result

    def test_predict_returns_exactly_15_pathologies(self, predictor):
        result = predictor.predict(MINIMAL_PNG_GRAYSCALE)
        assert len(result) == 15, f"Expected 15 pathologies, got {len(result)}"


# ---- Tests: ÙØ±Ù…Øªâ€ŒÙ‡Ø§ÛŒ Ù…Ø®ØªÙ„Ù ØªØµÙˆÛŒØ± ----

class TestPredictImageFormats:

    def test_predict_accepts_grayscale_png(self, predictor):
        png = _make_gray_png(64)
        result = predictor.predict(png)
        assert isinstance(result, dict)
        assert len(result) == 15

    def test_predict_accepts_rgb_png(self, predictor):
        """
        ØªØµØ§ÙˆÛŒØ± RGB Ø¨Ø§ÛŒØ¯ Ø¯Ø§Ø®Ù„ predict() Ø¨Ù‡ grayscale ØªØ¨Ø¯ÛŒÙ„ Ø¨Ø´Ù†.
        """
        png = _make_rgb_png(64)
        result = predictor.predict(png)
        assert isinstance(result, dict)
        assert len(result) == 15

    def test_predict_rgb_and_gray_return_same_keys(self, predictor):
        result_rgb = predictor.predict(_make_rgb_png(32))
        result_gray = predictor.predict(_make_gray_png(32))
        assert set(result_rgb.keys()) == set(result_gray.keys())

    def test_predict_two_calls_return_same_keys(self, predictor):
        r1 = predictor.predict(MINIMAL_PNG_GRAYSCALE)
        r2 = predictor.predict(MINIMAL_PNG_GRAYSCALE)
        assert set(r1.keys()) == set(r2.keys())

    def test_predict_accepts_minimal_1x1_png(self, predictor):
        result = predictor.predict(MINIMAL_PNG_GRAYSCALE)
        assert isinstance(result, dict)


# ---- Tests: Ø¹Ù…Ù„Ú©Ø±Ø¯ Ùˆ Ù¾Ø§ÛŒØ¯Ø§Ø±ÛŒ ----

class TestPredictReliability:

    def test_predict_does_not_raise_exception(self, predictor):
        try:
            predictor.predict(MINIMAL_PNG_GRAYSCALE)
        except Exception as exc:
            pytest.fail(f"predict() raised an unexpected exception: {exc}")

    def test_predict_accepts_bytes_input(self, predictor):
        """predict() Ø¨Ø§ÛŒØ¯ bytes Ø¨Ú¯ÛŒØ±Ù‡ Ù†Ù‡ path."""
        assert isinstance(MINIMAL_PNG_GRAYSCALE, bytes)
        result = predictor.predict(MINIMAL_PNG_GRAYSCALE)
        assert result is not None

    def test_predict_returns_new_dict_each_call(self, predictor):
        r1 = predictor.predict(MINIMAL_PNG_GRAYSCALE)
        r2 = predictor.predict(MINIMAL_PNG_GRAYSCALE)
        assert r1 is not r2
