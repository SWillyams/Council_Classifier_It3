import sys
from pathlib import Path

import joblib
import pytest
from fastapi import HTTPException

API_DIR = Path(__file__).resolve().parents[1] / "API"
sys.path.insert(0, str(API_DIR))

import main as api
from API_lib import get_confidence_feedback

#Prep
@pytest.fixture(scope="module")
def model_artifacts():
    model_dir = Path(__file__).resolve().parents[1] / "model"
    return {
        "model": joblib.load(model_dir / "model.joblib"),
        "vectoriser": joblib.load(model_dir / "vectoriser.joblib"),
    }


@pytest.fixture
def loaded_model(monkeypatch, model_artifacts):
    monkeypatch.setattr(api, "artefacts", dict(model_artifacts))
    return model_artifacts

#Tests:
def test_health_endpoint_returns_ok_when_model_is_loaded(loaded_model):
    assert api.health() == {"status": "ok"}


def test_health_endpoint_returns_503_when_model_is_not_loaded(monkeypatch):
    monkeypatch.setattr(api, "artefacts", {})

    with pytest.raises(HTTPException) as exc_info:
        api.health()

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "Model is not loaded"


def test_categories_endpoint_returns_model_categories(loaded_model):
    assert api.categories() == {
        "categories": loaded_model["model"].classes_.tolist()
    }


def test_categories_endpoint_returns_503_when_model_is_not_loaded(monkeypatch):
    monkeypatch.setattr(api, "artefacts", {})

    with pytest.raises(HTTPException) as exc_info:
        api.categories()

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "Model is not loaded"


def test_predict_endpoint_categorises_overflowing_rubbish_bin(loaded_model):
    #Valid request
    request_text = "Rubbish bin overflowing"

    result = api.predict(api.PredictionRequest(request_text=request_text))

    assert result["request_text"] == request_text
    assert result["predicted_category"] == "waste"
    assert result["confidence"] > 0.5
    assert result["confidence_rank"] == "high"
    assert result["needs_review"] is False

def test_predict_endpoint_categorises_unknown_request(loaded_model):
    #unknown request
    request_text = "doom"

    result = api.predict(api.PredictionRequest(request_text=request_text))

    assert result["request_text"] == request_text
    assert result["predicted_category"] == ""
    assert result["confidence"] == 0
    assert result["confidence_rank"] == "very_low"
    assert result["needs_review"] is True


def test_predict_endpoint_rejects_blank_text():
    request = api.PredictionRequest(request_text="  ")

    with pytest.raises(HTTPException) as exc_info:
        api.predict(request)

    assert exc_info.value.status_code == 422
    assert exc_info.value.detail == "text must not be blank"


def test_predict_endpoint_returns_503_when_model_is_not_loaded(monkeypatch):
    monkeypatch.setattr(api, "artefacts", {})

    with pytest.raises(HTTPException) as exc_info:
        api.predict(api.PredictionRequest(request_text="Rubbish bin overflowing"))

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "Model is not loaded"


@pytest.mark.parametrize(
    ("confidence", "expected"),
    [
        (0.8, ("high", False, None)),
        (0.51, ("medium", False, None)),
        (0.31, ("low", True, "low_confidence")),
        (0.2, ("very_low", True, "very_low_confidence")),
    ],
)
def test_get_confidence_feedback(confidence, expected):
    assert get_confidence_feedback(confidence) == expected