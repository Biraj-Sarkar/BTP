"""The HTTP contract the mobile app depends on.

The Flutter client (`lib/main.dart`, `lib/results.dart`) posts a multipart image
to `/anemia-detection` and reads five response fields by name. For a while the
server only answered `/predict`, so every request from the app would have
returned 404 while every field inside the request and response already matched —
the kind of break that no pipeline test can see. These tests pin the contract at
the HTTP layer instead.

Uses a linear model with a classical extractor, so no PyTorch model or trained
bundle is needed to run them.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from synthetic import make_eye  # noqa: E402

# The exact keys `ScreeningResult.fromJson` in lib/results.dart reads first.
APP_FIELDS = ("hb_g_dl", "anemic", "usable", "threshold_g_dl", "message")


@pytest.fixture()
def client(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from anemia.linear_model import fit_linear_model

    X = np.array([[0.10, -0.02], [0.20, 0.01], [0.15, 0.00], [0.05, -0.03], [0.25, 0.02]])
    y = np.array([9.0, 12.0, 11.0, 8.5, 13.0])
    model_path = tmp_path / "linear_model.json"
    fit_linear_model(X, y, "erythema", extractor="refined").save(model_path)

    monkeypatch.setenv("ANEMIA_LINEAR_MODEL", str(model_path))
    monkeypatch.delenv("ANEMIA_CHECKPOINT", raising=False)

    import serve.app as app_module
    app_module = importlib.reload(app_module)  # drop any predictor cached by another test
    return TestClient(app_module.app)


def _upload():
    image, _ = make_eye(seed=21)
    ok, png = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    assert ok
    return {"image": ("eye.png", png.tobytes(), "image/png")}


FORM = {"age_years": "30", "sex": "f", "pregnant": "false"}


@pytest.mark.parametrize("path", ["/anemia-detection", "/predict"])
def test_route_answers_with_every_field_the_app_reads(client, path):
    response = client.post(path, files=_upload(), data=FORM)

    assert response.status_code == 200, response.text
    body = response.json()
    missing = [k for k in APP_FIELDS if k not in body]
    assert not missing, f"{path} response lacks {missing}"


def test_both_routes_return_the_same_answer(client):
    """One handler behind two paths — they must never diverge."""

    first = client.post("/anemia-detection", files=_upload(), data=FORM).json()
    second = client.post("/predict", files=_upload(), data=FORM).json()
    assert first == second


def test_unreadable_upload_is_a_client_error_not_a_crash(client):
    response = client.post("/anemia-detection",
                           files={"image": ("eye.png", b"not an image", "image/png")},
                           data=FORM)
    assert response.status_code == 400
