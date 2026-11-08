"""Tests for the minimal classification API."""

from io import BytesIO
from pathlib import Path

import numpy as np
import requests
from fastapi.testclient import TestClient
from PIL import Image

from backend.app.main import app, classification_service


def _png_bytes() -> bytes:
    image = Image.fromarray(np.zeros((32, 32), dtype=np.uint8), mode="L")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


client = TestClient(app)


def test_health_endpoints() -> None:
    assert client.get("/health").json() == {
        "status": "ok",
        "service": "medical-imaging-api",
    }
    response = client.get("/health/models")
    assert response.status_code == 200
    assert response.json()["classification_model_available"] is True


def test_classification_inference_uses_real_checkpoint() -> None:
    response = client.post(
        "/predict/classification",
        content=_png_bytes(),
        headers={"content-type": "image/png"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "densenet121"
    assert body["model_artifact"] == "best_classifier.pt"
    assert len(body["predictions"]) == 14
    assert all(0.0 <= item["probability"] <= 1.0 for item in body["predictions"])


def test_classification_rejects_empty_image() -> None:
    response = client.post(
        "/predict/classification",
        content=b"",
        headers={"content-type": "image/png"},
    )

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "invalid_image"


def test_classification_rejects_corrupt_image() -> None:
    response = client.post(
        "/predict/classification",
        content=b"not an image",
        headers={"content-type": "image/png"},
    )

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "invalid_image"


def test_classification_rejects_oversized_upload(monkeypatch) -> None:
    monkeypatch.setattr(
        "backend.app.main.max_upload_size_bytes",
        lambda: 4,
    )
    response = client.post(
        "/predict/classification",
        content=b"12345",
        headers={"content-type": "image/png"},
    )

    assert response.status_code == 413
    assert response.json()["detail"]["code"] == "upload_too_large"


def test_model_health_reports_missing_checkpoint(monkeypatch) -> None:
    monkeypatch.setattr(
        classification_service,
        "checkpoint_path",
        Path("missing-classifier.pt"),
    )
    try:
        response = client.get("/health/models")
        assert response.status_code == 200
        assert response.json()["classification_model_available"] is False
    finally:
        monkeypatch.setattr(
            classification_service,
            "checkpoint_path",
            Path("models/classification/best_classifier.pt"),
        )


def test_grad_cam_returns_controlled_overlay() -> None:
    response = client.post(
        "/explainability/grad-cam?label=Cardiomegaly",
        content=_png_bytes(),
        headers={"content-type": "image/png"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["target_label"] == "Cardiomegaly"
    assert body["heatmap_shape"] == [7, 7]
    assert body["overlay_base64"]
    assert "image_path" not in body
    assert "overlay_path" not in body


def test_grad_cam_rejects_unknown_label() -> None:
    response = client.post(
        "/explainability/grad-cam?label=Unknown",
        content=_png_bytes(),
        headers={"content-type": "image/png"},
    )

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "invalid_grad_cam_request"


def test_dicom_rejects_non_dicom_payload() -> None:
    response = client.post(
        "/dicom/instances",
        content=b"not dicom",
        headers={"content-type": "application/dicom"},
    )

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "invalid_dicom"


def test_dicom_upload_validates_and_forwards(monkeypatch) -> None:
    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, str]:
            return {"ID": "orthanc-instance", "Type": "Instance"}

    def fake_upload(*args, **kwargs):
        return FakeResponse()

    monkeypatch.setattr("backend.app.services.dicom.requests.post", fake_upload)
    payload = (Path(__file__).parent / "fixtures" / "sample.dcm").read_bytes()

    response = client.post(
        "/dicom/instances",
        content=payload,
        headers={"content-type": "application/dicom"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["orthanc_id"] == "orthanc-instance"
    assert body["metadata"]["modality"] == "OT"


def test_dicom_studies_returns_controlled_orthanc_error(monkeypatch) -> None:
    def fail_listing(*args, **kwargs):
        raise requests.ConnectionError("offline")

    monkeypatch.setattr("backend.app.services.dicom.requests.get", fail_listing)

    response = client.get("/dicom/studies")

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "orthanc_unavailable"


def test_review_persists_across_repository_calls(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'reviews.sqlite3'}")
    payload = {
        "reference": "generated-image-1",
        "action": "approve",
        "model_artifact": "best_classifier.pt",
    }

    create_response = client.post("/reviews", json=payload)
    list_response = client.get("/reviews/generated-image-1")

    assert create_response.status_code == 201
    assert list_response.status_code == 200
    assert list_response.json()[0]["action"] == "approve"


def test_review_rejects_invalid_action(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'reviews.sqlite3'}")

    response = client.post(
        "/reviews",
        json={
            "reference": "generated-image-1",
            "action": "maybe",
            "model_artifact": "best_classifier.pt",
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "invalid_review"
