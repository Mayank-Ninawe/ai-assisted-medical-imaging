# AI-Assisted Medical Imaging

Explainable chest X-ray classification with a React workspace, FastAPI
backend, reviewer feedback, and local Orthanc/DICOM integration.

> Educational prototype only. Not for clinical use, diagnosis, treatment,
> or patient-care decisions.

## Overview

This internship project implements a local educational workflow for:

1. Uploading a chest X-ray image.
2. Running a 14-label DenseNet-121 classifier.
3. Viewing model scores and label-specific Grad-CAM overlays.
4. Recording reviewer feedback in SQLite.
5. Uploading, listing, and retrieving DICOM data through backend endpoints
   connected to Orthanc.

A separate MONAI-based spleen segmentation workflow is included as
experimental research. It is not a reliable application feature and is not
integrated into the classification workspace.

The chest X-ray classifier and CT spleen segmenter are separate models,
datasets, and tasks.

## Current scope

| Component | Current implementation | Important boundary |
|---|---|---|
| Classification | Training, evaluation, checkpoint inference, and web results | Educational scores; no clinical validation |
| Grad-CAM | Label-specific overlays displayed beside the original image | Attribution is not proof of diagnostic reasoning |
| Reviewer feedback | Approve/reject actions persisted in SQLite | Feedback does not retrain the model |
| Backend | FastAPI endpoints for classification, Grad-CAM, reviews, and DICOM | No authentication or authorization |
| Frontend | React/TypeScript workspace with image preview and sorted scores | No DICOM viewer or segmentation visualization |
| Orthanc | Local DICOM upload, study listing, and instance retrieval | API integration, not a complete PACS/viewer product |
| Segmentation | MONAI training, evaluation, checkpoints, and notebooks | Poor measured performance; experimental only |
| Deployment | Local Docker Compose stack | Not production-ready |
| CI | Python and frontend checks configured in GitHub Actions | Check the latest Actions run for its actual result |
| DVC | Initialization metadata | No operational pipeline or remote |

## Architecture

```text
Browser: React / TypeScript
    |
    | HTTP requests
    v
FastAPI backend
    |
    +-- DenseNet-121 classification checkpoint
    +-- Grad-CAM generation
    +-- SQLite reviewer feedback
    +-- Orthanc REST API
            |
            +-- Persistent DICOM storage and index

Separate research workflow:
MSD spleen CT data -> MONAI preprocessing -> 3D U-Net -> evaluation reports
```

The web workspace currently uses classification, Grad-CAM, and review
endpoints. Orthanc operations are exposed through the backend API; the
workspace does not include a DICOM viewer.

## Classification results

Reported evaluation artifact:

```text
reports/classification_test_metrics.json
```

| Property | Recorded value |
|---|---|
| Dataset | NIH ChestX-ray14, according to classification configuration |
| Checkpoint | `models/classification/best_classifier.pt` |
| Test images | 25,596 |
| Labels | 14 |
| Macro AUROC | 0.796025 |
| Macro average precision | 0.256642 |
| Decision threshold | 0.5 |
| Evaluation device | CUDA |

These are recorded evaluation results, not a new evaluation performed when
the repository is cloned.

### Interpretation and limitations

- Macro AUROC is not classification accuracy.
- At threshold 0.5, the report records zero recall and zero F1 for
  Consolidation, Edema, Fibrosis, Hernia, and Pneumonia.
- Other labels also have limited recall at this threshold.
- Thresholds have not been demonstrated to be calibrated or clinically
  appropriate.
- Model scores must not be interpreted as validated individual disease risk.
- The configuration reserves the official test list and specifies a
  patient-level validation split from the train/validation pool, with
  validation fraction 0.1 and random seed 42.
- The metrics JSON does not contain a dataset-manifest hash or complete run
  provenance. The split policy in the current configuration alone does not
  prove the exact lineage of a historical checkpoint.
- External validation, confidence intervals, and clinical validation have
  not been established.

### Input configuration

The current classification configuration specifies:

- Grayscale input.
- Resize to 224 x 224 pixels.
- Normalization mean 0.5 and standard deviation 0.5.
- Two training epochs.
- Batch size 16.
- Learning rate 0.0001.

See `ml/classification/config.yaml`. Current configuration values must not
be treated as an immutable snapshot of every historical experiment.

## Experimental spleen segmentation

The repository contains MONAI preprocessing, a 3D U-Net workflow,
training/evaluation scripts, notebooks, and historical artifacts.

Recorded results include:

- Current checkpoint audit: validation Dice approximately 0.0533.
- A historical held-out checkpoint: validation Dice approximately 0.1099.
- A historical six-volume evaluation report: mean Dice approximately 0.1477.

These values belong to different artifacts and runs. They must not be
combined into a single experiment result or presented as directly
comparable without reconciling their configurations and splits.

Segmentation remains unsuitable as a reliable product feature because of
poor overlap scores and reported over-segmentation. It is not exposed
through the current application UI.

## Requirements

For the documented development setup:

- Python 3.10.
- Node.js 22.14.0, matching the configured frontend CI environment.
- npm.
- Docker with Docker Compose for the containerized application and Orthanc.
- A separately provisioned classification checkpoint for real inference.

Large datasets and trained model files are not provided through normal Git
history. No automated model download or DVC remote is configured.

## Python development setup

Run from the repository root.

### Windows CMD

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install -e ".[dev,backend,ml]"
```

### Linux/macOS

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,backend,ml]"
```

Do not recreate an existing working environment unnecessarily.

### Model prerequisite

Place the compatible trained classification checkpoint at:

```text
models/classification/best_classifier.pt
```

The checkpoint must match the repository's classifier architecture and label
ordering. Use only trusted checkpoint files.

Without the checkpoint, the API can report degraded model availability;
real classification and Grad-CAM inference are unavailable.

### Run the backend locally

```bash
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

- API documentation: http://localhost:8000/docs
- Service health: http://localhost:8000/health
- Checkpoint availability: http://localhost:8000/health/models

Model health reports checkpoint-file availability. It is not a full model
load, prediction, or accuracy check.

## Frontend development

From the repository root:

```bash
cd frontend
npm ci
npm test
npm run build
npm run dev
```

The frontend API client defaults to `http://localhost:8000`.

`VITE_API_BASE_URL` can override the API base URL when starting Vite or
building the frontend. The container serves static built assets; changing a
runtime environment variable alone does not update an existing bundle.

The default CORS configuration permits the local development and Docker
frontend origins:

- http://localhost:5173
- http://localhost:8080

## Local Docker demonstration

From the repository root, with the checkpoint available:

```bash
docker compose up -d --build
docker compose ps
```

| Service | Local address |
|---|---|
| Web workspace | http://localhost:8080 |
| Backend API documentation | http://localhost:8000/docs |
| Orthanc HTTP API/interface | http://localhost:8042 |
| Orthanc DICOM port | 4242 |

The Compose configuration mounts the host `models` directory read-only.
It uses named volumes for backend runtime data and Orthanc storage.

The checked-in Orthanc configuration contains development credentials.
Review the configuration before starting services. Do not expose this stack
to an untrusted network.

### Rebuild only the frontend

```bash
docker compose up -d --build --no-deps frontend
```

### Inspect backend logs

```bash
docker compose logs --tail 80 backend
```

### Stop services

```bash
docker compose down
```

Do not remove persistent volumes unless you intentionally want to discard
their stored data.

## API overview

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Service health |
| GET | `/health/models` | Classification checkpoint availability |
| POST | `/predict/classification` | Classify raw image bytes |
| POST | `/explainability/grad-cam?label=Effusion` | Generate a label-specific overlay |
| POST | `/reviews` | Save reviewer feedback |
| GET | `/reviews/{reference}` | Retrieve reviews for a reference |
| POST | `/dicom/instances` | Validate and forward raw DICOM bytes to Orthanc |
| GET | `/dicom/studies` | List Orthanc study identifiers |
| GET | `/dicom/instances/{instance_id}` | Retrieve a DICOM instance |

Classification, Grad-CAM, and DICOM uploads use raw request bodies, not
multipart form uploads.

Use the generated API documentation for request and response schemas.

## Testing

### Python lint and tests

From the repository root:

```bash
python -m ruff check .
python -m pytest -q
```

Tests that require a real classification checkpoint explicitly skip if the
configured checkpoint is absent.

### Live Orthanc tests: Windows CMD

Start the local Orthanc service first:

```cmd
set RUN_ORTHANC_INTEGRATION=1
python -m pytest backend/tests/test_orthanc_integration.py -v
python -m pytest -q
```

Disable live integration tests in the current CMD session:

```cmd
set RUN_ORTHANC_INTEGRATION=
```

### Live Orthanc tests: Linux/macOS

```bash
RUN_ORTHANC_INTEGRATION=1 python -m pytest -q
```

Live integration tests upload the repository's DICOM test fixture into the
configured Orthanc instance. Use a dedicated development instance.

### Frontend checks

```bash
cd frontend
npm ci
npm test
npm run build
npm audit
```

### Local verification record

During local verification on 2026-11-08:

| Check | Result |
|---|---|
| Python suite with checkpoint and live Orthanc | 75 passed |
| API suite with a deliberately missing checkpoint | 12 passed, 3 skipped |
| Ruff | Passed |
| Frontend tests | 3 passed |
| Frontend TypeScript and production build | Passed |
| Full frontend dependency audit | 0 known vulnerabilities reported |
| Frontend production dependency audit | 0 known vulnerabilities reported |
| Frontend Docker build | Passed |
| Frontend HTTP check | 200 |
| Browser classification, Grad-CAM, and review submission | Manually verified |

These results describe a particular local environment and dependency state.
They do not guarantee future installations, clinical performance, or
absence of security vulnerabilities.

A Starlette/TestClient deprecation warning remained in the local Python
test run.

## Continuous integration

Workflow:

```text
.github/workflows/python-ci.yml
```

Configured jobs:

1. Python tests and Ruff, with CPU PyTorch.
2. Frontend locked dependency installation, tests, TypeScript/build, and
   dependency audit.

CI does not provision the trained checkpoint or a live Orthanc instance.
Real-checkpoint tests explicitly skip when the artifact is missing, and
live Orthanc tests are disabled.

The workflow does not currently verify Docker builds, live browser E2E
flows, or deployment.

Check GitHub Actions for the actual result of the latest run; local passing
tests are not evidence that the remote workflow passed.

## Repository guide

```text
backend/                 FastAPI application, services, and API tests
frontend/                React/TypeScript workspace and frontend tests
ml/classification/       Classification workflow and configuration
ml/explainability/       Grad-CAM workflow
ml/segmentation/         Experimental MONAI segmentation workflow
ml/notebooks/            Segmentation research notebooks
tests/                   ML/data/helper tests
dicom/orthanc/            Local Orthanc configuration
scripts/                 Supporting DICOM and segmentation utilities
models/                  Separately provisioned model artifacts
reports/                 Locally generated evaluation artifacts
.github/workflows/       CI configuration
```

Model and evaluation artifacts may be absent from a fresh clone.

## Known gaps

- No authentication, authorization, or production access controls.
- No production TLS, secret management, backup procedure, or monitoring.
- Development credentials remain in local deployment configuration.
- No integrated DICOM viewer or segmentation UI.
- Poor segmentation performance.
- No validated per-label decision thresholds or calibration.
- No external or clinical model validation.
- Incomplete experiment provenance.
- No operational DVC pipeline or remote.
- No automated browser E2E or Docker-build CI.
- Reviewer references currently use uploaded filenames, not immutable
  image or inference identifiers.
- Dependency version ranges are not a fully locked Python environment.

## Privacy and responsible use

- Use public or appropriately de-identified images only.
- Do not upload private patient data to this demonstration.
- Do not commit datasets, real patient identifiers, private credentials,
  or access tokens.
- Public dataset labels are not equivalent to independently verified
  clinical ground truth.
- Grad-CAM does not establish causal or clinical reasoning.
- Reviewer approval is not clinical approval.
- Model scores must not guide care decisions.

## License and dataset terms

Review the repository's code license and each dataset's terms separately.
A code license does not grant permission to redistribute datasets or
trained artifacts.

Dataset access, provenance, attribution, and redistribution permissions
must be documented before sharing third-party data or models.