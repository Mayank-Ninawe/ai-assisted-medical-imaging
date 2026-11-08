# \# AI-Assisted Medical Imaging and Diagnosis System

# 

# > \*\*For educational purposes only, not for clinical use.\*\*

# 

# \## Project status

# 

# \*\*Status: Implemented educational prototype with verified limits.\*\*

# 

# This repository is for an internship project to build an educational demonstration of AI-assisted medical-image workflows. The current prototype includes a tested Python/ML workflow, FastAPI service, React frontend, local SQLite review persistence, and DICOM/Orthanc integration boundaries. The local Docker Compose stack and Orthanc flow were live-validated on 2026-11-08.

# 

# \## Project overview

# 

# The planned system will demonstrate two separate machine-learning tasks:

# 

# 1\. Multi-label classification of chest radiographs using a suitable public dataset and a pretrained image model.

# 2\. 3D medical-image segmentation using a suitable Medical Segmentation Decathlon task and MONAI.

# 

# The planned application will also explore DICOM study handling through Orthanc, image validation and preprocessing, Grad-CAM visualizations, a FastAPI backend, a React and TypeScript viewer, persistent human-review audit records, automated testing, and containerized deployment.

# 

# The classifier and segmenter are separate educational workflows. This project is not a medical device and must not be used to diagnose, treat, or make decisions about a person’s health.

# 

# \## Current capabilities and verified limits

# 

# \- Validate supported image and DICOM inputs and reject unsupported or malformed files safely.

# \- Train and evaluate a transfer-learning chest-radiograph classifier, if dataset access and available compute make this feasible.

# \- Generate Grad-CAM visualizations for supported classification models.

# \- Develop a MONAI-based 3D segmentation workflow for one selected MSD task, subject to feasibility.

# \- Use Orthanc for a local DICOM demonstration where practical.

# \- Provide an API and web interface for supported project workflows.

# \- Record human review actions in persistent storage.

# \- Document test results, dataset limitations, privacy considerations, bias, and known gaps.

# 

# The classification checkpoint, Grad-CAM path, Docker Compose stack, and local Orthanc flow are exercised by automated tests and live local smoke tests. Segmentation remains a research prototype: the corrected held-out evaluation and bounded pilot showed poor Dice and severe over-segmentation, so it must not be presented as reliable.

# 

# \## Proposed technology areas

# 

# Technology and dataset choices are provisional until the development environment, dataset access, and project schedule have been checked.

# 

# \- \*\*Classification:\*\* PyTorch and a suitable pretrained image model.

# \- \*\*Segmentation:\*\* MONAI and a 3D U-Net-based workflow.

# \- \*\*DICOM:\*\* pydicom and Orthanc, if feasible.

# \- \*\*Backend:\*\* FastAPI and persistent database storage.

# \- \*\*Frontend:\*\* React, TypeScript, and Vite.

# \- \*\*Data and experiments:\*\* DVC where a practical storage setup is available.

# \- \*\*Development and deployment:\*\* Docker and Docker Compose, followed by a suitable hosted deployment if feasible.

# 

# \## Data, privacy, and limitations

# 

# \- Do not commit medical images, patient identifiers, credentials, access tokens, or private configuration to this repository.

# \- Use only data whose access terms and intended use have been reviewed.

# \- Keep large datasets and model artifacts outside normal Git history unless a suitable storage mechanism is deliberately configured.

# \- Public dataset labels may be incomplete, uncertain, or derived from reports; they are not equivalent to clinical ground truth.

# \- Grad-CAM highlights model-dependent image regions. It does not prove that a prediction is correct or explain clinical reasoning.

# \- Results from public datasets do not establish clinical validity or generalization to other populations or imaging systems.

# \- Any completed evaluation must state the dataset, split method, metrics, and limitations. Results will not be invented or presented as clinical evidence.

# 

# \## Development approach

# 

# The project will be built in documented phases. Each phase should include:

# 

# \- A specific goal and the files changed.

# \- Step-by-step commands and setup instructions.

# \- Relevant automated or manual tests.

# \- Expected results and known limitations.

# \- A Git checkpoint after meaningful verified work.

# 

# The implementation plan and testing checklist will be added to the `docs/` area as the project progresses.

# 

# \## Local setup

# 

# The local working copy is planned at:

# 

# ```text

# D:\\Nexvia-Internship\\ai-assisted-medical-imaging

# ```

# 

# Install Python dependencies with `python -m pip install -e ".[dev,backend,ml]"`, then run `python -m pytest -q` and `python -m ruff check backend ml tests`. The frontend can be validated with `Set-Location frontend; npm ci`, `npm test`, and `npm run build`.
#
# A local container configuration is available through `docker compose config` and `docker compose up --build`. It expects a classification checkpoint at `models/classification/best_classifier.pt`; model and dataset files are intentionally mounted from the host rather than copied into images. The stack was live-validated locally on 2026-11-08, but this is not evidence of production deployment or clinical readiness.

# 

# \## License and dataset terms

# 

# The repository's code license and any dataset terms must be reviewed and documented separately. A code license does not grant permission to redistribute or use third-party medical datasets.
