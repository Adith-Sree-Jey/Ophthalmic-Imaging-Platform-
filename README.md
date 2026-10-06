# Ophthalmic Imaging Ophthalmology Platform

> Source-only distribution: no pretrained weights, clinical datasets, or company branding are included. All model paths below are destinations for separately authorized, user-supplied assets. Inference is unavailable until the required weights are supplied.

This repository contains a multi-module ophthalmic imaging platform built around a clinical web application plus the supporting machine learning pipelines used for training and inference.

The main product surface is the `ophthalmic-web` application, which combines:

- cataract nuclear sclerosis grading
- glaucoma classification
- retina vessel segmentation
- patient history and longitudinal tracking
- PDF report generation

The repository includes standalone training/research code. All model weights must be supplied separately; see [DATA_SOURCES.md](./DATA_SOURCES.md) and [PRIVACY_REVIEW.md](./PRIVACY_REVIEW.md).

## Repository Overview

The codebase is split into two broad layers:

1. Product application
   - `ophthalmic-web/frontend`: React + Vite user interface
   - `ophthalmic-web/backend`: FastAPI backend, inference orchestration, reporting, and patient-history APIs
2. Model and research modules
   - `model`: multimodal cataract grading model training and inference
   - `autoDetect`: image-modality classifier for `anterior_segment`, `red_glow`, and `slit_lamp`
   - `pupil_crop`: pupil/lens localization utilities and YOLO-based crop flow
   - `glaucoma`: glaucoma model, reporting, and PDF utilities
   - `retina_segmentation`: retinal blood vessel segmentation project
   - `data_pipeline`, `report`, `metadata`: helper scripts and data assets

## Main Clinical Workflow

The web app supports the following workflow:

1. User signs in to the dashboard.
2. Patient details are entered.
3. One or more ophthalmic images are uploaded.
4. The `autoDetect` model assigns image modality labels.
5. The user reviews or corrects the labels if needed.
6. The selected analysis module runs:
   - cataract grading
   - glaucoma classification
   - retina vessel segmentation
7. Results are displayed in the UI.
8. Reports can be generated and downloaded.
9. Patient visits are stored for history and longitudinal review when the database is configured.

## Core Modules

### 1. Cataract Grading

The cataract module is a multimodal grading pipeline for `NS1` to `NS4`.

Inputs can include:

- `anterior_segment`
- `red_glow`
- `slit_lamp`

The backend inference path is centered on:

- [ophthalmic-web/backend/classifier.py](./ophthalmic-web/backend/classifier.py)
- [model/inference_multimodal.py](./model/inference_multimodal.py)
- [model/model_multimodal.py](./model/model_multimodal.py)

Supporting assets:

- `ophthalmic-web/checkpoints/best_acc_model.pth` or `best_qwk_model.pth` — **user-supplied; not included**. Set `CATARACT_CHECKPOINT_PATH` or place an authentic checkpoint at that location. [The backend validates this at startup](./ophthalmic-web/backend/classifier.py); the asset status is recorded in [PRIVACY_REVIEW.md](./PRIVACY_REVIEW.md).
- `pupil_crop/Yolo/weights/best.pt` — user-supplied crop-model asset; no weights are included.

Returned outputs include:

- predicted grade
- confidence and class probabilities
- modality attention weights
- review flags
- pupil crop image
- Grad-CAM visual explanations
- recommendation and severity text

### 2. autoDetect

The upload pipeline uses a local classifier to determine image type before cataract inference.

Main file:

- [ophthalmic-web/backend/image_classifier.py](./ophthalmic-web/backend/image_classifier.py)

Model assets:

- `ophthalmic-web/backend/model/best_model.pth`
- `ophthalmic-web/backend/model/class_map.json`

Supported classes:

- `anterior_segment`
- `red_glow`
- `slit_lamp`
- `unknown`

### 3. Glaucoma Classification

The glaucoma module is integrated into the same web application and supports prediction plus PDF report generation.

Primary files:

- [ophthalmic-web/backend/glaucoma_service.py](./ophthalmic-web/backend/glaucoma_service.py)
- [glaucoma/src/infer.py](./glaucoma/src/infer.py)
- [glaucoma/src/report_service.py](./glaucoma/src/report_service.py)
- [glaucoma/src/pdf_generator.py](./glaucoma/src/pdf_generator.py)

Primary runtime checkpoint:

- `glaucoma/outputs/checkpoints/glaucoma_mobilenetv3.pth` — **user-supplied; not included at the runtime path expected by `glaucoma_service.py`**. Its training-dataset source remains `MISSING` in [DATA_SOURCES.md](./DATA_SOURCES.md).

### 4. Retina Segmentation

The retina module performs retinal blood vessel segmentation and is also maintained as a standalone research project.

Main files:

- [ophthalmic-web/backend/retina_segmentation_service.py](./ophthalmic-web/backend/retina_segmentation_service.py)
- [retina_segmentation/src/infer.py](./retina_segmentation/src/infer.py)
- [retina_segmentation/src/train.py](./retina_segmentation/src/train.py)
- [retina_segmentation/src/model.py](./retina_segmentation/src/model.py)

This module supports:

- probability-map generation
- segmentation mask creation
- overlay rendering
- standalone training and evaluation workflows

## Application Architecture

### Frontend

The frontend is built with:

- React
- Vite
- Tailwind CSS
- React Router

Key files:

- [ophthalmic-web/frontend/src/App.jsx](./ophthalmic-web/frontend/src/App.jsx)
- [ophthalmic-web/frontend/src/pages/Dashboard.jsx](./ophthalmic-web/frontend/src/pages/Dashboard.jsx)
- [ophthalmic-web/frontend/src/pages/PatientHistory.jsx](./ophthalmic-web/frontend/src/pages/PatientHistory.jsx)
- [ophthalmic-web/frontend/src/api.js](./ophthalmic-web/frontend/src/api.js)

### Backend

The backend is a FastAPI service that provides:

- authentication
- cataract classification APIs
- glaucoma APIs
- retina segmentation APIs
- report generation
- patient history and statistics endpoints

Key files:

- [ophthalmic-web/backend/main.py](./ophthalmic-web/backend/main.py)
- [ophthalmic-web/backend/auth.py](./ophthalmic-web/backend/auth.py)
- [ophthalmic-web/backend/database.py](./ophthalmic-web/backend/database.py)
- [ophthalmic-web/backend/pdf_report.py](./ophthalmic-web/backend/pdf_report.py)

### Persistence

Patient history is stored in SQL Server through SQLAlchemy models:

- `patients`
- `visits`
- `classifications`

Database setup notes are in:

- [ophthalmic-web/MSSQL_SETUP.md](./ophthalmic-web/MSSQL_SETUP.md)

## Repository Layout

```text
Capstone_Final/
|-- ophthalmic-web/
|   |-- backend/
|   |-- frontend/
|   |-- MSSQL_SETUP.md
|   `-- README.md
|-- autoDetect/
|-- checkpoints/
|-- data_pipeline/
|-- glaucoma/
|-- metadata/
|-- model/
|-- output/
|-- pupil_crop/
|-- report/
|-- retina_segmentation/
|-- Sample_image/
|-- glaucoma_classification_fixed.py
`-- README.md
```

## Setup

## Prerequisites

Recommended local environment:

- Python 3.10+
- Node.js 18+
- SQL Server + ODBC Driver 17 for SQL Server
- CUDA-capable GPU for faster inference and report generation

## Backend Setup

From the backend folder:

```powershell
cd .\ophthalmic-web\backend
python -m pip install -r ..\..\requirements.txt
```

The active backend code also relies on ML libraries used by the integrated modules, such as:

- `torch`
- `torchvision`
- `albumentations`
- `pandas`
- `matplotlib`

If those are not already installed in the environment, install them before starting the backend.

Create or update:

```text
ophthalmic-web/.env
```

Typical variables:

```env
JWT_SECRET=your_secret_here
MSSQL_SERVER=localhost\SQLEXPRESS
MSSQL_DATABASE=ophthalmic_imaging_cataract
MSSQL_DRIVER=ODBC Driver 17 for SQL Server
MSSQL_USERNAME=
MSSQL_PASSWORD=
MSSQL_ENCRYPT=no
```

Run the backend:

```powershell
python -m uvicorn main:app --reload --port 8000
```

### Frontend Setup

From the frontend folder:

```powershell
cd .\ophthalmic-web\frontend
npm install
npm run dev
```

Default URLs:

- frontend: `http://localhost:5173`
- backend: `http://127.0.0.1:8000`

## Standalone Module Setup

Some subprojects can also be used independently:

- `retina_segmentation`: standalone segmentation training, evaluation, export, and Streamlit demo
- `model`: standalone cataract model training and evaluation
- `autoDetect`: standalone modality-classifier training and prediction
- `glaucoma`: standalone glaucoma training/inference utilities

Refer to each module's local files before running long training jobs.

## Key API Endpoints

Main backend routes currently include:

- `POST /auth/login`
- `POST /detect-image-types`
- `POST /classify`
- `POST /classify-batch`
- `GET /report`
- `POST /report-batch`
- `POST /retina/probability`
- `POST /retina-segmentation`
- `POST /glaucoma/predict`
- `POST /glaucoma/report`
- `GET /history/{mri_number}`
- `GET /longitudinal/{mri_number}/{eye_side}`
- `GET /patients`
- `GET /stats/overview`

## Demo Notes

The current app still includes a demo login flow for local development. Update authentication before using this in a shared or production deployment.

## Current State

This repository is best understood as a combined product-and-research workspace:

- The web application path is functional and integrates multiple models into a single workflow.
- The ML modules are usable, but some parts are still research-oriented.
- A working directory may contain ignored local datasets, checkpoints, temporary outputs, and generated assets; a clean tracked checkout does not supply the clinical training datasets or cataract/glaucoma runtime checkpoints listed above.
- SQL-backed history works when the MSSQL environment is correctly configured.

## Known Limitations

- authentication is still demo-oriented
- some runtime dependencies are spread across multiple modules instead of one fully unified environment file
- in-memory session result caching is used for report generation
- training and inference require separately supplied datasets and checkpoints
- the retina segmentation project still reflects research workflow conventions in parts of its training/evaluation path

## Recommended Cleanup Direction

If you are continuing this project, the highest-value improvements are:

1. replace demo authentication with real user management
2. consolidate dependencies into reproducible environment files
3. persist report state outside process memory
4. separate deployable app code from research assets and datasets
5. add automated API and UI smoke tests
6. standardize documentation across modules

## Additional Documentation

Module-specific references:

- [ophthalmic-web/README.md](./ophthalmic-web/README.md)
- [retina_segmentation/README.md](./retina_segmentation/README.md)
- [ophthalmic-web/MSSQL_SETUP.md](./ophthalmic-web/MSSQL_SETUP.md)
