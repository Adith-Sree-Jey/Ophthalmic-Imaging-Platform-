# Ophthalmic Imaging Cataract NS Grade Classifier

This source-only distribution includes no model weights or clinical datasets. Model paths in this document are destinations for separately authorized, user-supplied assets. Example API responses are illustrative, not measured validation results.

A full-stack clinical web application for cataract nuclear sclerosis grading from ophthalmic images.

The system combines three core pipelines:

- `Cataract Classifier`: a V7 multimodal classifier that predicts `NS1`, `NS2`, `NS3`, or `NS4`
- `autoDetect`: a local image-type classifier that automatically tags uploaded images as `anterior_segment`, `red_glow`, or `slit_lamp`
- `Pupil Cropping`: a YOLOv8-based crop pipeline that localizes the pupil/lens region before grading

## Overview

This application is built for a structured clinical workflow:

1. A clinician logs in to the web app.
2. The clinician uploads one or more eye images for a case.
3. The `autoDetect` module assigns each image to the correct modality.
4. The clinician can review or correct the detected image types.
5. The `Cataract Classifier` runs on the case, using pupil/lens localization from the `Pupil Cropping` pipeline.
6. The app returns grade, confidence, probability distribution, modality attention, cropped pupil image, and recommendation.
7. A PDF report is generated with patient details and deterministic clinical text.

## Tech Stack

- Frontend: React + Vite + Tailwind CSS
- Backend: FastAPI
- Authentication: JWT
- Database: `ophthalmic_database` is permanent; the app and Alembic must always share the same `MSSQL_DATABASE` value from `.env`.
- Report generation: ReportLab
- Cataract model: V7 multimodal inference engine
- Auto image type detection: local MobileNetV3 classifier
- Pupil cropping: YOLOv8

## Core Modules

### Cataract Classifier

The cataract grading pipeline is handled by the backend service in [backend/classifier.py](./backend/classifier.py).

It loads the V7 multimodal inference engine from the main repository root:

- `model/inference_multimodal.py`
- `checkpoints/best_acc_model.pth`
- `checkpoints/best_qwk_model.pth`

The classifier returns:

- predicted grade: `NS1`, `NS2`, `NS3`, `NS4`, or `Unknown`
- confidence score
- class probabilities
- modality attention weights
- review flags for low-confidence cases
- pupil/lens crop image
- recommendation text

The current result payload also includes:

- `patient_name`
- `mri_number`
- `case_id`
- original modality images as base64

### autoDetect

The image auto-detection pipeline is implemented in [backend/image_classifier.py](./backend/image_classifier.py).

Current behavior:

- local inference only
- no LLM
- no external API
- no internet required for type detection

The detector uses:

- MobileNetV3-Small
- weights file: `backend/model/best_model.pth`
- class map: `backend/model/class_map.json`

Supported output classes:

- `anterior_segment`
- `red_glow`
- `slit_lamp`
- `unknown`

This module powers the `POST /detect-image-types` endpoint and helps the upload UI auto-tag images before the main classification step.

### Pupil Cropping

The pupil crop stage is handled by YOLOv8 and is used by the V7 grading pipeline.

The backend looks for YOLO weights in this order:

1. `pupil_crop/Yolo/weights/best.pt`
2. `runs/detect/pupil_detector/weights/best.pt`

The crop is used to:

- localize the clinically relevant pupil/lens region
- support grading inference
- provide the cropped image returned in the API response
- include the crop in generated reports

## Repository Layout

```text
cataract_ns_grading_v3/
|-- ophthalmic-web/
|   |-- backend/
|   |   |-- auth.py
|   |   |-- classifier.py
|   |   |-- image_classifier.py
|   |   |-- main.py
|   |   |-- pdf_report.py
|   |   |-- report_text_engine.py
|   |   |-- requirements.txt
|   |   `-- model/
|   |       |-- best_model.pth
|   |       `-- class_map.json
|   |-- frontend/
|   |   |-- src/
|   |   |   |-- components/
|   |   |   |-- pages/
|   |   |   |-- api.js
|   |   |   `-- App.jsx
|   |   |-- package.json
|   |   `-- tailwind.config.js
|   `-- README.md
|-- model/
|   |-- inference_multimodal.py
|   |-- model_multimodal.py
|   `-- train_multimodal.py
|-- checkpoints/
|   |-- best_acc_model.pth
|   `-- best_qwk_model.pth
|-- autoDetect/
|   |-- train.py
|   |-- predict.py
|   `-- image_classifier_local.py
`-- pupil_crop/
    |-- detector/
    |-- processor/
    `-- Yolo/weights/best.pt
```

## Web Application Features

- JWT login with a database-backed admin or doctor account
- auto image-type detection before grading
- manual correction of detected image types
- patient metadata fields:
  - `Patient Name`
  - `MRI Number`
  - `Case ID`
- multimodal case submission
- result dashboard with:
  - grade
  - confidence
  - class probabilities
  - modality attention
  - cropped pupil image
  - recommendations
- PDF report export
- clear/reset workflow for the next patient

## Admin Setup

Before logging in, set `ADMIN_USERNAME` and `ADMIN_PASSWORD`, then run:

```powershell
cd .\backend
python scripts\seed_admin.py
```

## API Endpoints

### `POST /token`

Request body:

```text
Content-Type: application/x-www-form-urlencoded

username=<your_admin_username>&password=<your_admin_password>
```

Response:

```json
{
  "access_token": "jwt_token_here",
  "token_type": "bearer"
}
```

### `POST /detect-image-types`

Protected endpoint.

Accepts:

- multipart form data
- field name: `files`

Returns one result per image:

```json
[
  {
    "filename": "eye_001.jpg",
    "detected_type": "anterior_segment",
    "confidence": "high",
    "reasoning": "Local classifier: anterior_segment (96.2% confidence)"
  }
]
```

### `POST /classify`

Protected endpoint.

Accepts:

- `files`
- `types`
- `case_id`
- `patient_name`
- `mri_number`

Notes:

- exactly one image must be assigned as `anterior_segment`
- `red_glow` and `slit_lamp` are optional
- duplicate modality assignments are rejected

Example response:

```json
{
  "case_id": "CASE-001",
  "patient_name": "Ravi Kumar",
  "mri_number": "MRI-2024-00891",
  "detected": true,
  "grade": "NS3",
  "confidence": 0.84,
  "confidence_pct": "84%",
  "probabilities": {
    "NS1": 0.01,
    "NS2": 0.08,
    "NS3": 0.84,
    "NS4": 0.07
  },
  "attention": {
    "anterior": 0.62,
    "red_glow": 0.21,
    "slit_lamp": 0.17
  },
  "bbox": [232, 85, 1060, 987],
  "crop_image_base64": "<base64>",
  "anterior_segment_base64": "<base64>",
  "red_glow_base64": "<base64>",
  "slit_lamp_base64": "<base64>",
  "recommendation": "Moderate cataract. Consider surgical referral.",
  "severity": "Moderate",
  "model_info": "V7_best | YOLOv8 pupil crop"
}
```

### `GET /report`

Protected endpoint.

Returns a downloadable PDF report for the latest classified case in the current user session.

## Environment Configuration

Create an `.env` file in:

```text
ophthalmic-web/.env
```

Example:

```env
JWT_SECRET=your_jwt_secret_here
CATARACT_CHECKPOINT_PATH=checkpoints/best_qwk_model.pth
LOCAL_MEDGEMMA_PATH=backend/models/medgemma-4b-it
```

Notes:

- the backend loads `.env` from the `ophthalmic-web` root
- changing `JWT_SECRET` invalidates previously issued login tokens

## Installation

### Backend

```powershell
python -m pip install -r requirements.txt
python -m pip install -e .\glaucoma -e .\retina_segmentation -e .\model -e .\autoDetect -e .\data_pipeline -e .\pupil_crop
cd .\ophthalmic-web\backend
python -m uvicorn main:app --reload --port 8000
```

Run these commands from the repository root. The root requirements file is the complete install entrypoint; component manifests remain available for scoped development. See `DEPENDENCIES.md` for version-pin provenance and unresolved pin TODOs.

Run the backend test suite with:

```powershell
cd .\ophthalmic-web\backend
pytest tests\ -v
```

### Frontend

```powershell
cd .\ophthalmic-web\frontend
npm install
npm run dev
```

Frontend URL:

```text
http://localhost:5173
```

Backend URL:

```text
http://localhost:8000
```

## Runtime Workflow

1. Open the login page.
2. Sign in with the demo account.
3. Enter patient name, MRI number, and case ID.
4. Upload one or more clinical images.
5. Wait for `autoDetect` to classify the image types.
6. Correct any modality tags if needed.
7. Click `Run Classification`.
8. Review the result panel.
9. Download the PDF report.
10. Click `Clear Images` to reset the form for the next patient.

## Notes About Reports

The PDF layer uses:

- [backend/pdf_report.py](./backend/pdf_report.py)
- [backend/report_text_engine.py](./backend/report_text_engine.py)

Clinical report text is deterministic and generated from model outputs:

- grade
- confidence
- probabilities
- attention weights

No LLM is used for report text generation.

## Troubleshooting

### `401 Unauthorized` on classify or detect

This usually means the browser does not have a valid JWT token.

Try:

1. log in again
2. clear site storage for `localhost:5173`
3. restart the backend if you changed `JWT_SECRET`

### Login succeeds but dashboard is blank

Use:

- `http://localhost:5173/login`

The frontend now redirects correctly between `/login` and `/dashboard`. If an old tab still behaves oddly, refresh once after restarting the dev server.

### `POST /detect-image-types` returns `unknown`

Check:

- `backend/model/best_model.pth`
- `backend/model/class_map.json`
- local `torch` / `torchvision` installation

### V7 classifier startup fails

The trained cataract weights are external runtime assets and are not supplied by dependency installation. Provide one of the real trained files:

- `checkpoints/best_acc_model.pth` or `checkpoints/best_qwk_model.pth`
- alternatively, set `CATARACT_CHECKPOINT_PATH` to the real checkpoint location (relative paths resolve from `ophthalmic-web/`)
- `model/inference_multimodal.py`
- `pupil_crop/Yolo/weights/best.pt`

When neither checkpoint exists, startup and `/workspace-status` report the cataract module as unavailable, and classification jobs fail with an explicit checkpoint error. Do not create placeholder or randomly initialized weights.

## Summary

This project is not just a web UI. It packages three connected vision pipelines into one clinical workflow:

- `Cataract Classifier` for NS grade prediction
- `autoDetect` for modality assignment
- `Pupil Cropping` for localization and crop extraction

Together they support an end-to-end cataract screening and reporting flow from upload to report export
