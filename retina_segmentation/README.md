# DeepVessel_Net-main

> Path isolation, sigmoid handling, and train/validation/test isolation have been remediated. See the repository `PRIVACY_REVIEW.md` for the evidence and remaining flags.

DeepVessel_Net-main is a PyTorch project for retinal blood vessel segmentation from fundus images. The repository combines a compact residual U-Net style model, a local train/test segmentation workflow, export and inference utilities, a Streamlit demo, and a set of research-oriented helper scripts for baseline comparison, STARE evaluation, profiling, and reporting.

This README is written as a full project handoff document. It explains what the project does, how it is structured, what is important, what is optional, and what should be improved next.

## 1. Project Summary

The main goal of the project is to segment retinal vessels from color fundus images. This task is useful in medical image analysis because vessel structure can support research and screening related to:

- diabetic retinopathy
- glaucoma
- hypertension
- vascular morphology analysis
- downstream ophthalmic feature extraction

The model predicts a binary vessel mask where vessel pixels are separated from the retinal background.

At a high level, the repository supports:

- model training
- evaluation on a local dataset split
- optional evaluation on STARE annotations
- image-level inference
- TorchScript and ONNX export
- a Streamlit-based demo app
- visualization and reporting utilities

## 2. Current Project State

This repository is more than a minimal training script. It already contains:

- a working segmentation model
- saved checkpoints in `outputs/checkpoints/`
- generated evaluation reports
- saved prediction masks
- a browser demo in `app.py`
- optional research scripts for comparisons and extra analysis

At the same time, the codebase is still partly in research-prototype form. The core path is usable, but a few scripts have drifted away from the main model contract and should be cleaned up.

Important current observations:

- The local workspace currently contains a dataset split under `Data/`.
- The current local split appears to be:
- 80 training images and 80 masks
- 20 test images and 20 masks
- The repo also includes a bundled `stare-DatasetNinja/` folder for external evaluation.
- A trained checkpoint `outputs/checkpoints/best_model.pth` is already present.

## 3. Main Idea of the Pipeline

The end-to-end flow of the project is:

1. Load retinal images and binary masks.
2. Resize and augment the data.
3. Train a segmentation model using Focal Tversky loss.
4. Save checkpoints during training.
5. Evaluate predictions using Dice, IoU, sensitivity, specificity, and accuracy.
6. Export the trained model for deployment.
7. Run single-image inference through CLI or Streamlit.

## 4. Core Architecture

The main model lives in `src/model.py`.

### Model Type

The architecture is a compact residual U-Net style network:

- encoder-decoder structure
- skip connections between encoder and decoder
- residual blocks inside each stage
- transposed convolutions for upsampling
- single-channel segmentation output

### Channel Progression

The model uses:

- encoder: `32 -> 64 -> 128`
- bottleneck: `256`
- decoder mirrors the encoder back down to `32`
- output head: `1` channel

### Important Note About Output

The model currently applies `sigmoid` inside `forward()` in `src/model.py`.

That means the model returns probabilities, not raw logits.

The audited training, evaluation, export, and inference consumers use those probabilities directly. They do not apply a second sigmoid. This probability-output contract is intentional because the current losses also consume probabilities; changing it would make existing checkpoints incompatible without a coordinated retraining and migration.

## 5. Loss Function and Metrics

### Training Loss

The main training loss is implemented in `src/loss.py`:

- `TverskyLoss`
- `FocalTverskyLoss`
- `DiceLoss`

The default training script uses `FocalTverskyLoss`, which is suitable for imbalanced segmentation tasks like thin-vessel detection where foreground pixels are sparse compared to background.

### Metrics

Metrics are implemented in `src/utils.py`:

- Dice coefficient
- IoU
- sensitivity
- specificity
- accuracy

These are used during validation and evaluation.

## 6. Data Pipeline

The data loading code is in `src/data_loader.py`.

### Expected Folder Layout

```text
Data/
  train/
    image/
    mask/
  test/
    image/
    mask/
```

### Data Assumptions

- image and mask filenames must match
- masks are loaded as grayscale
- masks are binarized to `0` and `1`
- images are resized to `512 x 512`

### Augmentation

Training augmentation includes:

- horizontal flip
- vertical flip
- rotation
- elastic transform
- brightness and contrast shift
- CLAHE
- random gamma

There is also support for:

- optional random crop
- optional extra green-channel input

However, in the current training configuration:

- random crop is disabled
- extra green channel is disabled

## 7. Training Workflow

The main training script is `src/train.py`.

### Default Settings

- image size: `512`
- batch size: `2`
- epochs: `50`
- optimizer: `AdamW`
- learning rate: `1e-4`
- weight decay: `1e-4`
- scheduler: `CosineAnnealingLR`
- AMP: enabled on CUDA

### Outputs Generated by Training

Training saves:

- `outputs/checkpoints/best_model.pth`
- `outputs/checkpoints/last_model.pth`
- `outputs/checkpoints/epoch_*.pth`
- TensorBoard logs under `outputs/logs/`

### Validation Isolation

`src/train.py` deterministically holds out 20% of `Data/train/` for validation using seed `42`. Training augmentation is applied only to the training subset. The training process never loads `Data/test/`; that directory is reserved for one-time final evaluation after model and threshold selection are complete.

## 8. Evaluation Workflow

### Local Evaluation

The main local evaluation script is `src/evaluate.py`.

It performs:

- checkpoint loading
- threshold sweep
- metric computation
- prediction saving
- visual comparison saving
- report generation

Artifacts are written to:

- `outputs/predictions/`
- `outputs/visuals/`
- `outputs/results_eval/`

### STARE Evaluation

External evaluation support is available in:

- `src/evaluate_stare.py`
- `src/stare_datasetninja.py`

This path reads DatasetNinja/Supervisely-style JSON annotations from the bundled STARE export and pairs them with images under:

```text
stare-DatasetNinja/stare-DatasetNinja/ds/
  img/
  ann/
```

This is useful if you want cross-dataset evaluation beyond the local `Data/` split.

## 9. Inference and Deployment

### CLI Inference

The main inference script is `src/infer.py`.

It supports:

- image preprocessing
- running a TorchScript model
- running an ONNX model
- thresholding output masks
- morphological cleanup

### Model Export

The export script is `src/export_model.py`.

It exports:

- TorchScript model
- ONNX model
- a deployment `config.json`

Output folder:

```text
outputs/deploy/
```

### Streamlit Demo

The interactive demo is in `app.py`.

It allows:

- image upload
- segmentation preview
- threshold choice for higher sensitivity
- output download

This is useful for demo presentations and quick testing of the trained checkpoint.

## 10. Current Performance Snapshot

[invalidated by split fix — must re-run]

[pending leakage-free re-evaluation — run evaluate.py and insert results]

The previous values were produced while `Data/test/` participated in checkpoint selection. They are not valid held-out test estimates and have been removed. Retrain with the isolated validation split, choose all model settings on validation only, then run one final evaluation on `Data/test/` before reporting any replacement values.

## 11. Strengths of the Project

The strongest parts of the repo are:

- clear end-to-end pipeline from training to demo
- lightweight and readable model architecture
- practical export and inference support
- useful generated outputs and checkpoints already present
- additional research tools for baselines and STARE evaluation

This makes the project suitable for:

- academic submission support
- project demonstrations
- portfolio presentation
- medical imaging experimentation
- model cleanup and extension work

## 12. Known Issues and Technical Risks

This is the most important section for anyone taking over the project.

### 12.1 Output Contract (Resolved)

The model applies sigmoid once internally and returns probabilities. The audited consumers threshold or score those probabilities directly; no double sigmoid is present in the current Python code.

### 12.2 Test Set Used as Validation (Resolved in Code)

`src/train.py` now selects checkpoints on a deterministic held-out subset of `Data/train/`. Existing metrics remain invalidated until retraining and final test evaluation are completed.

### 12.3 Inconsistent Path Handling

Some scripts use script-relative paths, while others use `os.getcwd()`. This makes behavior depend on where commands are run from.

### 12.4 Documentation Drift

The previous README and some app text describe features that do not fully match the current implementation, including:

- patch-based training as if currently active
- dropout in the architecture
- some metric/reporting expectations

### 12.5 Helper Script Drift

A few optional research/reporting scripts appear to be less reliable than the core train/evaluate path. They should be reviewed before being used as authoritative sources in a report or publication.

## 13. Important Files

If you want to keep the project focused, these are the core files:

- `src/model.py`
- `src/data_loader.py`
- `src/loss.py`
- `src/train.py`
- `src/evaluate.py`
- `src/infer.py`
- `src/export_model.py`
- `src/utils.py`
- `src/main.py`
- `app.py`
- `requirements.txt`
- `README.md`

These files define the main training, evaluation, inference, and deployment workflow.

## 14. Optional Research Files

These files are useful, but not required for the main project path:

- `src/evaluate_stare.py`
- `src/stare_datasetninja.py`
- `src/baseline_models.py`
- `src/compare_baselines.py`
- `src/impact_analysis.py`
- `src/realtime_profiler.py`
- `src/metrics_report.py`
- `src/visualize_results.py`

Keep them if you want research comparisons, profiling, or extra reporting.

## 15. Low-Priority or Removable Files

These are not essential to the core project:

- `plot_loss.py`
- `check.py`
- `s.py`
- `convert_model.py`
- `temp/`
- many generated files inside `outputs/` once you no longer need them

These can be removed later if you want to simplify the repository.

## 16. Recommended Cleanup Plan

If you continue improving this repo, the best cleanup order is:

### Step 1

Retrain with the held-out validation split, keeping `Data/test/` sealed until final evaluation.

### Step 2

Unify path handling:

- use script-relative project paths consistently
- avoid depending on `os.getcwd()`

### Step 3

Clean and align documentation:

- update README
- update app text
- update helper scripts
- remove duplicate or outdated utilities

## 17. How to Run the Project

### Install

```bash
python -m venv deepvessel_env
deepvessel_env\Scripts\activate
pip install -r requirements.txt
```

### Train

```bash
python src/train.py
```

### Evaluate

```bash
python src/evaluate.py
```

### Evaluate on STARE

```bash
python src/evaluate_stare.py
```

### Export Model

```bash
python src/export_model.py
```

### Run Streamlit App

```bash
streamlit run app.py
```

### Main Entry Point

You can also use:

```bash
python src/main.py --mode train
python src/main.py --mode evaluate
python src/main.py --mode visualize
python src/main.py --mode export
```

## 18. Suggested Use Cases for This Repository

This project is a good fit if you want to:

- present a complete medical image segmentation project
- study a residual U-Net style architecture
- demonstrate PyTorch training, evaluation, and deployment flow
- build a cleaner academic or portfolio submission
- extend the project with better validation, better losses, or new datasets

## 19. Final Handoff Notes

If you are taking over this project, the repo should be understood as:

- a working retinal vessel segmentation pipeline
- a project with useful real artifacts already generated
- a codebase that needs cleanup more than a complete rewrite
- a strong candidate for polishing into a final research or portfolio project

The main engineering priority is not adding new features first. The main priority is making the current evaluation, inference, and documentation fully consistent with the model's true behavior.
