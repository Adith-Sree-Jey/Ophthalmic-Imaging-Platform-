# Python Dependencies

The repository now has one full-install entrypoint:

```powershell
python -m pip install -r requirements.txt
```

The root manifest includes these component manifests:

- `model/requirements.txt` for the multimodal cataract model (`timm`)
- `retina_segmentation/requirements.txt` for retinal segmentation and shared ML packages
- `ophthalmic-web/backend/requirements.txt` for the API, database, reporting, and MedGemma runtime

After installing dependencies, install the local packages from the repository root:

```powershell
python -m pip install -e .\model -e .\retina_segmentation -e .\glaucoma -e .\autoDetect -e .\data_pipeline -e .\pupil_crop
```

`ophthalmic-web/requirements_additions.txt` was removed. Its valid packages were folded into the backend manifest; the invalid shell command embedded in that file is gone.

Exact pins were added only for the newly audited imports whose working versions were observable in the remediation environment (`timm`, `transformers`, `accelerate`, and `bitsandbytes`). Existing lower bounds were retained. Dependencies for which the repository recorded no version are marked `TODO(pin)` instead of assigning an invented version.

PyTorch accelerator builds can require a platform-specific package index. The repository retains its existing minimum versions and does not guess a CUDA build. Select the correct PyTorch build for the deployment host before installing the full manifest if GPU execution is required.
