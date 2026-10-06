import sys
import os
from pathlib import Path
import json
import torch
import numpy as np
from retina_segmentation.src.model import DeepVesselNet
BASE_DIR = Path(__file__).resolve().parents[1]
CKPT = BASE_DIR / "outputs" / "checkpoints" / "best_model.pth"
EXPORT_DIR = BASE_DIR / "outputs" / "deploy"
os.makedirs(EXPORT_DIR, exist_ok=True)

IMG_SIZE = 512  # keep consistent with training

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepVesselNet(in_channels=3, out_channels=1).to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device))
    model.eval()

    # Dummy input
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=device)

    # ----- TorchScript -----
    traced = torch.jit.trace(model, dummy)
    ts_path = EXPORT_DIR / "deepvesselnet.torchscript.pt"
    traced.save(ts_path)
    print(f"✅ Saved TorchScript: {ts_path}")

    # ----- ONNX -----
    onnx_path = EXPORT_DIR / "deepvesselnet.onnx"
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=["input"], output_names=["logits"],
        opset_version=17,
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}
    )
    print(f"✅ Saved ONNX: {onnx_path}")

    # ----- Config -----
    cfg = {
        "img_size": IMG_SIZE,
        "normalize": {"mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]},
        "threshold": 0.5,
        "postprocess": {"morph_open": True, "morph_close": True, "kernel": 3}
    }
    with open(EXPORT_DIR / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    print("✅ Saved config.json")

if __name__ == "__main__":
    main()
