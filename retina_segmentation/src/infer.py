import os
import sys
import json
from functools import lru_cache
from pathlib import Path
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image

from retina_segmentation.src.utils import clean_prediction

# Optional: ONNX Runtime support
try:
    import onnxruntime as ort
except ImportError:
    ort = None
from retina_segmentation.src.model import DeepVesselNet

MODULE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_EXPORT_DIR = MODULE_DIR / "outputs" / "deploy"

# ===============================
# CONFIG LOAD
# ===============================
def load_config(export_dir):
    with open(Path(export_dir) / "config.json", "r") as f:
        return json.load(f)

# ===============================
# IMAGE PREPROCESSING
# ===============================
def preprocess(img_bgr, img_size, mean, std):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img_rgb, (img_size, img_size))
    img = img.astype(np.float32) / 255.0
    img = (img - np.array(mean)) / np.array(std)
    img = np.transpose(img, (2, 0, 1))  # CHW
    img = np.expand_dims(img, 0)        # NCHW
    return img

# ===============================
# POSTPROCESSING
# ===============================
def postprocess(prob, threshold):
    """
    Threshold probability map to binary mask.
    Expects values in [0, 1] (traced/ONNX models include sigmoid in the graph).
    """
    prob = np.asarray(prob, dtype=np.float64)
    mask = (prob > threshold).astype(np.uint8) * 255
    return mask

# ===============================
# TORCHSCRIPT BACKEND
# ===============================
def run_torchscript(ts_path, img_tensor):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.jit.load(ts_path, map_location=device)
    model.eval()
    with torch.no_grad():
        inp = torch.from_numpy(img_tensor).to(device)
        out = model(inp).cpu().numpy()
    return out

# ===============================
# ONNX BACKEND
# ===============================
def run_onnx(onnx_path, img_tensor):
    if ort is None:
        raise RuntimeError("onnxruntime not installed. pip install onnxruntime")
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(onnx_path, providers=providers)
    out = sess.run(None, {"input": img_tensor.astype(np.float32)})[0]
    return out

# ===============================
# MAIN CLI ENTRY POINT
# ===============================
def main():
    if len(sys.argv) < 3:
        print("Usage: python infer.py <path_to_image> <output_mask_path> [export_dir]")
        sys.exit(1)

    img_path = sys.argv[1]
    out_path = sys.argv[2]
    export_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_EXPORT_DIR

    cfg = load_config(export_dir)
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"Error: Could not read image: {img_path}")
        sys.exit(1)
    img_tensor = preprocess(img_bgr, cfg["img_size"], cfg["normalize"]["mean"], cfg["normalize"]["std"])

    backend = os.environ.get("DV_BACKEND", "torchscript").lower()
    if backend == "onnx":
        logits = run_onnx(Path(export_dir) / "deepvesselnet.onnx", img_tensor)
    else:
        logits = run_torchscript(Path(export_dir) / "deepvesselnet.torchscript.pt", img_tensor)

    # Model output is (1, 1, H, W); squeeze to (H, W)
    prob = np.squeeze(logits)

    mask = postprocess(prob, cfg["threshold"])
    mask = clean_prediction(mask)

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(out_path, mask)
    print(f"✅ Saved mask to: {out_path}")

# ===============================
# WEB APP HELPER FUNCTION
# ===============================
def load_model(model_path):
    model = DeepVesselNet()
    checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
    
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
    else:
        raise RuntimeError("Invalid checkpoint format.")
    
    return model

@lru_cache(maxsize=1)
def _get_default_model():
    model_path = MODULE_DIR / "outputs" / "checkpoints" / "best_model.pth"
    return load_model(model_path)

def predict_single_image(image_path, model, threshold=0.5):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])

    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor).squeeze().cpu().numpy()

    mask = (output > threshold).astype(np.uint8) * 255
    mask = clean_prediction(mask)
    return mask

def predict_probability_map(image_path: str) -> np.ndarray:
    """
    Run inference and return raw sigmoid probability map.
    No thresholding applied. Returns float32 ndarray shape (H, W), values in [0, 1].
    This is the pre-threshold output for real-time frontend thresholding.
    """
    model = _get_default_model()
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])

    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        probability_map = model(input_tensor).squeeze().detach().cpu().numpy().astype(np.float32)

    return probability_map

# ===============================
# CALL MAIN IF RUN DIRECTLY
# ===============================
if __name__ == "__main__":
    main()
