import os
import time
import argparse
from collections import deque

import cv2
import numpy as np
import matplotlib.pyplot as plt
import psutil

try:
    import pynvml
    pynvml.nvmlInit()
    _NVML_OK = True
except Exception:
    _NVML_OK = False

try:
    import torch
    _TORCH_OK = True
except Exception:
    _TORCH_OK = False


def get_gpu_util():
    """Returns (gpu_util%, gpu_mem_util%) if NVML available else (None, None)."""
    if not _NVML_OK:
        return None, None
    try:
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        mem_util = int(100 * (mem.used / max(mem.total, 1)))
        return util, mem_util
    except Exception:
        return None, None


def iter_images(img_dir, resize=(512, 512)):
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    paths = [os.path.join(img_dir, f) for f in sorted(os.listdir(img_dir)) if f.lower().endswith(exts)]
    if not paths:
        raise FileNotFoundError(f"No images found in: {img_dir}")

    while True:  # loop forever to simulate a stream
        for p in paths:
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            if resize is not None:
                img = cv2.resize(img, resize, interpolation=cv2.INTER_AREA)
            yield img


def preprocess_to_tensor(gray_img):
    """
    Basic preprocessing: [H,W] -> [1,1,H,W], float32 in [0,1].
    Replace this with your CLAHE/normalization if needed.
    """
    x = gray_img.astype(np.float32) / 255.0
    x = np.expand_dims(np.expand_dims(x, axis=0), axis=0)  # N,C,H,W
    return x


def load_model_torchscript(ts_path, device="cpu"):
    if not _TORCH_OK:
        raise RuntimeError("PyTorch not available. Install torch or switch to ONNX runtime approach.")
    dev = torch.device(device)
    m = torch.jit.load(ts_path, map_location=dev)
    m.eval()
    return m, dev


def run_inference(model, dev, x_np):
    """
    TorchScript inference.
    """
    import torch
    x = torch.from_numpy(x_np).to(dev)
    with torch.no_grad():
        y = model(x)
    # If output is tuple/list, take first
    if isinstance(y, (tuple, list)):
        y = y[0]
    return y


def main():
    # Script-based paths: work when run from project root as python src/realtime_profiler.py
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _base_dir = os.path.abspath(os.path.join(_script_dir, ".."))
    _data_root = os.path.join(_base_dir, "Data")
    _default_img_dir = os.path.join(_data_root, "test", "image")
    if not os.path.isdir(_default_img_dir):
        _fallback = os.path.join(_data_root, "train", "image")
        if os.path.isdir(_fallback):
            _default_img_dir = _fallback
    _default_out_dir = os.path.join(_base_dir, "outputs", "realtime_profiler")

    ap = argparse.ArgumentParser(
        description="Real-time FPS/latency profiler. Run from project root: python src/realtime_profiler.py"
    )
    ap.add_argument("--img_dir", default=_default_img_dir, help=f"Folder with input images (default: Data/test/image).")
    ap.add_argument("--torchscript", default=None, help="Path to TorchScript model (.pt). If not provided, runs 'simulation-only' timing.")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="cpu or cuda (needs torch + CUDA).")
    ap.add_argument("--seconds", type=int, default=60, help="How long to run the real-time profiling.")
    ap.add_argument("--warmup", type=int, default=20, help="Warmup iterations (not logged).")
    ap.add_argument("--window", type=int, default=200, help="How many recent points to show in the live graph.")
    ap.add_argument("--out_dir", default=_default_out_dir, help="Output folder for PNG + CSV.")
    ap.add_argument("--resize", type=int, nargs=2, default=[512, 512], help="Resize W H, e.g., 512 512")
    args = ap.parse_args()

    # Resolve to absolute paths so it works from any cwd
    args.img_dir = os.path.abspath(args.img_dir)
    args.out_dir = os.path.abspath(args.out_dir)

    if not os.path.isdir(args.img_dir):
        raise FileNotFoundError(
            f"Image dir not found: {args.img_dir}\n"
            "Create Data/test/image and add images, or pass --img_dir <path> (e.g. --img_dir Data/train/image)."
        )
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Image dir: {args.img_dir}")
    print(f"Output dir: {args.out_dir}")
    csv_path = os.path.join(args.out_dir, "realtime_metrics.csv")
    fig_path = os.path.join(args.out_dir, "realtime_fps_latency.png")

    # Model (optional)
    model = None
    dev = None
    if args.torchscript is not None:
        model, dev = load_model_torchscript(args.torchscript, device=args.device)

    # Stream
    stream = iter_images(args.img_dir, resize=tuple(args.resize))

    # Data buffers
    win = args.window
    t_buf = deque(maxlen=win)
    fps_buf = deque(maxlen=win)
    lat_buf = deque(maxlen=win)
    cpu_buf = deque(maxlen=win)
    ram_buf = deque(maxlen=win)
    gpu_buf = deque(maxlen=win)
    gpumem_buf = deque(maxlen=win)

    # Live plot setup
    plt.ion()
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    line_fps, = ax1.plot([], [], linewidth=1.5, label="FPS")
    line_cpu, = ax1.plot([], [], linewidth=1.0, label="CPU%")
    line_lat, = ax2.plot([], [], linewidth=1.5, label="Latency (ms)")

    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("FPS / CPU%")
    ax2.set_ylabel("Latency (ms)")
    ax1.grid(True, alpha=0.3)

    # Legends (merge both axes)
    lines = [line_fps, line_cpu, line_lat]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right")

    # Warmup
    for _ in range(args.warmup):
        img = next(stream)
        x_np = preprocess_to_tensor(img)
        if model is not None:
            _ = run_inference(model, dev, x_np)

    # CSV header
    with open(csv_path, "w", newline="") as f:
        f.write("t_sec,fps,latency_ms,cpu_percent,ram_percent,gpu_percent,gpu_mem_percent\n")

    t0 = time.perf_counter()
    prev = t0

    while True:
        now = time.perf_counter()
        elapsed = now - t0
        if elapsed >= args.seconds:
            break

        img = next(stream)
        x_np = preprocess_to_tensor(img)

        # Measure inference latency
        s = time.perf_counter()
        if model is not None:
            _ = run_inference(model, dev, x_np)
        else:
            # Simulation-only: burn a tiny compute to mimic work
            _ = (x_np * 1.0001).sum()
        e = time.perf_counter()

        latency_ms = (e - s) * 1000.0
        dt = max(now - prev, 1e-9)
        fps = 1.0 / dt
        prev = now

        cpu = psutil.cpu_percent(interval=None)
        ram = psutil.virtual_memory().percent
        gpu, gpumem = get_gpu_util()

        # Store
        t_buf.append(elapsed)
        fps_buf.append(fps)
        lat_buf.append(latency_ms)
        cpu_buf.append(cpu)
        ram_buf.append(ram)
        gpu_buf.append(gpu if gpu is not None else np.nan)
        gpumem_buf.append(gpumem if gpumem is not None else np.nan)

        # Append CSV
        with open(csv_path, "a", newline="") as f:
            f.write(f"{elapsed:.3f},{fps:.3f},{latency_ms:.3f},{cpu:.2f},{ram:.2f},{gpu if gpu is not None else ''},{gpumem if gpumem is not None else ''}\n")

        # Update plot
        line_fps.set_data(t_buf, fps_buf)
        line_cpu.set_data(t_buf, cpu_buf)
        line_lat.set_data(t_buf, lat_buf)

        ax1.set_xlim(max(0, elapsed - (args.seconds if len(t_buf) < win else (t_buf[-1] - t_buf[0]))), max(5, elapsed))
        ax1.set_ylim(0, max(5, np.nanmax(list(fps_buf)) * 1.2, np.nanmax(list(cpu_buf)) * 1.2))
        ax2.set_ylim(0, max(5, np.nanmax(list(lat_buf)) * 1.5))

        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.001)

    # Save final plot
    plt.ioff()
    fig.tight_layout()
    fig.savefig(fig_path, dpi=300)
    print(f"\nSaved: {fig_path}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
