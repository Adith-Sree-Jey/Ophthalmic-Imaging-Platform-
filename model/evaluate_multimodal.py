#!/usr/bin/env python3
"""
evaluate_multimodal.py — Evaluation for Multi-Model V7
=======================================================
Usage:
    python evaluate_multimodal.py --test_csv Metadata/test.csv --checkpoint best_qwk_model.pth
    python evaluate_multimodal.py --test_csv Metadata/test.csv --compare --ckpt_dir checkpoints_v7
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score

import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    GRADCAM_AVAILABLE = True
except ImportError:
    GRADCAM_AVAILABLE = False
    print("Warning: pytorch_grad_cam not installed")

from model.dataset_multimodal import MultiModalDatasetV6, ANT_SIZE, RG_SIZE, SL_SIZE, NORM
from model.model_multimodal import MultiModalCataractModelV7, hybrid_predict_v7

LABEL_NAMES = ["NS1", "NS2", "NS3", "NS4"]


# ═══════════════════════════════════════════════════════════════════
# TTA
# ═══════════════════════════════════════════════════════════════════

def get_tta_transforms(size):
    """5 TTA augmentations."""
    base = [A.Normalize(**NORM), ToTensorV2()]
    return [
        A.Compose([A.Resize(size, size)] + base),
        A.Compose([A.Resize(size, size), A.HorizontalFlip(p=1)] + base),
        A.Compose([A.Resize(size, size), A.VerticalFlip(p=1)] + base),
        A.Compose([A.Resize(size, size), A.Rotate(limit=15, p=1)] + base),
        A.Compose([A.Resize(size, size), A.CLAHE(clip_limit=3, p=1)] + base),
    ]


def load_img(path):
    try:
        return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)
    except:
        return np.zeros((256, 256, 3), dtype=np.uint8)


# ═══════════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════════

def evaluate(model, loader, device, ce_weight=0.65, temperature=1.1):
    """Standard evaluation."""
    model.eval()
    all_preds, all_labels, all_gids = [], [], []
    
    with torch.no_grad():
        for batch in loader:
            ant, rg, sl, labels, gids = batch[:5]
            ce_logits, coral_logits, binary_logits = model(
                ant.to(device), rg.to(device), sl.to(device)
            )
            preds = hybrid_predict_v7(
                ce_logits, coral_logits, binary_logits,
                ce_weight=ce_weight, temperature=temperature
            ).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())
            all_gids.extend(gids)
    
    return np.array(all_preds), np.array(all_labels), all_gids


def evaluate_tta(model, dataset, device, ce_weight=0.65, temperature=1.1):
    """Evaluation with TTA."""
    model.eval()
    
    tta_ant = get_tta_transforms(ANT_SIZE)
    tta_rg = get_tta_transforms(RG_SIZE)
    tta_sl = get_tta_transforms(SL_SIZE)
    
    all_preds, all_labels, all_gids = [], [], []
    
    for idx in range(len(dataset)):
        row = dataset.df.iloc[idx]
        label = dataset.labels[idx]
        
        ant_np = load_img(row["anterior_path"])
        rg_np = load_img(row["red_glow_path"])
        sl_np = load_img(row["slit_lamp_path"])
        
        ce_list, coral_list, binary_list = [], [], []
        
        with torch.no_grad():
            for t_ant, t_rg, t_sl in zip(tta_ant, tta_rg, tta_sl):
                ant = t_ant(image=ant_np)["image"].unsqueeze(0).to(device)
                rg = t_rg(image=rg_np)["image"].unsqueeze(0).to(device)
                sl = t_sl(image=sl_np)["image"].unsqueeze(0).to(device)
                
                ce_out, coral_out, binary_out = model(ant, rg, sl)
                ce_list.append(ce_out)
                coral_list.append(coral_out)
                binary_list.append(binary_out)
        
        avg_ce = torch.stack(ce_list).mean(dim=0)
        avg_coral = torch.stack(coral_list).mean(dim=0)
        avg_binary = torch.stack(binary_list).mean(dim=0)
        
        pred = hybrid_predict_v7(
            avg_ce, avg_coral, avg_binary,
            ce_weight=ce_weight, temperature=temperature
        ).item()
        
        all_preds.append(pred)
        all_labels.append(label)
        all_gids.append(row["group_id"])
    
    return np.array(all_preds), np.array(all_labels), all_gids


def compute_all_metrics(preds, labels):
    qwk = cohen_kappa_score(labels, preds, weights="quadratic")
    f1 = f1_score(labels, preds, average="macro", zero_division=0)
    acc = (preds == labels).mean()
    cm = confusion_matrix(labels, preds, labels=[0, 1, 2, 3])
    
    per_recall = {}
    for i, cls in enumerate(LABEL_NAMES):
        per_recall[cls] = round(cm[i, i] / cm[i].sum(), 3) if cm[i].sum() > 0 else 0.0
    
    return {"qwk": round(float(qwk), 4), "f1": round(float(f1), 4),
            "acc": round(float(acc), 4), "recall": per_recall, "cm": cm}


def print_metrics(m, title):
    print(f"\n{'=' * 55}")
    print(f"  {title}")
    print(f"{'=' * 55}")
    print(f"  QWK      : {m['qwk']:.4f}")
    print(f"  Macro F1 : {m['f1']:.4f}")
    print(f"  Accuracy : {m['acc']:.4f}")
    print(f"\n  Per-class recall:")
    for cls, r in m['recall'].items():
        bar = "█" * int(r * 20)
        print(f"    {cls}: {r:.3f}  {bar}")
    print(f"\n  Confusion Matrix:")
    print(pd.DataFrame(m['cm'], index=LABEL_NAMES, columns=LABEL_NAMES).to_string())


def plot_confusion_matrix(cm, save_path, title="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set(xticks=range(4), yticks=range(4),
           xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES,
           xlabel="Predicted", ylabel="True", title=title)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ═══════════════════════════════════════════════════════════════════
# GRAD-CAM
# ═══════════════════════════════════════════════════════════════════

class AntWrapper(torch.nn.Module):
    def __init__(self, model, rg, sl):
        super().__init__()
        self.model = model
        self.rg, self.sl = rg, sl
    def forward(self, x):
        ce, _, _ = self.model(x, self.rg, self.sl)
        return ce

class RgWrapper(torch.nn.Module):
    def __init__(self, model, ant, sl):
        super().__init__()
        self.model = model
        self.ant, self.sl = ant, sl
    def forward(self, x):
        ce, _, _ = self.model(self.ant, x, self.sl)
        return ce

class SlWrapper(torch.nn.Module):
    def __init__(self, model, ant, rg):
        super().__init__()
        self.model = model
        self.ant, self.rg = ant, rg
    def forward(self, x):
        ce, _, _ = self.model(self.ant, self.rg, x)
        return ce


def denorm(tensor):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = tensor.permute(1, 2, 0).cpu().numpy()
    return ((img * std + mean).clip(0, 1)).astype(np.float32)


def visualize_gradcam_batch(model, dataset, indices, device, save_dir):
    """Generate Grad-CAM visualizations."""
    if not GRADCAM_AVAILABLE:
        print("Grad-CAM not available")
        return
    
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
    val_ant = A.Compose([A.Resize(ANT_SIZE, ANT_SIZE), A.Normalize(**NORM), ToTensorV2()])
    val_rg = A.Compose([A.Resize(RG_SIZE, RG_SIZE), A.Normalize(**NORM), ToTensorV2()])
    val_sl = A.Compose([A.Resize(SL_SIZE, SL_SIZE), A.Normalize(**NORM), ToTensorV2()])
    
    target_ant = model.backbone_ant.blocks[-1]
    target_rg = model.backbone_rg.stages[-1]
    target_sl = model.backbone_sl.blocks[-1]
    
    for idx in indices:
        row = dataset.df.iloc[idx]
        true_label = dataset.labels[idx]
        group_id = row["group_id"]
        
        ant_np = load_img(row["anterior_path"])
        rg_np = load_img(row["red_glow_path"])
        sl_np = load_img(row["slit_lamp_path"])
        
        ant_t = val_ant(image=ant_np)["image"].unsqueeze(0).to(device)
        rg_t = val_rg(image=rg_np)["image"].unsqueeze(0).to(device)
        sl_t = val_sl(image=sl_np)["image"].unsqueeze(0).to(device)
        
        with torch.no_grad():
            ce_out, coral_out, binary_out = model(ant_t, rg_t, sl_t)
            pred = hybrid_predict_v7(ce_out, coral_out, binary_out).item()
        
        try:
            cam_ant = GradCAM(model=AntWrapper(model, rg_t, sl_t), target_layers=[target_ant])
            cam_rg = GradCAM(model=RgWrapper(model, ant_t, sl_t), target_layers=[target_rg])
            cam_sl = GradCAM(model=SlWrapper(model, ant_t, rg_t), target_layers=[target_sl])
            
            hm_ant = cam_ant(input_tensor=ant_t, targets=None)[0]
            hm_rg = cam_rg(input_tensor=rg_t, targets=None)[0]
            hm_sl = cam_sl(input_tensor=sl_t, targets=None)[0]
        except Exception as e:
            print(f"  Grad-CAM failed for {group_id}: {e}")
            continue
        
        ant_show = denorm(ant_t.squeeze(0))
        rg_show = denorm(rg_t.squeeze(0))
        sl_show = denorm(sl_t.squeeze(0))
        
        ov_ant = show_cam_on_image(ant_show, hm_ant, use_rgb=True)
        ov_rg = show_cam_on_image(rg_show, hm_rg, use_rgb=True)
        ov_sl = show_cam_on_image(sl_show, hm_sl, use_rgb=True)
        
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        correct = "correct" if pred == true_label else "wrong"
        fig.suptitle(f"{group_id} | True: {LABEL_NAMES[true_label]} Pred: {LABEL_NAMES[pred]} [{correct}]")
        
        for col, (orig, ov, title) in enumerate([
            (ant_show, ov_ant, "Anterior"),
            (rg_show, ov_rg, "Red Glow"),
            (sl_show, ov_sl, "Slit Lamp"),
        ]):
            axes[0, col].imshow(orig)
            axes[0, col].set_title(title)
            axes[0, col].axis("off")
            axes[1, col].imshow(ov)
            axes[1, col].set_title(f"Grad-CAM {title}")
            axes[1, col].axis("off")
        
        plt.tight_layout()
        fname = f"{correct}_{group_id.replace(' ', '_')}_t{LABEL_NAMES[true_label]}_p{LABEL_NAMES[pred]}.png"
        plt.savefig(os.path.join(save_dir, fname), dpi=120, bbox_inches="tight")
        plt.close()
    
    print(f"Grad-CAM saved to {save_dir}/")


def run_evaluation(checkpoint, test_ds, test_ldr, device, save_dir, cfg):
    os.makedirs(save_dir, exist_ok=True)
    
    model = MultiModalCataractModelV7(num_classes=4).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    label = os.path.splitext(os.path.basename(checkpoint))[0]
    print(f"\nLoaded [{label}]: {checkpoint}")
    
    # Standard eval
    preds, labels, gids = evaluate(model, test_ldr, device, cfg.ce_weight, cfg.temperature)
    m_std = compute_all_metrics(preds, labels)
    print_metrics(m_std, f"{label} — no TTA")
    plot_confusion_matrix(m_std['cm'], f"{save_dir}/cm_{label}_no_tta.png")
    
    pd.DataFrame({
        "group_id": gids, "true": [LABEL_NAMES[l] for l in labels],
        "pred": [LABEL_NAMES[p] for p in preds], "correct": preds == labels
    }).to_csv(f"{save_dir}/predictions_{label}.csv", index=False)
    
    # TTA eval
    m_tta = m_std
    if not cfg.no_tta:
        print(f"\nRunning TTA...")
        tp, tl, tg = evaluate_tta(model, test_ds, device, cfg.ce_weight, cfg.temperature)
        m_tta = compute_all_metrics(tp, tl)
        print_metrics(m_tta, f"{label} — with TTA")
        plot_confusion_matrix(m_tta['cm'], f"{save_dir}/cm_{label}_tta.png")
        
        pd.DataFrame({
            "group_id": tg, "true": [LABEL_NAMES[l] for l in tl],
            "pred": [LABEL_NAMES[p] for p in tp], "correct": tp == tl
        }).to_csv(f"{save_dir}/predictions_{label}_tta.csv", index=False)
        
        print(f"\n  TTA improvement: QWK {m_std['qwk']:.4f} → {m_tta['qwk']:.4f}, "
              f"ACC {m_std['acc']:.4f} → {m_tta['acc']:.4f}")
    
    # Grad-CAM
    if cfg.gradcam_n > 0:
        gradcam_idx = []
        for ci in range(4):
            mask = np.where(labels == ci)[0]
            gradcam_idx.extend(mask[:max(1, cfg.gradcam_n // 4)].tolist())
        print(f"\nGenerating Grad-CAM for {len(gradcam_idx)} samples...")
        visualize_gradcam_batch(model, test_ds, gradcam_idx, device, f"{save_dir}/gradcam_{label}")
    
    return m_std, m_tta


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test_csv", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--compare", action="store_true")
    p.add_argument("--ckpt_dir", default=None)
    p.add_argument("--save_dir", default="eval_outputs_v7")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--gradcam_n", type=int, default=20)
    p.add_argument("--no_tta", action="store_true")
    p.add_argument("--ce_weight", type=float, default=0.65)
    p.add_argument("--temperature", type=float, default=1.1)
    cfg = p.parse_args()
    
    os.makedirs(cfg.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    test_ds = MultiModalDatasetV6(cfg.test_csv, is_train=False)
    test_ldr = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    
    if cfg.compare:
        ckpt_dir = cfg.ckpt_dir or os.path.dirname(cfg.checkpoint or ".")
        qwk_ckpt = os.path.join(ckpt_dir, "best_qwk_model.pth")
        acc_ckpt = os.path.join(ckpt_dir, "best_acc_model.pth")
        
        print(f"\nComparison mode:")
        print(f"  QWK: {qwk_ckpt}")
        print(f"  ACC: {acc_ckpt}")
        
        m_qwk_std, m_qwk_tta = run_evaluation(qwk_ckpt, test_ds, test_ldr, device, cfg.save_dir, cfg)
        m_acc_std, m_acc_tta = run_evaluation(acc_ckpt, test_ds, test_ldr, device, cfg.save_dir, cfg)
        
        print(f"\n{'=' * 65}")
        print(f"  COMPARISON SUMMARY")
        print(f"{'=' * 65}")
        print(f"  {'Metric':<12} {'QWK(std)':>10} {'QWK(tta)':>10} {'ACC(std)':>10} {'ACC(tta)':>10}")
        for key in ["qwk", "acc", "f1"]:
            print(f"  {key.upper():<12} {m_qwk_std[key]:>10.4f} {m_qwk_tta[key]:>10.4f} "
                  f"{m_acc_std[key]:>10.4f} {m_acc_tta[key]:>10.4f}")
        
        # Per-class recall comparison
        print(f"\n  Per-class Recall (TTA):")
        print(f"  {'Class':<8} {'best_qwk':>10} {'best_acc':>10}")
        for cls in LABEL_NAMES:
            print(f"  {cls:<8} {m_qwk_tta['recall'][cls]:>10.3f} {m_acc_tta['recall'][cls]:>10.3f}")
    else:
        if not cfg.checkpoint:
            print("Provide --checkpoint or use --compare")
            return
        run_evaluation(cfg.checkpoint, test_ds, test_ldr, device, cfg.save_dir, cfg)


if __name__ == "__main__":
    main()
