#!/usr/bin/env python3
"""
train_multimodal.py — Training script for Multi-Model V7
=========================================================
V7 Changes from V6:
  - Reduced binary_weight (0.1 → 0.05)
  - Added ordinal consistency loss
  - Rebalanced class weights (boost NS1/NS3)
  - Lower temperature (1.2 → 1.1)
  - Adaptive binary correction in predictions

Usage:
    python train_multimodal.py --train_csv Metadata/train.csv --val_csv Metadata/val.csv
"""

import os
import time
import argparse
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_

from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score

from model.dataset_multimodal import MultiModalDatasetV6, make_weighted_sampler_v6
from model.model_multimodal import (
    MultiModalCataractModelV7,
    HybridLossV7,
    hybrid_predict_v7,
    attention_diversity_loss,
    compute_class_weights_v7
)


def get_config():
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", required=True)
    p.add_argument("--val_csv", required=True)
    p.add_argument("--save_dir", default="checkpoints_v7")
    
    # Training
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--phase1_epochs", type=int, default=12)
    p.add_argument("--phase2_epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=25)
    
    # Learning rates
    p.add_argument("--lr_head", type=float, default=1e-3)
    p.add_argument("--lr_backbone", type=float, default=5e-6)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    
    # Loss weights (V7 defaults)
    p.add_argument("--ce_weight", type=float, default=0.65)
    p.add_argument("--coral_weight", type=float, default=0.25)
    p.add_argument("--binary_weight", type=float, default=0.05)  # Reduced from V6
    p.add_argument("--ordinal_weight", type=float, default=0.05)  # NEW
    p.add_argument("--attn_reg_weight", type=float, default=0.1)
    
    # Focal loss
    p.add_argument("--focal_gamma", type=float, default=2.0)
    p.add_argument("--label_smoothing", type=float, default=0.05)
    p.add_argument("--temperature", type=float, default=1.1)  # Reduced from V6
    
    # Architecture
    p.add_argument("--dropout", type=float, default=0.4)
    p.add_argument("--unfreeze_blocks", type=int, default=3)
    
    # Class weights
    p.add_argument("--weight_strategy", default="balanced_v7",
                   choices=["balanced_v7", "sqrt_inverse", "uniform"])
    
    # Other
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true")
    
    return p.parse_args()


def compute_metrics(all_preds, all_labels):
    """Compute all evaluation metrics."""
    preds = np.array(all_preds)
    labels = np.array(all_labels)
    
    qwk = cohen_kappa_score(labels, preds, weights="quadratic")
    f1 = f1_score(labels, preds, average="macro", zero_division=0)
    acc = (preds == labels).mean()
    cm = confusion_matrix(labels, preds, labels=[0, 1, 2, 3])
    
    per_recall = {}
    for i, cls in enumerate(["NS1", "NS2", "NS3", "NS4"]):
        per_recall[cls] = round(cm[i, i] / cm[i].sum(), 3) if cm[i].sum() > 0 else 0.0
    
    return {
        "qwk": round(float(qwk), 4),
        "f1_macro": round(float(f1), 4),
        "accuracy": round(float(acc), 4),
        "per_class_recall": per_recall,
        "confusion_matrix": cm
    }


def run_epoch(model, loader, loss_fn, cfg, optimizer=None, scaler=None,
              device=None, use_attn_reg=False):
    """Run one training or validation epoch."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    
    total_loss = 0.0
    total_focal = 0.0
    total_coral = 0.0
    total_binary = 0.0
    total_ordinal = 0.0
    total_attn_reg = 0.0
    all_preds, all_labels = [], []
    attn_sum = torch.zeros(3, device=device)
    attn_count = 0
    
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    
    with ctx:
        for batch in loader:
            ant, rg, sl, labels, gids = batch[:5]
            
            ant = ant.to(device, non_blocking=True)
            rg = rg.to(device, non_blocking=True)
            sl = sl.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            with autocast("cuda", enabled=(device.type == "cuda")):
                ce_logits, coral_logits, binary_logits, attn_weights = model(
                    ant, rg, sl, return_attention=True
                )
                
                # Handle soft labels from mixup
                if labels.dim() == 2:
                    hard_labels = labels.argmax(dim=1)
                else:
                    hard_labels = labels
                
                loss, focal_l, coral_l, binary_l, ordinal_l = loss_fn(
                    ce_logits, coral_logits, binary_logits, hard_labels
                )
                
                # Attention diversity regularization (Phase 2 only)
                attn_reg = torch.tensor(0.0, device=device)
                if is_train and use_attn_reg:
                    attn_reg = attention_diversity_loss(attn_weights) * cfg.attn_reg_weight
                    loss = loss + attn_reg
            
            if is_train:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            
            batch_size = len(hard_labels)
            total_loss += loss.item() * batch_size
            total_focal += focal_l.item() * batch_size
            total_coral += coral_l.item() * batch_size
            total_binary += binary_l.item() * batch_size
            total_ordinal += ordinal_l.item() * batch_size
            total_attn_reg += attn_reg.item() * batch_size
            
            preds = hybrid_predict_v7(
                ce_logits, coral_logits, binary_logits,
                ce_weight=cfg.ce_weight, temperature=cfg.temperature
            )
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(hard_labels.cpu().numpy().tolist())
            
            attn_sum += attn_weights.sum(dim=0).detach()
            attn_count += batch_size
    
    n = len(loader.dataset)
    metrics = compute_metrics(all_preds, all_labels)
    metrics["loss"] = round(total_loss / n, 4)
    metrics["focal_loss"] = round(total_focal / n, 4)
    metrics["coral_loss"] = round(total_coral / n, 4)
    metrics["binary_loss"] = round(total_binary / n, 4)
    metrics["ordinal_loss"] = round(total_ordinal / n, 4)
    metrics["attn_reg"] = round(total_attn_reg / n, 6)
    
    avg_attn = (attn_sum / attn_count).cpu().numpy()
    metrics["attn_ant"] = round(float(avg_attn[0]), 2)
    metrics["attn_rg"] = round(float(avg_attn[1]), 2)
    metrics["attn_sl"] = round(float(avg_attn[2]), 2)
    
    return metrics


def check_logit_explosion(model, device):
    """Check for logit explosion."""
    with torch.no_grad():
        x = torch.randn(1, 3, 260, 260).to(device)
        x_rg = torch.randn(1, 3, 224, 224).to(device)
        ce_out, coral_out, binary_out = model(x, x_rg, x)
        ce_max = ce_out.abs().max().item()
        if ce_max > 50:
            print(f"  ⚠️  WARNING: logits may be exploding — ce={ce_max:.1f}")
            return True
    return False


def save_history(history, save_path):
    """Save training history to CSV (crash-safe)."""
    pd.DataFrame(history).to_csv(save_path, index=False)


def load_resume_state(save_dir):
    """Load resume state from training history."""
    history_path = os.path.join(save_dir, "training_history.csv")
    if not os.path.exists(history_path):
        return None, [], 0, 0
    
    history_df = pd.read_csv(history_path)
    if len(history_df) == 0:
        return None, [], 0, 0
    
    last_row = history_df.iloc[-1]
    phase = int(last_row['phase'])
    epoch = int(last_row['epoch'])
    
    ckpt_path = os.path.join(save_dir, "best_qwk_model.pth")
    if os.path.exists(ckpt_path):
        state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    else:
        state_dict = None
    
    history = history_df.to_dict('records')
    
    return state_dict, history, phase, epoch


def train(cfg):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
    
    os.makedirs(cfg.save_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"  Multi-Model V7 Training")
    print(f"{'='*60}")
    print(f"  Batch size: {cfg.batch_size}")
    print(f"  Phase 1: {cfg.phase1_epochs} epochs (frozen)")
    print(f"  Phase 2: {cfg.phase2_epochs} epochs (unfreeze last {cfg.unfreeze_blocks})")
    print(f"  Loss: Focal({cfg.ce_weight}) + CORAL({cfg.coral_weight}) + Binary({cfg.binary_weight}) + Ordinal({cfg.ordinal_weight})")
    print(f"  Focal gamma: {cfg.focal_gamma}")
    print(f"  Temperature: {cfg.temperature}")
    print(f"  Weight strategy: {cfg.weight_strategy}")
    print(f"{'='*60}")
    
    # Datasets
    print("\nLoading datasets...")
    train_ds = MultiModalDatasetV6(cfg.train_csv, is_train=True)
    val_ds = MultiModalDatasetV6(cfg.val_csv, is_train=False)
    
    sampler = make_weighted_sampler_v6(train_ds, strategy='sqrt_inverse')  # Use sqrt for V7
    
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, sampler=sampler,
        num_workers=cfg.num_workers, pin_memory=(device.type == "cuda"), drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=(device.type == "cuda")
    )
    
    # Model
    print("\nInitializing model...")
    model = MultiModalCataractModelV7(num_classes=4, dropout=cfg.dropout)
    model = model.to(device)
    
    # Class weights and loss
    class_weights = compute_class_weights_v7(cfg.train_csv, device, cfg.weight_strategy)
    
    loss_fn = HybridLossV7(
        class_weights=class_weights,
        focal_gamma=cfg.focal_gamma,
        ce_weight=cfg.ce_weight,
        coral_weight=cfg.coral_weight,
        binary_weight=cfg.binary_weight,
        ordinal_weight=cfg.ordinal_weight,
        label_smoothing=cfg.label_smoothing,
        temperature=cfg.temperature
    )
    
    scaler = GradScaler("cuda") if device.type == "cuda" else GradScaler()
    
    # Resume logic
    history = []
    start_phase = 1
    start_epoch = 1
    best_qwk = -1.0
    best_acc = -1.0
    
    if cfg.resume:
        state_dict, history, last_phase, last_epoch = load_resume_state(cfg.save_dir)
        if state_dict is not None:
            model.load_state_dict(state_dict)
            start_phase = last_phase
            start_epoch = last_epoch + 1
            best_qwk = max([h.get('va_qwk', -1) for h in history], default=-1)
            best_acc = max([h.get('va_accuracy', -1) for h in history], default=-1)
            print(f"Resumed from Phase {start_phase}, Epoch {start_epoch}")
            print(f"Best QWK: {best_qwk:.4f}, Best ACC: {best_acc:.4f}")
    
    history_path = os.path.join(cfg.save_dir, "training_history.csv")
    
    # ══════════════════════════════════════════════════════════════
    # PHASE 1: Frozen backbones
    # ══════════════════════════════════════════════════════════════
    if start_phase == 1:
        print("\n" + "=" * 60)
        print("Phase 1 — Frozen backbones")
        print("=" * 60)
        
        model.freeze_backbones()
        head_params = model.get_head_params()
        optimizer = AdamW(head_params, lr=cfg.lr_head, weight_decay=cfg.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg.phase1_epochs, eta_min=1e-6)
        
        for epoch in range(start_epoch if start_phase == 1 else 1, cfg.phase1_epochs + 1):
            t0 = time.time()
            
            tr = run_epoch(model, train_loader, loss_fn, cfg, optimizer, scaler,
                          device, use_attn_reg=False)
            va = run_epoch(model, val_loader, loss_fn, cfg, None, None,
                          device, use_attn_reg=False)
            scheduler.step()
            
            elapsed = time.time() - t0
            
            print(f"[P1 {epoch:02d}/{cfg.phase1_epochs}] "
                  f"loss={tr['loss']:.3f} ord={tr['ordinal_loss']:.3f} | "
                  f"val_qwk={va['qwk']:.3f} acc={va['accuracy']:.3f} | "
                  f"NS1={va['per_class_recall']['NS1']:.2f} NS2={va['per_class_recall']['NS2']:.2f} | "
                  f"{int(elapsed//60)}m{int(elapsed%60)}s")
            
            # Save history
            row = {"phase": 1, "epoch": epoch}
            for k, v in tr.items():
                if "matrix" not in k and "recall" not in k:
                    row[f"tr_{k}"] = v
            for k, v in va.items():
                if "matrix" not in k and "recall" not in k:
                    row[f"va_{k}"] = v
            for cls, r in va['per_class_recall'].items():
                row[f"va_recall_{cls}"] = r
            history.append(row)
            save_history(history, history_path)
            
            # Checkpointing
            if va["qwk"] > best_qwk:
                best_qwk = va["qwk"]
                torch.save(model.state_dict(), os.path.join(cfg.save_dir, "best_qwk_model.pth"))
                print(f"  >>> Best QWK: {best_qwk:.4f}")
            
            if va["accuracy"] > best_acc:
                best_acc = va["accuracy"]
                torch.save(model.state_dict(), os.path.join(cfg.save_dir, "best_acc_model.pth"))
                print(f"  >>> Best ACC: {best_acc:.4f}")
            
            check_logit_explosion(model, device)
        
        start_phase = 2
        start_epoch = 1
    
    # ══════════════════════════════════════════════════════════════
    # PHASE 2: Fine-tune backbones
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print(f"Phase 2 — Unfreeze last {cfg.unfreeze_blocks} blocks")
    print("=" * 60)
    
    model.unfreeze_backbones_last_n(cfg.unfreeze_blocks)
    
    optimizer = AdamW([
        {"params": model.get_backbone_params(), "lr": cfg.lr_backbone},
        {"params": model.get_head_params(), "lr": cfg.lr_head / 10},
    ], weight_decay=cfg.weight_decay)
    
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.phase2_epochs, eta_min=1e-7)
    no_improve = 0
    
    for epoch in range(start_epoch if start_phase == 2 else 1, cfg.phase2_epochs + 1):
        t0 = time.time()
        
        tr = run_epoch(model, train_loader, loss_fn, cfg, optimizer, scaler,
                      device, use_attn_reg=True)
        va = run_epoch(model, val_loader, loss_fn, cfg, None, None,
                      device, use_attn_reg=False)
        scheduler.step()
        
        elapsed = time.time() - t0
        
        print(f"[P2 {epoch:02d}/{cfg.phase2_epochs}] "
              f"loss={tr['loss']:.3f} ord={tr['ordinal_loss']:.3f} | "
              f"val_qwk={va['qwk']:.3f} acc={va['accuracy']:.3f} f1={va['f1_macro']:.3f} | "
              f"NS1={va['per_class_recall']['NS1']:.2f} NS2={va['per_class_recall']['NS2']:.2f} "
              f"NS3={va['per_class_recall']['NS3']:.2f} NS4={va['per_class_recall']['NS4']:.2f} | "
              f"wait={no_improve} | {int(elapsed//60)}m{int(elapsed%60)}s")
        
        # Save history
        row = {"phase": 2, "epoch": epoch}
        for k, v in tr.items():
            if "matrix" not in k and "recall" not in k:
                row[f"tr_{k}"] = v
        for k, v in va.items():
            if "matrix" not in k and "recall" not in k:
                row[f"va_{k}"] = v
        for cls, r in va['per_class_recall'].items():
            row[f"va_recall_{cls}"] = r
        history.append(row)
        save_history(history, history_path)
        
        # Checkpointing with early stopping
        if va["qwk"] > best_qwk:
            best_qwk = va["qwk"]
            no_improve = 0
            torch.save(model.state_dict(), os.path.join(cfg.save_dir, "best_qwk_model.pth"))
            print(f"  >>> Best QWK: {best_qwk:.4f}")
        else:
            no_improve += 1
            if no_improve >= cfg.patience:
                print(f"\n  Early stopping at epoch {epoch}")
                break
        
        if va["accuracy"] > best_acc:
            best_acc = va["accuracy"]
            torch.save(model.state_dict(), os.path.join(cfg.save_dir, "best_acc_model.pth"))
            print(f"  >>> Best ACC: {best_acc:.4f}")
        
        check_logit_explosion(model, device)
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"  Training Complete!")
    print(f"  Best val QWK = {best_qwk:.4f}")
    print(f"  Best val ACC = {best_acc:.4f}")
    print(f"{'='*60}")
    
    # Print final confusion matrix from best model
    print("\nLoading best QWK model for final evaluation...")
    model.load_state_dict(torch.load(
        os.path.join(cfg.save_dir, "best_qwk_model.pth"),
        map_location=device, weights_only=True
    ))
    final_metrics = run_epoch(model, val_loader, loss_fn, cfg, None, None, device, False)
    print("\nFinal Confusion Matrix:")
    print(pd.DataFrame(
        final_metrics['confusion_matrix'],
        index=["NS1", "NS2", "NS3", "NS4"],
        columns=["NS1", "NS2", "NS3", "NS4"]
    ).to_string())


if __name__ == "__main__":
    cfg = get_config()
    train(cfg)
