"""
model_multimodal.py — Multi-Model V7 Architecture
==================================================
Changes from V6:
  1. Reduced binary_weight (0.1 → 0.05) — less aggressive NS2 push
  2. Rebalanced class weights — boost NS1 and NS3
  3. Lower temperature (1.2 → 1.1) — slightly sharper predictions
  4. Ordinal consistency loss — penalize grade-skipping predictions
  5. Adaptive binary correction — weaker correction when uncertain

Architecture: Same as V6
    Anterior (260×260)  → EfficientNet-B2 → proj(1408→512) ─┐
    Red glow (224×224)  → ConvNeXt-Tiny   → proj(768→512)  ─┼─ CrossModalAttention
    Slit lamp (260×260) → EfficientNet-B2 → proj(1408→512) ─┘
                                    ↓
                    Concat [attended + f_ant + f_rg + f_sl] = 2048d
                                    ↓
                    FusionMLP → 128d features
                                    ↓
              ┌──────────────┬──────────────┬──────────────┐
           CE head       CORAL head     Binary head
         (4-class)       (ordinal)     (NS1 vs NS2+)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


# ═══════════════════════════════════════════════════════════════════
# FOCAL LOSS (same as V6)
# ═══════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean', label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
    
    def forward(self, inputs, targets):
        num_classes = inputs.size(1)
        if self.label_smoothing > 0:
            with torch.no_grad():
                smooth_targets = torch.zeros_like(inputs)
                smooth_targets.fill_(self.label_smoothing / (num_classes - 1))
                smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
        
        p = F.softmax(inputs, dim=1)
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1 - p_t) ** self.gamma
        
        if self.label_smoothing > 0:
            ce = -(smooth_targets * F.log_softmax(inputs, dim=1)).sum(dim=1)
        else:
            ce = F.cross_entropy(inputs, targets, reduction='none')
        
        loss = focal_weight * ce
        
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)
            loss = alpha_t * loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


# ═══════════════════════════════════════════════════════════════════
# CORAL ORDINAL LOSS (same as V6)
# ═══════════════════════════════════════════════════════════════════

class CoralHead(nn.Module):
    """CORAL head for ordinal regression."""
    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.fc = nn.Linear(in_features, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(num_classes - 1))
    
    def forward(self, x):
        return self.fc(x) + self.bias


def coral_loss(logits, targets, num_classes, class_weights=None, smoothing=0.02):
    """CORAL loss with optional label smoothing."""
    B = targets.size(0)
    K1 = num_classes - 1
    
    rank_targets = torch.zeros(B, K1, device=targets.device, dtype=torch.float32)
    for i in range(K1):
        rank_targets[:, i] = (targets > i).float()
    
    if smoothing > 0:
        rank_targets = rank_targets * (1 - smoothing) + (1 - rank_targets) * smoothing
    
    loss = F.binary_cross_entropy_with_logits(logits, rank_targets, reduction="none")
    
    if class_weights is not None:
        loss = loss * class_weights[targets].unsqueeze(1)
    
    return loss.mean()


def coral_logits_to_probs(logits):
    """Convert CORAL logits to class probabilities."""
    cumulative = torch.sigmoid(logits)
    B = logits.size(0)
    K = logits.size(1) + 1
    
    probs = torch.zeros(B, K, device=logits.device)
    probs[:, 0] = 1 - cumulative[:, 0]
    for k in range(1, K - 1):
        probs[:, k] = cumulative[:, k-1] - cumulative[:, k]
    probs[:, K-1] = cumulative[:, K-2]
    
    probs = probs.clamp(min=0)
    probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-8)
    
    return probs


# ═══════════════════════════════════════════════════════════════════
# ORDINAL CONSISTENCY LOSS (NEW in V7)
# ═══════════════════════════════════════════════════════════════════

def ordinal_consistency_loss(ce_logits, targets):
    """
    Penalize predictions that "skip" grades.
    E.g., if true=NS1, penalize high probability on NS3/NS4.
    E.g., if true=NS4, penalize high probability on NS1/NS2.
    """
    probs = F.softmax(ce_logits, dim=1)
    B = probs.size(0)
    
    loss = torch.zeros(B, device=probs.device)
    
    for i in range(B):
        true_grade = targets[i].item()
        
        # Penalize probability mass more than 1 grade away
        for pred_grade in range(4):
            distance = abs(pred_grade - true_grade)
            if distance > 1:
                # Quadratic penalty for skipping grades
                loss[i] += (distance - 1) * probs[i, pred_grade]
    
    return loss.mean()


# ═══════════════════════════════════════════════════════════════════
# HYBRID LOSS V7
# ═══════════════════════════════════════════════════════════════════

class HybridLossV7(nn.Module):
    """
    V7 Hybrid Loss:
      - Focal Loss (primary) — focuses on hard examples
      - CORAL (auxiliary) — ordinal structure
      - Binary (auxiliary) — NS1 vs NS2+ boundary (REDUCED weight)
      - Ordinal consistency (NEW) — penalize grade skipping
    """
    def __init__(self, class_weights, focal_gamma=2.0, 
                 ce_weight=0.65, coral_weight=0.25, binary_weight=0.05,
                 ordinal_weight=0.05, label_smoothing=0.05, temperature=1.1):
        super().__init__()
        self.class_weights = class_weights
        self.focal_gamma = focal_gamma
        self.ce_weight = ce_weight
        self.coral_weight = coral_weight
        self.binary_weight = binary_weight
        self.ordinal_weight = ordinal_weight
        self.label_smoothing = label_smoothing
        self.temperature = temperature
        
        self.focal_loss = FocalLoss(
            alpha=class_weights,
            gamma=focal_gamma,
            label_smoothing=label_smoothing
        )
    
    def forward(self, ce_logits, coral_logits, binary_logits, targets):
        # Temperature scaling
        ce_logits_scaled = ce_logits / self.temperature
        
        # Focal loss (primary)
        focal_l = self.focal_loss(ce_logits_scaled, targets)
        
        # CORAL loss
        coral_l = coral_loss(
            coral_logits, targets, num_classes=4,
            class_weights=self.class_weights, smoothing=0.02
        )
        
        # Binary loss: NS1 (0) vs NS2+ (1) — reduced pos_weight
        binary_targets = (targets >= 1).float()
        binary_l = F.binary_cross_entropy_with_logits(
            binary_logits.squeeze(-1), binary_targets,
            pos_weight=torch.tensor(1.2, device=targets.device)  # Reduced from 1.5
        )
        
        # Ordinal consistency loss (NEW)
        ordinal_l = ordinal_consistency_loss(ce_logits, targets)
        
        total = (self.ce_weight * focal_l + 
                 self.coral_weight * coral_l + 
                 self.binary_weight * binary_l +
                 self.ordinal_weight * ordinal_l)
        
        return total, focal_l, coral_l, binary_l, ordinal_l


def hybrid_predict_v7(ce_logits, coral_logits, binary_logits, 
                      ce_weight=0.65, temperature=1.1, binary_boost=0.05):
    """
    V7 prediction with adaptive binary correction.
    Binary correction is weaker and confidence-gated.
    """
    ce_logits_scaled = ce_logits / temperature
    ce_probs = F.softmax(ce_logits_scaled, dim=1)
    coral_probs = coral_logits_to_probs(coral_logits)
    
    # Combine CE and CORAL (more CORAL weight for ordinal consistency)
    combined = ce_weight * ce_probs + (1 - ce_weight) * coral_probs
    
    # Adaptive binary correction (weaker, confidence-gated)
    binary_prob = torch.sigmoid(binary_logits.squeeze(-1))  # P(NS2+)
    
    # Only apply correction when binary head is confident
    confidence_gate = (binary_prob - 0.5).abs() * 2  # 0 at 0.5, 1 at 0/1
    effective_boost = binary_boost * confidence_gate
    
    # Redistribute: reduce NS1, boost NS2
    ns1_reduction = effective_boost * binary_prob
    combined[:, 0] = combined[:, 0] * (1 - ns1_reduction)
    combined[:, 1] = combined[:, 1] + ns1_reduction * 0.5  # Only half goes to NS2
    
    # Renormalize
    combined = combined / (combined.sum(dim=1, keepdim=True) + 1e-8)
    
    return combined.argmax(dim=1)


# ═══════════════════════════════════════════════════════════════════
# CROSS-MODAL ATTENTION (same as V6)
# ═══════════════════════════════════════════════════════════════════

class CrossModalAttention(nn.Module):
    """Learns attention weights over the 3 modalities."""
    def __init__(self, feat_dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(feat_dim * 3, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, 3),
            nn.Softmax(dim=1)
        )
    
    def forward(self, f_ant, f_rg, f_sl):
        combined = torch.cat([f_ant, f_rg, f_sl], dim=1)
        weights = self.attn(combined)
        fused = weights[:, 0:1] * f_ant + weights[:, 1:2] * f_rg + weights[:, 2:3] * f_sl
        return fused, weights


def attention_diversity_loss(attn_weights):
    """Penalize attention collapse."""
    eps = 1e-8
    entropy = -(attn_weights * (attn_weights + eps).log()).sum(dim=1)
    max_entropy = torch.log(torch.tensor(3.0, device=attn_weights.device))
    return (1.0 - entropy / max_entropy).mean()


# ═══════════════════════════════════════════════════════════════════
# MAIN MODEL V7 (architecture same as V6)
# ═══════════════════════════════════════════════════════════════════

class MultiModalCataractModelV7(nn.Module):
    """
    V7: Same architecture as V6, different loss weights and prediction.
    """
    
    def __init__(self, num_classes: int = 4, dropout: float = 0.4):
        super().__init__()
        self.num_classes = num_classes
        
        # Load backbones
        print("Loading EfficientNet-B2 for anterior segment...")
        self.backbone_ant = timm.create_model(
            'efficientnet_b2', pretrained=True, num_classes=0, global_pool='avg'
        )
        ant_dim = 1408
        
        print("Loading ConvNeXt-Tiny for red glow...")
        self.backbone_rg = timm.create_model(
            'convnext_tiny', pretrained=True, num_classes=0, global_pool='avg'
        )
        rg_dim = 768
        
        print("Loading EfficientNet-B2 for slit lamp...")
        self.backbone_sl = timm.create_model(
            'efficientnet_b2', pretrained=True, num_classes=0, global_pool='avg'
        )
        sl_dim = 1408
        
        # Projection layers
        proj_dim = 512
        self.proj_ant = nn.Sequential(
            nn.Linear(ant_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        self.proj_rg = nn.Sequential(
            nn.Linear(rg_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        self.proj_sl = nn.Sequential(
            nn.Linear(sl_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        
        # Cross-modal attention
        self.cross_attn = CrossModalAttention(proj_dim)
        
        # Fusion MLP
        fusion_in = proj_dim * 4
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_in, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )
        
        # Three heads
        self.ce_head = nn.Linear(128, num_classes)
        self.coral_head = CoralHead(128, num_classes)
        self.binary_head = nn.Linear(128, 1)
        
        total = sum(p.numel() for p in self.parameters())
        print(f"Total parameters: {total:,}")
    
    def forward(self, x_ant, x_rg, x_sl, return_attention=False):
        f_ant = self.proj_ant(self.backbone_ant(x_ant))
        f_rg = self.proj_rg(self.backbone_rg(x_rg))
        f_sl = self.proj_sl(self.backbone_sl(x_sl))
        
        fused, attn_weights = self.cross_attn(f_ant, f_rg, f_sl)
        combined = torch.cat([fused, f_ant, f_rg, f_sl], dim=1)
        features = self.fusion_mlp(combined)
        
        ce_logits = self.ce_head(features)
        coral_logits = self.coral_head(features)
        binary_logits = self.binary_head(features)
        
        if return_attention:
            return ce_logits, coral_logits, binary_logits, attn_weights
        return ce_logits, coral_logits, binary_logits
    
    def freeze_backbones(self):
        for backbone in [self.backbone_ant, self.backbone_rg, self.backbone_sl]:
            for p in backbone.parameters():
                p.requires_grad = False
        print("All backbones frozen.")
    
    def unfreeze_backbones_last_n(self, n: int = 3):
        for backbone in [self.backbone_ant, self.backbone_sl]:
            for p in backbone.parameters():
                p.requires_grad = False
            for block in backbone.blocks[-n:]:
                for p in block.parameters():
                    p.requires_grad = True
        
        for p in self.backbone_rg.parameters():
            p.requires_grad = False
        for stage in self.backbone_rg.stages[-n:]:
            for p in stage.parameters():
                p.requires_grad = True
        for p in self.backbone_rg.head.parameters():
            p.requires_grad = True
        
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Unfrozen last {n} blocks — trainable params: {trainable:,}")
    
    def get_backbone_params(self):
        params = []
        for backbone in [self.backbone_ant, self.backbone_rg, self.backbone_sl]:
            params.extend([p for p in backbone.parameters() if p.requires_grad])
        return params
    
    def get_head_params(self):
        return (
            list(self.proj_ant.parameters()) +
            list(self.proj_rg.parameters()) +
            list(self.proj_sl.parameters()) +
            list(self.cross_attn.parameters()) +
            list(self.fusion_mlp.parameters()) +
            list(self.ce_head.parameters()) +
            list(self.coral_head.parameters()) +
            list(self.binary_head.parameters())
        )


# ═══════════════════════════════════════════════════════════════════
# CLASS WEIGHTS V7 — Balanced for NS1/NS2/NS3
# ═══════════════════════════════════════════════════════════════════

def compute_class_weights_v7(train_csv, device, strategy='balanced_v7'):
    """
    V7 class weights — more balanced across all classes.
    """
    import pandas as pd
    df = pd.read_csv(train_csv)
    df = df[df['has_anterior'] & df['has_red_glow'] & df['has_slit_lamp']]
    counts = df["label"].value_counts()
    
    ns1 = counts.get("NS1", 1)
    ns2 = counts.get("NS2", 1)
    ns3 = counts.get("NS3", 1)
    ns4 = counts.get("NS4", 1)
    
    print(f"Class counts: NS1={ns1}, NS2={ns2}, NS3={ns3}, NS4={ns4}")
    
    if strategy == 'balanced_v7':
        # V7: Boost NS1 and NS3, keep NS2 moderate
        weights = [0.85, 0.75, 0.90, 1.0]
    
    elif strategy == 'uniform':
        weights = [1.0, 1.0, 1.0, 1.0]
    
    elif strategy == 'sqrt_inverse':
        total = ns1 + ns2 + ns3 + ns4
        weights = [
            (total / ns1) ** 0.5,
            (total / ns2) ** 0.5,
            (total / ns3) ** 0.5,
            (total / ns4) ** 0.5,
        ]
        max_w = max(weights)
        weights = [w / max_w for w in weights]
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    print(f"Class weights ({strategy}): {[round(w, 3) for w in weights]}")
    return torch.tensor(weights, dtype=torch.float32).to(device)
