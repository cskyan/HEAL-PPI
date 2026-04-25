# -*- coding: utf-8 -*-
"""Training script for the public RBP top-k release.

Public interface goals:
- keep the current repository structure unchanged
- preserve the four input modalities through dataset/config switches
- keep the dual-output interface explicit:
  1) binary prediction via ``p_fused``
  2) evidence prediction via ``evi_score`` / ``evi_logit``
"""
import os, math, time, random, re, uuid
import numpy as np
from typing import Optional
import torch.nn as nn
import torch.nn.functional as F
from loss_innovations import integrate_new_losses, format_aux_log
from torch.utils.data import DataLoader, Subset
import torch
from sklearn.metrics import roc_auc_score

import json
from contextlib import nullcontext

try:
    from torch.cuda.amp import autocast, GradScaler
except Exception:
    autocast = None
    GradScaler = None


def _resolve_runtime_device(default_cuda_index: int = 2) -> str:
    if not torch.cuda.is_available():
        return "cpu"
    raw = str(os.environ.get("CUDA_DEVICE_INDEX", default_cuda_index)).strip()
    try:
        idx = int(raw)
    except Exception:
        idx = int(default_cuda_index)
    n = int(torch.cuda.device_count())
    if idx < 0 or idx >= max(1, n):
        print(f"[device][warn] invalid CUDA_DEVICE_INDEX={raw}, fallback to cuda:0", flush=True)
        idx = 0
    try:
        torch.cuda.set_device(idx)
    except Exception as e:
        print(f"[device][warn] torch.cuda.set_device({idx}) failed: {e}; fallback to cuda:0", flush=True)
        idx = 0
        torch.cuda.set_device(idx)
    return f"cuda:{idx}"


# ---------------- Binary-first defaults (enable list/rank later via env overrides) ----------------
from config_topk import (
    Params,
    build_model_config,
    infer_rbp_dataset_tag,
    validate_rbp_dataset_config,
)

# ---- optional AMP (off by default; enable with AMP=1) ----
USE_AMP = (os.environ.get('AMP', '0') not in ('0', 'false', 'False', '')) and torch.cuda.is_available() and (
        autocast is not None) and (GradScaler is not None)
AMP_DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16


# ---------------- EMA (stabilize Top-K, cheap gain) ----------------
class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow = {}
        self.backup = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = p.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if n not in self.shadow:
                self.shadow[n] = p.detach().clone()
            else:
                self.shadow[n].mul_(d).add_(p.detach(), alpha=(1.0 - d))

    @torch.no_grad()
    def apply_shadow(self, model: nn.Module):
        self.backup = {}
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.backup[n] = p.detach().clone()
                p.data.copy_(self.shadow[n].data)

    @torch.no_grad()
    def restore(self, model: nn.Module):
        for n, p in model.named_parameters():
            if n in self.backup:
                p.data.copy_(self.backup[n].data)
        self.backup = {}


def _loss_is_bad(loss_t: torch.Tensor, thr: float = 1e6):
    if not torch.is_tensor(loss_t):
        return True, "not_tensor"
    if not torch.isfinite(loss_t).all():
        return True, "non_finite"
    v = float(loss_t.detach().cpu())
    if (not np.isfinite(v)) or (abs(v) > float(thr)):
        return True, f"abs>{thr:g}"
    return False, ""


# ---------------- Params resolution (cfg-first) ----------------
def _get_model_cfg(model):
    for attr in ('cfg', 'config', 'model_cfg'):
        if hasattr(model, attr):
            return getattr(model, attr)
    return None


# ====================== LR Scheduler (Warmup + Cosine) ======================
# ============================================================================

def build_warmup_cosine_scheduler(opt, total_updates: int, base_lr: float, P: Params):
    """
    Warmup + cosine LR scheduler (update-step based; works with ACCUM_STEPS).

    Design goals:
      - Avoid "frozen" early training when total_updates is small or warmup is mis-set.
      - Keep behavior backward compatible with the 0.75+ stable version.
      - Allow optional knobs (if present in Params):
            lr_warmup_start_ratio (default 0.0)
            lr_warmup_min_updates (default 10)
            lr_warmup_max_updates (default max(200, 2*steps_per_epoch) if steps_per_epoch is known)
    """
    if not bool(P.lr_sched):
        return None, {"enabled": False}

    total_updates = int(max(1, total_updates))
    warmup_frac = float(P.lr_warmup_frac)
    min_ratio = float(P.lr_min_ratio)
    cycles = max(1, int(P.lr_cosine_cycles))

    # ---- warmup length clamp ----
    warmup_updates = int(round(max(0.0, warmup_frac) * total_updates))
    warmup_min = int(P.lr_warmup_min_updates)
    steps_per_epoch = P.steps_per_epoch
    default_max = 200
    if isinstance(steps_per_epoch, int) and steps_per_epoch > 0:
        default_max = max(default_max, 2 * int(steps_per_epoch))
    warmup_max = int(P.lr_warmup_max_updates)
    warmup_updates = max(warmup_min, warmup_updates)
    warmup_updates = min(max(1, warmup_max), warmup_updates)
    warmup_updates = min(total_updates, warmup_updates)

    # ---- warmup start ratio ----
    start_ratio = float(P.lr_warmup_start_ratio)
    start_ratio = max(0.0, min(1.0, start_ratio))

    def lr_lambda(step):
        if step < warmup_updates:
            # warmup
            if warmup_updates <= 1:
                return 1.0
            alpha = float(step) / float(warmup_updates)
            warmup_start_ratio = getattr(P, 'lr_warmup_start_ratio', 0.0)
            return warmup_start_ratio + (1.0 - warmup_start_ratio) * alpha
        else:
            # cosine
            progress = (step - warmup_updates) / float(max(1, total_updates - warmup_updates))
            cosine_lr = 0.5 * (1.0 + math.cos(math.pi * progress))

            lr_min_ratio = getattr(P, 'lr_min_ratio', 0.4)
            return max(lr_min_ratio, cosine_lr)

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
    info = {
        "enabled": True,
        "total_updates": int(total_updates),
        "warmup_updates": int(warmup_updates),
        "warmup_frac": float(warmup_frac),
        "warmup_start_ratio": float(start_ratio),
        "min_ratio": float(min_ratio),
        "cycles": int(cycles),
        "base_lr": float(base_lr),
    }
    return sched, info


# ====================== L2 stability & ESM cache ======================
def _sample_l2_pixels(
        logits_flat: torch.Tensor,
        labels_flat: torch.Tensor,
        neg_budget: int = 4096,  # [FIX]  budget
        min_pos: int = 8,
        hard_frac: float = 0.70,
        max_neg=None,
        neg_per_pos: int = 20,
        neg_min: int = 1024,
        pos_cap: int = None,
):
    if max_neg is not None: neg_budget = int(max_neg)

    with torch.no_grad():
        pos_idx = (labels_flat > 0.5).nonzero(as_tuple=False).squeeze(-1)
        neg_idx = (labels_flat <= 0.5).nonzero(as_tuple=False).squeeze(-1)
        n_pos = int(pos_idx.numel())

        pos_cap_eff = 4096 if (pos_cap is None) else int(pos_cap)
        if pos_cap_eff > 0 and n_pos > pos_cap_eff:
            perm = torch.randperm(n_pos, device=labels_flat.device)[:pos_cap_eff]
            pos_idx = pos_idx[perm]
            n_pos = int(pos_idx.numel())

        # 
        k_neg = max(int(neg_min), int(n_pos * neg_per_pos))
        k_neg = min(k_neg, neg_budget)
        k_neg = min(k_neg, int(neg_idx.numel()))

        if k_neg <= 0:
            return pos_idx, int(pos_idx.numel()), 0

        # [FIX] ard + Random
        k_hard = int(k_neg * hard_frac)
        k_rand = k_neg - k_hard

        keep_neg_list = []
        if k_hard > 0:
            # 1.  (Hard Mining)
            neg_scores = logits_flat[neg_idx]
            _, top_k_idx = torch.topk(neg_scores, k=k_hard)
            keep_neg_list.append(neg_idx[top_k_idx])

            if k_rand > 0:
                mask = torch.ones(neg_idx.numel(), dtype=torch.bool, device=neg_idx.device)
                mask[top_k_idx] = False
                neg_idx_rest = neg_idx[mask]
            else:
                neg_idx_rest = torch.tensor([], device=neg_idx.device)
        else:
            neg_idx_rest = neg_idx

        # 3. 
        if k_rand > 0 and neg_idx_rest.numel() > 0:
            perm = torch.randperm(neg_idx_rest.numel(), device=neg_idx.device)[:k_rand]
            keep_neg_list.append(neg_idx_rest[perm])

        if len(keep_neg_list) > 0:
            keep_neg = torch.cat(keep_neg_list, dim=0)
            keep = torch.cat([pos_idx, keep_neg], dim=0)
        else:
            keep = pos_idx

    return keep, int(pos_idx.numel()), int(keep_neg.numel())


def _ohem_keep_stats(y_sel: torch.Tensor):
    # y_sel: selected labels after OHEM (0/1 float)
    if y_sel is None or (not torch.is_tensor(y_sel)) or y_sel.numel() == 0:
        return 0, 0, 0
    npos_sel = int((y_sel > 0.5).sum().item())
    nsel = int(y_sel.numel())
    nneg_sel = int(nsel - npos_sel)
    return npos_sel, nneg_sel, nsel  # keepPos, keepNeg, keepPix


def _topk_margin_rank_loss(logits: torch.Tensor, labels: torch.Tensor, k_neg: int = 50, margin: float = 0.5,
                           k_pos: int = 50):
    """Top-K boundary margin ranking (directly aligned to precision@K).

    We compare *thresholds* rather than means:
      pos_ref = K-th largest positive logit (i.e., the boundary positive you need inside Top-K)
      neg_ref = K-th largest negative logit (i.e., the boundary negative you must beat)

    Objective: pos_ref >= neg_ref + margin
    Loss: softplus(margin + neg_ref - pos_ref)
    """
    pos = labels > 0.5
    neg = ~pos

    npos = int(pos.sum().item())
    nneg = int(neg.sum().item())
    if npos == 0 or nneg == 0:
        return torch.tensor(0.0, device=logits.device)

    pos_logits = logits[pos]
    neg_logits = logits[neg]

    kp = min(int(k_pos), int(pos_logits.numel()))
    kn = min(int(k_neg), int(neg_logits.numel()))
    if kp <= 0 or kn <= 0:
        return torch.tensor(0.0, device=logits.device)

    # K-th largest => sorted=True and take last element
    pos_ref = torch.topk(pos_logits, k=kp, largest=True, sorted=True).values[-1]
    neg_ref = torch.topk(neg_logits, k=kn, largest=True, sorted=True).values[-1]

    return F.softplus(float(margin) + neg_ref - pos_ref)


# ---------------- Extra L2 losses (optional; default off) ----------------
def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    # gt_sorted: [P] in {0,1}, sorted by errors desc
    p = gt_sorted.numel()
    if p == 0:
        return gt_sorted
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1.0 - gt_sorted.float()).cumsum(0)
    jaccard = 1.0 - intersection / union.clamp_min(1e-12)
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_hinge_flat(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Binary Lovasz hinge loss for a flat vector."""
    if logits.numel() == 0:
        return torch.zeros((), device=logits.device, dtype=logits.dtype)
    labels = (labels > 0.5).float()
    if (labels.max() == labels.min()):
        # undefined IoU gradient (all-0 or all-1); return 0 to avoid noise
        return torch.zeros((), device=logits.device, dtype=logits.dtype)
    signs = 2.0 * labels - 1.0
    errors = 1.0 - logits * signs
    errors_sorted, perm = torch.sort(errors, descending=True)
    gt_sorted = labels[perm]
    grad = _lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), grad.to(errors_sorted.dtype))
    return loss


def soft_dice_loss_from_logits(logits: torch.Tensor, labels: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Soft Dice on a flat vector."""
    if logits.numel() == 0:
        return torch.zeros((), device=logits.device, dtype=logits.dtype)
    y = (labels > 0.5).float()
    p = torch.sigmoid(logits)
    num = 2.0 * (p * y).sum() + eps
    den = p.sum() + y.sum() + eps
    return 1.0 - (num / den)


def l1_pairwise_rank_loss(
        logits: torch.Tensor,
        labels: torch.Tensor,
        margin: float = 0.5,
        n_pairs: int = 128,
        hard_neg_frac: float = 0.5,
) -> torch.Tensor:
    """
    Pairwise margin ranking loss for residue logits.

    Encourages positive residue logits to exceed negative residue logits by a
    margin. A mix of random and hard-negative pairs is used so the loss aligns
    better with top-k recall.
    """
    if logits.numel() == 0:
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

    pos_mask = (labels > 0.5)
    neg_mask = ~pos_mask
    n_pos = int(pos_mask.sum().item())
    n_neg = int(neg_mask.sum().item())

    if n_pos == 0 or n_neg == 0:
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

    pos_logits = logits[pos_mask]   # [n_pos]
    neg_logits = logits[neg_mask]   # [n_neg]

    # Split pairs: half random, half hard-negative
    n_hard = int(n_pairs * float(hard_neg_frac))
    n_rand = n_pairs - n_hard

    pairs_pos = []
    pairs_neg = []

    # Random pairs
    if n_rand > 0:
        pi = torch.randint(0, n_pos, (n_rand,), device=logits.device)
        ni = torch.randint(0, n_neg, (n_rand,), device=logits.device)
        pairs_pos.append(pos_logits[pi])
        pairs_neg.append(neg_logits[ni])

    # Hard-negative pairs: take top-k neg by logit
    if n_hard > 0:
        k_hard = min(n_hard, n_neg)
        hard_neg_idx = torch.topk(neg_logits, k=k_hard, largest=True, sorted=False).indices
        hard_neg_sel = neg_logits[hard_neg_idx]   # [k_hard]
        pi = torch.randint(0, n_pos, (k_hard,), device=logits.device)
        pairs_pos.append(pos_logits[pi])
        pairs_neg.append(hard_neg_sel)

    all_pos = torch.cat(pairs_pos)   # [n_pairs]
    all_neg = torch.cat(pairs_neg)   # [n_pairs]

    # Hinge loss: max(0, margin - (pos - neg))
    loss = torch.clamp(float(margin) - (all_pos - all_neg), min=0.0)
    return loss.mean()


def l1_listmle_loss(
        logits: torch.Tensor,
        labels: torch.Tensor,
        tau: float = 2.0,
        neg_k: int = 64,
) -> torch.Tensor:
    """
    Hybrid listwise loss for residue evidence ranking.

    Part A uses softplus pairwise comparisons between positive residues and a
    mixture of hard / random negatives. Part B adds a cheap listwise term that
    concentrates probability mass on positives and improves top-k behavior.
    """
    if logits.numel() == 0:
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

    pos_mask = (labels > 0.5)
    neg_mask = ~pos_mask
    n_pos = int(pos_mask.sum().item())
    n_neg = int(neg_mask.sum().item())

    if n_pos == 0 or n_neg == 0:
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

    pos_logits = logits[pos_mask]  # [n_pos]
    neg_logits = logits[neg_mask]  # [n_neg]

    #  Part A: Softplus Pairwise 
    n_pairs = min(256, n_pos * n_neg)
    k_hard  = min(int(neg_k), n_neg)
    hard_idx = torch.topk(neg_logits, k=k_hard, largest=True, sorted=False).indices
    hard_neg = neg_logits[hard_idx]

    n_hard = int(n_pairs * 0.7)
    n_rand = n_pairs - n_hard

    pi_h = torch.randint(0, n_pos,  (n_hard,), device=logits.device)
    ni_h = torch.randint(0, k_hard, (n_hard,), device=logits.device)
    pi_r = torch.randint(0, n_pos,  (n_rand,), device=logits.device)
    ni_r = torch.randint(0, n_neg,  (n_rand,), device=logits.device)

    sampled_pos = torch.cat([pos_logits[pi_h], pos_logits[pi_r]])
    sampled_neg = torch.cat([hard_neg[ni_h],   neg_logits[ni_r]])
    loss_sp = F.softplus(sampled_neg - sampled_pos).mean()

    #  Part B: ApproxNDCG 
    t = float(max(tau, 0.5))
    lse_all = torch.logsumexp(logits / t, dim=0)
    lse_pos = torch.logsumexp(pos_logits / t, dim=0)
    loss_ndcg = lse_all - lse_pos

    return 0.7 * loss_sp + 0.3 * loss_ndcg


def binary_focal_loss_with_logits(
        logits: torch.Tensor, labels: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'
) -> torch.Tensor:
    """Focal loss on a flat vector."""
    if logits.numel() == 0:
        return torch.zeros((), device=logits.device, dtype=logits.dtype)
    y = (labels > 0.5).float()
    ce = F.binary_cross_entropy_with_logits(logits, y, reduction='none')
    p = torch.sigmoid(logits)
    p_t = p * y + (1.0 - p) * (1.0 - y)
    alpha_t = alpha * y + (1.0 - alpha) * (1.0 - y)
    loss = alpha_t * ((1.0 - p_t).clamp_min(1e-6) ** gamma) * ce
    if reduction == 'sum':
        return loss.sum()
    if reduction == 'none':
        return loss
    return loss.mean()


def _rank_gap_stats(logits: torch.Tensor, labels: torch.Tensor, k_neg: int = 50, k_pos: int = 50):
    """Logging: (pos_ref - neg_ref) at Top-K boundary."""
    pos = labels > 0.5
    neg = ~pos
    if int(pos.sum().item()) == 0 or int(neg.sum().item()) == 0:
        return torch.tensor(0.0, device=logits.device)
    pos_logits = logits[pos]
    neg_logits = logits[neg]
    kp = min(int(k_pos), int(pos_logits.numel()))
    kn = min(int(k_neg), int(neg_logits.numel()))
    if kp <= 0 or kn <= 0:
        return torch.tensor(0.0, device=logits.device)
    pos_ref = torch.topk(pos_logits, k=kp, largest=True, sorted=True).values[-1]
    neg_ref = torch.topk(neg_logits, k=kn, largest=True, sorted=True).values[-1]
    return (pos_ref - neg_ref)


def _listwise_pos_mass_loss(logits: torch.Tensor, labels: torch.Tensor, neg_k=1024, tau=1.0, neg_push_w=1.0):
    pos = (labels > 0.5)
    npos = int(pos.sum().item())

    if npos == 0:
        return neg_push_w * F.softplus(logits).mean()

    neg = ~pos
    nneg = int(neg.sum().item())

    if nneg > 0:
        k = min(int(neg_k), nneg)
        neg_logits = logits[neg]
        hard_idx = torch.topk(neg_logits, k=k, largest=True).indices
        cand = torch.cat([logits[pos], neg_logits[hard_idx]], dim=0)
        pos_mask = torch.cat([
            torch.ones(npos, device=logits.device, dtype=torch.bool),
            torch.zeros(k, device=logits.device, dtype=torch.bool)
        ], dim=0)
    else:
        cand = logits[pos]
        pos_mask = torch.ones_like(cand, dtype=torch.bool)

    logp = F.log_softmax(cand / float(tau), dim=0)
    return -torch.logsumexp(logp[pos_mask], dim=0)


# ---------------- Robust helpers ----------------
def _to_tensor(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        return x
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    return torch.as_tensor(x)


def _valid2d_and_y2d(batch):
    """Return (y2d, valid2d). valid2d is derived from masks if missing."""
    y2d = batch.get("y2d", None)
    valid2d = batch.get("valid2d", None)

    if valid2d is None:
        # Derive from maskA/maskB when dataset provides per-residue masks but not a 2D valid mask.
        mA = batch.get("maskA", None)
        mB = batch.get("maskB", None)
        if (mA is not None) and (mB is not None):
            # support both single-sample (La) and batched (B,La)
            if mA.dim() == 1:
                valid2d = (mA[:, None] > 0.5) & (mB[None, :] > 0.5)
            else:
                valid2d = (mA[:, :, None] > 0.5) & (mB[:, None, :] > 0.5)
            valid2d = valid2d.to(torch.bool)

    return y2d, valid2d


def sample_has_pos_rate(batch, rate_thr=0.01):
    """Estimate whether a sample contains enough positives for weighted sampling.

    For RBP296 we use residue labels when available. For legacy DIPS-style
    samples we fall back to the 2D contact map.
    """
    # ---- RBP296: residue labels first ----
    y_res_A = batch.get("y_res_A", None)
    if torch.is_tensor(y_res_A) and y_res_A.numel() > 0:
        y_flat = y_res_A.float().reshape(-1)
        npos = int((y_flat > 0.5).sum().item())
        nall = int(y_flat.numel())
        rate = npos / max(1, nall)
        return (rate >= float(rate_thr)) and (npos > 0), npos

    # ---- DIPS fallback: y2d ----
    y2d, valid2d = _valid2d_and_y2d(batch)
    if y2d is None or valid2d is None or (not bool(valid2d.any())):
        return False, 0
    npos = int((y2d > 0.5).sum().item())
    nall = int(valid2d.sum().item())
    rate = npos / max(1, nall)
    return (rate >= float(rate_thr)) and (npos > 0), npos

def build_pos_aware_weights(subset, rate_thr=0.01, p_target=0.80,
                            base_w_pos=1.0, base_w_neg=1.0,
                            pos_oversample=8.0, max_ratio=50.0):
    """Build per-sample weights for positive-aware sampling."""
    n = len(subset)
    flags = np.zeros(n, dtype=np.int8)
    for i in range(n):
        item = subset[i]
        has_pos, _ = sample_has_pos_rate(item, rate_thr=rate_thr)
        flags[i] = 1 if has_pos else 0

    f = float(flags.mean()) if n > 0 else 0.0
    if f <= 1e-8:
        w_pos = 1.0
        w_neg = 1.0
    else:
        r = (p_target * (1.0 - f)) / (max(f, 1e-8) * (1.0 - p_target))
        r = float(max(1.0, min(r, max_ratio)))
        w_pos = base_w_pos * r * pos_oversample
        w_neg = base_w_neg

    weights = np.where(flags == 1, w_pos, w_neg).astype(np.float32)
    stats = {
        "n": n, "has_pos_rate": f, "w_pos": float(w_pos), "w_neg": float(w_neg),
        "rate_thr": float(rate_thr),
    }
    return torch.from_numpy(weights), stats, flags.astype(np.bool_)


# ----------------- Config (RBP296) -----------------
P = Params.from_env()
_RBP_DATASET_TAG = infer_rbp_dataset_tag(P)
print(f"[config] inferred_rbp_dataset={_RBP_DATASET_TAG}", flush=True)
for _warn in validate_rbp_dataset_config(P):
    print(f"[config][warn] {_warn}", flush=True)
# ---- validation threshold control (binary metrics) ----
VAL_THR_MODE = str(P.val_thr_mode)
VAL_THR_FIXED = float(P.val_thr)
VAL_THR_MIN = float(P.val_thr_min)
VAL_THR_MAX = float(P.val_thr_max)
VAL_THR_GRID = int(P.val_thr_grid)

# ---- primary objective ----
# "binary" -> best ckpt by AUPRC, threshold by MCC  (default)
# "topk"   -> best ckpt by TopL/10 Recall (positive-only proteins)
PRIMARY_OBJ = str(getattr(P, 'primary_objective', 'binary')).strip().lower()
if PRIMARY_OBJ not in ('binary', 'topk'):
    print(f"[warn] unknown primary_objective={PRIMARY_OBJ!r}, fallback to 'binary'", flush=True)
    PRIMARY_OBJ = 'binary'

# ---- paths (RBP296) ----
RBP_ROOT    = P.rbp_root
RBP_ID_LIST = P.rbp_id_list

ESM_LOCAL_DIR = P.esm_local_dir
SAVE_DIR = P.save_dir

# ---- dirs ----
ESM_HUB_CACHE = os.path.join(ESM_LOCAL_DIR, "hub")
os.makedirs(ESM_HUB_CACHE, exist_ok=True)
os.environ.setdefault("TORCH_HOME", ESM_HUB_CACHE)
os.environ.setdefault("TRANSFORMERS_CACHE", ESM_HUB_CACHE)
os.environ.setdefault("HF_HOME", ESM_HUB_CACHE)
os.environ.setdefault("XDG_CACHE_HOME", ESM_HUB_CACHE)

os.makedirs(SAVE_DIR, exist_ok=True)
CKPT_DIR = os.path.join(SAVE_DIR, "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)
# best ckpt path depends on primary objective (set after PRIMARY_OBJ is defined)
BEST_CKPT = os.path.join(CKPT_DIR, ("best_TOPK.pt" if PRIMARY_OBJ == "topk" else "best_AUPRC.pt"))

# ---- train (from Params) ----
SEED = P.seed
PRINT_EVERY = P.print_every
WARMUP_PRINT_STEPS = P.warmup_print_steps

BATCH_SITE = P.batch_site
NUM_WORKERS_SITE = P.num_workers
EPOCHS = P.epochs

LR = P.lr
WEIGHT_DECAY = P.weight_decay
MAX_GRAD_NORM = P.max_grad_norm
ACCUM_STEPS = P.accum_steps

CONTACT_CUTOFF = P.contact_cutoff
DEVICE = _resolve_runtime_device(default_cuda_index=2)
# Default ESM to CPU to avoid burning GPU memory before the main model is built.
ESM_DEVICE = os.environ.get("ESM_DEVICE", ("cpu" if DEVICE.startswith("cuda") else DEVICE)).strip().lower() or ("cpu" if DEVICE.startswith("cuda") else DEVICE)

# ---- crop/mining (from Params) ----
MAX_2D_TOKENS_TRAIN = P.max_2d_tokens_train
MAX_2D_TOKENS_EVAL = P.max_2d_tokens_eval
MIN_SIDE = P.min_side
MAX_SIDE = P.max_side

L2_ANCHOR_PROB = P.l2_anchor_prob
L2_FIXED_CROP = bool(getattr(P, 'l2_fixed_crop', False))
L2_FIXED_SIDE = int(getattr(P, 'l2_fixed_side', 0))
L2_REJECT_ALLNEG_P = float(getattr(P, 'l2_reject_allneg_p', getattr(P, 'l2_allneg_drop_p', 0.0)))
L2_REJECT_ALLNEG_MAX_TRIES = int(getattr(P, 'l2_reject_allneg_max_tries', 8))
L2_FORCE_ANCHOR_IF_HASPOS = bool(getattr(P, 'l2_force_anchor_if_haspos', True))
L2_HARDNEG_FRAC = P.l2_hardneg_frac
POS_OVERSAMPLE = P.pos_oversample

# ---- sampler state (cached for per-epoch updates) ----
_L2_SAMPLER_FLAGS = None  # np bool array: which samples are pos-bearing
_L2_SAMPLER_HASPOS_RATE = 0.0  # fraction of pos-bearing samples
_L2_SAMPLER_SAMPLER = None  # WeightedRandomSampler instance
_L2_SAMPLER_WEIGHTS = None  # current weight tensor
EARLY_MIN_EPOCHS = P.early_min_epochs
EARLY_PATIENCE = P.early_patience
EARLY_MIN_DELTA = P.early_min_delta
EARLY_EMA = P.early_ema

TOP5_GUARD_WINDOW = P.top5_guard_window
TOP5_GUARD_DROP = P.top5_guard_drop

# ---- external deps (your repo) ----
from train_sitepairs import SeqEmbedder as SiteEmbedder
from model import UnifiedInterfaceModel


def _gpu_mem_snapshot(device: str = DEVICE):
    if (not str(device).startswith("cuda")) or (not torch.cuda.is_available()):
        return None
    try:
        free_b, total_b = torch.cuda.mem_get_info()
        return {
            "free_gb": float(free_b) / (1024 ** 3),
            "total_gb": float(total_b) / (1024 ** 3),
            "allocated_gb": float(torch.cuda.memory_allocated()) / (1024 ** 3),
            "reserved_gb": float(torch.cuda.memory_reserved()) / (1024 ** 3),
        }
    except Exception:
        return None


def _print_gpu_mem_snapshot(tag: str, device: str = DEVICE):
    snap = _gpu_mem_snapshot(device)
    if snap is None:
        return
    print(
        f"[gpu] {tag} free={snap['free_gb']:.2f}GiB total={snap['total_gb']:.2f}GiB "
        f"allocated={snap['allocated_gb']:.2f}GiB reserved={snap['reserved_gb']:.2f}GiB",
        flush=True,
    )


def _topk_primary_score(metrics: dict) -> float:
    """Composite score aligned with final reported Top-K table."""
    topk_po = metrics.get("topk_posonly", {}) or {}

    def _get(subkey: str, field: str) -> float:
        obj = topk_po.get(subkey, {})
        if not isinstance(obj, dict):
            return 0.0
        return float(obj.get(field, 0.0))

    l5 = _get("L5", "rec")
    l10 = _get("L10", "rec")
    k10p = _get("10", "prec")
    hit20 = _get("20", "hit")
    score = (
        float(getattr(P, "topk_score_w_l5", 0.35)) * l5
        + float(getattr(P, "topk_score_w_l10", 0.35)) * l10
        + float(getattr(P, "topk_score_w_k10p", 0.20)) * k10p
        + float(getattr(P, "topk_score_w_hit20", 0.10)) * hit20
    )
    return float(score)


def _infer_rbp_feature_dims():
    """Infer input dims without forcing one DataLoader iteration.

    For the current RBP residue feature stack:
      ESM(1280) + optional PSSM(20) + optional DSSP(9)
    Chain-level features are not used in the current single-chain setup.
    """
    d_res = 1280
    if bool(getattr(P, "use_pssm", False)):
        d_res += 20
    if bool(getattr(P, "use_dssp", False)):
        d_res += 9
    d_chain = 0
    return int(d_res), int(d_chain)


def set_seed(s: int):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def save_ckpt(path, model, opt=None, combo=None, top50=None, medauc=None, epoch: int = 0, scaler=None, cfg=None,
              extra=None):
    os.makedirs(os.path.dirname(str(path)) or ".", exist_ok=True)
    ckpt = {"epoch": int(epoch), "model_state": model.state_dict()}
    if opt is not None:
        ckpt["opt_state"] = opt.state_dict()
    if scaler is not None:
        ckpt["scaler_state"] = scaler.state_dict()
    if cfg is not None:
        ckpt["cfg"] = cfg
    if combo is not None:
        ckpt["combo"] = combo
    if top50 is not None:
        ckpt["top50"] = top50
    if medauc is not None:
        ckpt["medauc"] = medauc
    if extra:
        ckpt["extra"] = extra
    torch.save(ckpt, str(path))


def load_ckpt(path, model, opt=None):
    print(f"[ckpt] loading from {path} ...", flush=True)
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    if opt is not None and "opt_state" in ckpt:
        opt.load_state_dict(ckpt["opt_state"])
    return ckpt.get("combo", 0.0), ckpt.get("medauc", 0.0), ckpt.get("epoch", 0)


def compute_medauc_from_batch_logits(logits2d: torch.Tensor, batch: dict) -> list:
    # returns per-sample AUC list (python floats), skip undefined

    y2d, valid2d = _valid2d_and_y2d(batch)
    if y2d is None or valid2d is None:
        return []
    if logits2d.ndim == 2:
        logits2d = logits2d.unsqueeze(0)
    prob = torch.sigmoid(logits2d.detach())
    aucs = []
    B = prob.shape[0]
    for b in range(B):
        vb = valid2d[b].bool()
        y = y2d[b][vb].detach().float().cpu().numpy()
        s = prob[b][vb].detach().cpu().numpy()
        if y.size == 0:
            continue
        if (y.max() == y.min()):
            continue  # all-0 or all-1, undefined AUC
        aucs.append(float(roc_auc_score(y, s)))
    return aucs


def dips_collate(batch):
    # ===== COLLATE ALIGN CHECK =====
    for i, d in enumerate(batch):
        y2d = d.get("y2d", None)
        rA = d.get("resA", None)
        rB = d.get("resB", None)

        if y2d is None or rA is None or rB is None:
            continue

        if y2d.shape[0] != rA.shape[0] or y2d.shape[1] != rB.shape[0]:
            raise RuntimeError(
                f"[COLLATE-ALIGN-ERROR] idx={i} "
                f"A={tuple(rA.shape)} B={tuple(rB.shape)} y2d={tuple(y2d.shape)}"
            )

    out = {}
    keys = batch[0].keys()

    def _pad_1d(x, Lmax):
        return F.pad(x, (0, Lmax - x.shape[0]))

    def _pad_2d_firstdim(x, Lmax):
        pad_len = Lmax - x.shape[0]
        return F.pad(x, (0, 0, 0, pad_len))

    for k in keys:
        v0 = batch[0][k]

        if k == "y2d":
            max_L = max(d[k].shape[0] for d in batch)
            max_M = max(d[k].shape[1] for d in batch)
            padded = []
            for d in batch:
                x = d[k]
                pad_L = max_L - x.shape[0]
                pad_M = max_M - x.shape[1]
                x = F.pad(x, (0, pad_M, 0, pad_L))
                padded.append(x)
            out[k] = torch.stack(padded)
            continue

        if k in ["resA", "resB", "chainA", "chainB"]:
            max_L = max(d[k].shape[0] for d in batch)
            padded = []
            for d in batch:
                x = d[k]
                if x.ndim != 2:
                    raise RuntimeError(f"[collate] {k} expect [L,D], got {tuple(x.shape)}")
                padded.append(_pad_2d_firstdim(x, max_L))
            out[k] = torch.stack(padded)
            continue

        if k in ["maskA", "maskB", "y_res_A", "y_res_B"]:
            max_L = max(d[k].shape[0] for d in batch)
            out[k] = torch.stack([_pad_1d(d[k], max_L) for d in batch])
            continue

        if torch.is_tensor(v0):
            out[k] = torch.stack([d[k] for d in batch])
        else:
            out[k] = [d[k] for d in batch]

    return out


# ------------------- DIPS indexed dataset -------------------
AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {a: i for i, a in enumerate(AA_ORDER)}
UNK_AA_IDX = 20

# ---- focus cfg (single source of truth) ----
_cfg_focus_epochs = int(getattr(P, 'l2_focus_epochs', 0))
_cfg_boost_w = float(getattr(P, 'l2_focus_rankw', 1.5))
_cfg_list_focus = float(getattr(P, 'l2_focus_listw', 1.2))
_cfg_margin_focus = float(getattr(P, 'l2_focus_margin', 1.5))

print(
    f"[cfg][focus] epochs={_cfg_focus_epochs} boost_w={_cfg_boost_w} list_w={_cfg_list_focus} margin_x={_cfg_margin_focus}",
    flush=True)


def _read_fasta_one(seq_path: str) -> str:
    if not seq_path or (not os.path.exists(seq_path)):
        return ""
    seq = []
    with open(seq_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line:
                continue
            if line.startswith(">"):
                continue
            seq.append(line.strip())
    return "".join(seq).strip()


def _read_fasta_two(path: str):
    if not path or (not os.path.exists(path)):
        return "", ""
    seqs = []
    cur = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur:
                    seqs.append("".join(cur))
                    cur = []
                continue
            cur.append(line)
        if cur:
            seqs.append("".join(cur))
    if len(seqs) == 0:
        return "", ""
    if len(seqs) == 1:
        return seqs[0], ""
    return seqs[0], seqs[1]


def _load_pssm(pssm_path: str, L: int) -> torch.Tensor:
    """Load PSSM [L,20]. Supports .npy (RBP296) and .npz (DIPS). Zero-fill on missing."""
    if pssm_path is None or (not os.path.exists(pssm_path)):
        return torch.zeros((L, 20), dtype=torch.float32)
    try:
        # ---- .npy (RBP296 format) ----
        if str(pssm_path).endswith('.npy'):
            arr = np.load(pssm_path).astype(np.float32)
        else:
            # ---- .npz (DIPS format) ----
            z = np.load(pssm_path)
            key = None
            for k in ("pssm", "PSSM", "arr_0"):
                if k in z.files:
                    key = k
                    break
            if key is None and len(z.files) > 0:
                key = z.files[0]
            arr = np.asarray(z[key], dtype=np.float32)
    except Exception:
        return torch.zeros((L, 20), dtype=torch.float32)

    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.shape[0] != L:
        arr = arr[:L] if arr.shape[0] > L else np.pad(arr, ((0, L - arr.shape[0]), (0, 0)), mode="constant")
    if arr.shape[1] != 20:
        arr = arr[:, :20] if arr.shape[1] > 20 else np.pad(arr, ((0, 0), (0, 20 - arr.shape[1])), mode="constant")
    return torch.from_numpy(arr)


def _load_dssp(dssp_path: str, L: int) -> torch.Tensor:
    """Load DSSP [L,9]. Supports .npy (RBP296) and TSV text (DIPS). Zero-fill on missing."""
    if dssp_path is None or (not os.path.exists(dssp_path)):
        return torch.zeros((L, 9), dtype=torch.float32)

    # ---- .npy (RBP296 format) ----
    if str(dssp_path).endswith('.npy'):
        try:
            arr = np.load(dssp_path).astype(np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            if arr.shape[0] != L:
                arr = arr[:L] if arr.shape[0] > L else np.pad(arr, ((0, L - arr.shape[0]), (0, 0)), mode="constant")
            if arr.shape[1] != 9:
                arr = arr[:, :9] if arr.shape[1] > 9 else np.pad(arr, ((0, 0), (0, 9 - arr.shape[1])), mode="constant")
            return torch.from_numpy(arr)
        except Exception:
            return torch.zeros((L, 9), dtype=torch.float32)

    # ---- .npz ----
    if str(dssp_path).endswith('.npz'):
        try:
            z = np.load(dssp_path)
            key = None
            for k in ("dssp", "DSSP", "arr_0"):
                if k in z.files:
                    key = k
                    break
            if key is None and len(z.files) > 0:
                key = z.files[0]
            arr = np.asarray(z[key], dtype=np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            if arr.shape[0] != L:
                arr = arr[:L] if arr.shape[0] > L else np.pad(arr, ((0, L - arr.shape[0]), (0, 0)), mode="constant")
            if arr.shape[1] != 9:
                arr = arr[:, :9] if arr.shape[1] > 9 else np.pad(arr, ((0, 0), (0, 9 - arr.shape[1])), mode="constant")
            return torch.from_numpy(arr)
        except Exception:
            return torch.zeros((L, 9), dtype=torch.float32)

    # ---- TSV text (DIPS format) ----
    feats = []
    try:
        with open(dssp_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                s = line.strip()
                if (not s) or s.startswith("#"):
                    continue
                parts = s.split("\t")
                nums = []
                for p in parts:
                    try:
                        nums.append(float(p))
                    except Exception:
                        continue
                if len(nums) > 0:
                    feats.append(nums)
    except Exception:
        return torch.zeros((L, 9), dtype=torch.float32)
    if len(feats) == 0:
        return torch.zeros((L, 9), dtype=torch.float32)

    arr = np.asarray(feats, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.shape[0] != L:
        arr = arr[:L] if arr.shape[0] > L else np.pad(arr, ((0, L - arr.shape[0]), (0, 0)), mode="constant")
    if arr.shape[1] != 9:
        arr = arr[:, :9] if arr.shape[1] > 9 else np.pad(arr, ((0, 0), (0, 9 - arr.shape[1])), mode="constant")
    return torch.from_numpy(arr)


def _run_embedder(embedder, cid, tag, seq_str: str):
    """Run the sequence embedder with flexible call signatures.

    Different embedder implementations expose different APIs:
      - encode(seq)
      - encode(pid, seq)
      - __call__(seq)

    We try a few common signatures in a safe order.
    """
    if embedder is None:
        raise RuntimeError('[ESM] embedder is None')

    key = f"{cid}_{tag}"
    last_err = None

    # Preference order: explicit methods first, then callable
    for name in ("encode", "embed", "forward", "compute", "__call__"):
        if not hasattr(embedder, name):
            continue
        fn = getattr(embedder, name)
        if not callable(fn):
            continue

        # Try (pid, seq) first (needed by train_sitepairs.SeqEmbedder.encode)
        for args in ((key, seq_str), (cid, seq_str), (tag, seq_str), (seq_str,)):
            try:
                return fn(*args)
            except TypeError as e:
                last_err = e
                continue

        raise TypeError(f"[ESM] '{name}' signature mismatch. Last error: {last_err}")

    raise AttributeError('[ESM] embedder has no callable encode/embed/forward/compute/__call__')


def _load_pair_npz_full(path: str):
    with np.load(path, allow_pickle=True) as z:
        caL = z["caL"] if "caL" in z else (z["XL"] if "XL" in z else None)
        caR = z["caR"] if "caR" in z else (z["XR"] if "XR" in z else None)

        # seq: either in npz or empty
        seqL = str(z["seqL"]) if "seqL" in z else ""
        seqR = str(z["seqR"]) if "seqR" in z else ""

    if caL is None or caR is None:
        return None, None, seqL, seqR

    caL = np.asarray(caL, dtype=np.float32)
    caR = np.asarray(caR, dtype=np.float32)

    # normalize to [L,3]
    if caL.ndim == 3: caL = caL[:, 0, :]
    if caR.ndim == 3: caR = caR[:, 0, :]
    if caL.ndim == 1: caL = caL.reshape(-1, 3)
    if caR.ndim == 1: caR = caR.reshape(-1, 3)

    return caL, caR, seqL, seqR


def _load_coords(coords_npz: str):
    if coords_npz is None or (not os.path.exists(coords_npz)):
        return None, None
    z = np.load(coords_npz)

    def _is_numeric(arr):
        try:
            return np.issubdtype(arr.dtype, np.number)
        except Exception:
            return False

    chosen_key = None
    chosen_arr = None
    for k in ("coords", "xyz", "CA"):
        if k in z.files:
            arr = np.asarray(z[k])
            if _is_numeric(arr):
                chosen_key, chosen_arr = k, arr
                break
    if chosen_key is None:
        for k in z.files:
            arr = np.asarray(z[k])
            if (arr.ndim >= 2) and (arr.shape[-1] == 3) and _is_numeric(arr):
                chosen_key, chosen_arr = k, arr
                break
    if chosen_key is None:
        for k in z.files:
            arr = np.asarray(z[k])
            if _is_numeric(arr):
                chosen_key, chosen_arr = k, arr
                break
    if chosen_key is None or chosen_arr is None:
        raise ValueError(f"[DIPS] no numeric coords array found in {coords_npz}, keys={list(z.files)}")

    arr = np.asarray(chosen_arr, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[:, 0, :]
    if arr.ndim == 1:
        arr = arr.reshape(-1, 3)
    if arr.shape[-1] != 3:
        n = arr.size
        if n % 3 != 0:
            raise ValueError(f"[DIPS] coords in {coords_npz} cannot be reshaped to (*,3), shape={arr.shape}")
        arr = arr.reshape(-1, 3)

    mask = np.isfinite(arr).all(axis=1)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.from_numpy(arr), torch.from_numpy(mask.astype(np.bool_))


def _pairwise_contact(coordsA: torch.Tensor, coordsB: torch.Tensor, maskA: torch.Tensor, maskB: torch.Tensor,
                      cutoff: float = 8.0, chunk: int = 1024) -> torch.Tensor:
    L = coordsA.shape[0]
    M = coordsB.shape[0]
    y = torch.zeros((L, M), dtype=torch.float32)
    if L == 0 or M == 0:
        return y
    validA = maskA.bool()
    validB = maskB.bool()
    if not bool(validA.any()) or not bool(validB.any()):
        return y

    cutoff2 = float(cutoff) * float(cutoff)
    for i0 in range(0, L, chunk):
        i1 = min(L, i0 + chunk)
        a = coordsA[i0:i1]
        diff = a[:, None, :] - coordsB[None, :, :]
        d2 = (diff * diff).sum(-1)
        hit = (d2 <= cutoff2)
        va = validA[i0:i1].unsqueeze(1)
        vb = validB.unsqueeze(0)
        hit = hit & va & vb
        y[i0:i1] = hit.float()
    return y


class DIPSIndexedPairs(torch.utils.data.Dataset):
    """DIPS-Plus indexed dataset with ESM cache and fixed 1309-d residue features."""

    def __init__(self, dips_root: str, complex_ids, contact_cutoff=8.0, embedder=None,
                 use_pssm: bool = False, use_dssp: bool = False, esm_cache_dir: str = None, skip_filter: bool = True,
                 verbose=False):
        self.root = str(dips_root)
        self.cutoff = float(contact_cutoff)
        self.use_pssm = bool(use_pssm)
        self.use_dssp = bool(use_dssp)
        self.embedder = embedder
        self.verbose = verbose

        self.dir_coords = os.path.join(self.root, "coords")
        self.dir_seq = os.path.join(self.root, "seq")
        self.dir_pssm = os.path.join(self.root, "pssm")
        self.dir_dssp = os.path.join(self.root, "dssp")

        self.dir_esm = str(esm_cache_dir) if esm_cache_dir else os.path.join(self.root, "esm_cache")
        os.makedirs(self.dir_esm, exist_ok=True)

        self._index_built = False
        self.ids = list(complex_ids) if skip_filter else self._filter_valid_ids(list(complex_ids), verbose=verbose)
        if len(self.ids) == 0:
            raise RuntimeError(f"[DIPS-index] empty complex list (root={self.root}).")

        self.reset_esm_stats()

    def __len__(self):
        return len(self.ids)

    def reset_esm_stats(self):
        self._esm_calls = 0
        self._esm_hit = 0
        self._esm_miss = 0
        try:
            self._esm_cache_files = len([f for f in os.listdir(self.dir_esm) if f.endswith(".pt")])
        except Exception:
            self._esm_cache_files = 0

    def _atomic_torch_save(self, obj, path: str):
        tmp = path + f".tmp.{os.getpid()}.{uuid.uuid4().hex}"
        torch.save(obj, tmp)
        os.replace(tmp, path)

    def _get_esm_cached(self, seq_path, cid, tag, length):
        safe_cid = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(cid))
        cache_path = os.path.join(self.dir_esm, f"{safe_cid}_{tag}_esm.pt")

        self._esm_calls += 1
        if os.path.exists(cache_path):
            self._esm_hit += 1
            if self.verbose and (self._esm_calls % 200 == 0):
                print(
                    f"[ESM-Cache] calls={self._esm_calls} hit={self._esm_hit} miss={self._esm_miss} cache_files={self._esm_cache_files}",
                    flush=True)
            try:
                feat = torch.load(cache_path, map_location="cpu")
                if (not torch.is_tensor(feat)) or feat.ndim != 2 or feat.shape[1] < 1280:
                    raise RuntimeError(f"bad esm cache tensor shape={getattr(feat, 'shape', None)}")
                return feat[:length, :1280]
            except Exception as e:
                try:
                    os.remove(cache_path)
                except Exception:
                    pass

        if isinstance(seq_path, tuple) and len(seq_path) == 2 and seq_path[0] == "SEQ":
            seq_str = seq_path[1] or ""
        else:
            seq_str = _read_fasta_one(seq_path) if seq_path else ""

        if not seq_str:
            self._esm_miss += 1
            return torch.zeros((length, 1280))

        feat = _run_embedder(self.embedder, cid, tag, seq_str)
        if isinstance(feat, np.ndarray):
            feat = torch.from_numpy(feat)
        if not torch.is_tensor(feat):
            feat = torch.tensor(feat)
        feat = feat.detach().float().cpu()

        try:
            self._atomic_torch_save(feat, cache_path)
        except Exception:
            pass
        self._esm_miss += 1
        self._esm_cache_files += 1
        if self.verbose and (self._esm_calls % 200 == 0):
            print(
                f"[ESM-Cache] calls={self._esm_calls} hit={self._esm_hit} miss={self._esm_miss} cache_files={self._esm_cache_files}",
                flush=True)

        return feat[:length, :1280]

    def _build_file_index(self, verbose: bool = False):
        """
        Build file indices for coords/seq/(optional pssm/dssp).

        Supports both:
          - split mode: <cid>_<tag>.(npz|npz.gz|fa|tsv|...)
          - pair mode:  <cid>.(npz|npz.gz|fa|tsv|...)

        IMPORTANT FIX:
          - previously only accepted tag in {0,1,A,B}. DIPS-Plus may use arbitrary
            alnum tags like _2, _5, etc. We now accept any [0-9A-Za-z]+ suffix.
        """
        if self._index_built:
            return

        def scan_dir(dir_path: str):
            try:
                return os.listdir(dir_path) if os.path.isdir(dir_path) else []
            except Exception:
                return []

        # cid -> {tag -> path}
        self._idx_coords_map = {}
        self._idx_seq_map = {}
        self._idx_pssm_map = {}
        self._idx_dssp_map = {}

        # cid -> path (pair mode)
        self._idx_coords_pair = {}
        self._idx_seq_pair = {}
        self._idx_pssm_pair = {}
        self._idx_dssp_pair = {}

        tag_re = re.compile(r"^(.*)_([0-9A-Za-z]+)$")

        # ---- coords ----
        for fn in scan_dir(self.dir_coords):
            if fn.endswith(".npz"):
                stem = fn[:-4]
            elif fn.endswith(".npz.gz"):
                stem = fn[:-7]
            else:
                continue

            m = tag_re.match(stem)
            path = os.path.join(self.dir_coords, fn)
            if m:
                cid, tag = m.group(1), m.group(2)
                self._idx_coords_map.setdefault(cid, {})[tag] = path
            else:
                self._idx_coords_pair[stem] = path

        # ---- seq ----
        seq_ext_priority = {
            ".fa": 0,
            ".fasta": 1,
            ".faa": 2,
            ".fa.gz": 3,
            ".fasta.gz": 4,
            ".faa.gz": 5,
            ".tsv": 6,
            ".tsv.gz": 7
        }
        seq_sufs = tuple(seq_ext_priority.keys())

        # : (cid, tag) -> [(priority, path), ...]
        seq_temp_map = {}
        seq_temp_pair = {}

        for fn in scan_dir(self.dir_seq):
            if not fn.endswith(seq_sufs):
                continue

            ext = None
            priority = 999
            for suf in seq_sufs:
                if fn.endswith(suf):
                    ext = suf
                    priority = seq_ext_priority[suf]
                    break

            if ext is None:
                continue

            stem = fn[:-len(ext)]
            path = os.path.join(self.dir_seq, fn)

            m = tag_re.match(stem)
            if m:
                cid, tag = m.group(1), m.group(2)
                key = (cid, tag)
                if key not in seq_temp_map or priority < seq_temp_map[key][0]:
                    seq_temp_map[key] = (priority, path)
            else:
                if stem not in seq_temp_pair or priority < seq_temp_pair[stem][0]:
                    seq_temp_pair[stem] = (priority, path)

        for (cid, tag), (priority, path) in seq_temp_map.items():
            self._idx_seq_map.setdefault(cid, {})[tag] = path

        for stem, (priority, path) in seq_temp_pair.items():
            self._idx_seq_pair[stem] = path

        # ---- optional pssm ----
        if self.dir_pssm and os.path.isdir(self.dir_pssm):
            pssm_sufs = (".npz", ".npz.gz", ".pt", ".pt.gz", ".tsv", ".tsv.gz")
            for fn in scan_dir(self.dir_pssm):
                if not fn.endswith(pssm_sufs):
                    continue
                stem = fn
                for suf in pssm_sufs:
                    if stem.endswith(suf):
                        stem = stem[:-len(suf)]
                        break
                m = tag_re.match(stem)
                path = os.path.join(self.dir_pssm, fn)
                if m:
                    cid, tag = m.group(1), m.group(2)
                    self._idx_pssm_map.setdefault(cid, {})
                    if (tag not in self._idx_pssm_map[cid]) or (
                            self._idx_pssm_map[cid][tag].endswith(".gz") and not path.endswith(".gz")):
                        self._idx_pssm_map[cid][tag] = path
                else:
                    self._idx_pssm_pair[stem] = path

        # ---- optional dssp ----
        if self.dir_dssp and os.path.isdir(self.dir_dssp):
            dssp_sufs = (".npz", ".npz.gz", ".pt", ".pt.gz", ".tsv", ".tsv.gz")
            for fn in scan_dir(self.dir_dssp):
                if not fn.endswith(dssp_sufs):
                    continue
                stem = fn
                for suf in dssp_sufs:
                    if stem.endswith(suf):
                        stem = stem[:-len(suf)]
                        break
                m = tag_re.match(stem)
                path = os.path.join(self.dir_dssp, fn)
                if m:
                    cid, tag = m.group(1), m.group(2)
                    self._idx_dssp_map.setdefault(cid, {})
                    if (tag not in self._idx_dssp_map[cid]) or (
                            self._idx_dssp_map[cid][tag].endswith(".gz") and not path.endswith(".gz")):
                        self._idx_dssp_map[cid][tag] = path
                else:
                    self._idx_dssp_pair[stem] = path

        self._index_built = True

        if verbose:
            n_split = sum(len(v) for v in self._idx_coords_map.values())
            print(f"[DIPS-index] coords: split={n_split} pair={len(self._idx_coords_pair)}", flush=True)
            n_split = sum(len(v) for v in self._idx_seq_map.values())
            print(f"[DIPS-index] seq:   split={n_split} pair={len(self._idx_seq_pair)}", flush=True)

    def _resolve_pair_files(self, cid: str):
        """
        Resolve files for an id.

        Returns:
          - ("pair", cP, sP, pP, dP)
          - ("split", cA, cB, sA, sB, pA, pB, dA, dB, tagA, tagB)

        FIX:
          - accept arbitrary tags.
          - if caller passes a "cid_with_tag", auto-normalize to base cid.
        """
        self._build_file_index()

        # If ids contain a trailing tag (e.g. "..._0"), normalize to base cid.
        base_cid = cid
        m = re.match(r"^(.*)_([0-9A-Za-z]+)$", cid)
        if (cid not in self._idx_coords_pair) and (cid not in self._idx_coords_map) and m:
            base_cid = m.group(1)

        # 1) pair mode
        cP = self._idx_coords_pair.get(base_cid)
        if cP:
            sP = self._idx_seq_pair.get(base_cid)
            pP = self._idx_pssm_pair.get(base_cid) if hasattr(self, "_idx_pssm_pair") else None
            dP = self._idx_dssp_pair.get(base_cid) if hasattr(self, "_idx_dssp_pair") else None
            return ("pair", cP, sP, pP, dP)

        # 2) split mode
        cMap = self._idx_coords_map.get(base_cid)
        if not cMap:
            return None

        tags = sorted(list(cMap.keys()))

        # Prefer canonical tags if present
        if ("0" in tags) and ("1" in tags):
            tagA, tagB = "0", "1"
        elif ("A" in tags) and ("B" in tags):
            tagA, tagB = "A", "B"
        else:
            # If exactly two tags, use them; else try to pick two deterministically.
            if len(tags) == 2:
                tagA, tagB = tags[0], tags[1]
            elif len(tags) > 2:
                # pick first two tags that have both coords (always) and seq if possible
                sMap = self._idx_seq_map.get(base_cid, {})
                cand = [t for t in tags if t in sMap] or tags
                if len(cand) >= 2:
                    tagA, tagB = cand[0], cand[1]
                else:
                    return None
            else:
                return None

        cA = cMap.get(tagA)
        cB = cMap.get(tagB)
        if not (cA and cB):
            return None

        sMap = self._idx_seq_map.get(base_cid, {})
        sA = sMap.get(tagA)
        sB = sMap.get(tagB)

        pMap = self._idx_pssm_map.get(base_cid, {}) if hasattr(self, "_idx_pssm_map") else {}
        dMap = self._idx_dssp_map.get(base_cid, {}) if hasattr(self, "_idx_dssp_map") else {}
        # if modality dirs are disabled, these can be None; caller handles.
        pA = pMap.get(tagA) if pMap else None
        pB = pMap.get(tagB) if pMap else None
        dA = dMap.get(tagA) if dMap else None
        dB = dMap.get(tagB) if dMap else None

        return ("split", cA, cB, sA, sB, pA, pB, dA, dB, tagA, tagB)

    def _filter_valid_ids(self, ids_in, verbose=False):
        self._build_file_index()
        kept, dropped = [], 0
        total = len(ids_in)
        t0 = time.time()
        for i, cid in enumerate(ids_in, 1):
            if self._resolve_pair_files(cid) is None:
                dropped += 1
            else:
                kept.append(cid)
            if verbose and (i % 2000 == 0 or i == total):
                dt = time.time() - t0
                print(f"[DIPS-index] filtering {i}/{total} kept={len(kept)} dropped={dropped} dt={dt:.1f}s", flush=True)
        if verbose:
            print(f"[DIPS-index] filter done: kept={len(kept)} dropped={dropped}", flush=True)
        return kept

    def __getitem__(self, idx: int):
        cid = self.ids[int(idx)]
        r = self._resolve_pair_files(cid)
        if r is None:
            raise FileNotFoundError(f"[DIPS-index] resolve failed for cid={cid}")

        mode = r[0]
        if mode == "split":
            _, cA_p, cB_p, sA_p, sB_p, pA_p, pB_p, dA_p, dB_p, tagA, tagB = r
            cA, mA = _load_coords(cA_p)
            cB, mB = _load_coords(cB_p)
            if cA is None or cB is None:
                raise FileNotFoundError(
                    f"[DIPS-index] split mode requires two chains but missing coords: cid={cid} "
                    f"(cA={cA is not None}, cB={cB is not None})"
                )
            L, M = cA.shape[0], cB.shape[0]
        else:
            _, cP_p, sP_p = r
            caL, caR, seqL_npz, seqR_npz = _load_pair_npz_full(cP_p)
            if caL is None or caR is None:
                raise RuntimeError(f"[DIPS-index] bad pair npz after resolve: cid={cid} cP={cP_p}")

            cA = torch.from_numpy(caL).float()
            cB = torch.from_numpy(caR).float()
            mA = torch.ones((cA.shape[0],), dtype=torch.bool)
            mB = torch.ones((cB.shape[0],), dtype=torch.bool)
            L, M = cA.shape[0], cB.shape[0]

            # prefer fasta if exists, else use npz strings
            if sP_p and os.path.exists(sP_p):
                seqL, seqR = _read_fasta_two(sP_p)
            else:
                seqL, seqR = seqL_npz, seqR_npz

            sA_p = ("SEQ", seqL)
            sB_p = ("SEQ", seqR)
            pA_p = pB_p = dA_p = dB_p = None

        def _assemble(s_p, p_p, d_p, length, tag):
            esm = self._get_esm_cached(s_p, cid, tag, length)
            if esm is None or (not torch.is_tensor(esm)) or esm.ndim != 2 or esm.shape[-1] < 1280:
                raise RuntimeError(f"[DIPS][{cid}] ESM dim error tag={tag}: {getattr(esm, 'shape', None)}")
            esm = esm[:length, :1280]

            feats = [esm]
            Lmin = int(esm.shape[0])

            # ---- optional PSSM ----
            if self.use_pssm:
                ps = _load_pssm(p_p, length)
                if ps is None or (not torch.is_tensor(ps)) or ps.ndim != 2 or ps.shape[-1] < 20:
                    raise RuntimeError(f"[DIPS][{cid}] PSSM load error tag={tag}: {getattr(ps, 'shape', None)}")
                ps = ps[:Lmin, :20]
                feats.append(ps)
                Lmin = min(Lmin, int(ps.shape[0]))

            # ---- optional DSSP ----
            if self.use_dssp:
                ds = _load_dssp(d_p, length)
                if ds is None or (not torch.is_tensor(ds)) or ds.ndim != 2 or ds.shape[-1] < 9:
                    raise RuntimeError(f"[DIPS][{cid}] DSSP load error tag={tag}: {getattr(ds, 'shape', None)}")
                ds = ds[:Lmin, :9]
                feats.append(ds)
                Lmin = min(Lmin, int(ds.shape[0]))

            # align lengths
            feats = [x[:Lmin] for x in feats]
            cat = torch.cat(feats, dim=-1)

            # baseline: USE_PSSM=0, USE_DSSP=0 -> dim=1280
            # full:     USE_PSSM=1, USE_DSSP=1 -> dim=1309
            return cat[:length, :cat.shape[-1]]

        resA = _assemble(sA_p, pA_p, dA_p, L, "0")
        resB = _assemble(sB_p, pB_p, dB_p, M, "1")
        y2d = _pairwise_contact(cA, cB, mA, mB, cutoff=self.cutoff)

        LA = resA.shape[0]
        LB = resB.shape[0]

        mA = mA[:LA]
        mB = mB[:LB]

        if y2d.shape[0] != LA or y2d.shape[1] != LB:
            y2d = y2d[:LA, :LB]

        y_res_A = y2d.max(dim=1).values
        y_res_B = y2d.max(dim=0).values

        return {
            "complex": cid,
            "resA": resA, "resB": resB,
            "maskA": mA.float(), "maskB": mB.float(),
            "y2d": y2d,
            "y_res_A": y_res_A,
            "y_res_B": y_res_B
        }


def _read_complex_list(txt_path):
    if not txt_path or (not os.path.exists(txt_path)):
        raise FileNotFoundError(f"complex list not found: {txt_path}")

    def _norm_id(t: str) -> str:
        t = t.strip()
        if not t:
            return t
        # drop extensions if user accidentally put filenames in the list
        t = re.sub(r"\.(npz|npy|pt|pth|pkl|pickle|fa|fasta|faa|tsv|txt)(\.gz)?$", "", t, flags=re.IGNORECASE)
        # strip single-chain suffix (common DIPS naming: xxx_0 / xxx_1 / xxx_A / xxx_B)
        # m = re.match(r"^(.*)_(0|1|A|B)$", t)
        # if m:
        #     t = m.group(1)
        return t

    out = []
    seen = set()
    with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            s = line.strip()
            if (not s) or s.startswith('#'):
                continue
            tok = s.split()[0]
            tok = _norm_id(tok)
            if tok and (tok not in seen):
                out.append(tok)
                seen.add(tok)
    return out


def get_sampling_params_for_epoch(ep, P):
    """Return epoch-dependent sampling parameters."""
    s1_end = int(getattr(P, 'samp_stage1_end', 10))
    s2_end = int(getattr(P, 'samp_stage2_end', 30))
    if ep <= s1_end:
        return {
            "p_target": float(getattr(P, 'samp_p_target_s1', 0.80)),
            "base_w_pos": float(getattr(P, 'samp_base_w_pos_s1', 1.00)),
            "base_w_neg": float(getattr(P, 'samp_base_w_neg_s1', 1.00)),
            "pos_oversample": float(getattr(P, 'samp_pos_oversample_s1', 8.0)),
        }
    elif ep <= s2_end:
        return {
            "p_target": float(getattr(P, 'samp_p_target_s2', 0.75)),
            "base_w_pos": float(getattr(P, 'samp_base_w_pos_s2', 1.20)),
            "base_w_neg": float(getattr(P, 'samp_base_w_neg_s2', 0.80)),
            "pos_oversample": float(getattr(P, 'samp_pos_oversample_s2', 4.0)),
        }
    else:
        return {
            "p_target": float(getattr(P, 'samp_p_target_s3', 0.70)),
            "base_w_pos": float(getattr(P, 'samp_base_w_pos_s3', 1.50)),
            "base_w_neg": float(getattr(P, 'samp_base_w_neg_s3', 0.60)),
            "pos_oversample": float(getattr(P, 'samp_pos_oversample_s3', 2.0)),
        }


def _read_rbp_id_list(txt_path: str):
    """Read UniProt IDs from a plain text file (one per line, # = comment)."""
    if not txt_path or not os.path.exists(txt_path):
        raise FileNotFoundError(f"RBP ID list not found: {txt_path}")
    out, seen = [], set()
    with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('#'):
                continue
            tok = s.split()[0].strip()
            # strip accidental extensions
            tok = re.sub(r'\.(npy|npz|pt|fa|fasta|txt)(\.(gz))?$', '', tok, flags=re.IGNORECASE)
            if tok and tok not in seen:
                out.append(tok)
                seen.add(tok)
    return out


class RBP296Dataset(torch.utils.data.Dataset):
    """
    RBP296 single-chain RNA-binding site dataset.

    Returns a batch dict with the SAME keys as DIPSIndexedPairs so that
    the rest of the training pipeline (forward_one, eval, crop, collate)
    works without modification:

        resA    [L, D]      real protein residue features
        maskA   [L]         valid-residue mask (all 1 for RBP296)
        y_res_A [L]         binary RNA-binding label
        resB    [Ld, D]     dummy chain B (Ld >= 2, zero-filled)
        maskB   [Ld]        dummy mask (all 1)
        y_res_B [Ld]        all 0  (not supervised)
        y2d     [L, Ld]     all 0  (not supervised; l2_w should be 0)
        complex  str        protein ID

    Feature assembly: ESM-1280 + optional PSSM-20 + optional DSSP-9 = D
    Default D = 1309  (use_pssm=True, use_dssp=True)
    """

    DUMMY_LEN = 2   # minimum dummy-B length to avoid BatchNorm errors

    def __init__(self, rbp_root: str, pid_list, embedder=None,
                 use_pssm: bool = True, use_dssp: bool = True,
                 esm_cache_dir: str = None, verbose: bool = False):
        self.root      = str(rbp_root)
        self.pids      = list(pid_list)
        self.embedder  = embedder
        self.use_pssm  = bool(use_pssm)
        self.use_dssp  = bool(use_dssp)
        self.verbose   = bool(verbose)

        self.dir_seq    = os.path.join(self.root, "seq")
        self.dir_labels = os.path.join(self.root, "labels")
        self.dir_pssm   = os.path.join(self.root, "pssm")
        self.dir_dssp   = os.path.join(self.root, "dssp")

        self.dir_esm = (str(esm_cache_dir) if esm_cache_dir
                        else os.path.join(self.root, "esm_cache"))
        os.makedirs(self.dir_esm, exist_ok=True)

        self.reset_esm_stats()

    def __len__(self):
        return len(self.pids)

    def reset_esm_stats(self):
        self._esm_calls = 0
        self._esm_hit   = 0
        self._esm_miss  = 0
        try:
            self._esm_cache_files = len([f for f in os.listdir(self.dir_esm) if f.endswith('.pt')])
        except Exception:
            self._esm_cache_files = 0

    def _atomic_torch_save(self, obj, path: str):
        tmp = path + f".tmp.{os.getpid()}.{uuid.uuid4().hex}"
        torch.save(obj, tmp)
        os.replace(tmp, path)

    def _get_esm_cached(self, pid: str, seq_str: str, length: int) -> torch.Tensor:
        safe_pid   = re.sub(r'[^A-Za-z0-9_.-]+', '_', str(pid))
        cache_path = os.path.join(self.dir_esm, f"{safe_pid}_esm.pt")
        self._esm_calls += 1

        if os.path.exists(cache_path):
            try:
                feat = torch.load(cache_path, map_location="cpu")
                if torch.is_tensor(feat) and feat.ndim == 2 and feat.shape[1] >= 1280:
                    self._esm_hit += 1
                    feat = feat[:length, :1280]
                    if feat.shape[0] < length:
                        feat = F.pad(feat, (0, 0, 0, length - feat.shape[0]))
                    return feat
                os.remove(cache_path)
            except Exception:
                try:
                    os.remove(cache_path)
                except Exception:
                    pass

        if not seq_str:
            self._esm_miss += 1
            return torch.zeros((length, 1280), dtype=torch.float32)

        feat = _run_embedder(self.embedder, pid, "A", seq_str)
        if isinstance(feat, np.ndarray):
            feat = torch.from_numpy(feat)
        feat = feat.detach().float().cpu()
        try:
            self._atomic_torch_save(feat, cache_path)
        except Exception:
            pass
        self._esm_miss += 1
        self._esm_cache_files += 1

        feat = feat[:length, :1280]
        if feat.shape[0] < length:
            feat = F.pad(feat, (0, 0, 0, length - feat.shape[0]))
        return feat

    def __getitem__(self, idx: int) -> dict:
        pid = self.pids[int(idx)]

        # ---- seq -> length ----
        seq_path = os.path.join(self.dir_seq, f"{pid}.fa")
        seq_str  = _read_fasta_one(seq_path)
        L        = len(seq_str) if seq_str else 0

        # ---- label (defines L if seq missing) ----
        label_path = os.path.join(self.dir_labels, f"{pid}.npy")
        if L == 0:
            if os.path.exists(label_path):
                try:
                    L = int(np.load(label_path).reshape(-1).shape[0])
                except Exception:
                    L = 1
            else:
                L = 1

        y_res_A = torch.zeros(L, dtype=torch.float32)
        if os.path.exists(label_path):
            try:
                arr = np.load(label_path).astype(np.float32).reshape(-1)
                if len(arr) != L:
                    arr = arr[:L] if len(arr) > L else np.pad(arr, (0, L - len(arr)))
                y_res_A = torch.from_numpy(arr)
            except Exception:
                pass

        # ---- ESM ----
        esm_feat = self._get_esm_cached(pid, seq_str, L)  # [L, 1280]

        # ---- assemble features ----
        feats = [esm_feat]
        Lmin  = int(esm_feat.shape[0])

        if self.use_pssm:
            pssm_path = os.path.join(self.dir_pssm, f"{pid}.npy")
            ps = _load_pssm(pssm_path, L)
            feats.append(ps[:Lmin])

        if self.use_dssp:
            dssp_path = os.path.join(self.dir_dssp, f"{pid}.npy")
            ds = _load_dssp(dssp_path, L)
            feats.append(ds[:Lmin])

        feats   = [x[:Lmin] for x in feats]
        resA    = torch.cat(feats, dim=-1)   # [Lmin, D]
        L       = Lmin
        y_res_A = y_res_A[:L]
        maskA   = torch.ones(L, dtype=torch.float32)

        # ---- dummy B chain (not supervised) ----
        Ld      = max(self.DUMMY_LEN, 2)
        D       = int(resA.shape[-1])
        resB    = torch.zeros((Ld, D), dtype=torch.float32)
        maskB   = torch.ones(Ld, dtype=torch.float32)
        y_res_B = torch.zeros(Ld, dtype=torch.float32)
        y2d     = torch.zeros((L, Ld), dtype=torch.float32)

        return {
            "complex": pid,
            "resA":    resA,
            "maskA":   maskA,
            "y_res_A": y_res_A,
            "resB":    resB,
            "maskB":   maskB,
            "y_res_B": y_res_B,
            "y2d":     y2d,
        }


def build_l2_loaders():
    """Build train / val DataLoaders for RBP296 with protein-level split.
    Keeps exactly the same return signature as the original DIPS version:
        dl_tr, dl_va, d_res, d_chain
    """
    print(f"[config] embedder_device={ESM_DEVICE}", flush=True)
    emb = SiteEmbedder(device=ESM_DEVICE)

    # ---- load and split IDs ----
    all_ids  = _read_rbp_id_list(RBP_ID_LIST)
    rnd      = random.Random(int(getattr(P, 'split_seed', 42)))
    rnd.shuffle(all_ids)
    N        = len(all_ids)
    n_tr     = max(1, int(round(N * float(getattr(P, 'split_train', 0.70)))))
    n_va     = max(1, int(round(N * float(getattr(P, 'split_val',   0.15)))))
    n_te     = N - n_tr - n_va
    if n_te < 0:
        n_va += n_te; n_te = 0

    tr_ids = all_ids[:n_tr]
    va_ids = all_ids[n_tr: n_tr + n_va]
    te_ids = all_ids[n_tr + n_va:]
    print(f"[RBP296] total={N} train={len(tr_ids)} val={len(va_ids)} test={len(te_ids)}", flush=True)

    # ---- save split ----
    split_path = os.path.join(SAVE_DIR, "split_ids.json")
    try:
        import json as _json
        with open(split_path, 'w', encoding='utf-8') as _f:
            _json.dump({"train": tr_ids, "val": va_ids, "test": te_ids}, _f, indent=2)
        print(f"[RBP296] split saved -> {split_path}", flush=True)
    except Exception as _e:
        print(f"[warn] could not save split: {_e}", flush=True)

    esm_cache_dir = os.path.join(SAVE_DIR, "esm_cache")
    os.makedirs(esm_cache_dir, exist_ok=True)

    common_kw = dict(embedder=emb, use_pssm=P.use_pssm, use_dssp=P.use_dssp,
                     esm_cache_dir=esm_cache_dir,
                     verbose=bool(int(getattr(P, 'dips_index_verbose', 0))))

    tr = RBP296Dataset(RBP_ROOT, tr_ids, **common_kw)
    va = RBP296Dataset(RBP_ROOT, va_ids, **common_kw)
    print(f"[RBP296] loaded: Train={len(tr)} Val={len(va)}", flush=True)

    # ---- pos-aware sampler (reuses existing build_pos_aware_weights) ----
    sp0      = get_sampling_params_for_epoch(1, P)
    p_target = float(sp0["p_target"])
    weights, st, flags = build_pos_aware_weights(
        tr,
        rate_thr=float(getattr(P, 'samp_rate_thr', 0.01)),
        p_target=p_target,
        base_w_pos=float(sp0["base_w_pos"]),
        base_w_neg=float(sp0["base_w_neg"]),
        pos_oversample=float(sp0["pos_oversample"]),
        max_ratio=50.0
    )

    sampler = torch.utils.data.WeightedRandomSampler(
        weights=weights.double(), num_samples=len(tr), replacement=True
    )

    global _L2_SAMPLER_FLAGS, _L2_SAMPLER_HASPOS_RATE, _L2_SAMPLER_SAMPLER, _L2_SAMPLER_WEIGHTS
    _L2_SAMPLER_FLAGS     = flags
    _L2_SAMPLER_HASPOS_RATE = float(st.get("has_pos_rate", 0.0) or 0.0)
    _L2_SAMPLER_SAMPLER   = sampler
    _L2_SAMPLER_WEIGHTS   = weights

    print(
        f"[sampler][RBP296] has_pos_rate={st['has_pos_rate']:.3f} "
        f"w_pos={st['w_pos']:.2f} w_neg={st['w_neg']:.2f} "
        f"rate_thr={st.get('rate_thr', 0.01):.3f} p_target={p_target:.2f}",
        flush=True)

    _nw = NUM_WORKERS_SITE
    if DEVICE.startswith('cuda') and _nw > 0:
        print('[dataloader][fix] DEVICE=cuda -> force num_workers_site=0.', flush=True)
        _nw = 0

    dl_kwargs = dict(
        batch_size=BATCH_SITE,
        num_workers=_nw,
        collate_fn=dips_collate,
        pin_memory=(DEVICE.startswith('cuda')),
        persistent_workers=(_nw > 0),
        drop_last=False
    )
    if _nw > 0:
        dl_kwargs['prefetch_factor'] = 2

    dl_tr = DataLoader(tr, sampler=sampler, shuffle=False, **dl_kwargs)
    dl_va = DataLoader(va, shuffle=False, **{**dl_kwargs, 'drop_last': False})

    d_res, d_chain = _infer_rbp_feature_dims()
    print(f"[RBP296] inferred feature dims without loader probe: d_res={d_res} d_chain={d_chain}", flush=True)

    # ===================== SANITY CHECK (optional) =====================
    n_sanity = int(P.sanity_n)
    if n_sanity > 0:
        print(f"[sanity] checking first {n_sanity} training samples ...", flush=True)
        n_sanity = min(n_sanity, len(tr))
        pos_pixels = []
        bad_shape = 0
        empty_cnt = 0
        for i in range(n_sanity):
            item = tr[i]
            y2d = item.get("y2d", None)
            rA  = item.get("resA", None)
            rB  = item.get("resB", None)
            if y2d is None or y2d.numel() == 0:
                empty_cnt += 1
                continue
            if (rA is None) or (rB is None) or (y2d.ndim != 2):
                bad_shape += 1
                continue
            print(
                f"[sanity][{i}] A={tuple(rA.shape)} B={tuple(rB.shape)} "
                f"y2d={tuple(y2d.shape)} npos={int((y2d > 0.5).sum().item())}",
                flush=True)
            pos_pixels.append(float((y2d > 0.5).sum().item()) / float(max(1, y2d.numel())))
        if len(pos_pixels) > 0:
            print(f"[sanity] pos_pixel_ratio: mean={np.mean(pos_pixels):.6f} max={np.max(pos_pixels):.6f}", flush=True)
        if empty_cnt or bad_shape:
            print(f"[sanity] empty={empty_cnt} bad_shape={bad_shape}", flush=True)
    # ============================================
    return dl_tr, dl_va, d_res, d_chain


# ---------------- Crop helpers ----------------
def _choose_crop(L, M, max_tokens, center=False):
    # Fixed-size crop (optional): helps fight extreme class-imbalance by increasing
    # the probability that a crop still contains positives around an interface.
    if L * M <= max_tokens:
        return 0, L, 0, M

    if bool(L2_FIXED_CROP) and int(L2_FIXED_SIDE) > 0:
        Lt = int(min(L, max(MIN_SIDE, int(L2_FIXED_SIDE))))
        Mt = int(min(M, max(MIN_SIDE, int(L2_FIXED_SIDE))))

        # respect max_tokens upper bound (avoid OOM)
        if Lt * Mt > int(max_tokens):
            cap = int(max(1, math.floor(math.sqrt(float(max_tokens)))))
            Lt = int(min(Lt, max(1, cap)))
            Mt = int(min(Mt, max(1, cap)))
    else:
        scale = math.sqrt(max_tokens / float(L * M))
        Lt = max(MIN_SIDE, int(L * scale))
        Lt = min(Lt, L)
        Mt = max(MIN_SIDE, int(M * scale))
        Mt = min(Mt, M)

    if center:
        i0 = (L - Lt) // 2
        j0 = (M - Mt) // 2
    else:
        i0 = 0 if (L == Lt) else random.randint(0, L - Lt)
        j0 = 0 if (M == Mt) else random.randint(0, M - Mt)
    return i0, i0 + Lt, j0, j0 + Mt

def _choose_window_start_1d(N: int, max_side: int, y_res: torch.Tensor = None, mask: torch.Tensor = None, *,
                            center_bias: bool = False) -> int:
    if N <= max_side:
        return 0
    if center_bias:
        return int((N - max_side) // 2)

    try:
        if y_res is not None and torch.is_tensor(y_res):
            yr = y_res[0] if y_res.ndim == 2 else y_res
            pos = (yr > 0.5).nonzero(as_tuple=False).view(-1)
            if pos.numel() > 0:
                k = int(pos[torch.randint(0, pos.numel(), (1,)).item()].item())
                i0 = k - max_side // 2
                i0 = max(0, min(int(N - max_side), int(i0)))
                return int(i0)
    except Exception:
        pass

    tries = 6
    for _ in range(tries):
        i0 = 0 if (N == max_side) else random.randint(0, N - max_side)
        if mask is None or (not torch.is_tensor(mask)):
            return int(i0)
        mk = mask[0] if mask.ndim == 2 else mask
        if bool((mk[i0:i0 + max_side] > 0.5).any().item()):
            return int(i0)
    return int(0 if (N == max_side) else random.randint(0, N - max_side))


def _trim_to_max_side_train(tb: dict) -> dict:
    L = int(tb['resA'].shape[1])
    M = int(tb['resB'].shape[1])

    if L > MAX_SIDE:
        i0 = _choose_window_start_1d(L, MAX_SIDE, tb.get('y_res_A', None), tb.get('maskA', None))
        i1 = i0 + MAX_SIDE
        for k in ('resA', 'maskA'):
            tb[k] = tb[k][:, i0:i1, ...]
        if 'y_res_A' in tb: tb['y_res_A'] = tb['y_res_A'][:, i0:i1]
        if 'y2d' in tb: tb['y2d'] = tb['y2d'][:, i0:i1, :]

    if M > MAX_SIDE:
        j0 = _choose_window_start_1d(M, MAX_SIDE, tb.get('y_res_B', None), tb.get('maskB', None))
        j1 = j0 + MAX_SIDE
        for k in ('resB', 'maskB'):
            tb[k] = tb[k][:, j0:j1, ...]
        if 'y_res_B' in tb: tb['y_res_B'] = tb['y_res_B'][:, j0:j1]
        if 'y2d' in tb: tb['y2d'] = tb['y2d'][:, :, j0:j1]

    return tb


def crop_batch_2d(tb, max_tokens, center=False, return_idx=False):
    tb = _trim_to_max_side_train(tb)
    # ===== FIX: align y2d/masks to resA/resB length before any crop =====
    if ("y2d" in tb) and ("maskA" in tb) and ("maskB" in tb):
        y2d = tb["y2d"]
        mA = tb["maskA"]
        mB = tb["maskB"]
        if torch.is_tensor(y2d) and torch.is_tensor(mA) and torch.is_tensor(mB):
            if y2d.ndim == 2: y2d = y2d.unsqueeze(0)
            if mA.ndim == 1: mA = mA.unsqueeze(0)
            if mB.ndim == 1: mB = mB.unsqueeze(0)
            L = int(min(y2d.shape[1], int(tb["resA"].shape[1]), mA.shape[1]))
            M = int(min(y2d.shape[2], int(tb["resB"].shape[1]), mB.shape[1]))
            tb["y2d"] = y2d[:, :L, :M]
            tb["maskA"] = mA[:, :L]
            tb["maskB"] = mB[:, :M]
            if "y_res_A" in tb and torch.is_tensor(tb["y_res_A"]):
                tb["y_res_A"] = tb["y_res_A"][:, :L]
            if "y_res_B" in tb and torch.is_tensor(tb["y_res_B"]):
                tb["y_res_B"] = tb["y_res_B"][:, :M]
    # ===================================================================
    L = tb["resA"].shape[1]
    M = tb["resB"].shape[1]

    if L * M <= max_tokens:
        return (tb, (0, L, 0, M)) if return_idx else tb

    # If the full map has positives but a random crop becomes all-negative,
    # reject ~80% of such crops and resample (reduces all-neg window dominance).
    global_has_pos = False
    valid2d = None
    if ("y2d" in tb) and ("maskA" in tb) and ("maskB" in tb):
        try:
            y2d_tmp = tb["y2d"]
            mA = tb["maskA"]
            mB = tb["maskB"]
            if torch.is_tensor(y2d_tmp) and torch.is_tensor(mA) and torch.is_tensor(mB):
                if y2d_tmp.ndim == 2: y2d_tmp = y2d_tmp.unsqueeze(0)
                if mA.ndim == 1: mA = mA.unsqueeze(0)
                if mB.ndim == 1: mB = mB.unsqueeze(0)
                mA = (mA > 0.5)
                mB = (mB > 0.5)
                L0 = int(min(y2d_tmp.shape[1], mA.shape[1]))
                M0 = int(min(y2d_tmp.shape[2], mB.shape[1]))
                y2d_tmp = y2d_tmp[:, :L0, :M0]
                mA = mA[:, :L0]
                mB = mB[:, :M0]
                valid2d = (mA[:, :, None] & mB[:, None, :])
                global_has_pos = bool((((y2d_tmp > 0.5) & valid2d).any()).item())
        except Exception:
            global_has_pos = False
            valid2d = None

    if global_has_pos and float(L2_REJECT_ALLNEG_P) > 0.0 and valid2d is not None:
        best = None
        tries = int(max(1, int(L2_REJECT_ALLNEG_MAX_TRIES)))
        for _ in range(tries):
            ci0, ci1, cj0, cj1 = _choose_crop(L, M, max_tokens, center=center)
            npos = int((((tb["y2d"][:, ci0:ci1, cj0:cj1] > 0.5) & (valid2d[:, ci0:ci1, cj0:cj1])).sum()).item())
            if npos > 0:
                best = (ci0, ci1, cj0, cj1)
                break
            # all-neg crop: reject with probability p
            if random.random() < float(L2_REJECT_ALLNEG_P):
                continue
            best = (ci0, ci1, cj0, cj1)
            break
        if best is None:
            best = (ci0, ci1, cj0, cj1)
        i0, i1, j0, j1 = best
    else:
        i0, i1, j0, j1 = _choose_crop(L, M, max_tokens, center=center)
    for k in ("resA", "maskA"):
        tb[k] = tb[k][:, i0:i1, ...]
    for k in ("resB", "maskB"):
        tb[k] = tb[k][:, j0:j1, ...]
    if "y2d" in tb: tb["y2d"] = tb["y2d"][:, i0:i1, j0:j1]
    if "y_res_A" in tb: tb["y_res_A"] = tb["y_res_A"][:, i0:i1]
    if "y_res_B" in tb: tb["y_res_B"] = tb["y_res_B"][:, j0:j1]
    return (tb, (i0, i1, j0, j1)) if return_idx else tb


def crop_batch_2d_pos_aware(tb, max_tokens, return_idx=False, max_tries=8, focus_max_tokens=131072, force=False):
    tb = _trim_to_max_side_train(tb)
    if (not bool(force)) and random.random() > float(L2_ANCHOR_PROB):
        return crop_batch_2d(tb, max_tokens, return_idx=return_idx)

    def _thr_from_max(vmax: float) -> float:
        if not np.isfinite(vmax) or vmax <= 0.0:
            return 0.5
        if vmax <= 0.5:
            return float(0.5 * vmax)
        return 0.5

    if ("y2d" not in tb) or ("maskA" not in tb) or ("maskB" not in tb):
        return crop_batch_2d(tb, max_tokens, center=False, return_idx=return_idx)

    y2d = tb["y2d"]
    maskA = (tb["maskA"] > 0.5)
    maskB = (tb["maskB"] > 0.5)

    # ===== FIX: force shape alignment among y2d / maskA / maskB =====
    # y2d: [B,L,M]   maskA:[B,L]   maskB:[B,M]
    if y2d.ndim == 2:
        y2d = y2d.unsqueeze(0)
    if maskA.ndim == 1:
        maskA = maskA.unsqueeze(0)
    if maskB.ndim == 1:
        maskB = maskB.unsqueeze(0)

    B = int(y2d.shape[0])
    L = int(min(y2d.shape[1], maskA.shape[1]))
    M = int(min(y2d.shape[2], maskB.shape[1]))

    # trim all to (B,L,M)
    y2d = y2d[:, :L, :M]
    maskA = maskA[:, :L]
    maskB = maskB[:, :M]

    # write back (important: later cropping uses tb content)
    tb["y2d"] = y2d
    tb["maskA"] = maskA.float()
    tb["maskB"] = maskB.float()

    valid2d = maskA.unsqueeze(-1) & maskB.unsqueeze(1)
    # ===============================================================

    try:
        vmax2d = float(torch.nan_to_num(y2d, nan=0.0).max().item())
    except Exception:
        vmax2d = 1.0
    thr_pos = _thr_from_max(vmax2d)

    pos = ((y2d > thr_pos) & valid2d).nonzero(as_tuple=False)
    if pos.numel() == 0:
        return crop_batch_2d(tb, max_tokens, center=False, return_idx=return_idx)

    L = int(tb["resA"].shape[1])
    M = int(tb["resB"].shape[1])

    def _crop_centered(tokens, ci, cj):
        if L * M <= tokens:
            return 0, L, 0, M

        if bool(L2_FIXED_CROP) and int(L2_FIXED_SIDE) > 0:
            Lt = int(min(L, max(MIN_SIDE, int(L2_FIXED_SIDE))))
            Mt = int(min(M, max(MIN_SIDE, int(L2_FIXED_SIDE))))
            if Lt * Mt > int(tokens):
                cap = int(max(1, math.floor(math.sqrt(float(tokens)))))
                Lt = int(min(Lt, max(1, cap)))
                Mt = int(min(Mt, max(1, cap)))
        else:
            scale = math.sqrt(tokens / float(L * M))
            Lt = max(MIN_SIDE, int(L * scale))
            Lt = min(Lt, L)
            Mt = max(MIN_SIDE, int(M * scale))
            Mt = min(Mt, M)

        i0 = int(max(0, min(L - Lt, ci - Lt // 2)))
        j0 = int(max(0, min(M - Mt, cj - Mt // 2)))
        return i0, i0 + Lt, j0, j0 + Mt

    def _try(tokens):
        for _ in range(max_tries):
            kk = random.randint(0, pos.shape[0] - 1)
            ci = int(pos[kk, 1].item())
            cj = int(pos[kk, 2].item())
            i0, i1, j0, j1 = _crop_centered(tokens, ci, cj)
            if bool((((y2d[:, i0:i1, j0:j1] > thr_pos) & (valid2d[:, i0:i1, j0:j1])).any()).item()):
                return i0, i1, j0, j1
        return None

    focus_tokens = int(min(max_tokens, focus_max_tokens))
    best = _try(focus_tokens)
    if best is None and focus_tokens != int(max_tokens):
        best = _try(int(max_tokens))
    if best is None:
        return crop_batch_2d(tb, max_tokens, center=False, return_idx=return_idx)

    i0, i1, j0, j1 = best
    for k in ("resA", "maskA"):
        tb[k] = tb[k][:, i0:i1, ...]
    for k in ("resB", "maskB"):
        tb[k] = tb[k][:, j0:j1, ...]
    if "y2d" in tb: tb["y2d"] = tb["y2d"][:, i0:i1, j0:j1]
    if "y_res_A" in tb: tb["y_res_A"] = tb["y_res_A"][:, i0:i1]
    if "y_res_B" in tb: tb["y_res_B"] = tb["y_res_B"][:, j0:j1]

    return (tb, (i0, i1, j0, j1)) if return_idx else tb


# ---------------- Eval: full-map multicrop + precision@k ----------------
@torch.no_grad()
def model_forward_from_batch(model, batch: dict):
    mA = batch.get("maskA", None)
    mB = batch.get("maskB", None)
    if torch.is_tensor(mA):
        mA = (mA > 0.5)
    else:
        mA = None
    if torch.is_tensor(mB):
        mB = (mB > 0.5)
    else:
        mB = None
    return model(batch["resA"], mA, batch.get("chainA", None),
                 batch["resB"], mB, batch.get("chainB", None))


@torch.no_grad()
def precision_at_k(y2d: torch.Tensor, s2d: torch.Tensor, valid2d: torch.Tensor, k: int, thr_true: float = 0.5) -> float:
    y2d = y2d.detach().cpu()
    s2d = s2d.detach().cpu()
    valid2d = valid2d.detach().cpu()
    y = y2d[valid2d].reshape(-1)
    s = s2d[valid2d].reshape(-1)
    n = int(s.numel())
    if n <= 0:
        return 0.0
    k = int(min(max(1, k), n))
    top = torch.topk(s, k=k, largest=True, sorted=False).indices
    tp = float((y[top] > thr_true).sum().item())
    return tp / float(k)


def _trim_max_side(tb: dict):
    L = int(tb["resA"].shape[1])
    M = int(tb["resB"].shape[1])

    if L > MAX_SIDE:
        i0 = _choose_window_start_1d(L, MAX_SIDE, None, tb.get('maskA', None), center_bias=True)
        i1 = i0 + MAX_SIDE
        for k in ("resA", "maskA"):
            if k in tb and torch.is_tensor(tb[k]):
                tb[k] = tb[k][:, i0:i1, ...]
        if "y_res_A" in tb: tb["y_res_A"] = tb["y_res_A"][:, i0:i1]
        if "y2d" in tb: tb["y2d"] = tb["y2d"][:, i0:i1, :]
        L = MAX_SIDE

    if M > MAX_SIDE:
        j0 = _choose_window_start_1d(M, MAX_SIDE, None, tb.get('maskB', None), center_bias=True)
        j1 = j0 + MAX_SIDE
        for k in ("resB", "maskB"):
            if k in tb and torch.is_tensor(tb[k]):
                tb[k] = tb[k][:, j0:j1, ...]
        if "y_res_B" in tb: tb["y_res_B"] = tb["y_res_B"][:, j0:j1]
        if "y2d" in tb: tb["y2d"] = tb["y2d"][:, :, j0:j1]
        M = MAX_SIDE

    return tb, L, M


def _slice_batch_2d(tb: dict, i0: int, i1: int, j0: int, j1: int) -> dict:
    out = dict(tb)
    for k in ("resA", "maskA"):
        if k in out and torch.is_tensor(out[k]):
            out[k] = out[k][:, i0:i1, ...]
    for k in ("resB", "maskB"):
        if k in out and torch.is_tensor(out[k]):
            out[k] = out[k][:, j0:j1, ...]
    if "y2d" in out and torch.is_tensor(out["y2d"]):
        out["y2d"] = out["y2d"][:, i0:i1, j0:j1]
    if "y_res_A" in out and torch.is_tensor(out["y_res_A"]):
        out["y_res_A"] = out["y_res_A"][:, i0:i1]
    if "y_res_B" in out and torch.is_tensor(out["y_res_B"]):
        out["y_res_B"] = out["y_res_B"][:, j0:j1]
    return out


def _ensure_mask_match(tb: dict) -> dict:
    try:
        if "resA" in tb and "maskA" in tb and torch.is_tensor(tb["resA"]) and torch.is_tensor(tb["maskA"]):
            if tb["maskA"].ndim == 1: tb["maskA"] = tb["maskA"].unsqueeze(0)
            La = int(tb["resA"].shape[1])
            if int(tb["maskA"].shape[1]) != La:
                L = int(min(La, int(tb["maskA"].shape[1])))
                tb["resA"] = tb["resA"][:, :L, ...]
                tb["maskA"] = tb["maskA"][:, :L]
                if "y_res_A" in tb and torch.is_tensor(tb["y_res_A"]):
                    tb["y_res_A"] = tb["y_res_A"][:, :L]
                if "y2d" in tb and torch.is_tensor(tb["y2d"]):
                    tb["y2d"] = tb["y2d"][:, :L, :]
        if "resB" in tb and "maskB" in tb and torch.is_tensor(tb["resB"]) and torch.is_tensor(tb["maskB"]):
            if tb["maskB"].ndim == 1: tb["maskB"] = tb["maskB"].unsqueeze(0)
            Mb = int(tb["resB"].shape[1])
            if int(tb["maskB"].shape[1]) != Mb:
                M = int(min(Mb, int(tb["maskB"].shape[1])))
                tb["resB"] = tb["resB"][:, :M, ...]
                tb["maskB"] = tb["maskB"][:, :M]
                if "y_res_B" in tb and torch.is_tensor(tb["y_res_B"]):
                    tb["y_res_B"] = tb["y_res_B"][:, :M]
                if "y2d" in tb and torch.is_tensor(tb["y2d"]):
                    tb["y2d"] = tb["y2d"][:, :, :M]
    except Exception:
        return tb
    return tb


@torch.no_grad()
def infer_logits_fullmap_multicrop(model, batch: dict, device: str, max_tokens: int, stride_frac: float = 0.5):
    tb = dict(batch)
    for k, v in list(tb.items()):
        if torch.is_tensor(v):
            tb[k] = v.contiguous()

    tb = _ensure_mask_match(tb)
    L0 = int(tb["resA"].shape[1])
    M0 = int(tb["resB"].shape[1])
    if L0 <= 0 or M0 <= 0:
        return None, None

    if L0 > MAX_SIDE or M0 > MAX_SIDE:
        def _win_starts(N):
            if N <= MAX_SIDE:
                return [0]
            a = 0
            b = int((N - MAX_SIDE) // 2)
            c = int(N - MAX_SIDE)
            xs = [a, b, c]
            out = []
            for x in xs:
                x = int(max(0, min(N - MAX_SIDE, x)))
                if x not in out:
                    out.append(x)
            return out

        startsA = _win_starts(L0)
        startsB = _win_starts(M0)
        B0 = int(tb["resA"].shape[0])  # batch size
        S_full = torch.full((B0, L0, M0), -1e4, dtype=torch.float16)

        for i0 in startsA:
            for j0 in startsB:
                i1, j1 = i0 + min(MAX_SIDE, L0 - i0), j0 + min(MAX_SIDE, M0 - j0)
                sub = _slice_batch_2d(tb, i0, i1, j0, j1)
                sub, Ls, Ms = _trim_max_side(sub)
                sub = _ensure_mask_match(sub)
                if Ls <= 0 or Ms <= 0:
                    continue
                sub_dev = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in sub.items()}
                out = model_forward_from_batch(model, sub_dev)
                S = out.get("S", None)
                if S is None:
                    continue
                if S.ndim == 4 and S.shape[1] == 1:
                    S = S[:, 0]
                if S.ndim == 2:
                    S = S.unsqueeze(0)
                S = S.detach().cpu().to(torch.float16)
                cur = S_full[:, i0:i0 + S.shape[1], j0:j0 + S.shape[2]]
                S_full[:, i0:i0 + S.shape[1], j0:j0 + S.shape[2]] = torch.maximum(cur, S)

        return S_full, tb

    tb, L, M = _trim_max_side(tb)
    if L <= 0 or M <= 0:
        return None, None

    if L * M <= max_tokens:
        def _to_dev_inputs(x: dict):
            need = ("resA", "maskA", "chainA", "resB", "maskB", "chainB")
            out = {}
            for k in need:
                if k in x:
                    v = x[k]
                    out[k] = v.to(device, non_blocking=True) if torch.is_tensor(v) else v
            return out

        tb_dev = _to_dev_inputs(tb)
        out = model_forward_from_batch(model, tb_dev)
        S = out.get("S", None)
        if S is None:
            return None, None
        if S.ndim == 4 and S.shape[1] == 1:
            S = S[:, 0]
        if S.ndim == 2:
            S = S.unsqueeze(0)
        return S.detach().cpu(), tb

    scale = math.sqrt(max_tokens / float(L * M))
    Lt = max(MIN_SIDE, int(L * scale));
    Lt = min(Lt, L)
    Mt = max(MIN_SIDE, int(M * scale));
    Mt = min(Mt, M)
    si = max(1, int(Lt * float(stride_frac)))
    sj = max(1, int(Mt * float(stride_frac)))

    i_starts = list(range(0, max(1, L - Lt + 1), si))
    j_starts = list(range(0, max(1, M - Mt + 1), sj))
    if i_starts[-1] != L - Lt: i_starts.append(L - Lt)
    if j_starts[-1] != M - Mt: j_starts.append(M - Mt)

    B0 = int(tb["resA"].shape[0])
    logits_full = torch.full((B0, L, M), -1e9, dtype=torch.float32)

    model.eval()
    for i0 in i_starts:
        i1 = i0 + Lt
        for j0 in j_starts:
            j1 = j0 + Mt
            tb_crop = _slice_batch_2d(tb, i0, i1, j0, j1)
            tb_crop = _ensure_mask_match(tb_crop)
            tb_dev = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in tb_crop.items()}
            out = model_forward_from_batch(model, tb_dev)
            S = out.get("S", None)
            if S is None:
                continue
            if S.ndim == 4 and S.shape[1] == 1:
                S = S[:, 0]
            if S.ndim == 2:
                S = S.unsqueeze(0)
            S = S.detach().cpu()
            prev = logits_full[:, i0:i1, j0:j1]
            logits_full[:, i0:i1, j0:j1] = torch.maximum(prev, S)
        # ---- TTA: swap A/B and average ----
    if bool(P.eval_tta_swap):
        tb_swap = dict(tb)
        tb_swap["resA"], tb_swap["resB"] = tb["resB"], tb["resA"]
        tb_swap["maskA"], tb_swap["maskB"] = tb["maskB"], tb["maskA"]

        if "chainA" in tb_swap and "chainB" in tb_swap:
            tb_swap["chainA"], tb_swap["chainB"] = tb["chainB"], tb["chainA"]

        # label 
        if "y2d" in tb_swap:
            tb_swap["y2d"] = tb_swap["y2d"].transpose(1, 2)

        tb_dev2 = {
            k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in tb_swap.items()
        }

        with torch.no_grad():
            out2 = model_forward_from_batch(model, tb_dev2)

        S2 = out2.get("S", None)
        if S2 is not None:
            if S2.ndim == 4 and S2.shape[1] == 1:
                S2 = S2[:, 0]
            if S2.ndim == 2:
                S2 = S2.unsqueeze(0)

            #  AB
            S2 = S2.transpose(1, 2).detach().cpu()
            if S2.shape[0] != logits_full.shape[0]:
                Bm = min(int(S2.shape[0]), int(logits_full.shape[0]))
                S2 = S2[:Bm]
                logits_full = logits_full[:Bm]
            logits_full = 0.5 * (logits_full + S2)

    return logits_full, tb


@torch.no_grad()
def eval_l2_topk_quick(model, dl, device: str = DEVICE, thr_true: float = 0.5, stride_frac: float = 0.5):
    skip_allneg = bool(P.eval_skip_allneg)

    model.eval()
    top50, top10, top5, topL5, topL10 = [], [], [], [], []
    max_tokens = int(MAX_2D_TOKENS_EVAL)

    cnt_total = 0
    cnt_used = 0
    cnt_allneg = 0
    cnt_haspos = 0
    cnt_bad = 0

    for batch in dl:
        logits2d, tb = infer_logits_fullmap_multicrop(
            model, batch, device=device, max_tokens=max_tokens, stride_frac=float(stride_frac)
        )
        if logits2d is None:
            cnt_bad += 1
            continue

        y2d, valid2d = _valid2d_and_y2d(tb)
        if y2d is None or valid2d is None or (not bool(valid2d.any())):
            cnt_bad += 1
            continue

        cnt_total += 1
        s2d = logits2d

        has_pos = bool(((y2d > thr_true) & valid2d).any())
        if has_pos:
            cnt_haspos += 1
        else:
            cnt_allneg += 1
            if skip_allneg:
                continue

        cnt_used += 1

        L = int(tb["resA"].shape[1])
        M = int(tb["resB"].shape[1])
        Lref = max(1, int(min(L, M)))

        topL5.append(precision_at_k(y2d, s2d, valid2d, k=max(1, Lref // 5), thr_true=thr_true))
        topL10.append(precision_at_k(y2d, s2d, valid2d, k=max(1, Lref // 10), thr_true=thr_true))
        top50.append(precision_at_k(y2d, s2d, valid2d, k=50, thr_true=thr_true))
        top10.append(precision_at_k(y2d, s2d, valid2d, k=10, thr_true=thr_true))
        top5.append(precision_at_k(y2d, s2d, valid2d, k=5, thr_true=thr_true))

    def _mean(xs):
        return float(np.mean(xs)) if len(xs) > 0 else 0.0

    out = {"L5": _mean(topL5), "L10": _mean(topL10), "50": _mean(top50), "10": _mean(top10), "5": _mean(top5)}

    print(
        f"[val_check] skip_allneg={int(skip_allneg)} total_valid={cnt_total} used={cnt_used} haspos={cnt_haspos} allneg={cnt_allneg} bad={cnt_bad}",
        flush=True)
    return out


# ---------------- Training forward ----------------
def forward_one(model, batch, ep, task="SITE"):
    aux_losses = {}
    pre_has_pos = False
    npos_before_crop = 0

    if task == "SITE":
        # [v2] ate-basedate_thr(0.0001)""rop
        pre_has_pos, npos_before_crop = sample_has_pos_rate(batch, rate_thr=0.0001)

        force_anchor = bool(pre_has_pos) and bool(L2_FORCE_ANCHOR_IF_HASPOS)
        _tries = 30 if force_anchor else 20
        batch, _ = crop_batch_2d_pos_aware(
            batch, MAX_2D_TOKENS_TRAIN, return_idx=True, max_tries=_tries, force=force_anchor
        )

        if pre_has_pos:
            y2d_c = batch.get("y2d", None)
            if torch.is_tensor(y2d_c):
                npos_after_crop = int((y2d_c > 0.5).sum().item())

                # 1) crop focus crop
                if npos_after_crop == 0:
                    batch, _ = crop_batch_2d_pos_aware(
                        batch, MAX_2D_TOKENS_TRAIN, return_idx=True,
                        focus_max_tokens=32768, max_tries=40, force=force_anchor
                    )
                    y2d_c2 = batch.get("y2d", None)
                    if torch.is_tensor(y2d_c2):
                        npos_after_crop = int((y2d_c2 > 0.5).sum().item())

                # 2) 
                if npos_before_crop > 0 and npos_after_crop < max(10, int(0.1 * npos_before_crop)):
                    batch, _ = crop_batch_2d(batch, MAX_2D_TOKENS_TRAIN, center=True, return_idx=True)

    else:
        batch, _ = crop_batch_2d(batch, MAX_2D_TOKENS_TRAIN, center=False, return_idx=True)

    L_len = batch["resA"].shape[1]
    M_len = batch["resB"].shape[1]
    if min(L_len, M_len) < 2:
        dummy_loss = torch.tensor(0.0, device=DEVICE, requires_grad=True)
        dummy_aux = dict(loss_2d=0.0, loss_resA=0.0, loss_resB=0.0, ohem_keepNeg=0, ohem_keepPix=0, crop_has_pos=0)
        return {}, dummy_loss, dummy_aux, batch, False

    resA = torch.nan_to_num(batch["resA"], nan=0.0).to(DEVICE).float()
    resB = torch.nan_to_num(batch["resB"], nan=0.0).to(DEVICE).float()
    maskA = (batch["maskA"] > 0.5).to(DEVICE)
    maskB = (batch["maskB"] > 0.5).to(DEVICE)

    chainA = batch.get("chainA", None)
    chainB = batch.get("chainB", None)
    chainA = torch.nan_to_num(chainA).to(DEVICE).float() if chainA is not None else None
    chainB = torch.nan_to_num(chainB).to(DEVICE).float() if chainB is not None else None

    def _align_1d(res, mask, chain):
        L = res.size(1) if res.dim() == 3 else res.size(0)
        if mask is not None and mask.size(-1) != L:
            mask = mask[..., :L]
        if chain is not None:
            if chain.dim() == 3 and chain.size(1) != L:
                chain = chain[:, :L, :]
            elif chain.dim() == 2:
                if chain.size(0) == L:
                    pass
                elif chain.size(1) == L:
                    chain = chain[:, :L]
                else:
                    if abs(chain.size(0) - L) <= abs(chain.size(1) - L):
                        chain = chain[:L, :]
                    else:
                        chain = chain[:, :L]
        return mask, chain

    maskA, chainA = _align_1d(resA, maskA, chainA)
    maskB, chainB = _align_1d(resB, maskB, chainB)

    out = model(resA, maskA, chainA, resB, maskB, chainB)
    aux = {}

    if task == "SITE":
        y2d = batch["y2d"].to(DEVICE).float()
        yA = batch["y_res_A"].to(DEVICE).float()
        yB = batch["y_res_B"].to(DEVICE).float()
        if yA.ndim == 1: yA = yA.unsqueeze(0)
        if yB.ndim == 1: yB = yB.unsqueeze(0)

        La = maskA.shape[1]
        Lb = maskB.shape[1]
        if y2d.shape[1] > La: y2d = y2d[:, :La, :]
        if y2d.shape[2] > Lb: y2d = y2d[:, :, :Lb]
        if yA.shape[1] > La: yA = yA[:, :La]
        if yB.shape[1] > Lb: yB = yB[:, :Lb]
        # ---- Loss computed below (binary BCE + optional list/rank; supports L1/L1.5 consistency) ----
        Slog = out["S"]
        Slog = torch.nan_to_num(Slog, nan=0.0, posinf=20.0, neginf=-20.0)
        Slog = Slog.clamp(-20.0, 20.0)
        if Slog.ndim == 4 and Slog.shape[1] == 1:
            Slog = Slog[:, 0]
        elif Slog.ndim == 2:
            Slog = Slog.unsqueeze(0)

        valid2d = maskA.unsqueeze(-1) & maskB.unsqueeze(1)

        # -------- L1<->L2 support consistency (bridges L1.5 without extra labels) --------
        tau = float(max(0.05, float(P.l12_support_tau)))  # always defined for logging (avoid UnboundLocal)

        loss_support = torch.tensor(0.0, device=DEVICE)
        if (float(P.l12_support_w) > 0) and (ep >= int(P.l12_support_start_epoch)) and valid2d.any():

            S = Slog.masked_fill(~valid2d, -1e9)  #  tau

            sA_sup = tau * torch.logsumexp(S / tau, dim=2)  # [B,L]
            sB_sup = tau * torch.logsumexp(S / tau, dim=1)  # [B,M]

            yA_bin = (yA > 0.5).float()
            yB_bin = (yB > 0.5).float()

            sA_shift = sA_sup - float(P.l12_support_margin)
            sB_shift = sB_sup - float(P.l12_support_margin)

            sA_shift = sA_shift.clamp(-30, 30)
            sB_shift = sB_shift.clamp(-30, 30)

            if maskA.any():
                loss_supportA = F.binary_cross_entropy_with_logits(sA_shift[maskA], yA_bin[maskA], reduction="mean")
            else:
                loss_supportA = torch.tensor(0.0, device=DEVICE)

            if maskB.any():
                loss_supportB = F.binary_cross_entropy_with_logits(sB_shift[maskB], yB_bin[maskB], reduction="mean")
            else:
                loss_supportB = torch.tensor(0.0, device=DEVICE)

            loss_support = loss_supportA + loss_supportB
            aux["loss_supportA"] = loss_supportA.detach()
            aux["loss_supportB"] = loss_supportB.detach()

        # -------- L1<->L2 consistency (softly align residue scores with max-contact evidence) --------
        cons_w_max = float(P.l12_cons_w)
        cons_start = int(P.l12_cons_start_epoch)
        cons_ramp = int(P.l12_cons_ramp_epochs)
        cons_tau = float(P.l12_cons_tau)
        cons_w = 0.0
        loss_cons = torch.tensor(0.0, device=DEVICE)

        if cons_w_max > 0 and ep >= cons_start and (maskA.any() and maskB.any()):
            # linear ramp
            if cons_ramp <= 0:
                cons_w = cons_w_max
            else:
                cons_w = cons_w_max * min(1.0, float(ep - cons_start + 1) / float(cons_ramp))

            with torch.no_grad():
                if Slog.ndim == 2:
                    Slog_3d = Slog.unsqueeze(0)  # [La, Lb] -> [1, La, Lb]
                else:
                    Slog_3d = Slog  # already [B, La, Lb]

                B, La, Lb = Slog_3d.shape

                # masksloghape
                if maskA.ndim == 1:
                    maskA = maskA.unsqueeze(0)  # [La] -> [1, La]
                if maskB.ndim == 1:
                    maskB = maskB.unsqueeze(0)  # [Lb] -> [1, Lb]

                mA_3d = maskA.unsqueeze(-1).expand(B, La, Lb)  # [B, La, 1] -> [B, La, Lb]
                mB_3d = maskB.unsqueeze(1).expand(B, La, Lb)  # [B, 1, Lb] -> [B, La, Lb]
                valid2d_cons = mA_3d & mB_3d  # [B, La, Lb]

                # mask: 
                S_masked_for_A = Slog_3d.masked_fill(~valid2d_cons, -1e9)
                S_masked_for_B = Slog_3d.masked_fill(~valid2d_cons, -1e9)

                # vidence (logit)
                maxA = torch.max(S_masked_for_A, dim=2).values  # [B, La]: max over Lb
                maxB = torch.max(S_masked_for_B, dim=1).values  # [B, Lb]: max over La

                # [BUG FIX v3]  maxA / cons_tau (tau=0.15)  20 ogit
                # cons_tau  sigmoid 
                tA_logit = torch.sigmoid(maxA / max(cons_tau, 1.0)).clamp(0.01, 0.99)
                tB_logit = torch.sigmoid(maxB / max(cons_tau, 1.0)).clamp(0.01, 0.99)

                # 
                if B == 1:
                    tA_logit = tA_logit.squeeze(0)  # [1, La] -> [La]
                    tB_logit = tB_logit.squeeze(0)  # [1, Lb] -> [Lb]

            sA = model._last_resA_logit
            sB = model._last_resB_logit

            if sA is not None and sB is not None:
                # 
                if sA.dim() == 3:
                    sA = sA.squeeze(-1)  # [B, L, 1] -> [B, L]
                if sB.dim() == 3:
                    sB = sB.squeeze(-1)  # [B, M, 1] -> [B, M]

                if sA.dim() == 2 and tA_logit.dim() == 1:
                    if sA.shape[0] == 1:
                        sA = sA.squeeze(0)  # [1, L] -> [L]
                    elif tA_logit.shape[0] == sA.shape[1]:
                        tA_logit = tA_logit.unsqueeze(0).expand_as(sA)

                if sB.dim() == 2 and tB_logit.dim() == 1:
                    if sB.shape[0] == 1:
                        sB = sB.squeeze(0)  # [1, M] -> [M]
                    elif tB_logit.shape[0] == sB.shape[1]:
                        tB_logit = tB_logit.unsqueeze(0).expand_as(sB)

                if maskA.any():
                    try:
                        sA_prob = torch.sigmoid(sA[maskA])
                        loss_cons = loss_cons + F.mse_loss(sA_prob, tA_logit[maskA], reduction="mean")
                    except (IndexError, RuntimeError) as e:
                        print(
                            f"[WARNING] consistency loss A indexing failed: "
                            f"sA.shape={sA.shape}, tA_logit.shape={tA_logit.shape}, "
                            f"maskA.shape={maskA.shape}"
                        )

                if maskB.any():
                    try:
                        sB_prob = torch.sigmoid(sB[maskB])
                        loss_cons = loss_cons + F.mse_loss(sB_prob, tB_logit[maskB], reduction="mean")
                    except (IndexError, RuntimeError) as e:
                        print(
                            f"[WARNING] consistency loss B indexing failed: "
                            f"sB.shape={sB.shape}, tB_logit.shape={tB_logit.shape}, "
                            f"maskB.shape={maskB.shape}"
                        )

        aux["loss_cons"] = loss_cons.detach()
        aux["alpha"] = float(cons_w)

        aux["sup_w"] = float(P.l12_support_w)
        aux["sup_tau"] = float(tau)

        # -------- Optional: fragment logits (model has logit_fragA/B) --------
        loss_frag = torch.tensor(0.0, device=DEVICE)
        frag_w = float(float(P.l15_w))
        if (frag_w > 0) and ("logit_fragA" in out) and ("logit_fragB" in out):
            lfA = out["logit_fragA"];
            lfB = out["logit_fragB"]
            if lfA.ndim == 1: lfA = lfA.unsqueeze(0)
            if lfB.ndim == 1: lfB = lfB.unsqueeze(0)
            lfA = torch.nan_to_num(lfA, nan=0.0, posinf=20.0, neginf=-20.0)
            lfB = torch.nan_to_num(lfB, nan=0.0, posinf=20.0, neginf=-20.0)
            lfA = torch.clamp(lfA, -20.0, 20.0)
            lfB = torch.clamp(lfB, -20.0, 20.0)
            yA_bin = (yA > 0.5).float();
            yB_bin = (yB > 0.5).float()
            lfA = lfA[:, :yA_bin.shape[1]];
            lfB = lfB[:, :yB_bin.shape[1]]
            loss_fragA = F.binary_cross_entropy_with_logits(lfA[maskA], yA_bin[maskA]) if maskA.any() else torch.tensor(
                0.0, device=DEVICE)
            loss_fragB = F.binary_cross_entropy_with_logits(lfB[maskB], yB_bin[maskB]) if maskB.any() else torch.tensor(
                0.0, device=DEVICE)
            loss_frag = loss_fragA + loss_fragB
            aux["loss_fragA"] = loss_fragA.detach()
            aux["loss_fragB"] = loss_fragB.detach()
            aux["frag_w"] = float(frag_w)

        loss_2d = torch.tensor(0.0, device=DEVICE)
        keepNeg = 0
        keepPix = 0

        aux_losses = {}
        # 
        _l2w_eff = float(P.l2_w)
        if valid2d.any() and _l2w_eff > 0.0:
            logit_v = Slog[valid2d]
            y_v = y2d[valid2d]
            y_bin = (y_v > 0.5).float()
            npos = int(y_bin.sum().item())
            aux["crop_npos"] = npos

            neg_per_pos = int(int(P.l2_neg_per_pos))
            neg_min = int(int(P.l2_neg_min))
            neg_cap = int(int(P.l2_neg_cap))
            hard_frac = float(float(P.l2_hardneg_frac))

            in_focus = (_cfg_focus_epochs > 0 and ep <= _cfg_focus_epochs)
            # ===== HARD DEFAULTS (avoid UnboundLocalError across branches) =====
            rank_w = 0.0
            list_w_eff = 0.0
            bce_w = 0.0
            poly_w = float(float(P.l2_poly_w))
            rank_margin = float(float(P.l2_rank_margin))
            aux["rank_w_used"] = float(rank_w)
            aux["list_w_used"] = float(list_w_eff)
            aux["bce_w_used"] = float(bce_w)
            aux["poly_w_used"] = float(poly_w)
            aux["rank_margin_used"] = float(rank_margin)
            # ================================================================
            if npos > 0:
                neg_budget = max(
                    int(neg_min),
                    min(int(neg_cap), int(npos * neg_per_pos))
                )
                sel_idx, npos_k, nneg_k = _sample_l2_pixels(
                    logit_v, y_bin, max_neg=neg_budget,
                    hard_frac=float(hard_frac),
                    neg_per_pos=int(neg_per_pos),
                    neg_min=int(neg_min)
                )

                # === FIX: define selected logits/labels ===
                l_sel = logit_v[sel_idx]
                y_sel = y_bin[sel_idx]
                l_sel = torch.nan_to_num(l_sel, nan=0.0, posinf=20.0, neginf=-20.0)
                l_sel = l_sel.clamp(-20.0, 20.0)
                # === FIX: recompute keep stats from final y_sel (ground truth) ===
                npos_sel, nneg_sel, nsel = _ohem_keep_stats(y_sel)
                keepNeg = int(nneg_sel)
                keepPix = int(nsel)
                aux["ohem_keepPos"] = int(npos_sel)

                # ----- BCE (brake only) -----
                ratio = float(nneg_sel) / float(max(1, npos_sel))
                pos_w_val = float(min(50.0, max(1.0, math.sqrt(ratio))))  # cap for Top-K focus
                loss_bce = F.binary_cross_entropy_with_logits(
                    l_sel, y_sel,
                    pos_weight=torch.tensor(pos_w_val, device=DEVICE),
                    reduction="mean"
                )
                # [FIX-F2] loss
                focal_w = float(getattr(P, 'l2_focal_w', 0.0))
                dice_w = float(getattr(P, 'l2_dice_w', 0.0))
                lovasz_w = float(getattr(P, 'l2_lovasz_w', 0.0))
                loss_focal = torch.tensor(0.0, device=DEVICE)
                loss_dice = torch.tensor(0.0, device=DEVICE)
                loss_lovasz = torch.tensor(0.0, device=DEVICE)
                if focal_w > 0.0:
                    loss_focal = binary_focal_loss_with_logits(
                        l_sel, y_sel,
                        alpha=float(getattr(P, 'l2_focal_alpha', 0.25)),
                        gamma=float(getattr(P, 'l2_focal_gamma', 2.0))
                    )
                if dice_w > 0.0:
                    loss_dice = soft_dice_loss_from_logits(l_sel, y_sel)
                if lovasz_w > 0.0:
                    loss_lovasz = lovasz_hinge_flat(l_sel, y_sel)
                aux['loss_focal'] = loss_focal.detach()
                aux['loss_dice'] = loss_dice.detach()
                aux['loss_lovasz'] = loss_lovasz.detach()

                # ----- weights + ramp (compute FIRST to avoid NaN poisoning when w==0) -----
                bce_w_max = float(float(P.l2_bce_w_max))
                bce_start = int(int(P.l2_bce_start_epoch))
                bce_ramp_ep = int(int(P.l2_bce_ramp_epochs))
                bce_hold_ep = int(int(P.l2_bce_hold_epochs))  # hold for stability
                bce_decay = float(float(P.l2_bce_decay))  # then exponential decay

                if ep < bce_start:
                    bce_w = 0.0
                else:
                    t = min(1.0, float(ep - bce_start + 1) / float(max(1, bce_ramp_ep)))
                    bce_w = bce_w_max * t
                    # [FIX-F4] BCEloss,0
                    if ep > bce_hold_ep:
                        # let Top-K dominate later: gradually down-weight BCE
                        decay_factor = bce_decay ** (ep - bce_hold_ep)
                        bce_w = max(bce_w * decay_factor, 0.5 * bce_w_max)

                # listwise
                list_w = float(float(P.l2_listwise_w))
                list_start = int(int(P.l2_list_start_epoch))
                list_ramp_ep = int(int(P.l2_list_ramp_epochs))
                if ep < list_start:
                    list_w_eff = 0.0
                else:
                    t = min(1.0, float(ep - list_start + 1) / float(max(1, list_ramp_ep)))
                    list_w_eff = list_w * t

                # rank
                rank_w_max = float(float(P.l2_rank_w_max))
                rank_start = int(int(P.l2_rank_start_epoch))
                rank_ramp_ep = int(int(P.l2_rank_ramp_epochs))
                if ep < rank_start:
                    rank_w = 0.0
                else:
                    t = min(1.0, float(ep - rank_start + 1) / float(max(1, rank_ramp_ep)))
                    rank_w = rank_w_max * t

                # poly (Top-K rescue) depends on rank mining; default off for binary-first
                poly_w = float(float(P.l2_poly_w))

                if in_focus:
                    rank_w = float(rank_w) * float(_cfg_boost_w)
                    list_w_eff = float(list_w_eff) * float(_cfg_list_focus)

                aux["rank_w_used"] = float(rank_w)
                aux["list_w_used"] = float(list_w_eff)
                aux["bce_w_used"] = float(bce_w)
                aux["poly_w_used"] = float(poly_w)

                # ----- compute aux losses only if they can contribute (avoid NaN * 0 == NaN) -----
                loss_list = torch.tensor(0.0, device=DEVICE)
                loss_rank = torch.tensor(0.0, device=DEVICE)
                loss_poly = torch.tensor(0.0, device=DEVICE)

                do_list = (list_w_eff > 0.0)
                do_rank = (rank_w > 0.0) or (poly_w > 0.0)

                if do_list:
                    base_tau = float(float(P.l2_listwise_tau))
                    tau_min = float(float(P.l2_listwise_tau_min))
                    anneal_ep = int(int(P.l2_listwise_anneal_epochs))
                    list_negk = int(int(P.l2_listwise_neg_k))

                    if ep <= anneal_ep:
                        t = float(ep) / float(max(1, anneal_ep))
                        list_tau = base_tau - t * (base_tau - tau_min)
                    else:
                        list_tau = tau_min

                    loss_list = _listwise_pos_mass_loss(
                        l_sel, y_sel,
                        neg_k=list_negk,
                        tau=list_tau
                    )
                    # listwise loss
                    loss_list = torch.nan_to_num(loss_list, nan=0.0, posinf=1e6, neginf=0.0).clamp_min(0.0)

                if do_rank:
                    rank_margin = float(float(P.l2_rank_margin))
                    rank_kneg = int(int(P.l2_rank_neg_k))
                    rank_kpos = int(int(P.l2_rank_pos_k))
                    if in_focus:
                        rank_margin = float(rank_margin) * float(_cfg_margin_focus)
                    aux["in_focus"] = int(in_focus)
                    aux["rank_margin_eff"] = float(rank_margin)

                    loss_rank = _topk_margin_rank_loss(
                        l_sel, y_sel,
                        k_neg=rank_kneg, k_pos=rank_kpos,
                        margin=rank_margin
                    )
                    # [v2 CRITICAL FIX] ard clamp
                    loss_rank = torch.nan_to_num(loss_rank, nan=0.0, posinf=1e6, neginf=0.0).clamp_min(0.0)

                    # Poly miss penalty: rescue when Top-K misses positives
                    if poly_w > 0.0:
                        poly_k = int(int(P.l2_poly_k))
                        poly_pw = float(float(P.l2_poly_pow))
                        k = min(poly_k, int(l_sel.numel()))
                        if k > 0:
                            top_idx = torch.topk(l_sel, k=k, largest=True).indices
                            top_y = y_sel[top_idx]
                            if (top_y > 0.5).sum() < 1:
                                all_pos_idx = (y_sel > 0.5).nonzero(as_tuple=False).view(-1)
                                if all_pos_idx.numel() > 0:
                                    worst_pos_logit = l_sel[all_pos_idx].min()
                                    kth_logit = l_sel[top_idx[-1]]
                                    if worst_pos_logit < kth_logit:
                                        gap = (kth_logit - worst_pos_logit)  # keep grad
                                        loss_poly = (gap ** poly_pw)
                                        loss_poly = torch.nan_to_num(loss_poly, nan=0.0, posinf=1e6,
                                                                     neginf=0.0).clamp_min(0.0)
                else:
                    # keep logs consistent even when rank is disabled
                    aux["in_focus"] = int(in_focus)
                    aux["rank_margin_eff"] = 0.0

                # ----- logit regularization -----
                reg_w = float(float(P.l2_logit_reg_w))
                loss_reg = reg_w * torch.mean(l_sel ** 2)
                loss_reg = loss_reg.clamp(0.0, float(float(P.l2_logit_reg_cap)))

                loss_2d = (
                        (bce_w * loss_bce)
                        + (list_w_eff * loss_list)
                        + (rank_w * loss_rank)
                        + (poly_w * loss_poly)
                        + (focal_w * loss_focal)
                        + (dice_w * loss_dice)
                        + (lovasz_w * loss_lovasz)
                        + loss_reg
                )
                aux_losses_extra = {}
                try:
                    class _AuxP:
                        pass

                    P_aux = _AuxP()
                    for _k in ["l2_calibration_w", "l2_calibration_bins"]:
                        if hasattr(P, _k):
                            setattr(P_aux, _k, getattr(P, _k))

                    #  loss
                    P_aux.l2_focal_w = 0.0
                    P_aux.l2_dice_w = 0.0
                    P_aux.l2_poly_w = 0.0
                    P_aux.l2_contrast_w = 0.0
                    calib_w = float(getattr(P, 'l2_calibration_w', 0.0))
                    n_pos = (y_sel > 0.5).sum().item()
                    if float(getattr(P, 'l2_calibration_w', 0.0)) > 0:
                        loss_2d, aux_losses_extra = integrate_new_losses(loss_2d, l_sel, y_sel, ep, P_aux)
                except Exception as _e:
                    aux_losses_extra = {"auxloss_err": str(_e)[:120]}

                if aux_losses_extra:
                    aux.update(aux_losses_extra)
                aux.update({
                    "loss_bce": loss_bce.detach(),
                    "loss_list": loss_list.detach(),
                    "loss_rank": loss_rank.detach(),
                    "loss_poly": loss_poly.detach(),
                    "rank_w": float(rank_w),
                    "list_w_eff": float(list_w_eff),
                    "ohem_keepNeg": int(keepNeg),
                    "ohem_keepPix": int(keepPix),
                    "loss_2d": loss_2d.detach(),
                    "rank_gap": float(_rank_gap_stats(l_sel, y_sel).detach().cpu())
                })


            else:
                drop_p = float(float(P.l2_allneg_drop_p))
                if random.random() < drop_p:
                    loss_2d = torch.tensor(0.0, device=DEVICE)
                    keepNeg = 0
                    keepPix = 0
                else:
                    k_bg = min(int(int(P.l2_bg_hard_k)), int(logit_v.numel()))
                    if k_bg > 0:
                        l_bg = torch.topk(logit_v, k=k_bg, largest=True).values

                        loss_bg = F.binary_cross_entropy_with_logits(l_bg, torch.zeros_like(l_bg))
                        bg_w = float(float(P.l2_bg_w))
                        loss_2d = bg_w * loss_bg
                    keepNeg = int(k_bg)
                    keepPix = int(k_bg)

                aux.update({
                    "ohem_keepPos": 0,
                    "ohem_keepNeg": int(keepNeg),
                    "ohem_keepPix": int(keepPix),
                    "crop_has_pos": int(pre_has_pos),
                    "loss_2d": loss_2d.detach(),
                })

        # ============================================================
        # :
        # ============================================================
        zA_logit = model._last_resA_logit  # [B, L, 1] rad
        zB_logit = model._last_resB_logit  # [B, M, 1] rad
        if zA_logit is not None and zB_logit is not None:
            zA_logit = zA_logit.squeeze(-1)  # [B, L]
            zB_logit = zB_logit.squeeze(-1)  # [B, M]
            # yA/yB
            zA_logit = zA_logit[:, :yA.shape[1]]
            zB_logit = zB_logit[:, :yB.shape[1]]
            # [Trick] Label smoothing ONLY on A chain (real labels)
            #   dummy0 smoothing
            _ls = float(getattr(P, 'label_smoothing_l1', 0.0))
            yA_f = yA.clamp(0, 1)
            yB_f = yB.clamp(0, 1)   # dummy B chain: NO smoothing
            if _ls > 0.0:
                yA_f = yA_f * (1.0 - _ls) + _ls * 0.5  # A chain only
            l1_pos_w = float(getattr(P, 'l1_pos_weight', 5.0))
            pw = torch.tensor(l1_pos_w, device=DEVICE, dtype=zA_logit.dtype)

            # ---- Base BCE loss ----
            _focal_w   = float(getattr(P, 'l1_focal_w',   0.0))
            _asl_w     = float(getattr(P, 'l1_asl_w',     0.0))
            _bce_w     = max(0.0, 1.0 - _focal_w - _asl_w)

            if maskA.any():
                logA_m = zA_logit[maskA]
                yA_m   = yA_f[maskA]
                # 1) BCE component
                loss_resA_bce = F.binary_cross_entropy_with_logits(
                    logA_m, yA_m, pos_weight=pw, reduction="mean"
                ) if _bce_w > 0.0 else torch.tensor(0.0, device=DEVICE)
                # 2) Focal component
                if _focal_w > 0.0:
                    _fgamma = float(getattr(P, 'l1_focal_gamma', 2.0))
                    _falpha = float(getattr(P, 'l1_focal_alpha', 0.25))
                    loss_resA_focal = binary_focal_loss_with_logits(
                        logA_m, yA_m, alpha=_falpha, gamma=_fgamma, reduction='mean'
                    )
                else:
                    loss_resA_focal = torch.tensor(0.0, device=DEVICE)
                # 3) ASL component (Asymmetric Shifting Loss)
                if _asl_w > 0.0:
                    _gn = float(getattr(P, 'l1_asl_gamma_neg', 0.1))
                    _pA = torch.sigmoid(logA_m)
                    _yA = (yA_m > 0.5).float()
                    # shift negative probs down by gamma_neg, clip at 0
                    _pA_neg_shifted = (_pA - _gn).clamp_min(0.0)
                    _pA_asl = _pA * _yA + _pA_neg_shifted * (1.0 - _yA)
                    _pA_asl = _pA_asl.clamp(1e-7, 1.0 - 1e-7)
                    loss_resA_asl = -(
                        _yA * torch.log(_pA_asl) +
                        (1.0 - _yA) * torch.log(1.0 - _pA_asl)
                    ).mean()
                else:
                    loss_resA_asl = torch.tensor(0.0, device=DEVICE)
                loss_resA = _bce_w * loss_resA_bce + _focal_w * loss_resA_focal + _asl_w * loss_resA_asl
            else:
                loss_resA = torch.tensor(0.0, device=DEVICE)

            loss_resB = F.binary_cross_entropy_with_logits(
                zB_logit[maskB], yB_f[maskB], pos_weight=pw, reduction="mean"
            ) if maskB.any() else torch.tensor(0.0, device=DEVICE)

            # ---- L1 Residue-level Rank Loss ----
            # "" loss
            # (rank1=0/list1=0)
            _l1_rank_w = float(getattr(P, 'l1_rank_w', 0.0))
            _l1_rank_ep = int(getattr(P, 'l1_rank_start_epoch', 999))
            loss_resA_rank = torch.tensor(0.0, device=DEVICE)
            if _l1_rank_w > 0.0 and ep >= _l1_rank_ep and maskA.any():
                _margin    = float(getattr(P, 'l1_rank_margin', 0.5))
                _n_pairs   = int(getattr(P, 'l1_rank_n_pairs', 128))
                _hard_frac = float(getattr(P, 'l1_rank_neg_hard_frac', 0.5))
                _rank_losses = []
                B_r = int(zA_logit.shape[0])
                for _b in range(B_r):
                    _m = maskA[_b].bool() if maskA.ndim > 1 else maskA.bool()
                    if not _m.any():
                        continue
                    _logit_b = zA_logit[_b][_m]
                    _lab_b   = yA_f[_b][_m] if yA_f.ndim > 1 else yA_f[_m]
                    if (_lab_b > 0.5).sum() == 0:
                        continue
                    _rl = l1_pairwise_rank_loss(
                        _logit_b, _lab_b,
                        margin=_margin, n_pairs=_n_pairs, hard_neg_frac=_hard_frac
                    )
                    _rank_losses.append(_rl)
                if _rank_losses:
                    loss_resA_rank = torch.stack(_rank_losses).mean()
            aux['loss_resA_rank'] = loss_resA_rank.detach() if torch.is_tensor(loss_resA_rank) else 0.0

            # ---- L1 List Loss v2 (Softplus+ApproxNDCG) ----
            _l1_list_w  = float(getattr(P, 'l1_list_w', 0.0))
            _l1_list_ep = int(getattr(P, 'l1_list_start_epoch', 999))
            loss_resA_list = torch.tensor(0.0, device=DEVICE)
            if _l1_list_w > 0.0 and ep >= _l1_list_ep and maskA.any():
                _tau    = float(getattr(P, 'l1_list_tau', 2.0))
                _neg_k  = int(getattr(P, 'l1_list_neg_k', 64))
                _list_losses = []
                B_l = int(zA_logit.shape[0])
                for _b in range(B_l):
                    _m = maskA[_b].bool() if maskA.ndim > 1 else maskA.bool()
                    if not _m.any():
                        continue
                    _logit_b = zA_logit[_b][_m]
                    _lab_b   = yA_f[_b][_m] if yA_f.ndim > 1 else yA_f[_m]
                    if (_lab_b > 0.5).sum() == 0:
                        continue
                    _ll = l1_listmle_loss(_logit_b, _lab_b, tau=_tau, neg_k=_neg_k)
                    _list_losses.append(_ll)
                if _list_losses:
                    loss_resA_list = torch.stack(_list_losses).mean()
            aux['loss_resA_list'] = loss_resA_list.detach() if torch.is_tensor(loss_resA_list) else 0.0
        else:
            p_res_A = out["p_res_A"]
            p_res_B = out["p_res_B"]
            eps = 1e-6
            p_res_A = torch.nan_to_num(p_res_A, nan=0.5).clamp(eps, 1.0 - eps)
            p_res_B = torch.nan_to_num(p_res_B, nan=0.5).clamp(eps, 1.0 - eps)
            yA_c = torch.nan_to_num(yA).clamp(0, 1)
            yB_c = torch.nan_to_num(yB).clamp(0, 1)
            l1a = F.binary_cross_entropy(p_res_A, yA_c, reduction='none')
            l1b = F.binary_cross_entropy(p_res_B, yB_c, reduction='none')
            loss_resA = l1a[maskA].mean() if maskA.any() else torch.tensor(0.0, device=DEVICE)
            loss_resB = l1b[maskB].mean() if maskB.any() else torch.tensor(0.0, device=DEVICE)

        l2w = float(P.l2_w)
        l1w = float(P.l1_w)

        # [v2] loss: aN/Infard clamp
        def safe_loss_component(loss_tensor, name, max_val=20.0):
            """Clamp or replace unstable loss components before aggregation."""
            if torch.isnan(loss_tensor) or torch.isinf(loss_tensor):
                print(f"[WARNING] {name} is NaN/Inf, setting to 0", flush=True)
                return torch.tensor(0.0, device=loss_tensor.device, requires_grad=True)
            # arnclamp
            if loss_tensor.item() > max_val:
                print(f"[WARN] {name}={loss_tensor.item():.4f} > {max_val} (not clamped, grad preserved)", flush=True)
            return loss_tensor

        # oss
        loss_2d = safe_loss_component(loss_2d, "loss_2d", max_val=15.0)
        loss_resA = safe_loss_component(loss_resA, "loss_resA", max_val=10.0)
        loss_resB = safe_loss_component(loss_resB, "loss_resB", max_val=10.0)
        loss_support = safe_loss_component(loss_support, "loss_support", max_val=8.0)
        if not torch.isfinite(loss_cons) or loss_cons.item() > 5.0:
            if not torch.isfinite(loss_cons):
                print(f"[GUARD] loss_cons is NaN/Inf, zeroing", flush=True)
            else:
                print(f"[GUARD] loss_cons={loss_cons.item():.4f} > 5.0, clamping (bug indicator!)", flush=True)
            loss_cons = torch.tensor(0.0, device=loss_cons.device)
        loss_frag = safe_loss_component(loss_frag, "loss_frag", max_val=5.0)

        # oss
        # safe_guard rank loss before adding
        loss_resA_rank = safe_loss_component(loss_resA_rank, "loss_resA_rank", max_val=5.0)
        if torch.is_tensor(loss_resA_list) and torch.isfinite(loss_resA_list) and loss_resA_list.item() > 3.0:
            loss_resA_list = loss_resA_list * (3.0 / loss_resA_list.item())
        loss_resA_list = safe_loss_component(loss_resA_list, "loss_resA_list", max_val=3.0)
        _l1_rank_w_eff = float(getattr(P, 'l1_rank_w', 0.0)) if int(getattr(P, 'l1_rank_start_epoch', 999)) <= ep else 0.0
        _l1_list_w_eff = float(getattr(P, 'l1_list_w', 0.0)) if int(getattr(P, 'l1_list_start_epoch', 999)) <= ep else 0.0

        loss = (l2w * loss_2d
                + l1w * (loss_resA + loss_resB)
                + float(P.l12_support_w) * loss_support
                + cons_w * loss_cons
                + frag_w * loss_frag
                + _l1_rank_w_eff * loss_resA_rank
                + _l1_list_w_eff * loss_resA_list)

        with torch.no_grad():
            if torch.isfinite(loss) and (loss.abs() > 1e6):
                print("[diag] huge loss parts:",
                      "loss_2d=", float(loss_2d.detach().cpu()),
                      "loss_resA=", float(loss_resA.detach().cpu()),
                      "loss_resB=", float(loss_resB.detach().cpu()),
                      "loss_support=", float(loss_support.detach().cpu()),
                      "loss_frag=", float(loss_frag.detach().cpu()),
                      "sup_w=", float(P.l12_support_w), "frag_w=", float(frag_w),
                      flush=True)
        aux["loss_resA"] = loss_resA.detach()
        aux["loss_resB"] = loss_resB.detach()
    else:
        loss = torch.tensor(0.0, device=DEVICE)

    if isinstance(aux_losses, dict) and aux_losses:
        aux.update(aux_losses)

    return out, loss, aux, batch, pre_has_pos


# ====================== RBP296  + Top-K eval ======================
@torch.no_grad()
def eval_rbp296_binary(
        model,
        dl,
        device: str = DEVICE,
        thr: float = None,
        thr_mode: str = "auto_mcc",
        thr_min: float = 0.001,
        thr_max: float = 0.5,
        thr_grid: int = 101,
        topk_ks: tuple = (10, 20, 50),
):
    """
    Evaluate RBP296 residue-level predictions.

    Uses ``model._last_resA_logit`` together with ``batch["y_res_A"]`` and
    reports binary metrics plus top-k recall / precision / hit-rate summaries.
    """
    model.eval()

    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
        have_sklearn = True
    except Exception:
        have_sklearn = False

    all_prob = []   # per-residue probabilities (flattened, for global binary metrics)
    all_lab  = []   # per-residue labels (0/1)

    # per-protein lists for AUROC/Top-K
    per_prob = []   # list of np arrays
    per_lab  = []   # list of np arrays
    per_len  = []   # sequence length

    n_used    = 0
    n_haspos  = 0
    n_allneg  = 0
    n_bad     = 0
    total_bce     = 0.0
    total_bce_pos = 0.0   # BCE for proteins with positives
    total_bce_neg = 0.0   # BCE for all-negative proteins
    n_bce         = 0
    n_bce_pos     = 0
    n_bce_neg     = 0

    def _choose_thr(prob_np, lab_np):
        if prob_np.size == 0:
            return 0.5
        mode = str(thr_mode or "auto_mcc").lower()
        if mode == "fixed":
            return float(0.5 if thr is None else thr)
        tmin = float(thr_min)
        tmax = float(thr_max)
        if tmax <= tmin:
            tmin, tmax = 0.05, 0.95
        grid = int(max(11, thr_grid))
        ts = np.linspace(tmin, tmax, grid, dtype=np.float32)
        y = (lab_np > 0.5).astype(np.int32)
        best_t = float(ts[grid // 2])
        best_s = -1e9
        for t in ts:
            p = (prob_np >= float(t)).astype(np.int32)
            tp = int(((p == 1) & (y == 1)).sum())
            fp = int(((p == 1) & (y == 0)).sum())
            fn = int(((p == 0) & (y == 1)).sum())
            tn = int(((p == 0) & (y == 0)).sum())
            if mode == "auto_f1":
                denom = (2 * tp + fp + fn)
                s = (2 * tp / denom) if denom > 0 else 0.0
            elif mode == "auto_acc":
                denom = tp + tn + fp + fn
                s = ((tp + tn) / denom) if denom > 0 else 0.0
            else:  # auto_mcc (default)
                denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
                s = ((tp * tn - fp * fn) / math.sqrt(denom)) if denom > 0 else 0.0
            if s > best_s:
                best_s = s
                best_t = float(t)
        return float(best_t)

    for batch in dl:
        # ---- move batch to device & run forward ----
        tb = {}
        for k, v in batch.items():
            tb[k] = v.to(device) if torch.is_tensor(v) else v

        y_res_A = tb.get("y_res_A", None)
        if y_res_A is None:
            n_bad += 1
            continue

        maskA = tb.get("maskA", None)

        # run forward to populate _last_resA_logit
        # : model(resA, maskA, chainA, resB, maskB, chainB)
        try:
            with torch.no_grad():
                _ = model_forward_from_batch(model, tb)
        except Exception as _e:
            n_bad += 1
            continue

        raw_logit = getattr(model, "_last_resA_logit", None)
        if raw_logit is None:
            n_bad += 1
            continue

        # ----  batch  ----
        # raw_logit: [B, L, 1] or [B, L] or [L]
        # y_res_A:   [B, L]    or [L]
        # maskA:     [B, L]    or [L]
        if raw_logit.ndim == 3:
            raw_logit = raw_logit.squeeze(-1)   # [B, L, 1] -> [B, L]
        if raw_logit.ndim == 1:
            raw_logit = raw_logit.unsqueeze(0)  # [L] -> [1, L]
        #  raw_logit: [B, L]

        if y_res_A.ndim == 1:
            y_res_A = y_res_A.unsqueeze(0)      # [L] -> [1, L]
        #  y_res_A: [B, L]

        if maskA is not None:
            if maskA.ndim == 1:
                maskA = maskA.unsqueeze(0)      # [L] -> [1, L]
        #  maskA: [B, L] or None

        B = int(raw_logit.shape[0])

        for b in range(B):
            logit_b = raw_logit[b].detach().float().cpu()   # [L]
            lab_b   = y_res_A[b].float().cpu()              # [L]

            if maskA is not None:
                m = maskA[b].bool().cpu()
            else:
                m = torch.ones(logit_b.shape[0], dtype=torch.bool)

            L = int(m.sum().item())
            if L == 0:
                n_bad += 1
                continue

            logit_np = logit_b[m].numpy()
            lab_np   = lab_b[m].numpy()
            prob_np  = (1.0 / (1.0 + np.exp(-logit_np.astype(np.float64)))).astype(np.float32)

            has_pos = bool((lab_np > 0.5).any())
            if has_pos:
                n_haspos += 1
            else:
                n_allneg += 1

            # BCE
            _bce_val = float(np.mean(
                -lab_np * np.log(np.clip(prob_np, 1e-7, 1.0))
                - (1 - lab_np) * np.log(np.clip(1 - prob_np, 1e-7, 1.0))
            ))
            total_bce += _bce_val
            n_bce += 1
            if has_pos:
                total_bce_pos += _bce_val
                n_bce_pos += 1
            else:
                total_bce_neg += _bce_val
                n_bce_neg += 1

            all_prob.append(prob_np)
            all_lab.append(lab_np)
            per_prob.append(prob_np)
            per_lab.append(lab_np)
            per_len.append(L)
            n_used += 1

    empty_ret = {
        "bce": float("nan"),
        "ACC": 0.0, "Precision": 0.0, "Recall": 0.0, "F1": 0.0,
        "AUROC": 0.0, "AUPRC": float("nan"),
        "MCC": 0.0,
        "MedAUC": 0.0,   # kept for compatibility, always 0
        "thr": 0.5,
        "n_samples_used": 0, "n_auc": 0,
        "haspos": int(n_haspos), "allneg": int(n_allneg), "bad": int(n_bad),
        # Top-K placeholders
        "topk_abs": {}, "topk_rel": {},
        "topk_abs_posonly": {}, "topk_rel_posonly": {},
    }

    if len(all_prob) == 0:
        return empty_ret

    probs = np.concatenate(all_prob, axis=0).astype(np.float32)
    labs  = np.concatenate(all_lab,  axis=0).astype(np.float32)

    thr_used = float(thr) if thr is not None else _choose_thr(probs, labs)
    pred = (probs >= thr_used).astype(np.int32)
    lab  = (labs > 0.5).astype(np.int32)

    tp = int(((pred == 1) & (lab == 1)).sum())
    tn = int(((pred == 0) & (lab == 0)).sum())
    fp = int(((pred == 1) & (lab == 0)).sum())
    fn = int(((pred == 0) & (lab == 1)).sum())

    acc  = (tp + tn) / max(1, tp + tn + fp + fn)
    prec = tp / max(1, tp + fp)
    rec  = tp / max(1, tp + fn)
    f1   = 2.0 * prec * rec / max(1e-12, prec + rec)
    denom_mcc = math.sqrt(max(1e-12, (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc  = (tp * tn - fp * fn) / denom_mcc if denom_mcc > 0 else 0.0

    # AUROC / AUPRC
    auroc = 0.0
    auprc = float("nan")
    if have_sklearn and lab.max() > 0:
        try:
            auroc = float(roc_auc_score(lab, probs))
        except Exception:
            pass
        try:
            auprc = float(average_precision_score(lab, probs))
        except Exception:
            pass

    bce         = float(total_bce     / max(1, n_bce))
    bce_posonly = float(total_bce_pos / max(1, n_bce_pos))
    bce_allneg  = float(total_bce_neg / max(1, n_bce_neg))

    # ---- Top-K metrics (per-protein) ----
    def _topk_metrics(plist, llist, lenlist, ks_abs, ks_rel, posonly=False):
        """Compute precision/recall/hitrate @k for each protein, then average."""
        result_prec = {k: [] for k in list(ks_abs) + [f"L{r}" for r in ks_rel]}
        result_rec  = {k: [] for k in list(ks_abs) + [f"L{r}" for r in ks_rel]}
        result_hit  = {k: [] for k in list(ks_abs) + [f"L{r}" for r in ks_rel]}
        for p, l, L in zip(plist, llist, lenlist):
            has_p = bool((l > 0.5).any())
            if posonly and not has_p:
                continue
            order = np.argsort(-p)
            y = (l > 0.5).astype(np.int32)
            n_pos = int(y.sum())
            # absolute K
            for k in ks_abs:
                kk = min(k, L)
                top = order[:kk]
                tp_k = int(y[top].sum())
                result_prec[k].append(tp_k / max(1, kk))
                result_rec[k].append(tp_k / max(1, n_pos) if n_pos > 0 else 0.0)
                result_hit[k].append(float(tp_k > 0))
            # relative K = L / r
            for r in ks_rel:
                kk = max(1, int(math.ceil(L / r)))
                kk = min(kk, L)
                top = order[:kk]
                tp_k = int(y[top].sum())
                key = f"L{r}"
                result_prec[key].append(tp_k / max(1, kk))
                result_rec[key].append(tp_k / max(1, n_pos) if n_pos > 0 else 0.0)
                result_hit[key].append(float(tp_k > 0))
        out = {}
        for k in result_prec:
            n = len(result_prec[k])
            out[str(k)] = {
                "prec": float(np.mean(result_prec[k])) if n > 0 else 0.0,
                "rec":  float(np.mean(result_rec[k]))  if n > 0 else 0.0,
                "hit":  float(np.mean(result_hit[k]))  if n > 0 else 0.0,
                "n": n,
            }
        return out

    ks_abs = list(topk_ks)
    ks_rel = [5, 10, 20]   # L/5, L/10, L/20

    topk_all     = _topk_metrics(per_prob, per_lab, per_len, ks_abs, ks_rel, posonly=False)
    topk_posonly = _topk_metrics(per_prob, per_lab, per_len, ks_abs, ks_rel, posonly=True)

    return {
        "bce":       bce,
        "bce_posonly": bce_posonly,
        "bce_allneg":  bce_allneg,
        "ACC":       float(acc),
        "Precision": float(prec),
        "Recall":    float(rec),
        "F1":        float(f1),
        "AUROC":     float(auroc),
        "AUPRC":     float(auprc),
        "MCC":       float(mcc),
        "MedAUC":    0.0,   # compatibility placeholder; not computed
        "thr":       float(thr_used),
        "n_samples_used": int(n_used),
        "n_auc":          int(n_used),
        "haspos":    int(n_haspos),
        "allneg":    int(n_allneg),
        "bad":       int(n_bad),
        "topk_all":     topk_all,
        "topk_posonly": topk_posonly,
    }


# ---------------- Early stopper ----------------
class EarlyStopper:
    """Early stop on primary metric (EMA-smoothed). Works for AUPRC (binary) or TopK recall."""

    def __init__(self):
        self.best = None
        self.best_epoch = 0
        self.bad_epochs = 0
        self.ema = None
        self.patience = int(EARLY_PATIENCE)
        self.min_epochs = int(EARLY_MIN_EPOCHS)

    def step(self, ep: int, medauc: float):
        v = float(medauc)
        self.ema = v if self.ema is None else (EARLY_EMA * self.ema + (1.0 - EARLY_EMA) * v)

        improved = (self.best is None) or (self.ema > self.best + EARLY_MIN_DELTA)
        if improved:
            self.best = float(self.ema)
            self.best_epoch = int(ep)
            self.bad_epochs = 0
        else:
            if ep >= EARLY_MIN_EPOCHS:
                self.bad_epochs += 1

        stop = (ep >= EARLY_MIN_EPOCHS) and (self.bad_epochs >= EARLY_PATIENCE)
        info = {"medauc": v, "medauc_ema": float(self.ema), "best_ema": float(self.best),
                "bad_epochs": int(self.bad_epochs)}
        return improved, stop, info


@torch.no_grad()
def eval_l2_binary_medauc(
        model,
        dl,
        device: str = DEVICE,
        thr: Optional[float] = None,
        stride_frac: float = 0.5,
        max_pixels_per_complex: int = 200000,
        neg_pos_ratio: int = 5,
        thr_mode: str = "auto_mcc",
        thr_min: float = 0.001,
        thr_max: float = 0.5,
        thr_grid: int = 101,
):
    """
    L2 pixel binary eval (sampled for global metrics) + MedAUC (per-complex ROC-AUC median, full valid pixels).
    Returns: ACC/Precision/Recall/F1/AUPRC/MCC/BCE/MedAUC + counts.
    """
    model.eval()

    # sklearn optional
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
        have_sklearn = True
    except Exception:
        have_sklearn = False

    all_prob = []
    all_lab = []
    auc_list = []

    def _choose_thr(prob_np: np.ndarray, lab_np: np.ndarray):
        # prob_np, lab_np are 1D float arrays
        if prob_np.size == 0:
            return 0.5
        mode = str(thr_mode or "auto_mcc").lower()
        if mode == "fixed":
            return float(0.5 if thr is None else thr)

        tmin = float(thr_min)
        tmax = float(thr_max)
        if tmax <= tmin:
            tmin, tmax = 0.05, 0.95
        grid = int(max(11, thr_grid))
        ts = np.linspace(tmin, tmax, grid, dtype=np.float32)

        # precompute
        y = (lab_np > 0.5).astype(np.int32)
        best_t = float(ts[grid // 2])
        best_s = -1e9

        for t in ts:
            p = (prob_np >= float(t)).astype(np.int32)
            tp = int(((p == 1) & (y == 1)).sum())
            fp = int(((p == 1) & (y == 0)).sum())
            fn = int(((p == 0) & (y == 1)).sum())
            tn = int(((p == 0) & (y == 0)).sum())

            if mode == "auto_f1":
                denom = (2 * tp + fp + fn)
                s = (2 * tp / denom) if denom > 0 else 0.0
            elif mode == "auto_acc":
                denom = (tp + tn + fp + fn)
                s = ((tp + tn) / denom) if denom > 0 else 0.0
            else:
                # default: MCC
                denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
                s = ((tp * tn - fp * fn) / math.sqrt(denom)) if denom > 0 else 0.0

            if s > best_s:
                best_s = s
                best_t = float(t)

        return float(best_t)

    total_bce = 0.0
    total_bce_pos = 0.0   # BCE for proteins with positives
    total_bce_neg = 0.0   # BCE for all-negative proteins
    n_bce = 0
    n_bce_pos = 0
    n_bce_neg = 0
    n_used = 0
    n_bad = 0
    n_allneg = 0
    n_haspos = 0

    max_tokens = int(MAX_2D_TOKENS_EVAL)

    for batch in dl:
        logits2d, tb = infer_logits_fullmap_multicrop(
            model, batch, device=device, max_tokens=max_tokens, stride_frac=float(stride_frac)
        )
        if logits2d is None:
            n_bad += 1
            continue

        y2d, valid2d = _valid2d_and_y2d(tb)
        if y2d is None or valid2d is None or (not bool(valid2d.any())):
            n_bad += 1
            continue

        # logits2d: [B,L,M] on cpu
        if logits2d.ndim == 2:
            logits2d = logits2d.unsqueeze(0)

        B = int(logits2d.shape[0])

        for b in range(B):
            vb = valid2d[b].bool()
            if not bool(vb.any()):
                continue

            y = y2d[b][vb].detach().float().cpu()
            l = logits2d[b][vb].detach().float().cpu()
            if y.numel() == 0:
                continue

            has_pos = bool((y > 0.5).any().item())
            if has_pos:
                n_haspos += 1
            else:
                n_allneg += 1

            # ---- MedAUC per complex (full valid pixels) ----
            if have_sklearn:
                y_np = y.numpy()
                if y_np.max() != y_np.min():  # skip all-0/all-1
                    p_np = torch.sigmoid(l.float()).detach().cpu().numpy()
                    try:
                        auc_list.append(float(roc_auc_score(y_np, p_np)))
                    except Exception:
                        pass

            # ---- Global binary metrics (sample pixels) ----
            y_bin = (y > 0.5)
            pos_idx = y_bin.nonzero(as_tuple=False).view(-1)
            neg_idx = (~y_bin).nonzero(as_tuple=False).view(-1)

            if pos_idx.numel() > 0:
                max_neg = int(min(neg_idx.numel(), max(pos_idx.numel() * int(neg_pos_ratio),
                                                       int(max_pixels_per_complex) - pos_idx.numel())))
                if max_neg > 0 and neg_idx.numel() > max_neg:
                    perm = torch.randperm(neg_idx.numel())[:max_neg]
                    neg_keep = neg_idx[perm]
                else:
                    neg_keep = neg_idx
                sel = torch.cat([pos_idx, neg_keep], dim=0)
            else:
                # all-neg: cap pixels
                max_neg = int(min(neg_idx.numel(), int(max_pixels_per_complex)))
                if neg_idx.numel() > max_neg and max_neg > 0:
                    perm = torch.randperm(neg_idx.numel())[:max_neg]
                    sel = neg_idx[perm]
                else:
                    sel = neg_idx

            if sel.numel() == 0:
                continue

            # extra cap
            if sel.numel() > int(max_pixels_per_complex):
                perm = torch.randperm(sel.numel())[:int(max_pixels_per_complex)]
                sel = sel[perm]

            l_sel = l[sel]
            y_sel = y[sel].float()

            # BCE on sampled pixels
            l_sel = l_sel.float()
            y_sel = y_sel.float()
            total_bce += float(F.binary_cross_entropy_with_logits(l_sel, y_sel, reduction="mean").item())
            n_bce += 1

            p_sel = torch.sigmoid(l_sel).float().detach().cpu()
            all_prob.append(p_sel)
            all_lab.append((y_sel > 0.5).int())

            n_used += 1

    if len(all_prob) == 0:
        return {
            "bce": float("nan"),
            "ACC": 0.0, "Precision": 0.0, "Recall": 0.0, "F1": 0.0, "AUPRC": 0.0, "MCC": 0.0,
            "MedAUC": 0.0,
            "thr": float(thr if thr is not None else 0.5),
            "n_samples_used": 0, "n_auc": 0,
            "haspos": int(n_haspos), "allneg": int(n_allneg), "bad": int(n_bad),
        }

    probs = torch.cat(all_prob, dim=0).numpy()
    labs = torch.cat(all_lab, dim=0).numpy()

    thr_used = float(thr) if thr is not None else _choose_thr(probs.astype(np.float32), labs.astype(np.float32))
    pred = (probs >= thr_used).astype("int32")
    lab = labs.astype("int32")

    tp = int(((pred == 1) & (lab == 1)).sum())
    tn = int(((pred == 0) & (lab == 0)).sum())
    fp = int(((pred == 1) & (lab == 0)).sum())
    fn = int(((pred == 0) & (lab == 1)).sum())

    acc = (tp + tn) / max(1, (tp + tn + fp + fn))
    prec = tp / max(1, (tp + fp))
    rec = tp / max(1, (tp + fn))
    f1 = 2.0 * prec * rec / max(1e-12, (prec + rec))

    denom = math.sqrt(max(1e-12, (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = ((tp * tn - fp * fn) / denom) if denom > 0 else 0.0

    # AUPRC
    if have_sklearn:
        try:
            auprc = float(average_precision_score(lab, probs))
        except Exception:
            auprc = 0.0
    else:
        auprc = 0.0

    bce = float(total_bce / max(1, n_bce))

    # MedAUC: median over per-complex AUC list
    if len(auc_list) > 0:
        medauc = float(np.median(np.asarray(auc_list, dtype=np.float64)))
    else:
        medauc = 0.0

    return {
        "bce": bce,
        "ACC": float(acc),
        "Precision": float(prec),
        "Recall": float(rec),
        "F1": float(f1),
        "AUPRC": float(auprc),
        "MCC": float(mcc),
        "MedAUC": float(medauc),
        "thr": float(thr_used),
        "n_samples_used": int(n_used),
        "n_auc": int(len(auc_list)),
        "haspos": int(n_haspos),
        "allneg": int(n_allneg),
        "bad": int(n_bad),
    }


# ====================== Explainability dumps (Top-K pairs) ======================
def dump_explain_topk_from_logits(logits2d: torch.Tensor, batch: dict, out_json: str, k: int = 50):
    """Save Top-K (i,j,logit,prob,label) for the first sample in batch."""
    cid = batch.get("complex", "unknown")
    if logits2d is None:
        return
    if logits2d.ndim == 2:
        logits2d = logits2d.unsqueeze(0)
    y2d = batch.get("y2d", None)
    if y2d is not None and torch.is_tensor(y2d) and y2d.ndim == 2:
        y2d = y2d.unsqueeze(0)

    B, La, Lb = logits2d.shape[0], logits2d.shape[1], logits2d.shape[2]
    b = 0
    flat = logits2d[b].reshape(-1)
    # torch.topk on CPU does not support float16/bfloat16 reliably; cast to fp32 for explain dump
    if flat.dtype in (torch.float16, torch.bfloat16):
        flat = flat.float()
    k_eff = int(min(int(k), int(flat.numel())))
    if k_eff <= 0:
        return
    topv, topi = torch.topk(flat, k=k_eff, largest=True)
    pairs = []
    prob = torch.sigmoid(topv).detach().float().cpu().numpy().tolist()
    topv_cpu = topv.detach().float().cpu().numpy().tolist()
    topi_cpu = topi.detach().cpu().numpy().tolist()
    for t, idx in enumerate(topi_cpu):
        i = int(idx // Lb)
        j = int(idx % Lb)
        lab = None
        if y2d is not None and torch.is_tensor(y2d):
            try:
                lab = float((y2d[b, i, j] > 0.5).item())
            except Exception:
                lab = None
        pairs.append({"i": i, "j": j, "logit": float(topv_cpu[t]), "prob": float(prob[t]), "label": lab})

    payload = {
        "complex": cid,
        "shape": [int(La), int(Lb)],
        "topk": int(k_eff),
        "pairs": pairs,
    }
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def dump_val_explain(model, dl, out_dir: str, ep: int, k: int = 50, stride_frac: float = 0.5):
    """Run one validation batch (EMA weights if already applied) and dump Top-K pairs + optional gate info."""
    if k <= 0:
        return
    try:
        batch = next(iter(dl))
    except Exception:
        return
    try:
        logits2d, _ = infer_logits_fullmap_multicrop(
            model, batch, device=DEVICE, max_tokens=int(MAX_2D_TOKENS_EVAL), stride_frac=float(stride_frac)
        )
    except Exception as e:
        print(f"[explain] failed: {e}", flush=True)
        return
    cid = batch.get("complex", f"ep{ep}")
    safe_cid = cid[0] if isinstance(cid, (list, tuple)) else cid
    safe_cid = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(safe_cid))[:64]
    out_json = os.path.join(out_dir, f"ep{int(ep):03d}_{safe_cid}_topk{k}.json")
    dump_explain_topk_from_logits(logits2d.detach().cpu(), batch, out_json, k=int(k))

    # add gate info if model supports it (best-effort)
    try:
        if hasattr(model, "get_last_gate_info"):
            gate = model.get_last_gate_info()
            if gate is not None:
                with open(out_json, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                payload["gate"] = gate
                with open(out_json, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2)
    except Exception:
        pass

    print(f"[explain] saved {out_json}", flush=True)


def main():
    # ---- CRITICAL FIX: sampler globals used/updated in main() ----
    global _L2_SAMPLER_FLAGS, _L2_SAMPLER_HASPOS_RATE, _L2_SAMPLER_SAMPLER, _L2_SAMPLER_WEIGHTS
    set_seed(SEED)
    print(f"[run] device={DEVICE}", flush=True)
    print("### VERSION: v4_7_0_fix_topk_half")

    dl_tr, dl_va, d_res_site, d_chain_site = build_l2_loaders()
    print(f"[info] Feature dims: d_res={d_res_site} d_chain={d_chain_site}", flush=True)

    cfg = build_model_config(P, d_res_site, d_chain_site)
    _print_gpu_mem_snapshot("before_model_build")
    model = UnifiedInterfaceModel(cfg)
    try:
        model = model.to(DEVICE)
    except torch.cuda.OutOfMemoryError:
        _print_gpu_mem_snapshot("oom_during_model_to")
        print(
            "[oom] model.to(cuda) failed. This usually means the GPU is already occupied by "
            "other processes. Try one of: 1) free the GPU, 2) switch GPU via CUDA_VISIBLE_DEVICES, "
            "3) run with ESM_DEVICE=cpu to save current-process VRAM.",
            flush=True,
        )
        raise
    if not hasattr(model, 'cfg'):
        model.cfg = cfg

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # ---- AMP scaler ----
    scaler = GradScaler(enabled=USE_AMP) if (GradScaler is not None) else None
    # ---- LR schedule: warmup + cosine (step-level) ----

    # ---- EMA (evaluate/save with EMA weights by default) ----
    ema_decay = float(P.ema_decay)
    ema = EMA(model, decay=ema_decay)

    steps_per_epoch = len(dl_tr)
    total_updates = int(math.ceil((EPOCHS * steps_per_epoch) / float(max(1, ACCUM_STEPS))))
    sched, sched_info = build_warmup_cosine_scheduler(opt, total_updates=total_updates, base_lr=P.lr, P=P)

    print(f"[lr_sched] {sched_info}", flush=True)

    early = EarlyStopper()
    best_top50_raw = -1e9
    best_epoch = 0

    global_update = 0
    loss_ema = None  # for loss spike guard

    for ep in range(1, EPOCHS + 1):
        model.train()
        model.training_epoch = ep

        _s1 = int(getattr(P, 'samp_stage1_end', 10))
        _s2 = int(getattr(P, 'samp_stage2_end', 30))
        if (
                _L2_SAMPLER_FLAGS is not None and _L2_SAMPLER_SAMPLER is not None and _L2_SAMPLER_WEIGHTS is not None and ep in (
        _s1 + 1, _s2 + 1)):
            sp = get_sampling_params_for_epoch(ep, P)
            _rate_thr = float(getattr(P, 'samp_rate_thr', 0.01))
            f = float(_L2_SAMPLER_HASPOS_RATE) if _L2_SAMPLER_HASPOS_RATE > 0 else 1e-8
            p_tgt = float(sp["p_target"])
            r = (p_tgt * (1.0 - f)) / (max(f, 1e-8) * (1.0 - p_tgt))
            r = float(max(1.0, min(r, 50.0)))
            w_pos = float(sp["base_w_pos"]) * r * float(sp["pos_oversample"])
            w_neg = float(sp["base_w_neg"])
            new_w = _L2_SAMPLER_WEIGHTS.clone()
            new_w[torch.from_numpy(_L2_SAMPLER_FLAGS)] = w_pos
            new_w[~torch.from_numpy(_L2_SAMPLER_FLAGS)] = w_neg
            _L2_SAMPLER_SAMPLER.weights = new_w.double()
            _L2_SAMPLER_WEIGHTS = new_w
            print(f"[sampler][v2] ep={ep} stage update: w_pos={w_pos:.2f} w_neg={w_neg:.2f} p_target={p_tgt:.2f}",
                  flush=True)

        if hasattr(dl_tr.dataset, 'reset_esm_stats'):
            dl_tr.dataset.reset_esm_stats()
            print(f"[ESM-Cache] reset stats at epoch={ep}", flush=True)
        _rank_on = int(P.l2_rank_start_epoch)  <= ep
        _list_on = int(P.l2_list_start_epoch)  <= ep
        _cons_on = int(P.l12_cons_start_epoch) <= ep and float(P.l12_cons_w) > 0
        _2d_on   = float(P.l2_w) > 0
        _focal_w_now = float(getattr(P, 'l1_focal_w', 0.0))
        _asl_w_now   = float(getattr(P, 'l1_asl_w', 0.0))
        _l1rank_on   = float(getattr(P, 'l1_rank_w', 0.0)) > 0.0 and ep >= int(getattr(P, 'l1_rank_start_epoch', 999))
        _l1list_on   = float(getattr(P, 'l1_list_w', 0.0)) > 0.0 and ep >= int(getattr(P, 'l1_list_start_epoch', 999))
        print(
            f"[sched] ep={ep}"
            f"  L1_BCE(pw={float(P.l1_pos_weight):.0f},ls={float(getattr(P,'label_smoothing_l1',0)):.2f})"
            f"  focal={'ON(w='+str(round(_focal_w_now,2))+',g='+str(round(float(getattr(P,'l1_focal_gamma',2.0)),1))+')' if _focal_w_now>0 else 'off'}"
            f"  asl={'ON(w='+str(round(_asl_w_now,2))+')' if _asl_w_now>0 else 'off'}"
            f"  l1rank={'ON(w='+str(round(float(getattr(P,'l1_rank_w',0)),2))+',m='+str(round(float(getattr(P,'l1_rank_margin',0.5)),1))+')' if _l1rank_on else 'off'}"
            f"  l1list={'ON(w='+str(round(float(getattr(P,'l1_list_w',0)),2))+')' if _l1list_on else 'off'}"
            f"  rank={'ON' if _rank_on else 'off'}"
            f"  list={'ON' if _list_on else 'off'}"
            f"  cons={'ON' if _cons_on else 'off'}"
            f"  2d={'ON' if _2d_on else 'off'}",
            flush=True
        )

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        t0 = time.time()
        opt.zero_grad(set_to_none=True)

        bad_loss = 0
        did_backward_in_accum = False

        for step, batch in enumerate(dl_tr, start=1):
            with (autocast(enabled=USE_AMP, dtype=AMP_DTYPE) if autocast is not None else nullcontext()):

                out, loss, aux, _, _ = forward_one(model, batch, ep)

            if (not torch.is_tensor(loss)):
                print(f"[warn] unexpected loss type={type(loss)}; set loss=0", flush=True)
                loss = torch.tensor(0.0, device=DEVICE, requires_grad=True)

            # optional: loss spike guard (prevents rare explosions poisoning grads)
            try:
                spike_thr = float(getattr(P, 'loss_spike_threshold', 0.0))
            except Exception:
                spike_thr = 0.0
            if spike_thr and spike_thr > 0.0:
                lv = float(loss.detach().cpu())
                if loss_ema is None:
                    loss_ema = lv
                else:
                    d = float(getattr(P, 'loss_ema_decay', 0.95))
                    loss_ema = d * loss_ema + (1.0 - d) * lv
                if (loss_ema is not None) and (loss_ema > 0.0) and (lv > spike_thr * loss_ema):
                    scale = float(getattr(P, 'loss_spike_scale', 0.25))
                    loss = loss * scale
                    aux['loss_spike_scaled'] = 1
                    aux['loss_ema'] = float(loss_ema)
                    aux['loss_raw'] = float(lv)

            is_bad, why = _loss_is_bad(loss, thr=float(P.bad_loss_thr))
            if is_bad:
                bad_loss += 1
                print(
                    f"[warn] ep={ep} step={step} bad loss={float(loss.detach().cpu() if torch.is_tensor(loss) else 0):.4e} why={why} "
                    f"| 2d={float(aux.get('loss_2d', 0.0)):.3e} list={float(aux.get('loss_list', 0.0)):.3e} "
                    f"rank={float(aux.get('loss_rank', 0.0)):.3e} bce={float(aux.get('loss_bce', 0.0)):.3e} "
                    f"poly={float(aux.get('loss_poly', 0.0)):.3e}",
                    flush=True
                )
                continue

            if USE_AMP:
                scaler.scale(loss / float(max(1, ACCUM_STEPS))).backward()
            else:
                (loss / float(max(1, ACCUM_STEPS))).backward()
            did_backward_in_accum = True

            if (step % ACCUM_STEPS) == 0:
                if did_backward_in_accum:
                    with torch.no_grad():
                        for p in model.parameters():
                            if p.grad is not None:
                                p.grad = torch.nan_to_num(p.grad, nan=0.0, posinf=0.0, neginf=0.0)

                    if USE_AMP and scaler is not None:
                        scaler.unscale_(opt)
                    total_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)
                    aux["grad_norm"] = float(total_norm)

                    if torch.isnan(total_norm) or torch.isinf(total_norm):
                        print(f"[FATAL] Step {step}: Gradient is NaN/Inf! Skipping.", flush=True)
                        opt.zero_grad(set_to_none=True)
                    else:
                        if USE_AMP and scaler is not None:
                            scaler.step(opt)
                            scaler.update()
                        else:
                            opt.step()

                        global_update += 1
                        ema.update(model)
                        if sched is not None:
                            sched.step()

                opt.zero_grad(set_to_none=True)
                did_backward_in_accum = False

            if (step <= WARMUP_PRINT_STEPS) or (step % PRINT_EVERY == 0):
                peak = f"{torch.cuda.max_memory_reserved() / (1024 ** 3):.2f}GiB" if torch.cuda.is_available() else "cpu"
                cur_lr = float(opt.param_groups[0]["lr"])
                extra = format_aux_log(aux)
                extra_str = (" | " + extra) if extra else ""
                print(
                    f"[train] ep={ep} step={step}/{steps_per_epoch} "
                    f"loss={float(loss.item()):.4f} "
                    f"| 2d={float(aux.get('loss_2d', 0.0)):.3f} "
                    f"cons={float(aux.get('loss_cons', 0.0)):.3f} "
                    f"resA={float(aux.get('loss_resA', 0.0)):.3f} resB={float(aux.get('loss_resB', 0.0)):.3f} "
                    f"rank1={float(aux.get('loss_resA_rank', 0.0)):.3f} list1={float(aux.get('loss_resA_list', 0.0)):.3f} "
                    f"| keepPos={int(aux.get('ohem_keepPos', 0))} keepNeg={int(aux.get('ohem_keepNeg', 0))} keepPix={int(aux.get('ohem_keepPix', 0))} "
                    f"| alpha={float(aux.get('alpha', 0.0)):.3f}{extra_str} "
                    f"| lr={cur_lr:.2e} | peak={peak}",
                    flush=True
                )
        # -------- FLUSH last partial accumulation --------
        if did_backward_in_accum:
            with torch.no_grad():
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad = torch.nan_to_num(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
            if USE_AMP and scaler is not None:
                scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)
            if USE_AMP and scaler is not None:
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()
            global_update += 1
            ema.update(model)
            if sched is not None:
                sched.step()
            opt.zero_grad(set_to_none=True)

        dt = time.time() - t0
        print(f"[epoch] ep={ep} done | time={dt:.1f}s bad_loss={bad_loss} lr={float(opt.param_groups[0]['lr']):.2e}",
              flush=True)
        # =================== VALIDATION (EMA weights) ===================
        ema.apply_shadow(model)
        # ---- RBP296: use residue-level eval; original eval_l2_binary_medauc kept below (commented)
        val_bin = eval_rbp296_binary(
            model, dl_va, device=DEVICE,
            thr=(VAL_THR_FIXED if VAL_THR_MODE.lower() == 'fixed' else None),
            thr_mode=VAL_THR_MODE, thr_min=VAL_THR_MIN, thr_max=VAL_THR_MAX, thr_grid=VAL_THR_GRID,
        )
        # ---- Original DIPS eval (kept for reference, disabled for RBP296) ----
        # val_bin = eval_l2_binary_medauc(
        #     model, dl_va, device=DEVICE,
        #     thr=(VAL_THR_FIXED if VAL_THR_MODE.lower() == 'fixed' else None),
        #     thr_mode=VAL_THR_MODE, thr_min=VAL_THR_MIN, thr_max=VAL_THR_MAX, thr_grid=VAL_THR_GRID,
        #     stride_frac=0.5,
        # )

        # Optional: Top-K quick check (NOT used for early stop; off by default)
        val_topk = None
        if bool(P.eval_topk):
            val_topk = eval_l2_topk_quick(model, dl_va, device=DEVICE, thr_true=0.5, stride_frac=0.5)
        # ---- Explainability: dump Top-K pairs for 1 val complex (EMA weights) ----
        try:
            k_ex = int(P.explain_topk_k)
        except Exception:
            k_ex = 0
        if k_ex > 0:
            ex_dir = os.path.join(CKPT_DIR, "explain")
            os.makedirs(ex_dir, exist_ok=True)
            dump_val_explain(model, dl_va, ex_dir, ep=ep, k=k_ex, stride_frac=0.5)
        ema.restore(model)

        topk_str = ""
        if val_topk is not None:
            topk_str = (
                f"|| TopK: L/5={val_topk.get('L5', 0.0):.3f} "
                f"L/10={val_topk.get('L10', 0.0):.3f} "
                f"50={val_topk.get('50', 0.0):.3f}"
            )

        # ---- Top-K summary string (from RBP296 eval) ----
        topk_po  = val_bin.get("topk_posonly", {})
        topk_all = val_bin.get("topk_all", {})
        l5_po  = topk_po.get("L5",  {});  l10_po = topk_po.get("L10", {})
        l20_po = topk_po.get("L20", {});  k50_po = topk_po.get("50",  {})
        l5_al  = topk_all.get("L5",  {}); l10_al = topk_all.get("L10",{})

        #  primary metric star marker 
        _auprc_val = val_bin.get("AUPRC", 0.0)
        _l5_val    = float(l5_po.get("rec", 0.0))
        _l10_val   = float(l10_po.get("rec", 0.0))
        _topk_score = _topk_primary_score(val_bin)
        _is_best   = (PRIMARY_OBJ == "topk" and _topk_score == best_top50_raw) or \
                     (PRIMARY_OBJ != "topk" and _auprc_val == best_top50_raw)

        #  build compact log lines 
        used  = val_bin.get("n_samples_used", 0)
        hp    = val_bin.get("haspos", 0)
        an    = val_bin.get("allneg", 0)
        bad   = val_bin.get("bad", 0)

        _bce_pos = val_bin.get('bce_posonly', 0.0)
        _bce_neg = val_bin.get('bce_allneg', 0.0)
        line1 = (
            f"[val] ep={ep:>3d}  thr={val_bin.get('thr', 0.5):.3f}"
            f"  AUROC={val_bin.get('AUROC', 0.0):.4f}"
            f"  AUPRC={_auprc_val:.4f}"
            f"  MCC={val_bin.get('MCC', 0.0):.4f}"
            f"  BCE={val_bin.get('bce', 0.0):.4f}"
            f"(+{_bce_pos:.3f}/-{_bce_neg:.3f})"
            f"  [n={used} +{hp} -{an} skip={bad}]"
        )
        line2 = (
            f"       "
            f"  ACC={val_bin.get('ACC', 0.0):.4f}"
            f"  Prec={val_bin.get('Precision', 0.0):.4f}"
            f"  Rec={val_bin.get('Recall', 0.0):.4f}"
            f"  F1={val_bin.get('F1', 0.0):.4f}"
        )
        line3_parts = []
        if topk_po:
            line3_parts.append(
                f"  TopK(pos-only):"
                f"  L/5={l5_po.get('rec',0.0):.3f}"
                f"  L/10={l10_po.get('rec',0.0):.3f}"
                f"  L/20={l20_po.get('rec',0.0):.3f}"
                f"  K10p={topk_po.get('10',{}).get('prec',0.0):.3f}"
                f"  Hit20={topk_po.get('20',{}).get('hit',0.0):.3f}"
                f"  Score={_topk_score:.3f}"
            )
        if topk_all:
            line3_parts.append(
                f"  TopK(all):"
                f"  L/5={l5_al.get('rec',0.0):.3f}"
                f"  L/10={l10_al.get('rec',0.0):.3f}"
            )
        line3 = "       " + "".join(line3_parts) if line3_parts else ""

        sep = "  NEW BEST" if _is_best else ""
        print(line1 + sep, flush=True)
        print(line2, flush=True)
        if line3:
            print(line3, flush=True)

        # ---- primary score for best-ckpt selection ----
        if PRIMARY_OBJ == 'topk':
            _primary_score = _topk_score
        else:
            _primary_score = float(val_bin.get('AUPRC', 0.0))

        improved, should_stop, es_info = early.step(ep, _primary_score)
        if improved:
            best_epoch = ep
            best_top50_raw = _primary_score
            print(
                f"[best] ep={ep}  {PRIMARY_OBJ.upper()}={_primary_score:.4f}"
                f"  ema={es_info['medauc_ema']:.4f}"
                f"  -> {os.path.basename(BEST_CKPT)}",
                flush=True)
            # save EMA weights (recommended)
            ema.apply_shadow(model)
            save_ckpt(BEST_CKPT, model, opt=opt, medauc=val_bin, epoch=ep, scaler=scaler, cfg=cfg,
                      extra={"early": es_info, "val_topk": val_topk})
            ema.restore(model)

        # always save latest
        last_ckpt = os.path.join(CKPT_DIR, "last.pt")
        save_ckpt(last_ckpt, model, opt=opt, medauc=val_bin, epoch=ep, scaler=scaler, cfg=cfg,
                  extra={"early": es_info, "val_topk": val_topk})

        if should_stop:
            print(
                f"[early_stop] ep={ep}"
                f"  bad={es_info.get('bad_epochs',0)}/{early.patience}"
                f"  best_ep={best_epoch}"
                f"  best_{PRIMARY_OBJ.upper()}={es_info.get('best_ema',0.0):.4f}",
                flush=True)
            break

    print(f"[done] best_epoch={best_epoch} best_{PRIMARY_OBJ.upper()}={best_top50_raw:.4f} ckpt={BEST_CKPT}", flush=True)


if __name__ == "__main__":
    main()

