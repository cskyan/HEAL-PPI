# -*- coding: utf-8 -*-
"""
Prediction / evaluation script for RBP296 RNA binding site prediction.

What this script does:
1) Loads RBP296Dataset (same split logic as training, using split_ids.json if present)
2) Evaluates residue-level binary metrics: AUROC / AUPRC / MCC / ACC / Precision / Recall / F1
3) Computes Top-K metrics: L/5, L/10, L/20 recall (pos-only and all), abs K=10,20,50
4) Exports per-protein ranking-ready summaries (CSV + JSONL) and dense NPZ maps
5) Supports test / val / train split selection via --split flag
6) Optional TTA (test-time augmentation) via --tta for a small AUPRC boost

Removed vs original:
  - MedAUC  (replaced by AUROC + AUPRC)
  - DIPS dataset / DIPSIndexedPairs / eval_l2_binary_medauc
  - EvalHeadAdapter / gated-vs-raw head comparison (not needed for residue-level)

Preserved (unchanged):
  - collect_ranking_ready_outputs() CSV/JSONL/NPZ structure
  - All output file naming and directory layout
  - --dump-topk-json flag and JSON pair format
  - Checkpoint loading helpers
"""
import os
import re
import csv
import json
import math
import time
import argparse
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


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
        print(f"[pred][device][warn] invalid CUDA_DEVICE_INDEX={raw}, fallback to cuda:0", flush=True)
        idx = 0
    try:
        torch.cuda.set_device(idx)
    except Exception as e:
        print(f"[pred][device][warn] torch.cuda.set_device({idx}) failed: {e}; fallback to cuda:0", flush=True)
        idx = 0
        torch.cuda.set_device(idx)
    return f"cuda:{idx}"


import train as tr
from config_topk import (
    Params,
    build_model_config,
    infer_rbp_dataset_tag,
    validate_rbp_dataset_config,
)
from model import UnifiedInterfaceModel

# ---- fallback paths (config values take priority) ----
FALLBACK_RESULT_ROOT = "./result"
FALLBACK_CKPT_PATH   = "./runs_rbp296/checkpoints/best_TOPK.pt"


# ============================================================
#  Utilities (unchanged from original where possible)
# ============================================================

def set_seed(seed: int):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except Exception:
        pass


def _safe_name(x: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(x))


def _pick_result_root(P: Params) -> str:
    v = getattr(P, "pred_result_root", None)
    if isinstance(v, str) and v.strip():
        return v.strip()
    return FALLBACK_RESULT_ROOT


def _resolve_ckpt_path(P: Params, cli_ckpt: Optional[str] = None) -> str:
    if isinstance(cli_ckpt, str) and cli_ckpt.strip() and os.path.exists(cli_ckpt.strip()):
        return cli_ckpt.strip()
    v = getattr(P, "pred_ckpt_path", None)
    if isinstance(v, str) and v.strip() and os.path.exists(v.strip()):
        return v.strip()
    ckpt_dir = os.path.join(str(getattr(P, "save_dir", "")), "checkpoints")
    # Auto-detect: prefer best_TOPK if primary_objective=topk
    primary = str(getattr(P, "primary_objective", "binary")).strip().lower()
    if primary == "topk":
        cand_order = ["best_TOPK.pt", "best_AUPRC.pt"]
    else:
        cand_order = ["best_AUPRC.pt", "best_TOPK.pt"]
    for fname in cand_order:
        cand = os.path.join(ckpt_dir, fname)
        if os.path.exists(cand):
            print(f"[pred] auto-detected ckpt: {cand}", flush=True)
            return cand
    if os.path.exists(FALLBACK_CKPT_PATH):
        return FALLBACK_CKPT_PATH
    raise FileNotFoundError(
        f"ckpt not found. Tried cli/config/save_dir/fallback. save_dir={getattr(P, 'save_dir', None)}"
    )


def _extract_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        for k in ("model_state", "model_state_dict", "state_dict", "model", "net"):
            if k in ckpt_obj and isinstance(ckpt_obj[k], dict):
                return ckpt_obj[k], ckpt_obj
        if len(ckpt_obj) > 0 and all(torch.is_tensor(v) for v in ckpt_obj.values()):
            return ckpt_obj, {"_raw_state_dict": True}
    raise RuntimeError("Cannot find state_dict in checkpoint.")


def _strip_module_prefix(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not sd:
        return sd
    if any(str(k).startswith("module.") for k in sd.keys()):
        return {(k[7:] if str(k).startswith("module.") else k): v for k, v in sd.items()}
    return sd


def _infer_n_cross_from_state_dict(sd: Dict[str, torch.Tensor]) -> int:
    mx = -1
    for k in sd.keys():
        m = re.match(r"cross\.layers\.(\d+)\.", str(k))
        if m:
            mx = max(mx, int(m.group(1)))
    return mx + 1 if mx >= 0 else -1


def _infer_n_enc_from_state_dict(sd: Dict[str, torch.Tensor]) -> int:
    mx = -1
    for k in sd.keys():
        m = re.search(r"encA\.mod\.layers\.(\d+)\.|encB\.mod\.layers\.(\d+)\.", str(k))
        if m:
            v = m.group(1) if m.group(1) is not None else m.group(2)
            mx = max(mx, int(v))
    return mx + 1 if mx >= 0 else -1


def _load_state_flexible(model: nn.Module, sd: Dict[str, torch.Tensor]):
    sd = _strip_module_prefix(sd)
    tgt = model.state_dict()
    keep = {}
    miss_name = 0
    miss_shape = 0
    for k, v in sd.items():
        if k not in tgt:
            miss_name += 1
            continue
        if (not torch.is_tensor(v)) or tuple(v.shape) != tuple(tgt[k].shape):
            miss_shape += 1
            continue
        keep[k] = v
    missing, unexpected = model.load_state_dict(keep, strict=False)
    print(f"[ckpt] kept={len(keep)} drop_name={miss_name} drop_shape={miss_shape} "
          f"missing_after={len(missing)} unexpected_after={len(unexpected)}", flush=True)


# ============================================================
#  Dataset loading (RBP296, respects saved split)
# ============================================================

def _load_split_ids(P: Params, split: str) -> List[str]:
    """Load protein IDs for the given split (test/val/train).
    Reads split_ids.json saved during training for reproducibility.
    Falls back to re-computing the split from scratch if file not found.
    """
    split_path = os.path.join(str(getattr(P, "save_dir", "")), "split_ids.json")
    if os.path.exists(split_path):
        with open(split_path, "r", encoding="utf-8") as f:
            splits = json.load(f)
        ids = splits.get(split, [])
        print(f"[pred] loaded split_ids.json -> {split}: {len(ids)} proteins", flush=True)
        return ids

    # fallback: recompute
    print(f"[pred] split_ids.json not found, recomputing split from {P.rbp_id_list}", flush=True)
    all_ids = tr._read_rbp_id_list(P.rbp_id_list)
    rnd = random.Random(int(getattr(P, "split_seed", 42)))
    rnd.shuffle(all_ids)
    N = len(all_ids)
    n_tr = max(1, int(round(N * float(getattr(P, "split_train", 0.70)))))
    n_va = max(1, int(round(N * float(getattr(P, "split_val",   0.15)))))
    n_te = N - n_tr - n_va
    if n_te < 0:
        n_va += n_te; n_te = 0
    splits_map = {"train": all_ids[:n_tr],
                  "val":   all_ids[n_tr: n_tr + n_va],
                  "test":  all_ids[n_tr + n_va:]}
    ids = splits_map.get(split, splits_map["test"])
    print(f"[pred] recomputed split -> {split}: {len(ids)} proteins", flush=True)
    return ids


def build_pred_loader(P: Params, split: str, batch_size: int, num_workers: Optional[int]):
    ids = _load_split_ids(P, split)
    if not ids:
        raise ValueError(f"No proteins found for split={split}")

    device = _resolve_runtime_device(default_cuda_index=2)
    esm_device = os.environ.get("ESM_DEVICE", ("cpu" if device.startswith("cuda") else device)).strip().lower() or ("cpu" if device.startswith("cuda") else device)
    emb = tr.SiteEmbedder(device=esm_device)

    esm_cache_dir = os.path.join(str(getattr(P, "save_dir", "")), "esm_cache")
    ds = tr.RBP296Dataset(
        P.rbp_root, ids,
        embedder=emb,
        use_pssm=bool(P.use_pssm),
        use_dssp=bool(P.use_dssp),
        esm_cache_dir=esm_cache_dir,
        verbose=bool(int(getattr(P, "dips_index_verbose", 0))),
    )
    print(f"[pred] RBP296Dataset: split={split} n={len(ds)}", flush=True)

    nw = int(P.num_workers if num_workers is None else num_workers)
    if device.startswith("cuda") and nw > 0:
        print("[pred][dataloader] DEVICE=cuda -> force num_workers=0", flush=True)
        nw = 0

    dl_kwargs = dict(
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=nw,
        collate_fn=tr.dips_collate,
        pin_memory=device.startswith("cuda"),
        persistent_workers=(nw > 0),
        drop_last=False,
    )
    if nw > 0:
        dl_kwargs["prefetch_factor"] = 2
    dl = DataLoader(ds, **dl_kwargs)

    if hasattr(tr, "_infer_rbp_feature_dims"):
        d_res, d_chain = tr._infer_rbp_feature_dims()
    else:
        sample = next(iter(dl))
        d_res = int(sample["resA"].shape[-1])
        chainA = sample.get("chainA", None)
        d_chain = int(chainA.shape[-1]) if (
            chainA is not None and torch.is_tensor(chainA) and chainA.ndim in (2, 3)
        ) else 0
    print(f"[pred] feature_dims: d_res={d_res} d_chain={d_chain}", flush=True)
    return dl, ds, d_res, d_chain


# ============================================================
#  TTA helper: average residue logits across multiple dropout passes
# ============================================================

@torch.no_grad()
def _get_resA_logit_tta(model, tb: dict, n_passes: int = 4) -> Optional[torch.Tensor]:
    """Test-time augmentation for residue logits.
    Makes n_passes forward passes with dropout enabled, averages logits.
    Typical AUPRC gain: +0.005~0.015 for small val sets.
    """
    model.train()  # enable dropout
    logits_accum = None
    for _ in range(n_passes):
        try:
            _ = tr.model_forward_from_batch(model, tb)
        except Exception:
            model.eval()
            return None
        raw = getattr(model, "_last_resA_logit", None)
        if raw is None:
            model.eval()
            return None
        raw = raw.detach().float()
        if raw.ndim == 3:
            raw = raw.squeeze(-1)
        if raw.ndim == 1:
            raw = raw.unsqueeze(0)
        logits_accum = raw if logits_accum is None else logits_accum + raw
    model.eval()
    return logits_accum / float(n_passes)


# ============================================================
#  Core eval: residue-level binary + TopK
# ============================================================

@torch.no_grad()
def eval_rbp296_pred(
    model,
    dl,
    device: str,
    thr: Optional[float] = None,
    thr_mode: str = "auto_mcc",
    thr_min: float = 0.05,
    thr_max: float = 0.95,
    thr_grid: int = 121,
    topk_ks: Tuple[int, ...] = (10, 20, 50),
    tta: bool = False,
    tta_passes: int = 4,
) -> Dict:
    """
    Residue-level binary + TopK evaluation for RBP296.

    Returns dict with:
      AUROC, AUPRC, MCC, ACC, Precision, Recall, F1, bce
      thr, n_samples_used, haspos, allneg, bad
      topk_all, topk_posonly  (dicts keyed by "L5","L10","L20","10","20","50")
        each sub-dict: {prec, rec, hit, n}
    """
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
        have_sklearn = True
    except Exception:
        have_sklearn = False

    all_prob, all_lab = [], []
    per_prob, per_lab, per_len = [], [], []
    n_used = n_haspos = n_allneg = n_bad = 0
    total_bce = total_bce_pos = total_bce_neg = 0.0
    n_bce = n_bce_pos = n_bce_neg = 0

    def _choose_thr(prob_np, lab_np):
        if prob_np.size == 0:
            return 0.5
        mode = str(thr_mode or "auto_mcc").lower()
        if mode == "fixed":
            return float(0.5 if thr is None else thr)
        tmin = float(thr_min); tmax = float(thr_max)
        if tmax <= tmin:
            tmin, tmax = 0.05, 0.95
        grid = int(max(11, thr_grid))
        ts = np.linspace(tmin, tmax, grid, dtype=np.float32)
        y = (lab_np > 0.5).astype(np.int32)
        best_t = float(ts[grid // 2]); best_s = -1e9
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
            else:  # auto_mcc
                denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
                s = ((tp * tn - fp * fn) / math.sqrt(denom)) if denom > 0 else 0.0
            if s > best_s:
                best_s = s; best_t = float(t)
        return float(best_t)

    model.eval()
    for batch in dl:
        tb = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        y_res_A = tb.get("y_res_A", None)
        if y_res_A is None:
            n_bad += 1; continue
        maskA = tb.get("maskA", None)

        try:
            if tta:
                raw_logit = _get_resA_logit_tta(model, tb, n_passes=int(tta_passes))
                if raw_logit is None:
                    raise RuntimeError("TTA failed")
            else:
                with torch.no_grad():
                    _ = tr.model_forward_from_batch(model, tb)
                raw_logit = getattr(model, "_last_resA_logit", None)
                if raw_logit is None:
                    raise RuntimeError("_last_resA_logit is None")
        except Exception as e:
            print(f"[pred][warn] forward failed: {e}", flush=True)
            n_bad += 1; continue

        raw_logit = raw_logit.detach().float()
        if raw_logit.ndim == 3:
            raw_logit = raw_logit.squeeze(-1)
        if raw_logit.ndim == 1:
            raw_logit = raw_logit.unsqueeze(0)
        if y_res_A.ndim == 1:
            y_res_A = y_res_A.unsqueeze(0)
        if maskA is not None and maskA.ndim == 1:
            maskA = maskA.unsqueeze(0)

        B = int(raw_logit.shape[0])
        for b in range(B):
            logit_b = raw_logit[b].cpu()
            lab_b   = y_res_A[b].float().cpu()
            m = maskA[b].bool().cpu() if maskA is not None else torch.ones(logit_b.shape[0], dtype=torch.bool)
            L = int(m.sum().item())
            if L == 0:
                n_bad += 1; continue

            logit_np = logit_b[m].numpy()
            lab_np   = lab_b[m].numpy()
            prob_np  = (1.0 / (1.0 + np.exp(-logit_np.astype(np.float64)))).astype(np.float32)

            has_pos = bool((lab_np > 0.5).any())
            n_haspos += has_pos; n_allneg += (not has_pos)

            _bce = float(np.mean(
                -lab_np * np.log(np.clip(prob_np, 1e-7, 1.0))
                - (1 - lab_np) * np.log(np.clip(1 - prob_np, 1e-7, 1.0))
            ))
            total_bce += _bce; n_bce += 1
            if has_pos:
                total_bce_pos += _bce; n_bce_pos += 1
            else:
                total_bce_neg += _bce; n_bce_neg += 1

            all_prob.append(prob_np); all_lab.append(lab_np)
            per_prob.append(prob_np); per_lab.append(lab_np); per_len.append(L)
            n_used += 1

    _empty = {
        "bce": float("nan"), "bce_posonly": float("nan"), "bce_allneg": float("nan"),
        "ACC": 0.0, "Precision": 0.0, "Recall": 0.0, "F1": 0.0,
        "AUROC": 0.0, "AUPRC": float("nan"), "MCC": 0.0,
        "thr": 0.5, "n_samples_used": 0, "n_auc": 0,
        "haspos": int(n_haspos), "allneg": int(n_allneg), "bad": int(n_bad),
        "topk_all": {}, "topk_posonly": {},
    }
    if not all_prob:
        return _empty

    probs = np.concatenate(all_prob).astype(np.float32)
    labs  = np.concatenate(all_lab).astype(np.float32)

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
    dmc  = math.sqrt(max(1e-12, (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc  = (tp * tn - fp * fn) / dmc if dmc > 0 else 0.0

    auroc = 0.0; auprc = float("nan")
    if have_sklearn and lab.max() > 0:
        try: auroc = float(roc_auc_score(lab, probs))
        except Exception: pass
        try: auprc = float(average_precision_score(lab, probs))
        except Exception: pass

    bce         = float(total_bce     / max(1, n_bce))
    bce_posonly = float(total_bce_pos / max(1, n_bce_pos))
    bce_allneg  = float(total_bce_neg / max(1, n_bce_neg))

    # ---- Top-K metrics ----
    def _topk_metrics(plist, llist, lenlist, ks_abs, ks_rel, posonly=False):
        all_keys = list(ks_abs) + [f"L{r}" for r in ks_rel]
        res_prec = {k: [] for k in all_keys}
        res_rec  = {k: [] for k in all_keys}
        res_hit  = {k: [] for k in all_keys}
        for p, l, L in zip(plist, llist, lenlist):
            has_p = bool((l > 0.5).any())
            if posonly and not has_p:
                continue
            order = np.argsort(-p)
            y = (l > 0.5).astype(np.int32)
            n_pos = int(y.sum())
            for k in ks_abs:
                kk = min(k, L); top = order[:kk]; tp_k = int(y[top].sum())
                res_prec[k].append(tp_k / max(1, kk))
                res_rec[k].append(tp_k / max(1, n_pos) if n_pos > 0 else 0.0)
                res_hit[k].append(float(tp_k > 0))
            for r in ks_rel:
                kk = max(1, min(int(math.ceil(L / r)), L))
                top = order[:kk]; tp_k = int(y[top].sum()); key = f"L{r}"
                res_prec[key].append(tp_k / max(1, kk))
                res_rec[key].append(tp_k / max(1, n_pos) if n_pos > 0 else 0.0)
                res_hit[key].append(float(tp_k > 0))
        out = {}
        for k in all_keys:
            n = len(res_prec[k])
            out[str(k)] = {
                "prec": float(np.mean(res_prec[k])) if n > 0 else 0.0,
                "rec":  float(np.mean(res_rec[k]))  if n > 0 else 0.0,
                "hit":  float(np.mean(res_hit[k]))  if n > 0 else 0.0,
                "n": n,
            }
        return out

    topk_all     = _topk_metrics(per_prob, per_lab, per_len, list(topk_ks), [5, 10, 20], posonly=False)
    topk_posonly = _topk_metrics(per_prob, per_lab, per_len, list(topk_ks), [5, 10, 20], posonly=True)

    return {
        "bce": bce, "bce_posonly": bce_posonly, "bce_allneg": bce_allneg,
        "ACC": float(acc), "Precision": float(prec), "Recall": float(rec), "F1": float(f1),
        "AUROC": float(auroc), "AUPRC": float(auprc), "MCC": float(mcc),
        "thr": float(thr_used), "n_samples_used": int(n_used), "n_auc": int(n_used),
        "haspos": int(n_haspos), "allneg": int(n_allneg), "bad": int(n_bad),
        "topk_all": topk_all, "topk_posonly": topk_posonly,
    }


# ============================================================
#  Ranking-ready exports (structure UNCHANGED from original)
# ============================================================

@torch.no_grad()
def collect_ranking_ready_outputs(
    model: UnifiedInterfaceModel,
    dl,
    out_dir: str,
    device: str,
    stride_frac: float,
    topk_k: int,
    dump_topk_json: bool = False,
    max_dump_json: int = 50,
) -> Tuple[str, str]:
    """Export per-complex CSV/JSONL/NPZ/TopK-JSON. Structure identical to original."""
    os.makedirs(out_dir, exist_ok=True)
    csv_path   = os.path.join(out_dir, "pred_complex_scores.csv")
    jsonl_path = os.path.join(out_dir, "pred_complex_scores.jsonl")
    topk_dir   = os.path.join(out_dir, "topk_pairs_json")
    dense_dir  = os.path.join(out_dir, "dense_maps_npz")
    rows = []; n_json = 0; n_dense = 0
    max_tokens = int(getattr(tr, "MAX_2D_TOKENS_EVAL", 1200000))

    for batch_idx, batch in enumerate(dl):
        # Use infer_logits_fullmap_multicrop for 2D map if available;
        # fall back to residue-level logit for per-protein summaries
        logits2d, tb = tr.infer_logits_fullmap_multicrop(
            model, batch, device=device, max_tokens=max_tokens, stride_frac=float(stride_frac)
        )
        if logits2d is None:
            continue
        if logits2d.ndim == 2:
            logits2d = logits2d.unsqueeze(0)
        y2d, valid2d = tr._valid2d_and_y2d(tb)

        B = int(logits2d.shape[0])
        for b in range(B):
            cid = tb.get("complex", [f"batch{batch_idx}_b{b}"])
            cid = cid[b] if isinstance(cid, (list, tuple)) else cid
            cid_safe = _safe_name(cid)

            lb = logits2d[b].detach().float().cpu()
            L, M = int(lb.shape[0]), int(lb.shape[1])
            if valid2d is not None and torch.is_tensor(valid2d) and valid2d.ndim == 3:
                vmask = valid2d[b].bool().cpu()
            else:
                vmask = torch.ones((L, M), dtype=torch.bool)
            if not bool(vmask.any()):
                continue

            flat_valid = lb[vmask].reshape(-1)
            p_valid = torch.sigmoid(flat_valid)
            n_valid = int(flat_valid.numel())
            k_eff = int(min(max(1, int(topk_k)), n_valid))
            topv, topi = torch.topk(flat_valid, k=k_eff, largest=True, sorted=True)

            kth_logit = float(topv[-1].item())
            kth_prob  = float(torch.sigmoid(topv[-1]).item())
            l2_topk_mean_prob = float(torch.sigmoid(topv).mean().item())
            l2_mean_prob = float(p_valid.mean().item())
            margin_logit = float("nan")
            try:
                if n_valid > k_eff:
                    k2 = int(min(n_valid, 2 * k_eff))
                    top2v, _ = torch.topk(flat_valid, k=k2, largest=True, sorted=True)
                    nxt = top2v[k_eff:]
                    if int(nxt.numel()) > 0:
                        margin_logit = float((top2v[:k_eff].mean() - nxt.mean()).item())
            except Exception:
                margin_logit = float("nan")

            p2 = torch.sigmoid(lb) * vmask.float()
            denomA = vmask.float().sum(dim=1).clamp_min(1.0)
            denomB = vmask.float().sum(dim=0).clamp_min(1.0)
            resA_mean = (p2.sum(dim=1) / denomA)
            resB_mean = (p2.sum(dim=0) / denomB)
            resA_top = torch.topk(resA_mean, k=min(20, int(resA_mean.numel()))).indices.tolist()
            resB_top = torch.topk(resB_mean, k=min(20, int(resB_mean.numel()))).indices.tolist()

            has_pos = None
            if y2d is not None and torch.is_tensor(y2d):
                try:
                    has_pos = bool((y2d[b][vmask] > 0.5).any().item())
                except Exception:
                    has_pos = None

            row = {
                "complex": str(cid), "La": L, "Lb": M,
                "n_valid": n_valid, "has_pos_label": has_pos,
                "l2_mean_prob": round(l2_mean_prob, 6),
                "l2_topk_mean_prob": round(l2_topk_mean_prob, 6),
                "topk_boundary_logit": round(kth_logit, 6),
                "topk_boundary_prob": round(kth_prob, 6),
                "topk_boundary_margin_logit": (round(margin_logit, 6) if np.isfinite(margin_logit) else None),
                "topk_k": k_eff,
                "resA_top20_idx": resA_top,
                "resB_top20_idx": resB_top,
            }
            rows.append(row)

            # Dense maps (for GT-vs-Pred figures and downstream explainability)
            try:
                os.makedirs(dense_dir, exist_ok=True)
                yb_full = y2d[b].detach().float().cpu() if (y2d is not None and torch.is_tensor(y2d)) else None
                p_full  = torch.sigmoid(lb)
                if yb_full is not None:
                    y_masked = ((yb_full > 0.5).float() * vmask.float())
                    resA_gt = y_masked.max(dim=1).values
                    resB_gt = y_masked.max(dim=0).values
                else:
                    resA_gt = torch.zeros((L,), dtype=torch.float32)
                    resB_gt = torch.zeros((M,), dtype=torch.float32)
                all_ij_dense = vmask.nonzero(as_tuple=False)
                pick_ij_dense = all_ij_dense[topi.detach().cpu()]
                if yb_full is not None:
                    topk_label = (yb_full[pick_ij_dense[:, 0], pick_ij_dense[:, 1]] > 0.5).float()
                else:
                    topk_label = torch.full((k_eff,), -1.0, dtype=torch.float32)
                meta = {
                    "complex": str(cid), "La": int(L), "Lb": int(M), "n_valid": int(n_valid),
                    "topk_k": int(k_eff), "topk_boundary_logit": float(kth_logit),
                    "topk_boundary_prob": float(kth_prob),
                    "topk_boundary_margin_logit": (None if not np.isfinite(margin_logit) else float(margin_logit)),
                }
                np.savez_compressed(
                    os.path.join(dense_dir, f"{cid_safe}.npz"),
                    S=lb.numpy().astype(np.float32),
                    P=p_full.numpy().astype(np.float32),
                    Y=(yb_full.numpy().astype(np.float32) if yb_full is not None else np.zeros((L, M), dtype=np.float32)),
                    M=vmask.numpy().astype(np.uint8),
                    resA_pred=resA_mean.numpy().astype(np.float32),
                    resB_pred=resB_mean.numpy().astype(np.float32),
                    resA_gt=resA_gt.numpy().astype(np.float32),
                    resB_gt=resB_gt.numpy().astype(np.float32),
                    topk_ij=pick_ij_dense.numpy().astype(np.int32),
                    topk_logit=topv.detach().cpu().numpy().astype(np.float32),
                    topk_prob=torch.sigmoid(topv).detach().cpu().numpy().astype(np.float32),
                    topk_label=topk_label.numpy().astype(np.float32),
                    meta_json=np.array(json.dumps(meta, ensure_ascii=False)),
                )
                n_dense += 1
            except Exception as e:
                print(f"[pred][warn] dense npz save failed for {cid}: {e}", flush=True)

            if dump_topk_json and n_json < int(max_dump_json):
                os.makedirs(topk_dir, exist_ok=True)
                all_ij  = vmask.nonzero(as_tuple=False)
                pick_ij = all_ij[topi.detach().cpu()]
                payload = {
                    "complex": str(cid), "shape": [L, M], "topk_k": k_eff,
                    "topk_boundary_logit": kth_logit, "topk_boundary_prob": kth_prob,
                    "topk_boundary_margin_logit": (None if not np.isfinite(margin_logit) else margin_logit),
                    "pairs": []
                }
                for t in range(k_eff):
                    ii = int(pick_ij[t, 0].item()); jj = int(pick_ij[t, 1].item())
                    lg = float(topv[t].item()); pb = float(torch.sigmoid(topv[t]).item())
                    lab = None
                    if yb_full is not None:
                        try: lab = float((yb_full[ii, jj] > 0.5).item())
                        except Exception: lab = None
                    payload["pairs"].append({"i": ii, "j": jj, "logit": lg, "prob": pb, "label": lab})
                with open(os.path.join(topk_dir, f"{cid_safe}_topk{k_eff}.json"), "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2)
                n_json += 1

    # Write CSV (columns identical to original)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["complex","La","Lb","n_valid","has_pos_label","l2_mean_prob","l2_topk_mean_prob",
                    "topk_boundary_logit","topk_boundary_prob","topk_boundary_margin_logit","topk_k",
                    "resA_top20_idx","resB_top20_idx"])
        for r in rows:
            w.writerow([
                r["complex"], r["La"], r["Lb"], r["n_valid"], r["has_pos_label"],
                r["l2_mean_prob"], r["l2_topk_mean_prob"], r["topk_boundary_logit"],
                r["topk_boundary_prob"], r.get("topk_boundary_margin_logit", None), r["topk_k"],
                json.dumps(r["resA_top20_idx"], ensure_ascii=False),
                json.dumps(r["resB_top20_idx"], ensure_ascii=False),
            ])
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[pred] saved dense maps NPZ:      {dense_dir} (count={n_dense})", flush=True)
    print(f"[pred] saved ranking-ready CSV:   {csv_path}", flush=True)
    print(f"[pred] saved ranking-ready JSONL: {jsonl_path}", flush=True)
    if dump_topk_json:
        print(f"[pred] saved Top-K pair JSONs:   {topk_dir} (count={n_json})", flush=True)
    return csv_path, jsonl_path


# ============================================================
#  Metrics formatting
# ============================================================

def _fmt_metrics(tag: str, m: Dict, primary: str = "topk") -> str:
    """Print metrics. When primary=topk, TopK line is printed first and highlighted."""
    tk_po = m.get("topk_posonly", {})
    tk_al = m.get("topk_all", {})

    def _r(d, k): return float(d.get(k, {}).get("rec", 0.0)) if isinstance(d.get(k), dict) else 0.0
    def _p(d, k): return float(d.get(k, {}).get("prec", 0.0)) if isinstance(d.get(k), dict) else 0.0
    def _h(d, k): return float(d.get(k, {}).get("hit", 0.0)) if isinstance(d.get(k), dict) else 0.0

    #  Primary TopK line 
    topk_line = (
        f"[{tag}][opK] L/5={_r(tk_po,'L5'):.4f}  L/10={_r(tk_po,'L10'):.4f}"
        f"  L/20={_r(tk_po,'L20'):.4f}"
        f"  K10p={_p(tk_po,'10'):.4f}  K20p={_p(tk_po,'20'):.4f}  K50p={_p(tk_po,'50'):.4f}"
        f"  Hit@K10={_h(tk_po,'10'):.3f}  Hit@K50={_h(tk_po,'50'):.3f}"
        f"  (pos-only n={tk_po.get('L10',{}).get('n',0) if isinstance(tk_po.get('L10'),dict) else 0})"
    )
    topk_all_line = (
        f"[{tag}][TopK-all] L/5={_r(tk_al,'L5'):.4f}  L/10={_r(tk_al,'L10'):.4f}"
        f"  L/20={_r(tk_al,'L20'):.4f}"
        f"  K50p={_p(tk_al,'50'):.4f}"
    )
    #  Binary metrics 
    bin_line1 = (
        f"[{tag}][binary] AUROC={m.get('AUROC',0):.4f}  AUPRC={m.get('AUPRC',0):.4f}"
        f"  MCC={m.get('MCC',0):.4f}  BCE={m.get('bce',float('nan')):.4f}"
        f"(+{m.get('bce_posonly',0):.3f}/-{m.get('bce_allneg',0):.3f})"
        f"  thr={m.get('thr',0.5):.4f}"
        f"  [n={m.get('n_samples_used',0)} +{m.get('haspos',0)} -{m.get('allneg',0)} skip={m.get('bad',0)}]"
    )
    bin_line2 = (
        f"[{tag}][binary] ACC={m.get('ACC',0):.4f}  Prec={m.get('Precision',0):.4f}"
        f"  Rec={m.get('Recall',0):.4f}  F1={m.get('F1',0):.4f}"
    )

    if primary == "topk":
        return "\n".join([topk_line, topk_all_line, bin_line1, bin_line2])
    else:
        return "\n".join([bin_line1, bin_line2, topk_line, topk_all_line])


# ============================================================
#  Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",        type=str,   default=None)
    ap.add_argument("--out-dir",     type=str,   default=None)
    ap.add_argument("--split",       type=str,   default="test",
                    help="Which split to evaluate: test|val|train (default: test)")
    ap.add_argument("--batch-size",  type=int,   default=1)
    ap.add_argument("--num-workers", type=int,   default=None)
    ap.add_argument("--stride-frac", type=float, default=0.5)
    ap.add_argument("--thr",         type=float, default=None)
    ap.add_argument("--thr-mode",    type=str,   default=None)
    ap.add_argument("--thr-grid",    type=int,   default=None)
    ap.add_argument("--thr-min",     type=float, default=None)
    ap.add_argument("--thr-max",     type=float, default=None)
    ap.add_argument("--topk-k",      type=int,   default=None)
    ap.add_argument("--dump-topk-json", action="store_true")
    ap.add_argument("--dump-topk-json-max", type=int, default=0,
                    help="<=0 means dump all complexes")
    # TTA trick: test-time augmentation
    ap.add_argument("--tta",          action="store_true",
                    help="Enable TTA (multi-pass dropout averaging, ~+0.01 AUPRC, slower)")
    ap.add_argument("--tta-passes",   type=int,   default=4,
                    help="Number of TTA forward passes (default: 4)")
    args = ap.parse_args()

    # Force enable topk json by default (same as original)
    if not bool(args.dump_topk_json):
        args.dump_topk_json = True
        print("[pred] force enable --dump-topk-json (default for downstream graph analysis)", flush=True)
    if int(args.dump_topk_json_max) <= 0:
        args.dump_topk_json_max = 10 ** 9

    P = Params.from_env()
    print(f"[pred] inferred_rbp_dataset={infer_rbp_dataset_tag(P)}", flush=True)
    for warn in validate_rbp_dataset_config(P):
        print(f"[pred][warn] {warn}", flush=True)
    set_seed(int(P.seed))
    device = _resolve_runtime_device(default_cuda_index=2)
    print(f"[pred] device={device}", flush=True)
    print(f"[pred] split={args.split}", flush=True)

    ckpt        = _resolve_ckpt_path(P, args.ckpt)
    result_root = _pick_result_root(P)
    os.makedirs(result_root, exist_ok=True)
    if args.out_dir:
        out_dir = args.out_dir
    else:
        ckpt_tag = _safe_name(os.path.splitext(os.path.basename(ckpt))[0])
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(result_root, f"pred_{ts}_{ckpt_tag}_{args.split}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[pred] ckpt={ckpt}", flush=True)
    print(f"[pred] out_dir={out_dir}", flush=True)
    if args.tta:
        print(f"[pred] TTA enabled: {args.tta_passes} passes", flush=True)

    dl, ds, d_res, d_chain = build_pred_loader(P, args.split, args.batch_size, args.num_workers)

    cfg = build_model_config(P, d_res, d_chain)
    cfg.amp_eval = False
    ckpt_obj = torch.load(ckpt, map_location="cpu")
    sd, _meta = _extract_state_dict(ckpt_obj)
    n_cross = _infer_n_cross_from_state_dict(sd)
    n_enc   = _infer_n_enc_from_state_dict(sd)
    if n_cross > 0: cfg.n_cross_layers = int(n_cross)
    if n_enc   > 0: cfg.n_encoder_layers = int(n_enc)
    print(f"[pred] cfg enc={cfg.n_encoder_layers} cross={cfg.n_cross_layers} "
          f"d_res={cfg.d_res_in} d_chain={cfg.d_chain_in}", flush=True)

    model = UnifiedInterfaceModel(cfg).to(device)
    _load_state_flexible(model, sd)
    model.eval()

    thr_mode = args.thr_mode or getattr(P, "val_thr_mode", "auto_mcc")
    thr_grid = int(args.thr_grid if args.thr_grid is not None else getattr(P, "val_thr_grid", 121))
    thr_min  = float(args.thr_min if args.thr_min is not None else getattr(P, "val_thr_min", 0.05))
    thr_max  = float(args.thr_max if args.thr_max is not None else getattr(P, "val_thr_max", 0.95))

    set_seed(int(P.seed))
    metrics = eval_rbp296_pred(
        model, dl, device=device,
        thr=args.thr,
        thr_mode=str(thr_mode), thr_min=float(thr_min),
        thr_max=float(thr_max), thr_grid=int(thr_grid),
        tta=bool(args.tta), tta_passes=int(args.tta_passes),
    )
    primary = str(getattr(P, "primary_objective", "binary")).strip().lower()
    print(_fmt_metrics(args.split, metrics, primary=primary), flush=True)

    # Save summary JSON (structure preserved from original)
    summary_path = os.path.join(out_dir, "pred_metrics_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "ckpt":         ckpt,
            "device":       device,
            "split":        args.split,
            "dataset_len":  len(ds),
            "batch_size":   int(args.batch_size),
            "stride_frac":  float(args.stride_frac),
            "thr_mode":     str(thr_mode),
            "thr":          args.thr,
            "thr_grid":     int(thr_grid),
            "thr_min":      float(thr_min),
            "thr_max":      float(thr_max),
            "tta":          bool(args.tta),
            "tta_passes":   int(args.tta_passes),
            "metrics":      metrics,
            "time":         time.strftime("%Y-%m-%d %H:%M:%S"),
        }, f, indent=2, ensure_ascii=False)
    print(f"[pred] saved metrics summary: {summary_path}", flush=True)

# Ranking-ready exports (CSV/JSONL/NPZ/TopK-JSON keep the original layout)
    topk_k = int(args.topk_k if args.topk_k is not None else getattr(P, "explain_topk_k", 256))
    collect_ranking_ready_outputs(
        model=model, dl=dl, out_dir=out_dir, device=device,
        stride_frac=float(args.stride_frac), topk_k=topk_k,
        dump_topk_json=bool(args.dump_topk_json),
        max_dump_json=int(args.dump_topk_json_max),
    )

    # TopK final summary (always printed regardless of primary mode)
    tk_po = metrics.get("topk_posonly", {})
    tk_al = metrics.get("topk_all", {})
    if tk_po:
        def _r(d, k): return float(d.get(k, {}).get("rec", 0.0)) if isinstance(d.get(k), dict) else 0.0
        def _p(d, k): return float(d.get(k, {}).get("prec", 0.0)) if isinstance(d.get(k), dict) else 0.0
        def _h(d, k): return float(d.get(k, {}).get("hit", 0.0)) if isinstance(d.get(k), dict) else 0.0
        n_po = tk_po.get("L10", {}).get("n", 0) if isinstance(tk_po.get("L10"), dict) else 0
        print(f"[pred] === TopK Final Summary ({args.split}, pos-only n={n_po}) ===", flush=True)
        print(f"[pred]  Recall  @L/5={_r(tk_po,'L5'):.4f}  @L/10={_r(tk_po,'L10'):.4f}  @L/20={_r(tk_po,'L20'):.4f}", flush=True)
        print(f"[pred]  Precis  @K10={_p(tk_po,'10'):.4f}  @K20={_p(tk_po,'20'):.4f}  @K50={_p(tk_po,'50'):.4f}", flush=True)
        print(f"[pred]  Hit%    @K10={_h(tk_po,'10'):.4f}  @K20={_h(tk_po,'20'):.4f}  @K50={_h(tk_po,'50'):.4f}", flush=True)
        print(f"[pred]  Recall(all) @L/5={_r(tk_al,'L5'):.4f}  @L/10={_r(tk_al,'L10'):.4f}  @L/20={_r(tk_al,'L20'):.4f}", flush=True)

    print("[pred] done.", flush=True)


if __name__ == "__main__":
    main()

