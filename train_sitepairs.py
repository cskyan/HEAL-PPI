#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared ESM sequence embedder utilities for the public release.

This module used to contain an older site-pair training pipeline. The current
top-k training and prediction scripts only depend on ``SeqEmbedder``, so the
legacy training code has been removed to keep the GitHub release focused.

Expected public data preparation:
- sequence input: FASTA or plain sequence strings
- optional residue features: PSSM and DSSP are loaded elsewhere
- structure-derived supervision is handled by dataset builders in the main
  training / prediction scripts
"""

import hashlib
import os
import re
from typing import Dict

import numpy as np
import torch


ESM_LOCAL_DIR = os.environ.get("ESM_LOCAL_DIR", "/path/to/resources/esm")
ESM_LOCAL_CKPT = os.path.join(ESM_LOCAL_DIR, "esm2_t33_650M_UR50D.pt")
ESM_HUB_CACHE = os.path.join(ESM_LOCAL_DIR, "hub")

os.makedirs(ESM_HUB_CACHE, exist_ok=True)
os.environ.setdefault("TORCH_HOME", ESM_HUB_CACHE)
os.environ.setdefault("TRANSFORMERS_CACHE", ESM_HUB_CACHE)
os.environ.setdefault("HF_HOME", ESM_HUB_CACHE)
os.environ.setdefault("XDG_CACHE_HOME", ESM_HUB_CACHE)


class SeqEmbedder:
    """Sequence-to-residue embedding helper backed by a local ESM checkpoint."""

    def __init__(self, backbone: str = "esm2_t33_650M_UR50D", cache_dir: str = None, device: str = "cpu"):
        self.backbone = str(backbone)
        self.cache = cache_dir or os.path.join(".", "esm_cache")
        self.device = str(device)
        self.mem = {}
        os.makedirs(self.cache, exist_ok=True)

        import esm

        print(f"[ESM] Loading local weights: {ESM_LOCAL_CKPT}")
        if not os.path.isfile(ESM_LOCAL_CKPT):
            raise FileNotFoundError(
                "Local ESM checkpoint not found. Set ESM_LOCAL_DIR or place the checkpoint at "
                f"{ESM_LOCAL_CKPT}"
            )

        try:
            self.esm_model, self.esm_alphabet = esm.pretrained.load_model_and_alphabet_local(ESM_LOCAL_CKPT)
        except Exception:
            from esm.pretrained import load_model_and_alphabet_core

            model_data = torch.load(ESM_LOCAL_CKPT, map_location="cpu")
            try:
                self.esm_model, self.esm_alphabet = load_model_and_alphabet_core(self.backbone, model_data)
            except TypeError:
                self.esm_model, self.esm_alphabet = load_model_and_alphabet_core(model_data)

        self.esm_model.to(self.device).eval()
        self.batch_converter = self.esm_alphabet.get_batch_converter()
        with torch.no_grad():
            self.d_out = int(self.esm_model.embed_dim)

    @staticmethod
    def _safe_filename(uid: str) -> str:
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(uid))[:80]
        hid = hashlib.md5(str(uid).encode("utf-8")).hexdigest()[:8]
        return f"{safe}__{hid}.npy"

    def encode(self, uid: str, seq: str) -> np.ndarray:
        uid = str(uid)
        seq = str(seq or "").strip()
        fp = os.path.join(self.cache, self._safe_filename(uid))

        if uid in self.mem:
            return self.mem[uid]

        if os.path.isfile(fp):
            try:
                arr = np.load(fp).astype(np.float32, copy=False)
                self.mem[uid] = arr
                return arr
            except Exception:
                pass

        if not seq:
            arr = np.zeros((1, self.d_out), dtype=np.float32)
            self.mem[uid] = arr
            return arr

        with torch.no_grad():
            _, _, batch_tokens = self.batch_converter([(uid, seq)])
            batch_tokens = batch_tokens.to(self.device)
            layer_id = 33 if "t33" in self.backbone else 12
            out = self.esm_model(batch_tokens, repr_layers=[layer_id])
            reps = out["representations"][max(out["representations"].keys())]
            arr = reps[0, 1:1 + len(seq), :].detach().cpu().float().numpy().astype(np.float32, copy=False)

        try:
            np.save(fp, arr)
        except Exception:
            pass

        self.mem[uid] = arr
        return arr


def warm_cache(embedder: SeqEmbedder, seqs: Dict[str, str]):
    """Precompute and cache embeddings for a sequence dictionary."""
    total = len(seqs)
    print(f"[WARM] Preloading ESM embeddings for {total} sequences...")
    for i, (uid, seq) in enumerate(seqs.items(), 1):
        try:
            embedder.encode(uid, seq)
        except Exception as exc:
            print(f"[WARM][WARN] {uid}: {exc}")
        if i % 50 == 0 or i == total:
            print(f"[WARM] {i}/{total} done")
    print("[WARM] Completed.")


def main():
    print(
        "train_sitepairs.py is now a lightweight shared embedder utility.\n"
        "Use train.py for training and predict.py for evaluation/prediction."
    )


if __name__ == "__main__":
    main()
