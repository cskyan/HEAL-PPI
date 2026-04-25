# -*- coding: utf-8 -*-
import os
import math
from typing import Optional, Dict, Tuple
import torch.utils.checkpoint as checkpoint
import torch
import torch.nn as nn
import torch.nn.functional as F


def _bmask(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if x is None:
        return None
    if x.dtype == torch.bool:
        return x
    return x > 0.5


from config_topk import ModelConfig



class FiLM(nn.Module):
    """Feature-wise linear modulation conditioned on pooled chain features."""

    def __init__(self, d_model: int, d_chain: int):
        super().__init__()
        dc = max(8, d_chain if d_chain > 0 else 8)
        self.proj = nn.Sequential(
            nn.Linear(dc, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model * 2),
        )

    def forward(self, x: torch.Tensor, chain_pooled: Optional[torch.Tensor]) -> torch.Tensor:
        if chain_pooled is None:
            return x
        B, L, D = x.shape
        g = self.proj(chain_pooled)  # [B, 2D]
        gamma, beta = g.chunk(2, dim=-1)  # [B, D] x2
        gamma = gamma.unsqueeze(1).expand(-1, L, -1)
        beta = beta.unsqueeze(1).expand(-1, L, -1)
        # Keep modulation bounded so the FiLM branch cannot dominate the backbone.
        gamma = torch.tanh(gamma) * 0.5
        return x * (1.0 + gamma) + beta * 0.1


class LayerScale(nn.Module):
    """LayerScale: learnable per-channel residual scaling (stability for deep transformers)."""

    def __init__(self, d: int, init_value: float = 1e-4):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d) * float(init_value))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward network (better conditioning than GELU MLP)."""

    def __init__(self, d: int, mult: float = 4.0, drop: float = 0.0):
        super().__init__()
        # keep params ~ comparable to 4x FFN by using (2/3)*mult as hidden factor
        hidden = max(32, int(d * float(mult) * 2.0 / 3.0))
        self.w1 = nn.Linear(d, hidden)
        self.w2 = nn.Linear(d, hidden)
        self.w3 = nn.Linear(hidden, d)
        self.drop = nn.Dropout(float(drop))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.w1(x) * F.silu(self.w2(x))
        x = self.drop(x)
        return self.w3(x)



class SelfEnc(nn.Module):
    """Transformer encoder stack for per-chain residue context modeling."""

    def __init__(self, d: int, n_layers: int, h: int, drop: float):
        super().__init__()
        if n_layers <= 0:
            self.mod = nn.Identity()
        else:
            layer = nn.TransformerEncoderLayer(
                d_model=d,
                nhead=h,
                dim_feedforward=d * 4,
                dropout=drop,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.mod = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if isinstance(self.mod, nn.Identity):
            return x
        kpm = None
        if mask is not None:
            kpm = ~_bmask(mask)  # True=pad
        return self.mod(x, src_key_padding_mask=kpm)


class CrossBlock(nn.Module):

    def __init__(
            self,
            d: int,
            h: int,
            drop: float,
            use_layerscale: bool = False,
            layerscale_init: float = 1e-4,
            use_swiglu: bool = False,
            ffn_mult: float = 4.0,
    ):
        super().__init__()
        self.use_layerscale = bool(use_layerscale)
        self.use_swiglu = bool(use_swiglu)

        # Multi-head attention blocks operate on [B, L, D].
        self.qA_kB = nn.MultiheadAttention(d, h, dropout=drop, batch_first=True)
        self.qB_kA = nn.MultiheadAttention(d, h, dropout=drop, batch_first=True)
        self.saA = nn.MultiheadAttention(d, h, dropout=drop, batch_first=True)
        self.saB = nn.MultiheadAttention(d, h, dropout=drop, batch_first=True)

        if self.use_swiglu:
            self.ffA = SwiGLUFFN(d, mult=ffn_mult, drop=drop)
            self.ffB = SwiGLUFFN(d, mult=ffn_mult, drop=drop)
        else:
            self.ffA = nn.Sequential(nn.Linear(d, d * 4), nn.GELU(), nn.Linear(d * 4, d))
            self.ffB = nn.Sequential(nn.Linear(d, d * 4), nn.GELU(), nn.Linear(d * 4, d))

        # Pre-layernorm stack for cross-attn, self-attn, and FFN.
        self.norm1A = nn.LayerNorm(d)
        self.norm1B = nn.LayerNorm(d)
        self.norm2A = nn.LayerNorm(d)
        self.norm2B = nn.LayerNorm(d)
        self.norm3A = nn.LayerNorm(d)
        self.norm3B = nn.LayerNorm(d)

        self.drop = nn.Dropout(drop)

        # Residual scaling: fixed 0.5 is ok, LayerScale is better
        if self.use_layerscale:
            self.res_scale = 1.0
            self.ls_crossA = LayerScale(d, init_value=layerscale_init)
            self.ls_crossB = LayerScale(d, init_value=layerscale_init)
            self.ls_selfA = LayerScale(d, init_value=layerscale_init)
            self.ls_selfB = LayerScale(d, init_value=layerscale_init)
            self.ls_ffA = LayerScale(d, init_value=layerscale_init)
            self.ls_ffB = LayerScale(d, init_value=layerscale_init)
        else:
            self.res_scale = 0.5
            self.ls_crossA = nn.Identity()
            self.ls_crossB = nn.Identity()
            self.ls_selfA = nn.Identity()
            self.ls_selfB = nn.Identity()
            self.ls_ffA = nn.Identity()
            self.ls_ffB = nn.Identity()

    def forward(self, xA, xB, mA=None, mB=None):
        kpmA = None if mA is None else ~_bmask(mA)
        kpmB = None if mB is None else ~_bmask(mB)

        # 1. Cross (Pre-LN + LayerScale)
        nA, nB = self.norm1A(xA), self.norm1B(xB)
        hA, _ = self.qA_kB(nA, nB, nB, key_padding_mask=kpmB, need_weights=False)
        hB, _ = self.qB_kA(nB, nA, nA, key_padding_mask=kpmA, need_weights=False)
        xA = xA + self.drop(self.ls_crossA(hA)) * self.res_scale
        xB = xB + self.drop(self.ls_crossB(hB)) * self.res_scale

        # 2. Self (Pre-LN + LayerScale)
        nA, nB = self.norm2A(xA), self.norm2B(xB)
        zA, _ = self.saA(nA, nA, nA, key_padding_mask=kpmA, need_weights=False)
        zB, _ = self.saB(nB, nB, nB, key_padding_mask=kpmB, need_weights=False)
        xA = xA + self.drop(self.ls_selfA(zA)) * self.res_scale
        xB = xB + self.drop(self.ls_selfB(zB)) * self.res_scale

        # 3. FFN (Pre-LN + LayerScale)
        uA = self.ffA(self.norm3A(xA))
        uB = self.ffB(self.norm3B(xB))
        xA = xA + self.drop(self.ls_ffA(uA)) * self.res_scale
        xB = xB + self.drop(self.ls_ffB(uB)) * self.res_scale

        return xA, xB

class CrossEnc(nn.Module):
    def __init__(
            self,
            d: int,
            n_layers: int,
            h: int,
            drop: float,
            use_layerscale: bool = False,
            layerscale_init: float = 1e-4,
            use_swiglu: bool = False,
            ffn_mult: float = 4.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            CrossBlock(d, h, drop,
                      use_layerscale=use_layerscale,
                      layerscale_init=layerscale_init,
                      use_swiglu=use_swiglu,
                      ffn_mult=ffn_mult)
            for _ in range(n_layers)
        ])

    def forward(
            self,
            xA: torch.Tensor,
            xB: torch.Tensor,
            mA: Optional[torch.Tensor] = None,
            mB: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        #  checkpoint
        def run_block(blk, xA, xB, mA, mB):
            return blk(xA, xB, mA, mB)

        for blk in self.layers:
            if self.training and xA.requires_grad:
                xA, xB = checkpoint.checkpoint(run_block, blk, xA, xB, mA, mB)
            else:
                xA, xB = blk(xA, xB, mA, mB)

        return xA, xB


class LiteUNet2D(nn.Module):
    """Lightweight UNet-style 2D refinement head."""

    def __init__(self, in_ch: int = 1, c: int = 16):
        super().__init__()
        self.e1 = nn.Conv2d(in_ch, c, 3, padding=1)
        # e2
        self.e2 = nn.Conv2d(c, c, 3, padding=2, dilation=2)
        self.down = nn.AvgPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.d1 = nn.Conv2d(c, c, 3, padding=2, dilation=2)
        self.out = nn.Conv2d(c, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        if H < 2 or W < 2:
            return torch.zeros((B, 1, H, W), device=x.device, dtype=x.dtype)

        z = F.gelu(self.e1(x))
        z = F.gelu(self.e2(z))
        z2 = self.down(z)
        z2 = F.gelu(self.d1(z2))
        z_up = self.up(z2)
        if z_up.shape[-2] != H or z_up.shape[-1] != W:
            z_up = F.interpolate(z_up, size=(H, W), mode="bilinear", align_corners=False)
        z = z + z_up
        out = self.out(z)
        return out



class _ResBlock2D(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        g = max(1, min(8, ch // 4))
        self.n1 = nn.GroupNorm(g, ch)
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.n2 = nn.GroupNorm(g, ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.n1(x))
        h = self.c1(h)
        h = F.silu(self.n2(h))
        h = self.c2(h)
        return x + h


class MSRefine2D(nn.Module):
    """Multi-scale dilated conv refine head (cheap)."""

    def __init__(self, in_ch: int = 3, c: int = 16):
        super().__init__()
        g = max(1, min(8, c // 4))
        self.inp = nn.Conv2d(in_ch, c, 1)
        self.n0 = nn.GroupNorm(g, c)
        self.d1 = nn.Conv2d(c, c, 3, padding=1, dilation=1)
        self.d2 = nn.Conv2d(c, c, 3, padding=2, dilation=2)
        self.d4 = nn.Conv2d(c, c, 3, padding=4, dilation=4)
        self.n1 = nn.GroupNorm(g, c)
        self.out = nn.Conv2d(c, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        if H < 2 or W < 2:
            return self.out(F.silu(self.n1(self.inp(x))))
        h = F.silu(self.n0(self.inp(x)))
        h = self.d1(h) + self.d2(h) + self.d4(h)
        h = F.silu(self.n1(h))
        return self.out(h)


class EnhancedUNet2D(nn.Module):
    """Shallow residual UNet refine head (still lightweight)."""

    def __init__(self, in_ch: int = 3, c: int = 16):
        super().__init__()
        g = max(1, min(8, c // 4))
        self.e0 = nn.Conv2d(in_ch, c, 3, padding=1)
        self.e1 = _ResBlock2D(c)
        self.down = nn.AvgPool2d(2)
        self.mid0 = nn.Conv2d(c, c, 3, padding=2, dilation=2)
        self.mid1 = _ResBlock2D(c)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.d0 = nn.Conv2d(c + c, c, 3, padding=1)
        self.d1 = _ResBlock2D(c)
        self.n = nn.GroupNorm(g, c)
        self.out = nn.Conv2d(c, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        if H < 2 or W < 2:
            h = F.silu(self.n(self.e0(x)))
            return self.out(h)
        x0 = self.e0(x)
        x0 = self.e1(x0)
        x1 = self.down(x0)
        x1 = self.mid0(x1)
        x1 = self.mid1(x1)
        x2 = self.up(x1)
        # handle odd shapes
        if x2.size(-2) != x0.size(-2) or x2.size(-1) != x0.size(-1):
            x2 = F.interpolate(x2, size=(x0.size(-2), x0.size(-1)), mode='bilinear', align_corners=False)
        x = torch.cat([x0, x2], dim=1)
        x = self.d0(x)
        x = self.d1(x)
        x = F.silu(self.n(x))
        return self.out(x)


class FragmentHead(nn.Module):
    """
    Fragment-level evidence head.

    This head replaces plain average pooling with a lightweight depthwise +
    pointwise convolution so fragment evidence remains local and learnable.
    """

    def __init__(
            self,
            d_model: int,
            kernel_size: int = 9,
            use_gelu: bool = True,
            dropout: float = 0.1
    ):
        super().__init__()

        # Depthwise convolution keeps fragment evidence local while staying cheap.
        self.depthwise = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=d_model,
            bias=False
        )

        self.bn1 = nn.BatchNorm1d(d_model)
        self.act = nn.GELU() if use_gelu else nn.ReLU()

        # Pointwise convolution mixes channels after the depthwise step.
        self.pointwise = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=1,
            bias=True
        )

        self.bn2 = nn.BatchNorm1d(d_model)
        self.dropout = nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity()

        # Final projection to scalar
        self.proj = nn.Linear(d_model, 1)
        self._init_weights()


    def _init_weights(self):
        """Initialize convolution and projection layers."""
        nn.init.xavier_uniform_(self.depthwise.weight)
        nn.init.xavier_uniform_(self.pointwise.weight)
        nn.init.zeros_(self.pointwise.bias)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, mask=None):
        """
        Args:
            x: [B, L, D] residue features
            mask: [B, L] optional padding mask

        Returns:
            logit: [B, L] fragment evidence logits
        """
        _, _, _ = x.shape

        # Transpose for Conv1d: [B, L, D] -> [B, D, L]
        x_t = x.transpose(1, 2)  # [B, D, L]

        # Depthwise Conv
        x_dw = self.depthwise(x_t)  # [B, D, L]
        x_dw = self.bn1(x_dw)
        x_dw = self.act(x_dw)

        # Pointwise Conv
        x_pw = self.pointwise(x_dw)  # [B, D, L]
        x_pw = self.bn2(x_pw)
        x_pw = self.act(x_pw)
        x_pw = self.dropout(x_pw)

        # Transpose back: [B, D, L] -> [B, L, D]
        x_out = x_pw.transpose(1, 2)  # [B, L, D]

        # Apply mask if provided
        if mask is not None:
            # mask: [B, L] -> [B, L, 1]
            mask_3d = mask.unsqueeze(-1).float()
            x_out = x_out * mask_3d

        # Project to logit
        logit = self.proj(x_out)  # [B, L, 1]
        logit = logit.squeeze(-1)  # [B, L]

        return logit


class AttnGates(nn.Module):
    """ & gate."""

    def __init__(self, d: int, gmin: float = 0.1, gmax: float = 0.8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(d * 2, d),
            nn.GELU(),
            nn.Linear(d, 5),
        )
        self.gmin = gmin
        self.gmax = gmax

    def forward(self, gA: torch.Tensor, gB: torch.Tensor):
        z = self.fc(torch.cat([gA, gB], dim=-1))  # [B, 5]
        _, _, w1, w2, g = z.unbind(-1)
        w_logits = torch.stack([w1, w2], dim=-1)  # [B,2]
        bias = torch.tensor([0.0, 0.4], device=w_logits.device)
        w = torch.softmax(w_logits + bias, dim=-1)
        w = 0.08 + 0.92 * w
        w = w / (w.sum(-1, keepdim=True) + 1e-6)
        w1, w2 = w.unbind(-1)

        g = torch.sigmoid(g)
        g = self.gmin + (self.gmax - self.gmin) * g
        return w1, w2, g


class UnifiedInterfaceModel(nn.Module):
    """Hierarchical evidence-aligned predictor.

    Expected residue input features:
    - sequence embedding (ESM or equivalent)
    - optional PSSM features
    - optional DSSP features
    - structure-derived supervision is used during training through labels/coordinates

    Public release output interface:
    - Output 1: ``p_fused`` for binary interaction/interface prediction
    - Output 2: ``evi_score`` / ``evi_logit`` for top-k evidence readout
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        # Residue-guided gating scale (learnable) to control L1.5 -> L2 influence
        # Unconstrained gate parameter (logit). We apply tanh() at use-time to keep alpha bounded.
        # Rationale:
        #   - init at 0 -> alpha=0 for a long time, L1.5 provides no usable prior to L2.
        #   - unbounded alpha can explode logits and hurt Top-K precision.
        # So we use a bounded mapping with a configurable max.
        self.interface_gate_scale = nn.Parameter(torch.tensor(0.5))
        DCH = cfg.d_chain_in
        D = cfg.d_model
        H = cfg.n_heads
        DR = cfg.dropout

        # inputs
        self.inA = nn.Linear(cfg.d_res_in, D)
        self.inB = nn.Linear(cfg.d_res_in, D)
        self.filmA = FiLM(D, DCH)
        self.filmB = FiLM(D, DCH)

        self.encA = SelfEnc(D, cfg.n_encoder_layers, H, DR)
        self.encB = SelfEnc(D, cfg.n_encoder_layers, H, DR)
        self.cross = CrossEnc(
            D, cfg.n_cross_layers, H, DR,
            use_layerscale=bool(getattr(cfg, 'use_layerscale', False)),
            layerscale_init=float(getattr(cfg, 'layerscale_init', 1e-4)),
            use_swiglu=bool(getattr(cfg, 'use_swiglu', False)),
            ffn_mult=float(getattr(cfg, 'ffn_mult', 4.0)),
        )

        # L1
        self.head_resA = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, 1))
        self.head_resB = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, 1))

        # L1.5 fragment head used by the residue-guided contact refinement stage.
        # kernel size is configurable; keep default=9 to match previous avg_pool1d behavior.
        frag_k = int(getattr(cfg, "frag_k", 9))
        self.frag_head = FragmentHead(d_model=D, kernel_size=frag_k)

        # L2
        self.l2_dim = 64
        self.l2_projA = nn.Linear(D, self.l2_dim)
        self.l2_projB = nn.Linear(D, self.l2_dim)
        variant = str(getattr(cfg, 'unet_variant', os.environ.get('MODEL_UNET_VARIANT', 'lite'))).lower()
        if variant in ('lite', 'l'):
            self.unet = LiteUNet2D(in_ch=3, c=cfg.unet_ch)
        elif variant in ('msfp', 'multi', 'pyramid'):
            self.unet = MSRefine2D(in_ch=3, c=cfg.unet_ch)
        elif variant in ('enhanced', 'unet', 'resunet'):
            self.unet = EnhancedUNet2D(in_ch=3, c=cfg.unet_ch)
        else:
            # fallback for unknown values
            self.unet = LiteUNet2D(in_ch=3, c=cfg.unet_ch)

        prior = float(cfg.prior_pos_pix)
        prior = max(min(prior, 0.5 - 1e-4), 1e-6)
        logit_prior = math.log(prior / (1.0 - prior))
        self.bias2d = nn.Parameter(torch.tensor(logit_prior, dtype=torch.float32))

        # prior-matching + temperature
        self.l2_prior_target = float(cfg.prior_pos_pix)
        self.l2_shift_strength = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.l2_temp = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.l2_prior_reg_w = 0.1

        self.gates = AttnGates(D, gmin=0.1, gmax=0.8)
        self.register_buffer("eps", torch.tensor(1e-6))

        # cache for loss/explain
        self._last_resA_logit = None
        self._last_resB_logit = None
        self._last_fragA_logit = None
        self._last_fragB_logit = None
        self._last_S = None
        self._last_S_raw = None
        self._last_evi_logit = None
        self._last_gate = None
        self._last_w = None
        self._last_topk_idx = None
        self._last_topk_val = None
        self._last_decomp = None
        self._last_gate_info = None

        self.training_epoch: int = 0
        self.temperature = nn.Parameter(torch.tensor(1.5))
    @staticmethod
    def _pool_chain(chain: Optional[torch.Tensor], mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if chain is None:
            return None
        if chain.dim() == 2:
            return chain
        if chain.dim() == 3:
            if mask is None:
                return chain.mean(dim=1)
            m = _bmask(mask).float().unsqueeze(-1)
            num = (chain * m).sum(dim=1)
            den = m.sum(dim=1).clamp_min(1.0)
            return num / den
        raise ValueError("Unsupported chain shape")

    @staticmethod
    def _global_pool(x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is None:
            return x.mean(dim=1)
        m = _bmask(mask).float().unsqueeze(-1)
        num = (x * m).sum(dim=1)
        den = m.sum(dim=1).clamp_min(1.0)
        return num / den

    # ==============================
    # Step-1/2 core: Top-k pooling
    # ==============================
    @staticmethod
    def topk_pool_from_logits(
            S_logits: torch.Tensor,
            valid_mask: Optional[torch.Tensor] = None,
            k: int = 256,
            frac: float = 0.02,
            invalid_fill: float = -1e9,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Top-k sparse pooling over a 2D contact-logit map.

        Returns:
            p_topk:   [B] mean sigmoid over valid top-k logits
            idx_flat: [B, K] flat indices in [0, La*Lb)
            val_topk: [B, K] top-k logits (padded)
        """
        if S_logits.dim() != 3:
            raise ValueError(f"S_logits must be [B,La,Lb], got {tuple(S_logits.shape)}")
        B, La, Lb = S_logits.shape
        device = S_logits.device

        if valid_mask is None:
            valid_mask = torch.ones((B, La, Lb), dtype=torch.bool, device=device)
        else:
            if valid_mask.dtype != torch.bool:
                valid_mask = valid_mask > 0.5
            if valid_mask.shape != (B, La, Lb):
                valid_mask = valid_mask.view(B, La, Lb)

        K = int(k) if int(k) > 0 else 0
        if frac is not None and float(frac) > 0:
            Kf = int(max(1, round(float(frac) * float(La * Lb))))
            K = min(K, Kf) if K > 0 else Kf
        K = max(1, K)

        flat = S_logits.view(B, -1)
        vflat = valid_mask.view(B, -1)
        masked = flat.masked_fill(~vflat, float(invalid_fill))

        kk = min(K, masked.shape[1])
        val, idx = torch.topk(masked, k=kk, largest=True, sorted=False)

        sel_valid = vflat.gather(1, idx)
        val_sig = torch.sigmoid(val)
        num = (val_sig * sel_valid.float()).sum(dim=1)
        den = sel_valid.float().sum(dim=1).clamp_min(1.0)
        p_topk = num / den

        if val.shape[1] < K:
            padn = K - val.shape[1]
            idx = torch.cat([idx, torch.zeros((B, padn), dtype=idx.dtype, device=device)], dim=1)
            val = torch.cat([val, torch.full((B, padn), float(invalid_fill), device=device, dtype=val.dtype)], dim=1)

        return p_topk, idx[:, :K], val[:, :K]

    @staticmethod
    @torch.no_grad()
    def topk_pool_from_logits_infer(
            S_logits: torch.Tensor,
            valid_mask: Optional[torch.Tensor] = None,
            k: int = 256,
            frac: float = 0.02,
            invalid_fill: float = -1e9,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Inference-only top-k pooling (no_grad wrapper)."""
        return UnifiedInterfaceModel.topk_pool_from_logits(
            S_logits=S_logits, valid_mask=valid_mask, k=k, frac=frac, invalid_fill=invalid_fill
        )

    # --------- L2 logits ---------

    @staticmethod
    def _listwise_pos_mass_loss(
            logits: torch.Tensor,
            labels: torch.Tensor,
            pos_idx: torch.Tensor,
            neg_idx: torch.Tensor,
            hard_neg_cap: int = 2048,
            tau: float = 1.0,
            eps: float = 1e-6,
    ) -> torch.Tensor:
        """Listwise ranking loss that directly pushes positives into the top of the logit list.

        loss = -log( sum softmax(logits/tau)[pos] )

        We build a candidate set = all positives + top hard negatives (by logit).
        This is cheap and aligns well with Top-K evaluation.
        """
        device = logits.device
        npos = int(pos_idx.numel())
        nneg = int(neg_idx.numel())
        if npos <= 0 or nneg <= 0:
            return torch.zeros((), device=device)

        # pick hard negatives by logit (largest)
        hard = min(int(hard_neg_cap), nneg)
        s_neg = logits[neg_idx]
        _, topi = torch.topk(s_neg, k=hard, largest=True, sorted=False)
        neg_pick = neg_idx[topi]

        cand = torch.cat([pos_idx, neg_pick], dim=0)
        s = logits[cand]
        # softmax distribution over candidates
        p = torch.softmax(s / max(float(tau), 1e-6), dim=0)
        pos_mass = p[:npos].sum().clamp_min(eps)
        return -torch.log(pos_mass)

    @staticmethod
    def _margin_pos_vs_hardneg(
            logits: torch.Tensor,
            pos_idx: torch.Tensor,
            neg_idx: torch.Tensor,
            hard_neg_cap: int = 512,
            margin: float = 1.0,
    ) -> torch.Tensor:
        """Softplus margin loss: want pos_mean >= neg_hard_mean + margin."""
        device = logits.device
        npos = int(pos_idx.numel())
        nneg = int(neg_idx.numel())
        if npos <= 0 or nneg <= 0:
            return torch.zeros((), device=device)
        pos_logits = logits[pos_idx]
        neg_logits = logits[neg_idx]
        hard = min(int(hard_neg_cap), int(neg_logits.numel()))
        neg_hard = torch.topk(neg_logits, k=hard, largest=True, sorted=False).values
        pos_mean = pos_logits.mean()
        neg_mean = neg_hard.mean()
        return F.softplus(float(margin) - (pos_mean - neg_mean))

    def _l2_logits(
            self,
            xA: torch.Tensor,
            xB: torch.Tensor,
            maskA: Optional[torch.Tensor],
            maskB: Optional[torch.Tensor],
    ) -> torch.Tensor:
        B, La, _ = xA.shape
        Lb = xB.size(1)
        maxL = int(self.cfg.l2_max_len)
        use_amp = bool(self.cfg.amp_eval and (not self.training))

        with torch.cuda.amp.autocast(enabled=use_amp):
            if La * Lb > maxL * maxL:
                rA = max(1, (La + maxL - 1) // maxL)
                rB = max(1, (Lb + maxL - 1) // maxL)
                xA_ds = F.avg_pool1d(xA.transpose(1, 2), rA, stride=rA).transpose(1, 2) if rA > 1 else xA
                xB_ds = F.avg_pool1d(xB.transpose(1, 2), rB, stride=rB).transpose(1, 2) if rB > 1 else xB

                La_ds = xA_ds.size(1)
                Lb_ds = xB_ds.size(1)

                zA = self.l2_projA(xA_ds)
                zB = self.l2_projB(xB_ds)
                d_l2 = zA.size(-1)
                S0 = torch.einsum("bid,bjd->bij", zA, zB) / math.sqrt(float(d_l2))

                i_idx = torch.arange(La_ds, device=xA_ds.device, dtype=torch.float32).view(La_ds, 1)
                j_idx = torch.arange(Lb_ds, device=xB_ds.device, dtype=torch.float32).view(1, Lb_ds)
                d_seq = torch.abs(i_idx - j_idx) / (max(La_ds, Lb_ds) + 1e-6)
                diag_band = (torch.abs(i_idx - j_idx) <= 2).float()
                d_seq = d_seq.unsqueeze(0).expand(B, -1, -1)
                diag_band = diag_band.unsqueeze(0).expand(B, -1, -1)

                s_in = torch.stack([S0, d_seq, diag_band], dim=1)  # [B,3,La_ds,Lb_ds]
                s_ref = self.unet(s_in).squeeze(1)
                s_ref = torch.tanh(s_ref) * 2.0
                S_low = S0 + s_ref

                S = F.interpolate(S_low.unsqueeze(1), size=(La, Lb), mode="bilinear", align_corners=False).squeeze(1)
            else:
                zA = self.l2_projA(xA)
                zB = self.l2_projB(xB)
                d_l2 = zA.size(-1)
                S0 = torch.einsum("bid,bjd->bij", zA, zB) / math.sqrt(float(d_l2))

                i_idx = torch.arange(La, device=xA.device, dtype=torch.float32).view(La, 1)
                j_idx = torch.arange(Lb, device=xB.device, dtype=torch.float32).view(1, Lb)
                d_seq = torch.abs(i_idx - j_idx) / (max(La, Lb) + 1e-6)
                diag_band = (torch.abs(i_idx - j_idx) <= 2).float()
                d_seq = d_seq.unsqueeze(0).expand(B, -1, -1)
                diag_band = diag_band.unsqueeze(0).expand(B, -1, -1)

                s_in = torch.stack([S0, d_seq, diag_band], dim=1)  # [B,3,La,Lb]
                s_ref = self.unet(s_in).squeeze(1)
                s_ref = torch.tanh(s_ref) * 2.0
                S = S0 + s_ref

        S = S + self.bias2d
        S = torch.nan_to_num(S, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)

        valid = None
        if maskA is not None and maskB is not None:
            valid = maskA[:, :, None] & maskB[:, None, :]
            S = S.masked_fill(~valid, -8.0)

        # prior-match calibration (detach)
        target = float(getattr(self, "l2_prior_target", self.cfg.prior_pos_pix))
        target = max(min(target, 1 - 1e-4), 1e-4)

        if valid is not None and valid.any():
            v = valid.view(B, -1)
            Sd = S.detach().view(B, -1)
            p = torch.sigmoid(Sd)
            num = (p * v.float()).sum(dim=1)
            den = v.float().sum(dim=1).clamp_min(1.0)
            p_mean = (num / den).clamp(1e-4, 1 - 1e-4)

            logit_target = math.log(target / (1.0 - target))
            logit_mean = torch.log(p_mean / (1.0 - p_mean))
            shift = (logit_target - logit_mean).view(B, 1, 1)

            strength = torch.sigmoid(self.l2_shift_strength)
            S = S + strength * shift

        temp = self.l2_temp.abs().clamp_min(0.5)
        S = (S / temp).clamp(-20.0, 20.0)
        return S

    # --------- forward ---------
    def forward(
            self,
            resA: torch.Tensor,
            maskA: Optional[torch.Tensor],
            chainA: Optional[torch.Tensor],
            resB: torch.Tensor,
            maskB: Optional[torch.Tensor],
            chainB: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        if resA.dim() == 2:
            resA = resA.unsqueeze(0)
        if resB.dim() == 2:
            resB = resB.unsqueeze(0)

        maskA = _bmask(maskA) if maskA is not None else None
        maskB = _bmask(maskB) if maskB is not None else None

        xA = self.inA(resA)
        xB = self.inB(resB)

        chainA_p = self._pool_chain(chainA, maskA)
        chainB_p = self._pool_chain(chainB, maskB)
        xA = self.filmA(xA, chainA_p)
        xB = self.filmB(xB, chainB_p)

        xA = self.encA(xA, maskA)
        xB = self.encB(xB, maskB)
        xA, xB = self.cross(xA, xB, maskA, maskB)

        gA = self._global_pool(xA, maskA)
        gB = self._global_pool(xB, maskB)
        w1, w2, g = self.gates(gA, gB)
        self._last_gate = g
        self._last_w = (w1, w2)

        zA = self.head_resA(xA)
        zB = self.head_resB(xB)
        # numerical guard for residue logits
        zA = torch.nan_to_num(zA, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
        zB = torch.nan_to_num(zB, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
        if maskA is not None:
            zA = zA.masked_fill(~maskA.unsqueeze(-1), -30.0)
        if maskB is not None:
            zB = zB.masked_fill(~maskB.unsqueeze(-1), -30.0)

        S = self._l2_logits(xA, xB, maskA, maskB)

        valid2d = None
        if maskA is not None and maskB is not None:
            valid2d = maskA[:, :, None] & maskB[:, None, :]



        # --- Residue-guided spatial gating (RGCR / global filter) ---
        # Step 1: explicit fragment evidence from the L1.5 bridge.
        zA1 = zA.squeeze(-1)  # [B, L]
        zB1 = zB.squeeze(-1)  # [B, M]
        logit_fragA = self.frag_head(xA, mask=maskA)
        logit_fragB = self.frag_head(xB, mask=maskB)
        # Numerical guard before the fragment signals are reused by the gate.
        logit_fragA = torch.nan_to_num(logit_fragA, nan=0.0, posinf=10.0, neginf=-10.0)
        logit_fragB = torch.nan_to_num(logit_fragB, nan=0.0, posinf=10.0, neginf=-10.0)
        # Clamp fragment logits to prevent L1->L2 gate explosion
        frag_clip = float(os.environ.get('L12_FRAG_LOGIT_CLAMP', '6.0'))
        logit_fragA = logit_fragA.clamp(-frag_clip, frag_clip)
        logit_fragB = logit_fragB.clamp(-frag_clip, frag_clip)

        # Step-1: logit-domain hard additive gating (ReLU-Gate variant)
        S_raw = torch.nan_to_num(S, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)

        # Warm up the residue-to-contact gate so L2 is not over-constrained too early.
        warmup = int(os.environ.get('L12_GATE_WARMUP_EPOCHS', '4'))
        cur_ep = int(getattr(self, 'training_epoch', 0) or 0)
        start_fac = float(os.environ.get('L12_GATE_WARMUP_START', '0.3'))
        if warmup <= 0:
            gate_factor = 1.0
        elif cur_ep < warmup:
            gate_factor = start_fac + (1.0 - start_fac) * (float(cur_ep) / float(max(1, warmup)))
        else:
            gate_factor = 1.0

        alpha_max = float(os.environ.get('L12_GATE_ALPHA_MAX', '0.8'))
        alpha = torch.tanh(self.interface_gate_scale) * alpha_max * gate_factor

        # Residue evidence is aligned to contact space and used as an additive gate.
        gate_signal = logit_fragA.unsqueeze(-1) + logit_fragB.unsqueeze(1)

        gate_sig_clip = float(os.environ.get('L12_GATE_SIGNAL_CLAMP', '6.0'))
        gate_signal = gate_signal.clamp(-gate_sig_clip, gate_sig_clip)

        gate_pos = F.relu(gate_signal)
        gate_neg = F.relu(-gate_signal)
        gate_adj = gate_pos - 0.3 * gate_neg

        S = S_raw + alpha * gate_adj
        S = torch.clamp(S, -20.0, 20.0)
        # ---- Top-K pooling AFTER RGCR gating ----
        p_topk, idx_flat, val_topk = UnifiedInterfaceModel.topk_pool_from_logits(
            S_logits=S,
            valid_mask=valid2d,
            k=int(getattr(self.cfg, "l2_topk_k", 256)),
            frac=float(getattr(self.cfg, "l2_topk_frac", 0.02)),
        )
        self._last_topk_idx = idx_flat
        self._last_topk_val = val_topk

        # ---- evidence logit (from gated S) ----
        if valid2d is not None:
            vflat = valid2d.view(valid2d.size(0), -1)
            sel_valid = vflat.gather(1, idx_flat)
            v = torch.nan_to_num(val_topk, nan=-20.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
            num = (v * sel_valid.float()).sum(dim=1)
            den = sel_valid.float().sum(dim=1).clamp_min(1.0)
            evi_logit = (num / den).view(-1, 1)
        else:
            evi_logit = torch.nan_to_num(
                val_topk.mean(dim=1, keepdim=True),
                nan=0.0, posinf=20.0, neginf=-20.0
            ).clamp(-20.0, 20.0)

        p_res_A = torch.sigmoid(zA.squeeze(-1))
        p_res_B = torch.sigmoid(zB.squeeze(-1))

        p2 = torch.sigmoid(S)

        sa = p_res_A.mean(dim=1, keepdim=True)
        sb = p_res_B.mean(dim=1, keepdim=True)
        s1 = 0.5 * (sa + sb)

        # Mean contact probability over valid residue pairs only.
        if valid2d is not None:
            v2 = valid2d.float()
            denom = v2.sum(dim=(1, 2), keepdim=True).clamp_min(1.0)
            s2 = (p2 * v2).sum(dim=(1, 2), keepdim=True) / denom
        else:
            s2 = p2.mean(dim=(1, 2), keepdim=True)

        w1m, w2m = (w1.unsqueeze(-1), w2.unsqueeze(-1))
        ws = (w1m + w2m).clamp_min(1e-6)
        w1n = w1m / ws
        w2n = w2m / ws

        p_fused = (w1n * s1 + w2n * s2).clamp(1e-6, 1 - 1e-6).view(-1)

        try:
            self._last_gate_info = {
                "alpha": float(alpha.detach().cpu().item()) if torch.is_tensor(alpha) else float(alpha),
                "gate_factor": float(gate_factor),
                "w_res": w1n.detach().view(-1).float().cpu().tolist(),
                "w_l2": w2n.detach().view(-1).float().cpu().tolist(),
                "gate": g.detach().view(-1).float().cpu().tolist(),
            }
        except Exception:
            self._last_gate_info = None

        self._last_decomp = {
            "s_res": s1.detach().view(-1),
            "s_l2_mean": s2.detach().view(-1),
            "evi_logit": evi_logit.detach().view(-1),
            "w_res": w1n.detach().view(-1),
            "w_l2": w2n.detach().view(-1),
            "gate": g.detach().view(-1),
        }

        # ===== cache for loss (training criterion) =====
        # NOTE: keep tensors with grad for criterion
        self._last_resA_logit = zA1
        self._last_resB_logit = zB1
        self._last_fragA_logit = logit_fragA
        self._last_fragB_logit = logit_fragB
        self._last_S = S
        self._last_S_shape = (S.shape[1], S.shape[2])
        self._last_S_raw = S_raw.detach()
        self._last_evi_logit = evi_logit.detach() if evi_logit is not None else None
        evi_score = torch.sigmoid(evi_logit.view(-1))
        return {
            "S": S,
            "S_raw": S_raw,
            "p_res_A": p_res_A,
            "p_res_B": p_res_B,
            "p_frag_A": torch.sigmoid(logit_fragA),
            "p_frag_B": torch.sigmoid(logit_fragB),
            "logit_fragA": logit_fragA,
            "logit_fragB": logit_fragB,
            # Dual-output interface kept explicit for the public release:
            # - p_fused: binary prediction
            # - evi_score/evi_logit: evidence readout from the contact field
            "evi_logit": evi_logit.view(-1),
            "evi_score": evi_score,
            "p_fused": p_fused,
        }


    # --------- explainability helpers ---------
    @torch.no_grad()
    def get_last_topk_pairs(self, topk: int = 200):
        """Return last forward() Top-K pairs as (i,j,logit) per batch element."""
        idx_flat = getattr(self, "_last_topk_idx", None)
        val_topk = getattr(self, "_last_topk_val", None)
        shape = getattr(self, "_last_S_shape", None)
        if idx_flat is None or val_topk is None or shape is None:
            return None
        L, M = int(shape[0]), int(shape[1])
        k = min(int(topk), int(idx_flat.size(1)))
        idx = idx_flat[:, :k]
        val = val_topk[:, :k]
        i = (idx // M).long()
        j = (idx % M).long()
        return {"i": i, "j": j, "logit": val}

    @torch.no_grad()
    def get_last_gate_info(self):
        """Return gating scalars cached in the last forward()."""
        return getattr(self, "_last_gate_info", None)


    # --------- criterion ---------
    def criterion(
            self,
            out: Dict[str, torch.Tensor],
            y2d=None,
            y_res_A=None,
            y_res_B=None,
            maskA=None,
            maskB=None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Deprecated placeholder. This method is not used in current training.

        Loss computation is handled in the training script, not inside the
        model class. This method is kept only for backward compatibility.
        """
        import warnings

        warnings.warn(
            "model.criterion() is deprecated and should not be called. "
            "Loss is computed in train.py::forward_one().",
            category=RuntimeWarning,
            stacklevel=2,
        )
        raise NotImplementedError(
            "model.criterion() is DEPRECATED and should not be called.\n"
            "Loss computation is handled in train.py::forward_one().\n"
            "If you see this error, check your training script for calls to model.criterion()."
        )
