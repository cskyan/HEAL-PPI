# -*- coding: utf-8 -*-
"""Public configuration for the RBP top-k training and evaluation pipeline."""
import os
from dataclasses import dataclass
from typing import List, Optional


def _env_bool(key: str, default: bool) -> bool:
    v = os.environ.get(key, None)
    if v is None or str(v).strip() == "":
        return default
    return str(v).strip().lower() not in ("0", "false", "no", "off")


def _env_int(key: str, default: int) -> int:
    v = os.environ.get(key, None)
    if v is None or str(v).strip() == "":
        return default
    try:
        return int(float(v))
    except Exception:
        return default


def _env_float(key: str, default: float) -> float:
    v = os.environ.get(key, None)
    if v is None or str(v).strip() == "":
        return default
    try:
        return float(v)
    except Exception:
        return default


def infer_rbp_dataset_tag(p) -> str:
    joined = " ".join([
        str(getattr(p, "rbp_root", "") or ""),
        str(getattr(p, "rbp_id_list", "") or ""),
    ]).lower()
    if "rbp296" in joined:
        return "RBP296"
    if "rbp109" in joined:
        return "RBP109"
    return "UNKNOWN"


def validate_rbp_dataset_config(p) -> List[str]:
    warnings = []
    tag = infer_rbp_dataset_tag(p)
    if tag != "RBP296":
        warnings.append(
            f"configured RBP dataset looks like {tag}, not RBP296 "
            f"(rbp_root={getattr(p, 'rbp_root', '')}, rbp_id_list={getattr(p, 'rbp_id_list', '')})"
        )
    if not str(getattr(p, "rbp_root", "")).strip():
        warnings.append("rbp_root is empty")
    if not str(getattr(p, "rbp_id_list", "")).strip():
        warnings.append("rbp_id_list is empty")
    return warnings


def apply_topk_preset(p, preset: str):
    """Apply a lightweight preset without overwriting the whole config."""
    name = str(preset or "").strip().lower()
    if not name:
        return p
    if name == "recall_boost":
        # Conservative preset for the current symptom:
        # Hit@20 and P@10 are already decent, while L/5 and L/10 recall are low.
        # We soften overly aggressive pairwise ranking, strengthen listwise mass,
        # and give EMA / early-stop more room to capture a better epoch.
        p.primary_objective = "topk"
        p.label_smoothing_l1 = 0.05
        p.l1_pos_weight = 5.0
        p.l1_rank_w = 0.20
        p.l1_rank_start_epoch = 5
        p.l1_rank_margin = 0.5
        p.l1_rank_n_pairs = 128
        p.l1_rank_neg_hard_frac = 0.5
        p.l1_list_w = 0.35
        p.l1_list_start_epoch = 4
        p.l1_list_tau = 2.0
        p.ema_decay = 0.995
        p.early_min_epochs = max(int(getattr(p, "early_min_epochs", 12)), 12)
        p.early_patience = max(int(getattr(p, "early_patience", 10)), 10)
    if name == "balanced_topk":
        # Prefer a checkpoint that balances L/5, L/10 and top-K precision
        # instead of over-optimizing one recall point.
        p.primary_objective = "topk"
        p.label_smoothing_l1 = 0.02
        p.l1_pos_weight = 4.0
        p.l1_focal_w = 0.35
        p.l1_rank_w = 0.18
        p.l1_rank_start_epoch = 5
        p.l1_rank_margin = 0.5
        p.l1_rank_n_pairs = 128
        p.l1_rank_neg_hard_frac = 0.5
        p.l1_list_w = 0.18
        p.l1_list_start_epoch = 6
        p.l1_list_tau = 2.2
        p.ema_decay = 0.995
        p.early_min_epochs = max(int(getattr(p, "early_min_epochs", 16)), 16)
        p.early_patience = max(int(getattr(p, "early_patience", 8)), 8)
        p.topk_score_w_l5 = 0.35
        p.topk_score_w_l10 = 0.35
        p.topk_score_w_k10p = 0.20
        p.topk_score_w_hit20 = 0.10
    return p


@dataclass
class Params:
    """Runtime and optimization settings for the public release."""

    # ---- primary objective ----
    primary_objective: str = "topk"     # "binary" or "topk"
    # ---- optional feature modalities ----
    # The public release keeps all four modalities configurable:
    # sequence embedding, structure-derived supervision, PSSM, and DSSP.
    use_pssm: bool = True
    use_dssp: bool = True
    dips_skip_filter: bool = False

    # ---- RBP dataset placeholders ----
    rbp_root: str = "/path/to/rbp_dataset"
    rbp_id_list: str = "/path/to/rbp_dataset/ids.txt"

    # protein-level split
    split_train: float = 0.70
    split_val:   float = 0.15
    split_test:  float = 0.15
    split_seed:  int   = 42

    # ---- optional legacy DIPS placeholders ----
    dips_root: str = "/path/to/dips_root"
    dips_train_list: str = "/path/to/dips_root/list/train.txt"
    dips_val_list: str = "/path/to/dips_root/list/val.txt"

    esm_local_dir: str = "/path/to/resources/esm"
    save_dir: str = "./runs_rbp296"

    # ---- prediction / analysis paths ----
    dips_pred_list: str = "/path/to/dips_root/list/test.txt"
    pred_result_root: str = "./result"
    pred_ckpt_path: str = "./runs_rbp296/checkpoints/best_TOPK.pt"

    # Structure-derived supervision distance cutoff.
    contact_cutoff: float = 8.0

    # ---- optimization basics ----
    seed: int = 1337
    epochs: int = 80
    batch_site: int = 4
    num_workers: int = 4

    lr: float = 2.0e-5
    weight_decay: float = 2e-4
    max_grad_norm: float = 0.5

    accum_steps: int = 1
    print_every: int = 25
    warmup_print_steps: int = 50
    sanity_n: int = 0

    # ---- stability ----
    loss_ema_decay: float = 0.95
    loss_spike_threshold: float = 3.5
    loss_spike_scale: float = 0.5

    # ---- loss clamp / adaptive clamp ----
    max_2d_loss: float = 3.0
    max_res_loss: float = 2.0
    max_support_loss: float = 8.0
    max_cons_loss: float = 15.0
    max_frag_loss: float = 15.0

    enable_adaptive_clamp: bool = True  # clamp
    adaptive_clamp_hist: int = 128
    adaptive_clamp_min_hist: int = 32
    adaptive_clamp_std_mul: float = 3.0

    loss_cap_mode: str = "soft"  # none|soft|hard
    warn_large_loss: bool = True
    warn_large_loss_mul: float = 2.5
    max_list_loss: float = 10.0  # listwise
    max_rank_loss: float = 10.0  # pairwise
    max_poly_loss: float = 60.0

    # ---- label smoothing ----
    label_smoothing_2d: float = 0.05
    label_smoothing_l1: float = 0.02

    # ---- progressive loss multipliers ----
    prog_stage1_end: int = 15
    prog_stage2_end: int = 40
    prog_w2d_s1: float = 0.8
    prog_w2d_s2: float = 1.0
    prog_w2d_s3: float = 1.2
    prog_wcons_s1: float = 0.5
    prog_wcons_s2: float = 1.0
    prog_wcons_s3: float = 2.0

    # ---- staged sampler knobs ----
    samp_stage1_end: int = 15
    samp_stage2_end: int = 40
    samp_p_target_s1: float = 0.75
    samp_p_target_s2: float = 0.70
    samp_p_target_s3: float = 0.65
    samp_base_w_pos_s1: float = 1.00
    samp_base_w_neg_s1: float = 1.00
    samp_base_w_pos_s2: float = 1.30
    samp_base_w_neg_s2: float = 0.75
    samp_base_w_pos_s3: float = 1.60
    samp_base_w_neg_s3: float = 0.55
    samp_pos_oversample_s1: float = 6.0
    samp_rate_thr: float = 0.001
    samp_pos_oversample_s2: float = 3.5
    samp_pos_oversample_s3: float = 2.0

    # ---- crop/mining ----
    max_2d_tokens_train: int = 160000
    max_2d_tokens_eval: int = 1200000
    min_side: int = 64
    max_side: int = 384
    l2_anchor_prob: float = 0.90
    l2_fixed_crop: bool = True
    l2_fixed_side: int = 192
    l2_reject_allneg_p: float = 0.8
    l2_reject_allneg_max_tries: int = 12
    l2_force_anchor_if_haspos: bool = True  # os-aware crop
    l2_hardneg_frac: float = 0.95

    # ---- model-side priors ----
    prior_pos_pix: float = 0.01
    explain_topk_k: int = 1024
    l2_topk_frac: float = 0.05
    l2_topk_k: int = 256

    # ---- evaluation toggles ----
    eval_topk: bool = False
    eval_thr_mode: str = "auto_mcc"  # auto_mcc|auto_f1|fixed
    eval_thr_grid: int = 101

    # ====================================================================
    # 80G,
    # ====================================================================
    small_tr_n: int = 2000
    small_va_n: int = 200

    # ---- sampler ----
    sampler_p_target: float = 0.75
    pos_oversample: float = 1.0

    # ====================================================================
    # ====================================================================
    lr_sched: bool = True
    lr_warmup_frac: float = 0.10
    lr_warmup_start_ratio: float = 0.0
    lr_warmup_min_updates: int = 20
    lr_warmup_max_updates: int = 300
    lr_min_ratio: float = 0.30
    lr_cosine_cycles: int = 3
    steps_per_epoch: Optional[int] = None

    # ---- EMA & Early Stopping ----
    ema_decay: float = 0.990
    early_min_epochs: int = 22
    early_patience: int = 6
    early_min_delta: float = 2e-4
    early_ema: float = 0.6

    top5_guard_window: int = 0
    top5_guard_drop: float = 0.0

    # ---- Top-K checkpoint selection ----
    # Use a composite validation score that matches the final reported table:
    # Recall@L/5, Recall@L/10, Precision@K10, Hit%@K20.
    topk_score_w_l5: float = 0.35
    topk_score_w_l10: float = 0.35
    topk_score_w_k10p: float = 0.20
    topk_score_w_hit20: float = 0.10

    # ---- eval ----
    eval_skip_allneg: bool = True
    eval_tta_swap: bool = True

    # ====================================================================
    # ====================================================================
    # oss
    l2_w: float = 0.0
    l1_w: float = 0.6
    l1_pos_weight: float = 4.0

    # ---- L1 Focal Loss (trick: UPRC) ----
    # : l1_focal_w=0.5, l1_focal_gamma=2.0, l1_focal_alpha=0.25
    l1_focal_w: float = 0.5
    l1_focal_gamma: float = 2.0
    l1_focal_alpha: float = 0.25  # 

    l1_asl_w: float = 0.0         # ASL 
    l1_asl_gamma_neg: float = 0.1

    # ---- L1 Residue-level Rank Loss (NEW: TopK) ----
    l1_rank_w: float = 0.35
    l1_rank_start_epoch: int = 3
    l1_rank_margin: float = 1.0
    l1_rank_n_pairs: int = 256          # pair
    l1_rank_neg_hard_frac: float = 0.7  # 70%

    # ---- L1 List Loss v2 (Softplus+ApproxNDCG ) ----
    l1_list_w: float = 0.25
    l1_list_start_epoch: int = 5
    l1_list_tau: float = 2.0
    l1_list_neg_k: int = 64

    # ---- L1.5 Support Loss ----
    l12_support_w: float = 0.05
    l12_support_start_epoch: int = 12
    l12_support_ramp_epochs: int = 10
    l12_support_tau: float = 0.5
    l12_support_margin: float = 0.6

    # ---- L1.5 Consistency Loss ----
    # loss
    l12_cons_w: float = 0.0
    l12_cons_start_epoch: int = 999  # RBP296: cons_w=0
    l12_cons_ramp_epochs: int = 10
    l12_cons_margin: float = 0.8
    l12_cons_tau: float = 1.0
                                     # tau=1.0  sigmoid

    # ---- L1.5 Fragment Loss ----
    l15_w: float = 0.08

    # ---- L2 mining ----
    # [v4]  hard mining
    l2_neg_per_pos: int = 5
    l2_neg_min: int = 128
    l2_neg_cap: int = 3072
    l2_bg_hard_k: int = 512
    l2_bg_w: float = 0.15
    l2_allneg_drop_p: float = 0.8

    # ---- L2 focus epochs ----
    l2_focus_epochs: int = 8
    l2_focus_rankw: float = 1.8
    l2_focus_listw: float = 1.5
    l2_focus_margin: float = 0.8

    # ---- L2 pairwise ranking ----
    # [v4]  ranking
    l2_rank_w_max: float = 0.15
    l2_rank_start_epoch: int = 999
    l2_rank_ramp_epochs: int = 15
    l2_rank_margin: float = 1.0
    l2_rank_neg_k: int = 256
    l2_rank_pos_k: int = 128

    # ---- L2 listwise ranking ----
    l2_listwise_w: float = 0.12
    l2_list_start_epoch: int = 999
    l2_list_ramp_epochs: int = 12
    l2_listwise_tau: float = 0.6
    l2_listwise_tau_min: float = 0.3
    l2_listwise_anneal_epochs: int = 25
    l2_listwise_neg_k: int = 512

    # ---- L2 BCE component () ----
    l2_bce_w_max: float = 0.25
    l2_bce_start_epoch: int = 0
    l2_bce_ramp_epochs: int = 5
    l2_bce_hold_epochs: int = 25
    l2_bce_decay: float = 0.99

    # ---- L2 poly-loss () ----
    l2_poly_w: float = 0.10
    l2_poly_k: int = 256
    l2_poly_pow: float = 1.8
    l2_poly_cap: float = 15.0


    # ---- extra aux losses (optional; default off) ----
    l2_focal_w: float = 0.1
    l2_focal_alpha: float = 0.25
    l2_focal_gamma: float = 2.0

    l2_dice_w: float = 0.1
    l2_dice_smooth: float = 1.0

    l2_calibration_w: float = 0.0
    l2_calibration_bins: int = 15

    l2_contrast_w: float = 0.0
    l2_contrast_temp: float = 0.5
    l2_contrast_margin: float = 1.0
    l2_contrast_hard_neg: bool = True

    # ---- L2 logit regularization ----
    l2_logit_reg_w: float = 0.02
    l2_logit_reg_cap: float = 8.0

    # ---- criterion (OHEM & RGCR) ----
    rgcr_w_pos: float = 2.5  # lass imbalance
    rgcr_w_neg: float = 0.9
    ohem_min_keep: int = 6144
    ohem_no_pos_keep: int = 1536
    ohem_ratio: float = 4.0
    l2_pair_neg_k: int = 1200

    # ---- validation thresholds ----
    val_thr_mode: str = "auto_mcc"
    val_thr: float = 0.5
    val_thr_grid: int = 121
    val_thr_min: float = 0.05
    val_thr_max: float = 0.95

    # ---- safety ----
    bad_loss_thr: float = 1e5
    dips_index_verbose: int = 1

    @classmethod
    def from_env(cls, p=None):
        """Load runtime overrides from environment variables."""
        if p is None:
            p = cls()

        # 
        p.seed = _env_int("SEED", p.seed)

        # RBP296  (env override)
        _po = os.environ.get("PRIMARY_OBJECTIVE", None)
        if _po is not None:
            p.primary_objective = str(_po).strip().lower()
        _rbp = os.environ.get("RBP_ROOT", None)
        if _rbp:
            p.rbp_root = str(_rbp)
        _rid = os.environ.get("RBP_ID_LIST", None)
        if _rid:
            p.rbp_id_list = str(_rid)
        p.split_train = _env_float("SPLIT_TRAIN", p.split_train)
        p.split_val   = _env_float("SPLIT_VAL",   p.split_val)
        p.split_test  = _env_float("SPLIT_TEST",  p.split_test)
        p.split_seed  = _env_int("SPLIT_SEED",    p.split_seed)
        p.epochs = _env_int("EPOCHS", p.epochs)
        p.early_patience = _env_int("EARLY_PATIENCE", p.early_patience)
        p.early_min_epochs = _env_int("EARLY_MIN_EPOCHS", p.early_min_epochs)
        p.early_min_delta = _env_float("EARLY_MIN_DELTA", p.early_min_delta)
        p.batch_site = _env_int("BATCH_SITE", p.batch_site)
        p.num_workers = _env_int("NUM_WORKERS", p.num_workers)
        p.lr = _env_float("LR", p.lr)
        p.weight_decay = _env_float("WEIGHT_DECAY", p.weight_decay)
        p.max_grad_norm = _env_float("MAX_GRAD_NORM", p.max_grad_norm)
        p.accum_steps = _env_int("ACCUM_STEPS", p.accum_steps)
        p.dips_skip_filter = _env_bool("DIPS_SKIP_FILTER", p.dips_skip_filter)
        p.dips_pred_list = os.environ.get("DIPS_PRED_LIST", getattr(p, "dips_pred_list", p.dips_val_list))
        p.pred_result_root = os.environ.get("PRED_RESULT_ROOT", getattr(p, "pred_result_root", ""))
        p.pred_ckpt_path = os.environ.get("PRED_CKPT_PATH", getattr(p, "pred_ckpt_path", ""))
        p.small_tr_n = _env_int("SMALL_TR_N", p.small_tr_n)
        p.small_va_n = _env_int("SMALL_VA_N", p.small_va_n)

        # contact cutoff
        p.contact_cutoff = _env_float("CONTACT_CUTOFF", p.contact_cutoff)

        # loss weights
        p.l2_w = _env_float("L2_W", p.l2_w)
        p.l1_w = _env_float("L1_W", p.l1_w)
        p.l1_pos_weight = _env_float("L1_POS_WEIGHT", p.l1_pos_weight)
        p.l1_focal_w = _env_float("L1_FOCAL_W", p.l1_focal_w)
        p.l1_focal_gamma = _env_float("L1_FOCAL_GAMMA", p.l1_focal_gamma)
        p.l1_focal_alpha = _env_float("L1_FOCAL_ALPHA", p.l1_focal_alpha)
        p.l1_asl_w = _env_float("L1_ASL_W", p.l1_asl_w)
        p.l1_asl_gamma_neg = _env_float("L1_ASL_GAMMA_NEG", p.l1_asl_gamma_neg)
        p.l1_rank_w = _env_float("L1_RANK_W", p.l1_rank_w)
        p.l1_rank_start_epoch = _env_int("L1_RANK_START_EPOCH", p.l1_rank_start_epoch)
        p.l1_rank_margin = _env_float("L1_RANK_MARGIN", p.l1_rank_margin)
        p.l1_rank_n_pairs = _env_int("L1_RANK_N_PAIRS", p.l1_rank_n_pairs)
        p.l1_rank_neg_hard_frac = _env_float("L1_RANK_NEG_HARD_FRAC", p.l1_rank_neg_hard_frac)
        p.l1_list_w = _env_float("L1_LIST_W", p.l1_list_w)
        p.l1_list_start_epoch = _env_int("L1_LIST_START_EPOCH", p.l1_list_start_epoch)
        p.l1_list_tau = _env_float("L1_LIST_TAU", p.l1_list_tau)
        p.l1_list_neg_k = _env_int("L1_LIST_NEG_K", p.l1_list_neg_k)
        p.topk_score_w_l5 = _env_float("TOPK_SCORE_W_L5", p.topk_score_w_l5)
        p.topk_score_w_l10 = _env_float("TOPK_SCORE_W_L10", p.topk_score_w_l10)
        p.topk_score_w_k10p = _env_float("TOPK_SCORE_W_K10P", p.topk_score_w_k10p)
        p.topk_score_w_hit20 = _env_float("TOPK_SCORE_W_HIT20", p.topk_score_w_hit20)

        # l12 support
        p.l12_support_w = _env_float("L12_SUPPORT_W", p.l12_support_w)
        p.l12_support_start_epoch = _env_int("L12_SUPPORT_START_EPOCH", p.l12_support_start_epoch)
        p.l12_support_ramp_epochs = _env_int("L12_SUPPORT_RAMP_EPOCHS", p.l12_support_ramp_epochs)
        p.l12_support_tau = _env_float("L12_SUPPORT_TAU", p.l12_support_tau)
        p.l12_support_margin = _env_float("L12_SUPPORT_MARGIN", p.l12_support_margin)
        p.l15_w = _env_float("L15_W", p.l15_w)

        # L2 mining
        p.l2_neg_per_pos = _env_int("L2_NEG_PER_POS", p.l2_neg_per_pos)
        p.l2_neg_min = _env_int("L2_NEG_MIN", p.l2_neg_min)
        p.l2_neg_cap = _env_int("L2_NEG_CAP", p.l2_neg_cap)

        # focus
        p.l2_focus_epochs = _env_int("L2_FOCUS_EPOCHS", p.l2_focus_epochs)

        # verbosity
        p.dips_index_verbose = _env_int("DIPS_INDEX_VERBOSE", p.dips_index_verbose)

        # validation threshold
        p.val_thr_mode = os.environ.get("VAL_THR_MODE", p.val_thr_mode)
        p.val_thr = _env_float("VAL_THR", p.val_thr)
        p.val_thr_grid = _env_int("VAL_THR_GRID", p.val_thr_grid)
        p.val_thr_min = _env_float("VAL_THR_MIN", p.val_thr_min)
        p.val_thr_max = _env_float("VAL_THR_MAX", p.val_thr_max)

        # criterion knobs
        p.rgcr_w_pos = _env_float("RGCR_W_POS", p.rgcr_w_pos)
        p.rgcr_w_neg = _env_float("RGCR_W_NEG", p.rgcr_w_neg)
        p.ohem_min_keep = _env_int("OHEM_MIN_KEEP", p.ohem_min_keep)
        p.ohem_no_pos_keep = _env_int("OHEM_NO_POS_KEEP", p.ohem_no_pos_keep)
        p.ohem_ratio = _env_float("OHEM_RATIO", p.ohem_ratio)
        p.l2_pair_neg_k = _env_int("L2_PAIR_NEG_K", p.l2_pair_neg_k)
        p.l12_cons_w = _env_float("L12_CONS_W", p.l12_cons_w)
        p.l12_cons_start_epoch = _env_int("L12_CONS_START_EPOCH", p.l12_cons_start_epoch)
        p.l12_cons_ramp_epochs = _env_int("L12_CONS_RAMP_EPOCHS", p.l12_cons_ramp_epochs)

        # L2 components
        p.l2_rank_w_max = _env_float("L2_RANK_W_MAX", p.l2_rank_w_max)
        p.l2_rank_start_epoch = _env_int("L2_RANK_START_EPOCH", p.l2_rank_start_epoch)
        p.l2_rank_ramp_epochs = _env_int("L2_RANK_RAMP_EPOCHS", p.l2_rank_ramp_epochs)
        p.l2_rank_margin = _env_float("L2_RANK_MARGIN", p.l2_rank_margin)
        p.l2_rank_neg_k = _env_int("L2_RANK_NEG_K", p.l2_rank_neg_k)
        p.l2_rank_pos_k = _env_int("L2_RANK_POS_K", p.l2_rank_pos_k)

        p.l2_listwise_w = _env_float("L2_LISTWISE_W", p.l2_listwise_w)
        p.l2_list_start_epoch = _env_int("L2_LIST_START_EPOCH", p.l2_list_start_epoch)
        p.l2_list_ramp_epochs = _env_int("L2_LIST_RAMP_EPOCHS", p.l2_list_ramp_epochs)
        p.l2_listwise_tau = _env_float("L2_LISTWISE_TAU", p.l2_listwise_tau)
        p.l2_listwise_tau_min = _env_float("L2_LISTWISE_TAU_MIN", p.l2_listwise_tau_min)
        p.l2_listwise_anneal_epochs = _env_int("L2_LISTWISE_ANNEAL_EPOCHS", p.l2_listwise_anneal_epochs)
        p.l2_listwise_neg_k = _env_int("L2_LISTWISE_NEG_K", p.l2_listwise_neg_k)

        p.l2_bce_w_max = _env_float("L2_BCE_W_MAX", p.l2_bce_w_max)
        p.l2_bce_start_epoch = _env_int("L2_BCE_START_EPOCH", p.l2_bce_start_epoch)
        p.l2_bce_ramp_epochs = _env_int("L2_BCE_RAMP_EPOCHS", p.l2_bce_ramp_epochs)
        p.l2_bce_hold_epochs = _env_int("L2_BCE_HOLD_EPOCHS", p.l2_bce_hold_epochs)
        p.l2_bce_decay = _env_float("L2_BCE_DECAY", p.l2_bce_decay)

        p.l2_poly_w = _env_float("L2_POLY_W", p.l2_poly_w)
        p.l2_poly_k = _env_int("L2_POLY_K", p.l2_poly_k)
        p.l2_poly_pow = _env_float("L2_POLY_POW", p.l2_poly_pow)
        p.l2_poly_cap = _env_float("L2_POLY_CAP", p.l2_poly_cap)

        p.l2_logit_reg_w = _env_float("L2_LOGIT_REG_W", p.l2_logit_reg_w)
        p.l2_logit_reg_cap = _env_float("L2_LOGIT_REG_CAP", p.l2_logit_reg_cap)

        # all-neg
        p.l2_allneg_drop_p = _env_float("L2_ALLNEG_DROP_P", p.l2_allneg_drop_p)
        p.l2_bg_hard_k = _env_int("L2_BG_HARD_K", p.l2_bg_hard_k)
        p.l2_bg_w = _env_float("L2_BG_W", p.l2_bg_w)

        # safety
        p.bad_loss_thr = _env_float("BAD_LOSS_THR", p.bad_loss_thr)
        p.print_every = _env_int("PRINT_EVERY", p.print_every)
        p.warmup_print_steps = _env_int("WARMUP_PRINT_STEPS", p.warmup_print_steps)

        p.min_side = _env_int("MIN_SIDE", p.min_side)
        p.max_side = _env_int("MAX_SIDE", p.max_side)
        p.l2_anchor_prob = _env_float("L2_ANCHOR_PROB", p.l2_anchor_prob)
        p.pos_oversample = _env_float("POS_OVERSAMPLE", p.pos_oversample)

        # loss cap
        p.loss_cap_mode = os.environ.get("LOSS_CAP_MODE", p.loss_cap_mode)
        p.samp_rate_thr = _env_float("SAMP_RATE_THR", p.samp_rate_thr)
        p = apply_topk_preset(p, os.environ.get("TOPK_PRESET", ""))
        return p


@dataclass
class ModelConfig:
    """Model architecture settings for the public release."""

    d_res_in: int = 1309
    d_chain_in: int = 18
    d_model: int = 384
    n_encoder_layers: int = 8
    n_cross_layers: int = 8
    n_heads: int = 8
    dropout: float = 0.20

    # stability
    use_layerscale: bool = True
    layerscale_init: float = 1e-4
    use_swiglu: bool = True
    ffn_mult: float = 4.0

    unet_ch: int = 128

    # priors & loss
    prior_pos_pix: float = 0.05
    focal_gamma: float = 1.0
    cb_beta: float = 0.99
    ohem_ratio: float = 4.0
    ohem_neg_frac: float = 0.5
    neg_per_pos: float = 5.0

    # L2 runtime
    l2_max_len: int = 384
    amp_eval: bool = True
    ohem_start_epoch: int = 5

    # L2 topk
    l2_topk_k: int = 256
    l2_topk_frac: float = 0.05

    # L2 components
    l2_topk_boost_start_epoch: int = 2
    l2_topk_boost_frac: float = 0.05
    l2_topk_boost_w: float = 0.8
    l2_rank_w: float = 0.40
    l2_rank_margin: float = 1.2
    l2_rank_pos_k: int = 128
    l2_rank_neg_k: int = 512

    l2_listwise_temp: float = 0.8
    l2_listwise_neg_k: int = 4096

    l2_multicrop_w: float = 0.25
    l2_multicrop_prob: float = 0.5
    l2_multicrop_until_ep: int = 10
    l2_multicrop_k: int = 1024

    l2_topk_focus_epochs: int = 8
    l2_listwise_tau: float = 0.8
    l2_hard_neg_cap: int = 2048
    l2_listwise_w_focus: float = 1.5
    l2_listwise_w: float = 0.30

    l2_margin: float = 1.2
    l2_margin_w_focus: float = 0.8
    l2_margin_w: float = 0.20
    l2_topk_boost_w_focus: float = 1.5

    # L1.5 support
    l12_support_w: float = 0.15
    l12_support_margin: float = 0.35
    l12_support_start_epoch: int = 3
    l1_pos_weight: float = 6.0

    # criterion
    rgcr_w_pos: float = 2.5
    rgcr_w_neg: float = 0.9
    ohem_min_keep: int = 6144
    ohem_no_pos_keep: int = 1536
    l2_pair_neg_k: int = 1200
    l12_cons_w: float = 0.15

    # future
    use_l3: bool = False
    explain_topk_k: int = 1024

    def get(self, key, default=None):
        return getattr(self, key, default)


def build_model_config(p: Params, d_res_in: int, d_chain_in: int) -> "ModelConfig":
    """aramsModelConfig"""
    cfg = ModelConfig(
        d_res_in=int(d_res_in),
        d_chain_in=int(d_chain_in),
        use_layerscale=_env_bool('USE_LAYERSCALE', True),
        layerscale_init=_env_float('LAYERSCALE_INIT', 1e-4),
        use_swiglu=_env_bool('USE_SWIGLU', True),
        ffn_mult=_env_float('FFN_MULT', 4.0),
        prior_pos_pix=float(p.prior_pos_pix),
        explain_topk_k=int(p.explain_topk_k),

        # sampling
        neg_per_pos=float(p.l2_neg_per_pos),
        l2_hard_neg_cap=int(p.l2_neg_cap),

        # topk
        l2_topk_frac=float(p.l2_topk_frac),
        l2_topk_k=int(p.l2_topk_k),

        # focus
        l2_topk_focus_epochs=int(p.l2_focus_epochs),
        l2_topk_boost_w_focus=float(p.l2_focus_rankw),
        l2_listwise_w_focus=float(p.l2_focus_listw),
        l2_margin_w_focus=float(p.l2_focus_margin),

        # support
        l12_support_w=float(p.l12_support_w),
        l12_support_margin=float(p.l12_support_margin),
        l12_support_start_epoch=int(p.l12_support_start_epoch),
        l1_pos_weight=float(p.l1_pos_weight),
        l2_listwise_tau=float(p.l2_listwise_tau),

        # criterion
        rgcr_w_pos=float(p.rgcr_w_pos),
        rgcr_w_neg=float(p.rgcr_w_neg),
        ohem_min_keep=int(p.ohem_min_keep),
        ohem_no_pos_keep=int(p.ohem_no_pos_keep),
        ohem_ratio=float(p.ohem_ratio),
        l2_pair_neg_k=int(p.l2_pair_neg_k),
        l12_cons_w=float(p.l12_cons_w),

        use_l3=False,
    )
    return cfg

