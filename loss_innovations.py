# -*- coding: utf-8 -*-
"""Auxiliary loss helpers used by the public HEAL-PPI training script."""

import torch
import torch.nn.functional as F


def focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    """Binary focal loss."""
    bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pt = torch.exp(-bce_loss)
    focal_term = (1 - pt) ** gamma
    alpha_t = targets * alpha + (1 - targets) * (1 - alpha)
    loss = alpha_t * focal_term * bce_loss
    return loss.mean()


def dice_loss(logits, targets, smooth=1.0):
    """Binary dice loss."""
    probs = torch.sigmoid(logits)
    intersection = (probs * targets).sum()
    dice = (2.0 * intersection + smooth) / (probs.sum() + targets.sum() + smooth)
    return 1 - dice


def expected_calibration_error(logits, labels, n_bins=15):
    """Expected calibration error (ECE)."""
    probs = torch.sigmoid(logits)
    confidences = probs
    predictions = (probs > 0.5).float()
    accuracies = (predictions == labels).float()

    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=logits.device)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = torch.zeros(1, device=logits.device)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) * (confidences <= bin_upper)
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece.squeeze()


def integrate_new_losses(loss_2d, l_sel, y_sel, epoch, P):
    """Integrate optional auxiliary losses into the current 2D loss."""
    aux = {}
    total_loss = loss_2d

    n_pos = (y_sel > 0.5).sum().item()
    n_neg = (y_sel < 0.5).sum().item()
    if n_pos > 0:
        aux["pos_ratio"] = n_pos / max(1, (n_pos + n_neg))

    focal_w = getattr(P, "l2_focal_w", 0.0)
    focal_start = getattr(P, "l2_focal_start_epoch", 1)
    if focal_w > 0 and epoch >= focal_start:
        focal_alpha = getattr(P, "l2_focal_alpha", 0.25)
        focal_gamma = getattr(P, "l2_focal_gamma", 2.0)
        loss_focal = focal_loss(l_sel, y_sel, alpha=focal_alpha, gamma=focal_gamma)
        if torch.isfinite(loss_focal):
            total_loss = total_loss + focal_w * loss_focal
            aux["loss_focal"] = loss_focal.item()
        else:
            aux["loss_focal"] = 0.0
            aux["focal_nan"] = True

    dice_w = getattr(P, "l2_dice_w", 0.0)
    dice_start = getattr(P, "l2_dice_start_epoch", 1)
    if dice_w > 0 and epoch >= dice_start:
        dice_smooth = getattr(P, "l2_dice_smooth", 1.0)
        loss_dice = dice_loss(l_sel, y_sel, smooth=dice_smooth)
        if torch.isfinite(loss_dice):
            total_loss = total_loss + dice_w * loss_dice
            aux["loss_dice"] = loss_dice.item()
        else:
            aux["loss_dice"] = 0.0
            aux["dice_nan"] = True

    calib_w = getattr(P, "l2_calibration_w", 0.0)
    calib_start = getattr(P, "l2_calibration_start_epoch", 46)
    if calib_w > 0 and epoch >= calib_start:
        calib_bins = getattr(P, "l2_calibration_bins", 15)
        loss_calib = expected_calibration_error(l_sel, y_sel, n_bins=calib_bins)
        if torch.isfinite(loss_calib):
            total_loss = total_loss + calib_w * loss_calib
            aux["loss_calibration"] = loss_calib.item()
        else:
            aux["loss_calibration"] = 0.0
            aux["calib_nan"] = True

    return total_loss, aux


def format_aux_log(aux):
    """Format auxiliary losses for logging."""
    parts = []
    if "loss_focal" in aux:
        parts.append(f"focal={aux['loss_focal']:.3f}")
    if "loss_dice" in aux:
        parts.append(f"dice={aux['loss_dice']:.3f}")
    if "loss_calibration" in aux:
        parts.append(f"calib={aux['loss_calibration']:.3f}")
    if "pos_ratio" in aux:
        parts.append(f"pos_r={aux['pos_ratio']:.3f}")
    if aux.get("focal_nan", False):
        parts.append("FOCAL_NAN")
    if aux.get("dice_nan", False):
        parts.append("DICE_NAN")
    if aux.get("calib_nan", False):
        parts.append("CALIB_NAN")
    return " | ".join(parts)


if __name__ == "__main__":
    print("loss_innovations.py sanity check")

    N = 1000
    logits = torch.randn(N) * 2
    labels = (torch.rand(N) > 0.9).float()

    print(f"Samples: N={N}, pos={labels.sum().item():.0f}, neg={(1 - labels).sum().item():.0f}")
    print(f"Focal Loss: {focal_loss(logits, labels).item():.4f}")
    print(f"Dice Loss: {dice_loss(logits, labels).item():.4f}")
    print(f"Calibration Loss: {expected_calibration_error(logits, labels).item():.4f}")

    class MockParams:
        l2_focal_w = 0.1
        l2_focal_start_epoch = 1
        l2_focal_alpha = 0.25
        l2_focal_gamma = 2.0
        l2_dice_w = 0.1
        l2_dice_start_epoch = 1
        l2_dice_smooth = 1.0
        l2_calibration_w = 0.0
        l2_calibration_start_epoch = 46
        l2_calibration_bins = 15

    base_loss = torch.tensor(0.5)
    total_loss, aux = integrate_new_losses(base_loss, logits, labels, epoch=1, P=MockParams())
    print(f"Integrated loss: {total_loss.item():.4f}")
    print(format_aux_log(aux))
