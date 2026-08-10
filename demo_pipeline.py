import numpy as np
from PIL import Image, ImageFilter

def _to_mask(gray, percentile, blur_radius=1.0):
    img = Image.fromarray(gray.astype(np.uint8))
    if blur_radius > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    a = np.array(img, dtype=np.float32)
    threshold = np.percentile(a, percentile)
    mask = (a > threshold).astype(np.uint8) * 255
    return mask

def _normalize_weights(values):
    v = np.asarray(values, dtype=float)
    v = np.clip(v, 1e-6, None)
    return (v / v.sum()).tolist()

def make_overlay(image, mask):
    base = image.astype(np.float32).copy()
    m = mask > 0
    if m.any():
        # Neutral bright overlay without claiming semantic accuracy.
        overlay = base.copy()
        overlay[m] = 0.55 * overlay[m] + 0.45 * np.array([255, 255, 255], dtype=np.float32)
        base = overlay
    return np.clip(base, 0, 255).astype(np.uint8)

def run_demo_pipeline(image_rgb, declared_gsd="Unknown", source="Unknown"):
    """
    UI DEMONSTRATION ONLY.
    This function does NOT implement the research MAPVNet-Q model.
    Replace its internals with real model inference later.
    """
    image_rgb = np.asarray(image_rgb, dtype=np.uint8)
    gray = (
        0.299 * image_rgb[..., 0]
        + 0.587 * image_rgb[..., 1]
        + 0.114 * image_rgb[..., 2]
    ).astype(np.uint8)

    # Three visually different placeholder predictions.
    mask_08 = _to_mask(gray, 78, blur_radius=2.2)
    mask_03 = _to_mask(gray, 74, blur_radius=1.3)
    mask_01 = _to_mask(gray, 70, blur_radius=0.7)

    # Image-derived demo diagnostics for stable, repeatable behavior.
    g = gray.astype(np.float32) / 255.0
    contrast = float(np.std(g))
    gx = np.abs(np.diff(g, axis=1)).mean() if g.shape[1] > 1 else 0.0
    gy = np.abs(np.diff(g, axis=0)).mean() if g.shape[0] > 1 else 0.0
    edge = float((gx + gy) / 2.0)

    entropy = float(np.clip(0.62 - 1.4 * contrast + 0.5 * edge, 0.08, 0.88))
    tta = float(np.clip(0.10 + 0.9 * edge, 0.05, 0.72))
    boundary = float(np.clip(0.12 + 1.1 * edge, 0.05, 0.80))
    pred_iou = float(np.clip(0.94 - 0.45 * entropy - 0.25 * boundary, 0.35, 0.96))

    quality_score = float(np.clip(
        0.45 * pred_iou
        + 0.20 * (1 - entropy)
        + 0.18 * (1 - tta)
        + 0.17 * (1 - boundary),
        0,
        1
    ))

    # Quality-conditioned placeholder weights.
    agent_quality = [
        np.clip(0.62 + 0.20 * (1 - entropy), 0.1, 1),
        np.clip(0.70 + 0.20 * (1 - boundary), 0.1, 1),
        np.clip(0.76 + 0.18 * pred_iou, 0.1, 1),
    ]
    weights = _normalize_weights(agent_quality)

    stack = np.stack([mask_08 / 255.0, mask_03 / 255.0, mask_01 / 255.0], axis=0)
    fused_prob = np.tensordot(np.asarray(weights), stack, axes=(0, 0))
    final_mask = (fused_prob >= 0.5).astype(np.uint8) * 255

    # Four-action demonstration policy.
    if quality_score >= 0.78:
        action = "ACCEPT"
        reason = "Demo quality threshold satisfied."
        route = "Fusion → instance extraction"
        retry_count = 0
    elif quality_score >= 0.62:
        action = "RETRY"
        reason = "Moderate uncertainty detected."
        route = "Repeat inference with enhanced preprocessing"
        retry_count = 1
    elif quality_score >= 0.45:
        action = "RE-ROUTE"
        reason = "Low quality; specialist route selected."
        route = "0.1 m specialist agent"
        retry_count = 1
    else:
        action = "ESCALATE"
        reason = "Quality remains below safe demo threshold."
        route = "Human / offline research review"
        retry_count = 3

    positive_fraction = float((final_mask > 0).mean())
    # These values are intentionally marked as demo approximations.
    approximate_scene_area = image_rgb.shape[0] * image_rgb.shape[1] * 0.01
    pv_area_m2 = positive_fraction * approximate_scene_area
    estimated_modules = int(max(0, round(pv_area_m2 / 1.8)))
    rejected = int(round(estimated_modules * 0.03))
    valid = max(0, estimated_modules - rejected)

    history = [
        {"Step": 1, "Stage": "Input", "State": "COMPLETE", "Action": "Read image", "Reason": source},
        {"Step": 2, "Stage": "Agents", "State": "COMPLETE", "Action": "3 demo masks", "Reason": declared_gsd},
        {"Step": 3, "Stage": "Quality", "State": "COMPLETE", "Action": f"Score {quality_score:.3f}", "Reason": "4 demo signals"},
        {"Step": 4, "Stage": "Decision", "State": "COMPLETE", "Action": action, "Reason": reason},
        {"Step": 5, "Stage": "Fusion", "State": "COMPLETE", "Action": "Adaptive weighted fusion", "Reason": "demo quality weights"},
        {"Step": 6, "Stage": "Extraction", "State": "COMPLETE", "Action": f"{valid} valid demo instances", "Reason": "approximation"},
    ]

    return {
        "agents": {"0.8m": mask_08, "0.3m": mask_03, "0.1m": mask_01},
        "quality": {
            "entropy": entropy,
            "tta_variance": tta,
            "boundary_disagreement": boundary,
            "pred_iou": pred_iou,
            "quality_score": quality_score,
        },
        "decision": {
            "action": action,
            "reason": reason,
            "route": route,
            "retry_count": retry_count,
            "compute_budget_pct": int(min(100, 36 + 17 * retry_count + round(25 * (1 - quality_score)))),
        },
        "fusion_weights": weights,
        "final_mask": final_mask,
        "instances": {
            "estimated_modules": estimated_modules,
            "valid_modules": valid,
            "rejected_objects": rejected,
            "pv_area_m2": pv_area_m2,
            "mean_module_area_m2": 1.8 if valid > 0 else 0.0,
        },
        "history": history,
    }
