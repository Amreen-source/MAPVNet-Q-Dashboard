import numpy as np
from PIL import Image, ImageFilter


# ============================================================
# MAPVNet-Q HEURISTIC DEMONSTRATION PIPELINE
# ============================================================
# IMPORTANT:
# This is NOT trained MAPVNet-Q inference.
# It is an improved image-processing demonstration designed
# to make the dashboard behavior more PV-oriented until the
# real trained models are connected.
# ============================================================


def _smooth(arr, radius=1.0):
    """Gaussian smoothing for a single-channel float image."""
    img = Image.fromarray(
        np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    )
    img = img.filter(ImageFilter.GaussianBlur(radius=radius))
    return np.asarray(img, dtype=np.float32) / 255.0


def _binary_cleanup(mask, open_size=3, close_size=5):
    """
    Simple morphological cleanup using PIL filters.
    Keeps dependencies light for Streamlit Community Cloud.
    """
    img = Image.fromarray((mask.astype(np.uint8) * 255))

    # Remove small isolated responses.
    if open_size >= 3:
        img = img.filter(ImageFilter.MinFilter(open_size))
        img = img.filter(ImageFilter.MaxFilter(open_size))

    # Fill small gaps.
    if close_size >= 3:
        img = img.filter(ImageFilter.MaxFilter(close_size))
        img = img.filter(ImageFilter.MinFilter(close_size))

    return np.asarray(img) > 127


def _pv_score(image_rgb):
    """
    Produce a PV-oriented heuristic score.

    Typical crystalline PV modules in aerial imagery often have:
      - blue / blue-gray appearance
      - moderate or low brightness
      - B channel >= R channel
      - relatively low extreme saturation

    This is only a visual demo heuristic.
    """

    rgb = image_rgb.astype(np.float32) / 255.0

    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]

    maximum = np.maximum.reduce([r, g, b])
    minimum = np.minimum.reduce([r, g, b])

    brightness = (r + g + b) / 3.0
    saturation = maximum - minimum

    # Blue-gray preference.
    blue_advantage = np.clip(
        (b - r + 0.10) / 0.35,
        0.0,
        1.0
    )

    # Reject very bright white roof structures.
    darkness = np.clip(
        (0.88 - brightness) / 0.55,
        0.0,
        1.0
    )

    # Moderate saturation preference.
    sat_preference = np.clip(
        1.0 - np.abs(saturation - 0.18) / 0.32,
        0.0,
        1.0
    )

    # Blue-gray relationship.
    blue_gray = np.clip(
        1.0 - np.abs(b - g) / 0.35,
        0.0,
        1.0
    )

    score = (
        0.38 * blue_advantage
        + 0.30 * darkness
        + 0.17 * sat_preference
        + 0.15 * blue_gray
    )

    # Explicitly suppress nearly white pixels.
    white_structure = (
        (brightness > 0.78)
        & (saturation < 0.15)
    )

    score[white_structure] *= 0.08

    return np.clip(score, 0.0, 1.0)


def _make_agent_mask(score, threshold, blur_radius):
    """
    Generate one heuristic agent prediction.
    Different thresholds simulate agents operating at
    different spatial-resolution sensitivities.
    """

    smoothed = _smooth(score, blur_radius)

    mask = smoothed >= threshold

    mask = _binary_cleanup(
        mask,
        open_size=3,
        close_size=5
    )

    return mask.astype(np.uint8) * 255


def _iou(mask_a, mask_b):
    """Binary IoU used only for demo agent-agreement estimation."""

    a = mask_a > 0
    b = mask_b > 0

    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()

    if union == 0:
        return 1.0

    return float(intersection / union)


def _normalize_weights(values):
    values = np.asarray(values, dtype=np.float32)
    values = np.clip(values, 1e-6, None)

    return (values / values.sum()).tolist()


def make_overlay(image, mask):
    """
    Produce a blue-tinted visualization of the heuristic mask.
    """

    base = image.astype(np.float32).copy()

    m = mask > 0

    if m.any():

        overlay_color = np.array(
            [40, 170, 255],
            dtype=np.float32
        )

        base[m] = (
            0.60 * base[m]
            + 0.40 * overlay_color
        )

    return np.clip(
        base,
        0,
        255
    ).astype(np.uint8)


def run_demo_pipeline(
    image_rgb,
    declared_gsd="Unknown",
    source="Unknown"
):

    """
    MAPVNet-Q HEURISTIC DEMONSTRATION.

    This function DOES NOT perform inference using trained
    MAPVNet-Q models.

    It provides PV-oriented visual behavior until the actual
    multi-resolution segmentation agents are connected.
    """

    image_rgb = np.asarray(
        image_rgb,
        dtype=np.uint8
    )

    # --------------------------------------------------------
    # 1. PV-oriented heuristic probability score
    # --------------------------------------------------------

    score = _pv_score(image_rgb)

    # --------------------------------------------------------
    # 2. Three simulated spatial-resolution agents
    # --------------------------------------------------------

    # Coarse agent:
    # smoother and more conservative.
    mask_08 = _make_agent_mask(
        score,
        threshold=0.58,
        blur_radius=2.2
    )

    # Medium-resolution agent.
    mask_03 = _make_agent_mask(
        score,
        threshold=0.54,
        blur_radius=1.3
    )

    # Fine-resolution agent:
    # preserves more panel detail.
    mask_01 = _make_agent_mask(
        score,
        threshold=0.50,
        blur_radius=0.7
    )

    # --------------------------------------------------------
    # 3. Agent agreement
    # --------------------------------------------------------

    iou_08_03 = _iou(mask_08, mask_03)
    iou_03_01 = _iou(mask_03, mask_01)
    iou_08_01 = _iou(mask_08, mask_01)

    agreement = float(
        np.mean([
            iou_08_03,
            iou_03_01,
            iou_08_01
        ])
    )

    # --------------------------------------------------------
    # 4. Demo uncertainty / quality indicators
    # --------------------------------------------------------

    entropy = float(
        np.clip(
            0.55 * (1.0 - agreement) + 0.10,
            0.05,
            0.85
        )
    )

    tta_variance = float(
        np.clip(
            0.42 * (1.0 - iou_03_01) + 0.05,
            0.03,
            0.70
        )
    )

    boundary_disagreement = float(
        np.clip(
            0.48 * (1.0 - iou_08_01) + 0.05,
            0.03,
            0.75
        )
    )

    # This is a DEMO proxy, not true predicted IoU.
    pred_iou = float(
        np.clip(
            0.55 + 0.40 * agreement,
            0.40,
            0.95
        )
    )

    quality_score = float(
        np.clip(
            0.50 * pred_iou
            + 0.20 * (1.0 - entropy)
            + 0.15 * (1.0 - tta_variance)
            + 0.15 * (1.0 - boundary_disagreement),
            0.0,
            1.0
        )
    )

    # --------------------------------------------------------
    # 5. Adaptive demo fusion
    # --------------------------------------------------------

    agent_quality = [
        0.72 + 0.18 * iou_08_03,
        0.78 + 0.18 * agreement,
        0.82 + 0.16 * iou_03_01
    ]

    weights = _normalize_weights(
        agent_quality
    )

    stack = np.stack(
        [
            mask_08 / 255.0,
            mask_03 / 255.0,
            mask_01 / 255.0
        ],
        axis=0
    )

    fused_probability = np.tensordot(
        np.asarray(weights),
        stack,
        axes=(0, 0)
    )

    # Require majority-like agreement.
    final_mask = (
        fused_probability >= 0.56
    ).astype(np.uint8) * 255

    final_mask = (
        _binary_cleanup(
            final_mask > 0,
            open_size=3,
            close_size=5
        ).astype(np.uint8)
        * 255
    )

    # --------------------------------------------------------
    # 6. Four-action demonstration decision policy
    # --------------------------------------------------------

    if quality_score >= 0.78:

        action = "ACCEPT"

        reason = (
            "High heuristic agreement among "
            "multi-resolution agents."
        )

        route = (
            "Fusion → instance extraction"
        )

        retry_count = 0

    elif quality_score >= 0.64:

        action = "RETRY"

        reason = (
            "Moderate disagreement detected "
            "between heuristic agents."
        )

        route = (
            "Repeat inference with enhanced preprocessing"
        )

        retry_count = 1

    elif quality_score >= 0.48:

        action = "RE-ROUTE"

        reason = (
            "Low cross-agent agreement; "
            "fine-resolution route selected."
        )

        route = (
            "0.1 m specialist agent"
        )

        retry_count = 1

    else:

        action = "ESCALATE"

        reason = (
            "Heuristic confidence remains below "
            "the demonstration threshold."
        )

        route = (
            "Manual / offline research review"
        )

        retry_count = 3

    # --------------------------------------------------------
    # 7. Demo area / module approximation
    # --------------------------------------------------------

    positive_fraction = float(
        (final_mask > 0).mean()
    )

    # Area is only calculated when GSD is explicitly known.
    gsd_lookup = {
        "0.8 m": 0.8,
        "0.3 m": 0.3,
        "0.1 m": 0.1
    }

    if declared_gsd in gsd_lookup:

        gsd = gsd_lookup[declared_gsd]

        pixel_area = gsd ** 2

        pv_area_m2 = float(
            (final_mask > 0).sum()
            * pixel_area
        )

        # Approximation for visualization only.
        estimated_modules = int(
            max(
                0,
                round(
                    pv_area_m2 / 1.8
                )
            )
        )

    else:

        pv_area_m2 = 0.0
        estimated_modules = 0

    rejected = int(
        round(
            estimated_modules * 0.03
        )
    )

    valid = max(
        0,
        estimated_modules - rejected
    )

    # --------------------------------------------------------
    # 8. Decision history
    # --------------------------------------------------------

    history = [

        {
            "Step": 1,
            "Stage": "Input",
            "State": "COMPLETE",
            "Action": "Read image",
            "Reason": source
        },

        {
            "Step": 2,
            "Stage": "Agents",
            "State": "COMPLETE",
            "Action": "3 heuristic demo masks",
            "Reason": declared_gsd
        },

        {
            "Step": 3,
            "Stage": "Quality",
            "State": "COMPLETE",
            "Action": f"Score {quality_score:.3f}",
            "Reason": (
                f"Cross-agent agreement "
                f"{agreement:.3f}"
            )
        },

        {
            "Step": 4,
            "Stage": "Decision",
            "State": "COMPLETE",
            "Action": action,
            "Reason": reason
        },

        {
            "Step": 5,
            "Stage": "Fusion",
            "State": "COMPLETE",
            "Action": "Agreement-weighted fusion",
            "Reason": "heuristic demo"
        },

        {
            "Step": 6,
            "Stage": "Extraction",
            "State": "COMPLETE",
            "Action": (
                f"{valid} demo module equivalents"
                if declared_gsd in gsd_lookup
                else "Area unavailable"
            ),
            "Reason": (
                "approximation"
                if declared_gsd in gsd_lookup
                else "GSD not declared"
            )
        }
    ]

    # --------------------------------------------------------
    # 9. Return dashboard-compatible output
    # --------------------------------------------------------

    return {

        "agents": {
            "0.8m": mask_08,
            "0.3m": mask_03,
            "0.1m": mask_01
        },

        "quality": {

            "entropy": entropy,

            "tta_variance":
                tta_variance,

            "boundary_disagreement":
                boundary_disagreement,

            "pred_iou":
                pred_iou,

            "quality_score":
                quality_score
        },

        "decision": {

            "action":
                action,

            "reason":
                reason,

            "route":
                route,

            "retry_count":
                retry_count,

            "compute_budget_pct":
                int(
                    min(
                        100,
                        35
                        + 18 * retry_count
                        + round(
                            20
                            * (
                                1.0
                                - quality_score
                            )
                        )
                    )
                )
        },

        "fusion_weights":
            weights,

        "final_mask":
            final_mask,

        "instances": {

            "estimated_modules":
                estimated_modules,

            "valid_modules":
                valid,

            "rejected_objects":
                rejected,

            "pv_area_m2":
                pv_area_m2,

            "mean_module_area_m2":
                1.8
                if valid > 0
                else 0.0
        },

        "history":
            history
    }
