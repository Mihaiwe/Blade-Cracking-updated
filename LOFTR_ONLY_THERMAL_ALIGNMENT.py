import cv2
import numpy as np
import os
from pathlib import Path

# Common functions used here are defined in thermal_alignment_common.py.
from thermal_alignment_common import (
    ThermalRawSpec,
    crop_xywh,
    load_dji_raw_temperature,
    load_grayscale_image,
    preprocess_temperature_for_loftr,
    save_heatmap,
    save_overlay,
)

try:
    import certifi

    # Anaconda's Python installation may not expose a usable Windows CA file.
    # Point urllib/Torch at certifi's trusted bundle instead of disabling SSL.
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    import torch
    import kornia.feature as KF
except ImportError as error:
    raise ImportError(
        "LoFTR requires PyTorch, Kornia, and certifi in the active PyCharm environment."
    ) from error


# ============================================================
# USER SETTINGS
# ============================================================

base_folder = Path(r"C:\Users\mihai.dobre\Desktop\Documents Mihai\drone imgs")
output_folder = base_folder / "alignment_outputs_thermal_loftr_0333T_0335T_ccheck"
output_folder.mkdir(parents=True, exist_ok=True)

before_thermal_path = base_folder / "DJI_0333_T.JPG"
after_thermal_path = base_folder / "DJI_0335_T.JPG"

before_raw_path = Path(
    r"C:\Users\mihai.dobre\Downloads\dji_thermal_sdk_v1.8_20250829\utility\bin\windows\release_x64\DJI_0333_T.raw"
)
after_raw_path = Path(
    r"C:\Users\mihai.dobre\Downloads\dji_thermal_sdk_v1.8_20250829\utility\bin\windows\release_x64\DJI_0335_T.raw"
)

ORIGINAL_THERMAL_W = 640
ORIGINAL_THERMAL_H = 512
TEMP_SCALE = 10.0
THERMAL_RAW_SPEC = ThermalRawSpec(
    width=ORIGINAL_THERMAL_W,
    height=ORIGINAL_THERMAL_H,
    temperature_scale=TEMP_SCALE,
)

# Crop the same physical region from the rendered JPG and raw temperature matrix.
# For best results, crop both images (Before and After) the same even if one
# image captures some noise, such as grass, sky, etc.
# Recommended cropping method: open image in Photos (Windows) -> click Edit ->
# Crop your desired amount from the top and note the amount cropped in
# THERMAL_CROP_Y -> crop your desired amount from the bottom -> note the final
# height in BEFORE_THERMAL_CROP_H. Repeat the process for the x axis, where the
# origin is at the left of the image.
BEFORE_THERMAL_CROP_X = 0  # distance from x=0 axis cropped from the left
BEFORE_THERMAL_CROP_Y = 156  # distance from y=0 axis cropped from the top
BEFORE_THERMAL_CROP_W = 640  # total width of the remaining image after crop
BEFORE_THERMAL_CROP_H = 250  # total height of the remaining image after crop

AFTER_THERMAL_CROP_X = 0
AFTER_THERMAL_CROP_Y = 156
AFTER_THERMAL_CROP_W = 640
AFTER_THERMAL_CROP_H = 250

# LoFTR's outdoor weights are generally the better starting point for large objects.
# The weights may be downloaded automatically the first time Kornia initializes LoFTR.
LOFTR_PRETRAINED = "outdoor"
# Set this to a downloaded .ckpt file to avoid automatic network downloading.
LOFTR_CHECKPOINT_PATH = None
LOFTR_CONFIDENCE_THRESHOLD = 0.90  # previously 0.20
MAX_MATCHES_FOR_MODEL = 10000  # previously 5000
MATCH_ROI = None

# Limit repetitive blade structures from supplying nearly all correspondences.
MATCH_GRID_COLUMNS = 10  # previously 8
MATCH_GRID_ROWS = 6  # previously 5
MAX_MATCHES_PER_CELL = 120  # previously 80

# Partial-affine RANSAC allows translation, rotation, and uniform scale only.
RANSAC_REPROJ_THRESHOLD = 2.0  # previously 2.5 pixels
RANSAC_MAX_ITERS = 1000000  # previously 100000
RANSAC_CONFIDENCE = 0.99999  # previously 0.999
RANSAC_REFINE_ITERS = 1000  # previously 200
MIN_MATCHES = 40  # previously 30
MIN_INLIERS = 20  # previously 12
MIN_INLIER_RATIO = 0.3  # previously 0.15

# Sanity limits for the direct after-to-before thermal transform.
MIN_SCALE = 0.85
MAX_SCALE = 1.15
MAX_ROTATION_DEGREES = 10.0
MAX_CENTER_SHIFT = 160.0


def pad_to_multiple(image, multiple=8):
    height, width = image.shape[:2]
    padded_height = int(np.ceil(height / multiple) * multiple)
    padded_width = int(np.ceil(width / multiple) * multiple)
    bottom = padded_height - height
    right = padded_width - width
    padded = cv2.copyMakeBorder(
        image,
        0,
        bottom,
        0,
        right,
        cv2.BORDER_REFLECT_101,
    )
    return padded


def image_to_tensor(image, device):
    tensor = torch.from_numpy(image.astype(np.float32) / 255.0)
    return tensor.unsqueeze(0).unsqueeze(0).to(device)


# ============================================================
# LOFTR MATCHING AND MODEL ESTIMATION
# ============================================================

def filter_points_to_image(points_after, points_before, confidence, shape):
    height, width = shape[:2]
    valid = (
        (confidence >= LOFTR_CONFIDENCE_THRESHOLD)
        & (points_after[:, 0] >= 0)
        & (points_after[:, 0] < width)
        & (points_after[:, 1] >= 0)
        & (points_after[:, 1] < height)
        & (points_before[:, 0] >= 0)
        & (points_before[:, 0] < width)
        & (points_before[:, 1] >= 0)
        & (points_before[:, 1] < height)
    )
    if MATCH_ROI is not None:
        x, y, roi_width, roi_height = MATCH_ROI
        valid &= (
            (points_after[:, 0] >= x)
            & (points_after[:, 0] < x + roi_width)
            & (points_after[:, 1] >= y)
            & (points_after[:, 1] < y + roi_height)
            & (points_before[:, 0] >= x)
            & (points_before[:, 0] < x + roi_width)
            & (points_before[:, 1] >= y)
            & (points_before[:, 1] < y + roi_height)
        )
    return points_after[valid], points_before[valid], confidence[valid]


def balance_matches_spatially(points_after, points_before, confidence, shape):
    height, width = shape[:2]
    order = np.argsort(-confidence)
    cell_counts = {}
    selected = []
    for index in order:
        x, y = points_after[index]
        column = min(MATCH_GRID_COLUMNS - 1, int(x * MATCH_GRID_COLUMNS / width))
        row = min(MATCH_GRID_ROWS - 1, int(y * MATCH_GRID_ROWS / height))
        cell = row, column
        count = cell_counts.get(cell, 0)
        if count >= MAX_MATCHES_PER_CELL:
            continue
        cell_counts[cell] = count + 1
        selected.append(index)
        if len(selected) >= MAX_MATCHES_FOR_MODEL:
            break
    selected = np.asarray(selected, dtype=np.int32)
    return points_after[selected], points_before[selected], confidence[selected]


def create_loftr_matcher(device):
    if LOFTR_CHECKPOINT_PATH is None:
        return KF.LoFTR(pretrained=LOFTR_PRETRAINED).to(device).eval()

    checkpoint_path = Path(LOFTR_CHECKPOINT_PATH)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"LoFTR checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("state_dict", checkpoint)
    matcher = KF.LoFTR(pretrained=None)
    matcher.load_state_dict(state_dict)
    return matcher.to(device).eval()


def run_loftr(after_image, before_image):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("LoFTR device:", device)
    matcher = create_loftr_matcher(device)

    padded_after = pad_to_multiple(after_image)
    padded_before = pad_to_multiple(before_image)
    batch = {
        "image0": image_to_tensor(padded_after, device),
        "image1": image_to_tensor(padded_before, device),
    }
    with torch.inference_mode():
        result = matcher(batch)

    points_after = result["keypoints0"].detach().cpu().numpy().astype(np.float32)
    points_before = result["keypoints1"].detach().cpu().numpy().astype(np.float32)
    confidence = result["confidence"].detach().cpu().numpy().astype(np.float32)
    points_after, points_before, confidence = filter_points_to_image(
        points_after,
        points_before,
        confidence,
        after_image.shape,
    )
    return balance_matches_spatially(
        points_after,
        points_before,
        confidence,
        after_image.shape,
    )


def estimate_partial_affine(points_after, points_before):
    if len(points_after) < MIN_MATCHES:
        raise RuntimeError(f"LoFTR produced only {len(points_after)} usable matches.")
    affine, inlier_mask = cv2.estimateAffinePartial2D(
        points_after,
        points_before,
        method=cv2.RANSAC,
        ransacReprojThreshold=RANSAC_REPROJ_THRESHOLD,
        maxIters=RANSAC_MAX_ITERS,
        confidence=RANSAC_CONFIDENCE,
        refineIters=RANSAC_REFINE_ITERS,
    )
    if affine is None or inlier_mask is None:
        raise RuntimeError("LoFTR partial-affine RANSAC failed.")
    inliers = inlier_mask.reshape(-1).astype(bool)
    inlier_count = int(inliers.sum())
    inlier_ratio = inlier_count / len(points_after)
    if inlier_count < MIN_INLIERS or inlier_ratio < MIN_INLIER_RATIO:
        raise RuntimeError(
            f"Weak LoFTR transform: {inlier_count}/{len(points_after)} inliers "
            f"({inlier_ratio:.3f})"
        )
    transform = np.eye(3, dtype=np.float32)
    transform[:2] = affine.astype(np.float32)
    return transform, inliers, inlier_count, inlier_ratio


def validate_transform(transform, shape):
    scale = float(np.hypot(transform[0, 0], transform[1, 0]))
    rotation = float(np.degrees(np.arctan2(transform[1, 0], transform[0, 0])))
    center = np.float32([[[shape[1] / 2.0, shape[0] / 2.0]]])
    moved_center = cv2.perspectiveTransform(center, transform).reshape(2)
    center_shift = float(np.linalg.norm(moved_center - center.reshape(2)))
    if not (
        MIN_SCALE <= scale <= MAX_SCALE
        and abs(rotation) <= MAX_ROTATION_DEGREES
        and center_shift <= MAX_CENTER_SHIFT
    ):
        raise RuntimeError(
            "LoFTR transform failed sanity limits: "
            f"scale={scale:.4f}, rotation={rotation:.3f}, shift={center_shift:.2f}"
        )
    return {
        "scale": scale,
        "rotation_degrees": rotation,
        "center_shift_pixels": center_shift,
    }


# ============================================================
# OUTPUT HELPERS
# ============================================================

def warp(image, transform, shape, interpolation):
    height, width = shape[:2]
    return cv2.warpPerspective(
        image,
        transform,
        (width, height),
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def save_match_visualization(after_image, before_image, points_after, points_before, inliers):
    height = max(after_image.shape[0], before_image.shape[0])
    width_after = after_image.shape[1]
    canvas = np.zeros((height, width_after + before_image.shape[1], 3), dtype=np.uint8)
    canvas[: after_image.shape[0], :width_after] = cv2.cvtColor(after_image, cv2.COLOR_GRAY2BGR)
    canvas[: before_image.shape[0], width_after:] = cv2.cvtColor(before_image, cv2.COLOR_GRAY2BGR)
    inlier_indices = np.flatnonzero(inliers)
    if len(inlier_indices) > 300:
        inlier_indices = inlier_indices[np.linspace(0, len(inlier_indices) - 1, 300).astype(int)]
    rng = np.random.default_rng(42)
    for index in inlier_indices:
        color = tuple(int(value) for value in rng.integers(40, 256, size=3))
        point_after = tuple(int(value) for value in np.round(points_after[index]))
        shifted_before = np.round(points_before[index]).astype(int) + np.array([width_after, 0])
        point_before = tuple(int(value) for value in shifted_before)
        cv2.circle(canvas, point_after, 2, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, point_before, 2, color, 1, cv2.LINE_AA)
        cv2.line(canvas, point_after, point_before, color, 1, cv2.LINE_AA)
    cv2.imwrite(str(output_folder / "loftr_partial_affine_inlier_matches.png"), canvas)


# ============================================================
# MAIN
# ============================================================

before_crop = (
    BEFORE_THERMAL_CROP_X,
    BEFORE_THERMAL_CROP_Y,
    BEFORE_THERMAL_CROP_W,
    BEFORE_THERMAL_CROP_H,
)
after_crop = (
    AFTER_THERMAL_CROP_X,
    AFTER_THERMAL_CROP_Y,
    AFTER_THERMAL_CROP_W,
    AFTER_THERMAL_CROP_H,
)

before_jpg = crop_xywh(
    load_grayscale_image(before_thermal_path), before_crop, "before thermal JPG"
)
after_jpg = crop_xywh(
    load_grayscale_image(after_thermal_path), after_crop, "after thermal JPG"
)
before_temperature = crop_xywh(
    load_dji_raw_temperature(before_raw_path, THERMAL_RAW_SPEC),
    before_crop,
    "before temperature",
)
after_temperature = crop_xywh(
    load_dji_raw_temperature(after_raw_path, THERMAL_RAW_SPEC),
    after_crop,
    "after temperature",
)
if before_temperature.shape != after_temperature.shape:
    raise ValueError(
        f"Temperature crop shapes differ: before={before_temperature.shape}, "
        f"after={after_temperature.shape}"
    )

target_shape = before_temperature.shape
before_features = preprocess_temperature_for_loftr(before_temperature)
after_features = preprocess_temperature_for_loftr(after_temperature)
cv2.imwrite(str(output_folder / "before_loftr_input.png"), before_features)
cv2.imwrite(str(output_folder / "after_loftr_input.png"), after_features)

points_after, points_before, match_confidence = run_loftr(after_features, before_features)
transform, inliers, inlier_count, inlier_ratio = estimate_partial_affine(
    points_after,
    points_before,
)
geometry = validate_transform(transform, target_shape)
save_match_visualization(after_features, before_features, points_after, points_before, inliers)

# Apply the selected transform once to the original raw temperature crop.
after_temperature_aligned = warp(after_temperature, transform, target_shape, cv2.INTER_LINEAR)
after_jpg_aligned = warp(after_jpg, transform, target_shape, cv2.INTER_LINEAR)
valid_mask = warp(
    np.ones(target_shape, dtype=np.uint8),
    transform,
    target_shape,
    cv2.INTER_NEAREST,
) > 0
delta_temperature = after_temperature_aligned - before_temperature
delta_temperature[~valid_mask] = np.nan

np.savetxt(output_folder / "H_loftr_partial_affine.csv", transform, delimiter=",")
np.save(output_folder / "after_temperature_aligned_C.npy", after_temperature_aligned)
np.save(output_folder / "delta_temperature_after_minus_before_C.npy", delta_temperature)
np.savetxt(
    output_folder / "delta_temperature_after_minus_before_C.csv",
    delta_temperature,
    delimiter=",",
)

save_overlay(
    before_temperature,
    after_temperature_aligned,
    output_folder / "temperature_overlay_after_loftr_alignment.png",
)
save_overlay(
    before_jpg,
    after_jpg_aligned,
    output_folder / "rendered_thermal_overlay_after_loftr_alignment.png",
)
save_heatmap(
    before_temperature,
    output_folder / "before_temperature_crop.png",
    "Before Temperature",
    "Temperature (C)",
)
save_heatmap(
    after_temperature_aligned,
    output_folder / "after_temperature_aligned_loftr.png",
    "After Temperature Aligned By LoFTR + Partial-Affine RANSAC",
    "Temperature (C)",
)
save_heatmap(
    delta_temperature,
    output_folder / "temperature_change_deltaT_loftr.png",
    "Temperature Change: After - Before",
    "Delta T (C)",
    symmetric=True,
)

log = [
    "LoFTR thermal-to-thermal partial-affine alignment",
    f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}",
    f"Pretrained weights: {LOFTR_PRETRAINED}",
    f"Matches after filtering: {len(points_after)}",
    f"Confidence min/mean/max: {match_confidence.min():.4f} / {match_confidence.mean():.4f} / {match_confidence.max():.4f}",
    f"Inliers: {inlier_count}",
    f"Inlier ratio: {inlier_ratio:.4f}",
    f"Geometry: {geometry}",
    f"Before crop: {before_crop}",
    f"After crop: {after_crop}",
    f"Transform after -> before:\n{transform}",
]
(output_folder / "alignment_log.txt").write_text("\n\n".join(log), encoding="utf-8")

print("\nDone.")
print("Outputs saved to:", output_folder)
