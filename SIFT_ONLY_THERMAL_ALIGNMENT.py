import cv2
import numpy as np
from pathlib import Path

# Common functions used here are defined in thermal_alignment_common.py.
from thermal_alignment_common import (
    ThermalRawSpec,
    build_roi_mask,
    crop_xywh,
    load_dji_raw_temperature,
    load_grayscale_image,
    preprocess_temperature_for_sift,
    save_heatmap,
    save_overlay,
)


# ============================================================
# USER SETTINGS
# ============================================================

BASE_FOLDER = Path(r"C:\Users\mihai.dobre\Desktop\Documents Mihai\drone imgs")
OUTPUT_FOLDER = BASE_FOLDER / "alignment_outputs_thermal_sift_0333T_0335T"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

BEFORE_THERMAL_JPG = BASE_FOLDER / "DJI_0333_T.JPG"
AFTER_THERMAL_JPG = BASE_FOLDER / "DJI_0335_T.JPG"
BEFORE_RAW = Path(r"C:\Users\mihai.dobre\Downloads\dji_thermal_sdk_v1.8_20250829\utility\bin\windows\release_x64\DJI_0333_T.raw")
AFTER_RAW = Path(r"C:\Users\mihai.dobre\Downloads\dji_thermal_sdk_v1.8_20250829\utility\bin\windows\release_x64\DJI_0335_T.raw")

FULL_WIDTH = 640
FULL_HEIGHT = 512
TEMPERATURE_SCALE = 10.0
THERMAL_RAW_SPEC = ThermalRawSpec(
    width=FULL_WIDTH,
    height=FULL_HEIGHT,
    temperature_scale=TEMPERATURE_SCALE,
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

# The coarse pass uses most of the thermal image. The fine pass emphasizes the
# band and blade-end structure after coarse prewarping.
COARSE_MATCH_ROI = (0, 0, 640, 250)
FINE_MATCH_ROI = (180, 0, 460, 250)

SIFT_FEATURES = 18000
SIFT_OCTAVE_LAYERS = 5
SIFT_CONTRAST_THRESHOLD = 0.008
SIFT_EDGE_THRESHOLD = 15
SIFT_SIGMA = 1.2
COARSE_LOWE_RATIO = 0.78
FINE_LOWE_RATIO = 0.82

COARSE_RANSAC_THRESHOLD = 3.5
FINE_RANSAC_THRESHOLD = 2.5
RANSAC_MAX_ITERS = 120000
RANSAC_CONFIDENCE = 0.999
RANSAC_REFINE_ITERS = 250
MIN_COARSE_INLIERS = 10
MIN_COARSE_INLIER_RATIO = 0.14
MIN_FINE_INLIERS = 7
MIN_FINE_INLIER_RATIO = 0.10

# Limit repetitive structures from dominating the fit.
MATCH_GRID_COLUMNS = 8
MATCH_GRID_ROWS = 5
MAX_MATCHES_PER_GRID_CELL = 14

# Coarse transform limits. Direct thermal alignment can require a large shift.
COARSE_MIN_SCALE = 0.85
COARSE_MAX_SCALE = 1.15
COARSE_MAX_ROTATION_DEGREES = 10.0
COARSE_MAX_CENTER_SHIFT = 140.0

# Fine residual limits prevent the second pass from undoing a good coarse fit.
FINE_MIN_SCALE = 0.97
FINE_MAX_SCALE = 1.03
FINE_MAX_ROTATION_DEGREES = 1.75
FINE_MAX_CENTER_SHIFT = 25.0
MIN_FINE_SCORE_IMPROVEMENT = 0.005

VALIDATION_BAND_ROI = (270, 0, 370, 250)
VALIDATION_BODY_ROI = (35, 25, 400, 195)


# ============================================================
# SIFT MATCHING
# ============================================================

def create_sift():
    return cv2.SIFT_create(
        nfeatures=SIFT_FEATURES,
        nOctaveLayers=SIFT_OCTAVE_LAYERS,
        contrastThreshold=SIFT_CONTRAST_THRESHOLD,
        edgeThreshold=SIFT_EDGE_THRESHOLD,
        sigma=SIFT_SIGMA,
    )


def ratio_filter(knn_matches, ratio):
    accepted = []
    for pair in knn_matches:
        if len(pair) != 2:
            continue
        first, second = pair
        if first.distance < ratio * second.distance:
            accepted.append(first)
    return accepted


def mutual_ratio_matches(descriptors_source, descriptors_target, ratio):
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    forward = ratio_filter(
        matcher.knnMatch(descriptors_source, descriptors_target, k=2), ratio
    )
    reverse = ratio_filter(
        matcher.knnMatch(descriptors_target, descriptors_source, k=2), ratio
    )
    reverse_lookup = {match.queryIdx: match.trainIdx for match in reverse}
    mutual = [
        match for match in forward
        if reverse_lookup.get(match.trainIdx) == match.queryIdx
    ]
    mutual.sort(key=lambda match: match.distance)
    return mutual


def wrap_angle_degrees(angle):
    return (angle + 180.0) % 360.0 - 180.0


def filter_orientation_and_scale(matches, source_keypoints, target_keypoints):
    if len(matches) < 8:
        return matches
    angle_differences = np.array([
        wrap_angle_degrees(
            target_keypoints[match.trainIdx].angle
            - source_keypoints[match.queryIdx].angle
        )
        for match in matches
    ])
    log_scale_ratios = np.array([
        np.log(
            max(target_keypoints[match.trainIdx].size, 1e-6)
            / max(source_keypoints[match.queryIdx].size, 1e-6)
        )
        for match in matches
    ])

    median_angle = float(np.median(angle_differences))
    median_scale = float(np.median(log_scale_ratios))
    angle_mad = float(np.median(np.abs(angle_differences - median_angle)))
    scale_mad = float(np.median(np.abs(log_scale_ratios - median_scale)))
    angle_limit = max(18.0, 3.5 * angle_mad)
    scale_limit = max(0.30, 3.5 * scale_mad)

    return [
        match for match, angle, scale in zip(matches, angle_differences, log_scale_ratios)
        if abs(angle - median_angle) <= angle_limit
        and abs(scale - median_scale) <= scale_limit
    ]


def balance_matches_spatially(matches, keypoints, shape):
    height, width = shape[:2]
    cells = {}
    selected = []
    for match in matches:
        x, y = keypoints[match.queryIdx].pt
        column = min(MATCH_GRID_COLUMNS - 1, int(x * MATCH_GRID_COLUMNS / width))
        row = min(MATCH_GRID_ROWS - 1, int(y * MATCH_GRID_ROWS / height))
        key = (row, column)
        count = cells.get(key, 0)
        if count >= MAX_MATCHES_PER_GRID_CELL:
            continue
        cells[key] = count + 1
        selected.append(match)
    return selected


def to_homography(affine):
    homography = np.eye(3, dtype=np.float32)
    homography[:2] = affine.astype(np.float32)
    return homography


def estimate_partial_affine(source_points, target_points, threshold):
    affine, inliers = cv2.estimateAffinePartial2D(
        source_points,
        target_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=threshold,
        maxIters=RANSAC_MAX_ITERS,
        confidence=RANSAC_CONFIDENCE,
        refineIters=RANSAC_REFINE_ITERS,
    )
    if affine is None or inliers is None:
        raise RuntimeError("SIFT partial-affine estimation failed.")
    return to_homography(affine), inliers.reshape(-1).astype(bool)


def transform_geometry(transform, shape):
    scale = float(np.hypot(transform[0, 0], transform[1, 0]))
    angle = float(np.degrees(np.arctan2(transform[1, 0], transform[0, 0])))
    center = np.float32([[[shape[1] / 2.0, shape[0] / 2.0]]])
    moved_center = cv2.perspectiveTransform(center, transform).reshape(2)
    center_shift = float(np.linalg.norm(moved_center - center.reshape(2)))
    return scale, angle, center_shift


def validate_transform(transform, shape, stage):
    scale, angle, center_shift = transform_geometry(transform, shape)
    if stage == "coarse":
        valid = (
            COARSE_MIN_SCALE <= scale <= COARSE_MAX_SCALE
            and abs(angle) <= COARSE_MAX_ROTATION_DEGREES
            and center_shift <= COARSE_MAX_CENTER_SHIFT
        )
    else:
        valid = (
            FINE_MIN_SCALE <= scale <= FINE_MAX_SCALE
            and abs(angle) <= FINE_MAX_ROTATION_DEGREES
            and center_shift <= FINE_MAX_CENTER_SHIFT
        )
    geometry = {
        "scale": scale,
        "rotation_degrees": angle,
        "center_shift_pixels": center_shift,
    }
    if not valid:
        raise RuntimeError(f"{stage} SIFT transform rejected: {geometry}")
    return geometry


def draw_matches(source, source_keypoints, target, target_keypoints, matches, inliers, path):
    visualization = cv2.drawMatches(
        source,
        source_keypoints,
        target,
        target_keypoints,
        matches,
        None,
        matchesMask=inliers.astype(np.uint8).tolist(),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    cv2.imwrite(str(path), visualization)


def estimate_sift_transform(
    source,
    target,
    source_mask,
    target_mask,
    lowe_ratio,
    ransac_threshold,
    minimum_inliers,
    minimum_ratio,
    stage,
):
    detector = create_sift()
    source_keypoints, source_descriptors = detector.detectAndCompute(source, source_mask)
    target_keypoints, target_descriptors = detector.detectAndCompute(target, target_mask)
    if source_descriptors is None or target_descriptors is None:
        raise RuntimeError(f"{stage} SIFT did not produce descriptors.")

    matches = mutual_ratio_matches(source_descriptors, target_descriptors, lowe_ratio)
    matches = filter_orientation_and_scale(matches, source_keypoints, target_keypoints)
    matches = balance_matches_spatially(matches, source_keypoints, source.shape)
    if len(matches) < minimum_inliers:
        raise RuntimeError(f"{stage} SIFT found only {len(matches)} filtered matches.")

    source_points = np.float32([source_keypoints[m.queryIdx].pt for m in matches])
    target_points = np.float32([target_keypoints[m.trainIdx].pt for m in matches])
    transform, inliers = estimate_partial_affine(source_points, target_points, ransac_threshold)
    inlier_count = int(inliers.sum())
    inlier_ratio = inlier_count / len(matches)
    if inlier_count < minimum_inliers or inlier_ratio < minimum_ratio:
        raise RuntimeError(
            f"Weak {stage} SIFT transform: {inlier_count}/{len(matches)} inliers"
        )
    geometry = validate_transform(transform, target.shape, stage)
    draw_matches(
        source,
        source_keypoints,
        target,
        target_keypoints,
        matches,
        inliers,
        OUTPUT_FOLDER / f"{stage}_sift_inlier_matches.png",
    )
    return transform, {
        "keypoints_source": len(source_keypoints),
        "keypoints_target": len(target_keypoints),
        "matches": len(matches),
        "inliers": inlier_count,
        "inlier_ratio": inlier_ratio,
        **geometry,
    }


# ============================================================
# WARPING AND VALIDATION
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


def roi_slices(shape, roi):
    height, width = shape[:2]
    x, y, roi_width, roi_height = roi
    x0 = max(0, int(x))
    y0 = max(0, int(y))
    x1 = min(width, x0 + int(roi_width))
    y1 = min(height, y0 + int(roi_height))
    return slice(y0, y1), slice(x0, x1)


def alignment_score(before_temperature, after_temperature, before_features, after_features, transform):
    shape = before_temperature.shape
    aligned_temperature = warp(after_temperature, transform, shape, cv2.INTER_LINEAR)
    aligned_features = warp(after_features, transform, shape, cv2.INTER_LINEAR)
    valid = warp(np.ones(shape, np.uint8), transform, shape, cv2.INTER_NEAREST) > 0

    before_edges = cv2.Canny(before_features, 70, 150)
    after_edges = cv2.Canny(aligned_features, 70, 150)
    edge_distance = cv2.distanceTransform(255 - before_edges, cv2.DIST_L2, 3)
    band_y, band_x = roi_slices(shape, VALIDATION_BAND_ROI)
    band_edge_mask = (after_edges[band_y, band_x] > 0) & valid[band_y, band_x]
    if int(band_edge_mask.sum()) < 30:
        chamfer = 40.0
    else:
        distances = edge_distance[band_y, band_x][band_edge_mask]
        chamfer = float(np.mean(np.clip(distances, 0.0, 20.0)))

    delta = aligned_temperature - before_temperature
    body_y, body_x = roi_slices(shape, VALIDATION_BODY_ROI)
    body_valid = valid[body_y, body_x]
    body_delta = delta[body_y, body_x]
    if int(body_valid.sum()) < 1000:
        smoothness = 40.0
    else:
        gradient_x = cv2.Sobel(body_delta, cv2.CV_32F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(body_delta, cv2.CV_32F, 0, 1, ksize=3)
        gradient = cv2.magnitude(gradient_x, gradient_y)
        smoothness = float(np.median(gradient[body_valid]))

    overlap = float(valid.mean())
    overlap_penalty = 40.0 * max(0.0, 0.90 - overlap)
    score = chamfer + 1.5 * smoothness + overlap_penalty
    return score, {
        "chamfer": chamfer,
        "delta_smoothness": smoothness,
        "overlap": overlap,
    }


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
    load_grayscale_image(BEFORE_THERMAL_JPG), before_crop, "before thermal JPG"
)
after_jpg = crop_xywh(
    load_grayscale_image(AFTER_THERMAL_JPG), after_crop, "after thermal JPG"
)
before_temperature = crop_xywh(
    load_dji_raw_temperature(BEFORE_RAW, THERMAL_RAW_SPEC),
    before_crop,
    "before temperature",
)
after_temperature = crop_xywh(
    load_dji_raw_temperature(AFTER_RAW, THERMAL_RAW_SPEC),
    after_crop,
    "after temperature",
)
if before_temperature.shape != after_temperature.shape:
    raise ValueError(
        f"Temperature crop shapes differ: before={before_temperature.shape}, "
        f"after={after_temperature.shape}"
    )
target_shape = before_temperature.shape

before_features = preprocess_temperature_for_sift(before_temperature)
after_features = preprocess_temperature_for_sift(after_temperature)
cv2.imwrite(str(OUTPUT_FOLDER / "before_sift_features.png"), before_features)
cv2.imwrite(str(OUTPUT_FOLDER / "after_sift_features.png"), after_features)

coarse_mask = build_roi_mask(target_shape, COARSE_MATCH_ROI, label="coarse SIFT ROI")
coarse_transform, coarse_info = estimate_sift_transform(
    after_features,
    before_features,
    coarse_mask,
    coarse_mask,
    COARSE_LOWE_RATIO,
    COARSE_RANSAC_THRESHOLD,
    MIN_COARSE_INLIERS,
    MIN_COARSE_INLIER_RATIO,
    "coarse",
)

after_features_coarse = warp(after_features, coarse_transform, target_shape, cv2.INTER_LINEAR)
coarse_valid = warp(
    np.ones(target_shape, np.uint8) * 255,
    coarse_transform,
    target_shape,
    cv2.INTER_NEAREST,
)
save_overlay(
    before_features,
    after_features_coarse,
    OUTPUT_FOLDER / "sift_features_overlay_after_coarse.png",
)

coarse_score, coarse_score_info = alignment_score(
    before_temperature,
    after_temperature,
    before_features,
    after_features,
    coarse_transform,
)
fine_transform = np.eye(3, dtype=np.float32)
fine_info = {"status": "failed"}
final_transform = coarse_transform.copy()
selection = "coarse SIFT"

try:
    fine_target_mask = build_roi_mask(target_shape, FINE_MATCH_ROI, label="fine SIFT ROI")
    fine_source_mask = build_roi_mask(
        target_shape,
        FINE_MATCH_ROI,
        valid_mask=coarse_valid,
        label="fine SIFT source ROI",
    )
    fine_transform, fine_info = estimate_sift_transform(
        after_features_coarse,
        before_features,
        fine_source_mask,
        fine_target_mask,
        FINE_LOWE_RATIO,
        FINE_RANSAC_THRESHOLD,
        MIN_FINE_INLIERS,
        MIN_FINE_INLIER_RATIO,
        "fine",
    )
    refined_transform = (fine_transform @ coarse_transform).astype(np.float32)
    refined_score, refined_score_info = alignment_score(
        before_temperature,
        after_temperature,
        before_features,
        after_features,
        refined_transform,
    )
    fine_info.update({"status": "valid", "score": refined_score_info})
    required_score = coarse_score * (1.0 - MIN_FINE_SCORE_IMPROVEMENT)
    if refined_score <= required_score:
        final_transform = refined_transform
        selection = "coarse + fine SIFT"
    else:
        fine_info["status"] = "rejected by validation score"
except RuntimeError as error:
    fine_info = {"status": "failed", "error": str(error)}
    print("Fine SIFT refinement failed:", error)

# Apply only the selected composed transform to the original thermal arrays.
after_temperature_aligned = warp(
    after_temperature, final_transform, target_shape, cv2.INTER_LINEAR
)
after_jpg_aligned = warp(after_jpg, final_transform, target_shape, cv2.INTER_LINEAR)
valid_mask = warp(
    np.ones(target_shape, np.uint8), final_transform, target_shape, cv2.INTER_NEAREST
) > 0
delta_temperature = after_temperature_aligned - before_temperature
delta_temperature[~valid_mask] = np.nan

np.savetxt(OUTPUT_FOLDER / "H_sift_coarse.csv", coarse_transform, delimiter=",")
np.savetxt(OUTPUT_FOLDER / "H_sift_fine_residual.csv", fine_transform, delimiter=",")
np.savetxt(OUTPUT_FOLDER / "H_sift_final.csv", final_transform, delimiter=",")
np.save(OUTPUT_FOLDER / "after_temperature_aligned_C.npy", after_temperature_aligned)
np.save(OUTPUT_FOLDER / "delta_temperature_after_minus_before_C.npy", delta_temperature)
np.savetxt(
    OUTPUT_FOLDER / "delta_temperature_after_minus_before_C.csv",
    delta_temperature,
    delimiter=",",
)

save_overlay(
    before_temperature,
    after_temperature_aligned,
    OUTPUT_FOLDER / "temperature_overlay_after_sift_only_alignment.png",
)
save_overlay(
    before_jpg,
    after_jpg_aligned,
    OUTPUT_FOLDER / "rendered_thermal_overlay_after_sift_only_alignment.png",
)
save_heatmap(
    before_temperature,
    OUTPUT_FOLDER / "before_temperature_crop.png",
    "Before Temperature",
    "Temperature (C)",
)
save_heatmap(
    after_temperature_aligned,
    OUTPUT_FOLDER / "after_temperature_aligned_sift_only.png",
    f"After Temperature Aligned: {selection}",
    "Temperature (C)",
)
save_heatmap(
    delta_temperature,
    OUTPUT_FOLDER / "temperature_change_deltaT_sift_only.png",
    "Temperature Change: After - Before",
    "Delta T (C)",
    symmetric=True,
)

log = [
    "SIFT-only thermal-to-thermal alignment",
    f"Selected: {selection}",
    f"Coarse info: {coarse_info}",
    f"Coarse score: {coarse_score_info}",
    f"Fine info: {fine_info}",
    f"Before crop: {before_crop}",
    f"After crop: {after_crop}",
    f"Coarse transform:\n{coarse_transform}",
    f"Fine residual:\n{fine_transform}",
    f"Final transform:\n{final_transform}",
]
(OUTPUT_FOLDER / "alignment_log.txt").write_text("\n\n".join(log), encoding="utf-8")

print("\nSelected:", selection)
print("Outputs saved to:", OUTPUT_FOLDER)
