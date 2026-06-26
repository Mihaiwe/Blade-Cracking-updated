import cv2
import numpy as np
from pathlib import Path

# Common functions used here are defined in thermal_alignment_common.py.
from thermal_alignment_common import (
    ThermalRawSpec,
    build_roi_mask,
    crop_roi,
    crop_xywh,
    load_dji_raw_temperature,
    load_grayscale_image,
    preprocess_temperature_for_orb,
    save_heatmap,
    save_overlay,
)


# ============================================================
# USER SETTINGS
# ============================================================

base_folder = Path(r"C:\Users\mihai.dobre\Desktop\Documents Mihai\drone imgs")

output_folder = base_folder / "alignment_outputs_thermal_orb_0333T_0335T" #adjust
output_folder.mkdir(parents=True, exist_ok=True)

before_thermal_path = base_folder / "DJI_0333_T.JPG" # adjust
after_thermal_path = base_folder / "DJI_0335_T.JPG" # adjust

before_raw_path = Path(r"C:\Users\mihai.dobre\Downloads\dji_thermal_sdk_v1.8_20250829\utility\bin\windows\release_x64\DJI_0333_T.raw") # adjust
after_raw_path = Path(r"C:\Users\mihai.dobre\Downloads\dji_thermal_sdk_v1.8_20250829\utility\bin\windows\release_x64\DJI_0335_T.raw") # adjust

ORIGINAL_THERMAL_W = 640
ORIGINAL_THERMAL_H = 512

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

# DJI int16 raw output is usually temp * 10.
TEMP_SCALE = 10.0
THERMAL_RAW_SPEC = ThermalRawSpec(
    width=ORIGINAL_THERMAL_W,
    height=ORIGINAL_THERMAL_H,
    temperature_scale=TEMP_SCALE,
)

# Optional feature ROI in cropped thermal coordinates.
# None uses the full crop except for a tiny border. This found more stable matches
# than a narrow ROI on this smooth cylindrical target.
MATCH_ROI = None

# More expensive than the first stable version, but still constrained enough to avoid bad warps.
ORB_FEATURES = 45000
ORB_SCALE_FACTOR = 1.05
ORB_LEVELS = 16
ORB_EDGE_THRESHOLD = 6
ORB_PATCH_SIZE = 31
ORB_FAST_THRESHOLD = 3

LOWE_RATIO = 0.80
MAX_MATCHES_FOR_MODEL = 2500

# Keep this conservative. Homography/ECC overfit this smooth cylindrical subject.
TRANSFORM_MODEL = "partial_affine"  # options: "auto", "affine", "homography", "partial_affine"

RANSAC_REPROJ_THRESHOLD = 3.0
RANSAC_MAX_ITERS = 60000
RANSAC_CONFIDENCE = 0.999
RANSAC_REFINE_ITERS = 150

MIN_GOOD_MATCHES = 25
MIN_INLIERS = 12
MIN_INLIER_RATIO = 0.18

# Let a high-ratio model through even if the absolute inlier count is modest.
# Thermal ORB may only find a few repeatable features on smooth pipe surfaces.
HIGH_CONFIDENCE_MIN_INLIERS = 12
HIGH_CONFIDENCE_INLIER_RATIO = 0.30

# After ORB, try only tiny x/y nudges. This can improve edge alignment without
# allowing scale/shear/perspective distortion.
LOCAL_TRANSLATION_REFINE = True
# None builds a right-side ROI that spans the current crop height. This matters
# when changing the crop from 320px to 360px tall.
LOCAL_REFINE_ROI = None
LOCAL_REFINE_MAX_SHIFT = 7.0
LOCAL_REFINE_STEP = 0.5

# Physical-objective refinement. This assumes the pipe-body ROI should have an
# almost constant after-before temperature change once alignment is correct.
UNIFORM_DELTA_REFINE = True
# None builds a pipe-body ROI from the current crop size.
UNIFORM_DELTA_ROI = None
UNIFORM_DELTA_EXCLUDE_GRADIENT_PERCENTILE = 82.0
UNIFORM_DELTA_MAX_SHIFT = 4.0
UNIFORM_DELTA_MAX_ROTATION_DEG = 0.8
UNIFORM_DELTA_MAX_SCALE_DELTA = 0.006
UNIFORM_DELTA_INITIAL_STEPS = (1.0, 1.0, 0.15, 0.0015)  # dx, dy, rotation_deg, scale_delta
UNIFORM_DELTA_MIN_STEPS = (0.125, 0.125, 0.025, 0.00025)
UNIFORM_DELTA_MAX_PASSES = 24

# Sanity limits for this pair of mostly similar images.
MAX_TRANSLATION_PIXELS = 180
MIN_AREA_RATIO = 0.55
MAX_AREA_RATIO = 1.80

# Manual last-mile nudge after ORB, in output pixels.
# Positive X moves the aligned after image right; positive Y moves it down.
MANUAL_SHIFT_X = -5.0
MANUAL_SHIFT_Y = 0.0


def polygon_area(points):
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    x = pts[:, 0]
    y = pts[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def transformed_corners(transform, model, shape):
    h, w = shape[:2]
    corners = np.float32([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1],
    ]).reshape(-1, 1, 2)

    if model == "homography":
        return cv2.perspectiveTransform(corners, transform).reshape(-1, 2)

    return cv2.transform(corners, transform).reshape(-1, 2)


def transform_is_reasonable(transform, model, shape):
    h, w = shape[:2]
    corners = transformed_corners(transform, model, shape)

    if not np.all(np.isfinite(corners)):
        return False, "non-finite transformed corners"

    original_area = float((w - 1) * (h - 1))
    warped_area = polygon_area(corners)
    area_ratio = warped_area / max(original_area, 1.0)

    center = np.float32([[[w / 2.0, h / 2.0]]])
    if model == "homography":
        warped_center = cv2.perspectiveTransform(center, transform).reshape(2)
    else:
        warped_center = cv2.transform(center, transform).reshape(2)

    center_shift = float(np.linalg.norm(warped_center - np.float32([w / 2.0, h / 2.0])))

    if center_shift > MAX_TRANSLATION_PIXELS:
        return False, f"center shift too large: {center_shift:.2f}px"

    if area_ratio < MIN_AREA_RATIO or area_ratio > MAX_AREA_RATIO:
        return False, f"area ratio out of range: {area_ratio:.3f}"

    return True, f"center_shift={center_shift:.2f}px, area_ratio={area_ratio:.3f}"


def ratio_matches(knn_matches, ratio):
    good = []
    for pair in knn_matches:
        if len(pair) < 2:
            continue

        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)

    return good


def mutual_ratio_matches(des_after, des_before):
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    # after -> before
    a2b_knn = matcher.knnMatch(des_after, des_before, k=2)
    a2b_good = ratio_matches(a2b_knn, LOWE_RATIO)

    # before -> after
    b2a_knn = matcher.knnMatch(des_before, des_after, k=2)
    b2a_good = ratio_matches(b2a_knn, LOWE_RATIO)
    reverse_best = {m.queryIdx: m.trainIdx for m in b2a_good}

    mutual = [
        m for m in a2b_good
        if reverse_best.get(m.trainIdx) == m.queryIdx
    ]

    mutual.sort(key=lambda match: match.distance)
    return mutual[:MAX_MATCHES_FOR_MODEL]


def estimate_affine(pts_after, pts_before, partial=False):
    if partial:
        M, inliers = cv2.estimateAffinePartial2D(
            pts_after,
            pts_before,
            method=cv2.RANSAC,
            ransacReprojThreshold=RANSAC_REPROJ_THRESHOLD,
            maxIters=RANSAC_MAX_ITERS,
            confidence=RANSAC_CONFIDENCE,
            refineIters=RANSAC_REFINE_ITERS,
        )
        return M, inliers

    M, inliers = cv2.estimateAffine2D(
        pts_after,
        pts_before,
        method=cv2.RANSAC,
        ransacReprojThreshold=RANSAC_REPROJ_THRESHOLD,
        maxIters=RANSAC_MAX_ITERS,
        confidence=RANSAC_CONFIDENCE,
        refineIters=RANSAC_REFINE_ITERS,
    )
    return M, inliers


def estimate_homography(pts_after, pts_before):
    H, inliers = cv2.findHomography(
        pts_after,
        pts_before,
        method=cv2.RANSAC,
        ransacReprojThreshold=RANSAC_REPROJ_THRESHOLD,
        maxIters=RANSAC_MAX_ITERS,
        confidence=RANSAC_CONFIDENCE,
    )
    return H, inliers


def candidate_score(candidate):
    model_penalty = {
        "homography": 8.0,
        "affine": 0.0,
        "partial_affine": -2.0,
    }[candidate["model"]]

    return candidate["inliers"] + 25.0 * candidate["inlier_ratio"] - model_penalty


def add_manual_shift(transform, model):
    if MANUAL_SHIFT_X == 0 and MANUAL_SHIFT_Y == 0:
        return transform

    if model == "homography":
        T = np.array([
            [1.0, 0.0, MANUAL_SHIFT_X],
            [0.0, 1.0, MANUAL_SHIFT_Y],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)
        return T @ transform

    shifted = transform.copy()
    shifted[0, 2] += MANUAL_SHIFT_X
    shifted[1, 2] += MANUAL_SHIFT_Y
    return shifted


def add_translation(transform, model, dx, dy):
    if model == "homography":
        T = np.array([
            [1.0, 0.0, dx],
            [0.0, 1.0, dy],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)
        return T @ transform

    shifted = transform.copy()
    shifted[0, 2] += dx
    shifted[1, 2] += dy
    return shifted


def resolve_local_refine_roi(shape):
    if LOCAL_REFINE_ROI is not None:
        return LOCAL_REFINE_ROI

    h, w = shape[:2]
    x = max(0, int(round(w * 0.46)))
    return (x, 0, w - x, h)


def resolve_uniform_delta_roi(shape):
    if UNIFORM_DELTA_ROI is not None:
        return UNIFORM_DELTA_ROI

    h, w = shape[:2]
    x = int(round(w * 0.07))
    y = int(round(h * 0.14))
    roi_w = int(round(w * 0.56))
    roi_h = int(round(h * 0.68))
    return (x, y, roi_w, roi_h)


def score_image_for_refine(img):
    img_f = img.astype(np.float32)
    gx = cv2.Sobel(img_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_f, cv2.CV_32F, 0, 1, ksize=3)
    score_img = cv2.magnitude(gx, gy)
    return cv2.GaussianBlur(score_img, (3, 3), 0)


def masked_ncc(a, b, mask):
    valid = mask > 0
    if int(valid.sum()) < 100:
        return -np.inf

    av = a[valid].astype(np.float32)
    bv = b[valid].astype(np.float32)

    av -= av.mean()
    bv -= bv.mean()

    denom = float(np.sqrt(np.sum(av * av) * np.sum(bv * bv)))
    if denom <= 1e-6:
        return -np.inf

    return float(np.sum(av * bv) / denom)


def refine_translation_only(before_ref, after_ref, transform, model):
    if not LOCAL_TRANSLATION_REFINE:
        return transform, (0.0, 0.0), None

    before_score = score_image_for_refine(before_ref)
    after_score = score_image_for_refine(after_ref)
    ones = np.ones_like(after_ref, dtype=np.uint8)

    dx_values = np.arange(-LOCAL_REFINE_MAX_SHIFT, LOCAL_REFINE_MAX_SHIFT + 0.001, LOCAL_REFINE_STEP)
    dy_values = np.arange(-LOCAL_REFINE_MAX_SHIFT, LOCAL_REFINE_MAX_SHIFT + 0.001, LOCAL_REFINE_STEP)

    best_score = -np.inf
    best_shift = (0.0, 0.0)
    best_transform = transform.copy()
    local_refine_roi = resolve_local_refine_roi(before_ref.shape)

    before_roi = crop_roi(before_score, local_refine_roi)

    for dy in dy_values:
        for dx in dx_values:
            shifted = add_translation(transform, model, float(dx), float(dy))
            warped = warp_with_model(after_score, shifted, model, before_ref.shape, cv2.INTER_LINEAR)
            valid = warp_with_model(ones, shifted, model, before_ref.shape, cv2.INTER_NEAREST)

            warped_roi = crop_roi(warped, local_refine_roi)
            valid_roi = crop_roi(valid, local_refine_roi)
            score = masked_ncc(before_roi, warped_roi, valid_roi)

            if score > best_score:
                best_score = score
                best_shift = (float(dx), float(dy))
                best_transform = shifted.astype(np.float32)

    print(
        "Local translation refinement:",
        f"dx={best_shift[0]:.2f}",
        f"dy={best_shift[1]:.2f}",
        f"score={best_score:.6f}",
    )

    return best_transform, best_shift, best_score


def to_homogeneous_transform(transform, model):
    if model == "homography":
        return transform.astype(np.float32).copy()

    H = np.eye(3, dtype=np.float32)
    H[:2, :] = transform.astype(np.float32)
    return H


def from_homogeneous_transform(transform_h, model):
    if model == "homography":
        return transform_h.astype(np.float32)

    return transform_h[:2, :].astype(np.float32)


def centered_similarity_adjustment(shape, roi, dx, dy, rotation_deg, scale_delta):
    x, y, roi_w, roi_h = roi
    cx = float(x + roi_w / 2.0)
    cy = float(y + roi_h / 2.0)
    scale = 1.0 + float(scale_delta)
    theta = np.deg2rad(float(rotation_deg))
    c = float(np.cos(theta) * scale)
    s = float(np.sin(theta) * scale)

    return np.array([
        [c, -s, (1.0 - c) * cx + s * cy + float(dx)],
        [s, c, -s * cx + (1.0 - c) * cy + float(dy)],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)


def apply_output_similarity_adjustment(transform, model, shape, roi, params):
    dx, dy, rotation_deg, scale_delta = params
    adjustment = centered_similarity_adjustment(
        shape,
        roi,
        dx,
        dy,
        rotation_deg,
        scale_delta,
    )
    combined = adjustment @ to_homogeneous_transform(transform, model)
    return from_homogeneous_transform(combined, model)


def build_uniform_delta_mask(before_temp):
    uniform_delta_roi = resolve_uniform_delta_roi(before_temp.shape)
    before_roi = crop_roi(before_temp, uniform_delta_roi)
    gx = cv2.Sobel(before_roi.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(before_roi.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    threshold = np.percentile(grad, UNIFORM_DELTA_EXCLUDE_GRADIENT_PERCENTILE)
    mask_roi = (grad <= threshold).astype(np.uint8)

    mask = np.zeros(before_temp.shape[:2], dtype=np.uint8)
    x, y, roi_w, roi_h = uniform_delta_roi
    mask[y:y + roi_h, x:x + roi_w] = mask_roi[:roi_h, :roi_w]
    return mask


def robust_delta_uniformity_score(before_temp, after_temp, transform, model, uniform_mask):
    warped_after = warp_with_model(after_temp, transform, model, before_temp.shape, cv2.INTER_LINEAR)
    valid = warp_with_model(
        np.ones_like(after_temp, dtype=np.uint8),
        transform,
        model,
        before_temp.shape,
        cv2.INTER_NEAREST,
    )

    mask = (uniform_mask > 0) & (valid > 0)
    if int(mask.sum()) < 500:
        return np.inf

    delta = warped_after[mask] - before_temp[mask]
    center = np.median(delta)
    mad = np.median(np.abs(delta - center))
    p95 = np.percentile(np.abs(delta - center), 95)

    # A tiny regularizer prevents the uniform-region score from drifting far
    # when several nearby corrections look similar.
    return float(mad + 0.08 * p95)


def refine_by_uniform_delta(before_temp, after_temp, transform, model):
    if not UNIFORM_DELTA_REFINE:
        return transform, (0.0, 0.0, 0.0, 0.0), None

    uniform_mask = build_uniform_delta_mask(before_temp)
    cv2.imwrite(str(output_folder / "uniform_delta_refine_mask.png"), uniform_mask * 255)

    params = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    steps = np.array(UNIFORM_DELTA_INITIAL_STEPS, dtype=np.float32)
    min_steps = np.array(UNIFORM_DELTA_MIN_STEPS, dtype=np.float32)
    limits = np.array([
        UNIFORM_DELTA_MAX_SHIFT,
        UNIFORM_DELTA_MAX_SHIFT,
        UNIFORM_DELTA_MAX_ROTATION_DEG,
        UNIFORM_DELTA_MAX_SCALE_DELTA,
    ], dtype=np.float32)

    def clipped(candidate):
        return np.clip(candidate, -limits, limits)

    current_transform = apply_output_similarity_adjustment(
        transform,
        model,
        before_temp.shape,
        resolve_uniform_delta_roi(before_temp.shape),
        params,
    )
    current_score = robust_delta_uniformity_score(
        before_temp,
        after_temp,
        current_transform,
        model,
        uniform_mask,
    )

    pass_count = 0
    while np.any(steps >= min_steps) and pass_count < UNIFORM_DELTA_MAX_PASSES:
        improved = False
        pass_count += 1

        for index in range(4):
            for direction in (-1.0, 1.0):
                candidate_params = params.copy()
                candidate_params[index] += direction * steps[index]
                candidate_params = clipped(candidate_params)

                if np.allclose(candidate_params, params):
                    continue

                candidate_transform = apply_output_similarity_adjustment(
                    transform,
                    model,
                    before_temp.shape,
                    resolve_uniform_delta_roi(before_temp.shape),
                    candidate_params,
                )
                candidate_score = robust_delta_uniformity_score(
                    before_temp,
                    after_temp,
                    candidate_transform,
                    model,
                    uniform_mask,
                )

                if candidate_score + 1e-5 < current_score:
                    params = candidate_params
                    current_transform = candidate_transform
                    current_score = candidate_score
                    improved = True

        if not improved:
            steps *= 0.5

    print(
        "Uniform-delta refinement:",
        f"dx={params[0]:.3f}",
        f"dy={params[1]:.3f}",
        f"rot={params[2]:.3f} deg",
        f"scale_delta={params[3]:.5f}",
        f"score={current_score:.6f}",
        f"passes={pass_count}",
    )

    return current_transform.astype(np.float32), tuple(float(v) for v in params), current_score


def improved_thermal_orb_alignment(before_temp, after_temp):
    before_orb = preprocess_temperature_for_orb(before_temp)
    after_orb = preprocess_temperature_for_orb(after_temp)
    feature_mask = build_roi_mask(before_orb.shape, MATCH_ROI, border=8, label="MATCH_ROI")

    cv2.imwrite(str(output_folder / "before_orb_preprocessed.png"), before_orb)
    cv2.imwrite(str(output_folder / "after_orb_preprocessed.png"), after_orb)
    cv2.imwrite(str(output_folder / "feature_mask.png"), feature_mask)

    orb = cv2.ORB_create(
        nfeatures=ORB_FEATURES,
        scaleFactor=ORB_SCALE_FACTOR,
        nlevels=ORB_LEVELS,
        edgeThreshold=ORB_EDGE_THRESHOLD,
        patchSize=ORB_PATCH_SIZE,
        fastThreshold=ORB_FAST_THRESHOLD,
        scoreType=cv2.ORB_HARRIS_SCORE,
    )

    kp_before, des_before = orb.detectAndCompute(before_orb, feature_mask)
    kp_after, des_after = orb.detectAndCompute(after_orb, feature_mask)

    if des_before is None or des_after is None:
        raise RuntimeError("ORB failed: not enough thermal features found.")

    matches = mutual_ratio_matches(des_after, des_before)

    if len(matches) < MIN_GOOD_MATCHES:
        raise RuntimeError(f"Too few mutual ORB matches: {len(matches)}")

    pts_after = np.float32([kp_after[m.queryIdx].pt for m in matches])
    pts_before = np.float32([kp_before[m.trainIdx].pt for m in matches])

    requested_models = {
        "auto": ["affine", "homography", "partial_affine"],
        "affine": ["affine"],
        "homography": ["homography"],
        "partial_affine": ["partial_affine"],
    }

    if TRANSFORM_MODEL not in requested_models:
        raise ValueError(f"Unknown TRANSFORM_MODEL: {TRANSFORM_MODEL}")

    candidates = []

    for model in requested_models[TRANSFORM_MODEL]:
        if model == "homography":
            transform, inlier_mask = estimate_homography(pts_after, pts_before)
        elif model == "affine":
            transform, inlier_mask = estimate_affine(pts_after, pts_before, partial=False)
        else:
            transform, inlier_mask = estimate_affine(pts_after, pts_before, partial=True)

        if transform is None or inlier_mask is None:
            print(f"{model}: failed")
            continue

        inliers = int(inlier_mask.sum())
        inlier_ratio = inliers / max(len(matches), 1)
        reasonable, reason = transform_is_reasonable(transform, model, before_temp.shape)

        print(f"{model}: inliers={inliers}, ratio={inlier_ratio:.3f}, {reason}")

        enough_standard_inliers = inliers >= MIN_INLIERS and inlier_ratio >= MIN_INLIER_RATIO
        enough_high_confidence_inliers = (
            inliers >= HIGH_CONFIDENCE_MIN_INLIERS
            and inlier_ratio >= HIGH_CONFIDENCE_INLIER_RATIO
        )

        if not (enough_standard_inliers or enough_high_confidence_inliers) or not reasonable:
            continue

        candidates.append({
            "model": model,
            "transform": transform.astype(np.float32),
            "inlier_mask": inlier_mask.astype(np.uint8).reshape(-1),
            "inliers": inliers,
            "inlier_ratio": inlier_ratio,
            "match_count": len(matches),
        })

    if not candidates:
        raise RuntimeError(
            "No acceptable ORB transform found. "
            f"Thresholds: MIN_INLIERS={MIN_INLIERS}, MIN_INLIER_RATIO={MIN_INLIER_RATIO}, "
            f"HIGH_CONFIDENCE_MIN_INLIERS={HIGH_CONFIDENCE_MIN_INLIERS}, "
            f"HIGH_CONFIDENCE_INLIER_RATIO={HIGH_CONFIDENCE_INLIER_RATIO}. "
            "Try setting MATCH_ROI or relaxing thresholds."
        )

    best = max(candidates, key=candidate_score)
    best["orb_transform"] = best["transform"].copy()
    best["local_refine_shift"] = (0.0, 0.0)
    best["local_refine_score"] = None
    best["uniform_delta_params"] = (0.0, 0.0, 0.0, 0.0)
    best["uniform_delta_score"] = None

    refined_transform, refine_shift, refine_score = refine_translation_only(
        before_orb,
        after_orb,
        best["transform"],
        best["model"],
    )
    best["transform"] = refined_transform
    best["local_refine_shift"] = refine_shift
    best["local_refine_score"] = refine_score

    uniform_transform, uniform_params, uniform_score = refine_by_uniform_delta(
        before_temp,
        after_temp,
        best["transform"],
        best["model"],
    )
    best["transform"] = uniform_transform
    best["uniform_delta_params"] = uniform_params
    best["uniform_delta_score"] = uniform_score

    best["transform"] = add_manual_shift(best["transform"], best["model"])

    print("\nStable thermal ORB alignment:")
    print("Keypoints before:", len(kp_before))
    print("Keypoints after:", len(kp_after))
    print("Mutual good matches:", len(matches))
    print("Selected model:", best["model"])
    print("Selected inliers:", best["inliers"])
    print("Selected inlier ratio:", best["inlier_ratio"])
    print("Local refine shift:", best["local_refine_shift"])
    print("Local refine score:", best["local_refine_score"])
    print("Uniform-delta params:", best["uniform_delta_params"])
    print("Uniform-delta score:", best["uniform_delta_score"])
    print("Transform after -> before:")
    print(best["transform"])

    match_vis = cv2.drawMatches(
        after_orb,
        kp_after,
        before_orb,
        kp_before,
        matches,
        None,
        matchesMask=best["inlier_mask"].tolist(),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    cv2.imwrite(str(output_folder / "thermal_orb_inlier_matches_after_to_before.png"), match_vis)

    return best, before_orb, after_orb


def warp_with_model(img, transform, model, output_shape, interpolation):
    h, w = output_shape[:2]

    if model == "homography":
        return cv2.warpPerspective(
            img,
            transform,
            (w, h),
            flags=interpolation,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    return cv2.warpAffine(
        img,
        transform,
        (w, h),
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def save_alignment_log(path, best):
    text = f"""Thermal ORB alignment log

Input:
before thermal JPG = {before_thermal_path}
after thermal JPG  = {after_thermal_path}
before raw         = {before_raw_path}
after raw          = {after_raw_path}

Thermal crop before:
x={BEFORE_THERMAL_CROP_X}
y={BEFORE_THERMAL_CROP_Y}
w={BEFORE_THERMAL_CROP_W}
h={BEFORE_THERMAL_CROP_H}

Thermal crop after:
x={AFTER_THERMAL_CROP_X}
y={AFTER_THERMAL_CROP_Y}
w={AFTER_THERMAL_CROP_W}
h={AFTER_THERMAL_CROP_H}

Matching:
MATCH_ROI={MATCH_ROI}
TRANSFORM_MODEL={TRANSFORM_MODEL}
selected_model={best["model"]}
inliers={best["inliers"]}
inlier_ratio={best["inlier_ratio"]}
match_count={best["match_count"]}
ORB_FEATURES={ORB_FEATURES}
ORB_SCALE_FACTOR={ORB_SCALE_FACTOR}
ORB_LEVELS={ORB_LEVELS}
ORB_EDGE_THRESHOLD={ORB_EDGE_THRESHOLD}
ORB_FAST_THRESHOLD={ORB_FAST_THRESHOLD}
LOWE_RATIO={LOWE_RATIO}
MAX_MATCHES_FOR_MODEL={MAX_MATCHES_FOR_MODEL}
RANSAC_REPROJ_THRESHOLD={RANSAC_REPROJ_THRESHOLD}
RANSAC_MAX_ITERS={RANSAC_MAX_ITERS}
RANSAC_REFINE_ITERS={RANSAC_REFINE_ITERS}
MIN_INLIERS={MIN_INLIERS}
MIN_INLIER_RATIO={MIN_INLIER_RATIO}
HIGH_CONFIDENCE_MIN_INLIERS={HIGH_CONFIDENCE_MIN_INLIERS}
HIGH_CONFIDENCE_INLIER_RATIO={HIGH_CONFIDENCE_INLIER_RATIO}
LOCAL_TRANSLATION_REFINE={LOCAL_TRANSLATION_REFINE}
LOCAL_REFINE_ROI={LOCAL_REFINE_ROI}
resolved_local_refine_roi={resolve_local_refine_roi((BEFORE_THERMAL_CROP_H, BEFORE_THERMAL_CROP_W))}
LOCAL_REFINE_MAX_SHIFT={LOCAL_REFINE_MAX_SHIFT}
LOCAL_REFINE_STEP={LOCAL_REFINE_STEP}
local_refine_shift={best["local_refine_shift"]}
local_refine_score={best["local_refine_score"]}
UNIFORM_DELTA_REFINE={UNIFORM_DELTA_REFINE}
UNIFORM_DELTA_ROI={UNIFORM_DELTA_ROI}
resolved_uniform_delta_roi={resolve_uniform_delta_roi((BEFORE_THERMAL_CROP_H, BEFORE_THERMAL_CROP_W))}
UNIFORM_DELTA_EXCLUDE_GRADIENT_PERCENTILE={UNIFORM_DELTA_EXCLUDE_GRADIENT_PERCENTILE}
UNIFORM_DELTA_MAX_SHIFT={UNIFORM_DELTA_MAX_SHIFT}
UNIFORM_DELTA_MAX_ROTATION_DEG={UNIFORM_DELTA_MAX_ROTATION_DEG}
UNIFORM_DELTA_MAX_SCALE_DELTA={UNIFORM_DELTA_MAX_SCALE_DELTA}
UNIFORM_DELTA_INITIAL_STEPS={UNIFORM_DELTA_INITIAL_STEPS}
UNIFORM_DELTA_MIN_STEPS={UNIFORM_DELTA_MIN_STEPS}
uniform_delta_params={best["uniform_delta_params"]}
uniform_delta_score={best["uniform_delta_score"]}
uniform_delta_offset_C={best.get("uniform_delta_offset_C", None)}
manual_shift_x={MANUAL_SHIFT_X}
manual_shift_y={MANUAL_SHIFT_Y}

Temperature scaling:
temp_C = raw_int16 / {TEMP_SCALE}

ORB transform before local/manual shift:
{best["orb_transform"]}

Transform after -> before:
{best["transform"]}
"""
    path.write_text(text)


# ============================================================
# MAIN
# ============================================================

before_thermal_original = load_grayscale_image(before_thermal_path)
after_thermal_original = load_grayscale_image(after_thermal_path)

print("\nOriginal rendered thermal image shapes:")
print("Before thermal original:", before_thermal_original.shape)
print("After thermal original:", after_thermal_original.shape)

before_thermal_crop = crop_xywh(
    before_thermal_original,
    (
        BEFORE_THERMAL_CROP_X,
        BEFORE_THERMAL_CROP_Y,
        BEFORE_THERMAL_CROP_W,
        BEFORE_THERMAL_CROP_H,
    ),
    "before thermal JPG",
)

after_thermal_crop = crop_xywh(
    after_thermal_original,
    (
        AFTER_THERMAL_CROP_X,
        AFTER_THERMAL_CROP_Y,
        AFTER_THERMAL_CROP_W,
        AFTER_THERMAL_CROP_H,
    ),
    "after thermal JPG",
)

cv2.imwrite(str(output_folder / "before_rendered_thermal_crop.png"), before_thermal_crop)
cv2.imwrite(str(output_folder / "after_rendered_thermal_crop.png"), after_thermal_crop)

before_temp_full = load_dji_raw_temperature(before_raw_path, THERMAL_RAW_SPEC)
after_temp_full = load_dji_raw_temperature(after_raw_path, THERMAL_RAW_SPEC)

before_temp_crop = crop_xywh(
    before_temp_full,
    (
        BEFORE_THERMAL_CROP_X,
        BEFORE_THERMAL_CROP_Y,
        BEFORE_THERMAL_CROP_W,
        BEFORE_THERMAL_CROP_H,
    ),
    "before temperature",
)

after_temp_crop = crop_xywh(
    after_temp_full,
    (
        AFTER_THERMAL_CROP_X,
        AFTER_THERMAL_CROP_Y,
        AFTER_THERMAL_CROP_W,
        AFTER_THERMAL_CROP_H,
    ),
    "after temperature",
)

if before_temp_crop.shape != after_temp_crop.shape:
    raise ValueError(
        f"Temperature crops must have same shape. "
        f"Before={before_temp_crop.shape}, after={after_temp_crop.shape}"
    )

target_shape = before_temp_crop.shape

np.save(output_folder / "before_temp_crop_C.npy", before_temp_crop)
np.save(output_folder / "after_temp_crop_C.npy", after_temp_crop)

save_overlay(
    before_temp_crop,
    after_temp_crop,
    output_folder / "temperature_overlay_before_alignment.png",
)

save_overlay(
    before_thermal_crop,
    after_thermal_crop,
    output_folder / "rendered_thermal_overlay_before_alignment.png",
)

best_transform, before_orb, after_orb = improved_thermal_orb_alignment(
    before_temp_crop,
    after_temp_crop,
)

model = best_transform["model"]
transform = best_transform["transform"]

np.save(output_folder / "transform_after_to_before_thermal_orb.npy", transform)
np.savetxt(output_folder / "transform_after_to_before_thermal_orb.csv", transform, delimiter=",")
save_alignment_log(output_folder / "alignment_log.txt", best_transform)

after_temp_aligned = warp_with_model(
    after_temp_crop,
    transform,
    model,
    target_shape,
    cv2.INTER_LINEAR,
)

valid_mask = warp_with_model(
    np.ones_like(after_temp_crop, dtype=np.uint8),
    transform,
    model,
    target_shape,
    cv2.INTER_NEAREST,
)

after_rendered_thermal_aligned = warp_with_model(
    after_thermal_crop,
    transform,
    model,
    target_shape,
    cv2.INTER_LINEAR,
)

after_orb_aligned = warp_with_model(
    after_orb,
    transform,
    model,
    target_shape,
    cv2.INTER_LINEAR,
)

delta_temp = after_temp_aligned - before_temp_crop
delta_temp[valid_mask == 0] = np.nan

uniform_delta_mask = build_uniform_delta_mask(before_temp_crop)
uniform_delta_valid = (uniform_delta_mask > 0) & np.isfinite(delta_temp)

if int(uniform_delta_valid.sum()) > 0:
    uniform_delta_offset = float(np.nanmedian(delta_temp[uniform_delta_valid]))
else:
    uniform_delta_offset = float("nan")

best_transform["uniform_delta_offset_C"] = uniform_delta_offset
delta_temp_uniform_centered = delta_temp - uniform_delta_offset

np.save(output_folder / "after_temp_aligned_C.npy", after_temp_aligned)
np.save(output_folder / "delta_temp_after_minus_before_C.npy", delta_temp)
np.save(output_folder / "delta_temp_uniform_roi_centered_C.npy", delta_temp_uniform_centered)
np.savetxt(output_folder / "delta_temp_after_minus_before_C.csv", delta_temp, delimiter=",")
np.savetxt(output_folder / "delta_temp_uniform_roi_centered_C.csv", delta_temp_uniform_centered, delimiter=",")
save_alignment_log(output_folder / "alignment_log.txt", best_transform)

cv2.imwrite(str(output_folder / "after_rendered_thermal_aligned.png"), after_rendered_thermal_aligned)
cv2.imwrite(str(output_folder / "after_orb_preprocessed_aligned.png"), after_orb_aligned)

save_overlay(
    before_orb,
    after_orb_aligned,
    output_folder / "orb_preprocessed_overlay_after_alignment.png",
)

save_overlay(
    before_temp_crop,
    after_temp_aligned,
    output_folder / "temperature_overlay_after_alignment.png",
)

save_overlay(
    before_thermal_crop,
    after_rendered_thermal_aligned,
    output_folder / "rendered_thermal_overlay_after_alignment.png",
)

save_heatmap(
    before_temp_crop,
    output_folder / "before_temperature_crop.png",
    "Before Temperature Crop",
    "Temperature (C)",
    cmap="inferno",
)

save_heatmap(
    after_temp_crop,
    output_folder / "after_temperature_crop_original.png",
    "After Temperature Crop Original",
    "Temperature (C)",
    cmap="inferno",
)

save_heatmap(
    after_temp_aligned,
    output_folder / "after_temperature_crop_aligned_thermal_orb.png",
    "After Temperature Crop Aligned Using Thermal ORB",
    "Temperature (C)",
    cmap="inferno",
)

save_heatmap(
    delta_temp,
    output_folder / "temperature_change_deltaT_thermal_orb.png",
    "Temperature Change: After - Before",
    "Delta T (C)",
    cmap="coolwarm",
    symmetric=True,
)

save_heatmap(
    delta_temp_uniform_centered,
    output_folder / "temperature_change_deltaT_uniform_roi_centered.png",
    "Temperature Change Residual After Uniform ROI Offset",
    "Residual Delta T (C)",
    cmap="coolwarm",
    symmetric=True,
)

print("\nDone.")
print("Outputs saved to:", output_folder)
