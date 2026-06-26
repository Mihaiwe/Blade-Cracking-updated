import json
from pathlib import Path

import cv2
import numpy as np

# Common functions used here are defined in thermal_alignment_common.py.
from thermal_alignment_common import (
    ThermalRawSpec,
    crop_xywh,
    load_dji_raw_temperature,
    load_grayscale_image,
    normalize_to_uint8,
    save_heatmap,
    save_overlay,
)


# ============================================================
# USER SETTINGS
# ============================================================

base_folder = Path(r"C:\Users\mihai.dobre\Desktop\Documents Mihai\drone imgs")
output_folder = base_folder / "alignment_outputs_manual_point_alignment"
output_folder.mkdir(parents=True, exist_ok=True)

before_thermal_path = base_folder / "DJI_0044_T.JPG"
after_thermal_path = base_folder / "DJI_0047_T.JPG"

before_raw_path = Path(r"C:\Users\mihai.dobre\Downloads\DJI_0044_T.raw")
after_raw_path = Path(r"C:\Users\mihai.dobre\Downloads\DJI_0047_T.raw")

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
BEFORE_THERMAL_CROP_Y = 84  # distance from y=0 axis cropped from the top
BEFORE_THERMAL_CROP_W = 640  # total width of the remaining image after crop
BEFORE_THERMAL_CROP_H = 320  # total height of the remaining image after crop

AFTER_THERMAL_CROP_X = 0
AFTER_THERMAL_CROP_Y = 84
AFTER_THERMAL_CROP_W = 640
AFTER_THERMAL_CROP_H = 320

# Best default for your case. It permits shift, rotation, and uniform scale,
# but avoids shear/perspective bending.
TRANSFORM_MODEL = "partial_affine"  # "translation", "partial_affine", "affine", "homography"

POINTS_JSON = output_folder / "manual_points.json"


def draw_points(img, points, color):
    out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for index, (x, y) in enumerate(points, start=1):
        cv2.circle(out, (int(round(x)), int(round(y))), 5, color, -1)
        cv2.putText(
            out,
            str(index),
            (int(round(x)) + 7, int(round(y)) - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    return out


def build_side_by_side(before_img, after_img, before_points, after_points):
    before_drawn = draw_points(before_img, before_points, (0, 255, 0))
    after_drawn = draw_points(after_img, after_points, (0, 200, 255))
    separator = np.full((before_img.shape[0], 8, 3), 255, dtype=np.uint8)
    return np.hstack([before_drawn, separator, after_drawn])


def collect_manual_points(before_display, after_display):
    if POINTS_JSON.exists():
        print(f"\nExisting points found: {POINTS_JSON}")
        answer = input("Reuse existing points? [y/N]: ").strip().lower()
        if answer == "y":
            data = json.loads(POINTS_JSON.read_text())
            return data["before_points"], data["after_points"]

    before_points = []
    after_points = []
    side_gap = 8
    before_w = before_display.shape[1]

    instructions = (
        "Click matching points in pairs:\n"
        "  1) click point on BEFORE image, left side\n"
        "  2) click matching point on AFTER image, right side\n"
        "Keys: u=undo last pair, s=save/solve, q=quit\n"
        "Use 4-8 points around the wrap-around band/cap if possible."
    )
    print("\n" + instructions)

    window_name = "Manual thermal point matching"

    def on_mouse(event, x, y, flags, param):
        del flags, param
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if x < before_w:
            before_points.append([float(x), float(y)])
            print(f"before #{len(before_points)}: ({x}, {y})")
        elif x >= before_w + side_gap:
            after_x = x - before_w - side_gap
            after_points.append([float(after_x), float(y)])
            print(f"after  #{len(after_points)}: ({after_x}, {y})")
        else:
            print("Clicked separator; ignored.")

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        canvas = build_side_by_side(before_display, after_display, before_points, after_points)
        cv2.imshow(window_name, canvas)
        key = cv2.waitKey(30) & 0xFF

        if key == ord("u"):
            if before_points:
                removed = before_points.pop()
                print("undo before:", removed)
            if after_points:
                removed = after_points.pop()
                print("undo after:", removed)

        elif key == ord("s"):
            if len(before_points) != len(after_points):
                print("Point counts do not match yet.")
                continue
            if len(before_points) < 2:
                print("Need at least 2 pairs; 4-8 is better.")
                continue
            break

        elif key == ord("q"):
            cv2.destroyWindow(window_name)
            raise SystemExit("Manual point selection canceled.")

    cv2.destroyWindow(window_name)

    data = {
        "before_points": before_points,
        "after_points": after_points,
        "note": "Coordinates are in cropped thermal-image coordinates.",
    }
    POINTS_JSON.write_text(json.dumps(data, indent=2))
    print("Saved points to:", POINTS_JSON)
    return before_points, after_points


def estimate_transform(after_points, before_points):
    pts_after = np.float32(after_points)
    pts_before = np.float32(before_points)

    if len(pts_after) != len(pts_before):
        raise ValueError("Point lists must have the same length.")

    if TRANSFORM_MODEL == "translation":
        shifts = pts_before - pts_after
        dx, dy = np.median(shifts, axis=0)
        M = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
        return M, None

    if TRANSFORM_MODEL == "partial_affine":
        if len(pts_after) < 2:
            raise ValueError("partial_affine needs at least 2 point pairs.")
        M, inliers = cv2.estimateAffinePartial2D(
            pts_after,
            pts_before,
            method=cv2.RANSAC,
            ransacReprojThreshold=4.0,
            maxIters=20000,
            confidence=0.999,
            refineIters=100,
        )
        return M, inliers

    if TRANSFORM_MODEL == "affine":
        if len(pts_after) < 3:
            raise ValueError("affine needs at least 3 point pairs.")
        M, inliers = cv2.estimateAffine2D(
            pts_after,
            pts_before,
            method=cv2.RANSAC,
            ransacReprojThreshold=4.0,
            maxIters=20000,
            confidence=0.999,
            refineIters=100,
        )
        return M, inliers

    if TRANSFORM_MODEL == "homography":
        if len(pts_after) < 4:
            raise ValueError("homography needs at least 4 point pairs.")
        H, inliers = cv2.findHomography(
            pts_after,
            pts_before,
            method=cv2.RANSAC,
            ransacReprojThreshold=4.0,
            maxIters=20000,
            confidence=0.999,
        )
        return H, inliers

    raise ValueError(f"Unknown TRANSFORM_MODEL: {TRANSFORM_MODEL}")


def warp_with_transform(img, transform, output_shape, interpolation):
    h, w = output_shape[:2]
    if TRANSFORM_MODEL == "homography":
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


# ============================================================
# MAIN
# ============================================================

before_thermal_full = load_grayscale_image(before_thermal_path)
after_thermal_full = load_grayscale_image(after_thermal_path)

before_thermal_crop = crop_xywh(
    before_thermal_full,
    (
        BEFORE_THERMAL_CROP_X,
        BEFORE_THERMAL_CROP_Y,
        BEFORE_THERMAL_CROP_W,
        BEFORE_THERMAL_CROP_H,
    ),
    "before thermal JPG",
)
after_thermal_crop = crop_xywh(
    after_thermal_full,
    (
        AFTER_THERMAL_CROP_X,
        AFTER_THERMAL_CROP_Y,
        AFTER_THERMAL_CROP_W,
        AFTER_THERMAL_CROP_H,
    ),
    "after thermal JPG",
)

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

before_display = normalize_to_uint8(before_temp_crop)
after_display = normalize_to_uint8(after_temp_crop)

cv2.imwrite(str(output_folder / "before_temperature_for_point_clicking.png"), before_display)
cv2.imwrite(str(output_folder / "after_temperature_for_point_clicking.png"), after_display)

before_points, after_points = collect_manual_points(before_display, after_display)

transform, inliers = estimate_transform(after_points, before_points)
if transform is None:
    raise RuntimeError("Manual transform estimation failed.")

print("\nManual transform after -> before:")
print(transform)

if inliers is not None:
    print("Inliers:", int(inliers.sum()), "of", len(inliers))

np.savetxt(output_folder / f"manual_transform_after_to_before_{TRANSFORM_MODEL}.csv", transform, delimiter=",")
np.save(output_folder / f"manual_transform_after_to_before_{TRANSFORM_MODEL}.npy", transform)

after_temp_aligned = warp_with_transform(
    after_temp_crop,
    transform,
    before_temp_crop.shape,
    cv2.INTER_LINEAR,
)
valid_mask = warp_with_transform(
    np.ones_like(after_temp_crop, dtype=np.uint8),
    transform,
    before_temp_crop.shape,
    cv2.INTER_NEAREST,
)
after_thermal_aligned = warp_with_transform(
    after_thermal_crop,
    transform,
    before_thermal_crop.shape,
    cv2.INTER_LINEAR,
)

delta_temp = after_temp_aligned - before_temp_crop
delta_temp[valid_mask == 0] = np.nan

np.save(output_folder / "before_temp_crop_C.npy", before_temp_crop)
np.save(output_folder / "after_temp_crop_C.npy", after_temp_crop)
np.save(output_folder / "after_temp_aligned_manual_C.npy", after_temp_aligned)
np.save(output_folder / "delta_temp_after_minus_before_manual_C.npy", delta_temp)
np.savetxt(output_folder / "delta_temp_after_minus_before_manual_C.csv", delta_temp, delimiter=",")

cv2.imwrite(str(output_folder / "after_rendered_thermal_aligned_manual.png"), after_thermal_aligned)

save_overlay(
    before_temp_crop,
    after_temp_aligned,
    output_folder / "temperature_overlay_after_manual_alignment.png",
)
save_overlay(
    before_thermal_crop,
    after_thermal_aligned,
    output_folder / "rendered_thermal_overlay_after_manual_alignment.png",
)

save_heatmap(
    before_temp_crop,
    output_folder / "before_temperature_crop.png",
    "Before Temperature Crop",
    "Temperature (C)",
)
save_heatmap(
    after_temp_crop,
    output_folder / "after_temperature_crop_original.png",
    "After Temperature Crop Original",
    "Temperature (C)",
)
save_heatmap(
    after_temp_aligned,
    output_folder / "after_temperature_crop_aligned_manual.png",
    "After Temperature Crop Aligned Manually",
    "Temperature (C)",
)
save_heatmap(
    delta_temp,
    output_folder / "temperature_change_deltaT_manual.png",
    "Temperature Change: After - Before",
    "Delta T (C)",
    cmap="coolwarm",
    symmetric=True,
)

point_preview = build_side_by_side(before_display, after_display, before_points, after_points)
cv2.imwrite(str(output_folder / "manual_point_pairs_preview.png"), point_preview)

log = {
    "transform_model": TRANSFORM_MODEL,
    "before_points": before_points,
    "after_points": after_points,
    "transform_after_to_before": transform.tolist(),
    "inliers": inliers.reshape(-1).astype(int).tolist() if inliers is not None else None,
}
(output_folder / "manual_alignment_log.json").write_text(json.dumps(log, indent=2))

print("\nDone.")
print("Outputs saved to:", output_folder)
