"""Shared utilities for thermal-image alignment scripts.

The alignment scripts in this folder use different matching methods
(manual points, ORB, SIFT, and LoFTR), but they should agree on the
basic preprocessing steps.  Keep reusable IO, crop, normalization,
display, and preprocessing helpers here so future scripts use the same
meaning for the same operation.
"""

from dataclasses import dataclass

import cv2
import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class ThermalRawSpec:
    """Shape and scale information for a DJI int16 raw thermal image."""

    width: int = 640
    height: int = 512
    temperature_scale: float = 10.0

    @property
    def expected_values(self):
        return self.width * self.height


def load_grayscale_image(path):
    """Load an image as a single-channel uint8 grayscale array."""

    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return image


def load_dji_raw_temperature(raw_path, raw_spec, verbose=True):
    """Load a DJI int16 thermal raw file and convert it to degrees C.

    Parameters
    ----------
    raw_path:
        Path to the `.raw` file exported by the DJI thermal tool.
    raw_spec:
        `ThermalRawSpec` with the raw image width, height, and scale factor.
    verbose:
        If true, print min/max/mean diagnostics while loading.
    """

    raw = np.fromfile(raw_path, dtype=np.int16)
    expected = raw_spec.expected_values

    if verbose:
        print("\nLoading raw temperature:")
        print(raw_path)
        print("Values:", raw.size)
        print("Expected:", expected)
        if raw.size:
            print("Raw min/max/mean:", raw.min(), raw.max(), raw.mean())

    if raw.size != expected:
        raise ValueError(f"{raw_path.name}: expected {expected} values, got {raw.size}")

    temperature = raw.reshape((raw_spec.height, raw_spec.width)).astype(np.float32)
    temperature /= float(raw_spec.temperature_scale)

    if verbose:
        print(
            "Temp C min/max/mean:",
            temperature.min(),
            temperature.max(),
            temperature.mean(),
        )

    return temperature


def crop_xywh(image, crop, label="image"):
    """Crop an image or matrix using `(x, y, width, height)` coordinates."""

    x, y, width, height = crop
    image_height, image_width = image.shape[:2]

    if x < 0 or y < 0 or width <= 0 or height <= 0:
        raise ValueError(f"{label}: invalid crop {crop}")

    if x + width > image_width or y + height > image_height:
        raise ValueError(f"{label}: crop {crop} exceeds image shape {image.shape}")

    return image[y : y + height, x : x + width]


def normalize_to_uint8(data, low_percentile=1.0, high_percentile=99.0):
    """Robustly scale a numeric array to uint8 using percentile clipping."""

    finite = np.isfinite(data)
    if not np.any(finite):
        raise ValueError("Cannot normalize an array with no finite values.")

    values = data[finite]
    low = float(np.percentile(values, low_percentile))
    high = float(np.percentile(values, high_percentile))

    if high <= low:
        high = low + 1.0

    normalized = (np.clip(data, low, high) - low) / (high - low)
    normalized[~finite] = 0.0
    return np.round(normalized * 255.0).astype(np.uint8)


def preprocess_temperature_for_orb(temperature):
    """Prepare a thermal crop for ORB feature detection.

    ORB works best with distinct edges and corners, so this uses robust
    thermal normalization, CLAHE contrast, unsharp masking, median
    filtering, and a small Canny-edge contribution.
    """

    image = normalize_to_uint8(temperature, 1.0, 99.0)
    contrast = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(image)
    blur = cv2.GaussianBlur(contrast, (0, 0), 1.2)
    sharp = cv2.addWeighted(contrast, 1.75, blur, -0.75, 0)
    sharp = cv2.medianBlur(sharp, 3)
    edges = cv2.Canny(sharp, 35, 110)
    return cv2.addWeighted(sharp, 0.78, edges, 0.22, 0)


def preprocess_temperature_for_sift(temperature):
    """Prepare a thermal crop for SIFT feature detection."""

    image = normalize_to_uint8(temperature, 1.0, 99.0)
    contrast = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(image)
    denoised = cv2.bilateralFilter(contrast, 7, 35, 35)
    blur = cv2.GaussianBlur(denoised, (0, 0), 1.0)
    sharp = cv2.addWeighted(denoised, 1.8, blur, -0.8, 0)
    gradient_x = cv2.Sobel(sharp, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(sharp, cv2.CV_32F, 0, 1, ksize=3)
    gradient = normalize_to_uint8(cv2.magnitude(gradient_x, gradient_y), 2.0, 98.0)
    return cv2.addWeighted(sharp, 0.72, gradient, 0.28, 0)


def preprocess_temperature_for_loftr(temperature):
    """Prepare a thermal crop for LoFTR matching."""

    image = normalize_to_uint8(temperature, 1.0, 99.0)
    contrast = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(image)
    denoised = cv2.bilateralFilter(contrast, 7, 30, 30)
    gradient_x = cv2.Sobel(denoised, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(denoised, cv2.CV_32F, 0, 1, ksize=3)
    gradient = normalize_to_uint8(cv2.magnitude(gradient_x, gradient_y), 2.0, 98.0)
    return cv2.addWeighted(denoised, 0.78, gradient, 0.22, 0)


def build_roi_mask(shape, roi=None, valid_mask=None, border=0, label="ROI"):
    """Create a uint8 mask from an optional `(x, y, width, height)` ROI."""

    height, width = shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)

    if roi is None:
        x0 = int(border)
        y0 = int(border)
        x1 = width - int(border)
        y1 = height - int(border)
    else:
        x, y, roi_width, roi_height = roi
        x0 = max(0, int(x))
        y0 = max(0, int(y))
        x1 = min(width, int(x + roi_width))
        y1 = min(height, int(y + roi_height))

    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"Invalid {label}: {roi}")

    mask[y0:y1, x0:x1] = 255

    if valid_mask is not None:
        mask = cv2.bitwise_and(mask, valid_mask.astype(np.uint8))

    return mask


def crop_roi(image, roi):
    """Crop an image/matrix using an ROI that is clipped to image bounds."""

    x, y, roi_width, roi_height = roi
    height, width = image.shape[:2]
    x0 = max(0, int(x))
    y0 = max(0, int(y))
    x1 = min(width, int(x + roi_width))
    y1 = min(height, int(y + roi_height))

    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"Invalid ROI: {roi}")

    return image[y0:y1, x0:x1]


def save_overlay(first, second, path):
    """Save a 50/50 visual overlay after robust uint8 normalization."""

    first_u8 = first if first.dtype == np.uint8 else normalize_to_uint8(first)
    second_u8 = second if second.dtype == np.uint8 else normalize_to_uint8(second)
    overlay = cv2.addWeighted(first_u8, 0.5, second_u8, 0.5, 0)
    cv2.imwrite(str(path), overlay)


def save_heatmap(data, path, title, label, cmap="inferno", symmetric=False):
    """Save a matplotlib heatmap for a thermal array or delta-T matrix."""

    plt.figure(figsize=(10, 6))
    if symmetric:
        limit = max(float(np.nanpercentile(np.abs(data), 98)), 0.1)
        plt.imshow(data, cmap="coolwarm", vmin=-limit, vmax=limit)
    else:
        plt.imshow(data, cmap=cmap)

    plt.colorbar(label=label)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
