##### Thermal Alignment Helper Functions

This folder contains several alignment scripts that use different matching
methods:

* `MANUAL\_THERMAL\_POINT\_ALIGNMENT.py`
* `ORB\_ONLY\_BESTVERSION\_copy.py`
* `ORB\_ONLY\_THERMAL\_ALIGNMENT\_STABLE\_REFINE.py`
* `SIFT\_ONLY\_THERMAL\_ALIGNMENT.py`
* `LOFTR\_ONLY\_THERMAL\_ALIGNMENT.py`

The method-specific matching code stays inside each script, but shared IO,
crop, normalization, preprocessing, ROI, overlay, and heatmap functions now
live in `thermal\_alignment\_common.py`.

##### Functions Included

* `load\_gray`
* `load\_temperature`
* `load\_dji\_raw\_temperature`
* `crop\_image`
* `robust\_u8`
* `robust\_normalize\_to\_u8`
* `save\_overlay`
* `save\_heatmap`



##### **Shared functions**

* ###### **ThermalRawSpec**

Stores the shape and scale for DJI raw thermal files.

Use it once near the user settings:

```python
THERMAL\_RAW\_SPEC = ThermalRawSpec(width=640, height=512, temperature\_scale=10.0)
```

* ###### **load\_grayscale\_image(path)**

Loads a rendered JPG/PNG image as a single-channel grayscale image.

Use it for thermal JPGs that are only needed for display, overlays, or manual
point selection.

* ###### **load\_dji\_raw\_temperature(raw\_path, raw\_spec)**

Loads a DJI `.raw` int16 thermal file and converts it to degrees C using the
scale in `ThermalRawSpec`.

Use it for quantitative temperature work, such as delta-T heatmaps.  Do not use
the rendered JPG for temperature subtraction.

* ###### **crop\_xywh(image, crop, label)**

Crops an image or temperature matrix using `(x, y, width, height)`.

Use the same crop coordinates for the rendered thermal JPG and the raw
temperature matrix when they refer to the same camera frame.

* ###### **normalize\_to\_uint8(data, low\_percentile=1.0, high\_percentile=99.0)**

Converts a numeric image, usually temperature in degrees C, to `uint8` for
feature detection or visualization.  It clips extreme values using percentiles
so one hot/cold pixel does not dominate the contrast.

Use this whenever a matching algorithm needs an 8-bit image.

* ###### **preprocess\_temperature\_for\_orb(temperature)**

Builds an ORB-friendly feature image:

1. Robustly normalize temperature to `uint8`.
2. Apply CLAHE contrast enhancement.
3. Sharpen with unsharp masking.
4. Add a small Canny-edge contribution.

###### Use it in ORB scripts before detecting ORB keypoints.

* ###### **preprocess\_temperature\_for\_sift(temperature)**

Builds a SIFT-friendly feature image:

1. Robustly normalize temperature to `uint8`.
2. Apply CLAHE contrast enhancement.
3. Denoise with bilateral filtering.
4. Sharpen.
5. Add gradient magnitude to emphasize edges.

Use it in SIFT scripts before detecting SIFT keypoints.

* ###### **preprocess\_temperature\_for\_loftr(temperature)**

Builds a LoFTR-friendly feature image:

1. Robustly normalize temperature to `uint8`.
2. Apply slightly gentler CLAHE than SIFT/ORB.
3. Denoise with bilateral filtering.
4. Add gradient magnitude lightly.

Use it before converting the image to a LoFTR tensor.

* ###### **build\_roi\_mask(shape, roi=None, valid\_mask=None, border=0, label="ROI")**

Creates a mask for feature matching.  If `roi` is `None`, it uses the full
image, optionally leaving out a border.

Use it when you want feature detection/matching to focus on a specific part of
the crop.

* ###### **crop\_roi(image, roi)**

Crops a region of interest while clipping to image bounds.

Use it for local refinement scoring or validation regions.

* ###### **save\_overlay(first, second, path)**

Saves a 50/50 overlay after robust normalization.  This is for visual checking,
not quantitative temperature analysis.

Use it before and after alignment to confirm that edges and hot/cold structures
line up.

* ###### **save heatmap(data, path, title, label, cmap="inferno", symmetric=False)**

Saves a temperature or delta-temperature heatmap.

Use `symmetric=True` for delta-T maps so positive and negative changes are
visually balanced around zero.



##### **How to add a new alignment script**



1. Import the shared functions from `thermal\_alignment\_common.py`.
2. Create one `THERMAL\_RAW\_SPEC` using the camera width, height, and scale.
3. Load rendered JPGs with `load\_grayscale\_image`.
4. Load raw temperatures with `load\_dji\_raw\_temperature`.
5. Crop both rendered images and raw temperature arrays with `crop\_xywh`.
6. Pick exactly one preprocessing function for the matching method:

   * ORB: `preprocess\_temperature\_for\_orb`
   * SIFT: `preprocess\_temperature\_for\_sift`
   * LoFTR: `preprocess\_temperature\_for\_loftr`
7. Keep the method-specific matching and transform estimation inside the script.

## Important usage rule

Use the preprocessed `uint8` image only for finding the transform.  Apply the
final transform to the original cropped temperature arrays when calculating
temperature differences.  This keeps the output physically meaningful.

