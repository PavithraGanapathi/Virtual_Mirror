# Virtual Mirror Saree Draping POC

Day 1 foundation for a 2D virtual saree draping prototype. The app detects a customer's pose, measures simple upper-body geometry, then places a transparent, pre-draped saree PNG on the image.

## Run locally

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run starter_virtual_mirror.py
```


## Day 1: concepts validated

1. **MediaPipe Pose** detects a human pose and returns normalized landmark coordinates.
2. **Landmarks 11, 12, 23 and 24** are the customer's left shoulder, right shoulder, left hip and right hip.
3. **Normalized coordinates become pixels** by multiplying `x` by image width and `y` by image height.
4. **Shoulder width** is the Euclidean distance between the two shoulder points.
5. **Shoulder tilt** is calculated with `atan2(dy, dx)` on the shoulder line and is used to rotate the garment.
6. **Torso height** is the distance between shoulder midpoint and hip midpoint.
7. The baseline drape is a rigid **resize → rotate → paste** pipeline with the PNG alpha channel used as the mask.
8. **Hip width** is the Euclidean distance between hip landmarks. It is now measured and displayed; it will inform a later geometry/warping stage rather than distort the current pre-draped garment asset.

## Current pipeline

```text
Customer image
  → MediaPipe Pose
  → shoulders and hips in pixel coordinates
  → body metrics
  → resize saree by shoulder width
  → rotate by shoulder tilt
  → anchor at the left shoulder
  → alpha composite
```

## Day 1 code updates

Only the following additions were made to the starter baseline.

### 1. Hip-width measurement

```python
hip_width = math.hypot(
    lh[0] - rh[0],
    lh[1] - rh[1]
)
```

The `metrics` dictionary now contains `hip_width_px`, and Streamlit displays it with the existing pose telemetry.

### 2. Named garment-anchor constants

The saree images in `assets/` are already pre-draped cutouts, not flat fabric textures. Their internal shoulder location is therefore expressed explicitly:

```python
SAREE_SHOULDER_ANCHOR_X = 0.55
SAREE_SHOULDER_ANCHOR_Y = 0.15
```

The detected anatomical left shoulder is used as the placement anchor. The garment keeps its original aspect ratio; it is not warped or stretched.

### 3. Sample calibration defaults

The sample image's manually verified starting values are:

```text
Scale multiplier: 0.95
Horizontal offset: -60 px
Vertical offset: +15 px
Opacity: 1.00
```

The UI retains all controls so the values can be inspected and adjusted during testing.

## Sample-image observation

For the bundled sample image, MediaPipe produced:

| Metric | Value |
| --- | ---: |
| Shoulder width | 272.0 px |
| Hip width | 162.0 px |
| Torso height | 394.0 px |
| Shoulder tilt | -0.63° |

The sample calibration can be represented as body-relative starting values:

```text
X offset ≈ -0.22 × shoulder width
Y offset ≈  0.04 × torso height
```

This identifies the first relationship between pose telemetry and drape placement. The detected shoulder coordinates already handle where the person stands in the frame; offsets are asset-alignment corrections, not replacements for body location.

## Day 1 conclusion and next step

The POC reliably extracts the relevant pose points and produces a calibrated result for the supplied sample using a pre-draped RGBA asset. A narrower shoulder measurement naturally produces a smaller on-screen garment because sizing is shoulder-width driven.

The current constants are a **starting calibration**, not a universal per-person model. The next step is to collect several uploaded/webcam examples, record the manually preferred scale/X/Y values for each, and express any consistent corrections as proportions of shoulder width, torso height, and body ratios such as `hip_width / shoulder_width`. A perspective warp should be introduced only for a flat fabric asset or a separately warpable pallu section; warping the entire pre-draped PNG would distort its folds and borders.


# Day 2 — Automatic Pose-Relative Saree Calibration

## Overview

Day 1 established a working 2D saree overlay using MediaPipe Pose and manual
Streamlit controls. Day 2 replaces the fixed sample-only calibration values
with pose-relative formulas, allowing the saree to adapt to different image
sizes, customer positions, shoulder widths, torso lengths, and shoulder tilt.

```text
Customer image
  → MediaPipe Pose landmarks
  → body telemetry
  → automatic scale, X, and Y corrections
  → resize + rotate + shoulder-anchor saree
  → alpha composite output
```

## Landmarks and Telemetry

The engine uses MediaPipe landmarks `11` and `12` for the customer’s left and
right shoulders, and `23` and `24` for the left and right hips. Normalized
coordinates are converted into image pixels before geometry is calculated.

The Day 2 telemetry includes shoulder width, hip width, torso height, shoulder
tilt, frame dimensions, hip/shoulder ratio, and torso/shoulder ratio.

For the supplied sample image:

| Measurement | Value |
| --- | ---: |
| Frame | 896 × 1200 px |
| Shoulder width | 272.0 px |
| Hip width | 162.0 px |
| Torso height | 394.0 px |
| Shoulder tilt | -0.63° |

## Automatic Calibration

The Day 1 visually verified sample result used scale `0.95`, X offset `-60 px`,
Y offset `+15 px`, and opacity `1.0`. Day 2 converts those values into
body-relative formulas:

```python
target_width = shoulder_width * 2.2 * 0.95
offset_x = (-60 / 272) * shoulder_width
offset_y = (15 / 394) * torso_height
opacity = 1.0
```

Shoulder width controls the apparent garment size. Torso height controls the
vertical asset-alignment correction. The detected shoulder coordinate itself
already moves the saree when the customer is positioned elsewhere in the frame;
the X/Y offsets only correct the internal alignment of the saree PNG.

## Shoulder Anchor and Rotation

The saree assets are transparent, pre-draped PNG cutouts with existing folds,
pleats, pallu, and borders. The code preserves their aspect ratio and anchors
their embedded shoulder point to the customer’s anatomical left shoulder.

```python
paste_x = shoulder_x - (target_width * 0.55) + offset_x
paste_y = shoulder_y - (target_height * 0.15) + offset_y
rotated_saree = resized_saree.rotate(-shoulder_tilt_deg, expand=True)
```

This allows the pallu to follow a slanted shoulder pose while avoiding distortion
of the garment’s existing folds and gold border. The alpha channel of the PNG is
used as the paste mask, so only the visible saree pixels are composited.

## Result and Next Direction

The Day 2 version produces automatic, plausible draping on the bundled sample
and on a tested slanted pose. It also shows calculated garment size, placement
point, frame dimensions, and body ratios for further debugging and calibration.

This remains a rigid 2D `resize → rotate → paste` POC. Hip width is collected
for future body-proportion work but does not yet warp the pre-draped asset.
The next step is to test more photos, record ideal corrections, refine the
proportional formulas, and reserve perspective warping for a flat-fabric or
separate pallu asset.
