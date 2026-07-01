# Dimension Extraction Pipeline — Technical Brief

> **Role:** Stage 2 — Dimension Extraction  
> **Input:** Segmentation masks from Stage 1 (instance segmentation)  
> **Output:** Structured JSON dimensional profile per subject → handed to Stage 3 (asset generation)  
> **Hardware:** iPhone 17 (no LiDAR) + MacBook Pro M1 8GB RAM

---

## Goal

Extract accurate real-world body and/or product dimensions from standard RGB camera input, without a depth sensor, precise enough for a colleague to generate fitting 3D assets. Target error tolerance is **under 2cm** on major body dimensions.

---

## Software Stack (all free)

| Tool | Purpose |
|------|---------|
| Python 3.10+ | Core language (MacBook) |
| YOLOv8 (Ultralytics) | Instance segmentation |
| Depth Anything V2 | Monocular depth estimation |
| MediaPipe Pose | Skeletal keypoint detection (33 landmarks) |
| OpenCV | Camera calibration, image processing, homography |
| NumPy | Measurement calculations and averaging |
| Open3D | Optional — 3D point cloud visualisation during dev |

### Install commands
```bash
pip install ultralytics
pip install mediapipe
pip install opencv-python
pip install numpy
pip install open3d
pip install transformers torch torchvision  # for Depth Anything V2
```

---

## Pipeline Overview

```
Capture (iPhone 17)
       ↓
Camera Calibration (one-time, OpenCV)
       ↓
Instance Segmentation (YOLOv8)
       ↓
Pose Estimation — Skeletal Keypoints (MediaPipe)
       ↓
Monocular Depth Estimation (Depth Anything V2)
       ↓
Reference Object Scale Anchoring
       ↓
Measurement Extraction + Multi-frame Averaging
       ↓
Validation
       ↓
JSON Output → Stage 3 (Asset Generation)
```

---

## Step-by-Step Implementation

### Step 1 — Camera Calibration (one-time setup)

Calibrate the iPhone 17 camera to extract intrinsic parameters (focal length, principal point). These convert pixel distances to real-world distances and must be done before any measurement work.

**How:**
- Print a checkerboard pattern (9×6 or similar)
- Take 20–30 photos of it at varied angles with your iPhone
- Run OpenCV calibration to extract the intrinsic matrix and distortion coefficients
- Save to file — reuse for all future captures

```python
import cv2
import numpy as np
import glob

# Checkerboard dimensions (inner corners)
CHECKERBOARD = (8, 5)
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

obj_points, img_points = [], []

for fname in glob.glob('calibration_images/*.jpg'):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
    if ret:
        obj_points.append(objp)
        img_points.append(corners)

ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    obj_points, img_points, gray.shape[::-1], None, None
)

np.save('camera_matrix.npy', camera_matrix)
np.save('dist_coeffs.npy', dist_coeffs)
print("Calibration complete. Reprojection error:", ret)
```

**Target reprojection error:** under 1.0 (lower is better)

---

### Step 2 — Controlled Capture Protocol

Accuracy is won or lost at capture time. Follow this protocol strictly:

- Subject stands **1.5–2 metres** from camera
- Camera on tripod or stable surface — **no handheld**
- Camera **perpendicular** to subject — not angled
- **Plain, high-contrast background**
- **A4 sheet** (210mm × 297mm) held flat at subject's side or placed on ground — visible in every frame — this is your scale anchor
- Even, diffuse lighting — no harsh shadows
- Capture **minimum 5 frames** per subject, same pose — measurements will be averaged across frames

---

### Step 3 — Instance Segmentation

Segment the person/product and reference object separately per frame.

```python
from ultralytics import YOLO

model = YOLO('yolov8n-seg.pt')  # nano — fast on M1
results = model('input_frame.jpg')

# Access masks
for result in results:
    masks = result.masks       # segmentation masks
    boxes = result.boxes       # bounding boxes
    classes = result.names     # class labels
```

---

### Step 4 — Pose Estimation (Skeletal Keypoints)

Use MediaPipe to extract 33 body landmarks per frame. Measure distances between landmark pairs — not silhouette edges. This is the most important accuracy decision in the pipeline.

```python
import mediapipe as mp
import cv2

mp_pose = mp.solutions.pose

with mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,       # highest accuracy
    enable_segmentation=False,
    min_detection_confidence=0.5
) as pose:
    image = cv2.imread('input_frame.jpg')
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    landmarks = results.pose_landmarks.landmark

    # Key landmark indices (MediaPipe convention)
    # 11 = left shoulder, 12 = right shoulder
    # 23 = left hip,      24 = right hip
    # 27 = left ankle,    28 = right ankle
    # 15 = left wrist,    16 = right wrist
    # 0  = nose (head proxy)

    h, w, _ = image.shape
    def get_point(idx):
        lm = landmarks[idx]
        return np.array([lm.x * w, lm.y * h, lm.z])  # z is relative depth

    left_shoulder  = get_point(11)
    right_shoulder = get_point(12)
    left_hip       = get_point(23)
    right_hip      = get_point(24)
    left_ankle     = get_point(27)
    right_ankle    = get_point(28)
```

---

### Step 5 — Monocular Depth Estimation + Scale Anchoring

Run Depth Anything V2 to get a relative depth map, then anchor it to real-world scale using your reference object.

```python
from transformers import pipeline
from PIL import Image
import numpy as np

# Load depth model
depth_pipeline = pipeline(
    task="depth-estimation",
    model="depth-anything/Depth-Anything-V2-Small-hf"  # small = faster on M1
)

image = Image.open('input_frame.jpg')
depth_output = depth_pipeline(image)
depth_map = np.array(depth_output['depth'])

# --- Scale anchoring via reference object (A4 sheet) ---
# 1. Detect A4 sheet bounding box in image (via YOLO or colour detection)
# 2. Measure its pixel width in the image
# 3. Known real width = 210mm = 0.21m
# 4. Known real distance to camera = your fixed capture distance (e.g. 1.8m)
# 5. Using camera intrinsics: pixel_width = (real_width * focal_length) / distance
#    → solve for scale factor

focal_length = camera_matrix[0, 0]   # from calibration
known_real_width = 0.21              # A4 width in metres
a4_pixel_width = 320                 # example — measure from detected A4 box

estimated_distance = (known_real_width * focal_length) / a4_pixel_width
print(f"Estimated subject distance: {estimated_distance:.3f}m")
```

---

### Step 6 — Measurement Extraction

Convert keypoint pixel positions to real-world measurements using camera intrinsics and estimated depth.

```python
def pixel_to_world(px, py, depth_m, camera_matrix):
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    x = (px - cx) * depth_m / fx
    y = (py - cy) * depth_m / fy
    return np.array([x, y, depth_m])

def measure_distance_3d(p1_px, p2_px, depth_map, camera_matrix):
    d1 = depth_map[int(p1_px[1]), int(p1_px[0])]
    d2 = depth_map[int(p2_px[1]), int(p2_px[0])]
    p1_world = pixel_to_world(p1_px[0], p1_px[1], d1, camera_matrix)
    p2_world = pixel_to_world(p2_px[0], p2_px[1], d2, camera_matrix)
    return np.linalg.norm(p1_world - p2_world)

# Example measurements
shoulder_width = measure_distance_3d(left_shoulder[:2], right_shoulder[:2], depth_map, camera_matrix)
hip_width      = measure_distance_3d(left_hip[:2], right_hip[:2], depth_map, camera_matrix)
torso_height   = measure_distance_3d(
    ((left_shoulder[0]+right_shoulder[0])/2, (left_shoulder[1]+right_shoulder[1])/2),
    ((left_hip[0]+right_hip[0])/2,           (left_hip[1]+right_hip[1])/2),
    depth_map, camera_matrix
)
```

---

### Step 7 — Multi-Frame Averaging + Outlier Rejection

Run steps 3–6 across all 5+ frames, collect measurements, discard outliers, average remainder.

```python
import numpy as np

def robust_average(measurements):
    measurements = np.array(measurements)
    mean = np.mean(measurements)
    std  = np.std(measurements)
    # Keep only measurements within 1 std of mean
    filtered = measurements[np.abs(measurements - mean) < std]
    return float(np.mean(filtered)), float(np.std(filtered))

# Collect across frames
shoulder_measurements = []  # append per-frame result here

final_shoulder_width, shoulder_error = robust_average(shoulder_measurements)
print(f"Shoulder width: {final_shoulder_width*100:.1f}cm ± {shoulder_error*100:.1f}cm")
```

---

### Step 8 — JSON Output

Structure your output for Stage 3 (asset generation).

```python
import json
from datetime import datetime

output = {
    "subject_id": "subject_001",
    "captured_at": datetime.now().isoformat(),
    "capture_distance_m": estimated_distance,
    "frame_count": 5,
    "measurements_cm": {
        "shoulder_width":  round(final_shoulder_width * 100, 1),
        "hip_width":       round(final_hip_width * 100, 1),
        "torso_height":    round(final_torso_height * 100, 1),
        "total_height":    round(final_total_height * 100, 1),
        "arm_length":      round(final_arm_length * 100, 1),
        "inseam":          round(final_inseam * 100, 1),
        "head_width":      round(final_head_width * 100, 1),
    },
    "error_estimates_cm": {
        "shoulder_width": round(shoulder_error * 100, 2),
        "hip_width":      round(hip_error * 100, 2),
    },
    "reference_object": "A4_sheet_210x297mm",
    "model_versions": {
        "segmentation":       "yolov8n-seg",
        "depth_estimation":   "Depth-Anything-V2-Small",
        "pose_estimation":    "mediapipe_pose_complexity_2"
    }
}

with open(f'measurements_{output["subject_id"]}.json', 'w') as f:
    json.dump(output, f, indent=2)
```

---

### Step 9 — Validation

Before using in production, validate against tape measure ground truth.

- Measure 3–5 known objects or people with a tape measure
- Run them through your pipeline
- Compare output vs ground truth
- Log error per measurement type
- If consistently over 2cm error: check calibration accuracy first, then reference object detection reliability, then frame count

---

## Key Accuracy Levers (in order of impact)

1. **Camera calibration quality** — low reprojection error = everything downstream is more accurate
2. **Reference object reliability** — if A4 detection is flaky, scale anchoring breaks
3. **Frame averaging** — more frames = lower random error
4. **Capture protocol discipline** — fixed distance, perpendicular angle, good lighting
5. **MediaPipe model complexity** — use `model_complexity=2` for best keypoint accuracy

---

## What to Read / Look Into Next

- `cv2.calibrateCamera()` — OpenCV docs
- MediaPipe Pose landmark map — know what each of the 33 indices represents
- Depth Anything V2 GitHub — `https://github.com/DepthAnything/Depth-Anything-V2`
- `cv2.findHomography()` — for more robust reference object scale extraction
- Perspective-n-Point (PnP) — next-level technique for recovering 3D positions from 2D points: `cv2.solvePnP()`
- Apple Vision framework — if processing moves to iPhone directly later

---

## Handoff to Stage 3

Deliver per-subject JSON files (as above) to your colleague for asset generation. Include the `error_estimates_cm` block so they know measurement confidence per dimension and can factor tolerances into asset fitting.
