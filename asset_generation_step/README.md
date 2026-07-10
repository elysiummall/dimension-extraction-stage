# Asset Generation — Stage 3 Technical Brief

> **Role:** Stage 3 — 3D Asset Generation
> **Input:** Stage 2 dimensional profile JSON + the segmentation masks Stage 2 already produced
> **Output:** One final, real-world-scaled **`.glb`** file per product, plus an asset validation report
> **Hardware:** iPhone 17 (no LiDAR) + MacBook Pro M1 8GB RAM — same as Stage 2, all-free stack

---

## Goal

Turn each measured product into a 3D mesh whose bounding box matches the Stage 2
measurements (width/height/depth in cm), exported as a single `.glb` file with correct
units, orientation, and embedded provenance metadata.

**Chosen approach: we build the mesh ourselves** from the pipeline's own outputs —
no external generator apps, no AI mesh models. Stage 2 already produces the two
ingredients a classic CV reconstruction needs: clean product **silhouette masks**
(front + side, from Grounding DINO + SAM 2) and exact **real-world dimensions**.
Route C below combines them. Routes A and B are kept only as fallbacks for products
whose shape silhouettes can't capture.

---

## What Stage 2 hands us (the input contract)

| Input | Where it comes from | Used for |
|-------|--------------------|----------|
| `measurements_cm` (width/height/depth) | `measurement_extraction_step/output/measurements_<subject_id>.json` | Exact size of the reconstruction volume |
| `error_estimates_cm` | same JSON | Pass/fail tolerance in validation |
| `height_cross_check.consistent` | same JSON | If `false`, don't generate yet — send it back to Stage 2 |
| **Front + side masks** | `instance_segmentation_step/output/` (per-frame SAM 2 masks) | The silhouettes that get carved into a 3D shape |
| Front + side photos | Stage 2 capture sets | Vertex colors / texture projection |
| `model_versions`, `frame_count`, `notes` | same JSON | Copied into the `.glb` metadata for provenance |

Also keep the latest `accuracy_validation_step/output/accuracy_report.json` at hand:
Stage 2's *ground-truth-checked* accuracy is what defines how tightly we can promise
the asset fits reality.

**Unit conventions to burn into memory now** (most bugs in this stage are unit/axis bugs):
- Stage 2 JSON → **centimetres**
- glTF/GLB spec → **metres**, **+Y up**, **+Z toward the viewer (front)**
- So: `width → X`, `height → Y`, `depth → Z`, all divided by 100.

---

## Route C — Silhouette Reconstruction (primary)

### The idea: shape-from-silhouette ("visual hull")

Think of the front mask as a cookie cutter pushed straight back through a block of
clay that is exactly `width × height × depth` in size. Now push the side mask through
the same block from the side. Keep only the clay that survived **both** cuts. What
remains is a 3D shape whose front outline, side outline, and overall dimensions are
all exactly what Stage 2 measured.

Why this route wins:

- **Dimensions are exact by construction.** The reconstruction volume is *built* at
  the measured size — there is no after-the-fact scaling step that can go wrong.
- **The shape is measured, not guessed.** No AI hallucinating unseen faces; the
  geometry comes from our own SAM 2 masks.
- **It's our code, our pipeline.** NumPy + scikit-image + trimesh — Python libraries
  in the same spirit as the OpenCV/NumPy work in Stage 2. Runs in seconds on the M1,
  no GPU, no RAM pressure, fully deterministic and re-runnable.
- **No new capture needed.** It consumes the front/side sets Stage 2 already has.

### Honest limitations (decide fallback per product)

- **Concavities are invisible to silhouettes.** A dent, a mug's interior, a hollow —
  if it doesn't change the outline, the hull fills it in solid.
- **Two views can't fully pin down curvature** — a sphere comes out slightly "puffy"
  (the intersection of two rounded extrusions). Outline dimensions stay exact.
  Adding a **top view** capture set sharpens this a lot and the pipeline already
  knows how to capture + mask a view.
- **Back/side texture is faked** (mirrored or stretched from the photos we have).
  Same weakness as AI generation — but here at least the *shape* is real.

Good fits: bottles, boxes, snowglobe, shoes, most convex-ish products.
Bad fits: wire whisks, open-handled mugs, heavily concave items → Route A/B fallback.

### Extra views are mask-only (cheap)

Carving needs each view's **mask only** — real-world scale comes entirely from the
front/side `measurements_cm`, because every mask is cropped to its tight bounding box
and stretched onto the corresponding face of the already-sized voxel grid. So when a
product needs a **top view** (or any additional carving view):

- Run it through **capture → undistort + segmentation only**
  (`VIEW=top make instance-segmentation`, then stop)
- **Skip** depth estimation and measurement extraction — the two slowest steps
- The **A4 sheet isn't even required** in extra-view frames (nothing is measured
  from them), and 1–2 sharp frames are enough — no 5-frame averaging needed
- Capture note: shoot as close to straight-down (perpendicular) as you can; a tilted
  top view distorts the silhouette's *shape*. Size is immune either way — the bbox
  normalization absorbs it — but shape is the whole point of the extra view

The same logic applies to the front/side masks used in carving: the full pipeline ran
on those sets to produce *measurements*; the carve itself only reads the masks + JSON.

### The script to build: `build_silhouette_mesh.py`

Behaviour spec, step by step:

1. **Load inputs** — Stage 2 JSON, the best front mask and best side mask (sharpest
   frame; the segmentation output JSON has per-frame confidence to pick by).
2. **Prepare masks** — crop each to its tight bounding box (`get_mask_tight_bbox`,
   same as the measurement step), clean specks/holes with OpenCV morphology
   (`cv2.morphologyEx`, open then close).
3. **Build the voxel grid** — a 3D boolean array (start at 256³) representing a box
   of exactly `width × height × depth` metres from `measurements_cm`.
4. **Carve** — a voxel survives if its (x, y) projects inside the front mask **and**
   its (z, y) projects inside the side mask. One vectorized NumPy expression
   (resample each mask to the grid's face resolution, then broadcast-AND), not loops.
5. **Mesh it** — `skimage.measure.marching_cubes` on the voxel grid → triangle mesh;
   light Laplacian smoothing (`trimesh.smoothing.filter_laplacian`) to remove the
   voxel staircase; decimate if over ~100k triangles.
6. **Orient + finish** — height along +Y, front facing +Z, origin at bottom-center
   of the bounding box (assets sit on the floor at y=0).
7. **Color (v1 → v2)** — v1: sample the front photo at each vertex's projected (x, y)
   → vertex colors; crude but fully automatic. v2: planar-project front + side photos
   as a real UV texture, blended by face normal direction.
8. **Embed provenance** — glTF `extras`: `subject_id`, full `measurements_cm` and
   `error_estimates_cm`, route (`"silhouette_hull"`), voxel resolution, mask frames
   used, Stage 2 `model_versions`, date.
9. **Export** — `work/<subject_id>_hull.glb` via trimesh.

Sanity expectation: because size is baked in at step 3, the validation step should
pass the dimension check *trivially*. If it doesn't, the bug is in axis mapping.

### Fallback routes (kept, not primary)

| | Route A — Photogrammetry (Apple Object Capture) | Route B — AI image-to-3D (TripoSR / Stable Fast 3D) |
|---|---|---|
| Use when | Product ships and has concavities Route C fills in | Quick draft, or Route A capture impractical |
| Geometry | Real, reconstructed | Plausible AI guess (mushy backs) |
| Needs | **New** 20–60 photo orbit capture, Xcode-built CLI, `.reduced` detail on 8GB RAM | One Stage 2 front frame, runs on CPU in minutes |
| Extra work | Output is arbitrary units/orientation → needs the cleanup + scaling steps below | Same |

Unlike Route C, both fallbacks output *unscaled, arbitrarily-oriented* meshes, so
they additionally require: Blender cleanup + canonical orientation, then
`scale_to_measurements.py` (compute per-axis scale factors from the measured dims;
if they agree within ~5% apply the median uniformly, else warn loudly — the
generator's proportions are wrong). Only build that script when a fallback product
first actually needs it.

---

## Software stack (all free)

| Tool | Purpose |
|------|---------|
| NumPy + OpenCV | Voxel grid, carving, mask cleanup (already installed) |
| scikit-image | Marching cubes (voxels → triangles) |
| trimesh + pygltflib | Smoothing, orientation, metadata, `.glb` export; validation script |
| Khronos glTF-Validator + gltfpack (meshoptimizer) | Spec-compliance check + final compression |
| https://gltf-viewer.donmccurdy.com | Visual check (drag-and-drop; macOS Quick Look can't open `.glb`) |
| *(fallbacks only)* Apple Object Capture / TripoSR / Blender | Routes A and B, per product |

### Install commands
```bash
pip install trimesh pygltflib scikit-image   # into the existing venv
brew install gltfpack
npm install -g gltf-validator                # or the web validator at github.khronos.org/glTF-Validator
```

---

## Pipeline overview

```
Stage 2 JSON + front/side masks + photos
       ↓
Silhouette reconstruction — voxel carve at measured size (build_silhouette_mesh.py)
       ↓                                                       [work/<id>_hull.glb]
Validation — bbox vs Stage 2 JSON, spec check, visual check (validate_glb.py)
       ↓
Optimization + final export (gltfpack)                         [output/<id>.glb]

(fallback per product: Route A/B generator → Blender cleanup →
 scale_to_measurements.py → rejoin at Validation)
```

---

## Validation (`validate_glb.py` — to build)

Never ship a file this script hasn't passed. It re-loads the *final* `.glb` and checks:

1. **Dimensions:** bbox extents vs `measurements_cm`, per axis. Tolerance:
   `max(0.5cm, 2 × error_estimates_cm[axis])` — and always within Stage 2's global
   2cm target. Catches axis mix-ups and cm/m bugs, the two most likely failures
2. **Units/orientation:** height is along +Y; no dimension is 100× off (the classic)
3. **Integrity:** watertight/manifold status reported; triangle count; colors or
   textures present (textures ≤2048px)
4. **Spec compliance:** shells out to `gltf-validator`, fails on errors
5. Writes `output/<subject_id>_asset_report.json` — the Stage 3 analogue of Stage 2's
   accuracy report: pass/fail per check, measured-vs-asset table, tolerances used

Plus one human check in gltf-viewer next to a product photo: the script can't see
that the shape lost a feature or the colors landed wrong.

## Optimize + final export

```bash
gltfpack -i work/<id>_hull.glb -o output/<id>.glb -cc -tc
```
(-cc = mesh compression, -tc = texture compression.) Re-run `validate_glb.py` on the
*output* file — compression must not have moved the bbox. Delivery budget for web/AR:
**≤ 50k triangles, ≤ 2K textures, ≤ 10MB** per asset.

---

## Directory layout

```
asset_generation_step/
├── README.md                      ← this file
├── build_silhouette_mesh.py       to build — Route C reconstruction (the main event)
├── validate_glb.py                to build — validation gate
├── work/                          intermediates (<id>_hull.glb, debug voxel dumps)
├── output/
│   ├── <subject_id>.glb           ← the deliverable
│   └── <subject_id>_asset_report.json
├── capture_sets/<subject_id>/     Route A orbit photos, only if a fallback is needed
└── raw_models/<subject_id>/       Route A/B raw output, only if a fallback is needed
```

Suggested Makefile additions once the scripts exist:
`make build-asset SUBJECT=x`, `make validate-asset SUBJECT=x`.

**Version control:** Stage 3 lives in this same repo — it consumes Stage 2's output
paths directly and shares the venv, requirements, and Makefile. Following the repo's
existing convention (`output/` dirs ignored, deliverables like `ground_truth.json`
tracked): ignore `work/`, `capture_sets/`, `raw_models/`, and `__pycache__/`, but
**track `output/*.glb` and the asset reports** — they're the stage's deliverables
and sit under the 10MB budget.

---

## Definition of done (per product)

- [ ] `output/<subject_id>.glb` exists, opens in gltf-viewer, looks like the product
- [ ] Upright (+Y), front-facing (+Z), origin at bottom-center, real-world metres
- [ ] `validate_glb.py` passes — all three dimensions within tolerance of Stage 2 JSON
- [ ] `gltf-validator` reports zero errors
- [ ] Within the 50k-tri / 2K-texture / 10MB budget
- [ ] Provenance metadata embedded in `extras`
- [ ] `<subject_id>_asset_report.json` committed alongside

## Milestones

1. **M1 — Pilot (snowglobe):** it already has merged front+side measurements and
   masks. Build `build_silhouette_mesh.py` (geometry only, single flat color),
   build `validate_glb.py`, get the first validated `.glb`. Proves the whole chain
2. **M2 — Color:** vertex-color sampling from the front photo (v1), then evaluate
   whether projected UV textures (v2) are worth it
3. **M3 — Top view (optional but likely):** add a top-view capture set for one curved
   product — mask-only lane, segmentation step and stop — and carve with three
   silhouettes; compare the "puffiness" improvement
4. **M4 — Production pass:** all measured products; per product, decide Route C vs
   fallback based on concavity

## Failure modes to expect

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Asset 100× too big/small in a viewer | cm vs m mixed | Unit conversion at grid build; validator check 2 catches it |
| Width and depth swapped | Front/side masks assigned to wrong grid faces | Axis mapping in carve step; the bbox check catches it |
| Mesh looks like blocky staircase | Voxel resolution too low / no smoothing | Raise grid to 384–512³, apply Laplacian pass |
| Product's hollow/handle came out solid | Concavity — silhouettes can't see it | Expected; add top view, or fall back to Route A |
| Curved product looks inflated | Two-view hull limit | Add top view (M3); dims are still exact |
| Ragged/noisy hull edges | Speckled or leaky mask on the chosen frame | Pick a different frame; strengthen morphology cleanup |
| Colors misaligned on the mesh | Mask crop offset vs photo coordinates | Project with the same tight-bbox offsets used in carving |

## What to read / look into next

- "Visual hull" / shape-from-silhouette — the classic technique Route C implements
- `skimage.measure.marching_cubes` docs — the voxels→mesh step
- trimesh docs — `Trimesh`, `smoothing.filter_laplacian`, `visual.ColorVisuals`, `export`
- glTF 2.0 quick reference — units, coordinate system, `extras`
- meshoptimizer/gltfpack README — compression flags
- *(fallbacks)* Apple "Creating a Photogrammetry Command-Line App"; VAST-AI-Research/TripoSR