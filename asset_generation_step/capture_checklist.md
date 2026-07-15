# Capture checklist — adding a new product to the knee study

The advisor's generalization question ("what polygon budget holds across
products?") needs each new product to reach the point where a hull master
exists. That means Stage 2 first, then Stage 3 masks. Per product:

## 1. Pick products that stress DIFFERENT shape families

The snowglobe is smooth/rotationally symmetric — the easiest case for low
budgets. To learn what generalises, the next two should be harder:

- **one box-like product** (cereal box, board game, book): hard edges and
  flat faces — decimation loves these, but corners must survive.
  Build with the default `CROSS_SECTION=silhouette` (the `round` lathe
  would shave its corners off).
- **one complex/organic product** (sneaker, plush toy, ceramic figure):
  concavities and fine detail the silhouette hull partially misses —
  likely the highest knee of the three.

Matte, non-reflective products segment much cleaner than glass (no
FILL_HOLES workarounds needed).

## 2. Shoot the capture sets (same recipe as the snowglobe)

Stage 2 sets (with the A4 sheet, coplanar with the product face — flat on
the same wall/surface plane, NOT offset behind it):

- **front**: 10–15 frames
- **side**: 8–12 frames (note which side: `SIDE_FROM=left|right`)

Stage 3 extra views (mask-only, NO A4 needed, 3–7 frames each):

- **top**: stand at the product's front, shoot straight down
- **bottom** (only if the product can be rolled upside down about its
  front-back axis — front still facing you): shoot straight down

Same camera as calibration (iPhone 17 main lens), same resolution
throughout a set if possible; mixed resolutions are handled but avoid
mixing within a view.

## 3. Run the pipeline per product

```bash
# Stage 2 (produces measurements_<subject>.json — required):
#   segmentation -> depth -> measurement -> merge-views per capture set
# then file the product masks:
asset_generation_step/masks/<subject>/{front,side,top,bottom}/*_product_mask.png

# Stage 3 master (smoothing knobs recommended after the snowglobe study):
SUBJECT=<subject> SMOOTH_ITERATIONS=15 PROFILE_SMOOTH=4 SIL_SMOOTH=2 \
    VOLUME_SMOOTH=1.5 make build-asset            # add CROSS_SECTION=round only for round products

# the knee study:
SUBJECT=<subject> venv/bin/python asset_generation_step/analysis/budget_knee.py
```

Each product's curve lands in `work/lods/knee_<subject>.{json,png}` —
compare `knee_budget` across products to answer the generalization question.
