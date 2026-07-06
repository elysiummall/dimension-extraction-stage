import os
import json
import glob
from datetime import datetime

# ─── Configuration ────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # folder this script lives in, so paths work from any cwd
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output')
VIEWS_DIR  = os.path.join(OUTPUT_DIR, 'views')

# The front and side views both see the product's vertical extent, so their two
# independent height measurements should agree. A gap larger than this is a
# capture problem (different A4 placement between the two sets, wrong product
# distance, bad mask in one view) and gets flagged loudly.
HEIGHT_AGREEMENT_TOLERANCE_CM = 1.5

# ─── Merge one subject's views ──────────────────────────────────────────────────

def merge_subject(subject_id, views):
    """
    Combine per-view measurements into one full dimensional profile.

    Axis mapping:
      - front view:  silhouette width  → product WIDTH
                     silhouette height → product HEIGHT
      - side view:   silhouette width  → product DEPTH (front-to-back)
                     silhouette height → product HEIGHT again (cross-check only)

    The front view is canonical for height; the side view's height is used to
    verify the two capture sets are consistent with each other, which is a free
    accuracy check no single view can provide.
    """
    front = views.get('front')
    side  = views.get('side')

    if front is None:
        print(f"  [!] {subject_id}: no front view found — cannot merge (side view alone "
              f"gives depth/height but no width). Run the pipeline on a front capture set.")
        return None

    measurements = {
        'width':  front['measurements_cm']['width'],
        'height': front['measurements_cm']['height'],
        'depth':  side['measurements_cm']['width'] if side else None,
    }
    errors = {
        'width':  front['error_estimates_cm']['width'],
        'height': front['error_estimates_cm']['height'],
    }

    height_check = None
    notes = []
    if side:
        errors['depth'] = side['error_estimates_cm']['width']

        # Cross-check: both views measured the product's height independently
        height_gap = round(abs(front['measurements_cm']['height'] - side['measurements_cm']['height']), 2)
        height_check = {
            'front_height_cm': front['measurements_cm']['height'],
            'side_height_cm':  side['measurements_cm']['height'],
            'gap_cm':          height_gap,
            'consistent':      height_gap <= HEIGHT_AGREEMENT_TOLERANCE_CM,
        }
        if height_check['consistent']:
            notes.append(f"Front/side height cross-check agrees within {height_gap}cm.")
        else:
            notes.append(
                f"WARNING: front and side views disagree on height by {height_gap}cm "
                f"(> {HEIGHT_AGREEMENT_TOLERANCE_CM}cm tolerance) — the two capture sets are "
                f"inconsistent. Check A4 placement/coplanarity in both sets before trusting depth."
            )
    else:
        notes.append(
            "Depth (front-to-back) not measured — only a front view was captured. "
            "Capture a side-view set, run the pipeline with VIEW=side, and re-merge."
        )

    return {
        'subject_id':        subject_id,
        'merged_at':         datetime.now().isoformat(),
        'views_used':        sorted(views.keys()),
        'measurements_cm':   measurements,
        'error_estimates_cm': errors,
        'height_cross_check': height_check,
        'reference_object':  front['reference_object'],
        'capture_metadata':  {v: views[v].get('capture_metadata') for v in sorted(views.keys())},
        'model_versions':    front['model_versions'],
        'frame_count':       {v: views[v].get('frame_count') for v in sorted(views.keys())},
        'notes':             ' '.join(notes),
    }


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=== Step 6b — Merge Views into Final Measurements ===\n")

    view_files = sorted(glob.glob(os.path.join(VIEWS_DIR, 'measurements_*.json')))
    if not view_files:
        print(f"[!] No per-view results found in '{VIEWS_DIR}'.")
        print("    Run measurement_extraction.py first (it writes one file per view).")
        return

    # Group view files by subject_id
    subjects = {}
    for path in view_files:
        with open(path) as f:
            data = json.load(f)
        subjects.setdefault(data['subject_id'], {})[data['view']] = data

    # Optionally restrict to one subject: SUBJECT=product_003 make merge-views
    only = os.environ.get('SUBJECT')
    if only:
        subjects = {k: v for k, v in subjects.items() if k == only}
        if not subjects:
            print(f"[!] No view files found for subject '{only}'.")
            return

    for subject_id, views in subjects.items():
        print(f"Subject: {subject_id}  (views: {', '.join(sorted(views.keys()))})")
        merged = merge_subject(subject_id, views)
        if merged is None:
            print()
            continue

        out_path = os.path.join(OUTPUT_DIR, f'measurements_{subject_id}.json')
        with open(out_path, 'w') as f:
            json.dump(merged, f, indent=2)

        m = merged['measurements_cm']
        depth_str = f"{m['depth']} cm" if m['depth'] is not None else "not measured (no side view)"
        print(f"  width  : {m['width']} cm")
        print(f"  height : {m['height']} cm")
        print(f"  depth  : {depth_str}")
        if merged['height_cross_check'] and not merged['height_cross_check']['consistent']:
            print(f"  [!] {merged['notes']}")
        print(f"  Saved: {out_path}\n")

    print("Merge complete. The final measurements_<subject>.json files are what")
    print("Stage 3 consumes, and what accuracy_validation.py validates.")


if __name__ == '__main__':
    main()
