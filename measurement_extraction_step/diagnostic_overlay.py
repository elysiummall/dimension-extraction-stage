import cv2
import numpy as np
import os
import json

# ─── Configuration ────────────────────────────────────────────────────────────

SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))  # folder this script lives in, so paths work from any cwd
DEPTH_RESULTS = os.path.join(SCRIPT_DIR, '..', 'depth_estimation_step', 'output', 'depth_results.json')
OUTPUT_DIR    = os.path.join(SCRIPT_DIR, 'output', 'diagnostics')
SUBJECT_ID    = os.environ.get('SUBJECT', 'product_000')     # change per product (or pass SUBJECT=... via make)
VIEW          = os.environ.get('VIEW', 'front')

# Overlays are saved downscaled — full iPhone frames are ~4000px and the
# diagnostic only needs to be readable, not archival.
OVERLAY_MAX_WIDTH_PX = 1400

# Colors (BGR)
COLOR_MASK   = (0, 200, 0)      # mask fill/contour
COLOR_BBOX   = (0, 165, 255)    # tight bbox the measurement actually uses
COLOR_POINTS = (0, 0, 255)      # width/height measurement endpoints
COLOR_A4     = (255, 100, 0)    # detected A4 reference quad
COLOR_ROW    = (255, 0, 255)    # widest single row of the mask

# ─── Geometry (mirrors measurement_extraction.py exactly) ─────────────────────

def px_span_to_cm(px_span, depth_m, focal_px):
    """Pinhole conversion of a pixel span at a known depth into centimetres."""
    return px_span * depth_m / focal_px * 100


def widest_mask_row(mask):
    """
    Find the single row with the largest left-to-right mask extent.

    The measurement step uses the GLOBAL bbox extremes: if the leftmost and
    rightmost mask pixels sit on different rows (angled object, protrusion,
    bleed at one corner), the bbox width exceeds the width of any physical
    horizontal slice of the product. Comparing the two tells us how much of
    the width comes from a single coherent edge vs. scattered extremes.

    Returns (row_y, x_left, x_right, span_px) or None for an empty mask.
    """
    rows = np.where(mask.any(axis=1))[0]
    if len(rows) == 0:
        return None
    best = None
    for y in rows:
        xs = np.where(mask[y] > 0)[0]
        span = xs[-1] - xs[0]
        if best is None or span > best[3]:
            best = (int(y), int(xs[0]), int(xs[-1]), int(span))
    return best


# ─── Per-frame overlay ─────────────────────────────────────────────────────────

def render_frame(depth_result, out_dir):
    """
    Draw everything the measurement step used onto the original frame:
      - the product mask (green) and its tight bbox (orange)
      - the four bbox-edge midpoints width/height are measured between (red)
      - the widest single mask row (magenta) vs the bbox width
      - the detected A4 quad (blue) that anchors the pixel→cm scale
    plus the numbers behind them, so mask bleed, angled-silhouette inflation,
    and plane-gap problems are each visible at a glance.

    Returns a per-frame stats dict, or None if inputs are missing.
    """
    frame_path = depth_result['frame']
    mask_path  = depth_result['product'].get('mask_path')
    image = cv2.imread(frame_path)
    mask  = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) if mask_path else None
    if image is None or mask is None:
        print(f"  [!] Missing frame or mask for {os.path.basename(frame_path)} — skipping.")
        return None

    camera_matrix = np.array(depth_result['camera_matrix'])
    fx, fy   = camera_matrix[0, 0], camera_matrix[1, 1]
    depth_m  = depth_result['estimated_distance_m']

    coords = cv2.findNonZero(mask)
    if coords is None:
        print(f"  [!] Empty mask for {os.path.basename(frame_path)} — skipping.")
        return None
    x, y, w, h = cv2.boundingRect(coords)
    x1, y1, x2, y2 = x, y, x + w, y + h
    mid_y, mid_x = (y1 + y2) // 2, (x1 + x2) // 2

    width_cm  = px_span_to_cm(x2 - x1, depth_m, fx)
    height_cm = px_span_to_cm(y2 - y1, depth_m, fy)

    row = widest_mask_row(mask)
    row_width_cm = px_span_to_cm(row[3], depth_m, fx) if row else None

    # Depth-map median over the mask interior vs. the A4 pinhole distance —
    # a large gap is the coplanarity red flag (same check Step 6 prints).
    depth_map = np.load(depth_result['depth_map_path'])
    kernel = np.ones((max(15, min(mask.shape) // 100),) * 2, np.uint8)
    eroded = cv2.erode(mask, kernel)
    if cv2.countNonZero(eroded) == 0:
        eroded = mask
    depth_median_m = float(np.median(depth_map[eroded > 0]))
    plane_gap_cm   = (depth_median_m - depth_m) * 100

    # ── draw ──
    overlay = image.copy()
    overlay[mask > 0] = (0.55 * overlay[mask > 0] + 0.45 * np.array(COLOR_MASK)).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, COLOR_MASK, 3)

    cv2.rectangle(overlay, (x1, y1), (x2, y2), COLOR_BBOX, 3)
    cv2.line(overlay, (x1, mid_y), (x2, mid_y), COLOR_POINTS, 2)
    cv2.line(overlay, (mid_x, y1), (mid_x, y2), COLOR_POINTS, 2)
    for px, py in [(x1, mid_y), (x2, mid_y), (mid_x, y1), (mid_x, y2)]:
        cv2.circle(overlay, (px, py), 14, COLOR_POINTS, -1)

    if row:
        ry, rx1, rx2, _ = row
        cv2.line(overlay, (rx1, ry), (rx2, ry), COLOR_ROW, 3)

    a4 = depth_result.get('a4_sheet', {})
    if a4.get('corners_px'):
        pts = np.array(a4['corners_px'], dtype=np.int32)
        cv2.polylines(overlay, [pts], isClosed=True, color=COLOR_A4, thickness=4)

    # Legend + numbers, scaled so text stays readable after the downscale
    scale = OVERLAY_MAX_WIDTH_PX / overlay.shape[1]
    small = cv2.resize(overlay, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    lines = [
        f"bbox width  {width_cm:6.2f} cm   (orange box, red endpoints)",
        f"widest row  {row_width_cm:6.2f} cm   (magenta - single-slice width)" if row_width_cm else "",
        f"bbox height {height_cm:6.2f} cm",
        f"A4 distance {depth_m:.3f} m   depth-map median {depth_median_m:.3f} m   gap {plane_gap_cm:+.1f} cm",
    ]
    ty = 34
    for line in lines:
        if not line:
            continue
        cv2.putText(small, line, (12, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(small, line, (12, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
        ty += 28

    stem = os.path.splitext(os.path.basename(frame_path))[0]
    out_path = os.path.join(out_dir, f'{stem}_overlay.jpg')
    cv2.imwrite(out_path, small, [cv2.IMWRITE_JPEG_QUALITY, 88])

    return {
        'frame':            os.path.basename(frame_path),
        'bbox_width_cm':    round(width_cm, 2),
        'widest_row_cm':    round(row_width_cm, 2) if row_width_cm else None,
        'bbox_height_cm':   round(height_cm, 2),
        'plane_gap_cm':     round(plane_gap_cm, 1),
        'overlay':          out_path,
    }


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=== Measurement Diagnostic Overlay ===\n")
    print(f"Subject: {SUBJECT_ID}   View: {VIEW}\n")

    if not os.path.exists(DEPTH_RESULTS):
        print(f"[!] Depth results not found: '{DEPTH_RESULTS}' — run the pipeline through Step 5 first.")
        return

    with open(DEPTH_RESULTS) as f:
        depth_results = json.load(f)

    out_dir = os.path.join(OUTPUT_DIR, f'{SUBJECT_ID}_{VIEW}')
    os.makedirs(out_dir, exist_ok=True)

    stats = []
    for dr in depth_results:
        s = render_frame(dr, out_dir)
        if s:
            stats.append(s)
            row_str = f"  widest-row={s['widest_row_cm']:6.2f}" if s['widest_row_cm'] is not None else ""
            print(f"  {s['frame']:<18} width={s['bbox_width_cm']:6.2f}{row_str}  "
                  f"height={s['bbox_height_cm']:6.2f}  plane-gap={s['plane_gap_cm']:+5.1f} cm")

    if not stats:
        print("[!] No overlays produced.")
        return

    widths  = [s['bbox_width_cm'] for s in stats]
    heights = [s['bbox_height_cm'] for s in stats]
    gaps    = [s['plane_gap_cm'] for s in stats]
    rows    = [s['widest_row_cm'] for s in stats if s['widest_row_cm'] is not None]
    print("\n" + "─" * 60)
    print(f"bbox width : mean {np.mean(widths):.2f} cm  (min {min(widths):.2f} / max {max(widths):.2f})")
    if rows:
        print(f"widest row : mean {np.mean(rows):.2f} cm  — bbox exceeds it by {np.mean(widths) - np.mean(rows):+.2f} cm "
              f"(scattered extremes, not one physical edge, if large)")
    print(f"bbox height: mean {np.mean(heights):.2f} cm  (min {min(heights):.2f} / max {max(heights):.2f})")
    print(f"plane gap  : mean {np.mean(gaps):+.1f} cm  (depth-map median minus A4 distance; "
          f"large |gap| = A4 not coplanar with product front face)")
    print(f"\nOverlays saved to: {out_dir}")


if __name__ == '__main__':
    main()
