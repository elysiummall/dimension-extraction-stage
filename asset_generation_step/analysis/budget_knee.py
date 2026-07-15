"""
Stage 3 experiment — dense polygon-budget sweep to find the fidelity knee.

Advisor's question: sweep many budgets and find the LOWEST face count that
keeps fidelity, then compare across products to see what number generalises.

Unlike fidelity_sweep.py (few budgets, renders, teaching artifacts), this
runs a dense grid with no renders and no per-level glb exports — just the
chamfer error curve. "Keeps fidelity" is made precise instead of eyeballed:

    1. the metric noise floor is measured FLOOR_REPEATS times (reference
       sampled against itself), giving floor mean + std
    2. a budget "keeps fidelity" if its mean error <= floor_mean +
       max(3 * floor_std, 2% of floor_mean)
    3. the KNEE is the smallest budget from which every larger budget also
       keeps fidelity (monotone, so one lucky noisy row can't win)

FACE_BUDGETS accepts comma-separated values and start:stop:step ranges,
e.g. "1000:20000:1000,50000:500000:5000". Budgets >= the reference face
count are skipped (decimation can't add faces — they'd be no-ops).

Usage:  SUBJECT=snowglobe venv/bin/python asset_generation_step/analysis/budget_knee.py

Output: work/lods/knee_<SUBJECT>.json    every row + floor + the knee
        work/lods/knee_<SUBJECT>.png     error-vs-budget curve
"""

import json
import os
import sys
import time

import numpy as np
import trimesh
from scipy.spatial import KDTree

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # asset_generation_step/
sys.path.insert(0, os.path.join(BASE_DIR, 'analysis'))
from fidelity_sweep import decimate, chamfer_mm, N_SAMPLES  # noqa: E402

SUBJECT_ID = os.environ.get('SUBJECT', 'product_000')
HULL_GLB   = os.path.join(BASE_DIR, 'work', f'{SUBJECT_ID}_hull.glb')
OUT_DIR    = os.path.join(BASE_DIR, 'work', 'lods')

# default grid: advisor's 50k-500k step 5k, plus a finer extension below 50k
# (earlier coarse sweeps put the snowglobe's knee under 20k — without the
# extension the whole requested range can come out flat and the knee unseen)
BUDGET_SPEC   = os.environ.get('FACE_BUDGETS',
                               '1000:20000:1000,20000:50000:2500,50000:500000:5000')
FLOOR_REPEATS = int(os.environ.get('FLOOR_REPEATS', '5'))


def parse_budgets(spec):
    out = set()
    for part in spec.split(','):
        part = part.strip()
        if ':' in part:
            a, b, s = (int(v) for v in part.split(':'))
            out.update(range(a, b + 1, s))
        elif part:
            out.add(int(part))
    return sorted(out)


def find_knee(budgets, errors, threshold):
    """Smallest budget from which every larger budget stays under threshold."""
    knee = None
    for b, e in sorted(zip(budgets, errors), reverse=True):
        if e <= threshold:
            knee = b
        else:
            break
    return knee


def plot_curve(rows, floor_mean, floor_std, threshold, knees, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    INK, MUTED, GRID = '#1a2530', '#5b6c7c', '#e3e9ee'
    SERIES, SERIES2 = '#35689e', '#9db8d2'

    budgets = [r['faces'] for r in rows]
    mean_e = [r['mean_err_mm'] for r in rows]
    p95_e = [r['p95_err_mm'] for r in rows]

    fig, ax = plt.subplots(figsize=(8.5, 4.6), dpi=130)
    ax.axhspan(floor_mean - 3 * floor_std, threshold, color=GRID, zorder=0,
               label='metric noise floor')
    ax.plot(budgets, p95_e, color=SERIES2, lw=1.4, label='p95 error')
    ax.plot(budgets, mean_e, color=SERIES, lw=2.0, label='mean error')

    def _mark(budget, label, dy, filled):
        if budget is None:
            return
        ke = next(r['mean_err_mm'] for r in rows if r['budget'] == budget)
        ax.scatter([budget], [ke], s=42, zorder=5, color=SERIES if filled
                   else 'white', edgecolors=SERIES, linewidths=1.4)
        ax.annotate(f'{label}: {budget:,}', (budget, ke),
                    xytext=(8, dy), textcoords='offset points',
                    fontsize=8.5, color=INK, fontweight='bold')

    _mark(knees.get('practical_excess_0.05mm'), 'practical knee', 12, True)
    _mark(knees.get('statistical_floor_plus_2pct'), 'statistical knee',
          -16, False)
    ax.set_xscale('log')
    ax.set_xlabel('polygon budget (faces, log scale)', color=MUTED, fontsize=9)
    ax.set_ylabel('chamfer error vs reference (mm)', color=MUTED, fontsize=9)
    ax.set_title(f'Where fidelity stops improving — {SUBJECT_ID}',
                 color=INK, fontsize=11, loc='left')
    ax.grid(True, which='both', color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.legend(frameon=False, fontsize=8, loc='upper right')
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    print("=" * 60)
    print("STAGE 3 — POLYGON-BUDGET KNEE FINDER")
    print("=" * 60)
    budgets = parse_budgets(BUDGET_SPEC)
    ref = trimesh.load(HULL_GLB, force='mesh')
    os.makedirs(OUT_DIR, exist_ok=True)
    budgets = [b for b in budgets if b < len(ref.faces)]
    print(f"Subject: {SUBJECT_ID}   reference: {len(ref.faces):,} faces   "
          f"{len(budgets)} budgets\n")

    print(f"[*] Measuring the noise floor ({FLOOR_REPEATS} repeats)...")
    ref_pts = ref.sample(N_SAMPLES)
    ref_tree = KDTree(ref_pts)
    floors = [chamfer_mm(ref_pts, ref_tree, ref)[0] for _ in range(FLOOR_REPEATS)]
    floor_mean, floor_std = float(np.mean(floors)), float(np.std(floors))
    threshold = floor_mean + max(3 * floor_std, 0.02 * floor_mean)
    print(f"    floor {floor_mean:.4f} mm  (std {floor_std:.4f})   "
          f"keep-fidelity threshold {threshold:.4f} mm")

    rows = []
    t_start = time.time()
    for i, budget in enumerate(sorted(budgets, reverse=True), 1):
        t0 = time.time()
        lod = decimate(ref, budget)
        mean_mm, p95_mm = chamfer_mm(ref_pts, ref_tree, lod)
        rows.append({'budget': budget, 'faces': len(lod.faces),
                     'vertices': len(lod.vertices),
                     'mean_err_mm': round(mean_mm, 4),
                     'p95_err_mm': round(p95_mm, 4)})
        mark = ' ' if mean_mm <= threshold else ' ^ above threshold'
        print(f"    [{i:>3}/{len(budgets)}] {budget:>8,}  "
              f"mean {mean_mm:.4f}  p95 {p95_mm:.4f}  "
              f"({time.time() - t0:.0f}s){mark}")

    rows.sort(key=lambda r: r['budget'])
    knee = find_knee([r['budget'] for r in rows],
                     [r['mean_err_mm'] for r in rows], threshold)

    # one strict knee is misleading when the curve has a long flat plateau —
    # report a tolerance ladder. The 'practical' criterion anchors to what
    # the pipeline can promise: excess error <= 0.05 mm is 5% of Stage 2's
    # ±1 mm measurement accuracy, i.e. far below anything downstream can see.
    budgets_sorted = [r['budget'] for r in rows]
    errs = [r['mean_err_mm'] for r in rows]
    knees = {
        'statistical_floor_plus_2pct': knee,
        'floor_plus_5pct':  find_knee(budgets_sorted, errs, floor_mean * 1.05),
        'floor_plus_10pct': find_knee(budgets_sorted, errs, floor_mean * 1.10),
        'practical_excess_0.05mm': find_knee(budgets_sorted, errs,
                                             floor_mean + 0.05),
    }

    out = {'subject': SUBJECT_ID, 'reference_faces': len(ref.faces),
           'n_samples': N_SAMPLES, 'budget_spec': BUDGET_SPEC,
           'floor_mean_mm': round(floor_mean, 4),
           'floor_std_mm': round(floor_std, 4),
           'threshold_mm': round(threshold, 4),
           'knee_budget': knee, 'knees': knees, 'rows': rows,
           'runtime_s': round(time.time() - t_start)}
    json_path = os.path.join(OUT_DIR, f'knee_{SUBJECT_ID}.json')
    with open(json_path, 'w') as f:
        json.dump(out, f, indent=2)
    png_path = os.path.join(OUT_DIR, f'knee_{SUBJECT_ID}.png')
    plot_curve(rows, floor_mean, floor_std, threshold, knees, png_path)

    print(f"\n[*] Wrote {json_path}")
    print(f"    curve: {png_path}")
    print("\n    Knees by fidelity criterion (lowest budget where every"
          " larger budget also passes):")
    for name, k in knees.items():
        print(f"      {name:<28} {k:,}" if k else
              f"      {name:<28} none in grid — extend upward")


if __name__ == '__main__':
    main()
