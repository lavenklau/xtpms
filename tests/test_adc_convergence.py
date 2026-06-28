"""
Discrete ADC convergence test: error vs mesh size h.

For each of the three revolution surfaces (k=π, 2π, 3π), generate meshes
at progressively finer resolutions, compute the discrete ADC using the
xtpms C++ solver, and compare against the high-precision analytical
reference values.

Outputs:
  - CSV data file
  - Log-log convergence plot (h: coarse → fine left → right)
"""
import os
import subprocess
import sys
import json
import tempfile

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '/opt/data/paper-tpms-heat/xtpms/tests')
from rev_surface_mesh import SURFACES, ADC_REF, generate_mesh, compute_resolution

# ── Configuration ──────────────────────────────────────────────────

XTPMS_BIN = '/opt/data/paper-tpms-heat/xtpms/build/xtpms'
TARGET_HS = [0.4, 0.2, 0.1, 0.05, 0.025, 0.0125]  # coarse → fine

OUTPUT_DIR = '/opt/data/paper-tpms-heat/xtpms/tests/convergence_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Run discrete ADC on a mesh via xtpms CLI ───────────────────────

def run_xtpms_compute(obj_path):
    """Run `xtpms compute --half-period 1,1,1 <obj>` and parse kA matrix.

    Returns the 3x3 ADC matrix (kappa=1).
    """
    cmd = [XTPMS_BIN, 'compute', '-i', obj_path, '--half-period', '1,1,1']
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        print(f"  ERROR: xtpms failed: {result.stderr[:500]}")
        return None

    # Parse output: "kA =\n  ... 3x3 matrix ...\n"
    lines = result.stdout.strip().split('\n')
    kA = None
    for i, line in enumerate(lines):
        if line.strip().startswith('kA'):
            # Next 3 lines are the matrix rows
            kA = []
            for j in range(1, 4):
                if i + j < len(lines):
                    row = [float(x) for x in lines[i + j].split()]
                    if len(row) == 3:
                        kA.append(row)
            if len(kA) == 3:
                kA = np.array(kA)
                break

    return kA


# ── Main convergence test ──────────────────────────────────────────

def main():
    print("=" * 80)
    print("Discrete ADC Convergence Test")
    print("  Surface of revolution: R(x) = (2 + cos(kx))/4")
    print("  ADC direction: x-axis (e^1)")
    print("  Reference values from Mathematica (30 digits)")
    print("=" * 80)

    results = {}  # name -> list of (target_h, actual_h, kA_11, error, n_verts, n_faces)

    for name, surf in SURFACES.items():
        k_pi = surf.k / np.pi
        ref = ADC_REF[name]

        print(f"\n{'─' * 80}")
        print(f"Surface: {name} (k = {k_pi:.0f}π)")
        print(f"  Reference ADC(e^1) = {ref:.15f}")
        print(f"{'─' * 80}")
        print(f"  {'target_h':>10} {'actual_h':>10} {'nv':>8} {'nf':>8} "
              f"{'kA_11':>14} {'error':>14} {'rel_err':>12}")
        print(f"  " + "-" * 90)

        results[name] = []
        for target_h in TARGET_HS:
            # Generate mesh
            N_x, N_theta = compute_resolution(surf, target_h)
            mesh = generate_mesh(surf, N_x, N_theta)

            # Save to temp OBJ
            obj_path = os.path.join(OUTPUT_DIR, f'{name}_h{target_h:.4f}.obj')
            mesh.save_obj(obj_path)

            # Run xtpms compute
            kA = run_xtpms_compute(obj_path)
            if kA is None:
                print(f"  {target_h:10.4f}  FAILED")
                continue

            # kA(0,0) = ADC along e^1
            kA_11 = kA[0, 0]
            error = abs(kA_11 - ref)
            rel_err = error / ref

            # Compute actual mesh size h (mean diagonal length)
            v, f_arr = mesh.vertices, mesh.faces
            period = 2.0
            e0 = v[f_arr[:, 1]] - v[f_arr[:, 0]]
            e1 = v[f_arr[:, 2]] - v[f_arr[:, 1]]
            e2 = v[f_arr[:, 0]] - v[f_arr[:, 2]]
            e0 = e0 - period * np.round(e0 / period)
            e1 = e1 - period * np.round(e1 / period)
            e2 = e2 - period * np.round(e2 / period)
            e0l = np.linalg.norm(e0, axis=1)
            e1l = np.linalg.norm(e1, axis=1)
            e2l = np.linalg.norm(e2, axis=1)
            # Diagonal = hypotenuse of isosceles right triangle
            # Tri1: e0=x-leg, e1=θ-leg, e2=diagonal
            # Tri2: e0=diagonal, e1=x-leg, e2=θ-leg
            n = len(f_arr)
            is_tri1 = np.arange(n) % 2 == 0
            diagonals = np.where(is_tri1, e2l, e0l)
            actual_h = float(np.mean(diagonals))

            results[name].append({
                'target_h': target_h,
                'actual_h': actual_h,
                'n_verts': mesh.n_vertices,
                'n_faces': mesh.n_faces,
                'kA_11': kA_11,
                'error': error,
                'rel_err': rel_err,
            })

            print(f"  {target_h:10.4f} {actual_h:10.6f} {mesh.n_vertices:8d} {mesh.n_faces:8d} "
                  f"{kA_11:14.10f} {error:14.6e} {rel_err:12.6e}")

        # Clean up OBJ files
        for r in results[name]:
            obj_path = os.path.join(OUTPUT_DIR, f'{name}_h{r["target_h"]:.4f}.obj')
            if os.path.exists(obj_path):
                os.remove(obj_path)

    # ── Save CSV ────────────────────────────────────────────────
    csv_path = os.path.join(OUTPUT_DIR, 'convergence_data.csv')
    with open(csv_path, 'w') as f:
        f.write('surface,target_h,actual_h,n_verts,n_faces,kA_11,error,rel_err\n')
        for name, data_list in results.items():
            for r in data_list:
                f.write(f'{name},{r["target_h"]:.6f},{r["actual_h"]:.6f},'
                        f'{r["n_verts"]},{r["n_faces"]},'
                        f'{r["kA_11"]:.15e},{r["error"]:.15e},{r["rel_err"]:.15e}\n')
    print(f"\nCSV saved: {csv_path}")

    # ── Convergence plot ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = {'k1pi': '#0072B2', 'k2pi': '#D55E00', 'k3pi': '#009E73'}
    labels = {'k1pi': r'$k = \pi$', 'k2pi': r'$k = 2\pi$', 'k3pi': r'$k = 3\pi$'}

    for name, data_list in results.items():
        if not data_list:
            continue
        hs = np.array([r['actual_h'] for r in data_list])
        errs = np.array([r['error'] for r in data_list])

        # Sort: coarse (large h) → fine (small h): LEFT → RIGHT
        sort_idx = np.argsort(hs)[::-1]  # descending = coarse first
        hs = hs[sort_idx]
        errs = errs[sort_idx]

        ax.loglog(hs, errs, 'o-', color=colors[name], label=labels[name],
                  markersize=6, linewidth=1.5)

    # Reference slope lines: O(h^2)
    # Place it in the middle of the plot
    h_min = min(min(r['actual_h'] for r in dl) for dl in results.values() if dl)
    h_max = max(max(r['actual_h'] for r in dl) for dl in results.values() if dl)
    h_ref = np.array([h_max * 0.5, h_max * 0.5 * 0.1])
    # Align with k1pi error at the first point
    if results.get('k1pi'):
        e0 = results['k1pi'][0]['error']
        scale = e0 / (h_max ** 2)
        e_ref = scale * h_ref ** 2
        ax.loglog(h_ref, e_ref, 'k--', alpha=0.4, linewidth=1)
        ax.text(h_ref[0] * 1.2, e_ref[0] * 1.5, r'$O(h^2)$', fontsize=12, alpha=0.6)

    ax.set_xlabel(r'Mesh size $h$', fontsize=13)
    ax.set_ylabel(r'$|\kappa_A^h - \kappa_A|$', fontsize=13)
    ax.set_title(r'Discrete ADC convergence: error vs mesh size', fontsize=14)
    ax.legend(fontsize=12, loc='lower right')
    ax.grid(True, which='both', alpha=0.3)

    # x-axis: coarse (large h) on LEFT, fine (small h) on RIGHT
    ax.invert_xaxis()

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, 'convergence_plot.png')
    plt.savefig(plot_path, dpi=200)
    print(f"Plot saved: {plot_path}")

    # ── Summary ────────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("Summary: convergence rate estimation")
    print("=" * 80)
    for name, data_list in results.items():
        if len(data_list) < 2:
            continue
        hs = np.array([np.log(r['actual_h']) for r in data_list])
        es = np.array([np.log(r['error']) for r in data_list])
        # Linear fit: log(e) = p * log(h) + c
        p = np.polyfit(hs, es, 1)
        print(f"  {name}: slope = {p[0]:.3f}  (expected ~2.0 for O(h^2))")


if __name__ == '__main__':
    main()
