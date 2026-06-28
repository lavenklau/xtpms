#!/usr/bin/env python3
"""
Mathematica-derived integrand coefficients — full convergence test.

Runs the asymptotic accuracy test for all 3 surfaces of revolution
(k=π, 2π, 3π) using the corrected Mathematica formula with proper
volume form (√[det g] = R·√(1+R'²) with curvature corrections).

The integrand is a quadratic polynomial in {du/dx, du/dz_norm}:
    I = c0 + c1_x·u_x + c1_z·u_z + C2_xx·u_x² + C2_zz·u_z²

Discretized on an N × (M+1) grid (x periodic, z ∈ [-1,1] normalized),
central FD in x, one-sided FD at z-boundaries. Total energy from
sparse quadratic minimization, κ_ε = E / |Y|.

Usage:
    python mathematica_convergence.py
"""
import sys
sys.path.insert(0, '/opt/data/paper-tpms-heat/xtpms/tests')

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from scipy.integrate import quad
from semi_analytical_kappa import principal_curvatures
from rev_surface_mesh import SURFACES, ADC_REF
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Surface area (for rho_eps) ────────────────────────────────────
def surface_area(k):
    R  = lambda x: (2.0 + np.cos(k*x)) / 4.0
    Rp = lambda x: -k * np.sin(k*x) / 4.0
    Ax, _ = quad(lambda x: R(x)*np.sqrt(1.0 + Rp(x)**2), -1, 1)
    return 2.0 * np.pi * Ax

# ── Coefficient computation (Di Zhang, Mathematica, corrected) ────
def compute_coeffs_corrected(rp, rpp, z_norm, eps):
    """
    Compute integrand coefficients with proper volume form.

    Parameters (all can be scalars or ndarrays):
        rp     : R'(x)
        rpp    : R''(x)
        z_norm : normalized z, z ∈ [-1,1]  (physical z = ε·z_norm)
        eps    : shell half-thickness

    Returns dict: c0, c1_x, c1_z, C2_xx, C2_zz, C2_xz
    All outputs broadcast to z_norm.shape.
    """
    sq      = np.sqrt(1.0 + rp**2)
    opd     = 1.0 + rp**2               # 1 + R'²
    opd32   = opd**1.5                  # (1+R'²)^(3/2)
    R_vals  = (2.0 + np.cos(np.pi * z_norm * 0 + 1)) / 4.0  # placeholder—see note

    # Note: R(x) must be provided separately; the coefficients below
    # use f[x] = R(x).  For brevity f[x] is replaced by R_vals which
    # must match R(x).  In compute_kappa(), we provide R(x) explicitly.
    # The formulas below are the Mathematica output with f[x] = R(x).

    z_eps = eps * z_norm  # physical z (for readability)

    denom_h = sq + rp**2 * sq + rpp * z_eps       # √(1+R'²)·(1+R'²) + R''·z·ε
    bracket = sq * R_vals - z_eps                   # √(1+R'²)·R - z·ε
    # ^^^ R_vals must be the actual R(x). See compute_kappa() below.

    # ── c0 ──
    c0 = (2.0 * np.pi * eps * denom_h * bracket) / opd32

    # ── c1 ──
    c1_x = (4.0 * np.pi * eps * bracket) / opd
    c1_z = (4.0 * rp * np.pi * denom_h * bracket) / (opd**2)

    # ── C2 ──
    num_xx = 2.0 * np.pi * eps * (-sq * z_eps + opd * R_vals)
    C2_xx = num_xx / denom_h

    C2_zz = (2.0 * np.pi * denom_h * bracket) / (opd32 * eps)

    shape = np.broadcast(z_norm, rp).shape if hasattr(z_norm, 'shape') else ()
    return {
        'c0':    np.broadcast_to(c0,    z_norm.shape if hasattr(z_norm, 'shape') else ()).copy(),
        'c1_x':  np.broadcast_to(c1_x,  z_norm.shape if hasattr(z_norm, 'shape') else ()).copy(),
        'c1_z':  np.broadcast_to(c1_z,  z_norm.shape if hasattr(z_norm, 'shape') else ()).copy(),
        'C2_xx': np.broadcast_to(C2_xx, z_norm.shape if hasattr(z_norm, 'shape') else ()).copy(),
        'C2_zz': np.broadcast_to(C2_zz, z_norm.shape if hasattr(z_norm, 'shape') else ()).copy(),
        'C2_xz': np.zeros_like(z_norm) if hasattr(z_norm, 'shape') else 0.0,
    }

# ── Full compute on grid ──────────────────────────────────────────
from functools import lru_cache

def compute_kappa(k, eps, R_f, Rp_f, Rpp_f, N=5000, M=20):
    """
    Discretize on N×(M+1) grid, normalized z, assemble and solve.

    κ_ε = E(energy_min) / |Y|,  |Y| = 8.
    Uses the CORRECTED Mathematica coefficient formulas with proper
    volume form (R(x) passed explicitly in coeffs).
    """
    dx     = 2.0 / N
    dz_n   = 2.0 / M               # normalized z spacing

    x     = np.linspace(-1.0, 1.0, N, endpoint=False)
    z_n   = np.linspace(-1.0, 1.0, M + 1)

    rpv   = Rp_f(x)                # (N,)
    rppv  = Rpp_f(x)               # (N,)
    rv    = R_f(x)                 # (N,)

    sqv   = np.sqrt(1.0 + rpv**2)
    opdv  = 1.0 + rpv**2
    opd32v = opdv**1.5

    # Grid arrays
    rp2   = rpv[:, np.newaxis]     # (N, 1)
    rpp2  = rppv[:, np.newaxis]
    r2    = rv[:, np.newaxis]
    _, Z  = np.meshgrid(x, z_n, indexing='ij')  # (N, M+1)

    z_eps = eps * Z
    sq    = sqv[:, np.newaxis]
    opd   = opdv[:, np.newaxis]
    opd32 = opd32v[:, np.newaxis]

    # ── Corrected Mathematica coefficients (with R(x) = r2) ──
    denom_h = sq * opd + rpp2 * z_eps
    bracket = sq * r2 - z_eps

    c0 = (2.0 * np.pi * eps * denom_h * bracket) / opd32
    c1_x = (4.0 * np.pi * eps * bracket) / opd
    c1_z = (4.0 * rp2 * np.pi * denom_h * bracket) / (opd**2)
    C2_xx = (2.0 * np.pi * eps * (-sq * z_eps + opd * r2)) / denom_h
    C2_zz = (2.0 * np.pi * denom_h * bracket) / (opd32 * eps)

    # ── Quadrature weights ──
    wz = np.full(M + 1, dz_n)
    wz[0] = dz_n / 2.0
    wz[M] = dz_n / 2.0
    W = dx * wz[np.newaxis, :] * np.ones((N, 1))

    nv = N * (M + 1)
    def idx(i, j):
        return i * (M + 1) + j

    C0 = float(np.sum(c0 * W))
    C1 = np.zeros(nv)
    C2 = lil_matrix((nv, nv))
    FX = 1.0 / (2.0 * dx)

    for i in range(N):
        ip, im = (i + 1) % N, (i - 1) % N
        for j in range(M + 1):
            w    = W[i, j]
            wc1x = c1_x[i, j]
            wc1z = c1_z[i, j]
            wc2x = C2_xx[i, j]
            wc2z = C2_zz[i, j]

            # ── x-derivative (periodic central) ──
            C1[idx(ip, j)] += w * wc1x * FX
            C1[idx(im, j)] -= w * wc1x * FX
            vx = w * wc2x * FX**2
            ipj, imj = idx(ip, j), idx(im, j)
            C2[ipj, ipj] += vx
            C2[imj, imj] += vx
            C2[ipj, imj] -= vx
            C2[imj, ipj] -= vx

            # ── z-derivative (one-sided at boundaries) ──
            if j == 0:
                fz = 1.0 / dz_n
                C1[idx(i, 1)] += w * wc1z * fz
                C1[idx(i, 0)] -= w * wc1z * fz
                vz = w * wc2z * fz**2
                C2[idx(i, 1), idx(i, 1)] += vz
                C2[idx(i, 0), idx(i, 0)] += vz
                C2[idx(i, 1), idx(i, 0)] -= vz
                C2[idx(i, 0), idx(i, 1)] -= vz
            elif j == M:
                fz = 1.0 / dz_n
                C1[idx(i, M)]     += w * wc1z * fz
                C1[idx(i, M - 1)] -= w * wc1z * fz
                vz = w * wc2z * fz**2
                C2[idx(i, M), idx(i, M)]         += vz
                C2[idx(i, M - 1), idx(i, M - 1)] += vz
                C2[idx(i, M), idx(i, M - 1)]     -= vz
                C2[idx(i, M - 1), idx(i, M)]     -= vz
            else:
                fz = 1.0 / (2.0 * dz_n)
                C1[idx(i, j + 1)] += w * wc1z * fz
                C1[idx(i, j - 1)] -= w * wc1z * fz
                vz = w * wc2z * fz**2
                C2[idx(i, j + 1), idx(i, j + 1)] += vz
                C2[idx(i, j - 1), idx(i, j - 1)] += vz
                C2[idx(i, j + 1), idx(i, j - 1)] -= vz
                C2[idx(i, j - 1), idx(i, j + 1)] -= vz

    # Pin s[0, 0] = 0
    C2p = C2.tocsr().tolil()
    C2p[0, :] = 0.0
    C2p[:, 0] = 0.0
    C2p[0, 0] = 1.0
    C2p = C2p.tocsr()
    C1p = C1.copy()
    C1p[0] = 0.0

    u     = spsolve(2.0 * C2p, -C1p)
    E     = C0 + C1p @ u + u @ (C2p @ u)
    kappa = E / 8.0

    As   = surface_area(k)
    V_s  = 2.0 * eps * As
    rho  = V_s / 8.0

    return {'kappa_eps': kappa, 'rho_eps': rho, 'energy': E,
            'C0': C0, 'V_shell': V_s}

# ═══════════════════════════════════════════════════════════════════
# Main: run convergence for all 3 surfaces and plot
# ═══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    epsilons = [0.10, 0.05, 0.025, 0.01, 0.005, 0.002, 0.001]
    all_res  = {}
    N, M     = 5000, 20

    print("=" * 80)
    print("Mathematica-derived coefficients — Asymptotic convergence test")
    print(f"  Grid: N={N}, M={M} (normalized z ∈ [-1,1])")
    print("  κ_ε = E_min / |Y|,  |Y| = 8")
    print("=" * 80)

    for name, surf in SURFACES.items():
        k   = surf.k
        ref = ADC_REF[name]
        R_f, Rp_f, Rpp_f, _, _ = principal_curvatures(k)

        print(f"\n{'─' * 80}")
        print(f"Surface: {name} (k = {k/np.pi:.0f}π), κ_A ref = {ref:.15f}")
        print(f"{'─' * 80}")
        print(f"  {'ε':>10} {'κ_ε':>16} {'ρ_ε':>12} {'ratio=κ_ε/ρ_ε':>16} {'E_r':>14}")
        print(f"  " + "-" * 80)

        pts = []
        for eps in epsilons:
            r = compute_kappa(k, eps, R_f, Rp_f, Rpp_f, N=N, M=M)
            ratio = r['kappa_eps'] / r['rho_eps']
            Er    = abs(r['kappa_eps'] - r['rho_eps'] * ref)
            pts.append({
                'eps': eps, 'kappa_eps': r['kappa_eps'],
                'rho_eps': r['rho_eps'], 'ratio': ratio, 'E_r': Er,
            })
            print(f"  {eps:10.4f} {r['kappa_eps']:16.12f} {r['rho_eps']:12.8f} "
                  f"{ratio:16.12f} {Er:14.6e}")

        all_res[name] = pts

        # Convergence order (exclude largest ε)
        eps_log = np.array([np.log(p['eps']) for p in pts[1:]])
        Er_log  = np.array([np.log(p['E_r'])  for p in pts[1:]])
        slope   = np.polyfit(eps_log, Er_log, 1)[0]
        print(f"\n  Convergence: O(ε^{slope:.2f})")

    # ── Save CSV ──
    csv_path = '/opt/data/paper-tpms-heat/xtpms/tests/convergence_results/mathematica_data.csv'
    with open(csv_path, 'w') as f:
        f.write('surface,eps,kappa_eps,rho_eps,ratio,E_r\n')
        for name, pts in all_res.items():
            for p in pts:
                f.write(f"{name},{p['eps']:.6f},"
                        f"{p['kappa_eps']:.15e},{p['rho_eps']:.15e},"
                        f"{p['ratio']:.15e},{p['E_r']:.15e}\n")
    print(f"\nCSV  saved → {csv_path}")

    # ── Plot ──
    fig, ax = plt.subplots(figsize=(8, 6))
    colors  = {'k1pi': '#0072B2', 'k2pi': '#D55E00', 'k3pi': '#009E73'}
    labels  = {'k1pi': r'$k=\pi$', 'k2pi': r'$k=2\pi$', 'k3pi': r'$k=3\pi$'}

    for name, pts in all_res.items():
        eps_a = np.array([p['eps'] for p in pts])
        Er_a  = np.array([p['E_r']  for p in pts])
        si    = np.argsort(eps_a)[::-1]  # descending: large ε left
        ax.loglog(eps_a[si], Er_a[si], 'o-', color=colors[name],
                  label=labels[name], markersize=6, linewidth=1.5)

    # O(ε³) reference line
    if all_res.get('k1pi'):
        r     = all_res['k1pi'][-1]
        scale = r['E_r'] / (r['eps']**3)
        ref_e = np.array([0.1, 0.001])
        ax.loglog(ref_e, scale * ref_e**3, 'k--', alpha=0.4, linewidth=1)
        ax.text(0.03, scale * 0.03**3 * 1.5,
                r'$O(\epsilon^3)$', fontsize=12, alpha=0.6)

    ax.set_xlabel(r'Shell half-thickness $\epsilon$', fontsize=13)
    ax.set_ylabel(r'$|\kappa_\epsilon - \rho_\epsilon \kappa_A|$', fontsize=13)
    ax.set_title('Asymptotic accuracy (corrected Mathematica coefficients)',
                 fontsize=14)
    ax.legend(fontsize=12, loc='lower right')
    ax.grid(True, which='both', alpha=0.3)
    ax.invert_xaxis()  # large ε LEFT → small ε RIGHT
    plt.tight_layout()

    plot_path = '/opt/data/paper-tpms-heat/xtpms/tests/convergence_results/mathematica_plot.png'
    plt.savefig(plot_path, dpi=200)
    print(f"Plot saved → {plot_path}")

    print(f"\n{'=' * 80}")
    print("Done.")
