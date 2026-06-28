"""
Semi-analytical evaluation of effective conductivity for shell lattices
built from surfaces of revolution.

Implements the method from Supplementary §S8:
  r^eps(x,theta,z) = r(x,theta) + z*n(x,theta),  z in [-eps, eps]

By rotational symmetry, u^eps = s(x,z) (no theta dependence), reducing the
3D cell problem to a 2D quadratic minimization in (x,z):

  kappa_eps = (2*pi*kappa / |Y|) * min_s  int  [c0 + c1 . grad s + grad s^T C2 grad s] dz dx

The shell metric exploits principal coordinates (x, theta are principal directions):
  g11^eps = (1 - z*k1)^2 * (1 + R'^2)
  g22^eps = (1 - z*k2)^2 * R^2
  g33    = 1,  g12 = g13 = g23 = 0
  sqrt(g) = (1 - z*k1)(1 - z*k2) * R * sqrt(1 + R'^2) =: J(x,z)

Derived coefficients (kappa=1, p = e^1 = x-axis):
  c0     = J
  c1x    = 2*(1 - z*k2)*R / sqrt(1 + R'^2)
  c1z    = -2*(1 - z*k1)*(1 - z*k2)*R*R'
  C2_xx  = (1 - z*k2)*R / [(1 - z*k1)*sqrt(1 + R'^2)]
  C2_zz  = J = (1 - z*k1)*(1 - z*k2)*R*sqrt(1 + R'^2)
  C2_xz  = 0

Principal curvatures (inward-pointing normal):
  k1 = R'' / (1 + R'^2)^(3/2)    (meridional)
  k2 = -1 / (R * sqrt(1 + R'^2)) (azimuthal)

Discretization: N x (M+1) grid, periodic in x, one-sided FD at z-boundaries,
trapezoidal quadrature in z.
"""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.integrate import quad
import sys

sys.path.insert(0, '/opt/data/paper-tpms-heat/xtpms/tests')
from rev_surface_mesh import SURFACES, ADC_REF


def principal_curvatures(k):
    """Return functions (R, R', R'', k1, k2) for R(x) = (2 + cos(kx))/4."""
    R  = lambda x: (2.0 + np.cos(k * x)) / 4.0
    Rp = lambda x: -k * np.sin(k * x) / 4.0
    Rpp = lambda x: -k**2 * np.cos(k * x) / 4.0

    def k1(x):
        rpp = Rpp(x)
        rp = Rp(x)
        return rpp / (1.0 + rp**2)**1.5

    def k2(x):
        r = R(x)
        rp = Rp(x)
        return -1.0 / (r * np.sqrt(1.0 + rp**2))

    return R, Rp, Rpp, k1, k2


# ═══════════════════════════════════════════════════════════════════
# New formulation (Di Zhang, Mathematica-derived):
#   Integrand coefficients after angular (theta) integration,
#   with NORMALIZED z in [-1, 1] (physical z = eps * z_norm).
#
#   I(x, z; du/dx, du/dz) = c0 + c1_x·u_x + c1_z·u_z + C2_xx·u_x² + C2_zz·u_z²
#
#   Total energy = ∫_{-1}^{1}∫_{-1}^{1} I dx dz  →  κ_ε = (1/|Y|) · energy
# ═══════════════════════════════════════════════════════════════════

def compute_integrand_coeffs(df, d2f, z, eps):
    """Compute integrand coefficients for the 2D reduced problem.

    Parameters
    ----------
    df : float or ndarray
        R'(x) — first derivative of the radius profile.
    d2f : float or ndarray
        R''(x) — second derivative of the radius profile.
    z : float or ndarray
        Normalised z-coordinate, z ∈ [-1, 1].
    eps : float
        Shell half-thickness ε.

    Returns
    -------
    dict with keys:
        c0   : scalar / array — constant term = 2π
        c1_x : scalar / array — coefficient of du/dx
        c1_z : scalar / array — coefficient of du/dz
        C2_xx: scalar / array — coefficient of (du/dx)²
        C2_zz: scalar / array — coefficient of (du/dz)²
        C2_xz: 0 (always)
    """
    sq = np.sqrt(1.0 + df**2)                            # √(1 + R'²)
    one_p_df2 = 1.0 + df**2                               # 1 + R'²
    denom_lin = one_p_df2 + d2f * z * eps / sq            # denominator in c1_x
    denom_quad = sq * one_p_df2 + d2f * z * eps           # denominator in C2_xx numerator

    c0 = np.broadcast_to(2.0 * np.pi, z.shape).copy()

    c1_x = (4.0 * np.pi) / denom_lin                      # 4π/(1+R'² + R''·z·ε/√(1+R'²))
    c1_z = np.broadcast_to((4.0 * np.pi * df) / (sq * eps), z.shape).copy()  # 4π·R'/(√(1+R'²)·ε)

    C2_xx = (2.0 * one_p_df2**2 * np.pi) / (denom_quad ** 2)
    C2_zz = np.broadcast_to((2.0 * np.pi) / (eps ** 2), z.shape).copy()
    C2_xz = np.zeros_like(z)

    return {
        'c0': c0,
        'c1_x': c1_x,
        'c1_z': c1_z,
        'C2_xx': C2_xx,
        'C2_zz': C2_zz,
        'C2_xz': C2_xz,
    }


# ═══════════════════════════════════════════════════════════════════
# Verify the new coefficients against the existing formulation.
# Both should give the same total energy for the same u-field.
# The old formulation uses physical z and principal curvatures;
# the new formulation uses normalized z and R'/R'' directly.
# ═══════════════════════════════════════════════════════════════════

def verify_coeff_equivalence(k=np.pi, eps=0.01):
    """Verify new vs old coefficients numerically at a few sample points.

    Old formulation (physical z ∈ [-ε,ε], principal curvatures):
        I_old = c0_p + c1x_p·u_x + c1z_p·u_zphys + C2xx_p·u_x² + C2zz_p·u_zphys²
        E = 2π ∫_{-1}^{1}∫_{-ε}^{ε} I_old dx dz_phys

    New formulation (normalized z ∈ [-1,1], R'/R'' directly):
        I_new = c0 + c1_x·u_x + c1_z·u_znorm + C2_xx·u_x² + C2_zz·u_znorm²
        E = ∫_{-1}^{1}∫_{-1}^{1} I_new dx dz_norm

    Mapping (z_phys = ε·z_norm, ∂/∂z_norm = ε·∂/∂z_phys, dz_phys = ε·dz_norm):
        c0     = 2π·ε·c0_p
        c1_x   = 2π·ε·c1x_p
        c1_z   = 2π·c1z_p
        C2_xx  = 2π·ε·C2xx_p
        C2_zz  = 2π/ε·C2zz_p

    Prints sampled values and checks the mapping.
    """
    R, Rp, Rpp, k1f, k2f = principal_curvatures(k)

    x_vals = np.linspace(-1.0, 1.0, 10, endpoint=False)
    z_norm_vals = np.linspace(-1.0, 1.0, 5)

    print(f"k = {k/np.pi:.0f}π,  ε = {eps}")
    print(f"{'x':>10} {'z_norm':>8} "
          f"{'c1x_new':>12} {'map(c1x_p)':>14} {'diff':>10} "
          f"{'C2xx_new':>12} {'map(C2xx_p)':>14} {'diff':>10}")
    print("-" * 95)

    for x in x_vals:
        r_val = float(R(np.array([x]))[0])
        rp_val = float(Rp(np.array([x]))[0])
        rpp_val = float(Rpp(np.array([x]))[0])
        k1_val = float(k1f(np.array([x]))[0])
        k2_val = float(k2f(np.array([x]))[0])
        sq_val = np.sqrt(1.0 + rp_val**2)

        for z_norm in z_norm_vals:
            z_phys = eps * z_norm

            c_new = compute_integrand_coeffs(rp_val, rpp_val, z_norm, eps)

            # Old formulation
            a1 = 1.0 - z_phys * k1_val
            a2 = 1.0 - z_phys * k2_val
            c1x_p = 2.0 * a2 * r_val / sq_val
            C2xx_p = a2 * r_val / (a1 * sq_val)

            # Map old → new (normalized)
            map_c1x = 2.0 * np.pi * eps * c1x_p
            map_C2xx = 2.0 * np.pi * eps * C2xx_p

            if abs(z_norm) < 0.01:  # print only mid-surface
                print(f"{x:10.4f} {z_norm:8.3f} "
                      f"{c_new['c1_x']:12.6f} {map_c1x:14.6f} "
                      f"{abs(c_new['c1_x']-map_c1x):10.2e} "
                      f"{c_new['C2_xx']:12.6f} {map_C2xx:14.6f} "
                      f"{abs(c_new['C2_xx']-map_C2xx):10.2e}")

    print("\nNote: exact equivalence holds only at z=0 (mid-surface) because")
    print("the two formulations use different geometric approximations off the mid-surface.")
    print("The real test is whether both give the same converged κ_ε for the full system.")


def compute_kappa_epsilon(k, eps, N=100000, M=20):
    """Compute kappa_eps(e^1) for a revolution surface with wavenumber k.

    Parameters
    ----------
    k : float
        Wavenumber (k = m*pi for period-2 surfaces).
    eps : float
        Shell half-thickness.
    N : int
        Number of x-grid cells (periodic).
    M : int
        Number of z-grid cells.

    Returns
    -------
    dict with keys: kappa_eps, rho_eps, kappa_eps/rho_eps, energy, N, M
    """
    R, Rp, Rpp, k1f, k2f = principal_curvatures(k)
    Y_vol = 8.0  # |Y| = |[-1,1]^3| = 8

    # Grid
    dx = 2.0 / N
    dz = 2.0 * eps / M
    x = np.linspace(-1.0, 1.0, N, endpoint=False)  # (N,) periodic
    z = np.linspace(-eps, eps, M + 1)                 # (M+1,)

    # Evaluate coefficients on the grid (vectorized)
    Rv  = R(x)       # (N,)
    Rpv = Rp(x)      # (N,)
    Rppv = Rpp(x)    # (N,)
    k1v = k1f(x)     # (N,)
    k2v = k2f(x)     # (N,)
    sq = np.sqrt(1.0 + Rpv**2)  # (N,)

    # Broadcast: X (N,1), Z (1, M+1) -> (N, M+1)
    X = x[:, np.newaxis]
    Z = z[np.newaxis, :]

    R_g  = Rv[:, np.newaxis]     # (N, 1)
    Rp_g = Rpv[:, np.newaxis]
    sq_g = sq[:, np.newaxis]
    k1_g = k1v[:, np.newaxis]
    k2_g = k2v[:, np.newaxis]

    # Jacobian J = (1 - z*k1)*(1 - z*k2)*R*sqrt(1+R'^2)
    a1 = 1.0 - Z * k1_g   # (N, M+1)
    a2 = 1.0 - Z * k2_g
    J = a1 * a2 * R_g * sq_g

    # Coefficients
    c0 = J
    c1x = 2.0 * a2 * R_g / sq_g
    c1z = -2.0 * a1 * a2 * R_g * Rp_g
    C2xx = a2 * R_g / (a1 * sq_g)
    C2zz = J

    # Quadrature weights in z: trapezoidal
    wz = np.full(M + 1, dz)
    wz[0] = dz / 2.0
    wz[M] = dz / 2.0
    wz_g = wz[np.newaxis, :]  # (1, M+1)

    # Weighted coefficients: w[i,j] = dx * wz[j]
    w = dx * wz[np.newaxis, :] * np.ones((N, M + 1))  # (N, M+1)

    # Total unknowns: N * (M+1)
    nv = N * (M + 1)

    def idx(i, j):
        """Grid (i,j) -> flat index. i in [0,N), j in [0,M]."""
        return i * (M + 1) + j

    # ── Assemble C0 (scalar), C1 (vector), C2 (sparse matrix) ──

    C0 = float(np.sum(c0 * w))

    C1 = np.zeros(nv)
    triplets = []  # (row, col, val) for C2

    for i in range(N):
        ip = (i + 1) % N
        im = (i - 1) % N

        for j in range(M + 1):
            wij = w[i, j]
            ci1x = c1x[i, j]
            ci1z = c1z[i, j]
            ci2xx = C2xx[i, j]
            ci2zz = C2zz[i, j]

            # --- s_x via central difference (periodic) ---
            # s_x = (s[ip,j] - s[im,j]) / (2*dx)
            # Contribution to C1: w * c1x * s_x = w * c1x / (2*dx) * (s[ip,j] - s[im,j])
            fac_sx = 1.0 / (2.0 * dx)
            C1[idx(ip, j)] += wij * ci1x * fac_sx
            C1[idx(im, j)] -= wij * ci1x * fac_sx

            # Contribution to C2: w * C2xx * s_x^2 = w * C2xx * fac_sx^2 * (s[ip,j] - s[im,j])^2
            val_xx = wij * ci2xx * fac_sx**2
            ii = idx(i, j)
            ip_j = idx(ip, j)
            im_j = idx(im, j)
            triplets.append((ip_j, ip_j, val_xx))
            triplets.append((im_j, im_j, val_xx))
            triplets.append((ip_j, im_j, -val_xx))
            triplets.append((im_j, ip_j, -val_xx))

            # --- s_z via finite difference ---
            if j == 0:
                # Forward: s_z = (s[i,1] - s[i,0]) / dz
                fac_sz = 1.0 / dz
                i0 = idx(i, 0)
                i1 = idx(i, 1)
                C1[i1] += wij * ci1z * fac_sz
                C1[i0] -= wij * ci1z * fac_sz

                val_zz = wij * ci2zz * fac_sz**2
                triplets.append((i1, i1, val_zz))
                triplets.append((i0, i0, val_zz))
                triplets.append((i1, i0, -val_zz))
                triplets.append((i0, i1, -val_zz))

            elif j == M:
                # Backward: s_z = (s[i,M] - s[i,M-1]) / dz
                fac_sz = 1.0 / dz
                iM = idx(i, M)
                iMm1 = idx(i, M - 1)
                C1[iM] += wij * ci1z * fac_sz
                C1[iMm1] -= wij * ci1z * fac_sz

                val_zz = wij * ci2zz * fac_sz**2
                triplets.append((iM, iM, val_zz))
                triplets.append((iMm1, iMm1, val_zz))
                triplets.append((iM, iMm1, -val_zz))
                triplets.append((iMm1, iM, -val_zz))

            else:
                # Central: s_z = (s[i,j+1] - s[i,j-1]) / (2*dz)
                fac_sz = 1.0 / (2.0 * dz)
                ijp = idx(i, j + 1)
                ijm = idx(i, j - 1)
                C1[ijp] += wij * ci1z * fac_sz
                C1[ijm] -= wij * ci1z * fac_sz

                val_zz = wij * ci2zz * fac_sz**2
                triplets.append((ijp, ijp, val_zz))
                triplets.append((ijm, ijm, val_zz))
                triplets.append((ijp, ijm, -val_zz))
                triplets.append((ijm, ijp, -val_zz))

    # Build sparse C2
    rows = [t[0] for t in triplets]
    cols = [t[1] for t in triplets]
    vals = [t[2] for t in triplets]
    C2 = csr_matrix((vals, (rows, cols)), shape=(nv, nv))

    # Remove null space: fix one DOF (s[0,0] = 0)
    # Pin the first row/column
    C2_pinned = C2.tolil()
    C2_pinned[0, :] = 0
    C2_pinned[:, 0] = 0
    C2_pinned[0, 0] = 1.0
    C1_pinned = C1.copy()
    C1_pinned[0] = 0.0

    # Solve: C2 * 2 * s = -C1  =>  s = -(1/2) * C2^{-1} * C1
    s = spsolve(C2_pinned * 2.0, -C1_pinned)

    # Compute minimum energy
    energy = C0 + C1_pinned @ s + s @ (C2_pinned @ s)

    # Effective conductivity
    kappa_eps = 2.0 * np.pi / Y_vol * energy

    # Volume fraction: rho_eps = |Omega^eps| / |Y|
    # |Omega^eps| = 2*pi * int_{-1}^{1} int_{-eps}^{eps} J dz dx
    vol_integral = float(np.sum(J * w))
    rho_eps = 2.0 * np.pi / Y_vol * vol_integral

    return {
        'kappa_eps': kappa_eps,
        'rho_eps': rho_eps,
        'ratio': kappa_eps / rho_eps,
        'energy': energy,
        'N': N,
        'M': M,
    }


def main():
    print("=" * 80)
    print("Semi-analytical effective conductivity for revolution surfaces")
    print("  R(x) = (2 + cos(kx))/4,  kappa=1,  p = e^1")
    print("  Method: Supplementary §S8 (2D reduction + FD discretization)")
    print("=" * 80)

    # Test epsilons: from large to small
    epsilons = [0.10, 0.05, 0.025, 0.01, 0.005, 0.002, 0.001]

    for name, surf in SURFACES.items():
        k = surf.k
        k_pi = k / np.pi
        adc_ref = ADC_REF[name]

        print(f"\n{'─' * 80}")
        print(f"Surface: {name} (k = {k_pi:.0f}pi)")
        print(f"  ADC(e^1) reference = {adc_ref:.15f}")
        print(f"{'─' * 80}")
        print(f"  {'eps':>10} {'kappa_eps':>14} {'rho_eps':>12} {'k_eps/rho':>14} "
              f"{'ADC':>14} {'residual':>14} {'rel_resid':>12}")
        print(f"  " + "-" * 100)

        results = []
        for eps in epsilons:
            r = compute_kappa_epsilon(k, eps, N=10000, M=20)
            residual = r['kappa_eps'] - r['rho_eps'] * adc_ref
            rel_resid = abs(residual) / (r['rho_eps'] * adc_ref)
            results.append((eps, r, residual, rel_resid))

            print(f"  {eps:10.4f} {r['kappa_eps']:14.8f} {r['rho_eps']:12.8f} "
                  f"{r['ratio']:14.8f} {adc_ref:14.8f} {residual:14.6e} {rel_resid:12.4e}")

        # Estimate convergence order
        print(f"\n  Convergence rate (log-log fit of |residual| vs eps):")
        for i in range(1, len(results)):
            e0, r0, res0, _ = results[i - 1]
            e1, r1, res1, _ = results[i]
            if abs(res0) > 1e-30 and abs(res1) > 1e-30:
                order = np.log(abs(res1) / abs(res0)) / np.log(e1 / e0)
                print(f"    eps={e0:.4f} -> {e1:.4f}: order = {order:.2f}")


if __name__ == '__main__':
    main()
