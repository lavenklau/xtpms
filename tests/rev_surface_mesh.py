#!/usr/bin/env python3
"""
Mesh generation for periodic surfaces of revolution — ADC convergence test module.

Generates tensor-product triangle meshes on periodic surfaces of revolution
    R(x) = (2 + cos(k·x)) / 4,   x ∈ [-1, 1],   rotating around the x-axis,
embedded in the flat torus [-1,1]^3 = R^3 / (2Z)^3.

Periodicity requirement: R(x+2) = R(x)  ⟺  k = mπ  (m ∈ Z).
Three test surfaces: k = π, 2π, 3π.

Mesh design:
  - Uniform sampling in x (N_x segments) and θ (N_θ segments), endpoint=False.
  - Each quad is split into 2 triangles along the diagonal (i,j)→(i+1,j+1).
  - N_x and N_θ are chosen so that arc-length-per-sement in x ≈ arc-length-per-segment in θ,
    making triangles approximately isosceles right triangles.
  - The mesh size h = diagonal (hypotenuse) length = circumcircle diameter for right triangles.
  - The mesh is topologically a torus (product of two circles): closed, no boundary, naturally periodic.

Usage:
    python rev_surface_mesh.py                    # run mesh-size control test
    python rev_surface_mesh.py --save OBJ_DIR     # also save OBJ files
"""

import argparse
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
from scipy.integrate import quad

# ═══════════════════════════════════════════════════════════════════
# Surface definition
# ═══════════════════════════════════════════════════════════════════

@dataclass
class RevSurface:
    """Periodic surface of revolution around the x-axis in [-1,1]^3.

    P(x, θ) = (x, R(x)·cos θ, R(x)·sin θ),   R(x) = (2 + cos(k·x)) / 4.

    Periodic in x with period 2 when k = mπ (m ∈ Z).
    """
    k: float  # wavenumber

    def R(self, x: np.ndarray) -> np.ndarray:
        """Radius profile R(x) = (2 + cos(kx)) / 4."""
        return (2.0 + np.cos(self.k * x)) / 4.0

    def R_prime(self, x: np.ndarray) -> np.ndarray:
        """Derivative R'(x) = -k·sin(kx) / 4."""
        return -self.k * np.sin(self.k * x) / 4.0

    def point(self, x: float, theta: float) -> np.ndarray:
        """Surface point P(x, θ) in R^3."""
        r = float(self.R(np.array([x]))[0])
        return np.array([x, r * np.cos(theta), r * np.sin(theta)])

    def arc_length_x(self) -> float:
        """Total arc length along x: ∫_{-1}^{1} √(1 + R'(x)²) dx."""
        result, _ = quad(
            lambda x: np.sqrt(1.0 + self.R_prime(np.array([x]))[0] ** 2),
            -1.0, 1.0, limit=200
        )
        return result

    def avg_radius(self) -> float:
        """Average radius = (1/2) ∫_{-1}^{1} R(x) dx.

        For k = mπ: ∫_{-1}^{1} cos(kx) dx = 2 sin(k)/k = 0, so avg_radius = 1/2 exactly.
        """
        result, _ = quad(lambda x: float(self.R(np.array([x]))[0]), -1.0, 1.0)
        return result / 2.0

    def circumference_avg(self) -> float:
        """Average circumference = 2π · avg_radius."""
        return 2.0 * np.pi * self.avg_radius()


# ═══════════════════════════════════════════════════════════════════
# Mesh size control
# ═══════════════════════════════════════════════════════════════════

def compute_resolution(surf: RevSurface, target_h: float) -> Tuple[int, int]:
    """Compute (N_x, N_θ) for approximately isosceles right triangles with h ≈ target_h.

    For isosceles right triangles:
        leg = h / √2
        N_x = L_x / leg     (L_x = total x-arc-length)
        N_θ = C_avg / leg   (C_avg = average circumference)
        h   = √2 · leg      (hypotenuse = circumcircle diameter)
    """
    L_x = surf.arc_length_x()
    C_avg = surf.circumference_avg()
    leg = target_h / np.sqrt(2.0)

    N_x = max(8, int(round(L_x / leg)))
    N_theta = max(8, int(round(C_avg / leg)))
    return N_x, N_theta


# ═══════════════════════════════════════════════════════════════════
# Mesh generation (vectorized)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Mesh:
    """Triangle mesh on a surface of revolution."""
    vertices: np.ndarray   # (N, 3) float64
    faces: np.ndarray      # (F, 3) int32
    N_x: int
    N_theta: int
    k: float

    @property
    def n_vertices(self) -> int:
        return len(self.vertices)

    @property
    def n_faces(self) -> int:
        return len(self.faces)

    def save_obj(self, filename: str):
        """Save as OBJ file (1-indexed faces)."""
        with open(filename, 'w') as f:
            f.write(f"# Surface of revolution: k={self.k/np.pi:.4g}pi\n")
            f.write(f"# N_x={self.N_x} N_theta={self.N_theta}\n")
            f.write(f"# {self.n_vertices} vertices, {self.n_faces} faces\n")
            for v in self.vertices:
                f.write(f"v {v[0]:.12e} {v[1]:.12e} {v[2]:.12e}\n")
            for face in self.faces:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


def generate_mesh(surf: RevSurface, N_x: int, N_theta: int) -> Mesh:
    """Generate tensor-product triangle mesh on surface of revolution.

    Vertices: (i, j) → x_i = -1 + 2i/N_x,  θ_j = 2πj/N_θ
              Periodic: i mod N_x, j mod N_θ (no boundary vertices).

    Faces: each quad (i,j) split along diagonal (i,j)→(i+1,j+1):
        Tri1: [v00, v10, v11]   edges: x-leg, θ-leg, diagonal
        Tri2: [v00, v11, v01]   edges: diagonal, x-leg, θ-leg
    """
    x = np.linspace(-1.0, 1.0, N_x, endpoint=False)          # (N_x,)
    theta = np.linspace(0.0, 2.0 * np.pi, N_theta, endpoint=False)  # (N_theta,)

    R_vals = surf.R(x)  # (N_x,)

    # Tensor-product grid (vectorized)
    # X[i,j] = x[i], TH[i,j] = theta[j]
    X, TH = np.meshgrid(x, theta, indexing='ij')      # (N_x, N_theta)
    R_grid = R_vals[:, np.newaxis]                      # (N_x, 1) broadcast

    # Vertices: flatten in row-major order (i*N_theta + j)
    vertices = np.stack([
        X.ravel(),
        (R_grid * np.cos(TH)).ravel(),
        (R_grid * np.sin(TH)).ravel(),
    ], axis=1)  # (N_x * N_theta, 3)

    # Face indices (vectorized)
    I, J = np.meshgrid(np.arange(N_x), np.arange(N_theta), indexing='ij')
    i1 = (I + 1) % N_x
    j1 = (J + 1) % N_theta

    v00 = (I * N_theta + J).ravel()
    v10 = (i1 * N_theta + J).ravel()
    v11 = (i1 * N_theta + j1).ravel()
    v01 = (I * N_theta + j1).ravel()

    n_quads = N_x * N_theta
    faces = np.empty((2 * n_quads, 3), dtype=np.int32)
    faces[0::2] = np.stack([v00, v10, v11], axis=1)  # Tri1: x-leg, θ-leg, diagonal
    faces[1::2] = np.stack([v00, v11, v01], axis=1)  # Tri2: diagonal, x-leg, θ-leg

    return Mesh(vertices=vertices, faces=faces, N_x=N_x, N_theta=N_theta, k=surf.k)


# ═══════════════════════════════════════════════════════════════════
# Mesh statistics
# ═══════════════════════════════════════════════════════════════════

@dataclass
class MeshStats:
    """Statistics for a generated mesh."""
    # Basic counts
    n_vertices: int
    n_faces: int
    N_x: int
    N_theta: int
    target_h: float

    # Diagonal (hypotenuse) statistics — this is the mesh size h
    h_min: float
    h_max: float
    h_mean: float
    h_median: float
    h_std: float

    # Leg statistics
    leg_x_mean: float
    leg_x_std: float
    leg_theta_mean: float
    leg_theta_std: float

    # Aspect ratio = x_leg / θ_leg (≈ 1 for isosceles)
    aspect_mean: float
    aspect_std: float

    # All-edge statistics (for reference)
    edge_min: float
    edge_max: float
    edge_mean: float

    @property
    def h_over_target(self) -> float:
        """Ratio of mean h to target h (≈ 1.0 means good control)."""
        return self.h_mean / self.target_h if self.target_h > 0 else 0.0


def _periodic_lengths(v0: np.ndarray, v1: np.ndarray, half_period: np.ndarray) -> np.ndarray:
    """Compute periodic edge lengths, wrapping each component into [-half, half).

    The surface lives in the flat torus [-1,1]^3 = R^3 / (2Z)^3, so the
    period is 2 in each axis.  For edge vector dv = v1 - v0, the wrapped
    vector is  dv - 2*round(dv / 2)  componentwise, and the periodic length
    is its Euclidean norm.
    """
    dv = v1 - v0
    dv_wrapped = dv - 2.0 * np.round(dv / 2.0)
    # After wrapping, components are in [-1, 1]. But if half_period is
    # asymmetric (not the case here), we'd use dv - 2*hp*round(dv/(2*hp)).
    return np.linalg.norm(dv_wrapped, axis=1)


def compute_stats(mesh: Mesh, target_h: float,
                  half_period: np.ndarray = None) -> MeshStats:
    """Compute mesh statistics, separating diagonal (hypotenuse) from leg edges.

    For each quad split along (i,j)→(i+1,j+1):
        Tri1 [v00, v10, v11]: e0=x-leg, e1=θ-leg, e2=diagonal
        Tri2 [v00, v11, v01]: e0=diagonal, e1=x-leg, e2=θ-leg

    Edge lengths are computed using periodic wrapping (torus metric) so that
    wrap-around edges at the periodic boundary are correctly measured.
    """
    if half_period is None:
        half_period = np.array([1.0, 1.0, 1.0])

    v = mesh.vertices
    f = mesh.faces

    # Periodic edge vectors and lengths (vectorized)
    e0_vec = v[f[:, 1]] - v[f[:, 0]]  # (n_faces, 3)
    e1_vec = v[f[:, 2]] - v[f[:, 1]]
    e2_vec = v[f[:, 0]] - v[f[:, 2]]

    # Wrap each component: dv - 2*hp*round(dv / (2*hp))
    period = 2.0 * half_period
    e0_wrapped = e0_vec - period * np.round(e0_vec / period)
    e1_wrapped = e1_vec - period * np.round(e1_vec / period)
    e2_wrapped = e2_vec - period * np.round(e2_vec / period)

    e0_len = np.linalg.norm(e0_wrapped, axis=1)
    e1_len = np.linalg.norm(e1_wrapped, axis=1)
    e2_len = np.linalg.norm(e2_wrapped, axis=1)

    all_edges = np.concatenate([e0_len, e1_len, e2_len])

    # Separate by triangle type
    n = len(f)
    is_tri1 = np.arange(n) % 2 == 0

    # Tri1: e0=x-leg, e1=θ-leg, e2=diagonal
    # Tri2: e0=diagonal, e1=x-leg, e2=θ-leg
    diagonals = np.where(is_tri1, e2_len, e0_len)
    x_legs = np.where(is_tri1, e0_len, e1_len)
    theta_legs = np.where(is_tri1, e1_len, e2_len)

    aspects = x_legs / np.maximum(theta_legs, 1e-30)

    return MeshStats(
        n_vertices=mesh.n_vertices,
        n_faces=mesh.n_faces,
        N_x=mesh.N_x,
        N_theta=mesh.N_theta,
        target_h=target_h,
        h_min=float(diagonals.min()),
        h_max=float(diagonals.max()),
        h_mean=float(diagonals.mean()),
        h_median=float(np.median(diagonals)),
        h_std=float(diagonals.std()),
        leg_x_mean=float(x_legs.mean()),
        leg_x_std=float(x_legs.std()),
        leg_theta_mean=float(theta_legs.mean()),
        leg_theta_std=float(theta_legs.std()),
        aspect_mean=float(aspects.mean()),
        aspect_std=float(aspects.std()),
        edge_min=float(all_edges.min()),
        edge_max=float(all_edges.max()),
        edge_mean=float(all_edges.mean()),
    )


# ═══════════════════════════════════════════════════════════════════
# Three test surfaces
# ═══════════════════════════════════════════════════════════════════

SURFACES: Dict[str, RevSurface] = {
    'k1pi': RevSurface(k=np.pi),
    'k2pi': RevSurface(k=2.0 * np.pi),
    'k3pi': RevSurface(k=3.0 * np.pi),
}

# ───────────────────────────────────────────────────────────────────
# High-precision reference ADC values (kappa = 1)
# ───────────────────────────────────────────────────────────────────
# Computed in Mathematica with 30 significant digits using the
# closed-form formula from Supplementary §S7 (Theorem S1):
#
#   kappa_A = 4 / (I1 * I2)
#   I1 = Integrate[Sqrt[1 + R'[x]^2] / R[x], {x, -1, 1}]
#   I2 = Integrate[R[x] * Sqrt[1 + R'[x]^2], {x, -1, 1}]
#   R(x) = (2 + cos(k*x)) / 4,  R'(x) = -k*sin(k*x)/4
#
# These serve as ground truth for the discrete ADC convergence test
# (error vs mesh size h) and the asymptotic accuracy test (residual
# vs shell thickness epsilon).

ADC_REF: Dict[str, float] = {
    'k1pi': 0.672319082696148908351292923939,
    'k2pi': 0.412907402350783021626364032370,
    'k3pi': 0.256191632219216291285901900428,
}

def adc_reference(name: str) -> float:
    """Return the high-precision reference ADC(e^1) for a test surface.

    Parameters
    ----------
    name : 'k1pi' | 'k2pi' | 'k3pi'

    Returns
    -------
    float
        kappa_A(tilde_omega_R; e^1) with kappa=1, accurate to ~30 digits
        (truncated to float64 precision ~15-16 significant digits).
    """
    return ADC_REF[name]

# Target mesh sizes (coarse → fine), each halved
TARGET_HS = [0.4, 0.2, 0.1, 0.05, 0.025, 0.0125]


# ═══════════════════════════════════════════════════════════════════
# Test / validation
# ═══════════════════════════════════════════════════════════════════

def run_test(save_dir: str = None):
    """Generate meshes at various resolutions and verify mesh size control."""
    print("=" * 90)
    print("Surface of Revolution Mesh Generation Test")
    print("  R(x) = (2 + cos(k·x)) / 4,  x ∈ [-1,1],  rotating around x-axis")
    print("  Three surfaces: k = π, 2π, 3π  (periodic with period 2)")
    print("=" * 90)

    all_stats: Dict[str, List[MeshStats]] = {}

    for name, surf in SURFACES.items():
        L_x = surf.arc_length_x()
        R_avg = surf.avg_radius()
        C_avg = surf.circumference_avg()
        k_pi = surf.k / np.pi

        print(f"\n{'─' * 90}")
        print(f"Surface: {name}  (k = {k_pi:.0f}π)")
        print(f"  L_x (x arc length)  = {L_x:.6f}")
        print(f"  R_avg (avg radius)  = {R_avg:.6f}")
        print(f"  C_avg (avg circumf) = {C_avg:.6f}")
        print(f"{'─' * 90}")

        header = (
            f"{'target_h':>10} {'N_x':>6} {'N_θ':>6} "
            f"{'n_vert':>8} {'n_face':>8} "
            f"{'h_mean':>10} {'h_min':>10} {'h_max':>10} "
            f"{'h/h_t':>7} {'h_std':>10} "
            f"{'aspect':>7} {'asp_std':>8}"
        )
        print(header)
        print("-" * len(header))

        stats_list = []
        for target_h in TARGET_HS:
            N_x, N_theta = compute_resolution(surf, target_h)
            mesh = generate_mesh(surf, N_x, N_theta)
            stats = compute_stats(mesh, target_h)
            stats_list.append(stats)

            print(
                f"{target_h:10.4f} {stats.N_x:6d} {stats.N_theta:6d} "
                f"{stats.n_vertices:8d} {stats.n_faces:8d} "
                f"{stats.h_mean:10.6f} {stats.h_min:10.6f} {stats.h_max:10.6f} "
                f"{stats.h_over_target:7.3f} {stats.h_std:10.6f} "
                f"{stats.aspect_mean:7.3f} {stats.aspect_std:8.4f}"
            )

            # Save OBJ if requested
            if save_dir:
                subdir = os.path.join(save_dir, name)
                os.makedirs(subdir, exist_ok=True)
                fname = f"{name}_h{target_h:.4f}.obj"
                mesh.save_obj(os.path.join(subdir, fname))

        all_stats[name] = stats_list

    # Summary
    print(f"\n{'=' * 90}")
    print("Summary")
    print("=" * 90)

    print(f"\n{'Surface':>8} {'k':>5} "
          f"{'h/h_t range':>16} {'aspect range':>16} "
          f"{'max h_std/h':>12}")

    for name, stats_list in all_stats.items():
        k_pi = SURFACES[name].k / np.pi
        h_ratios = [s.h_over_target for s in stats_list]
        aspects = [s.aspect_mean for s in stats_list]
        max_rel_std = max(s.h_std / s.h_mean for s in stats_list)

        print(f"{name:>8} {k_pi:5.0f}π "
              f"{min(h_ratios):8.3f}–{max(h_ratios):.3f} "
              f"{min(aspects):8.3f}–{max(aspects):.3f} "
              f"{max_rel_std:12.4f}")

    print(f"\nInterpretation:")
    print(f"  h/h_t ≈ 1.0  → mesh size matches target")
    print(f"  aspect ≈ 1.0 → triangles are approximately isosceles right")
    print(f"  h_std/h small → uniform mesh size across the surface")
    print(f"  (h_std/h grows with k because R'(x) variation increases)")

    if save_dir:
        print(f"\nOBJ files saved to: {save_dir}/")
        print(f"  Load with: xtpms compute --half-period 1,1,1 <file.obj>")

    return all_stats


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate test meshes for ADC convergence experiments.'
    )
    parser.add_argument('--save', metavar='DIR', default=None,
                        help='Directory to save OBJ files')
    parser.add_argument('--no-test', action='store_true',
                        help='Skip test output, only save meshes')
    args = parser.parse_args()

    stats = run_test(save_dir=args.save)
