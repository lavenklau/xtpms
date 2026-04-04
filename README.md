# xtpms

> **Computational design and optimization of triply periodic minimal surfaces**

Triply periodic minimal surfaces (TPMS) — such as the Gyroid, Schwarz P, and Diamond — are nature-inspired geometries found in butterfly wings, sea urchin skeletons, and block copolymer self-assembly. Their unique combination of high surface area, structural efficiency, and tunable porosity makes them ideal building blocks for:

- **Lightweight metamaterials** and lattice structures for aerospace and automotive
- **Tissue engineering scaffolds** with controlled pore architecture
- **Heat exchangers** and thermal management devices
- **Battery electrodes** and catalytic substrates
- **Acoustic and photonic crystals**

While TPMS geometries are widely used, **designing optimal TPMS for specific physical properties remains challenging**. xtpms addresses this gap: given a periodic surface, it automatically optimizes the shape to maximize effective thermal/electrical conductivity — a key performance metric for heat exchangers, electrodes, and conductive scaffolds.

<!-- TODO: add a figure showing optimization from rod-3 to near-optimal surface -->

## What xtpms Does

xtpms computes the **effective (homogenized) conductivity tensor** of a periodic surface using asymptotic analysis, then **iteratively deforms the surface** to optimize it. The optimizer drives the surface toward the theoretical upper bound (APAC = 2/3) — a limit achieved only by true minimal surfaces.

**Example: rod-3 scaffold optimization**

```
iter= 0   APAC = 0.421   (initial rod-3 geometry)
iter=10   APAC = 0.508
iter=20   APAC = 0.550
iter=30   APAC = 0.594
iter=40   APAC = 0.634   ← singularity surgery
iter=50   APAC = 0.666   ← converged to theoretical upper bound 2/3
```

The framework is not limited to conductivity — it supports **general objective functions** on the conductivity tensor (e.g., single component k₁₁, isotropy, anisotropy ratio), and can be extended to other homogenized properties.

## Features

- **Asymptotic homogenization** on discrete periodic surfaces via cotangent Laplacian FEM
- **Shape sensitivity** from adjoint-based Hadamard formula with discrete second fundamental form
- **Preconditioned gradient descent** with Laplacian smoothing and backtracking line search
- **Mean curvature flow** regularization to drive surfaces toward minimal surface configurations
- **Curvature-adaptive Delaunay remeshing** to maintain mesh quality during optimization
- **Singularity surgery** for automatic detection and removal of topological necks (high-curvature regions detected → CGAL hole filling → bilaplacian fairing)
- **Non-uniform tri-axis periods** — the unit cell can be rectangular, not just cubic
- **Periodic mesh utilities** — boundary merging, unit cell splitting, `saveUnitCell` visualization

## Building

### Prerequisites

Install dependencies via [vcpkg](https://github.com/microsoft/vcpkg):

```bash
vcpkg install cgal openmesh eigen3 gtest libigl
```

### Configure and build

```bash
cmake -B build -DCMAKE_TOOLCHAIN_FILE=<path-to-vcpkg>/scripts/buildsystems/vcpkg.cmake
cmake --build build --config Release
```

### Run tests

```bash
cd build
ctest -C Release --output-on-failure

# Run a specific test:
./Release/test_periodic_mesh_periodize --gtest_filter="*RevolutionSurface_K11*"
```

## Quick Start

### Optimize a TPMS surface

```cpp
#include "PeriodicMesh.h"
#include "AsymptoticConductivity.h"

// 1. Load mesh and set up periodic boundary
xtpms::PeriodicTriMesh mesh;
mesh.setHalfPeriod({1.0, 1.0, 1.0});  // period = [0, 2]^3
// ... load vertices and faces, then:
mesh.mergePeriodBoundary();

// 2. Run optimization
xtpms::TailorADCOptions opts;
opts.objectiveType = "apac";    // maximize trace(kA)/3
opts.maxIter = 100;
opts.maxStep = 1.0;
opts.mcfWeight = 0.1;           // mean curvature flow regularization
opts.enableRemesh = true;       // adaptive Delaunay remeshing
opts.enableSurgery = true;      // automatic neck removal
opts.surgeryStartIter = 40;     // delay surgery to let optimization settle

xtpms::tailorADC(mesh, opts);

// 3. Export
mesh.saveUnitCell("optimized.obj");
```

### Just compute effective conductivity

```cpp
auto geom = xtpms::computeVertexGeometry(mesh);
Eigen::MatrixX3d u;
Eigen::Matrix3d kA = xtpms::solveAsymptoticConductivity(mesh, geom, u);

double apac = kA.trace() / 3.0;
std::cout << "APAC = " << apac << std::endl;  // theoretical max: 2/3
std::cout << "kA =\n" << kA << std::endl;
```

### Custom objective functions

```cpp
// Maximize k11 (conductivity along x-axis)
opts.objectiveType = "k00";

// Or define via the evaluateADCObjective interface:
// "apac"  → trace(kA)/3
// "k00"   → kA(0,0)
// "k11"   → kA(1,1)
// "k22"   → kA(2,2)
// "iso"   → isotropy (minimize eigenvalue spread)
```

## How It Works

### Homogenized conductivity

For a periodic surface $S$ embedded in a unit cell, the effective conductivity tensor is:

$$\mathbf{k}_A = \frac{1}{|S|} \int_S \nabla\chi \cdot \nabla\chi^T \, dA$$

where $\chi = u + y$ is the corrector field, and $u$ solves the cell problem $\Delta_S u = -\mathrm{div}_S(\mathbf{e})$.

The averaged principal conductivity (APAC) = $\mathrm{tr}(\mathbf{k}_A)/3$ has a theoretical upper bound of $2/3$, achieved by triply periodic minimal surfaces (surfaces with zero mean curvature everywhere).

### Shape sensitivity

The shape derivative with respect to normal displacement $\delta n$ follows the Hadamard formula:

$$\delta k_{A,ij} = \frac{1}{|S|} \int_S \left[ 2H |\nabla\chi|^2 - 2\,\mathrm{II}(\nabla\chi, \nabla\chi) \right] \delta n \, dA$$

where $H$ is the mean curvature and $\mathrm{II}$ is the second fundamental form. This is computed entirely from the surface geometry and the cell problem solution — no adjoint solve is needed.

### Optimization pipeline

Each iteration:

1. **Solve** the cell problem on the current mesh (cotangent Laplacian)
2. **Compute** shape sensitivity via the Hadamard formula
3. **Precondition** the gradient with a Laplacian smoother
4. **Add** mean curvature flow for regularization
5. **Line search** with face-flip checking
6. **Remesh** adaptively based on local curvature
7. **(Optional) Surgery** to remove topological necks

## Project Structure

```
xtpms/src/
    PeriodicMesh.h/cpp           Periodic mesh operations
    PeriodicRemesh.h/cpp         Curvature-adaptive Delaunay remeshing
    AsymptoticConductivity.h/cpp ADC computation, sensitivity, optimizer
    MarchingCubes.h/cpp          Isosurface extraction
tests/
    test_periodic_mesh_periodize.cpp
    data/                        Test meshes
```

## Validation

The implementation is validated against:

- **Analytic k₁₁** for revolution surfaces with known closed-form conductivity
- **Variational sensitivity** compared with finite-difference on revolution surfaces
- **Schwarz P / Gyroid** known optimal APAC = 2/3
- **Mesh convergence** on multiple TPMS types at increasing resolution

## License

[MIT](LICENSE)

## Citation

If you find this useful, please consider citing:

```bibtex
@software{xtpms2025,
  title  = {xtpms: Computational design of triply periodic minimal surfaces},
  author = {Zhang, Di},
  year   = {2025},
  url    = {https://github.com/lavenklau/xtpms}
}
```
