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

- CMake 3.16+
- C++17 compiler (MSVC 2019+, GCC 9+, Clang 10+)

Install dependencies via **conda** (recommended) or **vcpkg**:

**Option A: conda**
```bash
conda env create -f environment.yml
conda activate xtpms
```

**Option B: vcpkg**
```bash
vcpkg install cgal openmesh eigen3 gtest libigl
```

### Configure and build

```bash
# With conda (auto-detected):
cmake -B build
cmake --build build --config Release

# With vcpkg (specify toolchain):
cmake -B build -DCMAKE_TOOLCHAIN_FILE=<path-to-vcpkg>/scripts/buildsystems/vcpkg.cmake
cmake --build build --config Release
```

## Command-Line Tool

The `xtpms` CLI provides five subcommands:

### Sample isosurface from level set

Extract a periodic surface from a level set expression. Expressions use **2π-periodic** convention — they are automatically scaled to match the physical period.

```bash
# Built-in types
xtpms sample -e gyroid -o gyroid.obj --half-period 0.5,0.5,0.5 -r 20
xtpms sample -e schwarzp -o schwarzp.obj -r 24
xtpms sample -e diamond -o diamond.obj --half-period 1,0.7,0.4

# Custom expression (2π-periodic in x, y, z)
xtpms sample -e "cos(x)+cos(y)+cos(z)" -o schwarzp.obj
xtpms sample -e "sin(x)*cos(y)+sin(y)*cos(z)+sin(z)*cos(x)" -o gyroid.obj
xtpms sample -e "cos(x)+cos(y)+cos(z)+0.3*sin(2*x)*sin(y)" -o perturbed.obj

# Random triperiodic surface
xtpms sample --random -o random_surface.obj -r 16
```

### Generate TPMS from seed mesh

Take any periodic mesh as a seed and optimize it toward a TPMS. The period is automatically detected from the bounding box, and boundary vertices are clamped to the bbox faces.

```bash
# Optimize seed mesh toward TPMS (auto-detect period)
xtpms generate -i scaffold.obj -o tpms.obj --max-iter 100

# With split output for visualization
xtpms generate -i seed.obj -o tpms_unit_cell.obj --split
```

### Periodize an existing mesh

```bash
# Merge periodic boundary of an OBJ mesh
xtpms periodize -i raw_mesh.obj -o periodic.obj --half-period 1,1,1

# Split unit cell (duplicate boundary vertices) for visualization
xtpms periodize -i raw_mesh.obj -o unit_cell.obj --half-period 1,1,1 --split
```

### Compute effective conductivity

```bash
xtpms compute -i periodic_mesh.obj --half-period 1,1,1
# Output:
#   kA =
#       0.6664  0.0006  0.0008
#       0.0006  0.6663 -0.0005
#       0.0008 -0.0005  0.6665
#   APAC = 0.6664
#   eigenvalues: 0.6651 0.6669 0.6673
```

### Optimize conductivity

```bash
# Maximize APAC (default objective)
xtpms optimize -i mesh.obj -o optimized.obj --half-period 1,1,1 \
    --objective apac --max-iter 100

# Maximize k11 with surgery enabled
xtpms optimize -i mesh.obj -o opt_k11.obj --half-period 1,1,1 \
    --objective k11 --surgery --surgery-start 40

# Custom objective expression (kA components: k00..k22)
xtpms optimize -i mesh.obj -o isotropic.obj --half-period 1,1,1 \
    --objective "(k00-k11)^2+(k11-k22)^2+(k00-k22)^2"

# Save intermediate meshes for animation
xtpms optimize -i mesh.obj -o final.obj --half-period 1,1,1 \
    --output-dir ./iterations --max-iter 50
```

## C++ API

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
