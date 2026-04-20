# xtpms

> **Computational design of triply periodic minimal surfaces via asymptotic conductivity optimization**

This project implements the algorithm described in:

> Di Zhang, Ligang Liu. *Asymptotic analysis and design of shell-based thermal lattice metamaterials.* [arXiv:2506.22319](https://arxiv.org/abs/2506.22319), 2025.

Given a periodic surface mesh, xtpms computes its **Asymptotic Directional Conductivity (ADC)** tensor and iteratively deforms the surface to maximize effective thermal conductivity. The optimizer drives the surface toward the theoretical upper bound (APAC = 2/3), which is achieved only by triply periodic minimal surfaces.

## Idea

The **Asymptotic Directional Conductivity (ADC)** measures the effective thermal conductivity contributed by a periodic surface in the thin-shell limit. Its trace-average (APAC) has a sharp upper bound of **2/3**, achieved if and only if the surface has zero mean curvature everywhere -- i.e., it is a minimal surface.

This means: **maximizing APAC is equivalent to generating a TPMS**. Starting from any periodic seed mesh, the optimizer drives APAC toward 2/3 and the surface converges to a minimal surface automatically.

The main challenges are topology changes (neck formation requiring surgery), mesh quality degradation during large deformations, and maintaining periodic boundary consistency throughout all operations.

## Features

- **Compute ADC tensor** for any periodic surface mesh
- **Generate TPMS** from arbitrary seed meshes by optimizing APAC
- **Sample periodic surfaces** from built-in types (Gyroid, Schwarz P, Diamond), custom level set expressions, or random Fourier series
- **Periodize meshes**: merge periodic boundaries to create topologically closed periodic meshes
- **Custom objectives**: optimize individual conductivity components (k00, k11, k22) or arbitrary expressions
- **Non-uniform periods**: supports rectangular unit cells, not limited to cubic

## Building

### Prerequisites

- CMake 3.16+
- C++17 compiler (MSVC 2019+, GCC 9+, Clang 10+)

Dependencies (via **vcpkg** or **conda**):
- OpenMesh, CGAL, Eigen3, libigl, CLI11, GoogleTest

### Build

```bash
# With vcpkg:
cmake -B build -DCMAKE_TOOLCHAIN_FILE=<vcpkg-root>/scripts/buildsystems/vcpkg.cmake
cmake --build build --config Release

# With conda:
conda activate xtpms
cmake -B build
cmake --build build --config Release
```

## Usage

The `xtpms` CLI provides four subcommands: `sample`, `generate`, `optimize`, `compute`, `periodize`.

### Sample a periodic surface

```bash
# Built-in TPMS
xtpms sample -e gyroid -o gyroid.obj --half-period 1,1,1 -r 24
xtpms sample -e diamond -o diamond.obj --half-period 0.5,0.7,1.0

# Custom level set (2pi-periodic in x,y,z, auto-scaled to period)
xtpms sample -e "cos(x)+cos(y)+cos(z)" -o schwarzp.obj

# Random triperiodic Fourier surface
xtpms sample --random --kmax 2 --decay 2.0 -o random.obj
```

### Generate TPMS from seed (optimize with default settings)

```bash
xtpms generate -i seed.obj -o tpms.obj --half-period 1,1,1 --max-iter 100
```

`generate` is an alias for `optimize --objective apac` with surgery enabled by default.

### Optimize with full control

```bash
# Maximize APAC (default)
xtpms optimize -i mesh.obj -o result.obj --half-period 1,1,1 --max-iter 100

# Maximize k11 (x-direction conductivity)
xtpms optimize -i mesh.obj -o result.obj --objective k11

# Custom expression objective
xtpms optimize -i mesh.obj -o result.obj --objective "(k00-k11)^2+(k11-k22)^2"

# Tune optimization parameters
xtpms optimize -i mesh.obj -o result.obj \
    --max-step 1.0 \
    --mcf-weight 0.1 \
    --adaptive-eps 1.0 \
    --surgery-tol 25 \
    --surgery-interval 4 \
    --nf-limit 100000 \
    --output-dir ./iterations
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
```

### Periodize a mesh

```bash
xtpms periodize -i raw_mesh.obj -o periodic.obj --half-period 1,1,1
```

## C++ API

```cpp
#include "PeriodicMesh.h"
#include "VertexGeometry.h"
#include "AsymptoticConductivity.h"

// Load and set up periodic mesh
xtpms::PeriodicTriMesh mesh;
mesh.setHalfPeriod({1.0, 1.0, 1.0});
// ... add vertices and faces ...
mesh.mergePeriodBoundary();

// Compute ADC
auto geom = xtpms::computeVertexGeometry(mesh);
Eigen::MatrixX3d u;
Eigen::Matrix3d kA = xtpms::solveAsymptoticConductivity(mesh, geom, u);
double apac = kA.trace() / 3.0;  // theoretical max: 2/3

// Optimize
xtpms::TailorADCOptions opts;
opts.objectiveType = "apac";
opts.maxIter = 100;
opts.enableRemesh = true;
opts.enableSurgery = true;
xtpms::tailorADC(mesh, opts);

// Export
mesh.saveUnitCell("optimized.obj");
```

## Optimization Pipeline

Each iteration:

1. **Surgery** (every N iterations): detect high-curvature singularities, excise faces, CGAL hole filling, bilaplacian fairing
2. **Remesh**: curvature-adaptive Delaunay remeshing (split/collapse/flip/circumcenter smooth)
3. **Solve**: cotangent Laplacian cell problem for 3 coordinate directions
4. **Sensitivity**: per-vertex shape derivative via second fundamental form
5. **Precondition**: Laplacian-smoothed gradient: $G = -cL + A_v$, solve $G\,dn = -A_v f_{weight}\,\partial f/\partial v_n$
6. **Line search**: backtracking with per-face cos(60) flip detection
7. **MCF regularization**: mean curvature flow weighted displacement

## Project Structure

```
xtpms/
    main.cpp                    CLI application
    src/
        MeshTypes.h             OpenMesh type definitions
        PeriodicMesh.h/cpp      Periodic mesh: boundary merge, surgery, split/save
        VertexGeometry.h/cpp    1-ring curvature, cotangent Laplacian, coordinate utilities
        PeriodicRemesh.h/cpp    Curvature-adaptive Delaunay remeshing
        AsymptoticConductivity.h/cpp  ADC solve, sensitivity, shape optimization
        MarchingCubes.h/cpp     Isosurface extraction with periodic boundary handling
        AABBTree.h              CGAL-based nearest-point queries
tests/
    test_periodic_mesh_periodize.cpp
    test_marching_cubes.cpp
    test_aabb_tree.cpp
    data/                       Test meshes (rod3, neck)
```

## License

[MIT](LICENSE)

## Citation

```bibtex
@article{zhang2025asymptotic,
  title   = {Asymptotic analysis and design of shell-based thermal lattice metamaterials},
  author  = {Zhang, Di and Liu, Ligang},
  journal = {arXiv preprint arXiv:2506.22319},
  year    = {2025}
}
```
