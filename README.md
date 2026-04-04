# xtpms

**Shape optimization of triply periodic minimal surfaces for extremal effective conductivity**

xtpms optimizes the geometry of periodic surfaces to maximize their asymptotic directional conductivity (ADC) tensor. Starting from an initial TPMS approximation (Schwarz P, Gyroid, etc.) or arbitrary periodic mesh, it iteratively deforms the surface toward the theoretical upper bound of effective conductivity.

## Key Features

- **Asymptotic homogenization** on discrete periodic surface meshes (cotangent Laplacian FEM)
- **Shape sensitivity** via adjoint-based Hadamard formula with second fundamental form
- **Gradient-based optimization** with Laplacian preconditioning and backtracking line search
- **Periodic Delaunay remeshing** with curvature-adaptive edge length control
- **Singularity surgery** for automatic neck removal (curvature detection + CGAL hole filling + bilaplacian fairing)
- **Non-uniform tri-axis periods** supported throughout
- **Revolution surface validation** with closed-form k11 and variational sensitivity

## Results

Starting from a rod-3 model (APAC = 0.42), the optimizer converges to the theoretical upper bound (APAC = 2/3) in ~50 iterations:

```
iter= 0  APAC=0.421  nv=4457
iter=10  APAC=0.508  nv=4865
iter=20  APAC=0.550  nv=5584
iter=30  APAC=0.594  nv=5470
iter=40  APAC=0.634  nv=4746  (surgery)
iter=50  APAC=0.666  nv=2156
```

## Building

### Prerequisites

Install via [vcpkg](https://github.com/microsoft/vcpkg):

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
# or run specific test:
./Release/test_periodic_mesh_periodize --gtest_filter="*RevolutionSurface_K11*"
```

## Usage

### Basic ADC optimization

```cpp
#include "PeriodicMesh.h"
#include "AsymptoticConductivity.h"
#include "PeriodicRemesh.h"

// Load and merge periodic boundary
xtpms::PeriodicTriMesh mesh;
mesh.setHalfPeriod({1.0, 1.0, 1.0});
// ... load mesh from OBJ, then:
mesh.mergePeriodBoundary();

// Optimize
xtpms::TailorADCOptions opts;
opts.objectiveType = "apac";   // maximize trace(kA)/3
opts.maxIter = 100;
opts.maxStep = 1.0;
opts.mcfWeight = 0.1;          // mean curvature flow regularization
opts.enableRemesh = true;
opts.enableSurgery = true;
opts.surgeryStartIter = 40;

xtpms::tailorADC(mesh, opts);

// Export result
mesh.saveUnitCell("optimized.obj");
```

### Compute effective conductivity

```cpp
auto geom = xtpms::computeVertexGeometry(mesh);
Eigen::MatrixX3d u;
Eigen::Matrix3d kA = xtpms::solveAsymptoticConductivity(mesh, geom, u);
std::cout << "APAC = " << kA.trace() / 3.0 << std::endl;
```

### Split unit cell for visualization

```cpp
mesh.splitUnitCell();  // cut at period boundaries, duplicate vertices
OpenMesh::IO::write_mesh(mesh, "unit_cell.obj");
```

## Mathematical Background

The effective conductivity tensor of a periodic surface is:

$$\mathbf{k}_A = \frac{1}{|S|} \int_S \nabla\chi \cdot \nabla\chi \, dA$$

where $\chi = u + y$ is the corrector field satisfying $\Delta_S u = -\text{div}_S(\mathbf{e})$ on the surface $S$.

The shape derivative with respect to normal displacement $\delta n$ is computed via the Hadamard formula:

$$\delta k_{A,ij} = \frac{1}{|S|} \int_S \left[ 2H |\nabla\chi|^2 - 2\,\text{II}(\nabla\chi, \nabla\chi) \right] \delta n \, dA$$

where $H$ is the mean curvature and $\text{II}$ is the second fundamental form.

The theoretical upper bound for APAC (averaged principal conductivity) is $2/3$, achieved by triply periodic minimal surfaces.

## Project Structure

```
xtpms/
  src/
    PeriodicMesh.h/cpp       - Periodic mesh: merge, split, surgery, saveUnitCell
    PeriodicRemesh.h/cpp      - Delaunay remesh with curvature-adaptive sizing
    AsymptoticConductivity.h/cpp - ADC solve, sensitivity, optimization loop
    MarchingCubes.h/cpp       - Marching cubes isosurface extraction
tests/
    test_periodic_mesh_periodize.cpp - All unit tests
    data/                     - Test meshes (rod3, neck, etc.)
CMakeLists.txt
```

## License

MIT

## Citation

If you use this code in your research, please cite:

```bibtex
@software{xtpms2025,
  title = {xtpms: Shape optimization of triply periodic surfaces},
  author = {Zhang, Di},
  year = {2025},
  url = {https://github.com/lavenklau/xtpms}
}
```
