#pragma once

// Periodic mesh: cross-period edge detection, period-aware neighborhood access; periodizeFrom uses AABB narrow-band SDF + periodic symmetry + Marching Cubes.
#include "MeshTypes.h"

#include <array>
#include <cmath>
#include <tuple>
#include <vector>

namespace xtpms {

// Vertex/edge periodic boundary flag bits: bit0=x_min, bit1=x_max, bit2=y_min, bit3=y_max, bit4=z_min, bit5=z_max
struct BoundaryFlag {
	static constexpr int kMinBit[3] = {0, 2, 4};
	static constexpr int kMaxBit[3] = {1, 3, 5};
	static constexpr int kPeriodMask = 0x3F;
	static constexpr int kMinMask = 0x15; // 010101
	static constexpr int kMaxMask = 0x2A; // 101010

	int flag = 0;

	void classify(double px, double py, double pz,
				  double xmin, double xmax, double ymin, double ymax,
				  double zmin, double zmax, double tol) {
		flag = 0;
		const double p[3] = {px, py, pz};
		const double lo[3] = {xmin, ymin, zmin};
		const double hi[3] = {xmax, ymax, zmax};
		for (int i = 0; i < 3; ++i) {
			if (p[i] - lo[i] < tol) flag |= (1 << kMinBit[i]);
			if (hi[i] - p[i] < tol) flag |= (1 << kMaxBit[i]);
		}
	}

	bool isMinBoundary(int axis) const { return (flag >> kMinBit[axis]) & 1; }
	bool isMaxBoundary(int axis) const { return (flag >> kMaxBit[axis]) & 1; }
	bool isMinBoundary() const { return flag & kMinMask; }
	bool isMaxBoundary() const { return flag & kMaxMask; }
	bool isBoundary() const { return flag & kPeriodMask; }
	int getMask() const { return flag & kPeriodMask; }
};

// Periodic boundary merging options
struct MergeBoundaryOptions {
	double shortEdgeTol{2e-5};      // Short edge collapse threshold
	double projectionTol{0.01};     // Projection distance warning threshold (squared distance)
	double vertexWeldTol{1e-6};     // Vertex welding tolerance
};

// periodizeFrom: translate mesh, periodic replication + AABB narrow-band voxels + signed distance + periodic symmetry + Marching Cubes to generate periodic mesh.
struct PeriodizeOptions {
	double bboxPaddingWorld{0.0};
	double bboxPaddingFraction{0.0};
	// When >0: attempt to collapse boundary edges shorter than this length (requires OpenMesh is_collapse_ok). 0 means no collapse.
	double collapseShortBoundaryEdgesBelow{0.0};

	/// AABB padding of the fundamental domain \([0,2\cdot\text{halfPeriod}]\)^3 (written to resolvedBBox*).
	/// Number of voxel cells per axis \(N\) (grid points \(N+1\)), domain is \([0,L_x]\times[0,L_y]\times[0,L_z]\).
	int mcCellsPerAxis{32};
	/// Number of dilation layers in the 26-neighborhood for surface-intersecting voxels (fills narrow band, asymmetric corners).
	int mcVoxelDilateLayers{2};
	double mcIsoValue{0.0};
	bool mcWeldVertices{true};

	std::array<double, 3> resolvedBBoxMin{};
	std::array<double, 3> resolvedBBoxMax{};
};

// Topological surgery options
struct SurgeryOptions {
	// Aligned with the mainstream minsurf settings (surgery-tol=25 surgery-type=2, used by 8/14 scripts)
	double singularityTol{25.0};      // Singularity absolute threshold
	double singularityRatio{0.0};     // Adaptive ratio (0 = disabled, use absolute threshold only)
	                                  // max(tol, ratio * avgH)
	double maxCutAreaFraction{0.3};   // Maximum fraction of cut area; surgery is aborted if exceeded (safety valve)
	int surgeryType{2};               // 1: |H|; 2: max(|kappa_1|,|kappa_2|) (minsurf mainstream)
	double islandCullRatio{0.1};      // Islands with face count < largest component * ratio are removed
};

class PeriodicTriMesh : public DefaultTriMesh {
public:
	using Base = DefaultTriMesh;
	using VertexHandle = typename Base::VertexHandle;
	using EdgeHandle = typename Base::EdgeHandle;
	using HalfedgeHandle = typename Base::HalfedgeHandle;
	using Vec3d = typename Base::Point;

public:
	PeriodicTriMesh();

	void setHalfPeriod(const Vec3d& halfPeriod);
	Vec3d halfPeriod() const;

	Vec3d wrapVector(const Vec3d& v) const;

	Vec3d shift2origin(const Vec3d& p) const;

	void shift2originInPlace();

	bool isPeriodicEdge(VertexHandle center, VertexHandle neighbor) const;

	bool isPeriodicEdge(EdgeHandle eh) const;

	std::tuple<std::vector<HalfedgeHandle>, std::vector<Vec3d>> e1ring(VertexHandle center) const;

	// Periodize: copy -> (optional) collapse short boundary edges -> translate AABB min to origin -> 3x3x3 periodic replication -> AABB tree -> narrow-band voxels ->
	// grid-point SDF (sign from closest face normal) -> periodic grid-point group weighted average -> Marching Cubes writes back to this mesh.
	// If &naiveMesh == this, the mesh is copied first before processing.
	void periodizeFrom(const DefaultTriMesh& naiveMesh, PeriodizeOptions& options);

	// Collapse boundary edges with length < max(shortTol, 1e-12) (if is_collapse_ok).
	void mergePeriodEdges(double shortEdgeTol);

	// Periodic boundary merging: edge matching + split alignment + vertex welding, turning a topologically non-periodic mesh into a truly periodic mesh.
	void mergePeriodBoundary(const MergeBoundaryOptions& options = MergeBoundaryOptions{});

	// Wrap vertices into the fundamental domain [0, 2*halfPeriod).
	void periodShift();

	// Detect and remove island connected components that do not span the full period.
	// Starting from the seed vertex of each connected component, BFS along 3 axes accumulating coordinate differences after periodic wrapping.
	// If the accumulated distance along an axis >= one full period, the component spans that axis.
	// A component is kept as long as it spans at least one axis; otherwise it is removed.
	// Returns the number of deleted faces.
	int removeNonPeriodicIslands();

	// Keep only the largest connected component by face count; return the number of removed components.
	// Used to clean up residual islands after surgery, preventing multi-dimensional null space in the FEM Laplacian.
	int keepLargestComponent();

	// Truncate cross-period edges at periodic boundaries; the mesh is no longer periodically closed after truncation.
	// splitEdges=true: Phase 1 splits edges at boundaries + Phase 2 dupPeriodFaces
	// splitEdges=false: Phase 2 dupPeriodFaces only (avoids bumps introduced by splitting)
	void splitUnitCell(bool splitEdges = true);

	// Save unit cell. When split=true, the mesh is copied and splitUnitCell is applied before saving (default).
	// When split=false, the original unwrap logic is used to save (does not modify the original mesh).
	// splitEdges controls whether splitUnitCell executes Phase 1 edge splitting.
	bool saveUnitCell(const std::string& filename, bool split = true,
					  bool splitEdges = true) const;

	// Topological surgery: detect neck singularities (high curvature), remove surrounding faces, fill holes, and remove islands.
	// Returns true if surgery was performed.
	bool surgery(const SurgeryOptions& opts = SurgeryOptions{});

private:
	Vec3d halfPeriod_{};
};

} // namespace xtpms
