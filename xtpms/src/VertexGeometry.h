#pragma once

// Discrete differential geometry on periodic triangle meshes:
// 1-ring curvature data (Compile1ring), per-vertex geometry (VertexGeometry),
// cotangent Laplacian assembly, and periodic coordinate utilities.

#include "PeriodicMesh.h"

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <string>
#include <vector>

namespace xtpms {

// ── Periodic coordinate utilities ──────────────────────────

// Wrap a difference vector into [-halfPeriod, halfPeriod) per axis.
DefaultTriMesh::Point makePeriod(const DefaultTriMesh::Point& v, const DefaultTriMesh::Point& hp);

// Convert between OpenMesh Vec3d and Eigen::Vector3d.
Eigen::Vector3d toEig(const DefaultTriMesh::Point& v);
DefaultTriMesh::Point toOM(const Eigen::Vector3d& v);

// Compute the periodic edge vector from v0 to v1.
Eigen::Vector3d periodicEdgeVec(const PeriodicTriMesh& mesh,
								PeriodicTriMesh::VertexHandle v0,
								PeriodicTriMesh::VertexHandle v1);

// Get the three vertex coordinates of a face, periodically wrapped near v0.
// Returns a 3x3 matrix with columns = vertex positions.
Eigen::Matrix3d getFacePeriodTri(const PeriodicTriMesh& mesh, PeriodicTriMesh::FaceHandle fh);

// Get vertex indices of a face.
void getFaceVertexIdx(const PeriodicTriMesh& mesh, PeriodicTriMesh::FaceHandle fh, int idx[3]);

// Get the periodic 1-ring neighborhood of a vertex.
// Face traversal ensures CCW order.
void getPeriodicRing(const PeriodicTriMesh& mesh,
					 PeriodicTriMesh::VertexHandle vh,
					 Eigen::Vector3d& outCenter,
					 std::vector<Eigen::Vector3d>& outRing);

// ── 1-ring curvature data ──────────────────────────────────

// Compiled geometric quantities from vertex 1-ring:
// Voronoi area, angle-weighted normal, mean curvature vector/scalar,
// Gaussian curvature.
struct Compile1ring {
	double As = 0;				 // Voronoi area
	Eigen::Vector3d nv{0, 0, 0}; // angle-weighted vertex normal
	Eigen::Vector3d Lx{0, 0, 0}; // mean curvature vector
	double H = 0;				 // mean curvature (scalar)
	double K = 0;				 // Gaussian curvature

	Compile1ring() = default;
	Compile1ring(const Eigen::Vector3d& o, const std::vector<Eigen::Vector3d>& ring);
};

// ── Per-vertex geometry ────────────────────────────────────

// Discrete differential geometry data computed from the mesh.
struct VertexGeometry {
	std::vector<double> cotWeights;				// per-edge cotangent weight
	std::vector<Eigen::Vector3d> edgeVectors;	// per-edge period-wrapped edge vector
	Eigen::VectorXd vertexAreas;				// per-vertex dual area
	std::vector<Eigen::Vector3d> vertexNormals; // per-vertex angle-weighted normal
	std::vector<Compile1ring> vrings;			// per-vertex compiled 1-ring
};

// Compute all per-vertex geometry from a periodic mesh.
VertexGeometry computeVertexGeometry(const PeriodicTriMesh& mesh);

// Per-vertex singularity measure used by surgery.
// surgeryType==1: |H|; surgeryType==2: max(|kappa_1|, |kappa_2|)
std::vector<double> computeSingularityMeasure(const PeriodicTriMesh& mesh, int surgeryType = 2);

// ── Cotangent Laplacian ────────────────────────────────────

// Compute the cotangent weight for a single edge.
double computeCotWeight(const PeriodicTriMesh& mesh, PeriodicTriMesh::EdgeHandle eh);

// Assemble the global cotangent Laplacian matrix.
Eigen::SparseMatrix<double> assembleLaplacian(const PeriodicTriMesh& mesh,
											  const std::vector<double>& cotWeights);

} // namespace xtpms
