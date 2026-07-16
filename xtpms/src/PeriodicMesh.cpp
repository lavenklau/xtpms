#include "PeriodicMesh.h"

#include "AABBTree.h"
#include "AsymptoticConductivity.h"
#include "MarchingCubes.h"
#include "VertexGeometry.h"

#include <Eigen/Dense>
#include <OpenMesh/Core/IO/MeshIO.hh>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <CGAL/AABB_segment_primitive.h>
#include <CGAL/AABB_traits.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/Simple_cartesian.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Polygon_mesh_processing/triangulate_hole.h>
#include <CGAL/Polygon_mesh_processing/border.h>

#include <igl/cotmatrix.h>
#include <igl/massmatrix.h>
#include <igl/min_quad_with_fixed.h>

namespace xtpms {

namespace {

using Vec3d = PeriodicTriMesh::Vec3d;

void computeAxisAlignedBounds(const DefaultTriMesh& mesh, Vec3d& outMin, Vec3d& outMax) {
	if (mesh.n_vertices() == 0) {
		throw std::invalid_argument("periodizeFrom: mesh has no vertices");
	}
	auto v_it = mesh.vertices_begin();
	outMin = outMax = mesh.point(*v_it);
	for (++v_it; v_it != mesh.vertices_end(); ++v_it) {
		const Vec3d& p = mesh.point(*v_it);
		for (int i = 0; i < 3; ++i) {
			outMin[i] = std::min(outMin[i], p[i]);
			outMax[i] = std::max(outMax[i], p[i]);
		}
	}
}

void applyBBoxPadding(Vec3d& min, Vec3d& max, double padWorld, double padFrac) {
	for (int i = 0; i < 3; ++i) {
		const double ext = static_cast<double>(max[i]) - static_cast<double>(min[i]);
		const double pad = std::max(padWorld, padFrac * ext);
		min[i] -= pad;
		max[i] += pad;
	}
}

void assignTriMesh(const DefaultTriMesh& in, PeriodicTriMesh& out) {
	out.clear();
	std::vector<PeriodicTriMesh::VertexHandle> vh;
	vh.reserve(in.n_vertices());
	for (auto v_it = in.vertices_begin(); v_it != in.vertices_end(); ++v_it) {
		vh.push_back(out.add_vertex(in.point(*v_it)));
	}
	for (auto f_it = in.faces_begin(); f_it != in.faces_end(); ++f_it) {
		auto fv_it = in.cfv_iter(*f_it);
		const int a = (*fv_it).idx();
		++fv_it;
		const int b = (*fv_it).idx();
		++fv_it;
		const int c = (*fv_it).idx();
		out.add_face(
			vh[static_cast<size_t>(a)], vh[static_cast<size_t>(b)], vh[static_cast<size_t>(c)]);
	}
}

void meshToRaw(const DefaultTriMesh& mesh,
			   std::vector<std::array<double, 3>>& verts,
			   std::vector<TriMeshFace>& faces) {
	verts.resize(mesh.n_vertices());
	for (auto v_it = mesh.vertices_begin(); v_it != mesh.vertices_end(); ++v_it) {
		const Vec3d& p = mesh.point(*v_it);
		verts[static_cast<size_t>((*v_it).idx())] = {
			static_cast<double>(p[0]), static_cast<double>(p[1]), static_cast<double>(p[2])};
	}
	faces.reserve(mesh.n_faces());
	for (auto f_it = mesh.faces_begin(); f_it != mesh.faces_end(); ++f_it) {
		auto fv_it = mesh.cfv_iter(*f_it);
		const std::size_t a = static_cast<std::size_t>((*fv_it).idx());
		++fv_it;
		const std::size_t b = static_cast<std::size_t>((*fv_it).idx());
		++fv_it;
		const std::size_t c = static_cast<std::size_t>((*fv_it).idx());
		faces.push_back({a, b, c});
	}
}

void translateMeshInPlace(PeriodicTriMesh& mesh, const Vec3d& delta) {
	for (auto v_it = mesh.vertices_begin(); v_it != mesh.vertices_end(); ++v_it) {
		mesh.set_point(*v_it, mesh.point(*v_it) - delta);
	}
}

void buildExtendedMesh27(const std::vector<std::array<double, 3>>& baseVerts,
						 const std::vector<TriMeshFace>& baseFaces,
						 double Lx,
						 double Ly,
						 double Lz,
						 std::vector<std::array<double, 3>>& outVerts,
						 std::vector<TriMeshFace>& outFaces) {
	outVerts.clear();
	outFaces.clear();
	if (baseVerts.empty() || baseFaces.empty()) {
		return;
	}
	outVerts.reserve(baseVerts.size() * 27);
	outFaces.reserve(baseFaces.size() * 27);
	for (int di = -1; di <= 1; ++di) {
		for (int dj = -1; dj <= 1; ++dj) {
			for (int dk = -1; dk <= 1; ++dk) {
				const std::size_t base = outVerts.size();
				const double ox = static_cast<double>(di) * Lx;
				const double oy = static_cast<double>(dj) * Ly;
				const double oz = static_cast<double>(dk) * Lz;
				for (const auto& p : baseVerts) {
					outVerts.push_back({p[0] + ox, p[1] + oy, p[2] + oz});
				}
				for (const TriMeshFace& f : baseFaces) {
					outFaces.push_back({base + f[0], base + f[1], base + f[2]});
				}
			}
		}
	}
}

inline int cellIndex(int i, int j, int k, int nx, int ny) {
	return i + nx * (j + ny * k);
}

inline int nodeIndex(int i, int j, int k, int np) {
	return i + np * (j + np * k);
}

void markCellsOverlappingFundDomain(const std::vector<std::array<double, 3>>& extVerts,
									const std::vector<TriMeshFace>& extFaces,
									double Lx,
									double Ly,
									double Lz,
									int nx,
									int ny,
									int nz,
									std::vector<uint8_t>& active) {
	const double dx = Lx / static_cast<double>(nx);
	const double dy = Ly / static_cast<double>(ny);
	const double dz = Lz / static_cast<double>(nz);
	active.assign(static_cast<size_t>(nx * ny * nz), 0);
	for (const TriMeshFace& f : extFaces) {
		const auto& pa = extVerts[f[0]];
		const auto& pb = extVerts[f[1]];
		const auto& pc = extVerts[f[2]];
		double bminx = std::min({pa[0], pb[0], pc[0]});
		double bminy = std::min({pa[1], pb[1], pc[1]});
		double bminz = std::min({pa[2], pb[2], pc[2]});
		double bmaxx = std::max({pa[0], pb[0], pc[0]});
		double bmaxy = std::max({pa[1], pb[1], pc[1]});
		double bmaxz = std::max({pa[2], pb[2], pc[2]});
		bminx = std::max(0.0, bminx);
		bminy = std::max(0.0, bminy);
		bminz = std::max(0.0, bminz);
		bmaxx = std::min(Lx, bmaxx);
		bmaxy = std::min(Ly, bmaxy);
		bmaxz = std::min(Lz, bmaxz);
		if (bmaxx < bminx || bmaxy < bminy || bmaxz < bminz) {
			continue;
		}
		const int i0 = std::clamp(static_cast<int>(std::floor(bminx / dx)), 0, nx - 1);
		const int i1 = std::clamp(static_cast<int>(std::floor(bmaxx / dx)), 0, nx - 1);
		const int j0 = std::clamp(static_cast<int>(std::floor(bminy / dy)), 0, ny - 1);
		const int j1 = std::clamp(static_cast<int>(std::floor(bmaxy / dy)), 0, ny - 1);
		const int k0 = std::clamp(static_cast<int>(std::floor(bminz / dz)), 0, nz - 1);
		const int k1 = std::clamp(static_cast<int>(std::floor(bmaxz / dz)), 0, nz - 1);
		for (int k = k0; k <= k1; ++k) {
			for (int j = j0; j <= j1; ++j) {
				for (int i = i0; i <= i1; ++i) {
					active[static_cast<size_t>(cellIndex(i, j, k, nx, ny))] = 1;
				}
			}
		}
	}
}

void dilateActiveCells(std::vector<uint8_t>& active, int nx, int ny, int nz, int layers) {
	if (layers <= 0) {
		return;
	}
	std::vector<uint8_t> tmp(static_cast<size_t>(nx * ny * nz));
	for (int t = 0; t < layers; ++t) {
		tmp = active;
		for (int k = 0; k < nz; ++k) {
			for (int j = 0; j < ny; ++j) {
				for (int i = 0; i < nx; ++i) {
					const size_t id = static_cast<size_t>(cellIndex(i, j, k, nx, ny));
					if (!active[id]) {
						continue;
					}
					for (int dk = -1; dk <= 1; ++dk) {
						for (int dj = -1; dj <= 1; ++dj) {
							for (int di = -1; di <= 1; ++di) {
								const int ni = i + di;
								const int nj = j + dj;
								const int nk = k + dk;
								if (ni < 0 || ni >= nx || nj < 0 || nj >= ny || nk < 0 ||
									nk >= nz) {
									continue;
								}
								tmp[static_cast<size_t>(cellIndex(ni, nj, nk, nx, ny))] = 1;
							}
						}
					}
				}
			}
		}
		active.swap(tmp);
	}
}

struct Dsu {
	std::vector<int> parent;
	explicit Dsu(int n) : parent(static_cast<size_t>(n)) {
		std::iota(parent.begin(), parent.end(), 0);
	}
	int find(int x) {
		return parent[static_cast<size_t>(x)] == x
				   ? x
				   : (parent[static_cast<size_t>(x)] = find(parent[static_cast<size_t>(x)]));
	}
	void unite(int a, int b) {
		a = find(a);
		b = find(b);
		if (a != b) {
			parent[static_cast<size_t>(b)] = a;
		}
	}
};

double signedDistanceAt(const std::array<double, 3>& q,
						const TriMeshAABBTree& tree,
						const std::vector<std::array<double, 3>>& extVerts,
						const std::vector<TriMeshFace>& extFaces) {
	const auto hit = tree.closest_point(q);
	if (!hit) {
		return std::numeric_limits<double>::quiet_NaN();
	}
	const std::array<double, 3>& c = hit->closest;
	Vec3d diff{static_cast<DefaultTriMesh::Scalar>(q[0] - c[0]),
			   static_cast<DefaultTriMesh::Scalar>(q[1] - c[1]),
			   static_cast<DefaultTriMesh::Scalar>(q[2] - c[2])};
	const double dist = std::sqrt(std::max(0.0, hit->squared_distance));
	const TriMeshFace& tf = extFaces[hit->primitive_index];
	const auto& pa = extVerts[tf[0]];
	const auto& pb = extVerts[tf[1]];
	const auto& pc = extVerts[tf[2]];
	Vec3d va{static_cast<DefaultTriMesh::Scalar>(pb[0] - pa[0]),
			 static_cast<DefaultTriMesh::Scalar>(pb[1] - pa[1]),
			 static_cast<DefaultTriMesh::Scalar>(pb[2] - pa[2])};
	Vec3d vb{static_cast<DefaultTriMesh::Scalar>(pc[0] - pa[0]),
			 static_cast<DefaultTriMesh::Scalar>(pc[1] - pa[1]),
			 static_cast<DefaultTriMesh::Scalar>(pc[2] - pa[2])};
	Vec3d n = va % vb;
	const double nl = static_cast<double>(n.norm());
	if (nl < 1e-30) {
		return 0.0;
	}
	n /= static_cast<DefaultTriMesh::Scalar>(nl);
	const double sgn = static_cast<double>(diff | n) >= 0.0 ? 1.0 : -1.0;
	return sgn * dist;
}

} // namespace

PeriodicTriMesh::PeriodicTriMesh() {
	halfPeriod_ = {1.0, 1.0, 1.0};
}

void PeriodicTriMesh::setHalfPeriod(const Vec3d& halfPeriod) {
	halfPeriod_ = halfPeriod;
}

PeriodicTriMesh::Vec3d PeriodicTriMesh::halfPeriod() const {
	return halfPeriod_;
}

PeriodicTriMesh::Vec3d PeriodicTriMesh::wrapVector(const Vec3d& v) const {
	Vec3d out = v;
	for (int i = 0; i < 3; ++i) {
		const double hp = halfPeriod_[i];
		const double step = 2.0 * hp;
		if (!(step > 0.0)) {
			continue;
		}

		const double shifted = static_cast<double>(v[i]) + hp;
		const double n = std::floor(shifted / step);
		out[i] = v[i] - n * step;
	}
	return out;
}

PeriodicTriMesh::Vec3d PeriodicTriMesh::shift2origin(const Vec3d& p) const {
	return wrapVector(p);
}

void PeriodicTriMesh::shift2originInPlace() {
	for (auto v_it = this->vertices_begin(); v_it != this->vertices_end(); ++v_it) {
		const VertexHandle vh = *v_it;
		const Vec3d p = this->point(vh);
		this->set_point(vh, shift2origin(p));
	}
}

bool PeriodicTriMesh::isPeriodicEdge(VertexHandle center, VertexHandle neighbor) const {
	const Vec3d p0 = this->point(center);
	const Vec3d p1 = this->point(neighbor);
	const Vec3d dp_raw = p1 - p0;
	const Vec3d dp_wrapped = wrapVector(dp_raw);

	const double eps = 1e-12;
	const double len2_raw = dp_raw.sqrnorm();
	const double len2_wrapped = dp_wrapped.sqrnorm();
	return (len2_wrapped + eps) < len2_raw;
}

bool PeriodicTriMesh::isPeriodicEdge(EdgeHandle eh) const {
	const HalfedgeHandle heh0 = this->halfedge_handle(eh, 0);
	const VertexHandle v0 = this->from_vertex_handle(heh0);
	const VertexHandle v1 = this->to_vertex_handle(heh0);
	return isPeriodicEdge(v0, v1);
}

std::tuple<std::vector<PeriodicTriMesh::HalfedgeHandle>, std::vector<PeriodicTriMesh::Vec3d>>
PeriodicTriMesh::e1ring(VertexHandle center) const {
	std::vector<HalfedgeHandle> oh;
	std::vector<Vec3d> ohvec;
	oh.reserve(8);
	ohvec.reserve(8);

	const Vec3d p0 = this->point(center);
	for (auto voh_it = this->cvoh_iter(center); voh_it.is_valid(); ++voh_it) {
		const HalfedgeHandle heh = *voh_it;
		const VertexHandle v1 = this->to_vertex_handle(heh);
		const Vec3d p1 = this->point(v1);

		const Vec3d dp_raw = p1 - p0;
		const Vec3d dp_wrapped = wrapVector(dp_raw);

		oh.push_back(heh);
		ohvec.push_back(dp_wrapped);
	}

	return std::make_tuple(std::move(oh), std::move(ohvec));
}

void PeriodicTriMesh::mergePeriodEdges(const double shortEdgeTol) {
	this->request_vertex_status();
	this->request_edge_status();
	this->request_halfedge_status();
	this->request_face_status();
	const double tol = std::max(shortEdgeTol, 1e-15);
	std::vector<EdgeHandle> edges;
	edges.reserve(n_edges());
	for (auto e_it = this->edges_begin(); e_it != this->edges_end(); ++e_it) {
		const EdgeHandle eh = *e_it;
		if (!this->is_boundary(eh)) {
			continue;
		}
		const HalfedgeHandle he = this->halfedge_handle(eh, 0);
		const Vec3d a = this->point(this->to_vertex_handle(he));
		const Vec3d b = this->point(this->from_vertex_handle(he));
		if ((a - b).norm() < tol) {
			edges.push_back(eh);
		}
	}
	for (const EdgeHandle& eh : edges) {
		if (!eh.is_valid()) {
			continue;
		}
		const HalfedgeHandle he = this->halfedge_handle(eh, 0);
		if (!this->is_collapse_ok(he)) {
			continue;
		}
		this->collapse(he);
	}
	this->garbage_collection();
}

// ──────────────────────────────────────────────────────────────────────────
// mergePeriodBoundary helpers
// ──────────────────────────────────────────────────────────────────────────

namespace {

using CgalK = CGAL::Simple_cartesian<double>;
using CgalPoint = CgalK::Point_3;
using CgalSegment = CgalK::Segment_3;

// CGAL AABB segment primitive: stores an iterator pointing to a vector element
using SegmentWithHE = std::pair<CgalSegment, std::size_t>; // segment + halfedge index
using SegVec = std::vector<SegmentWithHE>;
using SegIter = SegVec::iterator;

struct SegPrimitive {
	using Id = SegIter;
	using Point = CgalPoint;
	using Datum = CgalSegment;
	Id m_it;
	SegPrimitive() = default;
	SegPrimitive(SegIter it) : m_it(it) {}
	Id id() const { return m_it; }
	Datum datum() const { return m_it->first; }
	Point reference_point() const { return m_it->first.source(); }
};

using SegAABBTraits = CGAL::AABB_traits<CgalK, SegPrimitive>;
using SegAABBTree = CGAL::AABB_tree<SegAABBTraits>;

CgalPoint toCgal(const PeriodicTriMesh::Vec3d& p) {
	return CgalPoint(
		static_cast<double>(p[0]), static_cast<double>(p[1]), static_cast<double>(p[2]));
}

// Determine whether p is near an endpoint of the segment
int isEndpoint(const CgalPoint& p,
			   const PeriodicTriMesh::Vec3d& v0,
			   const PeriodicTriMesh::Vec3d& v1,
			   double eps) {
	double d0 = CGAL::squared_distance(p, toCgal(v0));
	double d1 = CGAL::squared_distance(p, toCgal(v1));
	if (d0 < eps * eps)
		return 0;
	if (d1 < eps * eps)
		return 1;
	return -1;
}

// Simple PeriodicGridIndex: used for vertex deduplication (periodic wrapping)
struct PeriodicVertexGrid {
	double invH;
	std::array<double, 3> origin;
	std::array<int, 3> ncell;
	// key = wrapped (ix,iy,iz), value = vertex id
	struct I3Hash {
		std::size_t operator()(const std::array<int, 3>& k) const noexcept {
			std::size_t h = static_cast<std::size_t>(k[0]) * 73856093u;
			h ^= static_cast<std::size_t>(k[1]) * 19349663u + 0x9e3779b9u + (h << 6) + (h >> 2);
			h ^= static_cast<std::size_t>(k[2]) * 83492791u + 0x9e3779b9u + (h << 6) + (h >> 2);
			return h;
		}
	};
	std::unordered_map<std::array<int, 3>, int, I3Hash> lattice;
	std::vector<std::array<double, 3>> points;
	int counter = 0;

	PeriodicVertexGrid(const std::array<double, 3>& orig,
					   const std::array<double, 3>& diag,
					   double eps)
		: origin(orig) {
		invH = 1.0 / eps;
		for (int i = 0; i < 3; ++i)
			ncell[i] = std::max(1, static_cast<int>(diag[i] * invH));
	}

	std::array<int, 3> raster(double px, double py, double pz) const {
		const double p[3] = {px, py, pz};
		std::array<int, 3> idx{};
		for (int i = 0; i < 3; ++i) {
			int raw = static_cast<int>(std::round((p[i] - origin[i]) * invH));
			idx[i] = ((raw % ncell[i]) + ncell[i]) % ncell[i];
		}
		return idx;
	}

	std::array<double, 3> deraster(const std::array<int, 3>& idx) const {
		std::array<double, 3> p{};
		for (int i = 0; i < 3; ++i)
			p[i] = static_cast<double>(idx[i]) / invH + origin[i];
		return p;
	}

	int insert(double px, double py, double pz) {
		auto idx = raster(px, py, pz);
		auto it = lattice.find(idx);
		if (it != lattice.end())
			return it->second;
		// Write to the 27-neighborhood
		auto pt = deraster(idx);
		points.push_back(pt);
		int id = counter++;
		for (int dz = -1; dz <= 1; ++dz) {
			for (int dy = -1; dy <= 1; ++dy) {
				for (int dx = -1; dx <= 1; ++dx) {
					std::array<int, 3> near = {idx[0] + dx, idx[1] + dy, idx[2] + dz};
					for (int i = 0; i < 3; ++i)
						near[i] = ((near[i] % ncell[i]) + ncell[i]) % ncell[i];
					lattice[near] = id;
				}
			}
		}
		return id;
	}

	int query(double px, double py, double pz) const {
		auto idx = raster(px, py, pz);
		auto it = lattice.find(idx);
		return it != lattice.end() ? it->second : -1;
	}
};

// ── collapseShortBoundaryEdges ──
// Step 2 helper: collapse short boundary edges, keeping the min-side vertex.
// After collapse, garbage_collection is called and vtag is re-classified.
static void collapseShortBoundaryEdges(
	PeriodicTriMesh& mesh,
	double shortEdgeTol,
	std::unordered_map<int, BoundaryFlag>& vtag,
	const std::function<std::unordered_map<int, BoundaryFlag>()>& classifyVertices) {

	using VertexHandle = PeriodicTriMesh::VertexHandle;
	using EdgeHandle = PeriodicTriMesh::EdgeHandle;
	using HalfedgeHandle = PeriodicTriMesh::HalfedgeHandle;

	std::vector<EdgeHandle> edgesToCollapse;
	for (auto e_it = mesh.edges_begin(); e_it != mesh.edges_end(); ++e_it) {
		const EdgeHandle eh = *e_it;
		if (!eh.is_valid())
			continue;
		const HalfedgeHandle he = mesh.halfedge_handle(eh, 0);
		const VertexHandle vFrom = mesh.from_vertex_handle(he);
		const VertexHandle vTo = mesh.to_vertex_handle(he);
		const Vec3d& pa = mesh.point(vFrom);
		const Vec3d& pb = mesh.point(vTo);
		if ((pa - pb).norm() < shortEdgeTol) {
			auto itFrom = vtag.find(vFrom.idx());
			auto itTo = vtag.find(vTo.idx());
			if (itFrom != vtag.end() || itTo != vtag.end()) {
				edgesToCollapse.push_back(eh);
			}
		}
	}
	for (const EdgeHandle& eh : edgesToCollapse) {
		if (!eh.is_valid() || eh.idx() < 0 || static_cast<std::size_t>(eh.idx()) >= mesh.n_edges())
			continue;
		if (mesh.status(eh).deleted())
			continue;
		const HalfedgeHandle he = mesh.halfedge_handle(eh, 0);
		const VertexHandle vFrom = mesh.from_vertex_handle(he);
		const VertexHandle vTo = mesh.to_vertex_handle(he);
		auto itFrom = vtag.find(vFrom.idx());
		auto itTo = vtag.find(vTo.idx());
		int flagFrom = (itFrom != vtag.end()) ? itFrom->second.getMask() : 0;
		int flagTo = (itTo != vtag.end()) ? itTo->second.getMask() : 0;
		int common = flagFrom & flagTo;
		// Collapse direction: keep the vertex with more flags (prefer min side)
		if ((~common) & flagFrom) {
			// from has flags that to doesn't -> collapse he.opp (keep from)
			const HalfedgeHandle oppHe = mesh.halfedge_handle(eh, 1);
			if (mesh.is_collapse_ok(oppHe))
				mesh.collapse(oppHe);
		} else {
			if (mesh.is_collapse_ok(he))
				mesh.collapse(he);
		}
	}
	mesh.garbage_collection();
	vtag = classifyVertices(); // re-classify
}

// ── weldAndRebuildMesh ──
// Step 4 helper: vertex welding (periodic deduplication) + mesh rebuild.
// Extracts V/F, deduplicates with periodic grid hash, rebuilds face table
// (skipping degenerate), unifies orientation via BFS, and rebuilds OpenMesh.
static void weldAndRebuildMesh(PeriodicTriMesh& mesh,
							   double weldTol,
							   double Lx,
							   double Ly,
							   double Lz,
							   const double domMin[3]) {

	using VertexHandle = PeriodicTriMesh::VertexHandle;
	using HalfedgeHandle = PeriodicTriMesh::HalfedgeHandle;
	using Vec3d = PeriodicTriMesh::Vec3d;

	// Extract V, F
	const std::size_t nv = mesh.n_vertices();
	const std::size_t nf = mesh.n_faces();
	std::vector<std::array<double, 3>> verts(nv);
	for (auto v_it = mesh.vertices_begin(); v_it != mesh.vertices_end(); ++v_it) {
		const Vec3d& p = mesh.point(*v_it);
		verts[static_cast<std::size_t>((*v_it).idx())] = {
			static_cast<double>(p[0]), static_cast<double>(p[1]), static_cast<double>(p[2])};
	}
	std::vector<std::array<int, 3>> faces;
	faces.reserve(nf);
	for (auto f_it = mesh.faces_begin(); f_it != mesh.faces_end(); ++f_it) {
		auto fv = mesh.cfv_iter(*f_it);
		int a = (*fv).idx();
		++fv;
		int b = (*fv).idx();
		++fv;
		int c = (*fv).idx();
		faces.push_back({a, b, c});
	}

	// Periodic deduplication: wrap to [0, L) + quantized hashing
	const double res = weldTol;
	struct GridKey {
		int64_t x, y, z;
		bool operator==(const GridKey& o) const { return x == o.x && y == o.y && z == o.z; }
	};
	struct GridHash {
		std::size_t operator()(const GridKey& k) const {
			return std::hash<int64_t>()(k.x * 1000003LL + k.y) * 1000003LL + k.z;
		}
	};

	std::unordered_map<GridKey, int, GridHash> gridMap;
	std::vector<std::array<double, 3>> uniqueVerts;
	std::vector<int> vmap(nv);

	for (std::size_t i = 0; i < nv; ++i) {
		// wrap to [0, L)
		double wx = verts[i][0] - domMin[0];
		double wy = verts[i][1] - domMin[1];
		double wz = verts[i][2] - domMin[2];
		while (wx < -1e-8)
			wx += Lx;
		while (wx >= Lx - 1e-8)
			wx -= Lx;
		if (wx < 0)
			wx = 0;
		while (wy < -1e-8)
			wy += Ly;
		while (wy >= Ly - 1e-8)
			wy -= Ly;
		if (wy < 0)
			wy = 0;
		while (wz < -1e-8)
			wz += Lz;
		while (wz >= Lz - 1e-8)
			wz -= Lz;
		if (wz < 0)
			wz = 0;

		GridKey key{static_cast<int64_t>(std::round(wx / res)),
					static_cast<int64_t>(std::round(wy / res)),
					static_cast<int64_t>(std::round(wz / res))};
		auto it = gridMap.find(key);
		if (it != gridMap.end()) {
			vmap[i] = it->second;
		} else {
			int idx = static_cast<int>(uniqueVerts.size());
			gridMap[key] = idx;
			uniqueVerts.push_back({wx + domMin[0], wy + domMin[1], wz + domMin[2]});
			vmap[i] = idx;
		}
	}

	// Rebuild face list, skipping degenerate triangles
	std::vector<std::array<int, 3>> newFaces;
	newFaces.reserve(nf);
	for (const auto& f : faces) {
		int fa = vmap[static_cast<std::size_t>(f[0])];
		int fb = vmap[static_cast<std::size_t>(f[1])];
		int fc = vmap[static_cast<std::size_t>(f[2])];
		if (fa == fb || fb == fc || fa == fc)
			continue;
		newFaces.push_back({fa, fb, fc});
	}

	// Build triangle-triangle adjacency and unify orientation via BFS
	const int nNewF = static_cast<int>(newFaces.size());
	const int nNewV = static_cast<int>(uniqueVerts.size());

	// Build edge-to-face adjacency
	struct EdgeKey {
		int lo, hi;
		bool operator==(const EdgeKey& o) const { return lo == o.lo && hi == o.hi; }
	};
	struct EdgeKeyHash {
		std::size_t operator()(const EdgeKey& e) const noexcept {
			return std::hash<long long>()(static_cast<long long>(e.lo) * 1000003LL + e.hi);
		}
	};

	// edge -> list of (faceIdx, localEdgeIdx)
	std::unordered_map<EdgeKey, std::vector<std::pair<int, int>>, EdgeKeyHash> edge2faces;
	for (int fi = 0; fi < nNewF; ++fi) {
		for (int ei = 0; ei < 3; ++ei) {
			int v0 = newFaces[static_cast<std::size_t>(fi)][static_cast<std::size_t>(ei)];
			int v1 = newFaces[static_cast<std::size_t>(fi)][static_cast<std::size_t>((ei + 1) % 3)];
			EdgeKey ek{std::min(v0, v1), std::max(v0, v1)};
			edge2faces[ek].push_back({fi, ei});
		}
	}

	// Build face-face adjacency (FF)
	std::vector<std::array<int, 3>> FF(static_cast<std::size_t>(nNewF), {-1, -1, -1});
	for (auto& [ek, flist] : edge2faces) {
		if (flist.size() == 2) {
			FF[static_cast<std::size_t>(flist[0].first)]
			  [static_cast<std::size_t>(flist[0].second)] = flist[1].first;
			FF[static_cast<std::size_t>(flist[1].first)]
			  [static_cast<std::size_t>(flist[1].second)] = flist[0].first;
		}
	}

	// BFS to unify face orientations
	std::vector<bool> visited(static_cast<std::size_t>(nNewF), false);
	std::queue<int> bfsQueue;
	// There may be multiple connected components
	for (int startF = 0; startF < nNewF; ++startF) {
		if (visited[static_cast<std::size_t>(startF)])
			continue;
		visited[static_cast<std::size_t>(startF)] = true;
		bfsQueue.push(startF);
		while (!bfsQueue.empty()) {
			const int fi = bfsQueue.front();
			bfsQueue.pop();
			const auto& srcF = newFaces[static_cast<std::size_t>(fi)];
			for (int ei = 0; ei < 3; ++ei) {
				int nb = FF[static_cast<std::size_t>(fi)][static_cast<std::size_t>(ei)];
				if (nb < 0 || visited[static_cast<std::size_t>(nb)])
					continue;
				visited[static_cast<std::size_t>(nb)] = true;
				// Check whether the shared edge has consistent orientation in both triangles
				int sv0 = srcF[static_cast<std::size_t>(ei)];
				int sv1 = srcF[static_cast<std::size_t>((ei + 1) % 3)];
				auto& dstF = newFaces[static_cast<std::size_t>(nb)];
				// Find the shared edge order in dst
				int ci0 = -1, ci1 = -1;
				for (int k = 0; k < 3; ++k) {
					if (dstF[static_cast<std::size_t>(k)] == sv0)
						ci0 = k;
					if (dstF[static_cast<std::size_t>(k)] == sv1)
						ci1 = k;
				}
				if (ci0 >= 0 && ci1 >= 0) {
					// In correct orientation, src's (v0,v1) edge should appear as (v1,v0) in dst
					// i.e., sv1 should precede sv0 in dst (mod 3), meaning (ci1+1)%3 == ci0
					if ((ci0 + 1) % 3 == ci1) {
						// Same direction -> need to flip dst
						std::swap(dstF[static_cast<std::size_t>(ci0)],
								  dstF[static_cast<std::size_t>(ci1)]);
						// Update FF (local edge indices change after flip, but BFS only uses
						// visited, no precise update needed)
					}
				}
				bfsQueue.push(nb);
			}
		}
	}

	// Fill small boundary holes (open edge loops from projection/weld)
	{
		auto packEdge = [](int u, int v) -> std::uint64_t {
			const int lo = std::min(u, v), hi = std::max(u, v);
			return (static_cast<std::uint64_t>(lo) << 32) | static_cast<std::uint64_t>(hi);
		};
		std::unordered_set<std::uint64_t> openEdges;
		std::unordered_map<int, std::vector<int>> openAdj;
		for (const auto& [ek, flist] : edge2faces) {
			if (flist.size() != 1)
				continue;
			openEdges.insert(packEdge(ek.lo, ek.hi));
			openAdj[ek.lo].push_back(ek.hi);
			openAdj[ek.hi].push_back(ek.lo);
		}

		auto orientHoleTri = [&](int va, int vb, int vc) -> std::array<int, 3> {
			const EdgeKey ek{std::min(va, vb), std::max(va, vb)};
			const auto it = edge2faces.find(ek);
			if (it != edge2faces.end() && it->second.size() == 1) {
				const int fi = it->second[0].first;
				const int ei = it->second[0].second;
				const int f0 = newFaces[static_cast<std::size_t>(fi)][static_cast<std::size_t>(ei)];
				const int f1 =
					newFaces[static_cast<std::size_t>(fi)][static_cast<std::size_t>((ei + 1) % 3)];
				if (f0 == va && f1 == vb)
					return {vb, va, vc};
				if (f0 == vb && f1 == va)
					return {va, vb, vc};
			}
			return {va, vb, vc};
		};

		std::unordered_set<std::uint64_t> usedEdges;
		for (const std::uint64_t ekey : openEdges) {
			if (usedEdges.count(ekey))
				continue;
			const int start = static_cast<int>(ekey >> 32);
			const int second = static_cast<int>(ekey & 0xFFFFFFFFu);
			std::vector<int> cycle;
			int cur = start, prev = -1;
			do {
				cycle.push_back(cur);
				int next = -1;
				for (int nb : openAdj[cur]) {
					if (nb == prev)
						continue;
					if (!openEdges.count(packEdge(cur, nb)))
						continue;
					next = nb;
					break;
				}
				if (next < 0)
					break;
				prev = cur;
				cur = next;
			} while (cur != start && static_cast<int>(cycle.size()) < 8);
			if (cur != start || static_cast<int>(cycle.size()) < 3)
				continue;

			bool loopOk = true;
			for (std::size_t i = 0; i < cycle.size(); ++i) {
				if (!openEdges.count(packEdge(cycle[i], cycle[(i + 1) % cycle.size()]))) {
					loopOk = false;
					break;
				}
			}
			if (!loopOk)
				continue;

			const int n = static_cast<int>(cycle.size());
			for (int i = 1; i + 1 < n; ++i) {
				std::array<int, 3> tri;
				if (i == 1) {
					tri = orientHoleTri(cycle[0], cycle[1], cycle[2]);
				} else {
					tri = orientHoleTri(cycle[static_cast<std::size_t>(i)],
										cycle[static_cast<std::size_t>(i + 1)],
										cycle[0]);
				}
				if (tri[0] == tri[1] || tri[1] == tri[2] || tri[0] == tri[2])
					continue;
				newFaces.push_back(tri);
			}
			for (std::size_t i = 0; i < cycle.size(); ++i) {
				usedEdges.insert(packEdge(cycle[i], cycle[(i + 1) % cycle.size()]));
			}
		}
	}

	// Rebuild OpenMesh
	mesh.clear();
	std::vector<VertexHandle> newVH;
	newVH.reserve(static_cast<std::size_t>(nNewV));
	for (int i = 0; i < nNewV; ++i) {
		const auto& pt = uniqueVerts[static_cast<std::size_t>(i)];
		newVH.push_back(mesh.add_vertex(Vec3d(static_cast<DefaultTriMesh::Scalar>(pt[0]),
											  static_cast<DefaultTriMesh::Scalar>(pt[1]),
											  static_cast<DefaultTriMesh::Scalar>(pt[2]))));
	}
	for (const auto& f : newFaces) {
		auto fh = mesh.add_face(newVH[static_cast<std::size_t>(f[0])],
								newVH[static_cast<std::size_t>(f[1])],
								newVH[static_cast<std::size_t>(f[2])]);
		if (!fh.is_valid()) {
			mesh.add_face(newVH[static_cast<std::size_t>(f[0])],
						  newVH[static_cast<std::size_t>(f[2])],
						  newVH[static_cast<std::size_t>(f[1])]);
		}
	}
	mesh.garbage_collection();

	int bndE2 = 0;
	for (auto e_it = mesh.edges_begin(); e_it != mesh.edges_end(); ++e_it)
		if (mesh.is_boundary(*e_it))
			++bndE2;
	std::cerr << "[merge] after weld: nv=" << mesh.n_vertices() << " nf=" << mesh.n_faces()
			  << " bnd=" << bndE2 << "\n";
}

} // namespace

void PeriodicTriMesh::mergePeriodBoundary(const MergeBoundaryOptions& options) {
	// OpenMesh requires status attributes to perform collapse/split/delete/garbage_collection
	this->request_vertex_status();
	this->request_edge_status();
	this->request_halfedge_status();
	this->request_face_status();

	const double Lx = 2.0 * static_cast<double>(halfPeriod_[0]);
	const double Ly = 2.0 * static_cast<double>(halfPeriod_[1]);
	const double Lz = 2.0 * static_cast<double>(halfPeriod_[2]);
	const double maxL = std::max({Lx, Ly, Lz});

	// Projection tolerance: the larger of the user-specified value and 1% of the mesh scale
	const double projTol = std::max(options.projectionTol, maxL * 0.02);
	// Vertex welding tolerance: the larger of the user value and 0.01% of the mesh scale
	const double weldTol = std::max(options.vertexWeldTol, maxL * 1e-4);
	// Short edge tolerance: the larger of the user value and 0.2% of the mesh scale
	const double shortTol = std::max(options.shortEdgeTol, maxL * 0.002);

	// domain: [0, Lx] x [0, Ly] x [0, Lz]
	const double domMin[3] = {0.0, 0.0, 0.0};
	const double domMax[3] = {Lx, Ly, Lz};

	// Boundary tolerance: strict face snap so slightly-inset verts are not tagged as period faces
	const double borderTol = 1e-5;

	// ── Step 1: classify all boundary vertices ──
	auto classifyVertices = [&]() {
		std::unordered_map<int, BoundaryFlag> vtag;
		for (auto v_it = this->vertices_begin(); v_it != this->vertices_end(); ++v_it) {
			if (!this->is_boundary(*v_it))
				continue;
			const Vec3d& p = this->point(*v_it);
			BoundaryFlag bf;
			bf.classify(static_cast<double>(p[0]),
						static_cast<double>(p[1]),
						static_cast<double>(p[2]),
						domMin[0],
						domMax[0],
						domMin[1],
						domMax[1],
						domMin[2],
						domMax[2],
						borderTol);
			if (bf.isBoundary())
				vtag[(*v_it).idx()] = bf;
		}
		return vtag;
	};

	auto vtag = classifyVertices();

	// ── Step 2: collapse short boundary edges (keep the min-side vertex) ──
	collapseShortBoundaryEdges(*this, shortTol, vtag, classifyVertices);

	// ── Step 3: project max->min boundary using CGAL AABB segment tree ──
	auto buildSegTree = [&](bool buildMin) {
		// Collect halfedges on the target boundary face: a halfedge is a candidate if both
		// endpoints are boundary vertices
		std::vector<std::pair<HalfedgeHandle, int>> bndHEs; // halfedge + periodic boundary mask
		for (auto e_it = this->edges_begin(); e_it != this->edges_end(); ++e_it) {
			if (!this->is_boundary(*e_it))
				continue;
			for (int side = 0; side < 2; ++side) {
				const HalfedgeHandle he = this->halfedge_handle(*e_it, side);
				if (!this->is_boundary(he))
					continue;
				const VertexHandle vf = this->from_vertex_handle(he);
				const VertexHandle vt = this->to_vertex_handle(he);
				auto itF = vtag.find(vf.idx());
				auto itT = vtag.find(vt.idx());
				// Both endpoints must be boundary vertices on the same face
				if (itF == vtag.end() || itT == vtag.end())
					continue;
				int mask = itF->second.getMask() & itT->second.getMask();
				if (!mask)
					continue;
				bool isMinEdge = (mask & BoundaryFlag::kMinMask) != 0;
				bool isMaxEdge = (mask & BoundaryFlag::kMaxMask) != 0;
				if (buildMin ? isMinEdge : isMaxEdge) {
					bndHEs.push_back({he, mask});
				}
			}
		}
		return bndHEs;
	};

	// Store the list of post-split halfedges corresponding to each original halfedge
	std::unordered_map<int, std::vector<HalfedgeHandle>> splittedHE;

	auto projectAndSplit = [&](bool projectMaxToMin) {
		auto bndHEs = buildSegTree(projectMaxToMin); // target edges (min or max side)
		SegVec segments;
		segments.reserve(bndHEs.size());
		for (std::size_t i = 0; i < bndHEs.size(); ++i) {
			const HalfedgeHandle he = bndHEs[i].first;
			const Vec3d& pf = this->point(this->from_vertex_handle(he));
			const Vec3d& pt = this->point(this->to_vertex_handle(he));
			segments.push_back({CgalSegment(toCgal(pf), toCgal(pt)), i});
			splittedHE[static_cast<int>(i)].clear();
			splittedHE[static_cast<int>(i)].push_back(he);
		}
		if (segments.empty())
			return;

		SegAABBTree tree(segments.begin(), segments.end());
		tree.build();

		// Project source endpoints
		for (auto& [vidx, bf] : vtag) {
			const VertexHandle vh(vidx);
			if (vidx < 0 || static_cast<std::size_t>(vidx) >= this->n_vertices())
				continue;
			if (this->status(vh).deleted())
				continue;
			bool isSource = projectMaxToMin ? bf.isMaxBoundary() : bf.isMinBoundary();
			if (!isSource)
				continue;

			Vec3d p = this->point(vh);
			// Translate to the target side (integer period only)
			Vec3d translated = p;
			for (int ax = 0; ax < 3; ++ax) {
				if (projectMaxToMin) {
					if (bf.isMaxBoundary(ax)) {
						const auto L = static_cast<DefaultTriMesh::Scalar>(domMax[ax] - domMin[ax]);
						translated[ax] -= L;
					}
				} else {
					if (bf.isMinBoundary(ax)) {
						const auto L = static_cast<DefaultTriMesh::Scalar>(domMax[ax] - domMin[ax]);
						translated[ax] += L;
					}
				}
			}

			auto result = tree.closest_point_and_primitive(toCgal(translated));
			CgalPoint closest = result.first;
			double sqDist = CGAL::squared_distance(closest, toCgal(translated));
			if (sqDist > projTol * projTol) {
				continue;
			}
			std::size_t primIdx = result.second->second;
			auto& heSplits = splittedHE[static_cast<int>(primIdx)];

			// Find the sub-halfedge (after splits) that contains the closest point
			for (std::size_t k = 0; k < heSplits.size(); ++k) {
				const HalfedgeHandle he = heSplits[k];
				if (!he.is_valid())
					continue;
				const EdgeHandle eeh = this->edge_handle(he);
				if (!eeh.is_valid() || this->status(eeh).deleted())
					continue;
				const Vec3d& hFrom = this->point(this->from_vertex_handle(he));
				const Vec3d& hTo = this->point(this->to_vertex_handle(he));
				// Use 1% of edge length as endpoint detection tolerance
				double heLen = (hTo - hFrom).norm();
				double endEps = std::max(1e-6, heLen * 0.01);
				int endId = isEndpoint(closest, hFrom, hTo, endEps);
				if (endId == 0) {
					// Matched the from-endpoint
					Vec3d trans{};
					for (int ax = 0; ax < 3; ++ax) {
						trans[ax] =
							projectMaxToMin
								? static_cast<DefaultTriMesh::Scalar>(domMax[ax] - domMin[ax])
								: static_cast<DefaultTriMesh::Scalar>(-(domMax[ax] - domMin[ax]));
						if (projectMaxToMin ? !bf.isMaxBoundary(ax) : !bf.isMinBoundary(ax))
							trans[ax] = 0;
					}
					const Vec3d newPos = hFrom + trans;
					this->set_point(vh, newPos);
					break;
				} else if (endId == 1) {
					Vec3d trans{};
					for (int ax = 0; ax < 3; ++ax) {
						trans[ax] =
							projectMaxToMin
								? static_cast<DefaultTriMesh::Scalar>(domMax[ax] - domMin[ax])
								: static_cast<DefaultTriMesh::Scalar>(-(domMax[ax] - domMin[ax]));
						if (projectMaxToMin ? !bf.isMaxBoundary(ax) : !bf.isMinBoundary(ax))
							trans[ax] = 0;
					}
					const Vec3d newPos = hTo + trans;
					this->set_point(vh, newPos);
					break;
				} else {
					// Need to split: insert a new vertex at the projected point on the target edge
					const Vec3d newPt(static_cast<DefaultTriMesh::Scalar>(closest.x()),
									  static_cast<DefaultTriMesh::Scalar>(closest.y()),
									  static_cast<DefaultTriMesh::Scalar>(closest.z()));
					// Check that the projected point is not at an edge endpoint (avoid degenerate
					// split) Only split if the parameter t is in the [0.02, 0.98] range (not too
					// close to endpoints)
					const Vec3d edgeVec = hTo - hFrom;
					const double edgeLen = edgeVec.norm();
					if (edgeLen < 1e-15)
						continue;
					double t = static_cast<double>((newPt - hFrom) | edgeVec) / (edgeLen * edgeLen);
					if (t < 0.02 || t > 0.98)
						continue;

					VertexHandle newVH = this->add_vertex(newPt);
					this->split(this->edge_handle(he), newVH);

					// Set the source vertex position
					Vec3d trans{};
					for (int ax = 0; ax < 3; ++ax) {
						trans[ax] =
							projectMaxToMin
								? static_cast<DefaultTriMesh::Scalar>(domMax[ax] - domMin[ax])
								: static_cast<DefaultTriMesh::Scalar>(-(domMax[ax] - domMin[ax]));
						if (projectMaxToMin ? !bf.isMaxBoundary(ax) : !bf.isMinBoundary(ax))
							trans[ax] = 0;
					}
					const Vec3d newPos = newPt + trans;
					this->set_point(vh, newPos);

					// Record the new halfedges produced by the split
					if (this->point(this->to_vertex_handle(he)) == newPt) {
						heSplits.push_back(this->next_halfedge_handle(he));
					} else {
						heSplits.push_back(this->prev_halfedge_handle(he));
					}
					break;
				}
			}
		}
	};

	// max->min projection
	projectAndSplit(true);
	vtag = classifyVertices();
	splittedHE.clear();

	// min->max projection
	projectAndSplit(false);
	this->garbage_collection();
	vtag = classifyVertices();

	// ── Step 4: vertex merging (periodic deduplication + mesh rebuild) ──
	weldAndRebuildMesh(*this, weldTol, Lx, Ly, Lz, domMin);
}

// ──────────────────────────────────────────────────────────────────────────
// removeNonPeriodicIslands
// Detect and remove connected components that do not span a full period (aligned with minsurf
// deleteisoisland BFS approach)
// ──────────────────────────────────────────────────────────────────────────

// Keep only the largest connected component (by face count)
int PeriodicTriMesh::keepLargestComponent() {
	this->request_vertex_status();
	this->request_edge_status();
	this->request_halfedge_status();
	this->request_face_status();

	const std::size_t nf = this->n_faces();
	if (nf == 0)
		return 0;
	std::vector<int> compId(nf, -1);
	std::vector<int> compSize;
	int nComp = 0;
	for (std::size_t fi = 0; fi < nf; ++fi) {
		if (compId[fi] >= 0)
			continue;
		std::queue<int> q;
		q.push(static_cast<int>(fi));
		compId[fi] = nComp;
		int sz = 0;
		while (!q.empty()) {
			int fid = q.front();
			q.pop();
			++sz;
			auto fh = this->face_handle(fid);
			for (auto fh_it = this->cfh_iter(fh); fh_it.is_valid(); ++fh_it) {
				auto opp = this->opposite_halfedge_handle(*fh_it);
				if (this->is_boundary(opp))
					continue;
				int adj = this->face_handle(opp).idx();
				if (adj >= 0 && static_cast<std::size_t>(adj) < nf &&
					compId[static_cast<std::size_t>(adj)] < 0) {
					compId[static_cast<std::size_t>(adj)] = nComp;
					q.push(adj);
				}
			}
		}
		compSize.push_back(sz);
		++nComp;
	}
	if (nComp <= 1)
		return 0;
	int maxComp =
		static_cast<int>(std::max_element(compSize.begin(), compSize.end()) - compSize.begin());
	for (auto f_it = this->faces_begin(); f_it != this->faces_end(); ++f_it) {
		int cid = compId[static_cast<std::size_t>((*f_it).idx())];
		if (cid != maxComp)
			this->delete_face(*f_it, true);
	}
	this->garbage_collection();
	return nComp - 1;
}

int PeriodicTriMesh::removeNonPeriodicIslands() {
	this->request_vertex_status();
	this->request_edge_status();
	this->request_halfedge_status();
	this->request_face_status();

	const Vec3d hp = halfPeriod_;
	const double fullPeriod[3] = {2.0 * static_cast<double>(hp[0]),
								  2.0 * static_cast<double>(hp[1]),
								  2.0 * static_cast<double>(hp[2])};

	// Find connected components
	const std::size_t nvTotal = this->n_vertices();
	std::vector<int> compId(nvTotal, -1);
	std::vector<VertexHandle> compSeed;
	int nComp = 0;
	for (auto v_it = this->vertices_begin(); v_it != this->vertices_end(); ++v_it) {
		if (compId[static_cast<std::size_t>((*v_it).idx())] >= 0)
			continue;
		std::queue<int> q;
		q.push((*v_it).idx());
		compId[static_cast<std::size_t>((*v_it).idx())] = nComp;
		while (!q.empty()) {
			int vi = q.front();
			q.pop();
			for (auto voh = this->cvoh_iter(VertexHandle(vi)); voh.is_valid(); ++voh) {
				int nb = this->to_vertex_handle(*voh).idx();
				if (compId[static_cast<std::size_t>(nb)] < 0) {
					compId[static_cast<std::size_t>(nb)] = nComp;
					q.push(nb);
				}
			}
		}
		compSeed.push_back(*v_it);
		++nComp;
	}

	if (nComp <= 1)
		return 0;

	// For each connected component, check whether it spans at least one axis period
	std::vector<bool> shouldRemove(static_cast<std::size_t>(nComp), true);
	for (int ci = 0; ci < nComp; ++ci) {
		VertexHandle seed = compSeed[static_cast<std::size_t>(ci)];
		for (int axis = 0; axis < 3; ++axis) {
			// BFS: accumulate periodically-wrapped coordinate differences along the axis
			std::vector<double> vdist(nvTotal, 0.0);
			std::queue<std::pair<int, double>> bfs;
			bfs.push({seed.idx(), 0.0});
			double maxDist = -1;
			while (!bfs.empty() && maxDist < fullPeriod[axis]) {
				auto [vi, dist] = bfs.front();
				bfs.pop();
				if (dist - vdist[static_cast<std::size_t>(vi)] < -1e-6)
					continue;
				vdist[static_cast<std::size_t>(vi)] = dist;
				maxDist = std::max(maxDist, dist);
				for (auto voh = this->cvoh_iter(VertexHandle(vi)); voh.is_valid(); ++voh) {
					int nb = this->to_vertex_handle(*voh).idx();
					Vec3d ev =
						wrapVector(this->point(VertexHandle(nb)) - this->point(VertexHandle(vi)));
					double nbDist = dist + static_cast<double>(ev[axis]);
					if (nbDist > vdist[static_cast<std::size_t>(nb)]) {
						bfs.push({nb, nbDist});
						vdist[static_cast<std::size_t>(nb)] = nbDist;
						maxDist = std::max(maxDist, nbDist);
					}
				}
			}
			if (maxDist >= fullPeriod[axis]) {
				shouldRemove[static_cast<std::size_t>(ci)] = false;
				break;
			}
		}
	}

	// Delete non-spanning components
	int deletedFaces = 0;
	for (auto v_it = this->vertices_begin(); v_it != this->vertices_end(); ++v_it) {
		int ci = compId[static_cast<std::size_t>((*v_it).idx())];
		if (ci >= 0 && shouldRemove[static_cast<std::size_t>(ci)]) {
			for (auto vf = this->cvf_iter(*v_it); vf.is_valid(); ++vf)
				if (!this->status(*vf).deleted()) {
					this->delete_face(*vf, false);
					++deletedFaces;
				}
		}
	}

	if (deletedFaces > 0) {
		this->garbage_collection();
		std::cerr << "[removeNonPeriodicIslands] removed " << deletedFaces << " faces from "
				  << std::count(shouldRemove.begin(), shouldRemove.end(), true)
				  << " non-periodic components\n";
	}

	return deletedFaces;
}

// ──────────────────────────────────────────────────────────────────────────
// periodShift
// ──────────────────────────────────────────────────────────────────────────

void PeriodicTriMesh::periodShift() {
	for (auto v_it = this->vertices_begin(); v_it != this->vertices_end(); ++v_it) {
		Vec3d p = this->point(*v_it);
		for (int i = 0; i < 3; ++i) {
			const double period = 2.0 * static_cast<double>(halfPeriod_[i]);
			double pi = static_cast<double>(p[i]);
			if (pi < -1e-5)
				p[i] += static_cast<DefaultTriMesh::Scalar>(period);
			else if (pi > period + 1e-5)
				p[i] -= static_cast<DefaultTriMesh::Scalar>(period);
		}
		this->set_point(*v_it, p);
	}
}

// ──────────────────────────────────────────────────────────────────────────
// splitUnitCell (aligned with minsurf split_unit_cell)
// Find edges crossing periodic boundaries and split them at boundary faces
// ──────────────────────────────────────────────────────────────────────────

void PeriodicTriMesh::splitUnitCell(bool doSplitEdges) {
	this->request_vertex_status();
	this->request_edge_status();
	this->request_halfedge_status();
	this->request_face_status();

	periodShift();

	const double hp[3] = {static_cast<double>(halfPeriod_[0]),
						  static_cast<double>(halfPeriod_[1]),
						  static_cast<double>(halfPeriod_[2])};
	const double L[3] = {2.0 * hp[0], 2.0 * hp[1], 2.0 * hp[2]};

	// ── Phase 1: split boundary-crossing edges at periodic boundaries ──
	if (doSplitEdges) {
		for (int axis = 0; axis < 3; ++axis) {
			std::vector<EdgeHandle> toSplit;
			for (auto e_it = this->edges_begin(); e_it != this->edges_end(); ++e_it) {
				if (this->status(*e_it).deleted())
					continue;
				HalfedgeHandle he = this->halfedge_handle(*e_it, 0);
				Vec3d p0 = this->point(this->from_vertex_handle(he));
				Vec3d p1 = p0 + wrapVector(this->point(this->to_vertex_handle(he)) - p0);

				double a0 = static_cast<double>(p0[axis]);
				double a1 = static_cast<double>(p1[axis]);

				if (std::abs(a0) < 1e-5 || std::abs(a0 - L[axis]) < 1e-5)
					continue;
				if (std::abs(a1) < 1e-5 || std::abs(a1 - L[axis]) < 1e-5)
					continue;
				if (a1 < -1e-5 || a1 > L[axis] + 1e-5)
					toSplit.push_back(*e_it);
			}

			for (auto eh : toSplit) {
				if (!eh.is_valid() || this->status(eh).deleted())
					continue;
				HalfedgeHandle he = this->halfedge_handle(eh, 0);
				VertexHandle vFrom = this->from_vertex_handle(he);
				VertexHandle vTo = this->to_vertex_handle(he);
				Vec3d p0 = this->point(vFrom);
				Vec3d p1 = p0 + wrapVector(this->point(vTo) - p0);
				double a0 = static_cast<double>(p0[axis]);
				double a1 = static_cast<double>(p1[axis]);
				double cut = (a1 > L[axis]) ? L[axis] : 0.0;
				double denom = a1 - a0;
				if (std::abs(denom) < 1e-15)
					continue;
				double t = (cut - a0) / denom;
				if (t <= 1e-6) {
					// split point nearly coincides with the from endpoint -> snap from to boundary
					auto pp = this->point(vFrom);
					pp[axis] = static_cast<DefaultTriMesh::Scalar>(cut);
					this->set_point(vFrom, pp);
				} else if (t >= 1.0 - 1e-6) {
					// split point nearly coincides with the to endpoint -> snap to to boundary
					auto pp = this->point(vTo);
					pp[axis] = static_cast<DefaultTriMesh::Scalar>(cut);
					this->set_point(vTo, pp);
				} else {
					Vec3d c;
					for (int k = 0; k < 3; ++k)
						c[k] = static_cast<DefaultTriMesh::Scalar>((1.0 - t) *
																	   static_cast<double>(p0[k]) +
																   t * static_cast<double>(p1[k]));
					this->split(eh, c);
				}
			}
			periodShift();
		}
		this->garbage_collection();
	}

	// ── Phase 2: dupPeriodFaces (aligned with minsurf) ──
	// Unfold periodic-crossing faces by edge, duplicate offset vertices, so all faces lie within
	// [0, L]
	{
		const std::size_t nv0 = this->n_vertices();
		std::vector<std::array<double, 3>> vpos(nv0);
		for (auto v_it = this->vertices_begin(); v_it != this->vertices_end(); ++v_it) {
			const Vec3d& p = this->point(*v_it);
			vpos[static_cast<std::size_t>((*v_it).idx())] = {
				static_cast<double>(p[0]), static_cast<double>(p[1]), static_cast<double>(p[2])};
		}
		struct F3 {
			int v[3];
		};
		std::vector<F3> flist;
		flist.reserve(this->n_faces());
		for (auto f_it = this->faces_begin(); f_it != this->faces_end(); ++f_it) {
			auto fv = this->cfv_iter(*f_it);
			int a = (*fv).idx();
			++fv;
			int b = (*fv).idx();
			++fv;
			int c = (*fv).idx();
			flist.push_back({a, b, c});
		}

		// Lookup existing vertices by position->index
		std::map<std::array<int, 3>, int> posMap; // quantized coordinates -> vertex index
		auto quantize = [&](double x, double y, double z) -> std::array<int, 3> {
			return {static_cast<int>(std::round(x * 1e5)),
					static_cast<int>(std::round(y * 1e5)),
					static_cast<int>(std::round(z * 1e5))};
		};
		for (std::size_t i = 0; i < nv0; ++i)
			posMap[quantize(vpos[i][0], vpos[i][1], vpos[i][2])] = static_cast<int>(i);

		std::vector<std::array<double, 3>> extraV;

		for (auto& f : flist) {
			double vnew[3][3];
			for (int j = 0; j < 3; ++j)
				for (int k = 0; k < 3; ++k)
					vnew[j][k] =
						vpos[static_cast<std::size_t>(f.v[j])][static_cast<std::size_t>(k)];

			bool hasPeriod = false;
			// Edge-based unfolding (aligned with minsurf dupPeriodFaces)
			for (int j = 0; j < 3; ++j) {
				int j1 = (j + 1) % 3;
				for (int k = 0; k < 3; ++k) {
					double ej = vnew[j1][k] - vnew[j][k];
					if (ej < -hp[k]) {
						vnew[j1][k] += L[k];
						hasPeriod = true;
					} else if (ej > hp[k]) {
						vnew[j][k] += L[k];
						hasPeriod = true;
					}
				}
			}
			// Shift the whole triangle back into [0, L]
			for (int k = 0; k < 3; ++k) {
				double cmax = std::max({vnew[0][k], vnew[1][k], vnew[2][k]});
				double cmin = std::min({vnew[0][k], vnew[1][k], vnew[2][k]});
				if (cmax > L[k] + 1e-5) {
					vnew[0][k] -= L[k];
					vnew[1][k] -= L[k];
					vnew[2][k] -= L[k];
				} else if (cmin < -1e-5) {
					vnew[0][k] += L[k];
					vnew[1][k] += L[k];
					vnew[2][k] += L[k];
				}
			}

			if (hasPeriod) {
				for (int j = 0; j < 3; ++j) {
					auto qk = quantize(vnew[j][0], vnew[j][1], vnew[j][2]);
					if (posMap.count(qk)) {
						f.v[j] = posMap[qk];
					} else {
						int newIdx = static_cast<int>(nv0 + extraV.size());
						posMap[qk] = newIdx;
						extraV.push_back({vnew[j][0], vnew[j][1], vnew[j][2]});
						f.v[j] = newIdx;
					}
				}
			}
		}

		// Rebuild mesh
		this->clear();
		std::vector<VertexHandle> vh;
		vh.reserve(nv0 + extraV.size());
		for (std::size_t i = 0; i < nv0; ++i)
			vh.push_back(this->add_vertex(Vec3d(static_cast<DefaultTriMesh::Scalar>(vpos[i][0]),
												static_cast<DefaultTriMesh::Scalar>(vpos[i][1]),
												static_cast<DefaultTriMesh::Scalar>(vpos[i][2]))));
		for (const auto& ev : extraV)
			vh.push_back(this->add_vertex(Vec3d(static_cast<DefaultTriMesh::Scalar>(ev[0]),
												static_cast<DefaultTriMesh::Scalar>(ev[1]),
												static_cast<DefaultTriMesh::Scalar>(ev[2]))));
		for (const auto& f : flist) {
			auto fh = this->add_face(vh[static_cast<std::size_t>(f.v[0])],
									 vh[static_cast<std::size_t>(f.v[1])],
									 vh[static_cast<std::size_t>(f.v[2])]);
			if (!fh.is_valid())
				this->add_face(vh[static_cast<std::size_t>(f.v[0])],
							   vh[static_cast<std::size_t>(f.v[2])],
							   vh[static_cast<std::size_t>(f.v[1])]);
		}
	}
}

// ──────────────────────────────────────────────────────────────────────────
// saveUnitCell / saveSplitUnitCellWithSingularity
// ──────────────────────────────────────────────────────────────────────────

bool PeriodicTriMesh::saveSplitUnitCellWithSingularity(const std::string& objPath,
													   const std::string& singularityPath,
													   int surgeryType,
													   bool splitEdges) const {
	PeriodicTriMesh mesh = *this;
	std::vector<double> curv = computeSingularityMeasure(mesh, surgeryType);

	mesh.request_vertex_status();
	mesh.request_edge_status();
	mesh.request_halfedge_status();
	mesh.request_face_status();
	mesh.periodShift();

	const double hp[3] = {static_cast<double>(mesh.halfPeriod_[0]),
						  static_cast<double>(mesh.halfPeriod_[1]),
						  static_cast<double>(mesh.halfPeriod_[2])};
	const double L[3] = {2.0 * hp[0], 2.0 * hp[1], 2.0 * hp[2]};

	using VertexHandle = PeriodicTriMesh::VertexHandle;
	using EdgeHandle = PeriodicTriMesh::EdgeHandle;
	using HalfedgeHandle = PeriodicTriMesh::HalfedgeHandle;
	using Vec3d = PeriodicTriMesh::Vec3d;

	if (splitEdges) {
		for (int axis = 0; axis < 3; ++axis) {
			std::vector<EdgeHandle> toSplit;
			for (auto e_it = mesh.edges_begin(); e_it != mesh.edges_end(); ++e_it) {
				if (mesh.status(*e_it).deleted())
					continue;
				HalfedgeHandle he = mesh.halfedge_handle(*e_it, 0);
				Vec3d p0 = mesh.point(mesh.from_vertex_handle(he));
				Vec3d p1 = p0 + mesh.wrapVector(mesh.point(mesh.to_vertex_handle(he)) - p0);

				const double a0 = static_cast<double>(p0[axis]);
				const double a1 = static_cast<double>(p1[axis]);

				if (std::abs(a0) < 1e-5 || std::abs(a0 - L[axis]) < 1e-5)
					continue;
				if (std::abs(a1) < 1e-5 || std::abs(a1 - L[axis]) < 1e-5)
					continue;
				if (a1 < -1e-5 || a1 > L[axis] + 1e-5)
					toSplit.push_back(*e_it);
			}

			for (const EdgeHandle& eh : toSplit) {
				if (!eh.is_valid() || mesh.status(eh).deleted())
					continue;
				HalfedgeHandle he = mesh.halfedge_handle(eh, 0);
				VertexHandle vFrom = mesh.from_vertex_handle(he);
				VertexHandle vTo = mesh.to_vertex_handle(he);
				Vec3d p0 = mesh.point(vFrom);
				Vec3d p1 = p0 + mesh.wrapVector(mesh.point(vTo) - p0);
				const double a0 = static_cast<double>(p0[axis]);
				const double a1 = static_cast<double>(p1[axis]);
				const double cut = (a1 > L[axis]) ? L[axis] : 0.0;
				const double denom = a1 - a0;
				if (std::abs(denom) < 1e-15)
					continue;
				const double t = (cut - a0) / denom;
				if (t <= 1e-6) {
					auto pp = mesh.point(vFrom);
					pp[axis] = static_cast<DefaultTriMesh::Scalar>(cut);
					mesh.set_point(vFrom, pp);
				} else if (t >= 1.0 - 1e-6) {
					auto pp = mesh.point(vTo);
					pp[axis] = static_cast<DefaultTriMesh::Scalar>(cut);
					mesh.set_point(vTo, pp);
				} else {
					Vec3d c;
					for (int k = 0; k < 3; ++k) {
						c[k] = static_cast<DefaultTriMesh::Scalar>((1.0 - t) *
																	   static_cast<double>(p0[k]) +
																   t * static_cast<double>(p1[k]));
					}
					const int nvBefore = static_cast<int>(mesh.n_vertices());
					mesh.split(eh, c);
					if (static_cast<int>(mesh.n_vertices()) == nvBefore + 1) {
						const double cv = (1.0 - t) * curv[static_cast<std::size_t>(vFrom.idx())] +
										  t * curv[static_cast<std::size_t>(vTo.idx())];
						curv.push_back(cv);
					}
				}
			}
			mesh.periodShift();
		}
		mesh.garbage_collection();
	}

	const std::size_t nv0 = mesh.n_vertices();
	if (curv.size() != nv0) {
		std::cerr << "[saveSplitUnitCellWithSingularity] curvature size mismatch after edge split: "
				  << curv.size() << " vs nv=" << nv0 << "\n";
		return false;
	}

	std::vector<std::array<double, 3>> vpos(nv0);
	for (auto v_it = mesh.vertices_begin(); v_it != mesh.vertices_end(); ++v_it) {
		const Vec3d& p = mesh.point(*v_it);
		vpos[static_cast<std::size_t>((*v_it).idx())] = {
			static_cast<double>(p[0]), static_cast<double>(p[1]), static_cast<double>(p[2])};
	}
	struct F3 {
		int v[3];
	};
	std::vector<F3> flist;
	flist.reserve(mesh.n_faces());
	for (auto f_it = mesh.faces_begin(); f_it != mesh.faces_end(); ++f_it) {
		auto fv = mesh.cfv_iter(*f_it);
		const int a = (*fv).idx();
		++fv;
		const int b = (*fv).idx();
		++fv;
		const int c = (*fv).idx();
		flist.push_back({a, b, c});
	}

	std::map<std::array<int, 3>, int> posMap;
	auto quantize = [](double x, double y, double z) -> std::array<int, 3> {
		return {static_cast<int>(std::round(x * 1e5)),
				static_cast<int>(std::round(y * 1e5)),
				static_cast<int>(std::round(z * 1e5))};
	};
	for (std::size_t i = 0; i < nv0; ++i) {
		posMap[quantize(vpos[i][0], vpos[i][1], vpos[i][2])] = static_cast<int>(i);
	}

	std::vector<std::array<double, 3>> extraV;
	std::vector<double> extraCurv;

	for (auto& f : flist) {
		double vnew[3][3];
		for (int j = 0; j < 3; ++j) {
			for (int k = 0; k < 3; ++k) {
				vnew[j][k] = vpos[static_cast<std::size_t>(f.v[j])][static_cast<std::size_t>(k)];
			}
		}

		bool hasPeriod = false;
		for (int j = 0; j < 3; ++j) {
			const int j1 = (j + 1) % 3;
			for (int k = 0; k < 3; ++k) {
				const double ej = vnew[j1][k] - vnew[j][k];
				if (ej < -hp[k]) {
					vnew[j1][k] += L[k];
					hasPeriod = true;
				} else if (ej > hp[k]) {
					vnew[j][k] += L[k];
					hasPeriod = true;
				}
			}
		}
		for (int k = 0; k < 3; ++k) {
			const double cmax = std::max({vnew[0][k], vnew[1][k], vnew[2][k]});
			const double cmin = std::min({vnew[0][k], vnew[1][k], vnew[2][k]});
			if (cmax > L[k] + 1e-5) {
				vnew[0][k] -= L[k];
				vnew[1][k] -= L[k];
				vnew[2][k] -= L[k];
			} else if (cmin < -1e-5) {
				vnew[0][k] += L[k];
				vnew[1][k] += L[k];
				vnew[2][k] += L[k];
			}
		}

		if (hasPeriod) {
			for (int j = 0; j < 3; ++j) {
				const int srcIdx = f.v[j];
				const double srcCurv = curv[static_cast<std::size_t>(srcIdx)];
				const auto qk = quantize(vnew[j][0], vnew[j][1], vnew[j][2]);
				if (posMap.count(qk)) {
					f.v[j] = posMap[qk];
				} else {
					const int newIdx = static_cast<int>(nv0 + extraV.size());
					posMap[qk] = newIdx;
					extraV.push_back({vnew[j][0], vnew[j][1], vnew[j][2]});
					extraCurv.push_back(srcCurv);
					f.v[j] = newIdx;
				}
			}
		}
	}

	std::vector<double> curvOut = curv;
	curvOut.insert(curvOut.end(), extraCurv.begin(), extraCurv.end());

	mesh.clear();
	std::vector<VertexHandle> vh;
	vh.reserve(nv0 + extraV.size());
	for (std::size_t i = 0; i < nv0; ++i) {
		vh.push_back(mesh.add_vertex(Vec3d(static_cast<DefaultTriMesh::Scalar>(vpos[i][0]),
										   static_cast<DefaultTriMesh::Scalar>(vpos[i][1]),
										   static_cast<DefaultTriMesh::Scalar>(vpos[i][2]))));
	}
	for (const auto& ev : extraV) {
		vh.push_back(mesh.add_vertex(Vec3d(static_cast<DefaultTriMesh::Scalar>(ev[0]),
										   static_cast<DefaultTriMesh::Scalar>(ev[1]),
										   static_cast<DefaultTriMesh::Scalar>(ev[2]))));
	}
	for (const auto& f : flist) {
		auto fh = mesh.add_face(vh[static_cast<std::size_t>(f.v[0])],
								vh[static_cast<std::size_t>(f.v[1])],
								vh[static_cast<std::size_t>(f.v[2])]);
		if (!fh.is_valid()) {
			mesh.add_face(vh[static_cast<std::size_t>(f.v[0])],
						  vh[static_cast<std::size_t>(f.v[2])],
						  vh[static_cast<std::size_t>(f.v[1])]);
		}
	}

	if (curvOut.size() != mesh.n_vertices()) {
		std::cerr << "[saveSplitUnitCellWithSingularity] curvature size mismatch after dup: "
				  << curvOut.size() << " vs nv=" << mesh.n_vertices() << "\n";
		return false;
	}

	if (!OpenMesh::IO::write_mesh(mesh, objPath)) {
		std::cerr << "[saveSplitUnitCellWithSingularity] failed to write " << objPath << "\n";
		return false;
	}

	std::ofstream out(singularityPath);
	if (!out) {
		std::cerr << "[saveSplitUnitCellWithSingularity] failed to write " << singularityPath
				  << "\n";
		return false;
	}
	out << std::setprecision(17);
	for (double v : curvOut) {
		out << v << "\n";
	}
	std::cerr << "[saveSplitUnitCellWithSingularity] wrote " << objPath << " and "
			  << singularityPath << " (nv=" << mesh.n_vertices() << ")\n";
	return true;
}

bool PeriodicTriMesh::saveUnitCell(const std::string& filename, bool split, bool splitEdges) const {
	if (split) {
		// Copy the mesh, then write out directly after splitUnitCell
		PeriodicTriMesh copy = *this;
		copy.splitUnitCell(splitEdges);
		return OpenMesh::IO::write_mesh(copy, filename);
	}
	// No split: write out directly after periodShift
	// All vertices are within the [0, L) domain; periodic-crossing faces keep their original
	// connectivity
	PeriodicTriMesh copy = *this;
	copy.periodShift();
	return OpenMesh::IO::write_mesh(copy, filename);
}

// ──────────────────────────────────────────────────────────────────────────
// surgery helpers (anonymous namespace)
// ──────────────────────────────────────────────────────────────────────────

namespace {

using VertexHandle = PeriodicTriMesh::VertexHandle;
using HalfedgeHandle = PeriodicTriMesh::HalfedgeHandle;

// ──────────────────────────────────────────────────────────────────────────
// computeSingularityMeasure (surgery uses VertexGeometry::computeSingularityMeasure)
// ──────────────────────────────────────────────────────────────────────────

// ── cullSmallIslands ──────────────────────────────────────────────────────
// Remove connected components whose face count < largest * cullRatio.
// Used after face deletion and after hole filling.
static void cullSmallIslands(PeriodicTriMesh& mesh, double cullRatio) {
	const std::size_t nf = mesh.n_faces();
	if (nf == 0)
		return;
	std::vector<int> compId(nf, -1);
	std::vector<int> compSize;
	int curComp = 0;
	for (std::size_t fi = 0; fi < nf; ++fi) {
		if (compId[fi] >= 0)
			continue;
		std::queue<int> q;
		q.push(static_cast<int>(fi));
		compId[fi] = curComp;
		int sz = 0;
		while (!q.empty()) {
			int fid = q.front();
			q.pop();
			++sz;
			auto fh = mesh.face_handle(fid);
			for (auto fh_it = mesh.cfh_iter(fh); fh_it.is_valid(); ++fh_it) {
				auto opp = mesh.opposite_halfedge_handle(*fh_it);
				if (mesh.is_boundary(opp))
					continue;
				int adj = mesh.face_handle(opp).idx();
				if (adj >= 0 && static_cast<std::size_t>(adj) < nf &&
					compId[static_cast<std::size_t>(adj)] < 0) {
					compId[static_cast<std::size_t>(adj)] = curComp;
					q.push(adj);
				}
			}
		}
		compSize.push_back(sz);
		++curComp;
	}
	if (curComp > 1) {
		int maxComp =
			static_cast<int>(std::max_element(compSize.begin(), compSize.end()) - compSize.begin());
		int threshold = static_cast<int>(compSize[static_cast<std::size_t>(maxComp)] * cullRatio);
		for (auto f_it = mesh.faces_begin(); f_it != mesh.faces_end(); ++f_it) {
			int cid = compId[static_cast<std::size_t>((*f_it).idx())];
			if (cid >= 0 && compSize[static_cast<std::size_t>(cid)] <= threshold)
				mesh.delete_face(*f_it, false);
		}
		mesh.garbage_collection();
	}
}

// ── expandSurgeryRegion ──────────────────────────────────────────────────
// Expand the set of faces to delete using bi-Laplacian smoothing,
// partitioning into patches, and BFS-based region growing.
// (aligned with minsurf expand_region)
static void expandSurgeryRegion(PeriodicTriMesh& mesh,
								std::set<int>& toDeleteFaceSet,
								const VertexGeometry& geom) {
	const std::size_t nv = mesh.n_vertices();
	auto L = assembleLaplacian(mesh, geom.cotWeights);
	Eigen::SparseMatrix<double> LtL = (L.transpose() * L).eval();
	Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver(LtL);

	// Partition the faces to delete into connected patches
	std::set<int> remaining = toDeleteFaceSet;
	std::vector<std::set<int>> patches;
	while (!remaining.empty()) {
		std::set<int> comp;
		std::queue<int> q;
		q.push(*remaining.begin());
		remaining.erase(remaining.begin());
		while (!q.empty()) {
			int fid = q.front();
			q.pop();
			comp.insert(fid);
			auto fh = mesh.face_handle(fid);
			for (auto fh_it = mesh.cfh_iter(fh); fh_it.is_valid(); ++fh_it) {
				auto opp = mesh.opposite_halfedge_handle(*fh_it);
				if (mesh.is_boundary(opp))
					continue;
				int adjF = mesh.face_handle(opp).idx();
				if (remaining.count(adjF)) {
					remaining.erase(adjF);
					q.push(adjF);
				}
			}
		}
		patches.push_back(comp);
	}

	// Perform Laplacian expansion for each connected patch
	toDeleteFaceSet.clear();
	for (auto& patch : patches) {
		Eigen::VectorXd rho = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(nv));
		for (int fi : patch) {
			auto fh = mesh.face_handle(fi);
			for (auto fv = mesh.cfv_iter(fh); fv.is_valid(); ++fv)
				rho[(*fv).idx()] = 1.0;
		}

		Eigen::VectorXd Mrho = geom.vertexAreas.cwiseProduct(rho);
		Eigen::VectorXd x = solver.solve((L.transpose() * Mrho).eval());
		x.array() -= geom.vertexAreas.dot(x) / geom.vertexAreas.sum();

		double xmin = 1e30, xmax = -1e30;
		for (int fi : patch) {
			auto fh = mesh.face_handle(fi);
			for (auto fv = mesh.cfv_iter(fh); fv.is_valid(); ++fv) {
				double xi = x[(*fv).idx()];
				xmin = std::min(xmin, xi);
				xmax = std::max(xmax, xi);
			}
		}

		// BFS expansion: include an adjacent face if its opposite vertex x value is within [xmin,
		// xmax]
		std::set<int> pext;
		std::queue<int> bfs;
		std::vector<bool> inBfs(mesh.n_faces(), false);
		for (int fi : patch) {
			bfs.push(fi);
			inBfs[static_cast<std::size_t>(fi)] = true;
		}
		while (!bfs.empty()) {
			int fi = bfs.front();
			bfs.pop();
			pext.insert(fi);
			auto fh = mesh.face_handle(fi);
			for (auto fh_it = mesh.cfh_iter(fh); fh_it.is_valid(); ++fh_it) {
				auto opp = mesh.opposite_halfedge_handle(*fh_it);
				if (mesh.is_boundary(opp))
					continue;
				int adjF = mesh.face_handle(opp).idx();
				if (inBfs[static_cast<std::size_t>(adjF)])
					continue;
				auto vop = mesh.to_vertex_handle(mesh.next_halfedge_handle(opp));
				if (x[vop.idx()] >= xmin && x[vop.idx()] <= xmax) {
					bfs.push(adjF);
					inBfs[static_cast<std::size_t>(adjF)] = true;
				}
			}
		}

		// Remove tip faces (faces with only one adjacent face in pext)
		for (int fi : pext) {
			int adjCount = 0;
			auto fh = mesh.face_handle(fi);
			for (auto fh_it = mesh.cfh_iter(fh); fh_it.is_valid(); ++fh_it) {
				auto opp = mesh.opposite_halfedge_handle(*fh_it);
				if (!mesh.is_boundary(opp) && pext.count(mesh.face_handle(opp).idx()))
					++adjCount;
			}
			if (adjCount >= 2)
				toDeleteFaceSet.insert(fi);
		}
	}
}

// ── fillSurgeryHoles ─────────────────────────────────────────────────────
// Extract boundary loops, build 4-ring submesh, CGAL hole filling,
// select correct patch, bilaplacian fairing.
// (aligned with minsurf fill_one_hole + localSmoothing)
// Returns the number of holes successfully processed.
static int fillSurgeryHoles(PeriodicTriMesh& mesh, const std::string& postFillPreFairDumpPath) {
	std::vector<std::vector<HalfedgeHandle>> loops;
	{
		std::set<int> visited;
		for (auto he_it = mesh.halfedges_begin(); he_it != mesh.halfedges_end(); ++he_it) {
			if (!mesh.is_boundary(*he_it))
				continue;
			if (visited.count((*he_it).idx()))
				continue;
			std::vector<HalfedgeHandle> loop;
			HalfedgeHandle he = *he_it, start = he;
			do {
				loop.push_back(he);
				visited.insert(he.idx());
				he = mesh.next_halfedge_handle(he);
			} while (he != start && loop.size() < 10000);
			if (!loop.empty())
				loops.push_back(loop);
		}
	}

	struct PatchJob {
		std::set<int> seedVerts;
		std::set<int> patchFaceIds;
		std::map<int, Vec3d> syncedPos;
	};
	std::vector<PatchJob> fairJobs;

	int holesProcessed = 0;
	for (auto& loop : loops) {
		// Collect seed vertices
		std::set<int> seedVerts;
		for (auto he : loop)
			seedVerts.insert(mesh.from_vertex_handle(he).idx());

		int holeEdgeNum = static_cast<int>(loop.size());
		(void)holeEdgeNum;

		// 4-ring expansion
		std::set<int> ringVerts = seedVerts;
		for (int kr = 0; kr < 4; ++kr) {
			std::set<int> toAdd;
			for (int vi : ringVerts)
				for (auto voh = mesh.cvoh_iter(VertexHandle(vi)); voh.is_valid(); ++voh)
					toAdd.insert(mesh.to_vertex_handle(*voh).idx());
			for (int v : toAdd)
				ringVerts.insert(v);
		}

		// Extract submesh faces
		std::set<int> ringFaces;
		for (int vi : ringVerts)
			for (auto vf = mesh.cvf_iter(VertexHandle(vi)); vf.is_valid(); ++vf)
				ringFaces.insert((*vf).idx());

		// BFS sync submesh coordinates
		std::map<int, Vec3d> syncedPos;
		{
			int startV = *ringVerts.begin();
			syncedPos[startV] = mesh.point(VertexHandle(startV));
			std::queue<int> bfs;
			bfs.push(startV);
			std::set<int> bfsVisited;
			bfsVisited.insert(startV);
			while (!bfs.empty()) {
				int cur = bfs.front();
				bfs.pop();
				for (auto voh = mesh.cvoh_iter(VertexHandle(cur)); voh.is_valid(); ++voh) {
					int nb = mesh.to_vertex_handle(*voh).idx();
					if (!ringVerts.count(nb) || bfsVisited.count(nb))
						continue;
					syncedPos[nb] = syncedPos[cur] + mesh.wrapVector(mesh.point(VertexHandle(nb)) -
																	 mesh.point(VertexHandle(cur)));
					bfsVisited.insert(nb);
					bfs.push(nb);
				}
			}
		}

		// Build CGAL mesh
		using CGALMesh = CGAL::Surface_mesh<CgalPoint>;
		CGALMesh cmesh;
		std::map<int, CGALMesh::Vertex_index> omToSm;
		std::vector<int> smToOm;

		for (int vi : ringVerts) {
			auto it = syncedPos.find(vi);
			if (it == syncedPos.end())
				continue;
			const Vec3d& p = it->second;
			auto smv = cmesh.add_vertex(CgalPoint(
				static_cast<double>(p[0]), static_cast<double>(p[1]), static_cast<double>(p[2])));
			omToSm[vi] = smv;
			smToOm.push_back(vi);
		}
		for (int fi : ringFaces) {
			auto fv = mesh.cfv_iter(mesh.face_handle(fi));
			int a = (*fv).idx();
			++fv;
			int b = (*fv).idx();
			++fv;
			int c = (*fv).idx();
			if (!omToSm.count(a) || !omToSm.count(b) || !omToSm.count(c))
				continue;
			cmesh.add_face(omToSm[a], omToSm[b], omToSm[c]);
		}

		// CGAL hole filling
		namespace PMP = CGAL::Polygon_mesh_processing;

		std::vector<CGALMesh::Halfedge_index> borderCycles;
		PMP::extract_boundary_cycles(cmesh, std::back_inserter(borderCycles));

		// Record the face count before hole filling
		std::size_t cmeshFacesBefore = cmesh.number_of_faces();

		// Repeatedly fill holes until only the outer boundary remains
		int filledCount = 0;
		for (int fillPass = 0; fillPass < 20; ++fillPass) {
			borderCycles.clear();
			PMP::extract_boundary_cycles(cmesh, std::back_inserter(borderCycles));
			if (borderCycles.size() <= 1)
				break;

			int maxCycleLen = 0, maxCycleIdx = 0;
			for (int bi = 0; bi < static_cast<int>(borderCycles.size()); ++bi) {
				int cl = 0;
				auto hh = borderCycles[static_cast<std::size_t>(bi)];
				do {
					++cl;
					hh = cmesh.next(hh);
				} while (hh != borderCycles[static_cast<std::size_t>(bi)] && cl < 100000);
				if (cl > maxCycleLen) {
					maxCycleLen = cl;
					maxCycleIdx = bi;
				}
			}
			int minCycleLen = 100000, minCycleIdx = -1;
			for (int bi = 0; bi < static_cast<int>(borderCycles.size()); ++bi) {
				if (bi == maxCycleIdx)
					continue;
				int cl = 0;
				auto hh = borderCycles[static_cast<std::size_t>(bi)];
				do {
					++cl;
					hh = cmesh.next(hh);
				} while (hh != borderCycles[static_cast<std::size_t>(bi)] && cl < 100000);
				if (cl < minCycleLen) {
					minCycleLen = cl;
					minCycleIdx = bi;
				}
			}
			if (minCycleIdx < 0)
				break;

			auto h = borderCycles[static_cast<std::size_t>(minCycleIdx)];
			std::vector<CGALMesh::Face_index> pf;
			std::vector<CGALMesh::Vertex_index> pv;
			try {
				PMP::triangulate_refine_and_fair_hole(
					cmesh, h, std::back_inserter(pf), std::back_inserter(pv));
				++filledCount;
			} catch (...) {
				std::cerr << "[surgery] CGAL fill failed len=" << minCycleLen << "\n";
				break;
			}
		}
		(void)filledCount;

		// Write newly created faces from the CGAL submesh back to the original mesh
		// Record the number of original faces in cmesh (faces that existed before hole filling)
		// Original faces = faces in ringFaces, distinguished by cmesh face index
		std::size_t origFaceCount = cmeshFacesBefore;
		std::set<int> patchFaceIds;
		int addOk = 0, addFail = 0;
		for (auto fi = cmesh.faces_begin(); fi != cmesh.faces_end(); ++fi) {
			// Skip original k-ring faces (they already exist in the original mesh)
			if (static_cast<std::size_t>((*fi).idx()) < origFaceCount)
				continue;

			auto hverts = cmesh.vertices_around_face(cmesh.halfedge(*fi));
			std::vector<int> fv;
			for (auto sv : hverts) {
				if (static_cast<std::size_t>(sv.idx()) < smToOm.size() &&
					smToOm[static_cast<std::size_t>(sv.idx())] >= 0) {
					fv.push_back(smToOm[static_cast<std::size_t>(sv.idx())]);
				} else {
					auto p = cmesh.point(sv);
					VertexHandle newV =
						mesh.add_vertex(Vec3d(static_cast<DefaultTriMesh::Scalar>(p.x()),
											  static_cast<DefaultTriMesh::Scalar>(p.y()),
											  static_cast<DefaultTriMesh::Scalar>(p.z())));
					while (smToOm.size() <= static_cast<std::size_t>(sv.idx()))
						smToOm.push_back(-1);
					smToOm[static_cast<std::size_t>(sv.idx())] = newV.idx();
					fv.push_back(newV.idx());
				}
			}
			if (fv.size() == 3) {
				auto fh =
					mesh.add_face(VertexHandle(fv[0]), VertexHandle(fv[1]), VertexHandle(fv[2]));
				if (!fh.is_valid())
					fh = mesh.add_face(
						VertexHandle(fv[0]), VertexHandle(fv[2]), VertexHandle(fv[1]));
				if (fh.is_valid()) {
					patchFaceIds.insert(fh.idx());
					++addOk;
				} else
					++addFail;
			}
		}
		(void)addOk;
		(void)addFail;

		if (!patchFaceIds.empty()) {
			fairJobs.push_back(
				{std::move(seedVerts), std::move(patchFaceIds), std::move(syncedPos)});
		}

		++holesProcessed;
	}

	if (!postFillPreFairDumpPath.empty()) {
		mesh.saveUnitCell(postFillPreFairDumpPath);
		std::cerr << "[surgery] saved post-fill pre-fair mesh: " << postFillPreFairDumpPath << "\n";
	}

	for (const auto& job : fairJobs) {
		const auto& seedVerts = job.seedVerts;
		const auto& patchFaceIds = job.patchFaceIds;
		const auto& syncedPos = job.syncedPos;

		// localSmoothing: bilaplacian fairing (aligned with minsurf)
		std::set<int> smoothFaces;
		for (int sv : seedVerts)
			for (auto vf = mesh.cvf_iter(VertexHandle(sv)); vf.is_valid(); ++vf)
				smoothFaces.insert((*vf).idx());
		for (int fi : patchFaceIds)
			smoothFaces.insert(fi);
		{
			std::set<int> toAdd;
			for (int fi : smoothFaces) {
				auto fh2 = mesh.face_handle(fi);
				for (auto fv2 = mesh.cfv_iter(fh2); fv2.is_valid(); ++fv2)
					for (auto vf2 = mesh.cvf_iter(*fv2); vf2.is_valid(); ++vf2)
						toAdd.insert((*vf2).idx());
			}
			for (int fi : toAdd)
				smoothFaces.insert(fi);
		}

		std::map<int, int> subVMap;
		std::vector<int> subVInv;
		for (int fi : smoothFaces) {
			auto fh2 = mesh.face_handle(fi);
			for (auto fv2 = mesh.cfv_iter(fh2); fv2.is_valid(); ++fv2) {
				int vi = (*fv2).idx();
				if (!subVMap.count(vi)) {
					subVMap[vi] = static_cast<int>(subVInv.size());
					subVInv.push_back(vi);
				}
			}
		}
		int snv = static_cast<int>(subVInv.size());
		if (snv == 0)
			continue;
		Eigen::MatrixX3d subV(snv, 3);
		std::vector<bool> synced(static_cast<std::size_t>(snv), false);
		synced[0] = true;
		{
			auto sp = syncedPos.count(subVInv[0]) ? syncedPos.at(subVInv[0])
												  : mesh.point(VertexHandle(subVInv[0]));
			subV.row(0) << static_cast<double>(sp[0]), static_cast<double>(sp[1]),
				static_cast<double>(sp[2]);
		}
		std::queue<int> syncQ;
		syncQ.push(0);
		while (!syncQ.empty()) {
			int si = syncQ.front();
			syncQ.pop();
			int omi = subVInv[static_cast<std::size_t>(si)];
			for (auto voh = mesh.cvoh_iter(VertexHandle(omi)); voh.is_valid(); ++voh) {
				int nb = mesh.to_vertex_handle(*voh).idx();
				if (!subVMap.count(nb))
					continue;
				int sn = subVMap[nb];
				if (synced[static_cast<std::size_t>(sn)])
					continue;
				Vec3d w =
					mesh.wrapVector(mesh.point(VertexHandle(nb)) - mesh.point(VertexHandle(omi)));
				subV.row(sn) = subV.row(si) + Eigen::RowVector3d(static_cast<double>(w[0]),
																 static_cast<double>(w[1]),
																 static_cast<double>(w[2]));
				synced[static_cast<std::size_t>(sn)] = true;
				syncQ.push(sn);
			}
		}
		Eigen::MatrixX3i subF(static_cast<int>(smoothFaces.size()), 3);
		int fc = 0;
		for (int fi : smoothFaces) {
			auto fh2 = mesh.face_handle(fi);
			auto fvi = mesh.cfv_iter(fh2);
			subF(fc, 0) = subVMap[(*fvi).idx()];
			++fvi;
			subF(fc, 1) = subVMap[(*fvi).idx()];
			++fvi;
			subF(fc, 2) = subVMap[(*fvi).idx()];
			++fc;
		}

		// Closed/periodic meshes have no OpenMesh boundary after hole fill.
		// Fix the submesh collar (rim + 1-ring) so bilaplacian is well-posed.
		std::set<int> rimSet;
		for (int si = 0; si < snv; ++si) {
			int omi = subVInv[static_cast<std::size_t>(si)];
			if (mesh.is_boundary(VertexHandle(omi)))
				rimSet.insert(si);
			for (auto voh = mesh.cvoh_iter(VertexHandle(omi)); voh.is_valid(); ++voh) {
				if (!subVMap.count(mesh.to_vertex_handle(*voh).idx())) {
					rimSet.insert(si);
					break;
				}
			}
		}
		std::set<int> fixSet = rimSet;
		for (int si : rimSet) {
			int omi = subVInv[static_cast<std::size_t>(si)];
			for (auto voh = mesh.cvoh_iter(VertexHandle(omi)); voh.is_valid(); ++voh) {
				int nb = mesh.to_vertex_handle(*voh).idx();
				if (subVMap.count(nb))
					fixSet.insert(subVMap[nb]);
			}
		}
		Eigen::MatrixX3d subVBefore = subV;
		Eigen::VectorXi fixDof(static_cast<int>(fixSet.size()));
		Eigen::MatrixX3d fixedY(static_cast<int>(fixSet.size()), 3);
		{
			int i = 0;
			for (int si : fixSet) {
				fixDof[i] = si;
				fixedY.row(i) = subV.row(si);
				++i;
			}
		}

		if (static_cast<int>(fixSet.size()) < snv && !fixSet.empty()) {
			for (int fair = 0; fair < 3; ++fair) {
				Eigen::SparseMatrix<double> L, M;
				igl::cotmatrix(subV, subF, L);
				igl::massmatrix(subV, subF, igl::MASSMATRIX_TYPE_DEFAULT, M);
				Eigen::SparseMatrix<double> L2 = (L.transpose() * M.cwiseInverse() * L).eval();
				Eigen::SparseMatrix<double> Aeq(0, snv);
				igl::min_quad_with_fixed_data<double> data;
				if (!igl::min_quad_with_fixed_precompute(L2, fixDof, Aeq, true, data))
					break;
				Eigen::MatrixXd B = Eigen::MatrixXd::Zero(snv, 3);
				Eigen::MatrixXd Beq(0, 3);
				if (!igl::min_quad_with_fixed_solve(data, B, fixedY, Beq, subV))
					break;
				if (subV.hasNaN()) {
					subV = subVBefore;
					break;
				}
			}
		}

		// Apply chart-space deltas onto original mesh coordinates (never write unwrapped abs
		// coords).
		for (int si = 0; si < snv; ++si) {
			const Eigen::RowVector3d delta = subV.row(si) - subVBefore.row(si);
			int omi = subVInv[static_cast<std::size_t>(si)];
			const Vec3d op = mesh.point(VertexHandle(omi));
			mesh.set_point(
				VertexHandle(omi),
				Vec3d(static_cast<DefaultTriMesh::Scalar>(static_cast<double>(op[0]) + delta[0]),
					  static_cast<DefaultTriMesh::Scalar>(static_cast<double>(op[1]) + delta[1]),
					  static_cast<DefaultTriMesh::Scalar>(static_cast<double>(op[2]) + delta[2])));
		}
	}

	return holesProcessed;
}

} // namespace

// ──────────────────────────────────────────────────────────────────────────
// surgery
// ──────────────────────────────────────────────────────────────────────────

bool PeriodicTriMesh::surgery(const SurgeryOptions& opts) {
	this->request_vertex_status();
	this->request_edge_status();
	this->request_halfedge_status();
	this->request_face_status();

	// ── [1] Compute singularity measure for each vertex ──
	const std::size_t nv = this->n_vertices();
	std::vector<double> singMeasure = computeSingularityMeasure(*this, opts.surgeryType);
	auto geom = computeVertexGeometry(*this);

	double avgH = 0;
	{
		double maxH = 0, sumH = 0;
		int cnt = 0;
		for (std::size_t i = 0; i < nv; ++i) {
			sumH += singMeasure[i];
			++cnt;
			maxH = std::max(maxH, singMeasure[i]);
		}
		avgH = (cnt > 0 ? sumH / cnt : 0);
		std::cerr << "[surgery] maxH=" << maxH << " avgH=" << avgH << " tol=" << opts.singularityTol
				  << " ratio=" << opts.singularityRatio << "\n";
	}

	// ── [2] Mark singular vertices -> collect faces to delete -> partition into connected patches
	// ── Adaptive threshold: max(singularityTol, singularityRatio * avgH) Avoids misclassifying
	// normal high-curvature vertices as singular during global wrinkling (when avgH is close to or
	// exceeds tol)
	const double effTol = std::max(opts.singularityTol, opts.singularityRatio * avgH);
	std::set<int> toDeleteFaceSet;
	for (auto v_it = this->vertices_begin(); v_it != this->vertices_end(); ++v_it) {
		if (singMeasure[static_cast<std::size_t>((*v_it).idx())] > effTol) {
			for (auto vf_it = this->cvf_iter(*v_it); vf_it.is_valid(); ++vf_it)
				toDeleteFaceSet.insert((*vf_it).idx());
		}
	}
	if (toDeleteFaceSet.empty())
		return false;

	// Limit excision area: if the candidate deletion fraction is too high, the high curvature is
	// global wrinkling rather than local necking
	{
		const std::size_t nfTotal = this->n_faces();
		double cutFrac = nfTotal > 0 ? (double)toDeleteFaceSet.size() / nfTotal : 0;
		if (cutFrac > opts.maxCutAreaFraction) {
			std::cerr << "[surgery] aborted: cut fraction " << cutFrac << " > maxCutAreaFraction "
					  << opts.maxCutAreaFraction << " (likely global wrinkle, not local neck)\n";
			return false;
		}
	}

	// ── [3] Expand the deletion region using bi-Laplacian smoothing ──
	expandSurgeryRegion(*this, toDeleteFaceSet, geom);

	// ── [4] Delete faces -> garbage collect -> remove islands ──
	for (int fi : toDeleteFaceSet)
		this->delete_face(this->face_handle(fi), false);
	this->garbage_collection();
	cullSmallIslands(*this, opts.islandCullRatio);

	if (!opts.preFillDumpPath.empty()) {
		this->saveUnitCell(opts.preFillDumpPath);
		std::cerr << "[surgery] saved pre-fill mesh: " << opts.preFillDumpPath << "\n";
	}

	// ── [5] Fill holes (+ optional dump before bilaplacian) then bilaplacian fairing ──
	fillSurgeryHoles(*this, opts.postFillPreFairDumpPath);

	// ── [6] Remove islands again (aligned with minsurf: deleteisoisland after fill_holes) ──
	cullSmallIslands(*this, opts.islandCullRatio);

	this->garbage_collection();
	// Cleanup: patch vertex coordinates from CGAL hole filling are in unwrapped space (may exceed
	// the domain), shift back to [0, 2*hp) to ensure correct behavior of subsequent
	// classify/mergePeriodBoundary
	this->periodShift();
	return true;
}

void PeriodicTriMesh::periodizeFrom(const DefaultTriMesh& naiveMesh, PeriodizeOptions& options) {
	DefaultTriMesh scratch;
	const DefaultTriMesh* src = &naiveMesh;
	if (this == &naiveMesh) {
		scratch = naiveMesh;
		src = &scratch;
	}

	assignTriMesh(*src, *this);

	if (options.collapseShortBoundaryEdgesBelow > 0.0) {
		mergePeriodEdges(options.collapseShortBoundaryEdgesBelow);
	}

	for (int ax = 0; ax < 3; ++ax) {
		if (!(static_cast<double>(halfPeriod_[ax]) > 0.0)) {
			throw std::invalid_argument("periodizeFrom: halfPeriod must be positive on every axis");
		}
	}

	Vec3d meshBmin{};
	Vec3d meshBmax{};
	computeAxisAlignedBounds(*this, meshBmin, meshBmax);
	translateMeshInPlace(*this, meshBmin);

	const double Lx = 2.0 * static_cast<double>(halfPeriod_[0]);
	const double Ly = 2.0 * static_cast<double>(halfPeriod_[1]);
	const double Lz = 2.0 * static_cast<double>(halfPeriod_[2]);

	Vec3d fundMin(0.0, 0.0, 0.0);
	Vec3d fundMax(static_cast<DefaultTriMesh::Scalar>(Lx),
				  static_cast<DefaultTriMesh::Scalar>(Ly),
				  static_cast<DefaultTriMesh::Scalar>(Lz));
	applyBBoxPadding(fundMin, fundMax, options.bboxPaddingWorld, options.bboxPaddingFraction);
	options.resolvedBBoxMin = {static_cast<double>(fundMin[0]),
							   static_cast<double>(fundMin[1]),
							   static_cast<double>(fundMin[2])};
	options.resolvedBBoxMax = {static_cast<double>(fundMax[0]),
							   static_cast<double>(fundMax[1]),
							   static_cast<double>(fundMax[2])};

	const int N = std::max(1, options.mcCellsPerAxis);
	const int nx = N;
	const int ny = N;
	const int nz = N;
	const int Np = N + 1;
	const int nNodes = Np * Np * Np;

	std::vector<std::array<double, 3>> baseVerts;
	std::vector<TriMeshFace> baseFaces;
	meshToRaw(*this, baseVerts, baseFaces);
	if (baseFaces.empty()) {
		throw std::invalid_argument("periodizeFrom: mesh has no faces");
	}

	std::vector<std::array<double, 3>> extVerts;
	std::vector<TriMeshFace> extFaces;
	buildExtendedMesh27(baseVerts, baseFaces, Lx, Ly, Lz, extVerts, extFaces);
	if (extFaces.empty()) {
		throw std::runtime_error("periodizeFrom: extended mesh is empty");
	}

	TriMeshAABBTree tree;
	tree.build(extVerts, extFaces);
	if (tree.empty()) {
		throw std::runtime_error("periodizeFrom: AABB tree build failed (degenerate triangles?)");
	}

	std::vector<uint8_t> activeCells;
	markCellsOverlappingFundDomain(extVerts, extFaces, Lx, Ly, Lz, nx, ny, nz, activeCells);
	dilateActiveCells(activeCells, nx, ny, nz, options.mcVoxelDilateLayers);

	const double dx = Lx / static_cast<double>(nx);
	const double dy = Ly / static_cast<double>(ny);
	const double dz = Lz / static_cast<double>(nz);

	std::vector<double> phi(static_cast<size_t>(nNodes), 0.0);
	for (int k = 0; k < Np; ++k) {
		for (int j = 0; j < Np; ++j) {
			for (int i = 0; i < Np; ++i) {
				const std::array<double, 3> q = {static_cast<double>(i) * dx,
												 static_cast<double>(j) * dy,
												 static_cast<double>(k) * dz};
				phi[static_cast<size_t>(nodeIndex(i, j, k, Np))] =
					signedDistanceAt(q, tree, extVerts, extFaces);
			}
		}
	}

	Dsu dsu(nNodes);
	for (int k = 0; k < Np; ++k) {
		for (int j = 0; j < Np; ++j) {
			for (int i = 0; i < Np; ++i) {
				const int id = nodeIndex(i, j, k, Np);
				if (i == 0) {
					dsu.unite(id, nodeIndex(N, j, k, Np));
				}
				if (j == 0) {
					dsu.unite(id, nodeIndex(i, N, k, Np));
				}
				if (k == 0) {
					dsu.unite(id, nodeIndex(i, j, N, Np));
				}
			}
		}
	}

	std::vector<double> sumPhi(static_cast<size_t>(nNodes), 0.0);
	std::vector<int> cnt(static_cast<size_t>(nNodes), 0);
	for (int id = 0; id < nNodes; ++id) {
		const int r = dsu.find(id);
		sumPhi[static_cast<size_t>(r)] += phi[static_cast<size_t>(id)];
		cnt[static_cast<size_t>(r)] += 1;
	}
	for (int id = 0; id < nNodes; ++id) {
		const int r = dsu.find(id);
		phi[static_cast<size_t>(id)] =
			sumPhi[static_cast<size_t>(r)] / static_cast<double>(cnt[static_cast<size_t>(r)]);
	}

	std::vector<LevelSetNode> nodes(static_cast<size_t>(nNodes));
	for (int k = 0; k < Np; ++k) {
		for (int j = 0; j < Np; ++j) {
			for (int i = 0; i < Np; ++i) {
				const int id = nodeIndex(i, j, k, Np);
				nodes[static_cast<size_t>(id)] = {static_cast<double>(i) * dx,
												  static_cast<double>(j) * dy,
												  static_cast<double>(k) * dz,
												  phi[static_cast<size_t>(id)]};
			}
		}
	}

	SparseVoxelCornerMap voxels;
	for (int k = 0; k < nz; ++k) {
		for (int j = 0; j < ny; ++j) {
			for (int i = 0; i < nx; ++i) {
				if (!activeCells[static_cast<size_t>(cellIndex(i, j, k, nx, ny))]) {
					continue;
				}
				VoxelCornerNodeIndices c{};
				c[0] = static_cast<std::size_t>(nodeIndex(i, j, k, Np));
				c[1] = static_cast<std::size_t>(nodeIndex(i + 1, j, k, Np));
				c[2] = static_cast<std::size_t>(nodeIndex(i + 1, j + 1, k, Np));
				c[3] = static_cast<std::size_t>(nodeIndex(i, j + 1, k, Np));
				c[4] = static_cast<std::size_t>(nodeIndex(i, j, k + 1, Np));
				c[5] = static_cast<std::size_t>(nodeIndex(i + 1, j, k + 1, Np));
				c[6] = static_cast<std::size_t>(nodeIndex(i + 1, j + 1, k + 1, Np));
				c[7] = static_cast<std::size_t>(nodeIndex(i, j + 1, k + 1, Np));
				voxels[{static_cast<std::int32_t>(i),
						static_cast<std::int32_t>(j),
						static_cast<std::int32_t>(k)}] = c;
			}
		}
	}

	if (voxels.empty()) {
		this->clear();
		return;
	}

	DefaultTriMesh mcOut;
	ExtractMarchingCubesOptions mcOpts;
	mcOpts.weldSharedEdges = options.mcWeldVertices;
	marchingCubesExtractToTriMesh(voxels, nodes, options.mcIsoValue, mcOut, mcOpts);
	assignTriMesh(mcOut, *this);
}

} // namespace xtpms
