#include "PeriodicMesh.h"

#include "AABBTree.h"
#include "AsymptoticConductivity.h"
#include "MarchingCubes.h"

#include <Eigen/Dense>
#include <OpenMesh/Core/IO/MeshIO.hh>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
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
		out.add_face(vh[static_cast<size_t>(a)], vh[static_cast<size_t>(b)], vh[static_cast<size_t>(c)]);
	}
}

void meshToRaw(const DefaultTriMesh& mesh, std::vector<std::array<double, 3>>& verts, std::vector<TriMeshFace>& faces) {
	verts.resize(mesh.n_vertices());
	for (auto v_it = mesh.vertices_begin(); v_it != mesh.vertices_end(); ++v_it) {
		const Vec3d& p = mesh.point(*v_it);
		verts[static_cast<size_t>((*v_it).idx())] = {static_cast<double>(p[0]), static_cast<double>(p[1]), static_cast<double>(p[2])};
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

void buildExtendedMesh27(
	const std::vector<std::array<double, 3>>& baseVerts,
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

void markCellsOverlappingFundDomain(
	const std::vector<std::array<double, 3>>& extVerts,
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
								if (ni < 0 || ni >= nx || nj < 0 || nj >= ny || nk < 0 || nk >= nz) {
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
		return parent[static_cast<size_t>(x)] == x ? x : (parent[static_cast<size_t>(x)] = find(parent[static_cast<size_t>(x)]));
	}
	void unite(int a, int b) {
		a = find(a);
		b = find(b);
		if (a != b) {
			parent[static_cast<size_t>(b)] = a;
		}
	}
};

double signedDistanceAt(
	const std::array<double, 3>& q,
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
	Vec3d va{static_cast<DefaultTriMesh::Scalar>(pb[0] - pa[0]), static_cast<DefaultTriMesh::Scalar>(pb[1] - pa[1]),
		static_cast<DefaultTriMesh::Scalar>(pb[2] - pa[2])};
	Vec3d vb{static_cast<DefaultTriMesh::Scalar>(pc[0] - pa[0]), static_cast<DefaultTriMesh::Scalar>(pc[1] - pa[1]),
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

void PeriodicTriMesh::setHalfPeriod(const Vec3d& halfPeriod) { halfPeriod_ = halfPeriod; }

PeriodicTriMesh::Vec3d PeriodicTriMesh::halfPeriod() const { return halfPeriod_; }

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

PeriodicTriMesh::Vec3d PeriodicTriMesh::shift2origin(const Vec3d& p) const { return wrapVector(p); }

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
// mergePeriodBoundary 辅助
// ──────────────────────────────────────────────────────────────────────────

namespace {

using CgalK = CGAL::Simple_cartesian<double>;
using CgalPoint = CgalK::Point_3;
using CgalSegment = CgalK::Segment_3;

// CGAL AABB segment primitive：存储指向 vector 元素的迭代器
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
	return CgalPoint(static_cast<double>(p[0]), static_cast<double>(p[1]), static_cast<double>(p[2]));
}

// 判断 p 是否在线段端点附近
int isEndpoint(const CgalPoint& p, const PeriodicTriMesh::Vec3d& v0, const PeriodicTriMesh::Vec3d& v1, double eps) {
	double d0 = CGAL::squared_distance(p, toCgal(v0));
	double d1 = CGAL::squared_distance(p, toCgal(v1));
	if (d0 < eps * eps) return 0;
	if (d1 < eps * eps) return 1;
	return -1;
}

// 简单的 PeriodicGridIndex：用于顶点去重（周期性 wrap）
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

	PeriodicVertexGrid(const std::array<double, 3>& orig, const std::array<double, 3>& diag, double eps)
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
		if (it != lattice.end()) return it->second;
		// 写入 27 邻域
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

} // namespace

void PeriodicTriMesh::mergePeriodBoundary(const MergeBoundaryOptions& options) {
	// OpenMesh 需要 status 属性才能执行 collapse/split/delete/garbage_collection
	this->request_vertex_status();
	this->request_edge_status();
	this->request_halfedge_status();
	this->request_face_status();

	const double Lx = 2.0 * static_cast<double>(halfPeriod_[0]);
	const double Ly = 2.0 * static_cast<double>(halfPeriod_[1]);
	const double Lz = 2.0 * static_cast<double>(halfPeriod_[2]);
	const double maxL = std::max({Lx, Ly, Lz});

	// 投影容差：取用户指定值和网格尺度的 1% 中较大的
	const double projTol = std::max(options.projectionTol, maxL * 0.02);
	// 顶点合并容差：取用户值和网格尺度的 0.01% 中较大的
	const double weldTol = std::max(options.vertexWeldTol, maxL * 1e-4);
	// 短边容差：取用户值和网格尺度的 0.2% 中较大的
	const double shortTol = std::max(options.shortEdgeTol, maxL * 0.002);

	// domain: [0, Lx] x [0, Ly] x [0, Lz]
	const double domMin[3] = {0.0, 0.0, 0.0};
	const double domMax[3] = {Lx, Ly, Lz};

	// 边界容差：取每轴尺寸的 1e-6 的最大值，至少 1e-10
	const double borderTol = std::max(1e-10, 1e-6 * std::max({Lx, Ly, Lz}));

	// ── Step 1: 标记所有边界顶点 ──
	auto classifyVertices = [&]() {
		std::unordered_map<int, BoundaryFlag> vtag;
		for (auto v_it = this->vertices_begin(); v_it != this->vertices_end(); ++v_it) {
			if (!this->is_boundary(*v_it)) continue;
			const Vec3d& p = this->point(*v_it);
			BoundaryFlag bf;
			bf.classify(static_cast<double>(p[0]), static_cast<double>(p[1]), static_cast<double>(p[2]),
						domMin[0], domMax[0], domMin[1], domMax[1], domMin[2], domMax[2], borderTol);
			if (bf.isBoundary()) vtag[(*v_it).idx()] = bf;
		}
		return vtag;
	};

	auto vtag = classifyVertices();

	{
		int nBndVerts = 0, nTagged = 0, nMin = 0, nMax = 0;
		for (auto v_it = this->vertices_begin(); v_it != this->vertices_end(); ++v_it) {
			if (this->is_boundary(*v_it)) ++nBndVerts;
		}
		for (auto& [idx, bf] : vtag) {
			++nTagged;
			if (bf.isMinBoundary()) ++nMin;
			if (bf.isMaxBoundary()) ++nMax;
		}
		std::cerr << "[merge] boundary verts=" << nBndVerts << " tagged=" << nTagged
				  << " min=" << nMin << " max=" << nMax << "\n";
		// 输出前几个 min 和 max 顶点的坐标
		int dbgCnt = 0;
		for (auto& [idx, bf] : vtag) {
			if (dbgCnt >= 6) break;
			const Vec3d& p = this->point(VertexHandle(idx));
			std::cerr << "  v" << idx << " p=(" << p[0] << "," << p[1] << "," << p[2]
					  << ") min=" << bf.isMinBoundary() << " max=" << bf.isMaxBoundary()
					  << " flag=" << bf.getMask() << "\n";
			++dbgCnt;
		}
	}

	// ── Step 2: 折叠短边界边（保留 min 端） ──
	{
		std::vector<EdgeHandle> edgesToCollapse;
		for (auto e_it = this->edges_begin(); e_it != this->edges_end(); ++e_it) {
			const EdgeHandle eh = *e_it;
			if (!eh.is_valid()) continue;
			const HalfedgeHandle he = this->halfedge_handle(eh, 0);
			const VertexHandle vFrom = this->from_vertex_handle(he);
			const VertexHandle vTo = this->to_vertex_handle(he);
			const Vec3d& pa = this->point(vFrom);
			const Vec3d& pb = this->point(vTo);
			if ((pa - pb).norm() < shortTol) {
				auto itFrom = vtag.find(vFrom.idx());
				auto itTo = vtag.find(vTo.idx());
				if (itFrom != vtag.end() || itTo != vtag.end()) {
					edgesToCollapse.push_back(eh);
				}
			}
		}
		for (const EdgeHandle& eh : edgesToCollapse) {
			if (!eh.is_valid() || eh.idx() < 0 || static_cast<std::size_t>(eh.idx()) >= this->n_edges()) continue;
			if (this->status(eh).deleted()) continue;
			const HalfedgeHandle he = this->halfedge_handle(eh, 0);
			const VertexHandle vFrom = this->from_vertex_handle(he);
			const VertexHandle vTo = this->to_vertex_handle(he);
			auto itFrom = vtag.find(vFrom.idx());
			auto itTo = vtag.find(vTo.idx());
			int flagFrom = (itFrom != vtag.end()) ? itFrom->second.getMask() : 0;
			int flagTo = (itTo != vtag.end()) ? itTo->second.getMask() : 0;
			int common = flagFrom & flagTo;
			// 折叠方向：保留 flag 更多的（min 优先）
			if ((~common) & flagFrom) {
				// from 有 to 没有的 flag -> collapse he.opp（保留 from）
				const HalfedgeHandle oppHe = this->halfedge_handle(eh, 1);
				if (this->is_collapse_ok(oppHe)) this->collapse(oppHe);
			} else {
				if (this->is_collapse_ok(he)) this->collapse(he);
			}
		}
		this->garbage_collection();
		vtag = classifyVertices(); // 重新标记
	}

	// ── Step 3: 用 CGAL AABB segment tree 投影 max→min 边界 ──
	auto buildSegTree = [&](bool buildMin) {
		// 收集位于目标边界面上的半边：只要半边两端都是边界顶点就作为候选
		std::vector<std::pair<HalfedgeHandle, int>> bndHEs; // halfedge + period mask
		for (auto e_it = this->edges_begin(); e_it != this->edges_end(); ++e_it) {
			if (!this->is_boundary(*e_it)) continue;
			for (int side = 0; side < 2; ++side) {
				const HalfedgeHandle he = this->halfedge_handle(*e_it, side);
				if (!this->is_boundary(he)) continue;
				const VertexHandle vf = this->from_vertex_handle(he);
				const VertexHandle vt = this->to_vertex_handle(he);
				auto itF = vtag.find(vf.idx());
				auto itT = vtag.find(vt.idx());
				// 两端都必须是边界顶点且在同一面上
				if (itF == vtag.end() || itT == vtag.end()) continue;
				int mask = itF->second.getMask() & itT->second.getMask();
				if (!mask) continue;
				bool isMinEdge = (mask & BoundaryFlag::kMinMask) != 0;
				bool isMaxEdge = (mask & BoundaryFlag::kMaxMask) != 0;
				if (buildMin ? isMinEdge : isMaxEdge) {
					bndHEs.push_back({he, mask});
				}
			}
		}
		return bndHEs;
	};

	// 存储每个原始半边对应的分裂后半边列表
	std::unordered_map<int, std::vector<HalfedgeHandle>> splittedHE;

	auto projectAndSplit = [&](bool projectMaxToMin) {
		auto bndHEs = buildSegTree(projectMaxToMin); // 目标边（min 或 max）
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
		std::cerr << "[merge] projectAndSplit(" << (projectMaxToMin?"max→min":"min→max")
				  << "): " << segments.size() << " target segments\n";
		if (segments.empty()) return;

		SegAABBTree tree(segments.begin(), segments.end());
		tree.build();

		int srcCount = 0, projectedCount = 0, splitCount = 0;
		// 投影源端点
		for (auto& [vidx, bf] : vtag) {
			const VertexHandle vh(vidx);
			if (vidx < 0 || static_cast<std::size_t>(vidx) >= this->n_vertices()) continue;
			if (this->status(vh).deleted()) continue;
			bool isSource = projectMaxToMin ? bf.isMaxBoundary() : bf.isMinBoundary();
			if (!isSource) continue;
			++srcCount;

			Vec3d p = this->point(vh);
			// 平移到目标侧
			Vec3d translated = p;
			for (int ax = 0; ax < 3; ++ax) {
				if (projectMaxToMin) {
					if (bf.isMaxBoundary(ax)) translated[ax] -= static_cast<DefaultTriMesh::Scalar>(domMax[ax] - domMin[ax]);
				} else {
					if (bf.isMinBoundary(ax)) translated[ax] += static_cast<DefaultTriMesh::Scalar>(domMax[ax] - domMin[ax]);
				}
			}

			auto result = tree.closest_point_and_primitive(toCgal(translated));
			CgalPoint closest = result.first;
			double sqDist = CGAL::squared_distance(closest, toCgal(translated));
			if (sqDist > projTol * projTol) {
				continue;
			}
			++projectedCount;
			if (projectedCount <= 3) {
				std::cerr << "[merge]   proj v" << vidx << " translated=(" << translated[0] << "," << translated[1] << "," << translated[2]
						  << ") closest=(" << closest.x() << "," << closest.y() << "," << closest.z()
						  << ") dist=" << std::sqrt(sqDist) << "\n";
			}

			std::size_t primIdx = result.second->second;
			auto& heSplits = splittedHE[static_cast<int>(primIdx)];

			// 在分裂后的子半边中找到包含 closest 的那条
			bool matched = false;
			for (std::size_t k = 0; k < heSplits.size(); ++k) {
				const HalfedgeHandle he = heSplits[k];
				if (!he.is_valid()) continue;
				const EdgeHandle eeh = this->edge_handle(he);
				if (!eeh.is_valid() || this->status(eeh).deleted()) continue;
				const Vec3d& hFrom = this->point(this->from_vertex_handle(he));
				const Vec3d& hTo = this->point(this->to_vertex_handle(he));
				// 用边长的 1% 作为端点判断容差
				double heLen = (hTo - hFrom).norm();
				double endEps = std::max(1e-6, heLen * 0.01);
				int endId = isEndpoint(closest, hFrom, hTo, endEps);
				if (endId == 0) {
					// 匹配 from 端点
					Vec3d trans{};
					for (int ax = 0; ax < 3; ++ax) {
						trans[ax] = projectMaxToMin
							? static_cast<DefaultTriMesh::Scalar>(domMax[ax] - domMin[ax])
							: static_cast<DefaultTriMesh::Scalar>(-(domMax[ax] - domMin[ax]));
						if (projectMaxToMin ? !bf.isMaxBoundary(ax) : !bf.isMinBoundary(ax))
							trans[ax] = 0;
					}
					this->set_point(vh, hFrom + trans);
					matched = true;
					break;
				} else if (endId == 1) {
					Vec3d trans{};
					for (int ax = 0; ax < 3; ++ax) {
						trans[ax] = projectMaxToMin
							? static_cast<DefaultTriMesh::Scalar>(domMax[ax] - domMin[ax])
							: static_cast<DefaultTriMesh::Scalar>(-(domMax[ax] - domMin[ax]));
						if (projectMaxToMin ? !bf.isMaxBoundary(ax) : !bf.isMinBoundary(ax))
							trans[ax] = 0;
					}
					this->set_point(vh, hTo + trans);
					matched = true;
					break;
				} else {
					// 需要 split：在目标边上投影点处插入新顶点
					const Vec3d newPt(static_cast<DefaultTriMesh::Scalar>(closest.x()),
									  static_cast<DefaultTriMesh::Scalar>(closest.y()),
									  static_cast<DefaultTriMesh::Scalar>(closest.z()));
					// 检查投影点不在边的端点上（避免退化 split）
					// 检查投影点不太靠近端点（参数 t 在 [0.02, 0.98] 范围内才 split）
					const Vec3d edgeVec = hTo - hFrom;
					const double edgeLen = edgeVec.norm();
					if (edgeLen < 1e-15) continue;
					double t = static_cast<double>((newPt - hFrom) | edgeVec) / (edgeLen * edgeLen);
					if (t < 0.02 || t > 0.98) continue;

					VertexHandle newVH = this->add_vertex(newPt);
					this->split(this->edge_handle(he), newVH);

					// 设置源顶点位置
					Vec3d trans{};
					for (int ax = 0; ax < 3; ++ax) {
						trans[ax] = projectMaxToMin
							? static_cast<DefaultTriMesh::Scalar>(domMax[ax] - domMin[ax])
							: static_cast<DefaultTriMesh::Scalar>(-(domMax[ax] - domMin[ax]));
						if (projectMaxToMin ? !bf.isMaxBoundary(ax) : !bf.isMinBoundary(ax))
							trans[ax] = 0;
					}
					this->set_point(vh, newPt + trans);

					// 记录分裂产生的新半边
					if (this->point(this->to_vertex_handle(he)) == newPt) {
						heSplits.push_back(this->next_halfedge_handle(he));
					} else {
						heSplits.push_back(this->prev_halfedge_handle(he));
					}
					matched = true;
					break;
				}
			}
			(void)matched;
		}
		std::cerr << "[merge]   src=" << srcCount << " projected=" << projectedCount << "\n";
	};

	// max→min 投影
	projectAndSplit(true);
	vtag = classifyVertices();
	splittedHE.clear();

	// min→max 投影
	projectAndSplit(false);
	this->garbage_collection();
	vtag = classifyVertices();

	{
		int bndE = 0;
		for (auto e_it = this->edges_begin(); e_it != this->edges_end(); ++e_it)
			if (this->is_boundary(*e_it)) ++bndE;
		std::cerr << "[merge] before weld: nv=" << this->n_vertices() << " nf=" << this->n_faces()
				  << " bnd=" << bndE << "\n";
	}

	// ── Step 4: 顶点合并（周期性去重 + 重建网格） ──
	{
		// 提取 V, F
		const std::size_t nv = this->n_vertices();
		const std::size_t nf = this->n_faces();
		std::vector<std::array<double, 3>> verts(nv);
		for (auto v_it = this->vertices_begin(); v_it != this->vertices_end(); ++v_it) {
			const Vec3d& p = this->point(*v_it);
			verts[static_cast<std::size_t>((*v_it).idx())] = {
				static_cast<double>(p[0]), static_cast<double>(p[1]), static_cast<double>(p[2])};
		}
		std::vector<std::array<int, 3>> faces;
		faces.reserve(nf);
		for (auto f_it = this->faces_begin(); f_it != this->faces_end(); ++f_it) {
			auto fv = this->cfv_iter(*f_it);
			int a = (*fv).idx(); ++fv;
			int b = (*fv).idx(); ++fv;
			int c = (*fv).idx();
			faces.push_back({a, b, c});
		}

		// 周期去重
		PeriodicVertexGrid grid(
			{domMin[0], domMin[1], domMin[2]},
			{Lx, Ly, Lz},
			weldTol);

		std::vector<int> vmap(nv);
		for (std::size_t i = 0; i < nv; ++i) {
			vmap[i] = grid.insert(verts[i][0], verts[i][1], verts[i][2]);
		}

		// 重建面表，跳过退化三角形
		std::vector<std::array<int, 3>> newFaces;
		newFaces.reserve(nf);
		for (const auto& f : faces) {
			int fa = vmap[static_cast<std::size_t>(f[0])];
			int fb = vmap[static_cast<std::size_t>(f[1])];
			int fc = vmap[static_cast<std::size_t>(f[2])];
			if (fa == fb || fb == fc || fa == fc) continue;
			newFaces.push_back({fa, fb, fc});
		}

		// 构建 triangle-triangle adjacency 并 BFS 统一朝向
		const int nNewF = static_cast<int>(newFaces.size());
		const int nNewV = grid.counter;

		// 建边→面邻接
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

		// 构建 FF 邻接
		std::vector<std::array<int, 3>> FF(static_cast<std::size_t>(nNewF), {-1, -1, -1});
		for (auto& [ek, flist] : edge2faces) {
			if (flist.size() == 2) {
				FF[static_cast<std::size_t>(flist[0].first)][static_cast<std::size_t>(flist[0].second)] = flist[1].first;
				FF[static_cast<std::size_t>(flist[1].first)][static_cast<std::size_t>(flist[1].second)] = flist[0].first;
			}
		}

		// BFS 统一朝向
		std::vector<bool> visited(static_cast<std::size_t>(nNewF), false);
		std::queue<int> bfsQueue;
		// 可能有多个连通分量
		for (int startF = 0; startF < nNewF; ++startF) {
			if (visited[static_cast<std::size_t>(startF)]) continue;
			visited[static_cast<std::size_t>(startF)] = true;
			bfsQueue.push(startF);
			while (!bfsQueue.empty()) {
				const int fi = bfsQueue.front();
				bfsQueue.pop();
				const auto& srcF = newFaces[static_cast<std::size_t>(fi)];
				for (int ei = 0; ei < 3; ++ei) {
					int nb = FF[static_cast<std::size_t>(fi)][static_cast<std::size_t>(ei)];
					if (nb < 0 || visited[static_cast<std::size_t>(nb)]) continue;
					visited[static_cast<std::size_t>(nb)] = true;
					// 检查共享边在两个三角形中的朝向是否一致
					int sv0 = srcF[static_cast<std::size_t>(ei)];
					int sv1 = srcF[static_cast<std::size_t>((ei + 1) % 3)];
					auto& dstF = newFaces[static_cast<std::size_t>(nb)];
					// 在 dst 中找共享边顺序
					int ci0 = -1, ci1 = -1;
					for (int k = 0; k < 3; ++k) {
						if (dstF[static_cast<std::size_t>(k)] == sv0) ci0 = k;
						if (dstF[static_cast<std::size_t>(k)] == sv1) ci1 = k;
					}
					if (ci0 >= 0 && ci1 >= 0) {
						// 在正确朝向中，src 的 (v0,v1) 边在 dst 中应该是 (v1,v0)
						// 即 dst 中 sv1 应在 sv0 前面（模3），或说 (ci1+1)%3 == ci0
						if ((ci0 + 1) % 3 == ci1) {
							// 同向 -> 需要翻转 dst
							std::swap(dstF[static_cast<std::size_t>(ci0)], dstF[static_cast<std::size_t>(ci1)]);
							// 更新 FF（翻转后局部边索引变化，但 BFS 只用 visited，不需精确更新）
						}
					}
					bfsQueue.push(nb);
				}
			}
		}

		// 重建 OpenMesh
		this->clear();
		std::vector<VertexHandle> newVH;
		newVH.reserve(static_cast<std::size_t>(nNewV));
		for (int i = 0; i < nNewV; ++i) {
			const auto& pt = grid.points[static_cast<std::size_t>(i)];
			newVH.push_back(this->add_vertex(Vec3d(
				static_cast<DefaultTriMesh::Scalar>(pt[0]),
				static_cast<DefaultTriMesh::Scalar>(pt[1]),
				static_cast<DefaultTriMesh::Scalar>(pt[2]))));
		}
		for (const auto& f : newFaces) {
			auto fh = this->add_face(
				newVH[static_cast<std::size_t>(f[0])],
				newVH[static_cast<std::size_t>(f[1])],
				newVH[static_cast<std::size_t>(f[2])]);
			if (!fh.is_valid()) {
				// 尝试翻转朝向
				this->add_face(
					newVH[static_cast<std::size_t>(f[0])],
					newVH[static_cast<std::size_t>(f[2])],
					newVH[static_cast<std::size_t>(f[1])]);
			}
		}
		this->garbage_collection();

		int bndE2 = 0;
		for (auto e_it = this->edges_begin(); e_it != this->edges_end(); ++e_it)
			if (this->is_boundary(*e_it)) ++bndE2;
		std::cerr << "[merge] after weld: nv=" << this->n_vertices() << " nf=" << this->n_faces()
				  << " bnd=" << bndE2 << "\n";
	}
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
			if (pi < -1e-5) p[i] += static_cast<DefaultTriMesh::Scalar>(period);
			else if (pi > period + 1e-5) p[i] -= static_cast<DefaultTriMesh::Scalar>(period);
		}
		this->set_point(*v_it, p);
	}
}

// ──────────────────────────────────────────────────────────────────────────
// splitUnitCell（和 minsurf split_unit_cell 对齐）
// 找跨周期边界的边，在边界面处 split
// ──────────────────────────────────────────────────────────────────────────

void PeriodicTriMesh::splitUnitCell() {
	this->request_vertex_status();
	this->request_edge_status();
	this->request_halfedge_status();
	this->request_face_status();

	periodShift();

	const double hp[3] = {
		static_cast<double>(halfPeriod_[0]),
		static_cast<double>(halfPeriod_[1]),
		static_cast<double>(halfPeriod_[2])
	};
	const double L[3] = { 2.0 * hp[0], 2.0 * hp[1], 2.0 * hp[2] };

	// ── Phase 1: 在周期边界处 split 跨界边 ──
	for (int axis = 0; axis < 3; ++axis) {
		std::vector<EdgeHandle> toSplit;
		for (auto e_it = this->edges_begin(); e_it != this->edges_end(); ++e_it) {
			if (this->status(*e_it).deleted()) continue;
			HalfedgeHandle he = this->halfedge_handle(*e_it, 0);
			Vec3d p0 = this->point(this->from_vertex_handle(he));
			Vec3d p1 = p0 + wrapVector(this->point(this->to_vertex_handle(he)) - p0);

			double a0 = static_cast<double>(p0[axis]);
			double a1 = static_cast<double>(p1[axis]);

			if (std::abs(a0) < 1e-5 || std::abs(a0 - L[axis]) < 1e-5) continue;
			if (std::abs(a1) < 1e-5 || std::abs(a1 - L[axis]) < 1e-5) continue;
			if (a1 < -1e-5 || a1 > L[axis] + 1e-5) toSplit.push_back(*e_it);
		}

		for (auto eh : toSplit) {
			if (!eh.is_valid() || this->status(eh).deleted()) continue;
			HalfedgeHandle he = this->halfedge_handle(eh, 0);
			Vec3d p0 = this->point(this->from_vertex_handle(he));
			Vec3d p1 = p0 + wrapVector(this->point(this->to_vertex_handle(he)) - p0);
			double a0 = static_cast<double>(p0[axis]);
			double a1 = static_cast<double>(p1[axis]);
			double cut = (a1 > L[axis]) ? L[axis] : 0.0;
			double denom = a1 - a0;
			if (std::abs(denom) < 1e-15) continue;
			double t = (cut - a0) / denom;
			if (t <= 0.01 || t >= 0.99) continue;
			Vec3d c;
			for (int k = 0; k < 3; ++k)
				c[k] = static_cast<DefaultTriMesh::Scalar>(
					(1.0 - t) * static_cast<double>(p0[k]) + t * static_cast<double>(p1[k]));
			this->split(eh, c);
		}
		periodShift();
	}
	this->garbage_collection();

	// ── Phase 2: 像 saveUnitCell 一样 unwrap 面并 duplicate 边界顶点 ──
	// 收集所有顶点（wrap 到 [0,L)）
	const std::size_t nvOrig = this->n_vertices();
	std::vector<std::array<double, 3>> verts(nvOrig);
	for (auto v_it = this->vertices_begin(); v_it != this->vertices_end(); ++v_it) {
		const Vec3d& p = this->point(*v_it);
		auto& v = verts[static_cast<std::size_t>((*v_it).idx())];
		for (int i = 0; i < 3; ++i) {
			double pi = static_cast<double>(p[i]);
			while (pi < -1e-8) pi += L[i];
			while (pi > L[i] + 1e-8) pi -= L[i];
			v[static_cast<std::size_t>(i)] = pi;
		}
	}

	// 逐面 unwrap，记录需要 dup 的顶点
	struct Face3 { int v[3]; };
	std::vector<Face3> newFaces;
	newFaces.reserve(this->n_faces());
	std::vector<std::array<double, 3>> extraVerts;
	std::vector<int> extraOrigIdx; // 额外顶点对应的原始顶点

	for (auto f_it = this->faces_begin(); f_it != this->faces_end(); ++f_it) {
		auto fv = this->cfv_iter(*f_it);
		int idx[3];
		idx[0] = (*fv).idx(); ++fv;
		idx[1] = (*fv).idx(); ++fv;
		idx[2] = (*fv).idx();

		std::array<double, 3> p[3];
		p[0] = verts[static_cast<std::size_t>(idx[0])];
		for (int k = 1; k < 3; ++k) {
			p[k] = verts[static_cast<std::size_t>(idx[k])];
			for (int ax = 0; ax < 3; ++ax) {
				double diff = p[k][static_cast<std::size_t>(ax)] - p[0][static_cast<std::size_t>(ax)];
				if (diff > hp[ax]) p[k][static_cast<std::size_t>(ax)] -= L[ax];
				else if (diff < -hp[ax]) p[k][static_cast<std::size_t>(ax)] += L[ax];
			}
		}
		// shift 整个面回 [0, L)
		for (int ax = 0; ax < 3; ++ax) {
			double centroid = (p[0][static_cast<std::size_t>(ax)] +
							   p[1][static_cast<std::size_t>(ax)] +
							   p[2][static_cast<std::size_t>(ax)]) / 3.0;
			while (centroid < 0) {
				for (int k = 0; k < 3; ++k) p[k][static_cast<std::size_t>(ax)] += L[ax];
				centroid += L[ax];
			}
			while (centroid > L[ax]) {
				for (int k = 0; k < 3; ++k) p[k][static_cast<std::size_t>(ax)] -= L[ax];
				centroid -= L[ax];
			}
		}

		Face3 face;
		for (int k = 0; k < 3; ++k) {
			const auto& orig = verts[static_cast<std::size_t>(idx[k])];
			bool moved = false;
			for (int ax = 0; ax < 3; ++ax) {
				if (std::abs(p[k][static_cast<std::size_t>(ax)] - orig[static_cast<std::size_t>(ax)]) > 1e-8) {
					moved = true; break;
				}
			}
			if (moved) {
				face.v[k] = static_cast<int>(nvOrig + extraVerts.size());
				extraVerts.push_back(p[k]);
				extraOrigIdx.push_back(idx[k]);
			} else {
				face.v[k] = idx[k];
			}
		}
		newFaces.push_back(face);
	}

	// 重建网格
	this->clear();
	std::vector<VertexHandle> vh;
	vh.reserve(nvOrig + extraVerts.size());
	for (std::size_t i = 0; i < nvOrig; ++i) {
		vh.push_back(this->add_vertex(Vec3d(
			static_cast<DefaultTriMesh::Scalar>(verts[i][0]),
			static_cast<DefaultTriMesh::Scalar>(verts[i][1]),
			static_cast<DefaultTriMesh::Scalar>(verts[i][2]))));
	}
	for (const auto& ev : extraVerts) {
		vh.push_back(this->add_vertex(Vec3d(
			static_cast<DefaultTriMesh::Scalar>(ev[0]),
			static_cast<DefaultTriMesh::Scalar>(ev[1]),
			static_cast<DefaultTriMesh::Scalar>(ev[2]))));
	}
	for (const auto& f : newFaces) {
		this->add_face(vh[static_cast<std::size_t>(f.v[0])],
					   vh[static_cast<std::size_t>(f.v[1])],
					   vh[static_cast<std::size_t>(f.v[2])]);
	}
}

// ──────────────────────────────────────────────────────────────────────────
// saveUnitCell
// ──────────────────────────────────────────────────────────────────────────

bool PeriodicTriMesh::saveUnitCell(const std::string& filename) const {
	// 提取 V/F 列表，对跨周期面把顶点 unwrap 到同一侧，然后写 OBJ。
	const double L[3] = {
		2.0 * static_cast<double>(halfPeriod_[0]),
		2.0 * static_cast<double>(halfPeriod_[1]),
		2.0 * static_cast<double>(halfPeriod_[2])
	};
	const double hp[3] = {
		static_cast<double>(halfPeriod_[0]),
		static_cast<double>(halfPeriod_[1]),
		static_cast<double>(halfPeriod_[2])
	};

	// 收集原始顶点（shift 到 [0, L) 域内）
	const std::size_t nv = this->n_vertices();
	std::vector<std::array<double, 3>> verts(nv);
	for (auto v_it = this->vertices_begin(); v_it != this->vertices_end(); ++v_it) {
		const Vec3d& p = this->point(*v_it);
		auto& v = verts[static_cast<std::size_t>((*v_it).idx())];
		for (int i = 0; i < 3; ++i) {
			double pi = static_cast<double>(p[i]);
			// wrap 到 [0, L[i])
			while (pi < -1e-8) pi += L[i];
			while (pi > L[i] + 1e-8) pi -= L[i];
			v[static_cast<std::size_t>(i)] = pi;
		}
	}

	// 收集面，对跨周期面创建新的顶点副本
	struct Face3 { int v[3]; };
	std::vector<Face3> faces;
	faces.reserve(this->n_faces());
	std::vector<std::array<double, 3>> extraVerts; // 新增的顶点

	for (auto f_it = this->faces_begin(); f_it != this->faces_end(); ++f_it) {
		auto fv = this->cfv_iter(*f_it);
		int idx[3];
		idx[0] = (*fv).idx(); ++fv;
		idx[1] = (*fv).idx(); ++fv;
		idx[2] = (*fv).idx();

		// 以 v0 为锚点，把 v1、v2 unwrap 到 v0 附近
		std::array<double, 3> p[3];
		p[0] = verts[static_cast<std::size_t>(idx[0])];

		for (int k = 1; k < 3; ++k) {
			p[k] = verts[static_cast<std::size_t>(idx[k])];
			for (int ax = 0; ax < 3; ++ax) {
				double diff = p[k][static_cast<std::size_t>(ax)] - p[0][static_cast<std::size_t>(ax)];
				if (diff > hp[ax]) p[k][static_cast<std::size_t>(ax)] -= L[ax];
				else if (diff < -hp[ax]) p[k][static_cast<std::size_t>(ax)] += L[ax];
			}
		}

		// 将整个面 shift 回 [0, L) 域内：用重心决定偏移方向
		for (int ax = 0; ax < 3; ++ax) {
			double centroid = (p[0][static_cast<std::size_t>(ax)] +
							   p[1][static_cast<std::size_t>(ax)] +
							   p[2][static_cast<std::size_t>(ax)]) / 3.0;
			while (centroid < 0) {
				for (int k = 0; k < 3; ++k) p[k][static_cast<std::size_t>(ax)] += L[ax];
				centroid += L[ax];
			}
			while (centroid > L[ax]) {
				for (int k = 0; k < 3; ++k) p[k][static_cast<std::size_t>(ax)] -= L[ax];
				centroid -= L[ax];
			}
		}

		// 所有三个顶点都可能与原始位置不同，需要创建副本
		Face3 face;
		for (int k = 0; k < 3; ++k) {
			const auto& orig = verts[static_cast<std::size_t>(idx[k])];
			bool moved = false;
			for (int ax = 0; ax < 3; ++ax) {
				if (std::abs(p[k][static_cast<std::size_t>(ax)] - orig[static_cast<std::size_t>(ax)]) > 1e-8) {
					moved = true;
					break;
				}
			}
			if (moved) {
				face.v[k] = static_cast<int>(nv + extraVerts.size());
				extraVerts.push_back(p[k]);
			} else {
				face.v[k] = idx[k];
			}
		}
		faces.push_back(face);
	}

	// 写 OBJ
	DefaultTriMesh out;
	const std::size_t totalV = nv + extraVerts.size();
	std::vector<VertexHandle> vh;
	vh.reserve(totalV);
	for (std::size_t i = 0; i < nv; ++i) {
		vh.push_back(out.add_vertex(Vec3d(
			static_cast<DefaultTriMesh::Scalar>(verts[i][0]),
			static_cast<DefaultTriMesh::Scalar>(verts[i][1]),
			static_cast<DefaultTriMesh::Scalar>(verts[i][2]))));
	}
	for (const auto& ev : extraVerts) {
		vh.push_back(out.add_vertex(Vec3d(
			static_cast<DefaultTriMesh::Scalar>(ev[0]),
			static_cast<DefaultTriMesh::Scalar>(ev[1]),
			static_cast<DefaultTriMesh::Scalar>(ev[2]))));
	}
	for (const auto& f : faces) {
		out.add_face(vh[static_cast<std::size_t>(f.v[0])],
					 vh[static_cast<std::size_t>(f.v[1])],
					 vh[static_cast<std::size_t>(f.v[2])]);
	}

	return OpenMesh::IO::write_mesh(out, filename);
}

// ──────────────────────────────────────────────────────────────────────────
// surgery
// ──────────────────────────────────────────────────────────────────────────

bool PeriodicTriMesh::surgery(const SurgeryOptions& opts) {
	this->request_vertex_status();
	this->request_edge_status();
	this->request_halfedge_status();
	this->request_face_status();

	// ══════════════════════════════════════════════════════════
	// [1] 计算每个顶点的奇异度（和 minsurf compute_singularity_measure 对齐）
	// ══════════════════════════════════════════════════════════
	const std::size_t nv = this->n_vertices();
	auto geom = computeVertexGeometry(*this);
	std::vector<double> singMeasure(nv, 0.0);
	for (std::size_t i = 0; i < nv; ++i) {
		double H = geom.vrings[i].H;
		double K = geom.vrings[i].K;
		if (opts.surgeryType == 1) {
			singMeasure[i] = std::abs(H);
		} else {
			double disc = std::abs(H * H - K);
			double sqrtDisc = std::sqrt(disc);
			singMeasure[i] = std::max(std::abs(H + sqrtDisc), std::abs(H - sqrtDisc));
		}
	}

	{
		double maxH = 0, sumH = 0; int cnt = 0;
		for (std::size_t i = 0; i < nv; ++i) {
			sumH += singMeasure[i]; ++cnt;
			maxH = std::max(maxH, singMeasure[i]);
		}
		std::cerr << "[surgery] maxH=" << maxH << " avgH=" << (cnt > 0 ? sumH / cnt : 0)
				  << " threshold=" << opts.singularityTol << "\n";
	}

	// ══════════════════════════════════════════════════════════
	// [2] 标记奇异顶点 → 收集要删除的面 → 分连通片
	// （和 minsurf search_singular_strips 对齐）
	// ══════════════════════════════════════════════════════════
	std::set<int> toDeleteFaceSet;
	for (auto v_it = this->vertices_begin(); v_it != this->vertices_end(); ++v_it) {
		if (singMeasure[static_cast<std::size_t>((*v_it).idx())] > opts.singularityTol) {
			for (auto vf_it = this->cvf_iter(*v_it); vf_it.is_valid(); ++vf_it)
				toDeleteFaceSet.insert((*vf_it).idx());
		}
	}
	if (toDeleteFaceSet.empty()) return false;

	// ══════════════════════════════════════════════════════════
	// [3] expand_region: 用 Laplacian 扩展删除区域使边界更光滑
	// （和 minsurf expand_region 对齐）
	// ══════════════════════════════════════════════════════════
	{
		// 组装 L（cotangent Laplacian）
		auto L = assembleLaplacian(*this, geom.cotWeights);
		// L^T L 是双 Laplacian，非奇异
		Eigen::SparseMatrix<double> LtL = (L.transpose() * L).eval();
		Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver(LtL);

		// 将要删除的面分成连通片
		std::set<int> remaining = toDeleteFaceSet;
		std::vector<std::set<int>> patches;
		while (!remaining.empty()) {
			std::set<int> comp;
			std::queue<int> q;
			q.push(*remaining.begin());
			remaining.erase(remaining.begin());
			while (!q.empty()) {
				int fid = q.front(); q.pop();
				comp.insert(fid);
				auto fh = this->face_handle(fid);
				for (auto fh_it = this->cfh_iter(fh); fh_it.is_valid(); ++fh_it) {
					auto opp = this->opposite_halfedge_handle(*fh_it);
					if (this->is_boundary(opp)) continue;
					int adjF = this->face_handle(opp).idx();
					if (remaining.count(adjF)) {
						remaining.erase(adjF);
						q.push(adjF);
					}
				}
			}
			patches.push_back(comp);
		}

		// 对每个连通片做 Laplacian 扩展
		toDeleteFaceSet.clear();
		for (auto& patch : patches) {
			// rho: 标记奇异区域的指示函数
			Eigen::VectorXd rho = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(nv));
			for (int fi : patch) {
				auto fh = this->face_handle(fi);
				for (auto fv = this->cfv_iter(fh); fv.is_valid(); ++fv)
					rho[(*fv).idx()] = 1.0;
			}

			// 解 L^T L x = L^T M rho（M = vertex area）
			Eigen::VectorXd Mrho = geom.vertexAreas.cwiseProduct(rho);
			Eigen::VectorXd x = solver.solve((L.transpose() * Mrho).eval());
			x.array() -= geom.vertexAreas.dot(x) / geom.vertexAreas.sum();

			// 找 patch 区域内 x 的范围
			double xmin = 1e30, xmax = -1e30;
			for (int fi : patch) {
				auto fh = this->face_handle(fi);
				for (auto fv = this->cfv_iter(fh); fv.is_valid(); ++fv) {
					double xi = x[(*fv).idx()];
					xmin = std::min(xmin, xi);
					xmax = std::max(xmax, xi);
				}
			}

			// BFS 扩展：邻接面的对顶点 x 值在 [xmin, xmax] 内则加入
			std::set<int> pext;
			std::queue<int> bfs;
			std::vector<bool> inBfs(this->n_faces(), false);
			for (int fi : patch) { bfs.push(fi); inBfs[static_cast<std::size_t>(fi)] = true; }
			while (!bfs.empty()) {
				int fi = bfs.front(); bfs.pop();
				pext.insert(fi);
				auto fh = this->face_handle(fi);
				for (auto fh_it = this->cfh_iter(fh); fh_it.is_valid(); ++fh_it) {
					auto opp = this->opposite_halfedge_handle(*fh_it);
					if (this->is_boundary(opp)) continue;
					int adjF = this->face_handle(opp).idx();
					if (inBfs[static_cast<std::size_t>(adjF)]) continue;
					// 对面顶点
					auto vop = this->to_vertex_handle(this->next_halfedge_handle(opp));
					if (x[vop.idx()] >= xmin && x[vop.idx()] <= xmax) {
						bfs.push(adjF);
						inBfs[static_cast<std::size_t>(adjF)] = true;
					}
				}
			}

			// 去掉尖端面（只有一个邻接面在 pext 中的面）
			for (int fi : pext) {
				int adjCount = 0;
				auto fh = this->face_handle(fi);
				for (auto fh_it = this->cfh_iter(fh); fh_it.is_valid(); ++fh_it) {
					auto opp = this->opposite_halfedge_handle(*fh_it);
					if (!this->is_boundary(opp) && pext.count(this->face_handle(opp).idx()))
						++adjCount;
				}
				if (adjCount >= 2)
					toDeleteFaceSet.insert(fi);
			}
		}
	}

	// ══════════════════════════════════════════════════════════
	// [4] 删除面 → gc → 删孤岛
	// （和 minsurf: delete_face → gc → deleteisoisland 对齐）
	// ══════════════════════════════════════════════════════════
	for (int fi : toDeleteFaceSet)
		this->delete_face(this->face_handle(fi), false);
	this->garbage_collection();

	// deleteisoisland: 删除面数 < 最大连通片 * islandCullRatio 的碎片
	{
		const std::size_t nf2 = this->n_faces();
		std::vector<int> compId(nf2, -1);
		std::vector<int> compSize;
		int curComp = 0;
		for (std::size_t fi = 0; fi < nf2; ++fi) {
			if (compId[fi] >= 0) continue;
			std::queue<int> q;
			q.push(static_cast<int>(fi));
			compId[fi] = curComp;
			int sz = 0;
			while (!q.empty()) {
				int fid = q.front(); q.pop(); ++sz;
				auto fh = this->face_handle(fid);
				for (auto fh_it = this->cfh_iter(fh); fh_it.is_valid(); ++fh_it) {
					auto opp = this->opposite_halfedge_handle(*fh_it);
					if (this->is_boundary(opp)) continue;
					int adj = this->face_handle(opp).idx();
					if (adj >= 0 && static_cast<std::size_t>(adj) < nf2 && compId[static_cast<std::size_t>(adj)] < 0) {
						compId[static_cast<std::size_t>(adj)] = curComp;
						q.push(adj);
					}
				}
			}
			compSize.push_back(sz); ++curComp;
		}
		if (curComp > 1) {
			int maxComp = static_cast<int>(std::max_element(compSize.begin(), compSize.end()) - compSize.begin());
			int threshold = static_cast<int>(compSize[static_cast<std::size_t>(maxComp)] * opts.islandCullRatio);
			for (auto f_it = this->faces_begin(); f_it != this->faces_end(); ++f_it) {
				int cid = compId[static_cast<std::size_t>((*f_it).idx())];
				if (cid >= 0 && compSize[static_cast<std::size_t>(cid)] <= threshold)
					this->delete_face(*f_it, false);
			}
			this->garbage_collection();
		}
	}

	// ══════════════════════════════════════════════════════════
	// [5] fill_holes: 提取边界环 → 提取 4-ring 子网格 → CGAL 填洞
	//     → 选正确补丁 → bilaplacian fairing
	// （和 minsurf fill_one_hole + localSmoothing 对齐）
	// ══════════════════════════════════════════════════════════
	{
		std::vector<std::vector<HalfedgeHandle>> loops;
		{
			std::set<int> visited;
			for (auto he_it = this->halfedges_begin(); he_it != this->halfedges_end(); ++he_it) {
				if (!this->is_boundary(*he_it)) continue;
				if (visited.count((*he_it).idx())) continue;
				std::vector<HalfedgeHandle> loop;
				HalfedgeHandle he = *he_it, start = he;
				do { loop.push_back(he); visited.insert(he.idx()); he = this->next_halfedge_handle(he); }
				while (he != start && loop.size() < 10000);
				if (!loop.empty()) loops.push_back(loop);
			}
		}
		std::cerr << "[surgery] found " << loops.size() << " boundary loops\n";

		for (auto& loop : loops) {
			// 收集种子顶点
			std::set<int> seedVerts;
			for (auto he : loop) seedVerts.insert(this->from_vertex_handle(he).idx());

			int holeEdgeNum = static_cast<int>(loop.size());

			// 4-ring 扩展
			std::set<int> ringVerts = seedVerts;
			for (int kr = 0; kr < 4; ++kr) {
				std::set<int> toAdd;
				for (int vi : ringVerts)
					for (auto voh = this->cvoh_iter(VertexHandle(vi)); voh.is_valid(); ++voh)
						toAdd.insert(this->to_vertex_handle(*voh).idx());
				for (int v : toAdd) ringVerts.insert(v);
			}

			// 提取子网格面
			std::set<int> ringFaces;
			for (int vi : ringVerts)
				for (auto vf = this->cvf_iter(VertexHandle(vi)); vf.is_valid(); ++vf)
					ringFaces.insert((*vf).idx());

			// BFS sync 子网格坐标
			std::map<int, Vec3d> syncedPos;
			{
				int startV = *ringVerts.begin();
				syncedPos[startV] = this->point(VertexHandle(startV));
				std::queue<int> bfs;
				bfs.push(startV);
				std::set<int> bfsVisited;
				bfsVisited.insert(startV);
				while (!bfs.empty()) {
					int cur = bfs.front(); bfs.pop();
					for (auto voh = this->cvoh_iter(VertexHandle(cur)); voh.is_valid(); ++voh) {
						int nb = this->to_vertex_handle(*voh).idx();
						if (!ringVerts.count(nb) || bfsVisited.count(nb)) continue;
						syncedPos[nb] = syncedPos[cur] + wrapVector(
							this->point(VertexHandle(nb)) - this->point(VertexHandle(cur)));
						bfsVisited.insert(nb);
						bfs.push(nb);
					}
				}
			}

			// 构建 CGAL mesh
			using CGALMesh = CGAL::Surface_mesh<CgalPoint>;
			CGALMesh cmesh;
			std::map<int, CGALMesh::Vertex_index> omToSm;
			std::vector<int> smToOm;

			for (int vi : ringVerts) {
				auto it = syncedPos.find(vi);
				if (it == syncedPos.end()) continue;
				const Vec3d& p = it->second;
				auto smv = cmesh.add_vertex(CgalPoint(
					static_cast<double>(p[0]), static_cast<double>(p[1]), static_cast<double>(p[2])));
				omToSm[vi] = smv;
				smToOm.push_back(vi);
			}
			for (int fi : ringFaces) {
				auto fv = this->cfv_iter(this->face_handle(fi));
				int a = (*fv).idx(); ++fv; int b = (*fv).idx(); ++fv; int c = (*fv).idx();
				if (!omToSm.count(a) || !omToSm.count(b) || !omToSm.count(c)) continue;
				cmesh.add_face(omToSm[a], omToSm[b], omToSm[c]);
			}

			// CGAL 填洞
			namespace PMP = CGAL::Polygon_mesh_processing;

			std::vector<CGALMesh::Halfedge_index> borderCycles;
			PMP::extract_boundary_cycles(cmesh, std::back_inserter(borderCycles));

			// 记录填洞前的面数
			std::size_t cmeshFacesBefore = cmesh.number_of_faces();

			// 反复填洞直到只剩外边界
			int filledCount = 0;
			for (int fillPass = 0; fillPass < 20; ++fillPass) {
				borderCycles.clear();
				PMP::extract_boundary_cycles(cmesh, std::back_inserter(borderCycles));
				if (borderCycles.size() <= 1) break;

				int maxCycleLen = 0, maxCycleIdx = 0;
				for (int bi = 0; bi < static_cast<int>(borderCycles.size()); ++bi) {
					int cl = 0; auto hh = borderCycles[static_cast<std::size_t>(bi)];
					do { ++cl; hh = cmesh.next(hh); } while (hh != borderCycles[static_cast<std::size_t>(bi)] && cl < 100000);
					if (cl > maxCycleLen) { maxCycleLen = cl; maxCycleIdx = bi; }
				}
				int minCycleLen = 100000, minCycleIdx = -1;
				for (int bi = 0; bi < static_cast<int>(borderCycles.size()); ++bi) {
					if (bi == maxCycleIdx) continue;
					int cl = 0; auto hh = borderCycles[static_cast<std::size_t>(bi)];
					do { ++cl; hh = cmesh.next(hh); } while (hh != borderCycles[static_cast<std::size_t>(bi)] && cl < 100000);
					if (cl < minCycleLen) { minCycleLen = cl; minCycleIdx = bi; }
				}
				if (minCycleIdx < 0) break;

				auto h = borderCycles[static_cast<std::size_t>(minCycleIdx)];
				std::vector<CGALMesh::Face_index> pf;
				std::vector<CGALMesh::Vertex_index> pv;
				try {
					PMP::triangulate_refine_and_fair_hole(cmesh, h,
						std::back_inserter(pf), std::back_inserter(pv));
					++filledCount;
				} catch (...) {
					std::cerr << "[surgery] CGAL fill failed len=" << minCycleLen << "\n";
					break;
				}
			}
			std::cerr << "[surgery] hole filled=" << filledCount << "\n";

			// 将 CGAL submesh 的新面写回原网格
			// 记录 cmesh 中原始面的数量（填洞前已有的面）
			// 原始面 = ringFaces 中的面，用 cmesh face 的索引区分
			std::size_t origFaceCount = cmeshFacesBefore;
			std::set<int> patchFaceIds;
			int addOk = 0, addFail = 0;
			for (auto fi = cmesh.faces_begin(); fi != cmesh.faces_end(); ++fi) {
				// 跳过原始 k-ring 的面（它们已经在原网格中了）
				if (static_cast<std::size_t>((*fi).idx()) < origFaceCount) continue;

				auto hverts = cmesh.vertices_around_face(cmesh.halfedge(*fi));
				std::vector<int> fv;
				for (auto sv : hverts) {
					if (static_cast<std::size_t>(sv.idx()) < smToOm.size() && smToOm[static_cast<std::size_t>(sv.idx())] >= 0) {
						fv.push_back(smToOm[static_cast<std::size_t>(sv.idx())]);
					} else {
						auto p = cmesh.point(sv);
						VertexHandle newV = this->add_vertex(Vec3d(
							static_cast<DefaultTriMesh::Scalar>(p.x()),
							static_cast<DefaultTriMesh::Scalar>(p.y()),
							static_cast<DefaultTriMesh::Scalar>(p.z())));
						while (smToOm.size() <= static_cast<std::size_t>(sv.idx())) smToOm.push_back(-1);
						smToOm[static_cast<std::size_t>(sv.idx())] = newV.idx();
						fv.push_back(newV.idx());
					}
				}
				if (fv.size() == 3) {
					auto fh = this->add_face(VertexHandle(fv[0]), VertexHandle(fv[1]), VertexHandle(fv[2]));
					if (!fh.is_valid())
						fh = this->add_face(VertexHandle(fv[0]), VertexHandle(fv[2]), VertexHandle(fv[1]));
					if (fh.is_valid()) { patchFaceIds.insert(fh.idx()); ++addOk; }
					else ++addFail;
				}
			}
			if (addFail > 0)
				std::cerr << "[surgery] writeback: ok=" << addOk << " FAILED=" << addFail << "\n";

				// localSmoothing: bilaplacian fairing（和 minsurf 对齐）
				// 收集补丁+邻域面（扩展一层）
				if (!patchFaceIds.empty()) {
					std::set<int> smoothFaces;
					// 先把 seedVerts 的所有面加入（和 minsurf post_smooth_patch 对齐）
					for (int sv : seedVerts)
						for (auto vf = this->cvf_iter(VertexHandle(sv)); vf.is_valid(); ++vf)
							smoothFaces.insert((*vf).idx());
					for (int fi : patchFaceIds)
						smoothFaces.insert(fi);
					// 再扩展一层
					{
						std::set<int> toAdd;
						for (int fi : smoothFaces) {
							auto fh2 = this->face_handle(fi);
							for (auto fv2 = this->cfv_iter(fh2); fv2.is_valid(); ++fv2)
								for (auto vf2 = this->cvf_iter(*fv2); vf2.is_valid(); ++vf2)
									toAdd.insert((*vf2).idx());
						}
						for (int fi : toAdd) smoothFaces.insert(fi);
					}

					// 提取子网格
					std::map<int, int> subVMap;
					std::vector<int> subVInv;
					for (int fi : smoothFaces) {
						auto fh2 = this->face_handle(fi);
						for (auto fv2 = this->cfv_iter(fh2); fv2.is_valid(); ++fv2) {
							int vi = (*fv2).idx();
							if (!subVMap.count(vi)) { subVMap[vi] = static_cast<int>(subVInv.size()); subVInv.push_back(vi); }
						}
					}
					int snv = static_cast<int>(subVInv.size());
					Eigen::MatrixX3d subV(snv, 3);
					// BFS sync
					std::vector<bool> synced(static_cast<std::size_t>(snv), false);
					synced[0] = true;
					{
						auto sp = syncedPos.count(subVInv[0]) ? syncedPos[subVInv[0]] : this->point(VertexHandle(subVInv[0]));
						subV.row(0) << static_cast<double>(sp[0]), static_cast<double>(sp[1]), static_cast<double>(sp[2]);
					}
					std::queue<int> syncQ; syncQ.push(0);
					while (!syncQ.empty()) {
						int si = syncQ.front(); syncQ.pop();
						int omi = subVInv[static_cast<std::size_t>(si)];
						for (auto voh = this->cvoh_iter(VertexHandle(omi)); voh.is_valid(); ++voh) {
							int nb = this->to_vertex_handle(*voh).idx();
							if (!subVMap.count(nb)) continue;
							int sn = subVMap[nb];
							if (synced[static_cast<std::size_t>(sn)]) continue;
							Vec3d w = wrapVector(this->point(VertexHandle(nb)) - this->point(VertexHandle(omi)));
							subV.row(sn) = subV.row(si) + Eigen::RowVector3d(static_cast<double>(w[0]), static_cast<double>(w[1]), static_cast<double>(w[2]));
							synced[static_cast<std::size_t>(sn)] = true;
							syncQ.push(sn);
						}
					}
					Eigen::MatrixX3i subF(static_cast<int>(smoothFaces.size()), 3);
					int fc = 0;
					for (int fi : smoothFaces) {
						auto fh2 = this->face_handle(fi);
						auto fvi = this->cfv_iter(fh2);
						subF(fc, 0) = subVMap[(*fvi).idx()]; ++fvi;
						subF(fc, 1) = subVMap[(*fvi).idx()]; ++fvi;
						subF(fc, 2) = subVMap[(*fvi).idx()]; ++fc;
					}

					// 固定顶点：边界+邻居（和 minsurf localSmoothing 对齐）
					std::set<int> fixSet;
					for (int si = 0; si < snv; ++si) {
						int omi = subVInv[static_cast<std::size_t>(si)];
						if (this->is_boundary(VertexHandle(omi))) {
							fixSet.insert(si);
							for (auto voh = this->cvoh_iter(VertexHandle(omi)); voh.is_valid(); ++voh) {
								int nb = this->to_vertex_handle(*voh).idx();
								if (subVMap.count(nb)) fixSet.insert(subVMap[nb]);
							}
						}
					}
					Eigen::VectorXi fixDof(static_cast<int>(fixSet.size()));
					Eigen::MatrixX3d fixedY(static_cast<int>(fixSet.size()), 3);
					{ int i = 0; for (int si : fixSet) { fixDof[i] = si; fixedY.row(i) = subV.row(si); ++i; } }

					// Bilaplacian fairing × 3
					for (int fair = 0; fair < 3; ++fair) {
						Eigen::SparseMatrix<double> L, M;
						igl::cotmatrix(subV, subF, L);
						igl::massmatrix(subV, subF, igl::MASSMATRIX_TYPE_DEFAULT, M);
						Eigen::SparseMatrix<double> L2 = (L.transpose() * M.cwiseInverse() * L).eval();
						Eigen::SparseMatrix<double> Aeq(0, snv);
						igl::min_quad_with_fixed_data<double> data;
						if (!igl::min_quad_with_fixed_precompute(L2, fixDof, Aeq, true, data)) break;
						Eigen::MatrixXd B = Eigen::MatrixXd::Zero(snv, 3);
						Eigen::MatrixXd Beq(0, 3);
						if (!igl::min_quad_with_fixed_solve(data, B, fixedY, Beq, subV)) break;
						// 检查 NaN
						if (subV.hasNaN()) {
							std::cerr << "[surgery] bilaplacian produced NaN, reverting\n";
							// 恢复原始坐标
							for (int si = 0; si < snv; ++si) {
								int omi = subVInv[static_cast<std::size_t>(si)];
								auto sp = syncedPos.count(omi) ? syncedPos[omi] : this->point(VertexHandle(omi));
								subV.row(si) << static_cast<double>(sp[0]), static_cast<double>(sp[1]), static_cast<double>(sp[2]);
							}
							break;
						}
					}

					// 写回
					for (int si = 0; si < snv; ++si) {
						this->set_point(VertexHandle(subVInv[static_cast<std::size_t>(si)]), Vec3d(
							static_cast<DefaultTriMesh::Scalar>(subV(si, 0)),
							static_cast<DefaultTriMesh::Scalar>(subV(si, 1)),
							static_cast<DefaultTriMesh::Scalar>(subV(si, 2))));
					}
				}
			}
		}

	// ══════════════════════════════════════════════════════════
	// [6] 再次删孤岛（和 minsurf 对齐：fill_holes 后再 deleteisoisland）
	// ══════════════════════════════════════════════════════════
	{
		const std::size_t nf2 = this->n_faces();
		std::vector<int> compId(nf2, -1);
		std::vector<int> compSize;
		int curComp = 0;
		for (std::size_t fi = 0; fi < nf2; ++fi) {
			if (compId[fi] >= 0) continue;
			std::queue<int> q;
			q.push(static_cast<int>(fi)); compId[fi] = curComp; int sz = 0;
			while (!q.empty()) {
				int fid = q.front(); q.pop(); ++sz;
				auto fh = this->face_handle(fid);
				for (auto fh_it = this->cfh_iter(fh); fh_it.is_valid(); ++fh_it) {
					auto opp = this->opposite_halfedge_handle(*fh_it);
					if (this->is_boundary(opp)) continue;
					int adj = this->face_handle(opp).idx();
					if (adj >= 0 && static_cast<std::size_t>(adj) < nf2 && compId[static_cast<std::size_t>(adj)] < 0) {
						compId[static_cast<std::size_t>(adj)] = curComp; q.push(adj);
					}
				}
			}
			compSize.push_back(sz); ++curComp;
		}
		if (curComp > 1) {
			int maxComp = static_cast<int>(std::max_element(compSize.begin(), compSize.end()) - compSize.begin());
			int threshold = static_cast<int>(compSize[static_cast<std::size_t>(maxComp)] * opts.islandCullRatio);
			for (auto f_it = this->faces_begin(); f_it != this->faces_end(); ++f_it) {
				int cid = compId[static_cast<std::size_t>((*f_it).idx())];
				if (cid >= 0 && compSize[static_cast<std::size_t>(cid)] <= threshold)
					this->delete_face(*f_it, false);
			}
			this->garbage_collection();
		}
	}

	this->garbage_collection();
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
	Vec3d fundMax(static_cast<DefaultTriMesh::Scalar>(Lx), static_cast<DefaultTriMesh::Scalar>(Ly),
		static_cast<DefaultTriMesh::Scalar>(Lz));
	applyBBoxPadding(fundMin, fundMax, options.bboxPaddingWorld, options.bboxPaddingFraction);
	options.resolvedBBoxMin = {static_cast<double>(fundMin[0]), static_cast<double>(fundMin[1]), static_cast<double>(fundMin[2])};
	options.resolvedBBoxMax = {static_cast<double>(fundMax[0]), static_cast<double>(fundMax[1]), static_cast<double>(fundMax[2])};

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
				const std::array<double, 3> q = {static_cast<double>(i) * dx, static_cast<double>(j) * dy, static_cast<double>(k) * dz};
				phi[static_cast<size_t>(nodeIndex(i, j, k, Np))] = signedDistanceAt(q, tree, extVerts, extFaces);
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
		phi[static_cast<size_t>(id)] = sumPhi[static_cast<size_t>(r)] / static_cast<double>(cnt[static_cast<size_t>(r)]);
	}

	std::vector<LevelSetNode> nodes(static_cast<size_t>(nNodes));
	for (int k = 0; k < Np; ++k) {
		for (int j = 0; j < Np; ++j) {
			for (int i = 0; i < Np; ++i) {
				const int id = nodeIndex(i, j, k, Np);
				nodes[static_cast<size_t>(id)] = {static_cast<double>(i) * dx, static_cast<double>(j) * dy, static_cast<double>(k) * dz,
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
				voxels[{static_cast<std::int32_t>(i), static_cast<std::int32_t>(j), static_cast<std::int32_t>(k)}] = c;
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
