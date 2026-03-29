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

	const Vec3d hp = halfPeriod_;

	// [1] 计算每个顶点的奇异度
	const std::size_t nv = this->n_vertices();
	auto geom = computeVertexGeometry(*this);
	std::vector<double> singMeasure(nv, 0.0);
	for (std::size_t i = 0; i < nv; ++i) {
		double H = geom.vrings[i].H;
		double K = geom.vrings[i].K;
		if (opts.surgeryType == 1) {
			// type 1: |H|
			singMeasure[i] = std::abs(H);
		} else {
			// type 2: max(|κ₁|, |κ₂|)，其中 κ₁,κ₂ = H ± sqrt(H²-K)
			double disc = H * H - K;
			if (disc < 0) disc = 0;
			double sqrtDisc = std::sqrt(disc);
			singMeasure[i] = std::max(std::abs(H + sqrtDisc), std::abs(H - sqrtDisc));
		}
	}

	// 输出奇异度统计
	{
		double maxH = 0, sumH = 0;
		int cnt = 0;
		for (std::size_t i = 0; i < nv; ++i) {
			if (singMeasure[i] > 0) { sumH += singMeasure[i]; ++cnt; }
			maxH = std::max(maxH, singMeasure[i]);
		}
		std::cerr << "[surgery] maxH=" << maxH << " avgH=" << (cnt > 0 ? sumH / cnt : 0)
				  << " threshold=" << opts.singularityTol << "\n";
	}

	// [2] 标记奇异顶点及周围面
	std::vector<bool> deleteFace(this->n_faces(), false);
	bool anyDeleted = false;
	for (auto v_it = this->vertices_begin(); v_it != this->vertices_end(); ++v_it) {
		if (singMeasure[static_cast<std::size_t>((*v_it).idx())] > opts.singularityTol) {
			for (auto vf_it = this->cvf_iter(*v_it); vf_it.is_valid(); ++vf_it) {
				deleteFace[static_cast<std::size_t>((*vf_it).idx())] = true;
				anyDeleted = true;
			}
		}
	}

	if (!anyDeleted) return false;

	// [3] 删除标记的面
	for (auto f_it = this->faces_begin(); f_it != this->faces_end(); ++f_it) {
		if (deleteFace[static_cast<std::size_t>((*f_it).idx())]) {
			this->delete_face(*f_it, false);
		}
	}
	this->garbage_collection();

	// [4] 删除孤岛（保留最大连通分量）
	{
		// BFS 找连通分量
		const std::size_t nf2 = this->n_faces();
		std::vector<int> compId(nf2, -1);
		std::vector<int> compSize;
		int curComp = 0;

		// 建 face-face 邻接
		for (std::size_t fi = 0; fi < nf2; ++fi) {
			if (compId[fi] >= 0) continue;
			// BFS
			std::queue<int> q;
			q.push(static_cast<int>(fi));
			compId[fi] = curComp;
			int sz = 0;
			while (!q.empty()) {
				int fid = q.front(); q.pop();
				++sz;
				auto fh = this->face_handle(fid);
				for (auto fh_it = this->cfh_iter(fh); fh_it.is_valid(); ++fh_it) {
					HalfedgeHandle opp = this->opposite_halfedge_handle(*fh_it);
					if (this->is_boundary(opp)) continue;
					auto adjF = this->face_handle(opp);
					int adjIdx = adjF.idx();
					if (adjIdx >= 0 && static_cast<std::size_t>(adjIdx) < nf2 && compId[static_cast<std::size_t>(adjIdx)] < 0) {
						compId[static_cast<std::size_t>(adjIdx)] = curComp;
						q.push(adjIdx);
					}
				}
			}
			compSize.push_back(sz);
			++curComp;
		}

		if (curComp > 1) {
			int maxComp = static_cast<int>(std::max_element(compSize.begin(), compSize.end()) - compSize.begin());
			int threshold = static_cast<int>(compSize[static_cast<std::size_t>(maxComp)] * opts.islandCullRatio);
			for (auto f_it = this->faces_begin(); f_it != this->faces_end(); ++f_it) {
				int cid = compId[static_cast<std::size_t>((*f_it).idx())];
				if (cid >= 0 && compSize[static_cast<std::size_t>(cid)] <= threshold) {
					this->delete_face(*f_it, false);
				}
			}
			this->garbage_collection();
		}
	}

	// [5] 填洞：提取边界环 → 提取 k-ring 子网格 → CGAL 填洞
	{
		// 5a: 提取所有边界环
		std::vector<std::vector<HalfedgeHandle>> loops;
		{
			std::set<int> visited;
			for (auto he_it = this->halfedges_begin(); he_it != this->halfedges_end(); ++he_it) {
				if (!this->is_boundary(*he_it)) continue;
				if (visited.count((*he_it).idx())) continue;
				std::vector<HalfedgeHandle> loop;
				HalfedgeHandle he = *he_it;
				HalfedgeHandle start = he;
				do {
					loop.push_back(he);
					visited.insert(he.idx());
					he = this->next_halfedge_handle(he);
				} while (he != start && loop.size() < 10000);
				if (!loop.empty()) loops.push_back(loop);
			}
		}
		std::cerr << "[surgery] found " << loops.size() << " boundary loops\n";

		// 5b: 对每个边界环，提取 k-ring 邻域子网格并用 CGAL 填洞
		for (auto& loop : loops) {
			// 收集边界环的顶点
			std::set<int> seedVerts;
			for (auto he : loop) seedVerts.insert(this->from_vertex_handle(he).idx());

			// k-ring 扩展（k=4）
			std::set<int> ringVerts = seedVerts;
			for (int kr = 0; kr < 4; ++kr) {
				std::set<int> toAdd;
				for (int vi : ringVerts) {
					for (auto voh = this->cvoh_iter(VertexHandle(vi)); voh.is_valid(); ++voh)
						toAdd.insert(this->to_vertex_handle(*voh).idx());
				}
				for (int v : toAdd) ringVerts.insert(v);
			}

			// 提取子网格面
			std::set<int> ringFaces;
			for (int vi : ringVerts) {
				for (auto vf = this->cvf_iter(VertexHandle(vi)); vf.is_valid(); ++vf)
					ringFaces.insert((*vf).idx());
			}

			// 构建 CGAL Surface_mesh
			using CGALMesh = CGAL::Surface_mesh<CgalPoint>;
			CGALMesh cmesh;
			std::map<int, CGALMesh::Vertex_index> omToSm;
			std::vector<int> smToOm;

			// 先 BFS sync 子网格坐标（周期展开到连续）
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
						// sync 到 cur 附近
						Vec3d diff = this->point(VertexHandle(nb)) - this->point(VertexHandle(cur));
						Vec3d wrapped = wrapVector(diff);
						syncedPos[nb] = syncedPos[cur] + wrapped;
						bfsVisited.insert(nb);
						bfs.push(nb);
					}
				}
			}

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

			// CGAL 填洞：只填边数匹配原始边界环的洞（跳过 k-ring 外边界）
			namespace PMP = CGAL::Polygon_mesh_processing;
			std::vector<CGALMesh::Halfedge_index> borderCycles;
			PMP::extract_boundary_cycles(cmesh, std::back_inserter(borderCycles));

			int holeEdgeNum = static_cast<int>(loop.size());

			for (auto h : borderCycles) {
				// 数这个边界环的边数
				int cycleLen = 0;
				auto hh = h;
				do { ++cycleLen; hh = cmesh.next(hh); } while (hh != h && cycleLen < 100000);
				// 只填匹配原始洞大小的边界（跳过 k-ring 外边界）
				if (cycleLen > holeEdgeNum * 2) continue;

				std::vector<CGALMesh::Face_index> patchFaces;
				std::vector<CGALMesh::Vertex_index> patchVerts;
				try {
					PMP::triangulate_refine_and_fair_hole(cmesh, h,
						std::back_inserter(patchFaces),
						std::back_inserter(patchVerts));
				} catch (...) {
					std::cerr << "[surgery] CGAL hole fill failed for a boundary loop\n";
					continue;
				}

				// 将新面和顶点添加回原网格
				for (auto smv : patchVerts) {
					auto p = cmesh.point(smv);
					VertexHandle newV = this->add_vertex(Vec3d(
						static_cast<DefaultTriMesh::Scalar>(p.x()),
						static_cast<DefaultTriMesh::Scalar>(p.y()),
						static_cast<DefaultTriMesh::Scalar>(p.z())));
					omToSm[newV.idx()] = smv;
					while (smToOm.size() <= static_cast<std::size_t>(smv.idx()))
						smToOm.push_back(-1);
					smToOm[static_cast<std::size_t>(smv.idx())] = newV.idx();
				}

				std::set<int> patchFaceIds;
				for (auto smf : patchFaces) {
					auto hverts = cmesh.vertices_around_face(cmesh.halfedge(smf));
					std::vector<int> fv;
					for (auto sv : hverts) {
						if (static_cast<std::size_t>(sv.idx()) < smToOm.size() && smToOm[static_cast<std::size_t>(sv.idx())] >= 0)
							fv.push_back(smToOm[static_cast<std::size_t>(sv.idx())]);
					}
					if (fv.size() == 3) {
						auto fh = this->add_face(VertexHandle(fv[0]), VertexHandle(fv[1]), VertexHandle(fv[2]));
						if (!fh.is_valid())
							fh = this->add_face(VertexHandle(fv[0]), VertexHandle(fv[2]), VertexHandle(fv[1]));
						if (fh.is_valid()) patchFaceIds.insert(fh.idx());
					}
				}

				// Bilaplacian 光滑：提取补丁+邻域子网格，固定边界顶点做 fairing
				if (!patchFaceIds.empty()) {
					// 收集补丁+邻域面
					std::set<int> smoothFaces = patchFaceIds;
					for (int fi : patchFaceIds) {
						auto fh2 = this->face_handle(fi);
						for (auto fv2 = this->cfv_iter(fh2); fv2.is_valid(); ++fv2) {
							for (auto vf2 = this->cvf_iter(*fv2); vf2.is_valid(); ++vf2)
								smoothFaces.insert((*vf2).idx());
						}
					}

					// 提取子网格 V, F
					std::map<int, int> subVMap; // om vertex → sub index
					std::vector<int> subVInv;   // sub index → om vertex
					Eigen::MatrixX3d subV;
					Eigen::MatrixX3i subF;
					{
						for (int fi : smoothFaces) {
							auto fh2 = this->face_handle(fi);
							for (auto fv2 = this->cfv_iter(fh2); fv2.is_valid(); ++fv2) {
								int vi = (*fv2).idx();
								if (!subVMap.count(vi)) {
									subVMap[vi] = static_cast<int>(subVInv.size());
									subVInv.push_back(vi);
								}
							}
						}
						int snv = static_cast<int>(subVInv.size());
						subV.resize(snv, 3);
						// BFS sync 坐标到连续
						std::vector<bool> synced(static_cast<std::size_t>(snv), false);
						synced[0] = true;
						{
							int startOm = subVInv[0];
							auto sp = syncedPos.count(startOm) ? syncedPos[startOm] : this->point(VertexHandle(startOm));
							subV.row(0) << static_cast<double>(sp[0]), static_cast<double>(sp[1]), static_cast<double>(sp[2]);
						}
						std::queue<int> syncQ;
						syncQ.push(0);
						while (!syncQ.empty()) {
							int si = syncQ.front(); syncQ.pop();
							int omi = subVInv[static_cast<std::size_t>(si)];
							for (auto voh = this->cvoh_iter(VertexHandle(omi)); voh.is_valid(); ++voh) {
								int nb = this->to_vertex_handle(*voh).idx();
								if (!subVMap.count(nb)) continue;
								int sn = subVMap[nb];
								if (synced[static_cast<std::size_t>(sn)]) continue;
								Vec3d diff = this->point(VertexHandle(nb)) - this->point(VertexHandle(omi));
								Vec3d w = wrapVector(diff);
								subV.row(sn) = subV.row(si) + Eigen::RowVector3d(
									static_cast<double>(w[0]), static_cast<double>(w[1]), static_cast<double>(w[2]));
								synced[static_cast<std::size_t>(sn)] = true;
								syncQ.push(sn);
							}
						}
						subF.resize(static_cast<int>(smoothFaces.size()), 3);
						int fc = 0;
						for (int fi : smoothFaces) {
							auto fh2 = this->face_handle(fi);
							auto fvi = this->cfv_iter(fh2);
							subF(fc, 0) = subVMap[(*fvi).idx()]; ++fvi;
							subF(fc, 1) = subVMap[(*fvi).idx()]; ++fvi;
							subF(fc, 2) = subVMap[(*fvi).idx()];
							++fc;
						}
					}

					// 找固定顶点（子网格边界顶点 + 非补丁顶点）
					std::set<int> patchVertSet;
					for (int fi : patchFaceIds) {
						auto fh2 = this->face_handle(fi);
						for (auto fv2 = this->cfv_iter(fh2); fv2.is_valid(); ++fv2)
							patchVertSet.insert(subVMap[(*fv2).idx()]);
					}
					std::vector<int> fixedIdx;
					Eigen::MatrixX3d fixedY;
					for (int si = 0; si < static_cast<int>(subVInv.size()); ++si) {
						if (!patchVertSet.count(si)) fixedIdx.push_back(si);
					}
					Eigen::VectorXi fixDof(static_cast<int>(fixedIdx.size()));
					fixedY.resize(static_cast<int>(fixedIdx.size()), 3);
					for (int i = 0; i < static_cast<int>(fixedIdx.size()); ++i) {
						fixDof[i] = fixedIdx[static_cast<std::size_t>(i)];
						fixedY.row(i) = subV.row(fixedIdx[static_cast<std::size_t>(i)]);
					}

					// Bilaplacian fairing (3 iterations)
					for (int fair = 0; fair < 3; ++fair) {
						Eigen::SparseMatrix<double> L, M;
						igl::cotmatrix(subV, subF, L);
						igl::massmatrix(subV, subF, igl::MASSMATRIX_TYPE_DEFAULT, M);
						Eigen::SparseMatrix<double> L2 = (L.transpose() * M.cwiseInverse() * L).eval();

						Eigen::SparseMatrix<double> Aeq(0, static_cast<int>(subVInv.size()));
						igl::min_quad_with_fixed_data<double> data;
						if (!igl::min_quad_with_fixed_precompute(L2, fixDof, Aeq, true, data))
							break;
						Eigen::MatrixXd B = Eigen::MatrixXd::Zero(static_cast<int>(subVInv.size()), 3);
						Eigen::MatrixXd Beq(0, 3);
						if (!igl::min_quad_with_fixed_solve(data, B, fixedY, Beq, subV))
							break;
					}

					// 写回原网格
					for (int si = 0; si < static_cast<int>(subVInv.size()); ++si) {
						int omi = subVInv[static_cast<std::size_t>(si)];
						this->set_point(VertexHandle(omi), Vec3d(
							static_cast<DefaultTriMesh::Scalar>(subV(si, 0)),
							static_cast<DefaultTriMesh::Scalar>(subV(si, 1)),
							static_cast<DefaultTriMesh::Scalar>(subV(si, 2))));
					}
				}
			}
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
