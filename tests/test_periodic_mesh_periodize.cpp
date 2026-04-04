#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include <OpenMesh/Core/IO/MeshIO.hh>

#include "PeriodicMesh.h"
#include "MarchingCubes.h"
#include "PeriodicRemesh.h"
#include "AsymptoticConductivity.h"

#ifndef XTPMS_TEST_STL_PATH
#define XTPMS_TEST_STL_PATH R"(tests/data/test_input.stl)"
#endif
// 周期配对率下限；默认 0 仅检查读写与 periodize 非空。需要回归时可于编译期定义为 0.85 等。
#ifndef XTPMS_TEST_STL_MIN_PARTNER_RATIO
#define XTPMS_TEST_STL_MIN_PARTNER_RATIO 0.0
#endif

namespace {

using Vec3d = xtpms::PeriodicTriMesh::Vec3d;

void meshBBox(const xtpms::DefaultTriMesh& mesh, Vec3d& outMin, Vec3d& outMax) {
	auto it = mesh.vertices_begin();
	outMin = outMax = mesh.point(*it);
	for (++it; it != mesh.vertices_end(); ++it) {
		const auto& p = mesh.point(*it);
		for (int i = 0; i < 3; ++i) {
			outMin[i] = std::min(outMin[i], p[i]);
			outMax[i] = std::max(outMax[i], p[i]);
		}
	}
}

struct Cell3 {
	std::int64_t x{};
	std::int64_t y{};
	std::int64_t z{};
};

struct Cell3Hash {
	std::size_t operator()(const Cell3& c) const noexcept {
		std::size_t h = static_cast<std::size_t>(c.x) * 73856093u;
		h ^= static_cast<std::size_t>(c.y) * 19349663u + 0x9e3779b9u + (h << 6) + (h >> 2);
		h ^= static_cast<std::size_t>(c.z) * 83492791u + 0x9e3779b9u + (h << 6) + (h >> 2);
		return h;
	}
};

bool operator==(const Cell3& a, const Cell3& b) {
	return a.x == b.x && a.y == b.y && a.z == b.z;
}

Cell3 cellOf(const Vec3d& p, double invCell) {
	return {static_cast<std::int64_t>(std::llround(static_cast<double>(p[0]) * invCell)),
		static_cast<std::int64_t>(std::llround(static_cast<double>(p[1]) * invCell)),
		static_cast<std::int64_t>(std::llround(static_cast<double>(p[2]) * invCell))};
}

bool nearAnyFaceOfBox(const Vec3d& p, const Vec3d& boxMin, const Vec3d& boxMax, double tol) {
	for (int i = 0; i < 3; ++i) {
		const double pi = static_cast<double>(p[i]);
		if (std::abs(pi - static_cast<double>(boxMin[i])) <= tol) return true;
		if (std::abs(pi - static_cast<double>(boxMax[i])) <= tol) return true;
	}
	return false;
}

bool isAxisPeriodDisplacement(const Vec3d& d, int axis, double periodLen, double tol) {
	if (!(periodLen > 0.0)) return false;
	for (int i = 0; i < 3; ++i) {
		const double di = static_cast<double>(d[i]);
		if (i == axis) {
			if (std::abs(std::abs(di) - periodLen) > tol) return false;
		} else {
			if (std::abs(di) > tol) return false;
		}
	}
	return true;
}

void buildSpatialGrid(const std::vector<Vec3d>& pts, double cell,
	std::unordered_map<Cell3, std::vector<int>, Cell3Hash>& outGrid) {
	outGrid.clear();
	if (!(cell > 0.0)) return;
	const double inv = 1.0 / cell;
	for (int i = 0; i < static_cast<int>(pts.size()); ++i) {
		outGrid[cellOf(pts[i], inv)].push_back(i);
	}
}

bool findPartnerInCells(const Vec3d& p, const std::vector<Vec3d>& pts,
	const std::unordered_map<Cell3, std::vector<int>, Cell3Hash>& grid, int selfIdx,
	const std::array<double, 3>& periodLen, double tol, double invCell) {
	for (int ax = 0; ax < 3; ++ax) {
		const double L = periodLen[static_cast<size_t>(ax)];
		if (!(L > 0.0)) continue;
		for (const double s : {-1.0, 1.0}) {
			Vec3d target{};
			for (int i = 0; i < 3; ++i) {
				if (i == ax) {
					target[i] = static_cast<double>(p[i]) + s * L;
				} else {
					target[i] = static_cast<double>(p[i]);
				}
			}
			const Cell3 c0 = cellOf(target, invCell);
			for (int dx = -1; dx <= 1; ++dx) {
				for (int dy = -1; dy <= 1; ++dy) {
					for (int dz = -1; dz <= 1; ++dz) {
						const Cell3 c{c0.x + dx, c0.y + dy, c0.z + dz};
						const auto it = grid.find(c);
						if (it == grid.end()) continue;
						for (int j : it->second) {
							if (j == selfIdx) continue;
							Vec3d d{};
							for (int i = 0; i < 3; ++i) {
								d[i] = static_cast<double>(pts[static_cast<size_t>(j)][i]) -
									static_cast<double>(p[i]);
							}
							if (isAxisPeriodDisplacement(d, ax, L, tol)) return true;
						}
					}
				}
			}
		}
	}
	return false;
}

xtpms::DefaultTriMesh makeUnitCube() {
	using Point = xtpms::DefaultTriMesh::Point;
	using VH = xtpms::DefaultTriMesh::VertexHandle;
	xtpms::DefaultTriMesh mesh;
	std::array<VH, 8> v{};
	for (int i = 0; i < 8; ++i) {
		const double x = (i & 1) ? 1.0 : 0.0;
		const double y = (i & 2) ? 1.0 : 0.0;
		const double z = (i & 4) ? 1.0 : 0.0;
		v[static_cast<size_t>(i)] = mesh.add_vertex(Point(x, y, z));
	}
	mesh.add_face(v[0], v[1], v[2]);
	mesh.add_face(v[0], v[2], v[3]);
	mesh.add_face(v[4], v[6], v[5]);
	mesh.add_face(v[4], v[7], v[6]);
	mesh.add_face(v[0], v[3], v[7]);
	mesh.add_face(v[0], v[7], v[4]);
	mesh.add_face(v[1], v[5], v[6]);
	mesh.add_face(v[1], v[6], v[2]);
	mesh.add_face(v[0], v[4], v[5]);
	mesh.add_face(v[0], v[5], v[1]);
	mesh.add_face(v[3], v[2], v[6]);
	mesh.add_face(v[3], v[6], v[7]);
	return mesh;
}

xtpms::DefaultTriMesh makeOpenZQuad() {
	using Point = xtpms::DefaultTriMesh::Point;
	using VH = xtpms::DefaultTriMesh::VertexHandle;
	xtpms::DefaultTriMesh mesh;
	const VH a = mesh.add_vertex(Point(0.0, 0.0, 0.0));
	const VH b = mesh.add_vertex(Point(1.0, 0.0, 0.0));
	const VH c = mesh.add_vertex(Point(1.0, 1.0, 0.0));
	const VH d = mesh.add_vertex(Point(0.0, 1.0, 0.0));
	mesh.add_face(a, b, c);
	mesh.add_face(a, c, d);
	return mesh;
}

} // namespace

TEST(PeriodicMeshPeriodize, UnitCube_MergeNonEmpty) {
	const xtpms::DefaultTriMesh src = makeUnitCube();
	xtpms::PeriodicTriMesh mesh;
	mesh.setHalfPeriod(Vec3d(0.5, 0.5, 0.5));

	xtpms::PeriodizeOptions opt;
	opt.bboxPaddingWorld = 0.0;
	opt.bboxPaddingFraction = 0.0;

	mesh.periodizeFrom(src, opt);

	EXPECT_GT(mesh.n_vertices(), 0u);
	EXPECT_GT(mesh.n_faces(), 0u);
	EXPECT_LE(opt.resolvedBBoxMin[0], opt.resolvedBBoxMax[0]);
	EXPECT_LE(opt.resolvedBBoxMin[1], opt.resolvedBBoxMax[1]);
	EXPECT_LE(opt.resolvedBBoxMin[2], opt.resolvedBBoxMax[2]);
}

TEST(PeriodicMeshPeriodize, OpenQuad_MergeNonEmpty) {
	const xtpms::DefaultTriMesh src = makeOpenZQuad();
	xtpms::PeriodicTriMesh mesh;
	mesh.setHalfPeriod(Vec3d(0.5, 0.5, 0.5));

	xtpms::PeriodizeOptions opt;
	opt.bboxPaddingWorld = 0.0;
	opt.bboxPaddingFraction = 0.0;

	mesh.periodizeFrom(src, opt);

	EXPECT_GT(mesh.n_vertices(), 0u);
	EXPECT_GT(mesh.n_faces(), 0u);
}

// ──────────────────────────────────────────────────────────
// mergePeriodBoundary 单元测试
// ──────────────────────────────────────────────────────────

// 通用 TPMS MC 生成器
using LevelSetFunc = std::function<double(double, double, double)>;

xtpms::DefaultTriMesh makeTPMSMC(const Vec3d& halfPeriod, const LevelSetFunc& func, int res = 24) {
	using namespace xtpms;
	const double Lx = 2.0 * static_cast<double>(halfPeriod[0]);
	const double Ly = 2.0 * static_cast<double>(halfPeriod[1]);
	const double Lz = 2.0 * static_cast<double>(halfPeriod[2]);
	const int nx = res, ny = res, nz = res;
	const int Np = res + 1;
	const double dx = Lx / nx, dy = Ly / ny, dz = Lz / nz;

	std::vector<LevelSetNode> nodes(static_cast<std::size_t>(Np * Np * Np));
	for (int k = 0; k <= nz; ++k) {
		for (int j = 0; j <= ny; ++j) {
			for (int i = 0; i <= nx; ++i) {
				const double x = i * dx, y = j * dy, z = k * dz;
				const int id = i + Np * (j + Np * k);
				nodes[static_cast<std::size_t>(id)] = {x, y, z, func(x, y, z)};
			}
		}
	}

	SparseVoxelCornerMap voxels;
	for (int k = 0; k < nz; ++k) {
		for (int j = 0; j < ny; ++j) {
			for (int i = 0; i < nx; ++i) {
				VoxelCornerNodeIndices c{};
				c[0] = static_cast<std::size_t>(i + Np * (j + Np * k));
				c[1] = static_cast<std::size_t>((i + 1) + Np * (j + Np * k));
				c[2] = static_cast<std::size_t>((i + 1) + Np * ((j + 1) + Np * k));
				c[3] = static_cast<std::size_t>(i + Np * ((j + 1) + Np * k));
				c[4] = static_cast<std::size_t>(i + Np * (j + Np * (k + 1)));
				c[5] = static_cast<std::size_t>((i + 1) + Np * (j + Np * (k + 1)));
				c[6] = static_cast<std::size_t>((i + 1) + Np * ((j + 1) + Np * (k + 1)));
				c[7] = static_cast<std::size_t>(i + Np * ((j + 1) + Np * (k + 1)));
				voxels[{static_cast<std::int32_t>(i), static_cast<std::int32_t>(j), static_cast<std::int32_t>(k)}] = c;
			}
		}
	}

	DefaultTriMesh mesh;
	ExtractMarchingCubesOptions mcOpts;
	mcOpts.weldSharedEdges = true;
	marchingCubesExtractToTriMesh(voxels, nodes, 0.0, mesh, mcOpts);
	return mesh;
}

xtpms::DefaultTriMesh makeGyroidMC(const Vec3d& halfPeriod, int res = 24) {
	const double Lx = 2.0 * static_cast<double>(halfPeriod[0]);
	const double Ly = 2.0 * static_cast<double>(halfPeriod[1]);
	const double Lz = 2.0 * static_cast<double>(halfPeriod[2]);
	return makeTPMSMC(halfPeriod, [Lx, Ly, Lz](double x, double y, double z) {
		double px = 2.0 * M_PI * x / Lx, py = 2.0 * M_PI * y / Ly, pz = 2.0 * M_PI * z / Lz;
		return std::sin(px) * std::cos(py) + std::sin(py) * std::cos(pz) + std::sin(pz) * std::cos(px);
	}, res);
}

xtpms::DefaultTriMesh makeSchwarzPMC(const Vec3d& halfPeriod, int res = 24) {
	const double Lx = 2.0 * static_cast<double>(halfPeriod[0]);
	const double Ly = 2.0 * static_cast<double>(halfPeriod[1]);
	const double Lz = 2.0 * static_cast<double>(halfPeriod[2]);
	return makeTPMSMC(halfPeriod, [Lx, Ly, Lz](double x, double y, double z) {
		// Schwarz P: cos(2π x/Lx) + cos(2π y/Ly) + cos(2π z/Lz) = 0
		return std::cos(2.0 * M_PI * x / Lx) + std::cos(2.0 * M_PI * y / Ly) + std::cos(2.0 * M_PI * z / Lz);
	}, res);
}

int countBoundaryEdges(const xtpms::DefaultTriMesh& mesh) {
	int count = 0;
	for (auto e_it = mesh.edges_begin(); e_it != mesh.edges_end(); ++e_it) {
		if (mesh.is_boundary(*e_it)) ++count;
	}
	return count;
}

TEST(MergePeriodBoundary, GyroidEqualPeriod_NoBoundaryEdges) {
	// Gyroid MC 输出 -> mergePeriodBoundary 后不应有边界边
	const Vec3d hp(0.5, 0.5, 0.5);
	xtpms::DefaultTriMesh src = makeGyroidMC(hp, 16);
	ASSERT_GT(src.n_faces(), 0u);

	xtpms::PeriodicTriMesh mesh;
	mesh.setHalfPeriod(hp);
	// 直接 assign src 到 mesh（不经过 periodizeFrom，测试纯 merge）
	for (auto v_it = src.vertices_begin(); v_it != src.vertices_end(); ++v_it) {
		mesh.add_vertex(src.point(*v_it));
	}
	for (auto f_it = src.faces_begin(); f_it != src.faces_end(); ++f_it) {
		auto fv = src.cfv_iter(*f_it);
		int a = (*fv).idx(); ++fv; int b = (*fv).idx(); ++fv; int c = (*fv).idx();
		mesh.add_face(xtpms::PeriodicTriMesh::VertexHandle(a),
					  xtpms::PeriodicTriMesh::VertexHandle(b),
					  xtpms::PeriodicTriMesh::VertexHandle(c));
	}

	// MC 生成的网格在边界处不是周期拓扑
	int bndBefore = countBoundaryEdges(mesh);
	EXPECT_GT(bndBefore, 0) << "MC output should have boundary edges before merge";

	xtpms::MergeBoundaryOptions opts;
	mesh.mergePeriodBoundary(opts);

	// 输出 merge 前后的网格供目视检查
	OpenMesh::IO::write_mesh(src, "gyroid_equal_before_merge.obj");
	OpenMesh::IO::write_mesh(mesh, "gyroid_equal_after_merge.obj");

	int bndAfter = countBoundaryEdges(mesh);
	EXPECT_EQ(bndAfter, 0) << "After mergePeriodBoundary, mesh should be closed (no boundary edges)";
	EXPECT_GT(mesh.n_vertices(), 0u);
	EXPECT_GT(mesh.n_faces(), 0u);
}

TEST(MergePeriodBoundary, GyroidTriAxisPeriod_NoBoundaryEdges) {
	// 三轴不等周期
	const Vec3d hp(0.5, 0.7, 0.3);
	xtpms::DefaultTriMesh src = makeGyroidMC(hp, 16);
	ASSERT_GT(src.n_faces(), 0u);

	xtpms::PeriodicTriMesh mesh;
	mesh.setHalfPeriod(hp);
	for (auto v_it = src.vertices_begin(); v_it != src.vertices_end(); ++v_it) {
		mesh.add_vertex(src.point(*v_it));
	}
	for (auto f_it = src.faces_begin(); f_it != src.faces_end(); ++f_it) {
		auto fv = src.cfv_iter(*f_it);
		int a = (*fv).idx(); ++fv; int b = (*fv).idx(); ++fv; int c = (*fv).idx();
		mesh.add_face(xtpms::PeriodicTriMesh::VertexHandle(a),
					  xtpms::PeriodicTriMesh::VertexHandle(b),
					  xtpms::PeriodicTriMesh::VertexHandle(c));
	}

	OpenMesh::IO::write_mesh(src, "gyroid_triaxis_before_merge.obj");

	xtpms::MergeBoundaryOptions opts;
	mesh.mergePeriodBoundary(opts);

	OpenMesh::IO::write_mesh(mesh, "gyroid_triaxis_after_merge.obj");

	int bndAfter = countBoundaryEdges(mesh);
	EXPECT_EQ(bndAfter, 0) << "Tri-axis periodic mesh should be closed after merge";
	EXPECT_GT(mesh.n_faces(), 0u);
}

TEST(MergePeriodBoundary, GyroidPeriodizeAndMerge_Closed) {
	// 完整流程：Gyroid MC → periodizeFrom → mergePeriodBoundary
	// 用 MC 生成的 Gyroid 作为 periodizeFrom 的输入，模拟实际用例
	const Vec3d hp(0.5, 0.5, 0.5);
	xtpms::DefaultTriMesh src = makeGyroidMC(hp, 12);
	ASSERT_GT(src.n_faces(), 0u);

	xtpms::PeriodicTriMesh mesh;
	mesh.setHalfPeriod(hp);

	xtpms::PeriodizeOptions popt;
	popt.mcCellsPerAxis = 20;
	popt.mcVoxelDilateLayers = 3;
	mesh.periodizeFrom(src, popt);
	ASSERT_GT(mesh.n_faces(), 0u);

	xtpms::MergeBoundaryOptions mopt;
	mesh.mergePeriodBoundary(mopt);

	OpenMesh::IO::write_mesh(mesh, "gyroid_periodize_merge.obj");

	int bndAfter = countBoundaryEdges(mesh);
	EXPECT_EQ(bndAfter, 0) << "periodize + merge should produce a closed mesh";
	EXPECT_GT(mesh.n_faces(), 0u);
}

TEST(PeriodicMeshPeriodize, StlPeriodizeSymmetry) {
	namespace fs = std::filesystem;
	const fs::path stlPath(XTPMS_TEST_STL_PATH);
	if (!fs::exists(stlPath)) {
		GTEST_SKIP() << "STL not found: " << stlPath.string();
	}

	xtpms::DefaultTriMesh src;
	if (!OpenMesh::IO::read_mesh(src, stlPath.string())) {
		GTEST_FAIL() << "OpenMesh::IO::read_mesh failed: " << stlPath.string();
	}
	ASSERT_GT(src.n_vertices(), 0u);
	ASSERT_GT(src.n_faces(), 0u) << "STL has vertices but no faces";

	Vec3d mn{};
	Vec3d mx{};
	meshBBox(src, mn, mx);
	const double ex = static_cast<double>(mx[0]) - static_cast<double>(mn[0]);
	const double ey = static_cast<double>(mx[1]) - static_cast<double>(mn[1]);
	const double ez = static_cast<double>(mx[2]) - static_cast<double>(mn[2]);
	ASSERT_GT(ex, 0.0);
	ASSERT_GT(ey, 0.0);
	ASSERT_GT(ez, 0.0);

	xtpms::PeriodicTriMesh mesh;
	mesh.setHalfPeriod(Vec3d(0.5 * ex, 0.5 * ey, 0.5 * ez));

	xtpms::PeriodizeOptions opt;
	opt.bboxPaddingWorld = 0.0;
	opt.bboxPaddingFraction = 0.0;

	mesh.periodizeFrom(src, opt);

	ASSERT_GT(mesh.n_vertices(), 0u);

	const fs::path objPath = stlPath.parent_path() / (stlPath.stem().string() + ".periodized.obj");
	if (!OpenMesh::IO::write_mesh(mesh, objPath.string())) {
		GTEST_FAIL() << "OpenMesh::IO::write_mesh failed: " << objPath.string();
	}

	const Vec3d boxMin(opt.resolvedBBoxMin[0], opt.resolvedBBoxMin[1], opt.resolvedBBoxMin[2]);
	const Vec3d boxMax(opt.resolvedBBoxMax[0], opt.resolvedBBoxMax[1], opt.resolvedBBoxMax[2]);
	const double extent = std::max({ex, ey, ez});
	const double tol = std::max(1e-4, 1e-3 * extent);
	const std::array<double, 3> periodLen = {opt.resolvedBBoxMax[0] - opt.resolvedBBoxMin[0],
		opt.resolvedBBoxMax[1] - opt.resolvedBBoxMin[1],
		opt.resolvedBBoxMax[2] - opt.resolvedBBoxMin[2]};

	std::vector<xtpms::PeriodicTriMesh::VertexHandle> cand;
	cand.reserve(mesh.n_vertices());
	for (auto v_it = mesh.vertices_begin(); v_it != mesh.vertices_end(); ++v_it) {
		if (mesh.is_boundary(*v_it)) cand.push_back(*v_it);
	}
	if (cand.empty()) {
		for (auto v_it = mesh.vertices_begin(); v_it != mesh.vertices_end(); ++v_it) {
			if (nearAnyFaceOfBox(mesh.point(*v_it), boxMin, boxMax, tol)) {
				cand.push_back(*v_it);
			}
		}
	}
	ASSERT_FALSE(cand.empty()) << "no boundary or box-face vertices to check periodic pairing";

	std::vector<Vec3d> pts;
	pts.reserve(cand.size());
	for (const auto& vh : cand) {
		pts.push_back(mesh.point(vh));
	}

	const double cell = 4.0 * tol;
	const double invCell = 1.0 / cell;
	std::unordered_map<Cell3, std::vector<int>, Cell3Hash> grid;
	buildSpatialGrid(pts, cell, grid);

	int missing = 0;
	for (int i = 0; i < static_cast<int>(cand.size()); ++i) {
		if (!findPartnerInCells(pts[static_cast<size_t>(i)], pts, grid, i, periodLen, tol, invCell)) {
			++missing;
		}
	}

	const double ratio =
		1.0 - static_cast<double>(missing) / static_cast<double>(cand.size());
	EXPECT_GE(ratio, static_cast<double>(XTPMS_TEST_STL_MIN_PARTNER_RATIO))
		<< "periodic partners missing for " << missing << " / " << cand.size()
		<< " boundary/box-face vertices (obj: " << objPath.string() << ")";
}

// ──────────────────────────────────────────────────────────
// Delaunay Remesh 单元测试
// ──────────────────────────────────────────────────────────

xtpms::PeriodicTriMesh makeClosedTPMS(const Vec3d& hp, const xtpms::DefaultTriMesh& src) {
	xtpms::PeriodicTriMesh mesh;
	mesh.setHalfPeriod(hp);
	for (auto v_it = src.vertices_begin(); v_it != src.vertices_end(); ++v_it)
		mesh.add_vertex(src.point(*v_it));
	for (auto f_it = src.faces_begin(); f_it != src.faces_end(); ++f_it) {
		auto fv = src.cfv_iter(*f_it);
		int a = (*fv).idx(); ++fv; int b = (*fv).idx(); ++fv; int c = (*fv).idx();
		mesh.add_face(xtpms::PeriodicTriMesh::VertexHandle(a),
					  xtpms::PeriodicTriMesh::VertexHandle(b),
					  xtpms::PeriodicTriMesh::VertexHandle(c));
	}
	xtpms::MergeBoundaryOptions mopt;
	mesh.mergePeriodBoundary(mopt);
	return mesh;
}

xtpms::PeriodicTriMesh makeClosedGyroid(const Vec3d& hp, int res = 16) {
	return makeClosedTPMS(hp, makeGyroidMC(hp, res));
}

xtpms::PeriodicTriMesh makeClosedSchwarzP(const Vec3d& hp, int res = 16) {
	return makeClosedTPMS(hp, makeSchwarzPMC(hp, res));
}

TEST(DelaunayRemesh, GyroidRemeshPreservesClosed) {
	const Vec3d hp(0.5, 0.5, 0.5);
	auto mesh = makeClosedGyroid(hp);
	ASSERT_EQ(countBoundaryEdges(mesh), 0);
	ASSERT_GT(mesh.n_faces(), 0u);

	int facesBefore = static_cast<int>(mesh.n_faces());

	xtpms::RemeshOptions opts;
	opts.outerIter = 1;
	opts.innerIter = 3;
	xtpms::delaunayRemesh(mesh, opts);

	EXPECT_GT(mesh.n_vertices(), 0u);
	EXPECT_GT(mesh.n_faces(), 0u);

	OpenMesh::IO::write_mesh(mesh, "gyroid_after_remesh.obj");
}

// ──────────────────────────────────────────────────────────
// Remesh + Merge 管线诊断测试
// ──────────────────────────────────────────────────────────

TEST(PipelineDiag, RemeshThenMerge_StepByStep) {
	const Vec3d hp(0.5, 0.7, 0.4);
	auto mesh = makeClosedGyroid(hp, 16);

	std::cout << "[step0] after makeClosedGyroid: nv=" << mesh.n_vertices()
			  << " nf=" << mesh.n_faces()
			  << " boundary_edges=" << countBoundaryEdges(mesh) << "\n";
	ASSERT_EQ(countBoundaryEdges(mesh), 0) << "initial mesh should be closed";
	OpenMesh::IO::write_mesh(mesh, "diag_step0_closed.obj");

	// remesh 1 轮
	xtpms::RemeshOptions ropts;
	ropts.outerIter = 1;
	ropts.innerIter = 3;
	xtpms::delaunayRemesh(mesh, ropts);

	int bndAfterRemesh = countBoundaryEdges(mesh);
	std::cout << "[step1] after remesh: nv=" << mesh.n_vertices()
			  << " nf=" << mesh.n_faces()
			  << " boundary_edges=" << bndAfterRemesh << "\n";
	OpenMesh::IO::write_mesh(mesh, "diag_step1_after_remesh.obj");

	if (bndAfterRemesh > 0) {
		std::cout << "[step1] remesh introduced " << bndAfterRemesh
				  << " boundary edges — need merge\n";

		// 看边界顶点分布
		int bndVerts = 0;
		for (auto v_it = mesh.vertices_begin(); v_it != mesh.vertices_end(); ++v_it) {
			if (mesh.is_boundary(*v_it)) ++bndVerts;
		}
		std::cout << "[step1] boundary vertices: " << bndVerts << "\n";

		// 尝试 merge
		xtpms::MergeBoundaryOptions mopt;
		mesh.mergePeriodBoundary(mopt);

		int bndAfterMerge = countBoundaryEdges(mesh);
		std::cout << "[step2] after merge: nv=" << mesh.n_vertices()
				  << " nf=" << mesh.n_faces()
				  << " boundary_edges=" << bndAfterMerge << "\n";
		OpenMesh::IO::write_mesh(mesh, "diag_step2_after_merge.obj");
	} else {
		std::cout << "[step1] mesh still closed after remesh — no merge needed\n";
	}

	// 一步法向位移模拟
	for (auto v_it = mesh.vertices_begin(); v_it != mesh.vertices_end(); ++v_it) {
		auto p = mesh.point(*v_it);
		p[2] += 0.001; // 微小位移
		mesh.set_point(*v_it, p);
	}

	// 再次 remesh
	xtpms::delaunayRemesh(mesh, ropts);
	int bnd2 = countBoundaryEdges(mesh);
	std::cout << "[step3] after 2nd remesh: nv=" << mesh.n_vertices()
			  << " nf=" << mesh.n_faces()
			  << " boundary_edges=" << bnd2 << "\n";
	OpenMesh::IO::write_mesh(mesh, "diag_step3_after_remesh2.obj");

	EXPECT_GT(mesh.n_faces(), 0u);
}

TEST(PipelineDiag, RemeshSizeStability) {
	const Vec3d hp(0.5, 0.7, 0.4);
	auto mesh = makeClosedGyroid(hp, 16);
	int nv0 = static_cast<int>(mesh.n_vertices());
	int nf0 = static_cast<int>(mesh.n_faces());
	{
		int E0 = static_cast<int>(mesh.n_edges());
		std::cout << "initial: nv=" << nv0 << " nf=" << nf0 << " ne=" << E0
				  << " euler=" << (nv0 - E0 + nf0)
				  << " bnd=" << countBoundaryEdges(mesh) << "\n";
	}

	// 自动 targetLength
	xtpms::RemeshOptions ropts;
	ropts.outerIter = 1;
	ropts.innerIter = 3;
	xtpms::delaunayRemesh(mesh, ropts);
	std::cout << "auto remesh: nv=" << mesh.n_vertices() << " nf=" << mesh.n_faces()
			  << " bnd=" << countBoundaryEdges(mesh) << "\n";
	EXPECT_EQ(countBoundaryEdges(mesh), 0);
	OpenMesh::IO::write_mesh(mesh, "diag_remesh_auto.obj");

	// 固定 targetLength 连续 remesh 3 次，检查 Euler 守恒和大小稳定
	mesh = makeClosedGyroid(hp, 16);
	int eulerInit = static_cast<int>(mesh.n_vertices()) - static_cast<int>(mesh.n_edges()) + static_cast<int>(mesh.n_faces());
	std::cout << "Euler initial = " << eulerInit << "\n";
	ropts.targetLength = 0.08;
	for (int i = 0; i < 3; ++i) {
		xtpms::delaunayRemesh(mesh, ropts);
		int bnd = countBoundaryEdges(mesh);
		int V = static_cast<int>(mesh.n_vertices());
		int E = static_cast<int>(mesh.n_edges());
		int F = static_cast<int>(mesh.n_faces());
		int euler = V - E + F;
		std::cout << "fixed remesh #" << i << ": nv=" << V << " nf=" << F
				  << " euler=" << euler << " bnd=" << bnd << "\n";
		EXPECT_EQ(bnd, 0) << "remesh should preserve closed mesh";
		EXPECT_EQ(euler, eulerInit) << "remesh should preserve Euler characteristic";
	}
	OpenMesh::IO::write_mesh(mesh, "diag_remesh_fixed3x.obj");
}

TEST(PipelineDiag, RemeshBoundaryDiagnostic) {
	// 测试多种分辨率和周期组合下 remesh 是否产生边界边
	struct TestCase { Vec3d hp; int res; double targetLen; };
	std::vector<TestCase> cases = {
		{{0.5, 0.5, 0.5}, 12, 0.06},
		{{0.5, 0.5, 0.5}, 16, 0.04},
		{{0.5, 0.7, 0.4}, 12, 0.06},
		{{0.5, 0.7, 0.4}, 16, 0.05},
		{{0.3, 0.3, 0.3}, 16, 0.03},
	};

	for (std::size_t ci = 0; ci < cases.size(); ++ci) {
		auto& tc = cases[ci];
		auto mesh = makeClosedGyroid(tc.hp, tc.res);
		int bndBefore = countBoundaryEdges(mesh);
		int eulerBefore = static_cast<int>(mesh.n_vertices()) - static_cast<int>(mesh.n_edges()) + static_cast<int>(mesh.n_faces());
		ASSERT_EQ(bndBefore, 0) << "case " << ci << " initial mesh not closed";

		xtpms::RemeshOptions ropts;
		ropts.outerIter = 1;
		ropts.innerIter = 3;
		ropts.targetLength = tc.targetLen;
		xtpms::delaunayRemesh(mesh, ropts);

		int bndAfter = countBoundaryEdges(mesh);
		int eulerAfter = static_cast<int>(mesh.n_vertices()) - static_cast<int>(mesh.n_edges()) + static_cast<int>(mesh.n_faces());

		std::cout << "case " << ci << " hp=(" << tc.hp[0] << "," << tc.hp[1] << "," << tc.hp[2]
				  << ") res=" << tc.res << " tgtLen=" << tc.targetLen
				  << ": nv=" << mesh.n_vertices() << " nf=" << mesh.n_faces()
				  << " bnd=" << bndAfter << " euler=" << eulerAfter << "\n";

		if (bndAfter > 0) {
			// 诊断：输出网格供检查
			std::string fname = "diag_remesh_bnd_case" + std::to_string(ci) + ".obj";
			OpenMesh::IO::write_mesh(mesh, fname);
			std::cout << "  -> saved " << fname << " for inspection\n";

			// 统计边界信息
			int bndVerts = 0;
			for (auto v_it = mesh.vertices_begin(); v_it != mesh.vertices_end(); ++v_it) {
				if (mesh.is_boundary(*v_it)) ++bndVerts;
			}
			std::cout << "  -> boundary verts=" << bndVerts << "\n";
		}

		EXPECT_EQ(bndAfter, 0) << "case " << ci << " remesh should not produce boundary edges";
		EXPECT_EQ(eulerAfter, eulerBefore) << "case " << ci << " Euler should be preserved";
	}
}

TEST(PipelineDiag, RemeshVisualCheck) {
	const Vec3d hp(0.5, 0.7, 0.4);
	auto mesh = makeClosedGyroid(hp, 16);
	mesh.saveUnitCell("remesh_vis_step0.obj");

	// 计算初始平均边长
	double totalEdgeLen = 0;
	int edgeCnt = 0;
	for (auto e_it = mesh.edges_begin(); e_it != mesh.edges_end(); ++e_it) {
		auto he = mesh.halfedge_handle(*e_it, 0);
		auto d = mesh.point(mesh.to_vertex_handle(he)) - mesh.point(mesh.from_vertex_handle(he));
		auto w = mesh.wrapVector(d);
		totalEdgeLen += w.norm();
		++edgeCnt;
	}
	double avgLen = totalEdgeLen / edgeCnt;
	std::cout << "step0: nv=" << mesh.n_vertices() << " nf=" << mesh.n_faces()
			  << " avgEdgeLen=" << avgLen << "\n";

	xtpms::RemeshOptions ropts;
	ropts.outerIter = 1;
	ropts.innerIter = 3;
	ropts.targetLength = avgLen;
	ropts.debugOutputDir = ".";

	// 只跑一轮 remesh 来生成每个子步骤的 debug 网格
	for (int i = 0; i < 1; ++i) {
		auto t0 = std::chrono::high_resolution_clock::now();
		xtpms::delaunayRemesh(mesh, ropts);
		auto t1 = std::chrono::high_resolution_clock::now();
		double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
		mesh.saveUnitCell("remesh_vis_step" + std::to_string(i + 1) + ".obj");
		std::cout << "step" << (i + 1) << ": nv=" << mesh.n_vertices()
				  << " nf=" << mesh.n_faces()
				  << " bnd=" << countBoundaryEdges(mesh)
				  << " time=" << ms << "ms\n";
	}
	EXPECT_EQ(countBoundaryEdges(mesh), 0);
}

TEST(PipelineDiag, RemeshNonManifoldCheck) {
	const Vec3d hp(0.5, 0.5, 0.5);
	auto mesh = makeClosedGyroid(hp, 16);
	xtpms::RemeshOptions ropts;
	ropts.outerIter = 1;
	ropts.innerIter = 3;
	ropts.targetLength = 0.06;

	xtpms::delaunayRemesh(mesh, ropts);

	// 检查非流形顶点
	int nonManifoldV = 0;
	for (auto v_it = mesh.vertices_begin(); v_it != mesh.vertices_end(); ++v_it) {
		// OpenMesh 中非流形顶点的标志：边界顶点在封闭网格中不该存在
		if (mesh.is_boundary(*v_it)) ++nonManifoldV;
	}
	std::cout << "after remesh: nv=" << mesh.n_vertices() << " nf=" << mesh.n_faces()
			  << " non-manifold=" << nonManifoldV << " bnd=" << countBoundaryEdges(mesh) << "\n";
	EXPECT_EQ(nonManifoldV, 0);
	EXPECT_EQ(countBoundaryEdges(mesh), 0);
}

TEST(AsymptoticConductivity, SensitivityFiniteDifference) {
	// 在远离 TPMS 的周期曲面上验证 sensitivity
	// 用大扰动的 Schwarz P: cos(πx)+cos(πy)+cos(πz) + 0.4*sin(2πx)sin(πy) = 0
	const Vec3d hp(1.0, 1.0, 1.0);
	auto levelSet = [](double x, double y, double z) {
		double px = M_PI * x, py = M_PI * y, pz = M_PI * z;
		return std::cos(px) + std::cos(py) + std::cos(pz) + 0.4 * std::sin(2 * px) * std::sin(py);
	};
	auto src = makeTPMSMC(hp, levelSet, 40);  // 加密到 res=40
	auto mesh = makeClosedTPMS(hp, src);
	ASSERT_EQ(countBoundaryEdges(mesh), 0);

	// Remesh 改善网格质量
	for (int round = 0; round < 3; ++round) {
		xtpms::RemeshOptions ropts;
		ropts.outerIter = 2;
		ropts.innerIter = 5;
		xtpms::delaunayRemesh(mesh, ropts);
		bool hasBnd = false;
		for (auto e_it = mesh.edges_begin(); e_it != mesh.edges_end() && !hasBnd; ++e_it)
			if (mesh.is_boundary(*e_it)) hasBnd = true;
		if (hasBnd) mesh.mergePeriodBoundary();
	}
	ASSERT_EQ(countBoundaryEdges(mesh), 0);

	// 沿法向二分搜索投影到隐函数水平集
	{
		auto geomProj = xtpms::computeVertexGeometry(mesh);
		for (auto v_it = mesh.vertices_begin(); v_it != mesh.vertices_end(); ++v_it) {
			int vi = (*v_it).idx();
			Eigen::Vector3d p(
				static_cast<double>(mesh.point(*v_it)[0]),
				static_cast<double>(mesh.point(*v_it)[1]),
				static_cast<double>(mesh.point(*v_it)[2]));
			Eigen::Vector3d n = geomProj.vertexNormals[static_cast<std::size_t>(vi)];
			double f0 = levelSet(p[0], p[1], p[2]);
			// 二分搜索：找 t 使得 levelSet(p + t*n) = 0
			double tlo = -0.1, thi = 0.1;
			double flo = levelSet(p[0]+tlo*n[0], p[1]+tlo*n[1], p[2]+tlo*n[2]);
			double fhi = levelSet(p[0]+thi*n[0], p[1]+thi*n[1], p[2]+thi*n[2]);
			if (flo * fhi < 0) {
				for (int iter = 0; iter < 50; ++iter) {
					double tmid = (tlo + thi) / 2.0;
					double fmid = levelSet(p[0]+tmid*n[0], p[1]+tmid*n[1], p[2]+tmid*n[2]);
					if (fmid * flo < 0) { thi = tmid; fhi = fmid; }
					else { tlo = tmid; flo = fmid; }
				}
				double t = (tlo + thi) / 2.0;
				mesh.set_point(*v_it, Vec3d(
					static_cast<xtpms::DefaultTriMesh::Scalar>(p[0]+t*n[0]),
					static_cast<xtpms::DefaultTriMesh::Scalar>(p[1]+t*n[1]),
					static_cast<xtpms::DefaultTriMesh::Scalar>(p[2]+t*n[2])));
			} else if (std::abs(f0) > 1e-10) {
				// 如果二分区间内没有零点，用 Newton 步
				double gx = -M_PI*std::sin(M_PI*p[0]) + 0.4*2*M_PI*std::cos(2*M_PI*p[0])*std::sin(M_PI*p[1]);
				double gy = -M_PI*std::sin(M_PI*p[1]) + 0.4*std::sin(2*M_PI*p[0])*M_PI*std::cos(M_PI*p[1]);
				double gz = -M_PI*std::sin(M_PI*p[2]);
				Eigen::Vector3d grad(gx, gy, gz);
				double dn = grad.dot(n);
				if (std::abs(dn) > 1e-12) {
					double t = -f0 / dn;
					if (std::abs(t) < 0.1) {
						mesh.set_point(*v_it, Vec3d(
							static_cast<xtpms::DefaultTriMesh::Scalar>(p[0]+t*n[0]),
							static_cast<xtpms::DefaultTriMesh::Scalar>(p[1]+t*n[1]),
							static_cast<xtpms::DefaultTriMesh::Scalar>(p[2]+t*n[2])));
					}
				}
			}
		}
		// 检查投影质量
		double maxErr = 0;
		for (auto v_it = mesh.vertices_begin(); v_it != mesh.vertices_end(); ++v_it) {
			auto p = mesh.point(*v_it);
			double f = levelSet(static_cast<double>(p[0]), static_cast<double>(p[1]), static_cast<double>(p[2]));
			maxErr = std::max(maxErr, std::abs(f));
		}
		std::cout << "After projection: max level set error = " << maxErr << "\n";
	}

	// saveUnitCell with per-vertex scalar field
	auto saveUnitCellWithScalar = [](const xtpms::PeriodicTriMesh& m,
									 const Eigen::VectorXd& scalar,
									 const std::string& filename) {
		const double L[3] = {2.0*m.halfPeriod()[0], 2.0*m.halfPeriod()[1], 2.0*m.halfPeriod()[2]};
		const double hp[3] = {(double)m.halfPeriod()[0], (double)m.halfPeriod()[1], (double)m.halfPeriod()[2]};
		const int nv = (int)m.n_vertices();

		// 原始顶点 wrap 到 [0,L)
		std::vector<std::array<double,3>> verts(nv);
		for (auto v = m.vertices_begin(); v != m.vertices_end(); ++v) {
			auto& p = m.point(*v);
			auto& vv = verts[(*v).idx()];
			for (int a = 0; a < 3; ++a) {
				double pi = (double)p[a];
				while (pi < -1e-8) pi += L[a];
				while (pi > L[a]+1e-8) pi -= L[a];
				vv[a] = pi;
			}
		}

		// 逐面 unwrap，记录 dup 顶点和它的原始索引
		struct F3 { int v[3]; };
		std::vector<F3> faces;
		std::vector<std::array<double,3>> extraVerts;
		std::vector<int> extraOrigIdx;  // 额外顶点对应的原始顶点索引

		for (auto f = m.faces_begin(); f != m.faces_end(); ++f) {
			auto fv = m.cfv_iter(*f);
			int idx[3]; idx[0]=(*fv).idx(); ++fv; idx[1]=(*fv).idx(); ++fv; idx[2]=(*fv).idx();
			std::array<double,3> p[3];
			p[0] = verts[idx[0]];
			for (int k = 1; k < 3; ++k) {
				p[k] = verts[idx[k]];
				for (int a = 0; a < 3; ++a) {
					double d = p[k][a] - p[0][a];
					if (d > hp[a]) p[k][a] -= L[a];
					else if (d < -hp[a]) p[k][a] += L[a];
				}
			}
			for (int a = 0; a < 3; ++a) {
				double c = (p[0][a]+p[1][a]+p[2][a])/3.0;
				while (c < 0) { for (int k=0;k<3;++k) p[k][a]+=L[a]; c+=L[a]; }
				while (c > L[a]) { for (int k=0;k<3;++k) p[k][a]-=L[a]; c-=L[a]; }
			}
			F3 face;
			for (int k = 0; k < 3; ++k) {
				bool moved = false;
				for (int a = 0; a < 3; ++a)
					if (std::abs(p[k][a] - verts[idx[k]][a]) > 1e-8) { moved = true; break; }
				if (moved) {
					face.v[k] = nv + (int)extraVerts.size();
					extraVerts.push_back(p[k]);
					extraOrigIdx.push_back(idx[k]);
				} else {
					face.v[k] = idx[k];
				}
			}
			faces.push_back(face);
		}

		// 写 OBJ：v x y z r g b (vertex color = scalar mapped to colormap)
		// 先算 scalar 的范围
		double smin = scalar.minCoeff(), smax = scalar.maxCoeff();
		double srange = smax - smin;
		if (srange < 1e-30) srange = 1.0;
		auto mapColor = [&](double s) -> std::array<double,3> {
			double t = (s - smin) / srange; // [0,1]
			// blue-white-red colormap
			if (t < 0.5) return {2*t, 2*t, 1.0};
			else return {1.0, 2*(1-t), 2*(1-t)};
		};

		std::ofstream ofs(filename);
		ofs << "# scalar range: " << smin << " " << smax << "\n";
		int totalV = nv + (int)extraVerts.size();
		for (int i = 0; i < nv; ++i) {
			auto c = mapColor(scalar[i]);
			ofs << "v " << verts[i][0] << " " << verts[i][1] << " " << verts[i][2]
				<< " " << c[0] << " " << c[1] << " " << c[2] << "\n";
		}
		for (int i = 0; i < (int)extraVerts.size(); ++i) {
			auto c = mapColor(scalar[extraOrigIdx[i]]);
			ofs << "v " << extraVerts[i][0] << " " << extraVerts[i][1] << " " << extraVerts[i][2]
				<< " " << c[0] << " " << c[1] << " " << c[2] << "\n";
		}
		for (auto& f : faces)
			ofs << "f " << f.v[0]+1 << " " << f.v[1]+1 << " " << f.v[2]+1 << "\n";

		// 同时输出 raw scalar txt（和顶点一一对应）
		std::string txtFile = filename.substr(0, filename.rfind('.')) + "_scalar.txt";
		std::ofstream tfs(txtFile);
		tfs << "# vertex_index original_index x y z scalar\n";
		for (int i = 0; i < nv; ++i)
			tfs << i << " " << i << " " << verts[i][0] << " " << verts[i][1] << " " << verts[i][2] << " " << scalar[i] << "\n";
		for (int i = 0; i < (int)extraVerts.size(); ++i)
			tfs << nv+i << " " << extraOrigIdx[i] << " " << extraVerts[i][0] << " " << extraVerts[i][1] << " " << extraVerts[i][2] << " " << scalar[extraOrigIdx[i]] << "\n";
	};

	auto geom = xtpms::computeVertexGeometry(mesh);
	Eigen::MatrixX3d ulist;
	Eigen::Matrix3d kA = xtpms::solveAsymptoticConductivity(mesh, geom, ulist);
	double As = geom.vertexAreas.sum();
	const int nv = static_cast<int>(mesh.n_vertices());

	std::cout << "Perturbed surface: nv=" << nv << " nf=" << mesh.n_faces() << "\n";
	std::cout << "kA =\n" << kA << "\n";
	std::cout << "APAC = " << kA.trace() / 3.0 << "\n\n";

	// 计算 sensitivity
	auto sens = xtpms::computeSensitivity(mesh, geom, ulist);
	auto kAv = xtpms::toVoigt(kA);
	kAv.tail<3>() /= 2.0;
	for (int i = 0; i < nv; ++i) {
		double ai = geom.vertexAreas[i];
		if (ai > 1e-15) { sens.vSens.row(i) /= ai; sens.aSens[i] /= ai; }
	}
	// 测试 APAC 和 k00 两种目标
	struct ObjCase { std::string name; std::string type; };
	std::vector<ObjCase> objCases = {{"APAC", "apac"}, {"k00", "k00"}};

	for (const auto& oc : objCases) {
		auto objRes = xtpms::evaluateADCObjective(oc.type, kA);
		Eigen::VectorXd dfdvn = (sens.vSens / As - sens.aSens * kAv.transpose() / As) * (-objRes.gradient);

		// 输出 sensitivity 场到网格
		saveUnitCellWithScalar(mesh, dfdvn, "perturbed_tpms_sens_" + oc.type + ".obj");
		std::cout << "Saved perturbed_tpms_sens_" << oc.type << ".obj"
				  << "  dfdvn: min=" << dfdvn.minCoeff() << " max=" << dfdvn.maxCoeff() << "\n";

		// vn 场：使用光滑三周期函数
		struct VnCase { std::string name; std::function<double(double,double,double)> fn; };
		std::vector<VnCase> vnCases = {
			{"sin*sin*sin", [](double x, double y, double z) {
				return std::sin(M_PI*x)*std::sin(M_PI*y)*std::sin(M_PI*z); }},
			{"cos(2pi*x)", [](double x, double y, double z) {
				(void)y; (void)z; return std::cos(2*M_PI*x); }},
			{"uniform", [](double, double, double) { return 1.0; }},
		};

		double obj0 = xtpms::evaluateADCObjective(oc.type, kA).value;
		std::cout << "=== Objective: " << oc.name << " = " << obj0 << " ===\n";

		for (const auto& vc : vnCases) {
			Eigen::VectorXd vn(nv);
			for (int i = 0; i < nv; ++i) {
				auto pt = mesh.point(xtpms::PeriodicTriMesh::VertexHandle(i));
				vn[i] = vc.fn(static_cast<double>(pt[0]), static_cast<double>(pt[1]), static_cast<double>(pt[2]));
			}

			double dk_formula = 0;
			for (int i = 0; i < nv; ++i)
				dk_formula += vn[i] * dfdvn[i] * geom.vertexAreas[i];

			auto perturbObj = [&](double s) {
				xtpms::PeriodicTriMesh copy = mesh;
				for (auto v_it = copy.vertices_begin(); v_it != copy.vertices_end(); ++v_it) {
					int vi = (*v_it).idx();
					Eigen::Vector3d p(
						static_cast<double>(copy.point(*v_it)[0]),
						static_cast<double>(copy.point(*v_it)[1]),
						static_cast<double>(copy.point(*v_it)[2]));
					Eigen::Vector3d n = geom.vertexNormals[static_cast<std::size_t>(vi)];
					p += s * vn[vi] * n;
					copy.set_point(*v_it, Vec3d(
						static_cast<xtpms::DefaultTriMesh::Scalar>(p[0]),
						static_cast<xtpms::DefaultTriMesh::Scalar>(p[1]),
						static_cast<xtpms::DefaultTriMesh::Scalar>(p[2])));
				}
				auto g = xtpms::computeVertexGeometry(copy);
				Eigen::MatrixX3d u;
				Eigen::Matrix3d k = xtpms::solveAsymptoticConductivity(copy, g, u);
				return xtpms::evaluateADCObjective(oc.type, k).value;
			};

			std::cout << "  vn=" << vc.name << "  formula=" << dk_formula << "\n";
			for (double stepFD : {1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 1e-4}) {
				double objP = perturbObj(stepFD);
				double objM = perturbObj(-stepFD);
				double dk_meshFD = (objP - objM) / (2.0 * stepFD);
				std::cout << "    step=" << stepFD
						  << "  obj+=" << objP << "  obj-=" << objM
						  << "  meshFD=" << dk_meshFD << "\n";
			}
		}
	}
}

// ──────────────────────────────────────────────────────────
// Revolution surface 旋转体辅助
// ──────────────────────────────────────────────────────────

// 旋转体 profile: R(x) 是关于 x 的周期函数，绕 x 轴旋转
// xtpms 域 [0, 2*hp]^3, 物理域 [-hp, hp]
// level set: y² + z² - R(x)² = 0
struct RevolutionProfile {
	using Func = std::function<double(double)>;
	Func R;       // radius profile R(x), x ∈ [-hp, hp]
	Func dR;      // R'(x)

	// 解析 k11: 4 / I1 / I2
	// I1 = ∫_{-hp}^{hp} sqrt(1 + R'(x)²) / R(x) dx
	// I2 = ∫_{-hp}^{hp} R(x) * sqrt(1 + R'(x)²) dx
	double analyticK11(double hp, int nquad = 10000) const {
		double I1 = 0, I2 = 0;
		double dx = 2.0 * hp / nquad;
		for (int i = 0; i < nquad; ++i) {
			double x = -hp + (i + 0.5) * dx;
			double r = R(x);
			double dr = dR(x);
			double ds = std::sqrt(1.0 + dr * dr);
			I1 += ds / r * dx;
			I2 += r * ds * dx;
		}
		return 4.0 / I1 / I2;
	}

	// dk11 的泛函一阶变分（解析公式 + 数值积分）
	// 给定轴对称法向速度 vn(x)，δR(x) = vn(x) / sqrt(1+R'²)
	//
	// k11 = 4 / (I1 * I2)
	// δk11 = -k11 * (δI1/I1 + δI2/I2)
	//
	// 对泛函 I = ∫ f(R, R') dx 的一阶变分:
	// δI = ∫ [∂f/∂R - d/dx(∂f/∂R')] * δR dx  (Euler-Lagrange)
	//
	// I1 的被积函数 f1 = s/R, s = sqrt(1+R'²):
	//   ∂f1/∂R = -s/R²
	//   ∂f1/∂R' = R'/(R*s)
	//   EL1(x) = -s/R² - d/dx[R'/(R*s)]
	//
	// I2 的被积函数 f2 = R*s:
	//   ∂f2/∂R = s
	//   ∂f2/∂R' = R*R'/s
	//   EL2(x) = s - d/dx[R*R'/s]
	//
	double variationalDk11(double hp, const Func& vn, int nquad = 100000) const {
		double dx = 2.0 * hp / nquad;
		double I1 = 0, I2 = 0;

		// d/dx 项用数值微分
		auto EL_term_R = [&](double x, double hd) {
			// ∂f1/∂R' 在 x 处的值: R'/(R*s)
			auto g1 = [&](double xx) {
				double rr = R(xx), dr = dR(xx);
				double ss = std::sqrt(1.0 + dr * dr);
				return dr / (rr * ss);
			};
			// ∂f2/∂R' 在 x 处的值: R*R'/s
			auto g2 = [&](double xx) {
				double rr = R(xx), dr = dR(xx);
				double ss = std::sqrt(1.0 + dr * dr);
				return rr * dr / ss;
			};
			double dg1dx = (g1(x + hd) - g1(x - hd)) / (2.0 * hd);
			double dg2dx = (g2(x + hd) - g2(x - hd)) / (2.0 * hd);
			return std::make_pair(dg1dx, dg2dx);
		};

		double dI1 = 0, dI2 = 0;
		double hd = dx * 0.5; // 数值微分步长
		for (int i = 0; i < nquad; ++i) {
			double x = -hp + (i + 0.5) * dx;
			double r = R(x), dr = dR(x);
			double s = std::sqrt(1.0 + dr * dr);
			double deltaR = vn(x) / s; // δR(x) = vn(x) / sqrt(1+R'²)

			I1 += s / r * dx;
			I2 += r * s * dx;

			auto [dg1dx, dg2dx] = EL_term_R(x, hd);

			// EL1 = ∂f1/∂R - d/dx(∂f1/∂R') = -s/R² - dg1dx
			double EL1 = -s / (r * r) - dg1dx;
			// EL2 = ∂f2/∂R - d/dx(∂f2/∂R') = s - dg2dx
			double EL2 = s - dg2dx;

			dI1 += EL1 * deltaR * dx;
			dI2 += EL2 * deltaR * dx;
		}

		double k11 = 4.0 / I1 / I2;
		return -k11 * (dI1 / I1 + dI2 / I2);
	}

	// 对比用：1D FD 验证变分公式
	double fdDk11(double hp, const Func& vn, double eps = 1e-7, int nquad = 100000) const {
		auto k11_at_eps = [&](double e) {
			auto R_e = [&](double x) {
				double dr = dR(x);
				return R(x) + e * vn(x) / std::sqrt(1.0 + dr * dr);
			};
			double I1 = 0, I2 = 0;
			double dx = 2.0 * hp / nquad;
			double hd = dx * 0.5;
			for (int i = 0; i < nquad; ++i) {
				double x = -hp + (i + 0.5) * dx;
				double r = R_e(x);
				// R_e' 用数值微分
				double dr = (R_e(x + hd) - R_e(x - hd)) / (2.0 * hd);
				double s = std::sqrt(1.0 + dr * dr);
				I1 += s / r * dx;
				I2 += r * s * dx;
			}
			return 4.0 / I1 / I2;
		};
		return (k11_at_eps(eps) - k11_at_eps(-eps)) / (2.0 * eps);
	}

	// 将顶点投影到精确曲面 (y-cy)²+(z-cz)²-R(x-cx)²=0
	// 沿径向(y,z)缩放到精确半径
	void projectToSurface(xtpms::PeriodicTriMesh& mesh, const Vec3d& halfPeriod) const {
		double cx = static_cast<double>(halfPeriod[0]);
		double cy = static_cast<double>(halfPeriod[1]);
		double cz = static_cast<double>(halfPeriod[2]);
		for (auto v_it = mesh.vertices_begin(); v_it != mesh.vertices_end(); ++v_it) {
			auto p = mesh.point(*v_it);
			double x = static_cast<double>(p[0]) - cx; // 映射到 [-hp, hp]
			double y = static_cast<double>(p[1]) - cy;
			double z = static_cast<double>(p[2]) - cz;
			double r_current = std::sqrt(y * y + z * z);
			double r_exact = R(x);
			if (r_current > 1e-12) {
				double scale = r_exact / r_current;
				mesh.set_point(*v_it, Vec3d(
					p[0],
					static_cast<xtpms::DefaultTriMesh::Scalar>(y * scale + cy),
					static_cast<xtpms::DefaultTriMesh::Scalar>(z * scale + cz)));
			}
		}
	}

	void projectToSurface(xtpms::DefaultTriMesh& mesh, const Vec3d& halfPeriod) const {
		double cx = static_cast<double>(halfPeriod[0]);
		double cy = static_cast<double>(halfPeriod[1]);
		double cz = static_cast<double>(halfPeriod[2]);
		for (auto v_it = mesh.vertices_begin(); v_it != mesh.vertices_end(); ++v_it) {
			auto p = mesh.point(*v_it);
			double x = static_cast<double>(p[0]) - cx;
			double y = static_cast<double>(p[1]) - cy;
			double z = static_cast<double>(p[2]) - cz;
			double r_current = std::sqrt(y * y + z * z);
			double r_exact = R(x);
			if (r_current > 1e-12) {
				double scale = r_exact / r_current;
				mesh.set_point(*v_it, Vec3d(
					p[0],
					static_cast<xtpms::DefaultTriMesh::Scalar>(y * scale + cy),
					static_cast<xtpms::DefaultTriMesh::Scalar>(z * scale + cz)));
			}
		}
	}

	// 生成 MC level set 网格
	xtpms::DefaultTriMesh generateMC(const Vec3d& halfPeriod, int res) const {
		const double Lx = 2.0 * static_cast<double>(halfPeriod[0]);
		const double Ly = 2.0 * static_cast<double>(halfPeriod[1]);
		const double Lz = 2.0 * static_cast<double>(halfPeriod[2]);
		const double hpx = static_cast<double>(halfPeriod[0]);
		auto levelSet = [&](double x, double y, double z) {
			// 映射 x ∈ [0, 2*hp] → [-hp, hp]
			double xp = x - hpx;
			// 映射 y, z 到以 0 为中心
			double yp = y - static_cast<double>(halfPeriod[1]);
			double zp = z - static_cast<double>(halfPeriod[2]);
			double r = R(xp);
			return yp * yp + zp * zp - r * r;
		};
		return makeTPMSMC(halfPeriod, levelSet, res);
	}

	// 生成周期闭合网格（可选投影到精确曲面）
	xtpms::PeriodicTriMesh generateClosed(const Vec3d& hp, int res, bool project = false) const {
		auto src = generateMC(hp, res);
		if (project) projectToSurface(src, hp);
		auto mesh = makeClosedTPMS(hp, src);
		if (project) projectToSurface(mesh, hp);
		return mesh;
	}
};

// ──────────────────────────────────────────────────────────
// minsurf 参考实现（直接拷贝），放在 msf_ref 命名空间
// ──────────────────────────────────────────────────────────
namespace msf_ref {

struct Compile1ring {
	double As = 0;
	Eigen::Vector3d nv{0, 0, 0};
	Eigen::Vector3d Lx{0, 0, 0};
	double H = 0, K = 0;
	double mass = 0;
	Compile1ring() = default;
	Compile1ring(const Eigen::Vector3d& o, const std::vector<Eigen::Vector3d>& ring) {
		Eigen::Vector3d Hv(0, 0, 0);
		double theta_sum = 0;
		int N = static_cast<int>(ring.size());
		std::vector<double> esq_inc(N), esq_opp(N);
		std::vector<double> cot_alpha(N), cot_beta(N), theta_v(N);
		for (int i = 0; i < N; i++) {
			esq_inc[i] = (ring[i] - o).squaredNorm();
			esq_opp[i] = (ring[(i + 1) % N] - ring[i]).squaredNorm();
		}
		for (int i = 0; i < N; i++) {
			double e_sq[] = {esq_inc[i], esq_inc[(i + 1) % N], esq_opp[i]};
			double cosA = (e_sq[0] + e_sq[2] - e_sq[1]) / 2 / std::sqrt(e_sq[0] * e_sq[2]);
			double cosB = (e_sq[1] + e_sq[2] - e_sq[0]) / 2 / std::sqrt(e_sq[1] * e_sq[2]);
			double cotA = cosA / std::sqrt(1 - cosA * cosA);
			double cotB = cosB / std::sqrt(1 - cosB * cosB);
			cot_alpha[(i + 1) % N] = cotA;
			cot_beta[i] = cotB;
			theta_v[i] = std::acos(std::clamp((e_sq[0] + e_sq[1] - e_sq[2]) / 2 / std::sqrt(e_sq[0] * e_sq[1]), -1.0, 1.0));
		}
		for (int i = 0; i < N; i++) {
			Eigen::Vector3d a = ring[i] - o;
			Eigen::Vector3d b = ring[(i + 1) % N] - o;
			double e_sq[] = {esq_inc[i], esq_inc[(i + 1) % N], esq_opp[i]};
			theta_sum += theta_v[i];
			Eigen::Vector3d axb = a.cross(b);
			mass += axb.norm() / 2 / 3;
			nv += axb.normalized() * theta_v[i];
			Hv += (cot_alpha[i] + cot_beta[i]) / 2 * a;
			double cotA = cot_alpha[(i + 1) % N];
			double cotB = cot_beta[i];
			if (e_sq[0] + e_sq[1] >= e_sq[2] && e_sq[1] + e_sq[2] >= e_sq[0] && e_sq[2] + e_sq[0] >= e_sq[1]) {
				As += cotA * e_sq[1] / 8 + cotB * e_sq[0] / 8;
			} else if (e_sq[0] + e_sq[1] < e_sq[2]) {
				As += axb.norm() / 4;
			} else {
				As += axb.norm() / 8;
			}
		}
		nv.normalize();
		K = (2 * M_PI - theta_sum) / As;
		Lx = Hv;
		H = Hv.norm() / As / 2;
		if (Hv.dot(nv) < 0) H *= -1;
	}
};

Eigen::Matrix3d face_frame(const Eigen::Matrix3d& tri) {
	Eigen::Vector3d n = (tri.col(1) - tri.col(0)).cross(tri.col(2) - tri.col(0)) / 2;
	Eigen::Vector3d e1 = (tri.col(1) - tri.col(0)).normalized();
	Eigen::Vector3d e2 = n.cross(e1).normalized();
	Eigen::Matrix3d fr;
	fr << e1, e2, n;
	return fr;
}

Eigen::Vector3d second_fundamental_form(const Eigen::Matrix3d& tri,
	const Compile1ring& v1, const Compile1ring& v2, const Compile1ring& v3) {
	double be12 = -(v2.nv - v1.nv).dot(tri.col(1) - tri.col(0));
	double be23 = -(v3.nv - v2.nv).dot(tri.col(2) - tri.col(1));
	double be31 = -(v1.nv - v3.nv).dot(tri.col(0) - tri.col(2));
	return {be12, be23, be31};
}

Eigen::Matrix3d strain_matrix_edge_stretch(const Eigen::Matrix3d& tri,
	const Eigen::Vector3d& e1, const Eigen::Vector3d& e2) {
	Eigen::Matrix<double, 3, 2> fram;
	fram << e1, e2;
	Eigen::Matrix3d tri1;
	tri1 << tri.col(1), tri.col(2), tri.col(0);
	Eigen::Matrix<double, 2, 3> d = fram.transpose() * (tri1 - tri);
	Eigen::Matrix3d A;
	A << d(0,0)*d(0,0), d(1,0)*d(1,0), d(0,0)*d(1,0),
	     d(0,1)*d(0,1), d(1,1)*d(1,1), d(0,1)*d(1,1),
	     d(0,2)*d(0,2), d(1,2)*d(1,2), d(0,2)*d(1,2);
	return A.inverse();
}

Eigen::Matrix2d fromvoigt(const Eigen::Vector3d& eps) {
	Eigen::Matrix2d E;
	E(0,0) = eps[0]; E(1,1) = eps[1];
	E(0,1) = eps[2] / 2; E(1,0) = E(0,1);
	return E;
}

Eigen::Matrix<double, 2, 3> scalar_gradient_matrix(const Eigen::Matrix3d& tri, const Eigen::Matrix3d& fr) {
	Eigen::Matrix2d V;
	V << fr.leftCols(2).transpose() * (tri.col(1) - tri.col(0)),
	     fr.leftCols(2).transpose() * (tri.col(2) - tri.col(0));
	Eigen::Matrix<double, 2, 3> S;
	S << -1, 1, 0, -1, 0, 1;
	return V.transpose().lu().solve(S);
}

Eigen::Vector<double, 6> voigt(const Eigen::Matrix3d& eps) {
	Eigen::Vector<double, 6> veps;
	for (int i = 0; i < 3; i++) {
		veps[i] = eps(i, i);
		veps[i + 3] = 2 * eps((i + 1) % 3, (i + 2) % 3);
	}
	return veps;
}

} // namespace msf_ref

// ──────────────────────────────────────────────────────────
// Revolution surface ADC 验证：k11 解析 vs 数值
// ──────────────────────────────────────────────────────────

TEST(AsymptoticConductivity, RevolutionSurface_K11) {
	// 旋转体绕 x 轴，R(x) 周期函数，域 [-1,1]^3
	// 测试多种 profile 在不同分辨率下的收敛性
	const Vec3d hp(1.0, 1.0, 1.0);

	struct TestCase {
		std::string name;
		RevolutionProfile profile;
	};

	std::vector<TestCase> cases;

	// Case 1: 圆柱 R(x) = 0.3 (常数)
	// k11 = 4 / (2/0.3) / (2*0.3) = 4 / 6.667 / 0.6 = 1.0
	// 实际上 k11 = 4 / (∫ 1/R dx) / (∫ R dx) = 4 / (2/0.3) / (2*0.3) = 4/6.667/0.6 = 1.0
	cases.push_back({"cylinder_R0.3", {
		[](double) { return 0.3; },
		[](double) { return 0.0; }
	}});

	// Case 2: 微扰圆柱 R(x) = 0.3 + 0.05*cos(pi*x)
	cases.push_back({"perturbed_cyl", {
		[](double x) { return 0.3 + 0.05 * std::cos(M_PI * x); },
		[](double x) { return -0.05 * M_PI * std::sin(M_PI * x); }
	}});

	// Case 3: 较大扰动 R(x) = 0.3 + 0.12*cos(pi*x)
	cases.push_back({"wavy_cyl", {
		[](double x) { return 0.3 + 0.12 * std::cos(M_PI * x); },
		[](double x) { return -0.12 * M_PI * std::sin(M_PI * x); }
	}});

	// Case 4: catenoid-like R(x) = 0.2*cosh(x/0.2) 但截断使其周期
	// 用 cos 近似: R(x) = 0.25 + 0.1*cos(pi*x)
	cases.push_back({"catenoid_approx", {
		[](double x) { return 0.25 + 0.1 * std::cos(M_PI * x); },
		[](double x) { return -0.1 * M_PI * std::sin(M_PI * x); }
	}});

	std::vector<int> resolutions = {16, 24, 32};

	for (const auto& tc : cases) {
		double k11_exact = tc.profile.analyticK11(1.0);
		std::cout << "\n=== " << tc.name << " ===\n";
		std::cout << "k11 analytic = " << k11_exact << "\n";

		for (int res : resolutions) {
			auto mesh = tc.profile.generateClosed(hp, res);
			if (countBoundaryEdges(mesh) > 0) {
				std::cout << "  res=" << res << " boundary edges present, skip\n";
				continue;
			}
			if (mesh.n_faces() == 0) {
				std::cout << "  res=" << res << " empty mesh, skip\n";
				continue;
			}

			auto geom = xtpms::computeVertexGeometry(mesh);
			Eigen::MatrixX3d ulist;
			Eigen::Matrix3d kA = xtpms::solveAsymptoticConductivity(mesh, geom, ulist);

			double k11_num = kA(0, 0);
			double err = std::abs(k11_num - k11_exact);
			double rel_err = err / std::abs(k11_exact);

			std::cout << "  res=" << res << " nv=" << mesh.n_vertices()
					  << " nf=" << mesh.n_faces()
					  << " k11_num=" << k11_num
					  << " err=" << err
					  << " rel=" << rel_err << "\n";
			std::cout << "  full kA diag: " << kA(0,0) << " " << kA(1,1) << " " << kA(2,2) << "\n";

			// 最高分辨率应该有合理精度
			if (res == resolutions.back()) {
				EXPECT_LT(rel_err, 0.1) << tc.name << " k11 relative error too large at res=" << res;
			}
		}
	}
}

TEST(AsymptoticConductivity, RevolutionSurface_SensitivityCheck) {
	// 在旋转体上验证 sensitivity 公式：
	// 指定轴对称法向速度 vn(x)，计算 δk11
	// 解析值: 通过 1D 积分公式的 FD 得到
	// 数值值: 通过 computeSensitivity 公式得到
	const Vec3d hp(1.0, 1.0, 1.0);

	// Profile: R(x) = 0.3 + 0.05*cos(pi*x)
	RevolutionProfile prof{
		[](double x) { return 0.3 + 0.05 * std::cos(M_PI * x); },
		[](double x) { return -0.05 * M_PI * std::sin(M_PI * x); }
	};

	// 法向速度场（只依赖 x，保持轴对称）
	struct VnField {
		std::string name;
		RevolutionProfile::Func vn;
	};
	std::vector<VnField> vnFields = {
		{"sin(pi*x)", [](double x) { return std::sin(M_PI * x); }},
		{"cos(2*pi*x)", [](double x) { return std::cos(2.0 * M_PI * x); }},
		{"1.0 (uniform)", [](double) { return 1.0; }},
	};

	// 多分辨率收敛测试
	std::vector<int> resolutions = {32};

	for (int res : resolutions) {
		auto mesh = prof.generateClosed(hp, res, true);  // 投影到精确曲面
		if (countBoundaryEdges(mesh) > 0 || mesh.n_faces() < 100) {
			std::cout << "  res=" << res << " bad mesh, skip\n";
			continue;
		}

		// Remesh 消除退化三角形
		for (int round = 0; round < 3; ++round) {
			xtpms::RemeshOptions ropts;
			ropts.outerIter = 2;
			ropts.innerIter = 5;
			xtpms::delaunayRemesh(mesh, ropts);
			bool hasBnd = false;
			for (auto e_it = mesh.edges_begin(); e_it != mesh.edges_end() && !hasBnd; ++e_it)
				if (mesh.is_boundary(*e_it)) hasBnd = true;
			if (hasBnd) mesh.mergePeriodBoundary();
		}
		// remesh 后再投影回精确曲面
		prof.projectToSurface(mesh, hp);

		if (countBoundaryEdges(mesh) > 0) {
			std::cout << "  res=" << res << " boundary after remesh, skip\n";
			continue;
		}

		// 输出网格供检查
		mesh.saveUnitCell("revolution_res" + std::to_string(res) + ".obj");
		OpenMesh::IO::write_mesh(mesh, "revolution_res" + std::to_string(res) + "_raw.obj");
		std::cout << "Saved revolution_res" << res << ".obj and _raw.obj\n";

		auto geom = xtpms::computeVertexGeometry(mesh);
		Eigen::MatrixX3d ulist;
		Eigen::Matrix3d kA = xtpms::solveAsymptoticConductivity(mesh, geom, ulist);
		double As = geom.vertexAreas.sum();
		const int nv = static_cast<int>(mesh.n_vertices());

		std::cout << "\n--- res=" << res << " nv=" << nv << " nf=" << mesh.n_faces()
				  << " k11=" << kA(0,0) << " (exact=" << prof.analyticK11(1.0) << ") ---\n";

		// 计算 sensitivity
		auto sens = xtpms::computeSensitivity(mesh, geom, ulist);
		for (int i = 0; i < nv; ++i) {
			double ai = geom.vertexAreas[i];
			if (ai > 1e-15) { sens.vSens.row(i) /= ai; sens.aSens[i] /= ai; }
		}
		auto kAv = xtpms::toVoigt(kA);
		kAv.tail<3>() /= 2.0;
		Eigen::Vector<double, 6> k11_grad = Eigen::Vector<double, 6>::Zero();
		k11_grad[0] = 1.0;
		Eigen::VectorXd dfdvn = (sens.vSens / As - sens.aSens * kAv.transpose() / As) * (-k11_grad);

		// ── 先验证面积导数 ──
		auto sensRaw = xtpms::computeSensitivity(mesh, geom, ulist);
		{
			// 用 uniform vn=1.0 验证面积导数
			// A_sens 不除以顶点面积，直接用 raw 值
			double dAs_ana = 0;
			for (int i = 0; i < nv; ++i)
				dAs_ana += sensRaw.aSens[i];  // vn=1 时就是直接求和

			// mesh FD: 扰动后重算面积
			for (double step : {1e-2, 1e-3}) {
				auto perturbArea = [&](double s) {
					xtpms::PeriodicTriMesh copy = mesh;
					for (auto v_it = copy.vertices_begin(); v_it != copy.vertices_end(); ++v_it) {
						int vi = (*v_it).idx();
						Eigen::Vector3d p(
							static_cast<double>(copy.point(*v_it)[0]),
							static_cast<double>(copy.point(*v_it)[1]),
							static_cast<double>(copy.point(*v_it)[2]));
						Eigen::Vector3d n = geom.vertexNormals[static_cast<std::size_t>(vi)];
						p += s * n;  // uniform vn=1
						copy.set_point(*v_it, Vec3d(
							static_cast<xtpms::DefaultTriMesh::Scalar>(p[0]),
							static_cast<xtpms::DefaultTriMesh::Scalar>(p[1]),
							static_cast<xtpms::DefaultTriMesh::Scalar>(p[2])));
					}
					auto g = xtpms::computeVertexGeometry(copy);
					return g.vertexAreas.sum();
				};
				double dAs_fd = (perturbArea(step) - perturbArea(-step)) / (2.0 * step);
				std::cout << "  Area deriv (vn=1): step=" << step
						  << "  FD=" << dAs_fd << "  analytic=" << dAs_ana
						  << "  ratio=" << (std::abs(dAs_fd) > 1e-12 ? dAs_ana / dAs_fd : 0) << "\n";
			}
		}
		std::cout << "\n";

		// ── 用解析 u₀ 代入 sensitivity 公式验证 ──
		// u₀(x) = ∫_{-1}^x [s(ζ)/R(ζ) * c3 - 1] dζ, s=sqrt(1+R'²)
		// c3 = 2 / I1, I1 = ∫_{-1}^1 s/R dx
		{
			double I1_quad = 0;
			int nq = 100000;
			double dxq = 2.0 / nq;
			for (int i = 0; i < nq; ++i) {
				double x = -1.0 + (i + 0.5) * dxq;
				double r = prof.R(x), dr = prof.dR(x);
				I1_quad += std::sqrt(1.0 + dr * dr) / r * dxq;
			}
			double c3 = 2.0 / I1_quad;

			// 用解析 u₀ 设置 ulist 的第 0 列
			Eigen::MatrixX3d ulist_analytic = ulist;  // 保留 u1, u2 不变
			for (int i = 0; i < nv; ++i) {
				auto pt = mesh.point(xtpms::PeriodicTriMesh::VertexHandle(i));
				double xv = static_cast<double>(pt[0]) - 1.0;  // 映射到 [-1,1]
				// u₀(xv) = ∫_{-1}^{xv} [s/R * c3 - 1] dζ
				double u0 = 0;
				int nstep = 10000;
				double dz = (xv + 1.0) / nstep;
				for (int j = 0; j < nstep; ++j) {
					double z = -1.0 + (j + 0.5) * dz;
					double rr = prof.R(z), dr = prof.dR(z);
					double ss = std::sqrt(1.0 + dr * dr);
					u0 += (ss / rr * c3 - 1.0) * dz;
				}
				ulist_analytic(i, 0) = u0;
			}
			// 减去均值（和求解器一致）
			ulist_analytic.col(0).array() -= ulist_analytic.col(0).mean();

			// 用解析 u 计算 sensitivity
			auto sens_ana_u = xtpms::computeSensitivity(mesh, geom, ulist_analytic);

			// 用数值 u 计算 sensitivity
			auto sens_num_u = xtpms::computeSensitivity(mesh, geom, ulist);

			// 比较 v_sens 的 k11 分量（Voigt index 0）
			double vSens_k11_ana = 0, vSens_k11_num = 0;
			double aSens_sum = 0;
			for (int i = 0; i < nv; ++i) {
				vSens_k11_ana += sens_ana_u.vSens(i, 0);  // vn=1 uniform
				vSens_k11_num += sens_num_u.vSens(i, 0);
				aSens_sum += sens_num_u.aSens[i];
			}

			// 从 v_sens 和 A_sens 算 dk11 (uniform vn=1)
			// dk11 = (-v_sens_00/As + k11 * A_sens/As) * (-1) ... 按公式
			// 但这里先看 raw v_sens 的对比
			std::cout << "  v_sens k11 (uniform vn=1):\n";
			std::cout << "    with analytic u₀: " << vSens_k11_ana << "\n";
			std::cout << "    with numeric u₀:  " << vSens_k11_num << "\n";
			std::cout << "    ratio: " << (std::abs(vSens_k11_num) > 1e-15 ? vSens_k11_ana / vSens_k11_num : 0) << "\n";

			// 也检查 u₀ 本身的差异
			double u_diff = 0, u_norm = 0;
			for (int i = 0; i < nv; ++i) {
				double d = ulist_analytic(i, 0) - ulist(i, 0);
				u_diff += d * d;
				u_norm += ulist_analytic(i, 0) * ulist_analytic(i, 0);
			}
			std::cout << "    u₀ relative diff: " << std::sqrt(u_diff / (u_norm + 1e-30)) << "\n";

			// 用解析 u 算完整的 dk11
			// dk11/dvn = -v_sens_00/As + k11 * A_sens/As (per vertex, uniform vn=1, 不除 Av)
			double dk11_formula_ana_u = (-vSens_k11_ana / As + kA(0,0) * aSens_sum / As);
			double dk11_formula_num_u = (-vSens_k11_num / As + kA(0,0) * aSens_sum / As);
			double dk11_var = prof.variationalDk11(1.0, [](double) { return 1.0; });

			std::cout << "    dk11 (analytic u): " << dk11_formula_ana_u << "\n";
			std::cout << "    dk11 (numeric u):  " << dk11_formula_num_u << "\n";
			std::cout << "    dk11 variational:  " << dk11_var << "\n";
			// mesh FD 用 step=1e-2
			auto perturbK11 = [&](double s) {
				xtpms::PeriodicTriMesh copy = mesh;
				for (auto v_it = copy.vertices_begin(); v_it != copy.vertices_end(); ++v_it) {
					int vi = (*v_it).idx();
					Eigen::Vector3d p(
						static_cast<double>(copy.point(*v_it)[0]),
						static_cast<double>(copy.point(*v_it)[1]),
						static_cast<double>(copy.point(*v_it)[2]));
					Eigen::Vector3d n = geom.vertexNormals[static_cast<std::size_t>(vi)];
					p += s * n;
					copy.set_point(*v_it, Vec3d(
						static_cast<xtpms::DefaultTriMesh::Scalar>(p[0]),
						static_cast<xtpms::DefaultTriMesh::Scalar>(p[1]),
						static_cast<xtpms::DefaultTriMesh::Scalar>(p[2])));
				}
				auto g = xtpms::computeVertexGeometry(copy);
				Eigen::MatrixX3d u;
				Eigen::Matrix3d k = xtpms::solveAsymptoticConductivity(copy, g, u);
				return k(0, 0);
			};
			double dk11_meshFD = (perturbK11(1e-2) - perturbK11(-1e-2)) / (2e-2);
			std::cout << "    dk11 mesh FD:      " << dk11_meshFD << "\n\n";
		}

		for (const auto& vnf : vnFields) {
			Eigen::VectorXd vn_vec(nv);
			for (int i = 0; i < nv; ++i) {
				auto pt = mesh.point(xtpms::PeriodicTriMesh::VertexHandle(i));
				double x = static_cast<double>(pt[0]) - 1.0;
				vn_vec[i] = vnf.vn(x);
			}

			// dfdvn 是逐顶点梯度（已除以 Av），方向导数需要面积加权内积
			double dk11_formula_wrong = vn_vec.dot(dfdvn);  // 错误：无权内积
			double dk11_formula = 0;
			for (int i = 0; i < nv; ++i)
				dk11_formula += vn_vec[i] * dfdvn[i] * geom.vertexAreas[i];  // 面积加权
			double dk11_var = prof.variationalDk11(1.0, vnf.vn);

			double stepMesh = 1e-2;
		{ // single step
			auto perturbAll = [&](double s) {
				xtpms::PeriodicTriMesh copy = mesh;
				for (auto v_it = copy.vertices_begin(); v_it != copy.vertices_end(); ++v_it) {
					int vi = (*v_it).idx();
					Eigen::Vector3d p(
						static_cast<double>(copy.point(*v_it)[0]),
						static_cast<double>(copy.point(*v_it)[1]),
						static_cast<double>(copy.point(*v_it)[2]));
					Eigen::Vector3d n = geom.vertexNormals[static_cast<std::size_t>(vi)];
					p += s * vn_vec[vi] * n;
					copy.set_point(*v_it, Vec3d(
						static_cast<xtpms::DefaultTriMesh::Scalar>(p[0]),
						static_cast<xtpms::DefaultTriMesh::Scalar>(p[1]),
						static_cast<xtpms::DefaultTriMesh::Scalar>(p[2])));
				}
				auto g = xtpms::computeVertexGeometry(copy);
				Eigen::MatrixX3d u;
				Eigen::Matrix3d k = xtpms::solveAsymptoticConductivity(copy, g, u);
				return k(0, 0);
			};
			double dk11_meshFD = (perturbAll(stepMesh) - perturbAll(-stepMesh)) / (2.0 * stepMesh);

			std::cout << "  vn=" << vnf.name
					  << "  var=" << dk11_var
					  << "  meshFD=" << dk11_meshFD
					  << "  formula=" << dk11_formula;
			if (std::abs(dk11_meshFD) > 1e-12)
				std::cout << "  f/mFD=" << dk11_formula / dk11_meshFD;
			if (std::abs(dk11_var) > 1e-12)
				std::cout << "  f/var=" << dk11_formula / dk11_var;
			std::cout << "\n";
		  }
		}
	}
}

TEST(AsymptoticConductivity, CompareWithMinsurf) {
	// 对同一网格的同一面，分别用 xtpms 和 minsurf 参考实现计算中间量并对比
	const Vec3d hp(1.0, 1.0, 1.0);
	auto mesh = makeClosedSchwarzP(hp, 20);
	ASSERT_EQ(countBoundaryEdges(mesh), 0);

	auto geom = xtpms::computeVertexGeometry(mesh);
	Eigen::MatrixX3d ulist;
	Eigen::Matrix3d kA = xtpms::solveAsymptoticConductivity(mesh, geom, ulist);

	// 取 1-ring 数据构建 msf_ref::Compile1ring
	auto toEig = [](const Vec3d& v) { return Eigen::Vector3d(v[0], v[1], v[2]); };
	auto makePeriodLocal = [&](const Vec3d& v) {
		Vec3d out = v;
		for (int a = 0; a < 3; ++a) {
			double period = 2.0 * hp[a];
			double vi = out[a];
			if (vi < -hp[a]) vi += period;
			else if (vi > hp[a]) vi -= period;
			out[a] = static_cast<xtpms::DefaultTriMesh::Scalar>(vi);
		}
		return out;
	};

	// 为每个顶点构建 msf_ref 版本的 Compile1ring
	int nv = static_cast<int>(mesh.n_vertices());
	std::vector<msf_ref::Compile1ring> msf_vrings(nv);
	for (int vid = 0; vid < nv; ++vid) {
		auto vh = xtpms::PeriodicTriMesh::VertexHandle(vid);
		Eigen::Vector3d center = toEig(mesh.point(vh));
		std::vector<Eigen::Vector3d> ring;
		// 用和 xtpms 完全相同的 1-ring 数据
		if (!mesh.is_boundary(vh)) {
			auto he_start = mesh.halfedge_handle(vh);
			auto he = he_start;
			do {
				auto vn = mesh.to_vertex_handle(he);
				ring.push_back(center + toEig(makePeriodLocal(mesh.point(vn) - mesh.point(vh))));
				he = mesh.opposite_halfedge_handle(mesh.prev_halfedge_handle(he));
			} while (he != he_start && ring.size() < 30);
		}
		msf_vrings[vid] = msf_ref::Compile1ring(center, ring);
	}

	// 逐面对比前几个面
	int checked = 0;
	int diffs = 0;
	for (auto f_it = mesh.faces_begin(); f_it != mesh.faces_end() && checked < 5; ++f_it, ++checked) {
		auto fv = mesh.cfv_iter(*f_it);
		auto v0h = *fv; ++fv; auto v1h = *fv; ++fv; auto v2h = *fv;
		int i0 = v0h.idx(), i1 = v1h.idx(), i2 = v2h.idx();

		// 构建周期包装后的三角形（两个版本用同一个 tri）
		Eigen::Vector3d p0 = toEig(mesh.point(v0h));
		Eigen::Vector3d p1 = p0 + toEig(makePeriodLocal(mesh.point(v1h) - mesh.point(v0h)));
		Eigen::Vector3d p2 = p0 + toEig(makePeriodLocal(mesh.point(v2h) - mesh.point(v0h)));
		Eigen::Matrix3d tri;
		tri.col(0) = p0; tri.col(1) = p1; tri.col(2) = p2;

		// === xtpms 版本 ===
		auto fr_x = xtpms::faceFrame(tri);
		auto be_x = xtpms::secondFundamentalFormEdge(tri, geom.vrings[i0], geom.vrings[i1], geom.vrings[i2]);
		auto Bn_x = xtpms::strainMatrixEdgeStretch(tri, fr_x.col(0), fr_x.col(1));
		Eigen::Vector3d bfv_x = Bn_x * be_x;
		auto bform_x = xtpms::fromVoigt3(bfv_x);
		auto G_x = xtpms::scalarGradientMatrix(tri, fr_x);

		// === msf_ref 版本 ===
		auto fr_m = msf_ref::face_frame(tri);
		auto be_m = msf_ref::second_fundamental_form(tri, msf_vrings[i0], msf_vrings[i1], msf_vrings[i2]);
		auto Bn_m = msf_ref::strain_matrix_edge_stretch(tri, fr_m.col(0), fr_m.col(1));
		Eigen::Vector3d bfv_m = Bn_m * be_m;
		auto bform_m = msf_ref::fromvoigt(bfv_m);
		auto G_m = msf_ref::scalar_gradient_matrix(tri, fr_m);

		// u values
		Eigen::Matrix3d ue;
		ue.row(0) = ulist.row(i0);
		ue.row(1) = ulist.row(i1);
		ue.row(2) = ulist.row(i2);

		Eigen::Matrix<double, 2, 3> gu_x = G_x * ue;
		Eigen::Matrix<double, 2, 3> pw_x = fr_x.leftCols<2>().transpose();
		double H_x = bform_x.trace() / 2.0;
		Eigen::Matrix3d sens_x = (gu_x + pw_x).transpose() * (bform_x - H_x * Eigen::Matrix2d::Identity()) * (gu_x + pw_x);

		Eigen::Matrix<double, 2, 3> gu_m = G_m * ue;
		Eigen::Matrix<double, 2, 3> pw_m = fr_m.leftCols<2>().transpose();
		double H_m = bform_m.trace() / 2.0;
		Eigen::Matrix3d sens_m = (gu_m + pw_m).transpose() * (bform_m - H_m * Eigen::Matrix2d::Identity()) * (gu_m + pw_m);

		std::cout << "Face " << (*f_it).idx() << " (v" << i0 << ",v" << i1 << ",v" << i2 << "):\n";

		// 对比法向
		double nv_dot0 = geom.vrings[i0].nv.dot(msf_vrings[i0].nv);
		double nv_dot1 = geom.vrings[i1].nv.dot(msf_vrings[i1].nv);
		double nv_dot2 = geom.vrings[i2].nv.dot(msf_vrings[i2].nv);
		std::cout << "  nv dots: " << nv_dot0 << " " << nv_dot1 << " " << nv_dot2 << "\n";
		std::cout << "  As: xtpms=[" << geom.vrings[i0].As << "," << geom.vrings[i1].As << "," << geom.vrings[i2].As
				  << "] msf=[" << msf_vrings[i0].As << "," << msf_vrings[i1].As << "," << msf_vrings[i2].As << "]\n";

		// 对比 face frame
		double fr_dot = (fr_x.col(2).normalized()).dot(fr_m.col(2).normalized());
		std::cout << "  fr_n dot: " << fr_dot << "\n";

		// 对比 be
		std::cout << "  be: xtpms=" << be_x.transpose() << " msf=" << be_m.transpose() << " diff=" << (be_x - be_m).norm() << "\n";

		// 对比 Bn
		std::cout << "  Bn diff: " << (Bn_x - Bn_m).norm() << "\n";

		// 对比 bform
		std::cout << "  bform: xtpms=\n" << bform_x << "\n    msf=\n" << bform_m << "\n    diff=" << (bform_x - bform_m).norm() << "\n";

		// 对比 G
		std::cout << "  G diff: " << (G_x - G_m).norm() << "\n";

		// 对比 gu
		std::cout << "  gu diff: " << (gu_x - gu_m).norm() << "\n";

		// 对比 pw
		std::cout << "  pw diff: " << (pw_x - pw_m).norm() << "\n";

		// 对比 sens
		std::cout << "  sens trace: xtpms=" << sens_x.trace() << " msf=" << sens_m.trace() << "\n";
		std::cout << "  sens diff: " << (sens_x - sens_m).norm() << "\n";

		if ((be_x - be_m).norm() > 1e-10 || (Bn_x - Bn_m).norm() > 1e-10 ||
		    (G_x - G_m).norm() > 1e-10 || (sens_x - sens_m).norm() > 1e-10) {
			++diffs;
		}
	}

	std::cout << "\nDifferences found in " << diffs << " / " << checked << " faces\n";
	EXPECT_EQ(diffs, 0);
}

TEST(Surgery, DiagnoseRingOrder) {
	// 检查 cvoh_iter 的遍历顺序是否与面环形一致
	const Vec3d hp(0.5, 0.5, 0.5);
	auto mesh = makeClosedGyroid(hp, 16);
	ASSERT_EQ(countBoundaryEdges(mesh), 0);

	// 取前几个内部顶点，比较 cvoh_iter 顺序和面遍历顺序
	int checked = 0;
	for (auto v_it = mesh.vertices_begin(); v_it != mesh.vertices_end() && checked < 5; ++v_it) {
		if (mesh.is_boundary(*v_it)) continue;
		auto vh = *v_it;

		// 方法1: cvoh_iter（当前用法）
		std::vector<int> ring_voh;
		for (auto voh = mesh.cvoh_iter(vh); voh.is_valid(); ++voh)
			ring_voh.push_back(mesh.to_vertex_handle(*voh).idx());

		// 方法2: 通过面遍历（保证 CCW）
		std::vector<int> ring_face;
		auto he_start = mesh.halfedge_handle(vh); // 一条 outgoing halfedge
		auto he = he_start;
		do {
			ring_face.push_back(mesh.to_vertex_handle(he).idx());
			// 转到下一条 outgoing: prev(opp(he))? 或 opp(next(he))?
			he = mesh.opposite_halfedge_handle(mesh.prev_halfedge_handle(he));
		} while (he != he_start && ring_face.size() < 20);

		bool same = (ring_voh == ring_face);
		std::cout << "v" << vh.idx() << " valence=" << mesh.valence(vh)
				  << " voh=[";
		for (int v : ring_voh) std::cout << v << ",";
		std::cout << "] face=[";
		for (int v : ring_face) std::cout << v << ",";
		std::cout << "] same=" << same << "\n";

		// 计算两种顺序的 Compile1ring H 值
		auto buildRing = [&](const std::vector<int>& order) {
			Eigen::Vector3d center = Eigen::Vector3d(
				static_cast<double>(mesh.point(vh)[0]),
				static_cast<double>(mesh.point(vh)[1]),
				static_cast<double>(mesh.point(vh)[2]));
			std::vector<Eigen::Vector3d> ring;
			for (int vi : order) {
				auto diff = mesh.point(xtpms::PeriodicTriMesh::VertexHandle(vi)) - mesh.point(vh);
				auto wrapped = mesh.wrapVector(diff);
				ring.push_back(center + Eigen::Vector3d(
					static_cast<double>(wrapped[0]),
					static_cast<double>(wrapped[1]),
					static_cast<double>(wrapped[2])));
			}
			return xtpms::Compile1ring(center, ring);
		};

		auto cr_voh = buildRing(ring_voh);
		auto cr_face = buildRing(ring_face);
		std::cout << "  H_voh=" << cr_voh.H << " H_face=" << cr_face.H
				  << " As_voh=" << cr_voh.As << " As_face=" << cr_face.As
				  << " Lx_voh=" << cr_voh.Lx.norm() << " Lx_face=" << cr_face.Lx.norm() << "\n";
		// 输出 ring 的实际坐标（看 wrap 是否正确）
		if (checked == 0) {
			auto center = Eigen::Vector3d(
				static_cast<double>(mesh.point(vh)[0]),
				static_cast<double>(mesh.point(vh)[1]),
				static_cast<double>(mesh.point(vh)[2]));
			std::cout << "  center=(" << center.transpose() << ")\n";
			for (size_t ri = 0; ri < ring_voh.size(); ++ri) {
				auto diff = mesh.point(xtpms::PeriodicTriMesh::VertexHandle(ring_voh[ri])) - mesh.point(vh);
				auto wrapped = mesh.wrapVector(diff);
				Eigen::Vector3d p = center + Eigen::Vector3d(
					static_cast<double>(wrapped[0]),
					static_cast<double>(wrapped[1]),
					static_cast<double>(wrapped[2]));
				auto raw = mesh.point(xtpms::PeriodicTriMesh::VertexHandle(ring_voh[ri]));
				std::cout << "  ring[" << ri << "] v" << ring_voh[ri]
						  << " raw=(" << raw[0] << "," << raw[1] << "," << raw[2]
						  << ") wrapped=(" << p.transpose() << ")"
						  << " dist=" << (p - center).norm() << "\n";
			}
		}

		++checked;
	}
}

TEST(Surgery, GyroidSurgerySmoke) {
	// MC 生成的 Gyroid 离散曲率统计（MC 网格不精确，离散 H 可能很大）
	const Vec3d hp(0.5, 0.5, 0.5);
	for (int res : {16, 24, 32}) {
		auto m = makeClosedGyroid(hp, res);
		auto g = xtpms::computeVertexGeometry(m);
		double maxH = 0, sumH = 0;
		for (auto& vr : g.vrings) { maxH = std::max(maxH, std::abs(vr.H)); sumH += std::abs(vr.H); }
		double avgH = sumH / m.n_vertices();
		std::cout << "res=" << res << " nv=" << m.n_vertices() << " maxH=" << maxH << " avgH=" << avgH << "\n";
	}

	// 对 remesh 后的 Gyroid 做 surgery 测试（remesh 后网格质量更好，H 更合理）
	auto mesh = makeClosedGyroid(hp, 16);
	xtpms::RemeshOptions ropts;
	ropts.outerIter = 1;
	ropts.innerIter = 3;
	xtpms::delaunayRemesh(mesh, ropts);
	ASSERT_EQ(countBoundaryEdges(mesh), 0);
	int nvBefore = static_cast<int>(mesh.n_vertices());

	auto geom = xtpms::computeVertexGeometry(mesh);
	double maxH = 0;
	for (auto& vr : geom.vrings) maxH = std::max(maxH, std::abs(vr.H));
	std::cout << "after remesh: nv=" << mesh.n_vertices() << " maxH=" << maxH << "\n";

	xtpms::PeriodicTriMesh::SurgeryOptions opts;
	opts.singularityTol = 25.0;
	bool performed = mesh.surgery(opts);

	std::cout << "surgery performed=" << performed << "\n";
	EXPECT_FALSE(performed) << "Remeshed Gyroid should not need surgery (maxH=" << maxH << ")";
	EXPECT_EQ(static_cast<int>(mesh.n_vertices()), nvBefore);
	EXPECT_EQ(countBoundaryEdges(mesh), 0);
}

TEST(Surgery, NeckMesh_SurgeryAndFill) {
	namespace fs = std::filesystem;
	const fs::path meshPath(R"(tests/data/neck.obj)");
	if (!fs::exists(meshPath)) {
		GTEST_SKIP() << "neck.obj not found";
	}

	xtpms::DefaultTriMesh src;
	ASSERT_TRUE(OpenMesh::IO::read_mesh(src, meshPath.string()));

	const Vec3d hp(5.0, 5.0, 5.0);
	auto mesh = makeClosedTPMS(hp, src);
	std::cout << "neck: nv=" << mesh.n_vertices() << " nf=" << mesh.n_faces()
			  << " bnd=" << countBoundaryEdges(mesh) << "\n";
	mesh.saveUnitCell("neck_before_surgery.obj");

	// remesh 先改善网格质量
	xtpms::RemeshOptions ropts;
	ropts.outerIter = 1;
	ropts.innerIter = 3;
	xtpms::delaunayRemesh(mesh, ropts);

	auto geom = xtpms::computeVertexGeometry(mesh);
	double maxH = 0;
	for (auto& vr : geom.vrings) maxH = std::max(maxH, std::abs(vr.H));
	std::cout << "after remesh: nv=" << mesh.n_vertices() << " nf=" << mesh.n_faces()
			  << " maxH=" << maxH << " bnd=" << countBoundaryEdges(mesh) << "\n";

	// surgery（降低阈值触发颈部切除）
	xtpms::PeriodicTriMesh::SurgeryOptions sopts;
	sopts.singularityTol = 5.0;
	bool performed = mesh.surgery(sopts);

	std::cout << "surgery: performed=" << performed
			  << " nv=" << mesh.n_vertices() << " nf=" << mesh.n_faces()
			  << " bnd=" << countBoundaryEdges(mesh) << "\n";
	mesh.saveUnitCell("neck_after_surgery.obj");

	EXPECT_GT(mesh.n_faces(), 0u);
	if (performed) {
		// surgery 后应保留大部分面
		EXPECT_GT(mesh.n_faces(), 1000u);
	}
}

TEST(Surgery, GyroidLowThreshold_SurgeryAndFill) {
	// remesh 后用低阈值触发 surgery，测试 CGAL 填洞 + bilaplacian 光滑
	const Vec3d hp(0.5, 0.5, 0.5);
	auto mesh = makeClosedGyroid(hp, 16);
	xtpms::RemeshOptions ropts;
	ropts.outerIter = 1;
	ropts.innerIter = 3;
	xtpms::delaunayRemesh(mesh, ropts);
	ASSERT_EQ(countBoundaryEdges(mesh), 0);
	int nfBefore = static_cast<int>(mesh.n_faces());

	auto geom = xtpms::computeVertexGeometry(mesh);
	double maxH = 0;
	for (auto& vr : geom.vrings) maxH = std::max(maxH, std::abs(vr.H));
	std::cout << "before surgery: maxH=" << maxH << " nf=" << nfBefore << "\n";

	mesh.saveUnitCell("surgery_before.obj");

	xtpms::PeriodicTriMesh::SurgeryOptions opts;
	opts.singularityTol = 5.0;
	bool performed = mesh.surgery(opts);

	mesh.saveUnitCell("surgery_after.obj");

	std::cout << "surgery: performed=" << performed
			  << " nv=" << mesh.n_vertices() << " nf=" << mesh.n_faces()
			  << " bnd=" << countBoundaryEdges(mesh) << "\n";

	// 低阈值会删较多面，只要不全删就行
	EXPECT_GT(mesh.n_faces(), 0u) << "Surgery should not delete all faces";
}

// ──────────────────────────────────────────────────────────
// ADC 单元测试
// ──────────────────────────────────────────────────────────

TEST(AsymptoticConductivity, GyroidSolveADC) {
	const Vec3d hp(0.5, 0.5, 0.5);
	auto mesh = makeClosedGyroid(hp);
	ASSERT_EQ(countBoundaryEdges(mesh), 0);

	auto geom = xtpms::computeVertexGeometry(mesh);
	EXPECT_EQ(geom.cotWeights.size(), mesh.n_edges());
	EXPECT_EQ(geom.vertexNormals.size(), mesh.n_vertices());
	EXPECT_GT(geom.vertexAreas.sum(), 0.0);

	Eigen::MatrixX3d ulist;
	Eigen::Matrix3d kA = xtpms::solveAsymptoticConductivity(mesh, geom, ulist);

	// kA 应该是对称正半定的，对角线在 (0, 1) 范围内
	std::cout << "kA =\n" << kA << "\n";
	for (int i = 0; i < 3; ++i) {
		EXPECT_GT(kA(i, i), 0.0) << "kA diagonal should be positive";
		EXPECT_LT(kA(i, i), 1.0) << "kA diagonal should be < 1";
	}
	// 对称性
	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < i; ++j) {
			EXPECT_NEAR(kA(i, j), kA(j, i), 1e-10) << "kA should be symmetric";
		}
	}
}

TEST(AsymptoticConductivity, ObjectiveAPAC) {
	Eigen::Matrix3d kA = Eigen::Matrix3d::Identity() * 0.3;
	auto obj = xtpms::evaluateADCObjective("apac", kA);
	EXPECT_NEAR(obj.value, 0.3, 1e-10);
	EXPECT_NEAR(obj.gradient[0], 1.0 / 3.0, 1e-10);
	EXPECT_NEAR(obj.gradient[1], 1.0 / 3.0, 1e-10);
	EXPECT_NEAR(obj.gradient[2], 1.0 / 3.0, 1e-10);
}

TEST(AsymptoticConductivity, ObjectiveK00) {
	Eigen::Matrix3d kA;
	kA << 0.4, 0.01, 0.0,
		  0.01, 0.3, 0.0,
		  0.0, 0.0, 0.2;
	auto obj = xtpms::evaluateADCObjective("k00", kA);
	EXPECT_NEAR(obj.value, 0.4, 1e-10);
	EXPECT_NEAR(obj.gradient[0], 1.0, 1e-10);
	EXPECT_NEAR(obj.gradient[1], 0.0, 1e-10);
}

TEST(FundamentalForms, FaceFrame_UnitTriangle) {
	Eigen::Matrix3d tri;
	tri.col(0) = Eigen::Vector3d(0, 0, 0);
	tri.col(1) = Eigen::Vector3d(1, 0, 0);
	tri.col(2) = Eigen::Vector3d(0, 1, 0);
	auto fr = xtpms::faceFrame(tri);
	// e1 should be (1,0,0)
	EXPECT_NEAR(fr(0, 0), 1.0, 1e-10);
	EXPECT_NEAR(fr(1, 0), 0.0, 1e-10);
	EXPECT_NEAR(fr(2, 0), 0.0, 1e-10);
	// e2 should be (0,1,0)
	EXPECT_NEAR(fr(0, 1), 0.0, 1e-10);
	EXPECT_NEAR(fr(1, 1), 1.0, 1e-10);
	// n should be (0,0,0.5) (area vector)
	EXPECT_NEAR(fr(2, 2), 0.5, 1e-10);
}

TEST(FundamentalForms, Compile1ring_FlatPlane) {
	// 平面上的正六边形 1-ring：H=0, K=0
	Eigen::Vector3d center(0, 0, 0);
	std::vector<Eigen::Vector3d> ring;
	for (int i = 0; i < 6; ++i) {
		double angle = i * M_PI / 3.0;
		ring.push_back(Eigen::Vector3d(std::cos(angle), std::sin(angle), 0));
	}
	xtpms::Compile1ring cr(center, ring);
	EXPECT_NEAR(cr.H, 0.0, 1e-10) << "Flat plane should have H=0";
	EXPECT_NEAR(cr.K, 0.0, 1e-2) << "Flat plane should have K=0";
	EXPECT_NEAR(cr.nv[2], 1.0, 1e-10) << "Normal should be (0,0,1)";
	EXPECT_GT(cr.As, 0.0) << "Area should be positive";
}

TEST(FundamentalForms, SecondFundamentalForm_Flat) {
	// 平面三角形 + 平面法向：II 应为 0
	Eigen::Matrix3d tri;
	tri.col(0) = Eigen::Vector3d(0, 0, 0);
	tri.col(1) = Eigen::Vector3d(1, 0, 0);
	tri.col(2) = Eigen::Vector3d(0.5, 0.866, 0);
	// 所有顶点法向均为 (0,0,1)
	xtpms::Compile1ring v0, v1, v2;
	v0.nv = v1.nv = v2.nv = Eigen::Vector3d(0, 0, 1);
	auto be = xtpms::secondFundamentalFormEdge(tri, v0, v1, v2);
	EXPECT_NEAR(be[0], 0.0, 1e-10);
	EXPECT_NEAR(be[1], 0.0, 1e-10);
	EXPECT_NEAR(be[2], 0.0, 1e-10);
}

TEST(FundamentalForms, StrainMatrix_Invertible) {
	Eigen::Matrix3d tri;
	tri.col(0) = Eigen::Vector3d(0, 0, 0);
	tri.col(1) = Eigen::Vector3d(1, 0, 0);
	tri.col(2) = Eigen::Vector3d(0.5, 0.866, 0);
	auto fr = xtpms::faceFrame(tri);
	auto Bn = xtpms::strainMatrixEdgeStretch(tri, fr.col(0), fr.col(1));
	EXPECT_GT(std::abs(Bn.determinant()), 1e-10) << "Bn should be invertible";
}

TEST(FundamentalForms, Compile1ring_Sphere) {
	// 球面上的 1-ring：H ≈ 1/R, K ≈ 1/R^2
	const double R = 2.0;
	Eigen::Vector3d center(0, 0, R); // 北极
	std::vector<Eigen::Vector3d> ring;
	const double dphi = 0.1; // 约 5.7 度
	for (int i = 0; i < 6; ++i) {
		double theta = i * M_PI / 3.0;
		ring.push_back(Eigen::Vector3d(R * std::sin(dphi) * std::cos(theta),
									   R * std::sin(dphi) * std::sin(theta),
									   R * std::cos(dphi)));
	}
	xtpms::Compile1ring cr(center, ring);
	// H 的符号取决于法向约定；球面 |H| = 1/R
	EXPECT_NEAR(std::abs(cr.H), 1.0 / R, 0.1) << "Sphere |mean curvature| should be ~1/R";
	EXPECT_NEAR(cr.K, 1.0 / (R * R), 0.1) << "Sphere Gauss curvature should be ~1/R^2";
}

TEST(AsymptoticConductivity, NonUniformPeriod_ADC) {
	// 三轴不等周期的 Gyroid，验证 kA 的计算
	const Vec3d hp(0.5, 0.7, 0.4);
	auto mesh = makeClosedGyroid(hp, 20);
	ASSERT_EQ(countBoundaryEdges(mesh), 0);
	ASSERT_GT(mesh.n_faces(), 100u);

	auto geom = xtpms::computeVertexGeometry(mesh);
	Eigen::MatrixX3d u;
	Eigen::Matrix3d kA = xtpms::solveAsymptoticConductivity(mesh, geom, u);
	double apac = kA.trace() / 3.0;

	std::cout << "=== Non-uniform period Gyroid (hp=" << hp[0] << "," << hp[1] << "," << hp[2] << ") ===\n";
	std::cout << "kA =\n" << kA << "\n";
	std::cout << "APAC = " << apac << "\n";

	OpenMesh::IO::write_mesh(mesh, "gyroid_triaxis_adc.obj");

	// Gyroid 的 APAC 理论值为 2/3，非均匀周期下因为离散化精度可能偏离
	// 但各轴 kA 分量应该不同（反映周期不等性），且 APAC 仍接近 2/3
	EXPECT_NEAR(apac, 2.0 / 3.0, 0.05) << "APAC should be near 2/3 for Gyroid";
	EXPECT_GT(kA(0, 0), 0.0);
	EXPECT_GT(kA(1, 1), 0.0);
	EXPECT_GT(kA(2, 2), 0.0);
	// 非均匀周期下对角线不应完全相等
	EXPECT_GT(std::abs(kA(0, 0) - kA(1, 1)), 0.01)
		<< "Non-uniform period should break isotropy";
	// 对称性
	for (int i = 0; i < 3; ++i)
		for (int j = 0; j < i; ++j)
			EXPECT_NEAR(kA(i, j), kA(j, i), 1e-8);
}

TEST(AsymptoticConductivity, TailorAPAC_NonUniformPeriod) {
	// 三轴不等周期的 Gyroid，用 apac 目标做优化，开启 remesh + surgery
	const Vec3d hp(0.5, 0.7, 0.4);
	auto mesh = makeClosedGyroid(hp, 20);
	ASSERT_EQ(countBoundaryEdges(mesh), 0);
	ASSERT_GT(mesh.n_faces(), 100u);

	// 优化前的 kA
	double apac_before;
	{
		auto geom = xtpms::computeVertexGeometry(mesh);
		Eigen::MatrixX3d u;
		Eigen::Matrix3d kA0 = xtpms::solveAsymptoticConductivity(mesh, geom, u);
		apac_before = kA0.trace() / 3.0;
		std::cout << "=== Before optimization ===\n";
		std::cout << "kA =\n" << kA0 << "\n";
		std::cout << "APAC = " << apac_before << "\n";
		mesh.saveUnitCell("tailor_apac_triaxis_before.obj");
	}

	// 优化
	xtpms::TailorADCOptions opts;
	opts.objectiveType = "apac";
	opts.maxIter = 50;
	opts.maxStep = 1.0;
	opts.convergeTol = 1e-5;
	opts.stepTol = 1e-4;
	opts.mcfWeight = 0.1;

	opts.enableRemesh = true;
	opts.remeshOpts.outerIter = 1;
	opts.remeshOpts.innerIter = 5;
	opts.remeshOpts.adaptiveEps = 1.0;

	opts.enableSurgery = false;
	opts.outputDir = ".";
	opts.saveInterval = 10;

	xtpms::tailorADC(mesh, opts);

	// 用最后一组参数的结果做后续检查
	auto geom = xtpms::computeVertexGeometry(mesh);
	Eigen::MatrixX3d u;
	Eigen::Matrix3d kA = xtpms::solveAsymptoticConductivity(mesh, geom, u);
	double apac_after = kA.trace() / 3.0;

	std::cout << "=== After optimization ===\n";
	std::cout << "kA =\n" << kA << "\n";
	std::cout << "APAC = " << apac_after << "\n";
	std::cout << "APAC change: " << apac_before << " -> " << apac_after << "\n";

	// 曲率分布统计
	{
		double maxH = 0, maxK = 0, maxTotalK = 0;
		double sumH = 0, sumK = 0, sumTotalK = 0;
		int nvv = static_cast<int>(mesh.n_vertices());
		for (int i = 0; i < nvv; ++i) {
			double H = geom.vrings[i].H;
			double K = geom.vrings[i].K;
			double totalK = 4*H*H - 2*K;
			maxH = std::max(maxH, std::abs(H));
			maxK = std::max(maxK, std::abs(K));
			maxTotalK = std::max(maxTotalK, std::abs(totalK));
			sumH += std::abs(H);
			sumK += std::abs(K);
			sumTotalK += std::abs(totalK);
		}
		std::cout << "\n=== Curvature stats ===\n";
		std::cout << "|H|:  max=" << maxH << " avg=" << sumH/nvv << "\n";
		std::cout << "|K|:  max=" << maxK << " avg=" << sumK/nvv << "\n";
		std::cout << "|4H²-2K|: max=" << maxTotalK << " avg=" << sumTotalK/nvv << "\n";

		// 自适应目标边长分布
		double flatLen = opts.remeshOpts.targetLength;
		if (flatLen < 0) {
			double totalEdgeLen = 0; int edgeCnt = 0;
			for (auto e = mesh.edges_begin(); e != mesh.edges_end(); ++e) {
				auto he = mesh.halfedge_handle(*e, 0);
				auto ev = mesh.point(mesh.to_vertex_handle(he)) - mesh.point(mesh.from_vertex_handle(he));
				totalEdgeLen += ev.norm(); ++edgeCnt;
			}
			flatLen = totalEdgeLen / edgeCnt;
		}
		double eps = 1.0 / opts.remeshOpts.adaptiveEps;
		double minTarget = 1e30, maxTarget = 0, sumTarget = 0;
		int edgeCnt2 = 0;
		for (auto e = mesh.edges_begin(); e != mesh.edges_end(); ++e) {
			// 简单估算：用两端点的平均 totalK
			auto he = mesh.halfedge_handle(*e, 0);
			int v0 = mesh.from_vertex_handle(he).idx();
			int v1 = mesh.to_vertex_handle(he).idx();
			double H0 = geom.vrings[v0].H, K0 = geom.vrings[v0].K;
			double H1 = geom.vrings[v1].H, K1 = geom.vrings[v1].K;
			double avgTK = ((4*H0*H0-2*K0) + (4*H1*H1-2*K1)) / 2.0;
			double L = flatLen * eps / (std::sqrt(std::abs(avgTK)) + eps);
			minTarget = std::min(minTarget, L);
			maxTarget = std::max(maxTarget, L);
			sumTarget += L;
			++edgeCnt2;
		}
		std::cout << "flatLen=" << flatLen << " eps=" << eps << "\n";
		std::cout << "adaptive target: min=" << minTarget << " max=" << maxTarget << " avg=" << sumTarget/edgeCnt2 << "\n";

		// 实际边长分布
		double minEdge = 1e30, maxEdge = 0, sumEdge = 0;
		for (auto e = mesh.edges_begin(); e != mesh.edges_end(); ++e) {
			auto he = mesh.halfedge_handle(*e, 0);
			auto ev = mesh.point(mesh.to_vertex_handle(he)) - mesh.point(mesh.from_vertex_handle(he));
			double len = ev.norm();
			minEdge = std::min(minEdge, len);
			maxEdge = std::max(maxEdge, len);
			sumEdge += len;
		}
		std::cout << "actual edge len: min=" << minEdge << " max=" << maxEdge << " avg=" << sumEdge/edgeCnt2 << "\n";
	}

	mesh.saveUnitCell("tailor_apac_triaxis_after.obj");

	// Post MCF smooth：纯 MCF 跑若干步看效果
	{
		std::cout << "\n=== Post MCF smooth ===\n";
		auto geomMcf = xtpms::computeVertexGeometry(mesh);
		for (int mcfIter = 0; mcfIter < 20; ++mcfIter) {
			double mcfStep = 0.01;
			double maxLx = 0;
			for (auto v_it = mesh.vertices_begin(); v_it != mesh.vertices_end(); ++v_it) {
				int vi = (*v_it).idx();
				Eigen::Vector3d lx(
					geomMcf.vrings[vi].Lx[0],
					geomMcf.vrings[vi].Lx[1],
					geomMcf.vrings[vi].Lx[2]);
				maxLx = std::max(maxLx, lx.norm());
				auto p = mesh.point(*v_it);
				mesh.set_point(*v_it, Vec3d(
					p[0] + static_cast<xtpms::DefaultTriMesh::Scalar>(mcfStep * lx[0]),
					p[1] + static_cast<xtpms::DefaultTriMesh::Scalar>(mcfStep * lx[1]),
					p[2] + static_cast<xtpms::DefaultTriMesh::Scalar>(mcfStep * lx[2])));
			}
			geomMcf = xtpms::computeVertexGeometry(mesh);
			Eigen::MatrixX3d uMcf;
			Eigen::Matrix3d kMcf = xtpms::solveAsymptoticConductivity(mesh, geomMcf, uMcf);
			double apacMcf = kMcf.trace() / 3.0;
			if (mcfIter % 5 == 0 || mcfIter == 19)
				std::cout << "  mcf iter=" << mcfIter << " APAC=" << apacMcf
						  << " maxLx=" << maxLx << "\n";
		}
		mesh.saveUnitCell("tailor_apac_triaxis_after_mcf.obj");
		auto geomFinal = xtpms::computeVertexGeometry(mesh);
		Eigen::MatrixX3d uFinal;
		Eigen::Matrix3d kFinal = xtpms::solveAsymptoticConductivity(mesh, geomFinal, uFinal);
		std::cout << "  Final APAC after MCF = " << kFinal.trace() / 3.0 << "\n";
	}

	// splitUnitCell 测试
	{
		int nvBefore = static_cast<int>(mesh.n_vertices());
		int nfBefore = static_cast<int>(mesh.n_faces());
		int bndBefore = countBoundaryEdges(mesh);

		mesh.splitUnitCell();

		int nvAfter = static_cast<int>(mesh.n_vertices());
		int nfAfter = static_cast<int>(mesh.n_faces());
		int bndAfter = countBoundaryEdges(mesh);

		std::cout << "\n=== splitUnitCell ===\n";
		std::cout << "Before: nv=" << nvBefore << " nf=" << nfBefore << " bnd=" << bndBefore << "\n";
		std::cout << "After:  nv=" << nvAfter << " nf=" << nfAfter << " bnd=" << bndAfter << "\n";

		OpenMesh::IO::write_mesh(mesh, "tailor_apac_triaxis_split.obj");
		std::cout << "Saved tailor_apac_triaxis_split.obj\n";

		EXPECT_GT(bndAfter, 0) << "splitUnitCell should produce boundary edges";
	}

	EXPECT_GT(apac_after, 0.5) << "APAC should remain reasonable";
	EXPECT_GT(mesh.n_faces(), 0u);
}

TEST(AsymptoticConductivity, TailorAPAC_PerturbedP) {
	// 加三周期扰动的 Schwarz P：不再是极小曲面
	const Vec3d hp(1.0, 1.0, 1.0);
	const double Lx = 2.0, Ly = 2.0, Lz = 2.0;
	auto src = makeTPMSMC(hp, [Lx, Ly, Lz](double x, double y, double z) {
		double px = 2.0 * M_PI * x / Lx, py = 2.0 * M_PI * y / Ly, pz = 2.0 * M_PI * z / Lz;
		// Schwarz P + 三周期扰动
		double base = std::cos(px) + std::cos(py) + std::cos(pz);
		double perturb = 0.6 * std::sin(2 * px) * std::sin(py)
					   + 0.5 * std::cos(3 * py) * std::sin(pz)
					   + 0.4 * std::sin(px) * std::cos(2 * pz)
					   + 0.3 * std::sin(2 * px) * std::cos(2 * py)
					   + 0.35 * std::cos(px) * std::sin(3 * pz);
		return base + perturb;
	}, 24);
	auto mesh = makeClosedTPMS(hp, src);
	ASSERT_EQ(countBoundaryEdges(mesh), 0);
	ASSERT_GT(mesh.n_faces(), 100u);

	// 优化前
	double apac_before;
	{
		auto geom = xtpms::computeVertexGeometry(mesh);
		Eigen::MatrixX3d u;
		Eigen::Matrix3d kA0 = xtpms::solveAsymptoticConductivity(mesh, geom, u);
		apac_before = kA0.trace() / 3.0;
		std::cout << "=== Perturbed P Before optimization ===\n";
		std::cout << "kA =\n" << kA0 << "\n";
		std::cout << "APAC = " << apac_before << "\n";
		mesh.saveUnitCell("perturbed_p_before.obj");
	}

	// 优化
	xtpms::TailorADCOptions opts;
	opts.objectiveType = "apac";
	opts.maxIter = 200;
	opts.maxStep = 0.3;
	opts.convergeTol = 1e-5;
	opts.stepTol = 1e-4;
	opts.preconditionStrength = 0.1;

	opts.enableRemesh = true;
	opts.remeshOpts.outerIter = 1;
	opts.remeshOpts.innerIter = 3;

	opts.enableSurgery = false;
	opts.mcfWeight = 0.1;  // 均曲率流正则化

	opts.outputDir = ".";
	opts.saveInterval = 1;

	xtpms::tailorADC(mesh, opts);

	// 优化后
	auto geom = xtpms::computeVertexGeometry(mesh);
	Eigen::MatrixX3d u;
	Eigen::Matrix3d kA = xtpms::solveAsymptoticConductivity(mesh, geom, u);
	double apac_after = kA.trace() / 3.0;

	std::cout << "=== Perturbed P After optimization ===\n";
	std::cout << "kA =\n" << kA << "\n";
	std::cout << "APAC = " << apac_after << "\n";
	std::cout << "APAC change: " << apac_before << " -> " << apac_after << "\n";

	mesh.saveUnitCell("perturbed_p_after.obj");

	EXPECT_GT(apac_after, apac_before - 0.01) << "APAC should improve or stay";
	EXPECT_GT(mesh.n_faces(), 0u);
}

TEST(AsymptoticConductivity, TailorAPAC_Rod3) {
	// 从文件读取 rod3-2.obj，周期 [0,2]^3
	namespace fs = std::filesystem;
	const fs::path meshPath(R"(tests/data/rod3-2.obj)");
	if (!fs::exists(meshPath)) {
		GTEST_SKIP() << "rod3-2.obj not found: " << meshPath.string();
	}

	xtpms::DefaultTriMesh src;
	ASSERT_TRUE(OpenMesh::IO::read_mesh(src, meshPath.string()));
	ASSERT_GT(src.n_vertices(), 0u);

	const Vec3d hp(1.0, 1.0, 1.0);  // 半周期 = 1，全周期 = 2
	// 先输出原始网格（merge 前）
	OpenMesh::IO::write_mesh(src, "rod3_raw.obj");

	auto mesh = makeClosedTPMS(hp, src);
	ASSERT_GT(mesh.n_faces(), 0u);
	std::cout << "rod3: nv=" << mesh.n_vertices() << " nf=" << mesh.n_faces()
			  << " bnd=" << countBoundaryEdges(mesh) << "\n";
	mesh.saveUnitCell("rod3_after_merge.obj");

	// 预处理：smooth + remesh 循环平滑连接处
	std::cout << "=== Pre-smoothing ===\n";
	for (int pre = 0; pre < 5; ++pre) {
		// MCF smooth
		auto geomPre = xtpms::computeVertexGeometry(mesh);
		double maxLx = 0;
		for (auto v_it = mesh.vertices_begin(); v_it != mesh.vertices_end(); ++v_it) {
			int vi = (*v_it).idx();
			auto& lx = geomPre.vrings[vi].Lx;
			double lxn = lx.norm();
			maxLx = std::max(maxLx, lxn);
			auto p = mesh.point(*v_it);
			double step = 0.05;
			mesh.set_point(*v_it, Vec3d(
				p[0] + static_cast<xtpms::DefaultTriMesh::Scalar>(step * lx[0]),
				p[1] + static_cast<xtpms::DefaultTriMesh::Scalar>(step * lx[1]),
				p[2] + static_cast<xtpms::DefaultTriMesh::Scalar>(step * lx[2])));
		}
		// remesh
		xtpms::RemeshOptions ropts;
		ropts.outerIter = 1;
		ropts.innerIter = 5;
		ropts.targetLength = 2.0 * 0.15;  // minPeriod * 0.15
		ropts.minLength = ropts.targetLength * 0.1;
		ropts.adaptiveEps = 1.0;
		xtpms::delaunayRemesh(mesh, ropts);
		bool hasBnd = false;
		for (auto e = mesh.edges_begin(); e != mesh.edges_end() && !hasBnd; ++e)
			if (mesh.is_boundary(*e)) hasBnd = true;
		if (hasBnd) mesh.mergePeriodBoundary();

		std::cout << "  pre " << pre << ": nv=" << mesh.n_vertices()
				  << " nf=" << mesh.n_faces() << " maxLx=" << maxLx << "\n";
	}
	mesh.saveUnitCell("rod3_pre_smoothed.obj");

	// 优化前
	double apac_before;
	{
		auto geom = xtpms::computeVertexGeometry(mesh);
		Eigen::MatrixX3d u;
		Eigen::Matrix3d kA0 = xtpms::solveAsymptoticConductivity(mesh, geom, u);
		apac_before = kA0.trace() / 3.0;
		std::cout << "=== Rod3 Before optimization ===\n";
		std::cout << "kA =\n" << kA0 << "\n";
		std::cout << "APAC = " << apac_before << "\n";
	}

	// 优化（和 NonUniformPeriod 相同参数）
	xtpms::TailorADCOptions opts;
	opts.objectiveType = "apac";
	opts.maxIter = 41;  // surgery 在 iter=40 执行，41 步看结果
	opts.maxStep = 1.0;
	opts.convergeTol = 1e-5;
	opts.stepTol = 1e-4;
	opts.mcfWeight = 0.1;

	opts.enableRemesh = true;
	opts.remeshOpts.outerIter = 1;
	opts.remeshOpts.innerIter = 5;
	opts.remeshOpts.adaptiveEps = 1.0;

	opts.enableSurgery = true;
	opts.surgeryInterval = 20;
	opts.surgeryStartIter = 40;
	opts.surgeryOpts.singularityTol = 50.0;

	opts.outputDir = ".";
	opts.saveInterval = 10;

	xtpms::tailorADC(mesh, opts);

	// 优化后
	auto geom = xtpms::computeVertexGeometry(mesh);
	Eigen::MatrixX3d u;
	Eigen::Matrix3d kA = xtpms::solveAsymptoticConductivity(mesh, geom, u);
	double apac_after = kA.trace() / 3.0;

	std::cout << "=== Rod3 After optimization ===\n";
	std::cout << "kA =\n" << kA << "\n";
	std::cout << "APAC = " << apac_after << "\n";
	std::cout << "APAC change: " << apac_before << " -> " << apac_after << "\n";

	mesh.saveUnitCell("rod3_after.obj");

	EXPECT_GT(apac_after, apac_before - 0.01);
	EXPECT_GT(mesh.n_faces(), 0u);
}
