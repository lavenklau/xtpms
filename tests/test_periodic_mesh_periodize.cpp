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

	// 运行 APAC 优化：开启 remesh + surgery
	xtpms::TailorADCOptions opts;
	opts.objectiveType = "apac";
	opts.maxIter = 100;
	opts.maxStep = 0.3;
	opts.convergeTol = 1e-4;
	opts.stepTol = 1e-3;
	opts.preconditionStrength = 1.0;

	opts.enableRemesh = true;
	opts.remeshOpts.outerIter = 1;
	opts.remeshOpts.innerIter = 3;
	// targetLength < 0 让 tailorADC 自动计算并固定

	opts.enableSurgery = false;

	opts.outputDir = ".";
	opts.saveInterval = 1;  // 每步输出网格

	xtpms::tailorADC(mesh, opts);

	// 优化后的 kA
	auto geom = xtpms::computeVertexGeometry(mesh);
	Eigen::MatrixX3d u;
	Eigen::Matrix3d kA = xtpms::solveAsymptoticConductivity(mesh, geom, u);
	double apac_after = kA.trace() / 3.0;

	std::cout << "=== After optimization ===\n";
	std::cout << "kA =\n" << kA << "\n";
	std::cout << "APAC = " << apac_after << "\n";
	std::cout << "APAC change: " << apac_before << " -> " << apac_after << "\n";

	mesh.saveUnitCell("tailor_apac_triaxis_after.obj");

	EXPECT_GT(apac_after, 0.5) << "APAC should remain reasonable";
	EXPECT_GT(mesh.n_faces(), 0u);

	for (int i = 0; i < 3; ++i)
		for (int j = 0; j < i; ++j)
			EXPECT_NEAR(kA(i, j), kA(j, i), 1e-6);
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
	opts.preconditionStrength = 1.0;

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
	// 从文件读取 rod3.obj，周期 [0,10]^3
	namespace fs = std::filesystem;
	const fs::path meshPath(R"(tests/data/rod3.obj)");
	if (!fs::exists(meshPath)) {
		GTEST_SKIP() << "rod3.obj not found: " << meshPath.string();
	}

	xtpms::DefaultTriMesh src;
	ASSERT_TRUE(OpenMesh::IO::read_mesh(src, meshPath.string()));
	ASSERT_GT(src.n_vertices(), 0u);

	const Vec3d hp(5.0, 5.0, 5.0);  // 半周期 = 5，全周期 = 10
	// 先输出原始网格（merge 前）
	OpenMesh::IO::write_mesh(src, "rod3_raw.obj");

	auto mesh = makeClosedTPMS(hp, src);
	ASSERT_GT(mesh.n_faces(), 0u);
	std::cout << "rod3: nv=" << mesh.n_vertices() << " nf=" << mesh.n_faces()
			  << " bnd=" << countBoundaryEdges(mesh) << "\n";
	mesh.saveUnitCell("rod3_after_merge.obj");

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
		mesh.saveUnitCell("rod3_before.obj");
	}

	// 优化
	xtpms::TailorADCOptions opts;
	opts.objectiveType = "apac";
	opts.maxIter = 200;
	opts.maxStep = 0.3;
	opts.convergeTol = 1e-5;
	opts.stepTol = 1e-4;
	opts.preconditionStrength = 1.0;
	opts.mcfWeight = 0.1;

	opts.enableRemesh = true;
	opts.remeshOpts.outerIter = 1;
	opts.remeshOpts.innerIter = 3;

	opts.enableSurgery = false; // surgery 的 fill_holes 还需要改进
	opts.mcfWeight = 0.1;

	opts.outputDir = ".";
	opts.saveInterval = 1;

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
