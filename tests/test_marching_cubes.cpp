#include <gtest/gtest.h>

#include <OpenMesh/Core/IO/MeshIO.hh>

#include <array>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

#include "MarchingCubes.h"

namespace {

TEST(MarchingCubes, singleCellOneCornerInsideProducesOneTriangle) {
	// Unit cube corners as nodal samples; only vertex 0 is "inside" (negative).
	std::vector<xtpms::LevelSetNode> nodes(8);
	nodes[0] = {0.0, 0.0, 0.0, -1.0};
	nodes[1] = {1.0, 0.0, 0.0, 1.0};
	nodes[2] = {1.0, 1.0, 0.0, 1.0};
	nodes[3] = {0.0, 1.0, 0.0, 1.0};
	nodes[4] = {0.0, 0.0, 1.0, 1.0};
	nodes[5] = {1.0, 0.0, 1.0, 1.0};
	nodes[6] = {1.0, 1.0, 1.0, 1.0};
	nodes[7] = {0.0, 1.0, 1.0, 1.0};

	xtpms::SparseVoxelCornerMap voxels;
	xtpms::VoxelCornerNodeIndices corners{};
	for (std::size_t i = 0; i < 8; ++i) {
		corners[i] = i;
	}
	voxels[{0, 0, 0}] = corners;

	std::vector<std::array<double, 3>> verts;
	std::vector<std::array<std::size_t, 3>> tris;
	xtpms::marchingCubesExtract(voxels, nodes, 0.0, verts, tris, {});

	ASSERT_EQ(tris.size(), 1u);
	ASSERT_EQ(verts.size(), 3u);

	// Midpoints on edges from node 0: (0.5,0,0), (0,0.5,0), (0,0,0.5) in some order.
	auto near = [](double a, double b) { return std::abs(a - b) < 1e-9; };
	std::array<double, 3> exp[3] = {
		{0.5, 0.0, 0.0},
		{0.0, 0.5, 0.0},
		{0.0, 0.0, 0.5},
	};
	for (const auto& p : verts) {
		bool ok = false;
		for (int e = 0; e < 3; ++e) {
			if (near(p[0], exp[e][0]) && near(p[1], exp[e][1]) && near(p[2], exp[e][2])) {
				ok = true;
				break;
			}
		}
		EXPECT_TRUE(ok);
	}
}

TEST(MarchingCubes, openMeshExtractMatchesBufferPath) {
	std::vector<xtpms::LevelSetNode> nodes(8);
	for (int z = 0; z < 2; ++z) {
		for (int y = 0; y < 2; ++y) {
			for (int x = 0; x < 2; ++x) {
				const int id = x + 2 * y + 4 * z;
				nodes[static_cast<std::size_t>(id)] = {static_cast<double>(x), static_cast<double>(y), static_cast<double>(z),
					(static_cast<double>(x) + static_cast<double>(y) + static_cast<double>(z) - 1.2)};
			}
		}
	}

	xtpms::SparseVoxelCornerMap voxels;
	xtpms::VoxelCornerNodeIndices corners{};
	for (std::size_t i = 0; i < 8; ++i) {
		corners[i] = i;
	}
	voxels[{0, 0, 0}] = corners;

	std::vector<std::array<double, 3>> v0;
	std::vector<std::array<std::size_t, 3>> f0;
	xtpms::marchingCubesExtract(voxels, nodes, 0.0, v0, f0, {});

	xtpms::DefaultTriMesh mesh;
	xtpms::marchingCubesExtractToTriMesh(voxels, nodes, 0.0, mesh, {});

	EXPECT_EQ(mesh.n_faces(), f0.size());
	EXPECT_EQ(mesh.n_vertices(), v0.size());
}

// Triply periodic gyroid (iso = 0). Domain is one unit cell [0,1]^3; the implicit is periodic in space,
// but the mesh is naturally open on the six box faces (no periodic gluing).
TEST(MarchingCubes, periodicGyroidGridWritesTriMeshFile) {
	const int nx = 28;
	const int ny = 28;
	const int nz = 28;
	const auto gyroid = [](double x, double y, double z) {
		const double tx = 2.0 * M_PI * x;
		const double ty = 2.0 * M_PI * y;
		const double tz = 2.0 * M_PI * z;
		return std::sin(tx) * std::cos(ty) + std::sin(ty) * std::cos(tz) + std::sin(tz) * std::cos(tx);
	};

	const int nxp = nx + 1;
	const int nyp = ny + 1;
	const auto nodeOf = [nxp, nyp](int ix, int iy, int iz) -> std::size_t {
		return static_cast<std::size_t>(ix + nxp * (iy + nyp * iz));
	};

	std::vector<xtpms::LevelSetNode> nodes(static_cast<std::size_t>(nxp * nyp * (nz + 1)));
	for (int iz = 0; iz <= nz; ++iz) {
		for (int iy = 0; iy <= ny; ++iy) {
			for (int ix = 0; ix <= nx; ++ix) {
				const double x = static_cast<double>(ix) / static_cast<double>(nx);
				const double y = static_cast<double>(iy) / static_cast<double>(ny);
				const double z = static_cast<double>(iz) / static_cast<double>(nz);
				const std::size_t id = nodeOf(ix, iy, iz);
				nodes[id] = {x, y, z, gyroid(x, y, z)};
			}
		}
	}

	xtpms::SparseVoxelCornerMap voxels;
	voxels.reserve(static_cast<std::size_t>(nx * ny * nz));
	for (int k = 0; k < nz; ++k) {
		for (int j = 0; j < ny; ++j) {
			for (int i = 0; i < nx; ++i) {
				xtpms::VoxelCornerNodeIndices c{};
				c[0] = nodeOf(i, j, k);
				c[1] = nodeOf(i + 1, j, k);
				c[2] = nodeOf(i + 1, j + 1, k);
				c[3] = nodeOf(i, j + 1, k);
				c[4] = nodeOf(i, j, k + 1);
				c[5] = nodeOf(i + 1, j, k + 1);
				c[6] = nodeOf(i + 1, j + 1, k + 1);
				c[7] = nodeOf(i, j + 1, k + 1);
				voxels[{static_cast<std::int32_t>(i), static_cast<std::int32_t>(j), static_cast<std::int32_t>(k)}] = c;
			}
		}
	}

	xtpms::DefaultTriMesh mesh;
	xtpms::marchingCubesExtractToTriMesh(voxels, nodes, 0.0, mesh, {});

	ASSERT_GT(mesh.n_faces(), 500u);
	ASSERT_GT(mesh.n_vertices(), 200u);

	const double tol = 1.0e-6;
	for (auto vh : mesh.vertices()) {
		const auto& p = mesh.point(vh);
		for (int a = 0; a < 3; ++a) {
			EXPECT_GE(static_cast<double>(p[a]), -tol);
			EXPECT_LE(static_cast<double>(p[a]), 1.0 + tol);
		}
	}

	namespace fs = std::filesystem;
	fs::path outPath;
#ifdef XTPMS_TEST_MC_MESH_PATH
	outPath = fs::path(XTPMS_TEST_MC_MESH_PATH);
#elif defined(XTPMS_MC_RESULTS_DIR)
	outPath = fs::path(XTPMS_MC_RESULTS_DIR) / "xtpms_marching_cubes_gyroid.obj";
#else
	outPath = fs::current_path() / "results" / "xtpms_marching_cubes_gyroid.obj";
#endif
	{
		std::error_code ec;
		fs::create_directories(outPath.parent_path(), ec);
		ASSERT_FALSE(ec) << "create_directories failed: " << outPath.parent_path().string();
	}
	ASSERT_TRUE(OpenMesh::IO::write_mesh(mesh, outPath.string()))
		<< "write_mesh failed: " << outPath.string();

	ASSERT_TRUE(fs::exists(outPath));
	EXPECT_GT(fs::file_size(outPath), 100u);
}

} // namespace
