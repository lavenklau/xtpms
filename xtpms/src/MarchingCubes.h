#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include "MeshTypes.h"

namespace xtpms {

struct VoxelIndex {
	std::int32_t i{};
	std::int32_t j{};
	std::int32_t k{};

	bool operator==(VoxelIndex o) const noexcept { return i == o.i && j == o.j && k == o.k; }
};

struct VoxelIndexHash {
	std::size_t operator()(VoxelIndex v) const noexcept;
};

/// One sample in the global nodal list: position and scalar used for iso-surfacing.
struct LevelSetNode {
	double x{};
	double y{};
	double z{};
	double value{};
};

/// Indices into `nodes` for the eight cube corners, matching standard marching-cubes order:
/// 0=(0,0,0), 1=(1,0,0), 2=(1,1,0), 3=(0,1,0), 4=(0,0,1), 5=(1,0,1), 6=(1,1,1), 7=(0,1,1) in
/// cell-local axes.
using VoxelCornerNodeIndices = std::array<std::size_t, 8>;

using SparseVoxelCornerMap = std::unordered_map<VoxelIndex, VoxelCornerNodeIndices, VoxelIndexHash>;

struct ExtractMarchingCubesOptions {
	/// If true, reuse one mesh vertex per undirected pair of nodal indices along a cut edge
	/// (consistent tessellation across adjacent cells).
	bool weldSharedEdges{true};
};

/// Classic marching cubes on a sparse set of voxels. Skips a cell if any corner index is out of
/// range for `nodes`. Corner `c` counts as "inside" when `nodes[corner[c]].value < isoValue`
/// (negative-distance style SDF with iso 0).
void marchingCubesExtract(const SparseVoxelCornerMap& voxels,
						  const std::vector<LevelSetNode>& nodes,
						  double isoValue,
						  std::vector<std::array<double, 3>>& outVertices,
						  std::vector<std::array<std::size_t, 3>>& outTriangles,
						  const ExtractMarchingCubesOptions& opts = ExtractMarchingCubesOptions{});

void marchingCubesExtractToTriMesh(
	const SparseVoxelCornerMap& voxels,
	const std::vector<LevelSetNode>& nodes,
	double isoValue,
	DefaultTriMesh& outMesh,
	const ExtractMarchingCubesOptions& opts = ExtractMarchingCubesOptions{});

} // namespace xtpms
