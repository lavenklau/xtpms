#include "MarchingCubes.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>

#include "MarchingCubesTables.inl"

namespace xtpms {

namespace {

// Endpoints (local corner ids 0..7) for edges 0..11 — same layout as Paul Bourke / OctoMap tables.
constexpr int kEdgeCorners[12][2] = {
	{0, 1},
	{1, 2},
	{2, 3},
	{3, 0},
	{4, 5},
	{5, 6},
	{6, 7},
	{7, 4},
	{0, 4},
	{1, 5},
	{2, 6},
	{3, 7},
};

struct PairHash {
	std::size_t operator()(const std::pair<std::size_t, std::size_t>& p) const noexcept {
		return p.first ^ (p.second + 0x9e3779b9u + (p.first << 6) + (p.first >> 2));
	}
};

} // namespace

std::size_t VoxelIndexHash::operator()(VoxelIndex v) const noexcept {
	auto u = [](std::int32_t a) -> std::size_t {
		return static_cast<std::size_t>(static_cast<std::uint32_t>(a));
	};
	std::size_t h = u(v.i) * 73856093u;
	h ^= u(v.j) * 19349663u + 0x9e3779b9u + (h << 6) + (h >> 2);
	h ^= u(v.k) * 83492791u + 0x9e3779b9u + (h << 6) + (h >> 2);
	return h;
}

void marchingCubesExtract(
	const SparseVoxelCornerMap& voxels,
	const std::vector<LevelSetNode>& nodes,
	double isoValue,
	std::vector<std::array<double, 3>>& outVertices,
	std::vector<std::array<std::size_t, 3>>& outTriangles,
	const ExtractMarchingCubesOptions& opts) {
	outVertices.clear();
	outTriangles.clear();

	std::unordered_map<std::pair<std::size_t, std::size_t>, std::size_t, PairHash> edgeCache;
	if (opts.weldSharedEdges) {
		edgeCache.reserve(std::min<std::size_t>(voxels.size() * 6u, 1u << 20));
	}

	for (const auto& kv : voxels) {
		const VoxelCornerNodeIndices& cornerIdx = kv.second;
		double s[8]{};
		double px[8]{};
		double py[8]{};
		double pz[8]{};
		bool bad = false;
		for (int c = 0; c < 8; ++c) {
			const std::size_t id = cornerIdx[static_cast<std::size_t>(c)];
			if (id >= nodes.size()) {
				bad = true;
				break;
			}
			const LevelSetNode& n = nodes[id];
			s[c] = n.value;
			px[c] = n.x;
			py[c] = n.y;
			pz[c] = n.z;
		}
		if (bad) {
			continue;
		}

		int cubeIndex = 0;
		for (int c = 0; c < 8; ++c) {
			if (s[c] < isoValue) {
				cubeIndex |= (1 << c);
			}
		}

		const int edgeFlags = mc_detail::edgeTable[cubeIndex];
		if (edgeFlags == 0) {
			continue;
		}

		auto vertexOnEdge = [&](int edge) -> std::size_t {
			const int a = kEdgeCorners[edge][0];
			const int b = kEdgeCorners[edge][1];
			const std::size_t ia = cornerIdx[static_cast<std::size_t>(a)];
			const std::size_t ib = cornerIdx[static_cast<std::size_t>(b)];
			const std::size_t lo = ia < ib ? ia : ib;
			const std::size_t hi = ia < ib ? ib : ia;
			const auto key = std::make_pair(lo, hi);

			if (opts.weldSharedEdges) {
				const auto it = edgeCache.find(key);
				if (it != edgeCache.end()) {
					return it->second;
				}
			}

			const double sa = s[a];
			const double sb = s[b];
			double denom = sb - sa;
			double t = 0.5;
			if (std::abs(denom) > 1e-30) {
				t = (isoValue - sa) / denom;
				t = std::clamp(t, 0.0, 1.0);
			}

			const std::size_t vid = outVertices.size();
			outVertices.push_back({
				px[a] + t * (px[b] - px[a]),
				py[a] + t * (py[b] - py[a]),
				pz[a] + t * (pz[b] - pz[a]),
			});

			if (opts.weldSharedEdges) {
				edgeCache.emplace(key, vid);
			}
			return vid;
		};

		const int* triRow = mc_detail::triTable[cubeIndex];
		for (int k = 0; k < 16 && triRow[k] != -1; k += 3) {
			const std::size_t i0 = vertexOnEdge(triRow[k]);
			const std::size_t i1 = vertexOnEdge(triRow[k + 1]);
			const std::size_t i2 = vertexOnEdge(triRow[k + 2]);
			outTriangles.push_back({i0, i1, i2});
		}
	}
}

void marchingCubesExtractToTriMesh(
	const SparseVoxelCornerMap& voxels,
	const std::vector<LevelSetNode>& nodes,
	double isoValue,
	DefaultTriMesh& outMesh,
	const ExtractMarchingCubesOptions& opts) {
	std::vector<std::array<double, 3>> verts;
	std::vector<std::array<std::size_t, 3>> tris;
	marchingCubesExtract(voxels, nodes, isoValue, verts, tris, opts);

	outMesh.clear();

	std::vector<DefaultTriMesh::VertexHandle> handles;
	handles.reserve(verts.size());
	for (const auto& p : verts) {
		handles.push_back(outMesh.add_vertex(DefaultTriMesh::Point(
			static_cast<DefaultTriMesh::Scalar>(p[0]),
			static_cast<DefaultTriMesh::Scalar>(p[1]),
			static_cast<DefaultTriMesh::Scalar>(p[2]))));
	}

	for (const auto& tri : tris) {
		outMesh.add_face(handles[tri[0]], handles[tri[1]], handles[tri[2]]);
	}
}

} // namespace xtpms
