/**
 * @file test_periodic_spatial_hash.cpp
 * @brief Unit tests for PeriodicSpatialHash.h (insert, neighborhood occupancy query, toroidal neighbors).
 */

#include "PeriodicSpatialHash.h"
#include "PeriodicSpatialHashGeometries.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace xtpms {
namespace {

using Hash = PeriodicSpatialHash<std::uint32_t>;
using Vec3d = Hash::Vec3d;
using VoxelKey = Hash::VoxelKey;

Vec3d v3(double x, double y, double z) {
	using S = typename Vec3d::value_type;
	return Vec3d(static_cast<S>(x), static_cast<S>(y), static_cast<S>(z));
}

struct FixedVoxelsGeom {
	std::vector<VoxelKey> keys;
	template <class HashT, class EmitFn>
	void getOccupiedVox(const HashT& /*hash*/, EmitFn&& emit) const {
		for (const VoxelKey& k : keys) emit(k);
	}
};

TEST(PeriodicSpatialHash, VoxelDimsAndVoxelAtOrigin) {
	Hash h(v3(1.0, 1.0, 1.0), 1.0);
	EXPECT_EQ(h.voxelDim(0), 2);
	EXPECT_EQ(h.voxelDim(1), 2);
	EXPECT_EQ(h.voxelDim(2), 2);

	const VoxelKey k = h.voxelAtWorld(v3(0.0, 0.0, 0.0));
	EXPECT_EQ(k.ix, 1);
	EXPECT_EQ(k.iy, 1);
	EXPECT_EQ(k.iz, 1);
}

TEST(PeriodicSpatialHash, Idx2WorldCenterRoundTripsWithVoxelAtWorld) {
	Hash h(v3(1.0, 1.0, 1.0), 1.0);
	for (int32_t ix = 0; ix < h.voxelDim(0); ++ix) {
		for (int32_t iy = 0; iy < h.voxelDim(1); ++iy) {
			for (int32_t iz = 0; iz < h.voxelDim(2); ++iz) {
				const VoxelKey k{ix, iy, iz};
				const VoxelKey k2 = h.voxelAtWorld(h.idx2World(k));
				EXPECT_EQ(k2.ix, k.ix);
				EXPECT_EQ(k2.iy, k.iy);
				EXPECT_EQ(k2.iz, k.iz);
			}
		}
	}
}

TEST(PeriodicSpatialHash, WrapVectorReducesAlongPeriod) {
	Hash h(v3(1.0, 1.0, 1.0), 1.0);
	const Vec3d w = h.wrapVector(v3(2.7, -3.4, 0.0));
	// DefaultTraits::Point may be float; allow ULP slack.
	EXPECT_NEAR(static_cast<double>(w[0]), 0.7, 1e-5);
	EXPECT_NEAR(static_cast<double>(w[1]), 0.6, 1e-5);
	EXPECT_NEAR(static_cast<double>(w[2]), 0.0, 1e-5);
}

TEST(PeriodicSpatialHash, InsertCustomGeometry) {
	Hash h(v3(1.0, 1.0, 1.0), 1.0);
	FixedVoxelsGeom g;
	g.keys = {VoxelKey{0, 1, 0}, VoxelKey{1, 1, 1}};
	h.insert(g, 42U);

	const auto& a = h.ids(VoxelKey{0, 1, 0});
	const auto& b = h.ids(VoxelKey{1, 1, 1});
	ASSERT_EQ(a.size(), 1u);
	ASSERT_EQ(b.size(), 1u);
	EXPECT_EQ(a[0], 42U);
	EXPECT_EQ(b[0], 42U);
	EXPECT_TRUE(h.ids(VoxelKey{0, 0, 0}).empty());
}

TEST(PeriodicSpatialHash, ClearRemovesEntries) {
	Hash h(v3(1.0, 1.0, 1.0), 1.0);
	FixedVoxelsGeom g;
	g.keys = {VoxelKey{0, 0, 0}};
	h.insert(g, 7U);
	EXPECT_FALSE(h.ids(VoxelKey{0, 0, 0}).empty());
	h.clear();
	EXPECT_TRUE(h.ids(VoxelKey{0, 0, 0}).empty());
}

TEST(PeriodicSpatialHash, OccupiedVoxelsNearPoint_EmptyWhenNoGeometry) {
	Hash h(v3(1.0, 1.0, 1.0), 1.0);
	const auto v = h.occupiedVoxelsNearPoint(v3(0.0, 0.0, 0.0), 1);
	EXPECT_TRUE(v.empty());
}

TEST(PeriodicSpatialHash, OccupiedVoxelsNearPoint_FindsInsertedSegment) {
	Hash h(v3(1.0, 1.0, 1.0), 1.0);
	SegmentGeometry seg;
	seg.a = v3(-0.5, 0.0, 0.0);
	seg.b = v3(0.5, 0.0, 0.0);
	h.insert(seg, 11U);

	const auto occ = h.occupiedVoxelsNearPoint(v3(0.1, 0.2, 0.0), 1);
	ASSERT_FALSE(occ.empty());
	bool found = false;
	for (const auto& o : occ) {
		if (o.id == 11U) found = true;
	}
	EXPECT_TRUE(found);
}

TEST(PeriodicSpatialHash, OccupiedVoxelsNearPoint_SameIdsAfterQueryPointPeriodShift) {
	Hash h(v3(1.0, 1.0, 1.0), 1.0);
	SegmentGeometry seg;
	seg.a = v3(-0.5, 0.0, 0.0);
	seg.b = v3(0.5, 0.0, 0.0);
	h.insert(seg, 11U);

	auto collectIds = [&](double x, double y, double z) {
		std::vector<std::uint32_t> ids;
		for (const auto& o : h.occupiedVoxelsNearPoint(v3(x, y, z), 1)) ids.push_back(o.id);
		return ids;
	};

	const auto a = collectIds(0.1, 0.2, 0.0);
	const auto b = collectIds(0.1 + 4.0, 0.2, 0.0);
	const auto c = collectIds(0.1 + 4.0, 0.2 - 6.0, 2.0);
	ASSERT_FALSE(a.empty());
	ASSERT_FALSE(b.empty());
	ASSERT_FALSE(c.empty());
	EXPECT_NE(std::find(a.begin(), a.end(), 11U), a.end());
	EXPECT_NE(std::find(b.begin(), b.end(), 11U), b.end());
	EXPECT_NE(std::find(c.begin(), c.end(), 11U), c.end());
}

TEST(PeriodicSpatialHash, OccupiedVoxelsNearPoint_SegmentInsertedWithWorldPeriodShift) {
	Hash h(v3(1.0, 1.0, 1.0), 1.0);
	SegmentGeometry segShifted;
	segShifted.a = v3(-0.5 + 10.0, 0.0, 0.0);
	segShifted.b = v3(0.5 + 10.0, 0.0, 0.0);
	h.insert(segShifted, 11U);

	const auto occ = h.occupiedVoxelsNearPoint(v3(0.1, 0.2, 0.0), 1);
	bool found = false;
	for (const auto& o : occ) {
		if (o.id == 11U) found = true;
	}
	EXPECT_TRUE(found);
}

TEST(PeriodicSpatialHash, OccupiedVoxelsNearPoint_TwoIdsSameVoxel) {
	Hash h(v3(1.0, 1.0, 1.0), 1.0);
	FixedVoxelsGeom g;
	g.keys = {VoxelKey{1, 1, 1}};
	h.insert(g, 100U);
	h.insert(g, 200U);

	const auto occ = h.occupiedVoxelsNearPoint(v3(0.0, 0.0, 0.0), 1);
	std::uint32_t n100 = 0;
	std::uint32_t n200 = 0;
	for (const auto& o : occ) {
		if (o.key.ix == 1 && o.key.iy == 1 && o.key.iz == 1) {
			if (o.id == 100U) ++n100;
			if (o.id == 200U) ++n200;
		}
	}
	EXPECT_EQ(n100, 1u);
	EXPECT_EQ(n200, 1u);
}

TEST(PeriodicSpatialHash, OccupiedVoxelsNearPoint_ToroidalNeighborhood) {
	Hash h(v3(1.0, 1.0, 1.0), 1.0);
	FixedVoxelsGeom g;
	g.keys = {VoxelKey{0, 0, 0}};
	h.insert(g, 55U);

	const auto occ = h.occupiedVoxelsNearPoint(v3(0.9, 0.9, 0.9), 1);
	bool hit = false;
	for (const auto& o : occ) {
		if (o.id == 55U && o.key.ix == 0 && o.key.iy == 0 && o.key.iz == 0) hit = true;
	}
	EXPECT_TRUE(hit);
}

TEST(PeriodicSpatialHash, OccupiedVoxelsNearPoint_RadiusZeroSkipsNeighbor) {
	Hash h(v3(1.0, 1.0, 1.0), 1.0);
	FixedVoxelsGeom g;
	g.keys = {VoxelKey{0, 0, 0}};
	h.insert(g, 66U);

	const auto occ = h.occupiedVoxelsNearPoint(v3(0.9, 0.9, 0.9), 0);
	EXPECT_TRUE(occ.empty());
}

// Long world-space x segment (many period wraps): occupancy must follow the covering-space line, not only
// the chord between independently wrapped endpoints (which would miss outer x voxels).
TEST(PeriodicSpatialHash, SegmentGeometryWorldLineOccupiesFullXSpan) {
	Hash h(v3(1.0, 1.0, 1.0), 0.25);
	ASSERT_EQ(h.voxelDim(0), 8);
	const int32_t iy = h.voxelAtWorld(v3(0.0, 0.0, 0.0)).iy;
	const int32_t iz = h.voxelAtWorld(v3(0.0, 0.0, 0.0)).iz;

	SegmentGeometry seg;
	seg.a = v3(100.5, 0.0, 0.0);
	seg.b = v3(-100.5, 0.0, 0.0);
	h.insert(seg, 99U);

	int hitIx = 0;
	for (int32_t ix = 0; ix < h.voxelDim(0); ++ix) {
		if (!h.ids(VoxelKey{ix, iy, iz}).empty()) ++hitIx;
	}
	EXPECT_EQ(hitIx, 8);
}

} // namespace
} // namespace xtpms
