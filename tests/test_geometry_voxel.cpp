/**
 * @file test_geometry_voxel.cpp
 * @brief Unit tests for GeometryVoxel.h
 */

#include "GeometryVoxel.h"

#include <gtest/gtest.h>

#include <set>
#include <tuple>
#include <vector>

namespace xtpms {
namespace {

using Vec3d = VoxelizationGrid::Vec3d;

Vec3d v3(double x, double y, double z) {
	using S = typename Vec3d::value_type;
	return Vec3d(static_cast<S>(x), static_cast<S>(y), static_cast<S>(z));
}

VoxelizationGrid unitBoxGrid2() {
	VoxelizationGrid g;
	g.boxMin = v3(0.0, 0.0, 0.0);
	g.boxMax = v3(1.0, 1.0, 1.0);
	g.invResolution = 2.0;
	g.n[0] = g.n[1] = g.n[2] = 2;
	return g;
}

std::vector<VoxelIndex> collectSeg(const SegmentVoxelGeometry& geom, const VoxelizationGrid& grid) {
	std::vector<VoxelIndex> out;
	geom.getOccupiedVox(grid, [&out](const VoxelIndex& v) { out.push_back(v); });
	return out;
}

bool operator<(const VoxelIndex& a, const VoxelIndex& b) {
	return std::tie(a.ix, a.iy, a.iz) < std::tie(b.ix, b.iy, b.iz);
}

TEST(GeometryVoxel, SegmentCrossesMultipleVoxels) {
	const VoxelizationGrid g = unitBoxGrid2();
	SegmentVoxelGeometry seg;
	seg.a = v3(0.05, 0.25, 0.25);
	seg.b = v3(0.95, 0.25, 0.25);
	const auto v = collectSeg(seg, g);
	std::set<std::tuple<int32_t, int32_t, int32_t>> uniq;
	for (const auto& k : v) uniq.insert({k.ix, k.iy, k.iz});
	EXPECT_GE(uniq.size(), 2u);
	for (const auto& t : uniq) {
		EXPECT_GE(std::get<0>(t), 0);
		EXPECT_LE(std::get<0>(t), 1);
		EXPECT_EQ(std::get<1>(t), 0);
		EXPECT_EQ(std::get<2>(t), 0);
	}
}

TEST(GeometryVoxel, SegmentDegenerateEmitsOneVoxel) {
	const VoxelizationGrid g = unitBoxGrid2();
	SegmentVoxelGeometry seg;
	seg.a = v3(0.25, 0.25, 0.25);
	seg.b = v3(0.25, 0.25, 0.25);
	const auto v = collectSeg(seg, g);
	ASSERT_EQ(v.size(), 1u);
}

TEST(GeometryVoxel, SegmentOutsideBoxEmitsNothing) {
	const VoxelizationGrid g = unitBoxGrid2();
	SegmentVoxelGeometry seg;
	seg.a = v3(-2.0, 0.5, 0.5);
	seg.b = v3(-1.5, 0.5, 0.5);
	const auto v = collectSeg(seg, g);
	EXPECT_TRUE(v.empty());
}

TEST(GeometryVoxel, PointSegmentDistanceMidpoint) {
	const Vec3d p = v3(0.5, 1.0, 0.0);
	const Vec3d a = v3(0.0, 0.0, 0.0);
	const Vec3d b = v3(1.0, 0.0, 0.0);
	const auto r = pointSegmentDistanceSquared(p, a, b);
	EXPECT_NEAR(r.distanceSquared, 1.0, 1e-5);
	EXPECT_NEAR(static_cast<double>(r.closest[0]), 0.5, 1e-5);
	EXPECT_NEAR(static_cast<double>(r.closest[1]), 0.0, 1e-5);
	EXPECT_NEAR(static_cast<double>(r.closest[2]), 0.0, 1e-5);
}

TEST(GeometryVoxel, PointSegmentDistanceDegenerate) {
	const Vec3d p = v3(2.0, 0.0, 0.0);
	const Vec3d a = v3(1.0, 0.0, 0.0);
	const Vec3d b = v3(1.0, 0.0, 0.0);
	const auto r = pointSegmentDistanceSquared(p, a, b);
	EXPECT_NEAR(r.distanceSquared, 1.0, 1e-5);
}

TEST(GeometryVoxel, SegmentVoxelGeometryDistanceToPoint) {
	SegmentVoxelGeometry seg;
	seg.a = v3(0.0, 0.0, 0.0);
	seg.b = v3(3.0, 0.0, 0.0);
	const auto r = seg.distanceSquaredToPoint(v3(2.0, 0.0, 0.0));
	EXPECT_NEAR(r.distanceSquared, 0.0, 1e-5);
}

} // namespace
} // namespace xtpms
