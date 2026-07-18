#include <gtest/gtest.h>

#include <array>
#include <stdexcept>
#include <vector>

#include "AABBTree.h"

namespace {

using P = std::array<double, 3>;

} // namespace

TEST(AABBTree_TriMesh, EmptyNoHit) {
	xtpms::TriMeshAABBTree tree;
	EXPECT_TRUE(tree.empty());
	const P q{{0.0, 0.0, 0.0}};
	EXPECT_FALSE(tree.closest_point(q).has_value());
}

TEST(AABBTree_TriMesh, SingleTriangleClosestPoint) {
	std::vector<P> v{{{0.0, 0.0, 0.0}}, {{1.0, 0.0, 0.0}}, {{0.0, 1.0, 0.0}}};
	std::vector<xtpms::TriMeshFace> f{{{0, 1, 2}}};

	xtpms::TriMeshAABBTree tree;
	tree.build(v, f);
	ASSERT_FALSE(tree.empty());

	const P q{{0.25, 0.25, 2.0}};
	const auto hit = tree.closest_point(q);
	ASSERT_TRUE(hit.has_value());
	EXPECT_EQ(hit->primitive_index, 0u);
	EXPECT_NEAR(hit->squared_distance, 4.0, 1e-9);
	EXPECT_NEAR(hit->closest[0], 0.25, 1e-9);
	EXPECT_NEAR(hit->closest[1], 0.25, 1e-9);
	EXPECT_NEAR(hit->closest[2], 0.0, 1e-9);
}

TEST(AABBTree_TriMesh, QueryAtVertex_ZeroDistance) {
	std::vector<P> v{{{0.0, 0.0, 0.0}}, {{1.0, 0.0, 0.0}}, {{0.0, 1.0, 0.0}}};
	std::vector<xtpms::TriMeshFace> f{{{0, 1, 2}}};
	xtpms::TriMeshAABBTree tree;
	tree.build(v, f);
	const auto hit = tree.closest_point(P{{0.0, 0.0, 0.0}});
	ASSERT_TRUE(hit.has_value());
	EXPECT_NEAR(hit->squared_distance, 0.0, 1e-18);
	EXPECT_EQ(hit->primitive_index, 0u);
}

TEST(AABBTree_TriMesh, TwoTriangles_PicksNearestPrimitive) {
	// 三角形 0：z=0；三角形 1：z=10
	std::vector<P> v{{{0.0, 0.0, 0.0}},
					 {{1.0, 0.0, 0.0}},
					 {{0.0, 1.0, 0.0}},
					 {{0.0, 0.0, 10.0}},
					 {{1.0, 0.0, 10.0}},
					 {{0.0, 1.0, 10.0}}};
	std::vector<xtpms::TriMeshFace> f{{{0, 1, 2}}, {{3, 4, 5}}};
	xtpms::TriMeshAABBTree tree;
	tree.build(v, f);
	ASSERT_FALSE(tree.empty());

	const auto hit = tree.closest_point(P{{0.25, 0.25, 10.0}});
	ASSERT_TRUE(hit.has_value());
	EXPECT_EQ(hit->primitive_index, 1u);
	EXPECT_NEAR(hit->squared_distance, 0.0, 1e-18);
	EXPECT_NEAR(hit->closest[2], 10.0, 1e-9);
}

TEST(AABBTree_TriMesh, ClearThenEmpty) {
	std::vector<P> v{{{0.0, 0.0, 0.0}}, {{1.0, 0.0, 0.0}}, {{0.0, 1.0, 0.0}}};
	std::vector<xtpms::TriMeshFace> f{{{0, 1, 2}}};
	xtpms::TriMeshAABBTree tree;
	tree.build(v, f);
	ASSERT_FALSE(tree.empty());
	tree.clear();
	EXPECT_TRUE(tree.empty());
	EXPECT_FALSE(tree.closest_point(P{{0.0, 0.0, 0.0}}).has_value());
}

TEST(AABBTree_TriMesh, AllDegenerateFaces_EmptyTree) {
	// 共线三点，三角形退化，应被跳过
	std::vector<P> v{{{0.0, 0.0, 0.0}}, {{1.0, 0.0, 0.0}}, {{2.0, 0.0, 0.0}}};
	std::vector<xtpms::TriMeshFace> f{{{0, 1, 2}}};
	xtpms::TriMeshAABBTree tree;
	tree.build(v, f);
	EXPECT_TRUE(tree.empty());
	EXPECT_FALSE(tree.closest_point(P{{0.5, 0.0, 0.0}}).has_value());
}

TEST(AABBTree_TriMesh, FaceIndexOutOfRange_Throws) {
	std::vector<P> v{{{0.0, 0.0, 0.0}}, {{1.0, 0.0, 0.0}}, {{0.0, 1.0, 0.0}}};
	std::vector<xtpms::TriMeshFace> f{{{0, 1, 10}}};
	xtpms::TriMeshAABBTree tree;
	EXPECT_THROW(tree.build(v, f), std::out_of_range);
	EXPECT_TRUE(tree.empty());
}

TEST(AABBTree_TriMesh, Rebuild_ReplacesPreviousTree) {
	std::vector<P> v1{{{0.0, 0.0, 0.0}}, {{1.0, 0.0, 0.0}}, {{0.0, 1.0, 0.0}}};
	std::vector<xtpms::TriMeshFace> f1{{{0, 1, 2}}};
	xtpms::TriMeshAABBTree tree;
	tree.build(v1, f1);
	ASSERT_TRUE(tree.closest_point(P{{0.0, 0.0, 5.0}}).has_value());

	std::vector<P> v2{{{10.0, 0.0, 0.0}}, {{11.0, 0.0, 0.0}}, {{10.0, 1.0, 0.0}}};
	std::vector<xtpms::TriMeshFace> f2{{{0, 1, 2}}};
	tree.build(v2, f2);
	const auto hit = tree.closest_point(P{{10.25, 0.25, 2.0}});
	ASSERT_TRUE(hit.has_value());
	EXPECT_NEAR(hit->closest[0], 10.25, 1e-9);
}

struct Vec3 {
	double x{};
	double y{};
	double z{};
};

template <>
struct xtpms::AABBPointTraits<Vec3> {
	static double cx(const Vec3& p) { return p.x; }
	static double cy(const Vec3& p) { return p.y; }
	static double cz(const Vec3& p) { return p.z; }
	static Vec3 make(double x, double y, double z) { return Vec3{x, y, z}; }
};

TEST(AABBTree_TriMesh, QueryWithDifferentPointType_ViaTraits) {
	std::vector<P> v{{{0.0, 0.0, 0.0}}, {{1.0, 0.0, 0.0}}, {{0.0, 1.0, 0.0}}};
	std::vector<xtpms::TriMeshFace> f{{{0, 1, 2}}};
	xtpms::TriMeshAABBTree tree;
	tree.build(v, f);

	const Vec3 q{0.25, 0.25, 2.0};
	const auto hit = tree.closest_point(q);
	ASSERT_TRUE(hit.has_value());
	EXPECT_NEAR(hit->squared_distance, 4.0, 1e-9);
	EXPECT_NEAR(hit->closest.x, 0.25, 1e-9);
	EXPECT_NEAR(hit->closest.y, 0.25, 1e-9);
	EXPECT_NEAR(hit->closest.z, 0.0, 1e-9);
}
