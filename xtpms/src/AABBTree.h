#pragma once

// 模块 AABBTree：基于 CGAL 的 AABB 最近点查询。
// - TriMeshAABBTree：三角网格（已实现）
// - SegmentAABBTree：折线/线段集（待实现，API 与三角网相同：非模板类 + 模板 build / closest_point）

#include <array>
#include <cstddef>
#include <memory>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits.h>
#include <CGAL/Simple_cartesian.h>
#include <CGAL/squared_distance_3.h>

namespace xtpms {

/// 用户点类型与 double 坐标互转；非常规类型请特化。
template<typename Point3>
struct AABBPointTraits {
	static double cx(const Point3& p) { return static_cast<double>(p[0]); }
	static double cy(const Point3& p) { return static_cast<double>(p[1]); }
	static double cz(const Point3& p) { return static_cast<double>(p[2]); }
	static Point3 make(double x, double y, double z) { return Point3{x, y, z}; }
};

template<>
struct AABBPointTraits<std::array<double, 3>> {
	static double cx(const std::array<double, 3>& p) { return p[0]; }
	static double cy(const std::array<double, 3>& p) { return p[1]; }
	static double cz(const std::array<double, 3>& p) { return p[2]; }
	static std::array<double, 3> make(double x, double y, double z) { return {x, y, z}; }
};

/// 三角形面片：顶点索引，对应 build 时传入的顶点表下标。
using TriMeshFace = std::array<std::size_t, 3>;

/// 最近点查询结果（三角/线段树共用字段名：primitive_index 在三角网中为面片下标）。
template<typename QueryPoint>
struct AABBNearestHit {
	QueryPoint closest{};
	double squared_distance{0.0};
	std::size_t primitive_index{0};
};

class TriMeshAABBTree {
public:
	TriMeshAABBTree() = default;
	TriMeshAABBTree(const TriMeshAABBTree&) = delete;
	TriMeshAABBTree& operator=(const TriMeshAABBTree&) = delete;
	TriMeshAABBTree(TriMeshAABBTree&&) noexcept = default;
	TriMeshAABBTree& operator=(TriMeshAABBTree&&) noexcept = default;

	void clear() {
		tree_.reset();
		storage_.clear();
	}

	bool empty() const { return !tree_ || storage_.empty(); }

	template<typename Vert, typename VertTraits = AABBPointTraits<std::remove_cv_t<Vert>>>
	void build(const std::vector<Vert>& vertices, const std::vector<TriMeshFace>& faces) {
		clear();
		using K = CGAL::Simple_cartesian<double>;
		using CgalPoint = typename K::Point_3;
		using CgalTriangle = typename K::Triangle_3;

		storage_.reserve(faces.size());
		for (std::size_t fi = 0; fi < faces.size(); ++fi) {
			const TriMeshFace& f = faces[fi];
			for (int k = 0; k < 3; ++k) {
				if (f[static_cast<size_t>(k)] >= vertices.size()) {
					throw std::out_of_range("TriMeshAABBTree::build: face vertex index out of range");
				}
			}
			const Vert& pa = vertices[f[0]];
			const Vert& pb = vertices[f[1]];
			const Vert& pc = vertices[f[2]];
			const CgalTriangle tri(CgalPoint(VertTraits::cx(pa), VertTraits::cy(pa), VertTraits::cz(pa)),
				CgalPoint(VertTraits::cx(pb), VertTraits::cy(pb), VertTraits::cz(pb)),
				CgalPoint(VertTraits::cx(pc), VertTraits::cy(pc), VertTraits::cz(pc)));
			if (tri.is_degenerate()) continue;
			storage_.emplace_back(tri, fi);
		}
		if (storage_.empty()) return;

		tree_ = std::make_unique<Tree>(storage_.cbegin(), storage_.cend());
	}

	template<typename Query, typename QueryTraits = AABBPointTraits<std::remove_cv_t<Query>>>
	std::optional<AABBNearestHit<Query>> closest_point(const Query& query) const {
		if (empty()) return std::nullopt;
		using K = CGAL::Simple_cartesian<double>;
		using CgalPoint = typename K::Point_3;
		const CgalPoint q(QueryTraits::cx(query), QueryTraits::cy(query), QueryTraits::cz(query));
		const auto pr = tree_->closest_point_and_primitive(q);
		const CgalPoint& cgal_closest = pr.first;
		const TriConstIterator prim_it = pr.second;
		AABBNearestHit<Query> hit;
		hit.closest = QueryTraits::make(static_cast<double>(cgal_closest.x()), static_cast<double>(cgal_closest.y()),
			static_cast<double>(cgal_closest.z()));
		hit.squared_distance = static_cast<double>(CGAL::squared_distance(q, cgal_closest));
		hit.primitive_index = prim_it->second;
		return hit;
	}

private:
	using K = CGAL::Simple_cartesian<double>;
	using CgalTriangle = typename K::Triangle_3;
	using TriPair = std::pair<CgalTriangle, std::size_t>;
	using TriStorage = std::vector<TriPair>;
	using TriConstIterator = typename TriStorage::const_iterator;

	struct TriPrimitive {
		using Id = TriConstIterator;
		using Point = typename K::Point_3;
		using Datum = typename K::Triangle_3;
		Id m_it{};
		TriPrimitive() = default;
		explicit TriPrimitive(Id it) : m_it(it) {}
		Id id() const { return m_it; }
		Datum datum() const { return m_it->first; }
		Point reference_point() const { return m_it->first.vertex(0); }
	};

	using TriATraits = CGAL::AABB_traits<K, TriPrimitive>;
	using Tree = CGAL::AABB_tree<TriATraits>;

	TriStorage storage_{};
	std::unique_ptr<Tree> tree_{};
};

} // namespace xtpms
