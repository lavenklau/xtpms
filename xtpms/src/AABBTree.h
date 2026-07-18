#pragma once

// Module AABBTree: AABB nearest-point queries based on CGAL.
// - TriMeshAABBTree: triangle mesh (implemented)
// - SegmentAABBTree: polyline/segment set (to be implemented; same API as triangle mesh:
// non-template class + template build / closest_point)

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

/// Converts between user point types and double coordinates; specialize for non-standard types.
template <typename Point3>
struct AABBPointTraits {
	static double cx(const Point3& p) { return static_cast<double>(p[0]); }
	static double cy(const Point3& p) { return static_cast<double>(p[1]); }
	static double cz(const Point3& p) { return static_cast<double>(p[2]); }
	static Point3 make(double x, double y, double z) { return Point3{x, y, z}; }
};

template <>
struct AABBPointTraits<std::array<double, 3>> {
	static double cx(const std::array<double, 3>& p) { return p[0]; }
	static double cy(const std::array<double, 3>& p) { return p[1]; }
	static double cz(const std::array<double, 3>& p) { return p[2]; }
	static std::array<double, 3> make(double x, double y, double z) { return {x, y, z}; }
};

/// Triangle face: vertex indices corresponding to the vertex array passed to build.
using TriMeshFace = std::array<std::size_t, 3>;

/// Nearest-point query result (shared field names for triangle/segment trees: primitive_index is
/// the face index in triangle meshes).
template <typename QueryPoint>
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

	template <typename Vert, typename VertTraits = AABBPointTraits<std::remove_cv_t<Vert>>>
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
					throw std::out_of_range(
						"TriMeshAABBTree::build: face vertex index out of range");
				}
			}
			const Vert& pa = vertices[f[0]];
			const Vert& pb = vertices[f[1]];
			const Vert& pc = vertices[f[2]];
			const CgalTriangle tri(
				CgalPoint(VertTraits::cx(pa), VertTraits::cy(pa), VertTraits::cz(pa)),
				CgalPoint(VertTraits::cx(pb), VertTraits::cy(pb), VertTraits::cz(pb)),
				CgalPoint(VertTraits::cx(pc), VertTraits::cy(pc), VertTraits::cz(pc)));
			if (tri.is_degenerate())
				continue;
			storage_.emplace_back(tri, fi);
		}
		if (storage_.empty())
			return;

		tree_ = std::make_unique<Tree>(storage_.cbegin(), storage_.cend());
	}

	template <typename Query, typename QueryTraits = AABBPointTraits<std::remove_cv_t<Query>>>
	std::optional<AABBNearestHit<Query>> closest_point(const Query& query) const {
		if (empty())
			return std::nullopt;
		using K = CGAL::Simple_cartesian<double>;
		using CgalPoint = typename K::Point_3;
		const CgalPoint q(QueryTraits::cx(query), QueryTraits::cy(query), QueryTraits::cz(query));
		const auto pr = tree_->closest_point_and_primitive(q);
		const CgalPoint& cgal_closest = pr.first;
		const TriConstIterator prim_it = pr.second;
		AABBNearestHit<Query> hit;
		hit.closest = QueryTraits::make(static_cast<double>(cgal_closest.x()),
										static_cast<double>(cgal_closest.y()),
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
