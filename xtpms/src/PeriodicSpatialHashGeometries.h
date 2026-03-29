#pragma once

/**
 * @file PeriodicSpatialHashGeometries.h
 * @brief Adapters for PeriodicSpatialHash::insert (world-lattice segment DDA).
 *
 * SegmentGeometry voxelizes the segment on the unbounded world lattice aligned with @c halfPeriod and
 * @c invResolution (same as @c voxelAtWorld), using 3D DDA in cell space; each visited cell is folded through
 * @c voxelAtWorld at the cell center (authoritative periodic key).
 */

#include "PeriodicSpatialHash.h"

#include <OpenMesh/Core/Utils/vector_traits.hh>

#include <cmath>
#include <cstdint>
#include <limits>
#include <set>
#include <utility>

namespace xtpms {

namespace detail {

template <class HashT>
struct PeriodicVoxelKeyLess {
	bool operator()(const typename HashT::VoxelKey& x, const typename HashT::VoxelKey& y) const {
		if (x.ix != y.ix) return x.ix < y.ix;
		if (x.iy != y.iy) return x.iy < y.iy;
		return x.iz < y.iz;
	}
};

template <class HashT>
typename HashT::Vec3d worldLatticeCellCenterWorld(const HashT& hash, int64_t vx, int64_t vy, int64_t vz) {
	using Vec3 = typename HashT::Vec3d;
	using Scalar = typename OpenMesh::vector_traits<Vec3>::value_type;
	const double inv = hash.invResolution();
	const auto& hp = hash.halfPeriod();
	const double cx = (static_cast<double>(vx) + 0.5) / inv - static_cast<double>(hp[0]);
	const double cy = (static_cast<double>(vy) + 0.5) / inv - static_cast<double>(hp[1]);
	const double cz = (static_cast<double>(vz) + 0.5) / inv - static_cast<double>(hp[2]);
	return Vec3{Scalar(cx), Scalar(cy), Scalar(cz)};
}

/**
 * 3D DDA over world lattice: cell coordinate is @f$ \lfloor (p_d + hp_d)\, invRes \rfloor @f$ (same anchor as @c voxelAtWorld).
 * @param onCell Invoked with each visited world cell (vx, vy, vz).
 */
template <class HashT, class CellFn>
void traverseSegmentWorldLattice(const HashT& hash, const typename HashT::Vec3d& wa, const typename HashT::Vec3d& wb, CellFn&& onCell) {
	using Vec3 = typename HashT::Vec3d;
	const double invRes = hash.invResolution();
	const auto& hp = hash.halfPeriod();

	auto toCell = [invRes, &hp](const Vec3& p, int axis) {
		return (static_cast<double>(p[axis]) + static_cast<double>(hp[axis])) * invRes;
	};

	double ca[3] {toCell(wa, 0), toCell(wa, 1), toCell(wa, 2)};
	double cb[3] {toCell(wb, 0), toCell(wb, 1), toCell(wb, 2)};

	double dir[3] {cb[0] - ca[0], cb[1] - ca[1], cb[2] - ca[2]};
	const double glen2 = dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2];
	if (!(glen2 > 1e-30)) {
		const int64_t vx = static_cast<int64_t>(std::floor(ca[0]));
		const int64_t vy = static_cast<int64_t>(std::floor(ca[1]));
		const int64_t vz = static_cast<int64_t>(std::floor(ca[2]));
		onCell(vx, vy, vz);
		return;
	}

	const double glen = std::sqrt(glen2);
	double du[3] {dir[0] / glen, dir[1] / glen, dir[2] / glen};

	int64_t vx = static_cast<int64_t>(std::floor(ca[0]));
	int64_t vy = static_cast<int64_t>(std::floor(ca[1]));
	int64_t vz = static_cast<int64_t>(std::floor(ca[2]));

	const int64_t ex = static_cast<int64_t>(std::floor(cb[0]));
	const int64_t ey = static_cast<int64_t>(std::floor(cb[1]));
	const int64_t ez = static_cast<int64_t>(std::floor(cb[2]));

	double tMax[3];
	double tDelta[3];
	const double inf = std::numeric_limits<double>::infinity();

	auto axisV = [](int axis, int64_t ix, int64_t iy, int64_t iz) -> int64_t {
		return axis == 0 ? ix : (axis == 1 ? iy : iz);
	};
	for (int axis = 0; axis < 3; ++axis) {
		const double cv = static_cast<double>(axisV(axis, vx, vy, vz));
		if (std::abs(du[axis]) < 1e-18) {
			tMax[axis] = inf;
			tDelta[axis] = inf;
		} else if (du[axis] > 0.0) {
			tMax[axis] = ((cv + 1.0) - ca[axis]) / du[axis];
			tDelta[axis] = 1.0 / du[axis];
		} else {
			tMax[axis] = (cv - ca[axis]) / du[axis];
			tDelta[axis] = -1.0 / du[axis];
		}
	}

	const int64_t maxSteps =
		(vx > ex ? vx - ex : ex - vx) + (vy > ey ? vy - ey : ey - vy) + (vz > ez ? vz - ez : ez - vz) + 4;
	int64_t steps = 0;

	for (;;) {
		onCell(vx, vy, vz);
		if (vx == ex && vy == ey && vz == ez) break;
		if (++steps > maxSteps) break;

		if (tMax[0] <= tMax[1]) {
			if (tMax[0] <= tMax[2]) {
				vx += du[0] > 0 ? int64_t{1} : int64_t{-1};
				tMax[0] += tDelta[0];
			} else {
				vz += du[2] > 0 ? int64_t{1} : int64_t{-1};
				tMax[2] += tDelta[2];
			}
		} else {
			if (tMax[1] <= tMax[2]) {
				vy += du[1] > 0 ? int64_t{1} : int64_t{-1};
				tMax[1] += tDelta[1];
			} else {
				vz += du[2] > 0 ? int64_t{1} : int64_t{-1};
				tMax[2] += tDelta[2];
			}
		}
	}
}

} // namespace detail

/**
 * @brief Euclidean segment @c a–@c b in world space: 3D DDA on the world lattice, then @c voxelAtWorld(cell center).
 *
 * World cell indices use @f$ \lfloor (p_d + hp_d)\, invRes \rfloor @f$ per axis (aligned with the hash). Multiple
 * world cells can map to the same periodic @c VoxelKey; duplicates are merged before @p emit.
 *
 * @note This is not the shortest torus geodesic between two canonicalized points; for that, callers should
 *       pass endpoints already chosen so the desired minimal image is the Euclidean chord, or add a
 *       dedicated primitive.
 */
struct SegmentGeometry {
	using Vec3d = typename PeriodicSpatialHash<>::Vec3d;
	Vec3d a{};
	Vec3d b{};

	template <class HashT, class EmitFn>
	void getOccupiedVox(const HashT& hash, EmitFn&& emit) const {
		using Key = typename HashT::VoxelKey;
		std::set<Key, detail::PeriodicVoxelKeyLess<HashT>> seen;
		detail::traverseSegmentWorldLattice(hash, a, b, [&](int64_t vx, int64_t vy, int64_t vz) {
			const typename HashT::Vec3d c = detail::worldLatticeCellCenterWorld(hash, vx, vy, vz);
			const Key k = hash.voxelAtWorld(c);
			if (seen.insert(k).second) emit(k);
		});
	}
};

} // namespace xtpms
