#pragma once

/**
 * @file GeometryVoxel.h
 * @brief Voxelize geometric primitives against a fixed axis-aligned voxel grid (no periodicity).
 *
 * Period wrapping, tiling, and PeriodicSpatialHash integration live in
 * PeriodicSpatialHashGeometries.h, not here.
 */

#include "MeshTypes.h"

#include <OpenMesh/Core/Utils/vector_traits.hh>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>

namespace xtpms {

/**
 * @brief Squared distance from a point to a closed segment @c ab, and the closest point on the segment.
 * @tparam Vec3 Type with @c [0],@c [1],@c [2] readable as scalars (e.g. DefaultTriMesh::Point).
 */
template <class Vec3>
struct PointSegmentDistanceResult {
	double distanceSquared = 0.0;
	Vec3 closest{};
};

/**
 * @brief Point-segment distance in Euclidean metric (segment endpoints included).
 * @param p Query point.
 * @param a Segment start.
 * @param b Segment end.
 */
template <class Vec3>
PointSegmentDistanceResult<Vec3> pointSegmentDistanceSquared(const Vec3& p, const Vec3& a, const Vec3& b) {
	PointSegmentDistanceResult<Vec3> out;
	const double abx = static_cast<double>(b[0]) - static_cast<double>(a[0]);
	const double aby = static_cast<double>(b[1]) - static_cast<double>(a[1]);
	const double abz = static_cast<double>(b[2]) - static_cast<double>(a[2]);
	const double apx = static_cast<double>(p[0]) - static_cast<double>(a[0]);
	const double apy = static_cast<double>(p[1]) - static_cast<double>(a[1]);
	const double apz = static_cast<double>(p[2]) - static_cast<double>(a[2]);
	const double ab2 = abx * abx + aby * aby + abz * abz;
	if (!(ab2 > 1e-30)) {
		out.closest = a;
		const double dx = apx;
		const double dy = apy;
		const double dz = apz;
		out.distanceSquared = dx * dx + dy * dy + dz * dz;
		return out;
	}
	double t = (apx * abx + apy * aby + apz * abz) / ab2;
	t = std::max(0.0, std::min(1.0, t));
	const double cx = static_cast<double>(a[0]) + t * abx;
	const double cy = static_cast<double>(a[1]) + t * aby;
	const double cz = static_cast<double>(a[2]) + t * abz;
	using Scalar = typename OpenMesh::vector_traits<Vec3>::value_type;
	out.closest = Vec3(Scalar(cx), Scalar(cy), Scalar(cz));
	const double dx = static_cast<double>(p[0]) - cx;
	const double dy = static_cast<double>(p[1]) - cy;
	const double dz = static_cast<double>(p[2]) - cz;
	out.distanceSquared = dx * dx + dy * dy + dz * dz;
	return out;
}

/**
 * @brief Regular grid: voxels cover [boxMin, boxMax] per axis with uniform spacing 1/invResolution.
 *
 * Voxel @c (ix,iy,iz) spans world coordinates
 * [boxMin[d] + ix/res, boxMin[d] + (ix+1)/res) with res = invResolution^{-1}.
 */
struct VoxelizationGrid {
	using Vec3d = typename DefaultTriMesh::Point;
	Vec3d boxMin{};
	Vec3d boxMax{};
	double invResolution = 1.0;
	int32_t n[3] {1, 1, 1};
};

/** @brief Integer cell indices inside the grid. */
struct VoxelIndex {
	int32_t ix = 0;
	int32_t iy = 0;
	int32_t iz = 0;
};

namespace detail {

inline int32_t clampDim(int32_t v, int32_t n) {
	if (v < 0) return 0;
	if (v >= n) return n - 1;
	return v;
}

inline int32_t coordToVoxelIndex(double x, int axis, const VoxelizationGrid& g) {
	const double t = (x - static_cast<double>(g.boxMin[axis])) * g.invResolution;
	int32_t idx = static_cast<int32_t>(std::floor(t));
	return clampDim(idx, g.n[axis]);
}

/** Liang-Barsky clip of segment (wa, wb) to closed box [boxMin, boxMax]. */
template <class Vec3>
inline bool clipSegmentToBox(Vec3& wa, Vec3& wb, const Vec3& boxMin, const Vec3& boxMax) {
	Vec3 d;
	for (int i = 0; i < 3; ++i) d[i] = wb[i] - wa[i];

	double t0 = 0.0;
	double t1 = 1.0;
	for (int i = 0; i < 3; ++i) {
		const double lo = static_cast<double>(boxMin[i]);
		const double hi = static_cast<double>(boxMax[i]);
		const double di = static_cast<double>(d[i]);
		const double p = static_cast<double>(wa[i]);
		if (std::abs(di) < 1e-30) {
			if (p < lo || p > hi) return false;
			continue;
		}
		double u0 = (lo - p) / di;
		double u1 = (hi - p) / di;
		if (u0 > u1) std::swap(u0, u1);
		t0 = std::max(t0, u0);
		t1 = std::min(t1, u1);
		if (t0 > t1) return false;
	}
	if (t0 > t1) return false;

	const Vec3 a0 = wa;
	for (int i = 0; i < 3; ++i) {
		const double di = static_cast<double>(d[i]);
		wa[i] = a0[i] + di * t0;
		wb[i] = a0[i] + di * t1;
	}
	return true;
}

template <class EmitFn>
inline void traverseSegmentInGrid(
	const VoxelizationGrid& g, const typename VoxelizationGrid::Vec3d& wa, const typename VoxelizationGrid::Vec3d& wb, EmitFn&& emit) {
	using Vec3d = typename VoxelizationGrid::Vec3d;

	Vec3d d = wb - wa;
	const double len2 = static_cast<double>(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
	if (!(len2 > 1e-30)) {
		emit(VoxelIndex{
			coordToVoxelIndex(static_cast<double>(wa[0]), 0, g),
			coordToVoxelIndex(static_cast<double>(wa[1]), 1, g),
			coordToVoxelIndex(static_cast<double>(wa[2]), 2, g)});
		return;
	}

	const double invRes = g.invResolution;
	auto toCell = [&g, invRes](const Vec3d& p, int axis) {
		return (static_cast<double>(p[axis]) - static_cast<double>(g.boxMin[axis])) * invRes;
	};

	double ca[3] {toCell(wa, 0), toCell(wa, 1), toCell(wa, 2)};
	double cb[3] {toCell(wb, 0), toCell(wb, 1), toCell(wb, 2)};

	double dir[3] {cb[0] - ca[0], cb[1] - ca[1], cb[2] - ca[2]};
	const double glen = std::sqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
	double du[3] {dir[0] / glen, dir[1] / glen, dir[2] / glen};

	int32_t vx = static_cast<int32_t>(std::floor(ca[0]));
	int32_t vy = static_cast<int32_t>(std::floor(ca[1]));
	int32_t vz = static_cast<int32_t>(std::floor(ca[2]));

	const int32_t ex = static_cast<int32_t>(std::floor(cb[0]));
	const int32_t ey = static_cast<int32_t>(std::floor(cb[1]));
	const int32_t ez = static_cast<int32_t>(std::floor(cb[2]));

	auto emitClamped = [&](int32_t ix, int32_t iy, int32_t iz) {
		emit(VoxelIndex{
			clampDim(ix, g.n[0]),
			clampDim(iy, g.n[1]),
			clampDim(iz, g.n[2])});
	};

	double tMax[3];
	double tDelta[3];
	const double inf = std::numeric_limits<double>::infinity();

	auto axisV = [](int axis, int32_t ix, int32_t iy, int32_t iz) -> int32_t {
		return axis == 0 ? ix : (axis == 1 ? iy : iz);
	};
	for (int axis = 0; axis < 3; ++axis) {
		const int32_t cv = axisV(axis, vx, vy, vz);
		if (std::abs(du[axis]) < 1e-18) {
			tMax[axis] = inf;
			tDelta[axis] = inf;
		} else if (du[axis] > 0.0) {
			tMax[axis] = ((cv + 1) - ca[axis]) / du[axis];
			tDelta[axis] = 1.0 / du[axis];
		} else {
			tMax[axis] = (cv - ca[axis]) / du[axis];
			tDelta[axis] = -1.0 / du[axis];
		}
	}

	const int32_t maxSteps = (g.n[0] + g.n[1] + g.n[2] + 8) * 4;
	int32_t steps = 0;

	for (;;) {
		emitClamped(vx, vy, vz);
		if (vx == ex && vy == ey && vz == ez) break;
		if (++steps > maxSteps) break;

		if (tMax[0] <= tMax[1]) {
			if (tMax[0] <= tMax[2]) {
				vx += du[0] > 0 ? 1 : -1;
				tMax[0] += tDelta[0];
			} else {
				vz += du[2] > 0 ? 1 : -1;
				tMax[2] += tDelta[2];
			}
		} else {
			if (tMax[1] <= tMax[2]) {
				vy += du[1] > 0 ? 1 : -1;
				tMax[1] += tDelta[1];
			} else {
				vz += du[2] > 0 ? 1 : -1;
				tMax[2] += tDelta[2];
			}
		}
	}
}

} // namespace detail

/**
 * @brief Line segment: voxels intersected after clipping to the grid box, then 3D DDA in cell space.
 */
struct SegmentVoxelGeometry {
	using Vec3d = typename VoxelizationGrid::Vec3d;
	Vec3d a{};
	Vec3d b{};

	template <class EmitFn>
	void getOccupiedVox(const VoxelizationGrid& grid, EmitFn&& emit) const {
		Vec3d wa = a;
		Vec3d wb = b;
		if (!detail::clipSegmentToBox(wa, wb, grid.boxMin, grid.boxMax)) return;
		detail::traverseSegmentInGrid(grid, wa, wb, std::forward<EmitFn>(emit));
	}

	/** @brief Squared distance from @p p to this segment (infinite line clamped to [a,b]). */
	PointSegmentDistanceResult<Vec3d> distanceSquaredToPoint(const Vec3d& p) const {
		return pointSegmentDistanceSquared(p, a, b);
	}
};

} // namespace xtpms
