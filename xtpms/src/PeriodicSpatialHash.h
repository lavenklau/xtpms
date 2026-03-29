#pragma once

/**
 * @file PeriodicSpatialHash.h
 * @brief Periodic voxel spatial hash for microstructure and TPMS-style workflows.
 */

#include "MeshTypes.h"

#include <OpenMesh/Core/Utils/vector_traits.hh>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace xtpms {

/**
 * @brief Spatial hash on a 3D voxel grid with periodic boundaries.
 *
 * Used to associate identifiers with voxels intersected by geometric primitives
 * (e.g. narrow features on periodic TPMS or microstructure meshes).
 *
 * Construction: pass @p halfPeriod (half the period length per axis) and
 * @p resolution (voxel edge length). The fundamental domain per axis is
 * approximately [-halfPeriod, +halfPeriod).
 *
 * @tparam IdT Type stored per voxel (default `std::uint32_t`).
 *
 * **Geometry concept** - types passed to insert() must provide:
 * @code
 * template <class HashT, class EmitFn>
 * void getOccupiedVox(const HashT& hash, EmitFn&& emit) const;
 * @endcode
 * where `emit` is callable as `emit(const VoxelKey&)`, once per occupied voxel.
 *
 * @see GeometryVoxel.h for periodicity-free voxelization on a regular grid.
 * @see PeriodicSpatialHashGeometries.h for periodic adapters (e.g. SegmentGeometry) used with insert().
 */
template <class IdT = std::uint32_t>
class PeriodicSpatialHash {
public:
	using Vec3d = typename DefaultTriMesh::Point;

	/** @brief Integer voxel coordinates in the periodic grid. */
	struct VoxelKey {
		int32_t ix = 0;
		int32_t iy = 0;
		int32_t iz = 0;

		bool operator==(const VoxelKey& rhs) const {
			return ix == rhs.ix && iy == rhs.iy && iz == rhs.iz;
		}
	};

	/** @brief Hash for VoxelKey in unordered_map. */
	struct VoxelKeyHasher {
		std::size_t operator()(const VoxelKey& k) const noexcept {
			std::size_t h = 1469598103934665603ull;
			auto mix = [&h](std::uint64_t v) {
				h ^= static_cast<std::size_t>(v) + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
			};
			mix(static_cast<std::uint64_t>(static_cast<int64_t>(k.ix)));
			mix(static_cast<std::uint64_t>(static_cast<int64_t>(k.iy)));
			mix(static_cast<std::uint64_t>(static_cast<int64_t>(k.iz)));
			return h;
		}
	};

	/**
	 * @brief Build a hash grid covering one period per axis.
	 * @param halfPeriod Half-period (positive extent) per axis; full period is `2 * halfPeriod[i]`.
	 * @param resolution Voxel edge length (must be positive).
	 */
	PeriodicSpatialHash(const Vec3d& halfPeriod, double resolution)
		: halfPeriod_(halfPeriod), resolution_(resolution) {
		invResolution_ = 1.0 / resolution_;

		for (int i = 0; i < 3; ++i) {
			const double period = 2.0 * halfPeriod_[i];
			n_[i] = static_cast<int32_t>(std::max(1.0, std::ceil(period / resolution_)));
		}
	}

	/** @brief Remove all voxel-to-id lists. */
	void clear() { map_.clear(); }

	/**
	 * @brief Ids associated with a voxel, or an empty vector if none.
	 * @param key Voxel coordinates.
	 */
	const std::vector<IdT>& ids(const VoxelKey& key) const {
		static const std::vector<IdT> empty;
		auto it = map_.find(key);
		if (it == map_.end()) return empty;
		return it->second;
	}

	/**
	 * @brief Wrap a vector/point per axis into the fundamental domain.
	 * @param v Input coordinates (any component may lie outside the period).
	 * @return Equivalent point with each component in [-halfPeriod, +halfPeriod) when the period is positive.
	 */
	Vec3d wrapVector(const Vec3d& v) const {
		Vec3d out = v;
		for (int i = 0; i < 3; ++i) {
			const double hp = halfPeriod_[i];
			const double step = 2.0 * hp;
			if (!(step > 0.0)) continue;

			const double shifted = static_cast<double>(v[i]) + hp;
			const double n = std::floor(shifted / step);
			out[i] = v[i] - n * step;
		}
		return out;
	}

	/**
	 * @brief Map a world-space point to a clamped voxel key after periodic wrap.
	 * @param p World coordinates.
	 */
	VoxelKey voxelAtWorld(const Vec3d& p) const {
		const Vec3d w = wrapVector(p);
		return {coordToIndex(w[0], 0), coordToIndex(w[1], 1), coordToIndex(w[2], 2)};
	}

	/**
	 * @brief Fundamental-domain voxel center in world coordinates (inverse of @c voxelAtWorld at cell center).
	 *
	 * Per axis @c d: @f$ x_d = (i_d + \tfrac12) / invResolution - halfPeriod_d @f$, matching @c coordToIndex.
	 */
	Vec3d idx2World(const VoxelKey& k) const {
		using Scalar = typename OpenMesh::vector_traits<Vec3d>::value_type;
		const double cx = (static_cast<double>(k.ix) + 0.5) / invResolution_ - static_cast<double>(halfPeriod_[0]);
		const double cy = (static_cast<double>(k.iy) + 0.5) / invResolution_ - static_cast<double>(halfPeriod_[1]);
		const double cz = (static_cast<double>(k.iz) + 0.5) / invResolution_ - static_cast<double>(halfPeriod_[2]);
		return Vec3d{Scalar(cx), Scalar(cy), Scalar(cz)};
	}

	/**
	 * @brief Insert an id into every voxel reported by the geometry.
	 * @tparam Geometry Type implementing `getOccupiedVox`.
	 * @param geometry Primitive or adapter implementing getOccupiedVox (see PeriodicSpatialHashGeometries.h).
	 * @param id Value to append to each occupied voxel's list.
	 */
	template <class Geometry>
	void insert(const Geometry& geometry, IdT id) {
		geometry.getOccupiedVox(*this, [this, id](const VoxelKey& k) { map_[k].push_back(id); });
	}

	/** @brief One stored id reported for an occupied voxel in a neighborhood query. */
	struct VoxelOccupancy {
		VoxelKey key{};
		IdT id{};
	};

	/**
	 * @brief List occupied voxels (and their stored ids) in a toroidal Chebyshev neighborhood of @p pointWorld.
	 *
	 * The anchor voxel is @c voxelAtWorld(pointWorld). Every voxel within @p chebyshevNeighborhoodRadius
	 * steps along each axis (L-infinity ball), with indices wrapped periodically, is visited. For each
	 * voxel that has at least one stored id, one @c VoxelOccupancy entry is appended per id.
	 *
	 * @return Empty if no geometry occupies that neighborhood; otherwise deterministic order (dz, dy, dx, then id list order).
	 */
	std::vector<VoxelOccupancy> occupiedVoxelsNearPoint(
		const Vec3d& pointWorld,
		int32_t chebyshevNeighborhoodRadius = 1) const {
		if (chebyshevNeighborhoodRadius < 0) chebyshevNeighborhoodRadius = 0;

		std::vector<VoxelOccupancy> out;
		const VoxelKey k0 = voxelAtWorld(pointWorld);
		for (int32_t dz = -chebyshevNeighborhoodRadius; dz <= chebyshevNeighborhoodRadius; ++dz) {
			for (int32_t dy = -chebyshevNeighborhoodRadius; dy <= chebyshevNeighborhoodRadius; ++dy) {
				for (int32_t dx = -chebyshevNeighborhoodRadius; dx <= chebyshevNeighborhoodRadius; ++dx) {
					const VoxelKey k{
						wrapVoxelIndex(k0.ix + dx, 0),
						wrapVoxelIndex(k0.iy + dy, 1),
						wrapVoxelIndex(k0.iz + dz, 2)};
					for (IdT id : ids(k)) out.push_back(VoxelOccupancy{k, id});
				}
			}
		}
		return out;
	}

	/** @brief Half-period vector used at construction (same convention as wrapVector). */
	const Vec3d& halfPeriod() const { return halfPeriod_; }
	/** @brief Reciprocal voxel edge length. */
	double invResolution() const { return invResolution_; }
	/** @brief Number of voxels along @p axis (0=x, 1=y, 2=z). */
	int32_t voxelDim(int axis) const { return n_[axis]; }

private:
	// Toroidal voxel index (for neighborhood queries across periodic boundaries).
	int32_t wrapVoxelIndex(int32_t idx, int axis) const {
		const int32_t n = n_[axis];
		if (n <= 0) return 0;
		int32_t m = idx % n;
		if (m < 0) m += n;
		return m;
	}

	// Maps coordinate in [-hp, hp] to voxel index; clamps to [0, n_[axis]-1].
	int32_t coordToIndex(double x, int axis) const {
		const double hp = halfPeriod_[axis];
		const double t = (x + hp) * invResolution_;
		int32_t idx = static_cast<int32_t>(std::floor(t));
		if (idx < 0) idx = 0;
		if (idx >= n_[axis]) idx = n_[axis] - 1;
		return idx;
	}

private:
	Vec3d halfPeriod_;
	double resolution_ = 1.0;
	double invResolution_ = 1.0;
	int32_t n_[3] {1, 1, 1};

	std::unordered_map<VoxelKey, std::vector<IdT>, VoxelKeyHasher> map_;
};

} // namespace xtpms
