#include "VertexGeometry.h"

#include <cmath>

namespace xtpms {

using Vec3d = PeriodicTriMesh::Vec3d;
using VH = PeriodicTriMesh::VertexHandle;
using EH = PeriodicTriMesh::EdgeHandle;
using HH = PeriodicTriMesh::HalfedgeHandle;
using FH = PeriodicTriMesh::FaceHandle;

// ── Periodic coordinate utilities ──────────────────────────

Vec3d makePeriod(const Vec3d& v, const Vec3d& hp) {
	Vec3d out = v;
	for (int i = 0; i < 3; ++i) {
		const double period = 2.0 * static_cast<double>(hp[i]);
		if (period <= 0.0) continue;
		double vi = static_cast<double>(out[i]);
		if (vi < -static_cast<double>(hp[i])) vi += period;
		else if (vi > static_cast<double>(hp[i])) vi -= period;
		out[i] = static_cast<DefaultTriMesh::Scalar>(vi);
	}
	return out;
}

Eigen::Vector3d toEig(const Vec3d& v) {
	return Eigen::Vector3d(static_cast<double>(v[0]), static_cast<double>(v[1]), static_cast<double>(v[2]));
}

Vec3d toOM(const Eigen::Vector3d& v) {
	return Vec3d(static_cast<DefaultTriMesh::Scalar>(v[0]),
				 static_cast<DefaultTriMesh::Scalar>(v[1]),
				 static_cast<DefaultTriMesh::Scalar>(v[2]));
}

Eigen::Vector3d periodicEdgeVec(const PeriodicTriMesh& mesh, VH v0, VH v1) {
	return toEig(makePeriod(mesh.point(v1) - mesh.point(v0), mesh.halfPeriod()));
}

Eigen::Matrix3d getFacePeriodTri(const PeriodicTriMesh& mesh, FH fh) {
	const Vec3d hp = mesh.halfPeriod();
	auto fv = mesh.cfv_iter(fh);
	VH v0 = *fv; ++fv; VH v1 = *fv; ++fv; VH v2 = *fv;
	Eigen::Vector3d p0 = toEig(mesh.point(v0));
	Eigen::Vector3d p1 = p0 + toEig(makePeriod(mesh.point(v1) - mesh.point(v0), hp));
	Eigen::Vector3d p2 = p0 + toEig(makePeriod(mesh.point(v2) - mesh.point(v0), hp));
	Eigen::Matrix3d tri;
	tri.col(0) = p0; tri.col(1) = p1; tri.col(2) = p2;
	return tri;
}

void getFaceVertexIdx(const PeriodicTriMesh& mesh, FH fh, int idx[3]) {
	auto fv = mesh.cfv_iter(fh);
	idx[0] = (*fv).idx(); ++fv;
	idx[1] = (*fv).idx(); ++fv;
	idx[2] = (*fv).idx();
}

void getPeriodicRing(const PeriodicTriMesh& mesh, VH vh,
					 Eigen::Vector3d& outCenter, std::vector<Eigen::Vector3d>& outRing) {
	const Vec3d hp = mesh.halfPeriod();
	outCenter = toEig(mesh.point(vh));
	outRing.clear();

	if (mesh.is_boundary(vh)) {
		for (auto voh_it = mesh.cvoh_iter(vh); voh_it.is_valid(); ++voh_it) {
			VH vn = mesh.to_vertex_handle(*voh_it);
			outRing.push_back(outCenter + toEig(makePeriod(mesh.point(vn) - mesh.point(vh), hp)));
		}
		return;
	}

	// Face traversal ensures CCW order
	HH he_start = mesh.halfedge_handle(vh);
	HH he = he_start;
	do {
		VH vn = mesh.to_vertex_handle(he);
		outRing.push_back(outCenter + toEig(makePeriod(mesh.point(vn) - mesh.point(vh), hp)));
		he = mesh.opposite_halfedge_handle(mesh.prev_halfedge_handle(he));
	} while (he != he_start && outRing.size() < 30);
}

// ── Compile1ring ───────────────────────────────────────────

Compile1ring::Compile1ring(const Eigen::Vector3d& o, const std::vector<Eigen::Vector3d>& ring) {
	if (ring.size() < 3) return;
	const int N = static_cast<int>(ring.size());

	std::vector<double> esq_inc(static_cast<std::size_t>(N));
	std::vector<double> esq_opp(static_cast<std::size_t>(N));
	for (int i = 0; i < N; ++i) {
		esq_inc[static_cast<std::size_t>(i)] = (ring[static_cast<std::size_t>(i)] - o).squaredNorm();
		esq_opp[static_cast<std::size_t>(i)] = (ring[static_cast<std::size_t>((i + 1) % N)] - ring[static_cast<std::size_t>(i)]).squaredNorm();
	}

	// Precompute cotangents of all angles (aligned with minsurf Compile1ring)
	// Triangle i = (o, ring[i], ring[i+1])
	// cotA_i = cot(angle at ring[i])   -> stored in cot_alpha[(i+1)%N]
	// cotB_i = cot(angle at ring[i+1]) -> stored in cot_beta[i]
	std::vector<double> cot_alpha(static_cast<std::size_t>(N), 0.0);
	std::vector<double> cot_beta(static_cast<std::size_t>(N), 0.0);
	std::vector<double> theta_v(static_cast<std::size_t>(N), 0.0);
	for (int i = 0; i < N; ++i) {
		double e0 = esq_inc[static_cast<std::size_t>(i)];
		double e1 = esq_inc[static_cast<std::size_t>((i + 1) % N)];
		double e2 = esq_opp[static_cast<std::size_t>(i)];
		double dA = 2.0 * std::sqrt(e0 * e2);
		double dB = 2.0 * std::sqrt(e1 * e2);
		if (dA > 1e-30 && dB > 1e-30) {
			double cosA = std::clamp((e0 + e2 - e1) / dA, -1.0, 1.0);
			double cosB = std::clamp((e1 + e2 - e0) / dB, -1.0, 1.0);
			double sinA = std::sqrt(1.0 - cosA * cosA);
			double sinB = std::sqrt(1.0 - cosB * cosB);
			cot_alpha[static_cast<std::size_t>((i + 1) % N)] = (sinA > 1e-15) ? cosA / sinA : 0.0;
			cot_beta[static_cast<std::size_t>(i)] = (sinB > 1e-15) ? cosB / sinB : 0.0;
		}
		double dT = 2.0 * std::sqrt(e0 * e1);
		theta_v[static_cast<std::size_t>(i)] = (dT > 1e-30) ?
			std::acos(std::clamp((e0 + e1 - e2) / dT, -1.0, 1.0)) : 0.0;
	}

	Eigen::Vector3d Hv = Eigen::Vector3d::Zero();
	double theta_sum = 0;

	for (int i = 0; i < N; ++i) {
		Eigen::Vector3d a = ring[static_cast<std::size_t>(i)] - o;
		Eigen::Vector3d b = ring[static_cast<std::size_t>((i + 1) % N)] - o;
		double e_sq0 = esq_inc[static_cast<std::size_t>(i)];
		double e_sq1 = esq_inc[static_cast<std::size_t>((i + 1) % N)];
		double e_sq2 = esq_opp[static_cast<std::size_t>(i)];

		theta_sum += theta_v[static_cast<std::size_t>(i)];

		Eigen::Vector3d axb = a.cross(b);
		double axb_norm = axb.norm();

		// Angle-weighted normal
		if (axb_norm > 1e-15) nv += (axb / axb_norm) * theta_v[static_cast<std::size_t>(i)];

		// Mean curvature vector: using opposite-angle cotangent weights (aligned with minsurf)
		// Weight of edge (o, ring[i]) = (cot_alpha[i] + cot_beta[i]) / 2
		// cot_alpha[i] = cot at ring[i-1] (in triangle i-1)
		// cot_beta[i]  = cot at ring[i+1] (in triangle i)
		Hv += (cot_alpha[static_cast<std::size_t>(i)] + cot_beta[static_cast<std::size_t>(i)]) / 2.0 * a;

		// Voronoi area (using cotA, cotB of the current triangle)
		double cotA = cot_alpha[static_cast<std::size_t>((i + 1) % N)];
		double cotB = cot_beta[static_cast<std::size_t>(i)];
		if (e_sq0 + e_sq1 >= e_sq2 && e_sq1 + e_sq2 >= e_sq0 && e_sq2 + e_sq0 >= e_sq1) {
			As += cotA * e_sq1 / 8.0 + cotB * e_sq0 / 8.0;
		} else if (e_sq0 + e_sq1 < e_sq2) {
			As += axb_norm / 4.0;
		} else {
			As += axb_norm / 8.0;
		}
	}

	if (nv.norm() > 1e-15) nv.normalize();
	if (As > 1e-20) {
		K = (2.0 * M_PI - theta_sum) / As;
		Lx = Hv;
		H = Hv.norm() / As / 2.0;
		if (Hv.dot(nv) < 0) H *= -1.0;
	}
}

// ── computeVertexGeometry ──────────────────────────────────

VertexGeometry computeVertexGeometry(const PeriodicTriMesh& mesh) {
	VertexGeometry g;
	const std::size_t ne = mesh.n_edges();
	const std::size_t nv = mesh.n_vertices();
	const Vec3d hp = mesh.halfPeriod();

	g.cotWeights.resize(ne);
	g.edgeVectors.resize(ne);
	for (std::size_t eid = 0; eid < ne; ++eid) {
		EH eh(static_cast<int>(eid));
		g.cotWeights[eid] = computeCotWeight(mesh, eh);
		HH he = mesh.halfedge_handle(eh, 0);
		Vec3d ev = makePeriod(mesh.point(mesh.to_vertex_handle(he)) - mesh.point(mesh.from_vertex_handle(he)), hp);
		g.edgeVectors[eid] = toEig(ev);
	}

	g.vertexAreas.resize(static_cast<Eigen::Index>(nv));
	g.vertexAreas.setZero();
	g.vertexNormals.resize(nv, Eigen::Vector3d::Zero());

	// Compute 1-ring (used for sensitivity and normals)
	g.vrings.resize(nv);
	for (std::size_t vid = 0; vid < nv; ++vid) {
		VH vh(static_cast<int>(vid));
		Eigen::Vector3d center;
		std::vector<Eigen::Vector3d> ring;
		getPeriodicRing(mesh, vh, center, ring);
		g.vrings[vid] = Compile1ring(center, ring);
		g.vertexNormals[vid] = g.vrings[vid].nv;
		g.vertexAreas[static_cast<Eigen::Index>(vid)] = g.vrings[vid].As;
	}

	return g;
}

// ── computeCotWeight ───────────────────────────────────────

double computeCotWeight(const PeriodicTriMesh& mesh, EH eh) {
	double cot_sum = 0.0;
	const Vec3d hp = mesh.halfPeriod();
	for (int side = 0; side < 2; ++side) {
		HH he = mesh.halfedge_handle(eh, side);
		if (mesh.is_boundary(he)) continue;
		HH heN = mesh.next_halfedge_handle(he);
		VH vOpp = mesh.to_vertex_handle(heN);
		VH v0 = mesh.from_vertex_handle(he);
		VH v1 = mesh.to_vertex_handle(he);
		Vec3d e0 = makePeriod(mesh.point(v0) - mesh.point(vOpp), hp);
		Vec3d e1 = makePeriod(mesh.point(v1) - mesh.point(vOpp), hp);
		Vec3d cross = e0 % e1;
		double sinA = cross.norm();
		double cosA = static_cast<double>(e0 | e1);
		if (std::abs(sinA) < 1e-15) continue;
		cot_sum += cosA / sinA;
	}
	return cot_sum / 2.0;
}

// ── assembleLaplacian ──────────────────────────────────────

Eigen::SparseMatrix<double> assembleLaplacian(
	const PeriodicTriMesh& mesh,
	const std::vector<double>& cotWeights) {
	const int nv = static_cast<int>(mesh.n_vertices());
	std::vector<Eigen::Triplet<double>> triplets;
	triplets.reserve(mesh.n_edges() * 2 + static_cast<std::size_t>(nv));

	for (auto v_it = mesh.vertices_begin(); v_it != mesh.vertices_end(); ++v_it) {
		double wsum = 0;
		for (auto voh_it = mesh.cvoh_iter(*v_it); voh_it.is_valid(); ++voh_it) {
			EH eh = mesh.edge_handle(*voh_it);
			double w = cotWeights[static_cast<std::size_t>(eh.idx())];
			VH vTo = mesh.to_vertex_handle(*voh_it);
			triplets.emplace_back((*v_it).idx(), vTo.idx(), w);
			wsum += w;
		}
		triplets.emplace_back((*v_it).idx(), (*v_it).idx(), -wsum);
	}

	Eigen::SparseMatrix<double> L(nv, nv);
	L.setFromTriplets(triplets.begin(), triplets.end());
	return L;
}

} // namespace xtpms
