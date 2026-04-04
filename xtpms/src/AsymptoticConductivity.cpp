#include "AsymptoticConductivity.h"

#include <Eigen/SparseCholesky>
#include <cmath>
#include <iostream>
#include <numeric>

namespace xtpms {

namespace {

using Vec3d = PeriodicTriMesh::Vec3d;
using VH = PeriodicTriMesh::VertexHandle;
using EH = PeriodicTriMesh::EdgeHandle;
using HH = PeriodicTriMesh::HalfedgeHandle;
using FH = PeriodicTriMesh::FaceHandle;

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

// 获取面的三个顶点坐标（周期包装到 v0 附近），返回 3x3 矩阵（列为顶点）
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

// 获取顶点的周期 1-ring 邻域（通过面遍历保证 CCW 顺序）
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

	// 面遍历保证 CCW 顺序
	HH he_start = mesh.halfedge_handle(vh);
	HH he = he_start;
	do {
		VH vn = mesh.to_vertex_handle(he);
		outRing.push_back(outCenter + toEig(makePeriod(mesh.point(vn) - mesh.point(vh), hp)));
		he = mesh.opposite_halfedge_handle(mesh.prev_halfedge_handle(he));
	} while (he != he_start && outRing.size() < 30);
}

} // namespace

// ══════════════════════════════════════════════════════════════
// Compile1ring
// ══════════════════════════════════════════════════════════════

Compile1ring::Compile1ring(const Eigen::Vector3d& o, const std::vector<Eigen::Vector3d>& ring) {
	if (ring.size() < 3) return;
	const int N = static_cast<int>(ring.size());

	std::vector<double> esq_inc(static_cast<std::size_t>(N));
	std::vector<double> esq_opp(static_cast<std::size_t>(N));
	for (int i = 0; i < N; ++i) {
		esq_inc[static_cast<std::size_t>(i)] = (ring[static_cast<std::size_t>(i)] - o).squaredNorm();
		esq_opp[static_cast<std::size_t>(i)] = (ring[static_cast<std::size_t>((i + 1) % N)] - ring[static_cast<std::size_t>(i)]).squaredNorm();
	}

	// 预计算所有角的余切（和 minsurf Compile1ring 对齐）
	// 三角形 i = (o, ring[i], ring[i+1])
	// cotA_i = cot(angle at ring[i])   → 存到 cot_alpha[(i+1)%N]
	// cotB_i = cot(angle at ring[i+1]) → 存到 cot_beta[i]
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

		// 角度加权法向
		if (axb_norm > 1e-15) nv += (axb / axb_norm) * theta_v[static_cast<std::size_t>(i)];

		// 均曲向量：用对角余切权（和 minsurf 对齐）
		// 边 (o, ring[i]) 的权重 = (cot_alpha[i] + cot_beta[i]) / 2
		// cot_alpha[i] = cot at ring[i-1] (在三角形 i-1 中)
		// cot_beta[i]  = cot at ring[i+1] (在三角形 i 中)
		Hv += (cot_alpha[static_cast<std::size_t>(i)] + cot_beta[static_cast<std::size_t>(i)]) / 2.0 * a;

		// Voronoi 面积（用当前三角形的 cotA, cotB）
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

// ══════════════════════════════════════════════════════════════
// faceFrame
// ══════════════════════════════════════════════════════════════

Eigen::Matrix3d faceFrame(const Eigen::Matrix3d& tri) {
	Eigen::Vector3d n = (tri.col(1) - tri.col(0)).cross(tri.col(2) - tri.col(0)) / 2.0;
	Eigen::Vector3d e1 = (tri.col(1) - tri.col(0)).normalized();
	Eigen::Vector3d e2 = n.cross(e1);
	if (e2.norm() > 1e-15) e2.normalize();
	Eigen::Matrix3d fr;
	fr.col(0) = e1; fr.col(1) = e2; fr.col(2) = n;
	return fr;
}

// ══════════════════════════════════════════════════════════════
// secondFundamentalFormEdge
// ══════════════════════════════════════════════════════════════

Eigen::Vector3d secondFundamentalFormEdge(
	const Eigen::Matrix3d& tri,
	const Compile1ring& v0, const Compile1ring& v1, const Compile1ring& v2) {
	// be_ij = -(n_j - n_i) · (v_j - v_i)
	double be12 = -(v1.nv - v0.nv).dot(tri.col(1) - tri.col(0));
	double be23 = -(v2.nv - v1.nv).dot(tri.col(2) - tri.col(1));
	double be31 = -(v0.nv - v2.nv).dot(tri.col(0) - tri.col(2));
	return Eigen::Vector3d(be12, be23, be31);
}

// ══════════════════════════════════════════════════════════════
// strainMatrixEdgeStretch
// ══════════════════════════════════════════════════════════════

Eigen::Matrix3d strainMatrixEdgeStretch(
	const Eigen::Matrix3d& tri,
	const Eigen::Vector3d& e1, const Eigen::Vector3d& e2) {
	Eigen::Matrix<double, 3, 2> fram;
	fram.col(0) = e1; fram.col(1) = e2;

	// tri1 = cyclic shift: [v1, v2, v0]
	Eigen::Matrix3d tri1;
	tri1.col(0) = tri.col(1); tri1.col(1) = tri.col(2); tri1.col(2) = tri.col(0);

	// d(i,j) = e_i · (v_{j+1} - v_j) for tangent components
	Eigen::Matrix<double, 2, 3> d = fram.transpose() * (tri1 - tri);

	// Build A from d
	Eigen::Matrix3d A;
	A << d(0, 0) * d(0, 0), d(1, 0) * d(1, 0), d(0, 0) * d(1, 0),
		 d(0, 1) * d(0, 1), d(1, 1) * d(1, 1), d(0, 1) * d(1, 1),
		 d(0, 2) * d(0, 2), d(1, 2) * d(1, 2), d(0, 2) * d(1, 2);

	return A.inverse();
}

// ══════════════════════════════════════════════════════════════
// areaShapeDerivative
// ══════════════════════════════════════════════════════════════

Eigen::Vector3d areaShapeDerivative(
	const Eigen::Matrix3d& tri, const Eigen::Matrix3d& fr,
	const Compile1ring v[3]) {
	auto Bn = strainMatrixEdgeStretch(tri, fr.col(0), fr.col(1));
	auto be = secondFundamentalFormEdge(tri, v[0], v[1], v[2]);
	Eigen::Vector3d bform = Bn * be; // {eps11, eps22, 2*eps12}
	double H2 = bform[0] + bform[1]; // trace of 2nd fund. form = 2H
	double A = fr.col(2).norm();
	return Eigen::Vector3d(-H2 * A / 3.0, -H2 * A / 3.0, -H2 * A / 3.0);
}

// ══════════════════════════════════════════════════════════════
// Voigt
// ══════════════════════════════════════════════════════════════

Eigen::Vector<double, 6> toVoigt(const Eigen::Matrix3d& M) {
	Eigen::Vector<double, 6> v;
	for (int i = 0; i < 3; ++i) {
		v[i] = M(i, i);
		v[i + 3] = 2.0 * M((i + 1) % 3, (i + 2) % 3);
	}
	return v;
}

Eigen::Matrix3d fromVoigt6(const Eigen::Vector<double, 6>& v) {
	Eigen::Matrix3d M;
	for (int i = 0; i < 3; ++i) {
		M(i, i) = v[i];
		int j = (i + 1) % 3, k = (i + 2) % 3;
		M(j, k) = M(k, j) = v[i + 3] / 2.0;
	}
	return M;
}

Eigen::Vector3d toVoigt2(const Eigen::Matrix2d& M) {
	return Eigen::Vector3d(M(0, 0), M(1, 1), 2.0 * M(0, 1));
}

Eigen::Matrix2d fromVoigt3(const Eigen::Vector3d& v) {
	Eigen::Matrix2d M;
	M(0, 0) = v[0]; M(1, 1) = v[1];
	M(0, 1) = M(1, 0) = v[2] / 2.0;
	return M;
}

// ══════════════════════════════════════════════════════════════
// scalarGradientMatrix
// ══════════════════════════════════════════════════════════════

Eigen::Matrix<double, 2, 3> scalarGradientMatrix(
	const Eigen::Matrix3d& tri, const Eigen::Matrix3d& fr) {
	Eigen::Matrix2d V;
	V <<
		fr.leftCols<2>().transpose() * (tri.col(1) - tri.col(0)),
		fr.leftCols<2>().transpose() * (tri.col(2) - tri.col(0));
	Eigen::Matrix<double, 2, 3> S;
	S << -1, 1, 0, -1, 0, 1;
	return V.transpose().lu().solve(S);
}

// ══════════════════════════════════════════════════════════════
// computeVertexGeometry
// ══════════════════════════════════════════════════════════════

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

	// 计算 1-ring（用于 sensitivity 和法向）
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

// ══════════════════════════════════════════════════════════════
// assembleLaplacian
// ══════════════════════════════════════════════════════════════

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

// ══════════════════════════════════════════════════════════════
// assembleRHS
// ══════════════════════════════════════════════════════════════

Eigen::MatrixX3d assembleRHS(
	const PeriodicTriMesh& mesh,
	const std::vector<double>& cotWeights,
	const std::vector<Eigen::Vector3d>& edgeVectors) {
	const int nv = static_cast<int>(mesh.n_vertices());
	Eigen::MatrixX3d blist(nv, 3);
	blist.setZero();

	for (auto v_it = mesh.vertices_begin(); v_it != mesh.vertices_end(); ++v_it) {
		const int vi = (*v_it).idx();
		for (auto voh_it = mesh.cvoh_iter(*v_it); voh_it.is_valid(); ++voh_it) {
			EH eh = mesh.edge_handle(*voh_it);
			Eigen::Vector3d ev = edgeVectors[static_cast<std::size_t>(eh.idx())];
			if (mesh.halfedge_handle(eh, 0) != *voh_it) ev = -ev;
			double w = cotWeights[static_cast<std::size_t>(eh.idx())];
			blist.row(vi) -= w * ev.transpose();
		}
	}
	return blist;
}

// ══════════════════════════════════════════════════════════════
// evaluateConductivityTensor
// ══════════════════════════════════════════════════════════════

Eigen::Matrix3d evaluateConductivityTensor(
	const PeriodicTriMesh& mesh,
	const Eigen::MatrixX3d& blist,
	const Eigen::MatrixX3d& ulist,
	double totalArea) {
	Eigen::Matrix3d nnTsum = Eigen::Matrix3d::Zero();
	for (auto f_it = mesh.faces_begin(); f_it != mesh.faces_end(); ++f_it) {
		Eigen::Matrix3d tri = getFacePeriodTri(mesh, *f_it);
		Eigen::Vector3d an = (tri.col(1) - tri.col(0)).cross(tri.col(2) - tri.col(0)) * 0.5;
		double anl = an.norm();
		if (anl > 1e-15) {
			nnTsum += an * an.transpose() / anl;
		}
	}

	Eigen::Matrix3d kA;
	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j <= i; ++j) {
			double uD2u = -blist.col(i).dot(ulist.col(j));
			kA(i, j) = (i == j ? 1.0 : 0.0) - nnTsum(i, j) / totalArea - uD2u / totalArea;
			kA(j, i) = kA(i, j);
		}
	}
	return kA;
}

// ══════════════════════════════════════════════════════════════
// solveAsymptoticConductivity
// ══════════════════════════════════════════════════════════════

Eigen::Matrix3d solveAsymptoticConductivity(
	PeriodicTriMesh& mesh,
	const VertexGeometry& geom,
	Eigen::MatrixX3d& outU) {
	auto L = assembleLaplacian(mesh, geom.cotWeights);
	auto blist = assembleRHS(mesh, geom.cotWeights, geom.edgeVectors);

	for (int k = 0; k < 3; ++k)
		blist.col(k).array() -= blist.col(k).mean();

	Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver(L);
	if (solver.info() != Eigen::Success) {
		std::cerr << "solveAsymptoticConductivity: factorization failed"
				  << " nv=" << mesh.n_vertices() << " nf=" << mesh.n_faces() << "\n";
		outU = Eigen::MatrixX3d::Zero(mesh.n_vertices(), 3);
		return Eigen::Matrix3d::Zero();
	}

	outU = solver.solve(blist);
	for (int k = 0; k < 3; ++k)
		outU.col(k).array() -= outU.col(k).mean();

	double As = geom.vertexAreas.sum();
	return evaluateConductivityTensor(mesh, blist, outU, As);
}

// ══════════════════════════════════════════════════════════════
// computeSensitivity（完整版，含第二基本形式）
// ══════════════════════════════════════════════════════════════

SensitivityResult computeSensitivity(
	const PeriodicTriMesh& mesh,
	const VertexGeometry& geom,
	const Eigen::MatrixX3d& ulist) {
	const int nv = static_cast<int>(mesh.n_vertices());
	SensitivityResult result;
	result.vSens = Eigen::MatrixXd::Zero(nv, 6);
	result.aSens = Eigen::VectorXd::Zero(nv);

	for (auto f_it = mesh.faces_begin(); f_it != mesh.faces_end(); ++f_it) {
		int idx[3];
		getFaceVertexIdx(mesh, *f_it, idx);

		// 周期包装后的三角形
		Eigen::Matrix3d tri = getFacePeriodTri(mesh, *f_it);

		// 面局部坐标系
		Eigen::Matrix3d fr = faceFrame(tri);
		double A = fr.col(2).norm(); // = area
		if (A < 1e-20) continue;

		// 取三个顶点的 1-ring 数据
		Compile1ring vring[3] = {
			geom.vrings[static_cast<std::size_t>(idx[0])],
			geom.vrings[static_cast<std::size_t>(idx[1])],
			geom.vrings[static_cast<std::size_t>(idx[2])]
		};

		// 第二基本形式
		Eigen::Vector3d be = secondFundamentalFormEdge(tri, vring[0], vring[1], vring[2]);
		Eigen::Matrix3d Bn = strainMatrixEdgeStretch(tri, fr.col(0), fr.col(1));
		Eigen::Vector3d bformVoigt = Bn * be;
		Eigen::Matrix2d bform = fromVoigt3(bformVoigt); // 2x2 第二基本形式

		// 标量梯度矩阵 2x3
		Eigen::Matrix<double, 2, 3> G = scalarGradientMatrix(tri, fr);

		// u 在面顶点的值
		Eigen::Matrix3d ue;
		ue.row(0) = ulist.row(idx[0]);
		ue.row(1) = ulist.row(idx[1]);
		ue.row(2) = ulist.row(idx[2]);

		// grad u 在局部坐标系下 2x3
		Eigen::Matrix<double, 2, 3> gu = G * ue;

		// P_w = [t1, t2]^T 2x3
		Eigen::Matrix<double, 2, 3> pw = fr.leftCols<2>().transpose();

		// 均曲率
		double H = bform.trace() / 2.0;
		Eigen::Matrix2d eye = Eigen::Matrix2d::Identity();

		// 敏感度张量: (gu + pw)^T * (bform - H*I) * (gu + pw)
		Eigen::Matrix3d sens_ij = (gu + pw).transpose() * (bform - H * eye) * (gu + pw);

		// 单点积分：每个顶点分配 1/3
		Eigen::Vector<double, 6> sv = toVoigt(2.0 * sens_ij * A / 3.0);
		sv.tail<3>() /= 2.0; // Voigt shear scaling

		// 面积导数
		Eigen::Vector3d dAdv = areaShapeDerivative(tri, fr, vring);

		for (int k = 0; k < 3; ++k) {
			result.vSens.row(idx[k]) += sv.transpose();
			result.aSens[idx[k]] += dAdv[k];
		}
	}

	// 注意：不在这里除以顶点面积，由 tailorADC 中统一处理
	// （与 minsurf 一致：asym_cond_sensitivity 返回累加值）

	return result;
}

// ══════════════════════════════════════════════════════════════
// evaluateADCObjective
// ══════════════════════════════════════════════════════════════

ADCObjective evaluateADCObjective(const std::string& type, const Eigen::Matrix3d& kA) {
	ADCObjective obj;
	obj.value = 0;
	obj.gradient.setZero();

	if (type == "apac" || type == "trace") {
		obj.value = kA.trace() / 3.0;
		obj.gradient[0] = 1.0 / 3.0;
		obj.gradient[1] = 1.0 / 3.0;
		obj.gradient[2] = 1.0 / 3.0;
	} else if (type == "k00") {
		obj.value = kA(0, 0);
		obj.gradient[0] = 1.0;
	} else if (type == "k11") {
		obj.value = kA(1, 1);
		obj.gradient[1] = 1.0;
	} else if (type == "k22") {
		obj.value = kA(2, 2);
		obj.gradient[2] = 1.0;
	} else if (type == "iso") {
		Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(kA);
		Eigen::Vector3d lambda = eig.eigenvalues();
		double spread = lambda[2] - lambda[0];
		obj.value = 1.0 - spread * spread;
		Eigen::Vector3d vMax = eig.eigenvectors().col(2);
		Eigen::Vector3d vMin = eig.eigenvectors().col(0);
		Eigen::Matrix3d dObjdK = -2.0 * spread * (vMax * vMax.transpose() - vMin * vMin.transpose());
		auto dv = toVoigt(dObjdK);
		dv.tail<3>() *= 2.0;
		obj.gradient = dv;
	} else {
		std::cerr << "Unknown ADC objective type: " << type << "\n";
		obj.value = kA.trace() / 3.0;
		obj.gradient[0] = obj.gradient[1] = obj.gradient[2] = 1.0 / 3.0;
	}

	return obj;
}

// ══════════════════════════════════════════════════════════════
// ConvergenceChecker
// ══════════════════════════════════════════════════════════════

bool ConvergenceChecker::operator()(double obj, double step) {
	objHistory.push_back(obj);
	stepHistory.push_back(step);

	if (static_cast<int>(objHistory.size()) < histSize) return false;

	const int n = histSize - 1;
	const int start = static_cast<int>(objHistory.size()) - n;
	double sx = 0, sy = 0, sxy = 0, sx2 = 0;
	for (int i = 0; i < n; ++i) {
		double x = static_cast<double>(i);
		double y = objHistory[static_cast<std::size_t>(start + i)];
		sx += x; sy += y; sxy += x * y; sx2 += x * x;
	}
	double slope = (n * sxy - sx * sy) / (n * sx2 - sx * sx);

	if (std::abs(slope) < objTol) {
		bool allSmall = true;
		for (int i = 0; i < n; ++i) {
			if (std::abs(stepHistory[static_cast<std::size_t>(start + i)]) > stepTol) {
				allSmall = false;
				break;
			}
		}
		if (allSmall || std::abs(slope) < objTol * 0.1) return true;
	}
	return false;
}

double ConvergenceChecker::estimatePrecondition(double cmax) const {
	return std::min(cmax, c0);
}

double ConvergenceChecker::estimateNextStep(double tmax) const {
	if (stepHistory.empty()) return tmax;
	return std::min(tmax, std::abs(stepHistory.back()) * 1.5 + 0.01);
}

// ══════════════════════════════════════════════════════════════
// tailorADC
// ══════════════════════════════════════════════════════════════

void tailorADC(PeriodicTriMesh& mesh, const TailorADCOptions& opts) {
	ConvergenceChecker conv(opts.convergeTol, opts.stepTol * opts.maxStep, opts.preconditionStrength);

	// 固定 remesh 目标边长：根据周期大小估算（和 minsurf 对齐）
	RemeshOptions remeshOpts = opts.remeshOpts;
	if (opts.enableRemesh && remeshOpts.targetLength < 0) {
		const Vec3d hp = mesh.halfPeriod();
		double minPeriod = 2.0 * std::min({static_cast<double>(hp[0]),
										   static_cast<double>(hp[1]),
										   static_cast<double>(hp[2])});
		remeshOpts.targetLength = minPeriod * 0.15; // 0.3 for period=2
		if (remeshOpts.minLength < 0)
			remeshOpts.minLength = remeshOpts.targetLength * 0.1;
		std::cout << "fixed remesh targetLength = " << remeshOpts.targetLength
				  << " (minPeriod=" << minPeriod << ")\n";
	}

	for (int iter = 0; iter < opts.maxIter; ++iter) {
		// [1] surgery
		if (opts.enableSurgery && iter >= opts.surgeryStartIter && iter % opts.surgeryInterval == 0 && iter > 0) {
			if (mesh.surgery(opts.surgeryOpts)) {
				mesh.garbage_collection();
				mesh.removeNonPeriodicIslands();
				std::cout << "surgery performed at iter " << iter
						  << " nv=" << mesh.n_vertices() << " nf=" << mesh.n_faces() << "\n";
				if (!opts.outputDir.empty()) {
					mesh.saveUnitCell(opts.outputDir + "/aftsur_" + std::to_string(iter) + ".obj");
				}
			}
		}

		// [2] remesh
		if (opts.enableRemesh && iter > 0) {
			delaunayRemesh(mesh, remeshOpts);
			bool hasBoundary = false;
			for (auto e_it = mesh.edges_begin(); e_it != mesh.edges_end() && !hasBoundary; ++e_it) {
				if (mesh.is_boundary(*e_it)) hasBoundary = true;
			}
			if (hasBoundary) mesh.mergePeriodBoundary();
			// 清理孤立顶点
			{
				mesh.request_vertex_status();
				bool cleaned = false;
				for (auto v_it = mesh.vertices_begin(); v_it != mesh.vertices_end(); ++v_it)
					if (mesh.valence(*v_it) == 0) { mesh.delete_vertex(*v_it, false); cleaned = true; }
				if (cleaned) mesh.garbage_collection();
			}
		}

		// [3] save intermediate
		if (!opts.outputDir.empty() && iter % opts.saveInterval == 0) {
			mesh.saveUnitCell(opts.outputDir + "/iter_" + std::to_string(iter) + ".obj");
		}

		// [4] geometry
		auto geom = computeVertexGeometry(mesh);
		const int nv = static_cast<int>(mesh.n_vertices());

		// [5] solve ADC
		Eigen::MatrixX3d ulist;
		Eigen::Matrix3d kA = solveAsymptoticConductivity(mesh, geom, ulist);

		// [6] objective
		ADCObjective obj = evaluateADCObjective(opts.objectiveType, kA);
		if (std::isnan(obj.value)) {
			std::cerr << "tailorADC: NaN objective at iter " << iter << "\n";
			break;
		}

		// [7] sensitivity
		auto sens = computeSensitivity(mesh, geom, ulist);

		// [8] gradient（与 minsurf 一致：先除 vertex area，再除 As）
		double As = geom.vertexAreas.sum();
		// 除以顶点面积（从面积密度到每顶点值）
		for (int i = 0; i < nv; ++i) {
			double ai = geom.vertexAreas[i];
			if (ai > 1e-15) {
				sens.vSens.row(i) /= ai;
				sens.aSens[i] /= ai;
			}
		}
		auto kAv = toVoigt(kA);
		kAv.tail<3>() /= 2.0;
		Eigen::VectorXd dfdvn = (sens.vSens / As - sens.aSens * kAv.transpose() / As) * (-obj.gradient);

		// [9] preconditioned descent direction
		double c = conv.estimatePrecondition(20.0);
		auto L = assembleLaplacian(mesh, geom.cotWeights);
		Eigen::SparseMatrix<double> G = -c * L;
		for (int i = 0; i < nv; ++i) G.coeffRef(i, i) += geom.vertexAreas[i];

		Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> precSolver(G);
		double fweight = c + 1.0;
		Eigen::VectorXd dn = -precSolver.solve(geom.vertexAreas.cwiseProduct(fweight * dfdvn).eval());

		if (iter == 0) {
			std::cout << "  dfdvn: min=" << dfdvn.minCoeff() << " max=" << dfdvn.maxCoeff()
					  << " norm=" << dfdvn.norm() << "\n";
			std::cout << "  dn: min=" << dn.minCoeff() << " max=" << dn.maxCoeff()
					  << " norm=" << dn.norm() << "\n";
			std::cout << "  As=" << As << " c=" << c << " fweight=" << fweight << "\n";
			std::cout << "  vSens norm=" << sens.vSens.norm() << " aSens norm=" << sens.aSens.norm() << "\n";
		}

		// [10] 构造每个顶点的位移向量：ADC 梯度（法向）+ 均曲率流
		// mean_curvature_vector = Σ cot_e * edge_vec（Laplace-Beltrami 算子作用于坐标）
		// 驱动曲面趋向极小曲面（H→0），加速收敛
		std::vector<Eigen::Vector3d> stepVec(static_cast<std::size_t>(nv), Eigen::Vector3d::Zero());
		for (int i = 0; i < nv; ++i) {
			// ADC 梯度部分：法向位移
			stepVec[static_cast<std::size_t>(i)] = dn[i] * geom.vertexNormals[static_cast<std::size_t>(i)];
			// 均曲率流部分（area regularization）
			if (opts.mcfWeight > 0) {
				stepVec[static_cast<std::size_t>(i)] += opts.mcfWeight * geom.vrings[static_cast<std::size_t>(i)].Lx;
			}
		}

		// maximal unflip step
		double tbar = conv.estimateNextStep(opts.maxStep);
		double step = tbar;
		const Vec3d hp = mesh.halfPeriod();
		for (int trial = 0; trial < 20; ++trial) {
			bool valid = true;
			for (auto fit = mesh.faces_begin(); fit != mesh.faces_end() && valid; ++fit) {
				int fidx[3];
				getFaceVertexIdx(mesh, *fit, fidx);
				auto shift = [&](int vi) {
					return toEig(mesh.point(VH(vi))) + step * stepVec[static_cast<std::size_t>(vi)];
				};
				Eigen::Vector3d np0 = shift(fidx[0]);
				Eigen::Vector3d np1 = shift(fidx[1]);
				Eigen::Vector3d np2 = shift(fidx[2]);
				Eigen::Vector3d oldE1 = toEig(makePeriod(mesh.point(VH(fidx[1])) - mesh.point(VH(fidx[0])), hp));
				Eigen::Vector3d oldE2 = toEig(makePeriod(mesh.point(VH(fidx[2])) - mesh.point(VH(fidx[0])), hp));
				Eigen::Vector3d oldN = oldE1.cross(oldE2);
				Eigen::Vector3d newE1 = toEig(makePeriod(toOM(np1) - toOM(np0), hp));
				Eigen::Vector3d newE2 = toEig(makePeriod(toOM(np2) - toOM(np0), hp));
				Eigen::Vector3d newN = newE1.cross(newE2);
				if (oldN.dot(newN) <= 0) valid = false;
			}
			if (valid) break;
			step *= 0.7;
		}

		// [11] 回溯线搜索：尝试步长，求解 ADC，检查目标是否改善
		{
			// 保存当前顶点位置
			std::vector<Vec3d> savedPos(static_cast<std::size_t>(nv));
			for (auto v_it = mesh.vertices_begin(); v_it != mesh.vertices_end(); ++v_it)
				savedPos[static_cast<std::size_t>((*v_it).idx())] = mesh.point(*v_it);

			for (int ls = 0; ls < 15; ++ls) {
				// 试探位移
				for (auto v_it = mesh.vertices_begin(); v_it != mesh.vertices_end(); ++v_it) {
					VH vh = *v_it;
					Eigen::Vector3d p = toEig(savedPos[static_cast<std::size_t>(vh.idx())]);
					p += step * stepVec[static_cast<std::size_t>(vh.idx())];
					mesh.set_point(vh, toOM(p));
				}
				// 评估目标
				auto trialGeom = computeVertexGeometry(mesh);
				Eigen::MatrixX3d trialU;
				Eigen::Matrix3d trialKA = solveAsymptoticConductivity(mesh, trialGeom, trialU);
				ADCObjective trialObj = evaluateADCObjective(opts.objectiveType, trialKA);

				if (trialObj.value > obj.value || ls >= 10) {
					// 接受此步长
					break;
				}
				// 恢复并缩小步长
				for (auto v_it = mesh.vertices_begin(); v_it != mesh.vertices_end(); ++v_it)
					mesh.set_point(*v_it, savedPos[static_cast<std::size_t>((*v_it).idx())]);
				step *= 0.7;
			}
		}

		// 计算位移统计
		double maxDisp = 0, maxDn = 0, maxMcf = 0;
		for (int i = 0; i < nv; ++i) {
			maxDn = std::max(maxDn, std::abs(dn[i]));
			if (opts.mcfWeight > 0)
				maxMcf = std::max(maxMcf, geom.vrings[static_cast<std::size_t>(i)].Lx.norm());
			maxDisp = std::max(maxDisp, step * stepVec[static_cast<std::size_t>(i)].norm());
		}
		std::cout << "iter=" << iter << " obj=" << obj.value
				  << " step=" << step << " kA_trace=" << kA.trace()
				  << " nv=" << nv << " nf=" << mesh.n_faces()
				  << " maxDisp=" << maxDisp << " maxDn=" << maxDn
				  << " maxMcf=" << maxMcf << "\n";

		// [12] convergence
		if (conv(obj.value, step)) {
			std::cout << "tailorADC converged at iter " << iter << "\n";
			break;
		}
	}

	// save final
	if (!opts.outputDir.empty()) {
		mesh.saveUnitCell(opts.outputDir + "/final.obj");
	}
}

} // namespace xtpms
