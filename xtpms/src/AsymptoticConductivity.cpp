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

} // namespace

// ── faceFrame ──────────────────────────────────────────────

Eigen::Matrix3d faceFrame(const Eigen::Matrix3d& tri) {
	Eigen::Vector3d n = (tri.col(1) - tri.col(0)).cross(tri.col(2) - tri.col(0)) / 2.0;
	Eigen::Vector3d e1 = (tri.col(1) - tri.col(0)).normalized();
	Eigen::Vector3d e2 = n.cross(e1);
	if (e2.norm() > 1e-15) e2.normalize();
	Eigen::Matrix3d fr;
	fr.col(0) = e1; fr.col(1) = e2; fr.col(2) = n;
	return fr;
}

// ── secondFundamentalFormEdge ──────────────────────────────

Eigen::Vector3d secondFundamentalFormEdge(
	const Eigen::Matrix3d& tri,
	const Compile1ring& v0, const Compile1ring& v1, const Compile1ring& v2) {
	// be_ij = -(n_j - n_i) · (v_j - v_i)
	double be12 = -(v1.nv - v0.nv).dot(tri.col(1) - tri.col(0));
	double be23 = -(v2.nv - v1.nv).dot(tri.col(2) - tri.col(1));
	double be31 = -(v0.nv - v2.nv).dot(tri.col(0) - tri.col(2));
	return Eigen::Vector3d(be12, be23, be31);
}

// ── strainMatrixEdgeStretch ────────────────────────────────

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

// ── areaShapeDerivative ───────────────────────────────────

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

// ── Voigt ─────────────────────────────────────────────────

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

// ── scalarGradientMatrix ──────────────────────────────────

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

// ── assembleRHS ────────────────────────────────────────────

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

// ── evaluateConductivityTensor ─────────────────────────────

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

// ── solveAsymptoticConductivity ────────────────────────────

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

// ── computeSensitivity (full version, including second fundamental form) ──

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

		// Periodically wrapped triangle
		Eigen::Matrix3d tri = getFacePeriodTri(mesh, *f_it);

		// Face local coordinate frame
		Eigen::Matrix3d fr = faceFrame(tri);
		double A = fr.col(2).norm(); // = area
		if (A < 1e-20) continue;

		// Get 1-ring data for the three vertices
		Compile1ring vring[3] = {
			geom.vrings[static_cast<std::size_t>(idx[0])],
			geom.vrings[static_cast<std::size_t>(idx[1])],
			geom.vrings[static_cast<std::size_t>(idx[2])]
		};

		// Second fundamental form
		Eigen::Vector3d be = secondFundamentalFormEdge(tri, vring[0], vring[1], vring[2]);
		Eigen::Matrix3d Bn = strainMatrixEdgeStretch(tri, fr.col(0), fr.col(1));
		Eigen::Vector3d bformVoigt = Bn * be;
		Eigen::Matrix2d bform = fromVoigt3(bformVoigt); // 2x2 second fundamental form

		// Scalar gradient matrix 2x3
		Eigen::Matrix<double, 2, 3> G = scalarGradientMatrix(tri, fr);

		// Values of u at face vertices
		Eigen::Matrix3d ue;
		ue.row(0) = ulist.row(idx[0]);
		ue.row(1) = ulist.row(idx[1]);
		ue.row(2) = ulist.row(idx[2]);

		// grad u in local coordinate frame, 2x3
		Eigen::Matrix<double, 2, 3> gu = G * ue;

		// P_w = [t1, t2]^T 2x3
		Eigen::Matrix<double, 2, 3> pw = fr.leftCols<2>().transpose();

		// Mean curvature
		double H = bform.trace() / 2.0;
		Eigen::Matrix2d eye = Eigen::Matrix2d::Identity();

		// Sensitivity tensor: (gu + pw)^T * (bform - H*I) * (gu + pw)
		Eigen::Matrix3d sens_ij = (gu + pw).transpose() * (bform - H * eye) * (gu + pw);

		// Single-point quadrature: each vertex receives 1/3
		Eigen::Vector<double, 6> sv = toVoigt(2.0 * sens_ij * A / 3.0);
		sv.tail<3>() /= 2.0; // Voigt shear scaling

		// Area derivative
		Eigen::Vector3d dAdv = areaShapeDerivative(tri, fr, vring);

		for (int k = 0; k < 3; ++k) {
			result.vSens.row(idx[k]) += sv.transpose();
			result.aSens[idx[k]] += dAdv[k];
		}
	}

	// Note: do not divide by vertex area here; handled uniformly in tailorADC
	// (consistent with minsurf: asym_cond_sensitivity returns accumulated values)

	return result;
}

// ── evaluateADCObjective ──────────────────────────────────

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

// ── ConvergenceChecker ────────────────────────────────────

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

// ── tailorADC ─────────────────────────────────────────────

void tailorADC(PeriodicTriMesh& mesh, const TailorADCOptions& opts) {
	ConvergenceChecker conv(opts.convergeTol, opts.stepTol * opts.maxStep, opts.preconditionStrength);

	RemeshOptions remeshOpts = opts.remeshOpts;
	if (opts.enableRemesh && remeshOpts.targetLength < 0) {
		remeshOpts = defaultRemeshOptions(mesh);
		// Preserve user-specified non-default parameters
		if (opts.remeshOpts.outerIter != 1) remeshOpts.outerIter = opts.remeshOpts.outerIter;
		if (opts.remeshOpts.innerIter != 5) remeshOpts.innerIter = opts.remeshOpts.innerIter;
		if (opts.remeshOpts.adaptiveEps != 0.6) remeshOpts.adaptiveEps = opts.remeshOpts.adaptiveEps;
		std::cout << "fixed remesh targetLength = " << remeshOpts.targetLength
				  << " minLength = " << remeshOpts.minLength << "\n";
	}

	// Sanity check helper: nfLimit < 0 means no upper limit on face count
	auto meshSanityCheck = [&](const char* stage, int nfLimit) -> std::string {
		const int nv = static_cast<int>(mesh.n_vertices());
		const int nf = static_cast<int>(mesh.n_faces());
		if (nf == 0 || nv == 0)
			return std::string(stage) + ": mesh is empty (nv=" + std::to_string(nv) + " nf=" + std::to_string(nf) + ")";
		if (nfLimit > 0 && nf > nfLimit)
			return std::string(stage) + ": mesh exploded (nf=" + std::to_string(nf) + " > " + std::to_string(nfLimit) + ")";
		// Check for NaN vertices
		for (auto v_it = mesh.vertices_begin(); v_it != mesh.vertices_end(); ++v_it) {
			auto p = mesh.point(*v_it);
			if (std::isnan(p[0]) || std::isnan(p[1]) || std::isnan(p[2]))
				return std::string(stage) + ": NaN vertex detected (v" + std::to_string((*v_it).idx()) + ")";
		}
		// Check for excessive boundary edges (holes) — allow a few from remesh artifacts
		int nBnd = 0;
		for (auto e_it = mesh.edges_begin(); e_it != mesh.edges_end(); ++e_it)
			if (mesh.is_boundary(*e_it)) ++nBnd;
		if (nBnd > nf / 10)
			return std::string(stage) + ": too many boundary edges (" + std::to_string(nBnd) + "/" + std::to_string(nf) + " faces)";
		return "";
	};

	// Save the mesh from the best iteration; can roll back to it if mesh degenerates late in optimization
	PeriodicTriMesh bestMesh;
	double bestObjValue = -std::numeric_limits<double>::infinity();
	bool haveBest = false;
	auto saveBest = [&](double objVal, const Eigen::Matrix3d& kA) {
		// Physical validity check: APAC=kA_trace/3 should be in [0, 1], each diagonal element should also be in [0, 1]
		// Out-of-range values are typically numerical artifacts from sliver triangles making L non-PSD; reject them
		if (objVal <= 0 || objVal >= 1.0) return;
		for (int i = 0; i < 3; ++i) {
			if (kA(i, i) < -1e-6 || kA(i, i) > 1.0 + 1e-6) return;
		}
		if (objVal > bestObjValue) {
			bestObjValue = objVal;
			bestMesh = mesh;   // OpenMesh deep copy
			haveBest = true;
		}
	};
	auto restoreBest = [&]() {
		if (haveBest) {
			mesh = bestMesh;
			std::cerr << "[tailorADC] restored best mesh (obj=" << bestObjValue << ")\n";
		}
	};

	for (int iter = 0; iter < opts.maxIter; ++iter) {
		// [1] surgery
		if (opts.enableSurgery && iter >= opts.surgeryStartIter && iter % opts.surgeryInterval == 0 && iter > 0) {
			if (mesh.surgery(opts.surgeryOpts)) {
				mesh.garbage_collection();
				mesh.removeNonPeriodicIslands();
				// Strictly ensure single connected component: avoid multi-dimensional null space
				// of FEM Laplacian causing ill-conditioned kA solution (negative kA diagonals observed)
				int removed = mesh.keepLargestComponent();
				if (removed > 0)
					std::cout << "surgery: removed " << removed << " residual component(s)\n";
				std::cout << "surgery performed at iter " << iter
						  << " nv=" << mesh.n_vertices() << " nf=" << mesh.n_faces() << "\n";
				if (!opts.outputDir.empty()) {
					mesh.saveUnitCell(opts.outputDir + "/aftsur_" + std::to_string(iter) + ".obj");
				}
				// Reset convergence history after surgery (topology changed)
				conv.objHistory.clear();
				conv.stepHistory.clear();
			}
		}

		// [2] remesh
		if (opts.enableRemesh) {
			delaunayRemesh(mesh, remeshOpts);
			bool hasBoundary = false;
			for (auto e_it = mesh.edges_begin(); e_it != mesh.edges_end() && !hasBoundary; ++e_it) {
				if (mesh.is_boundary(*e_it)) hasBoundary = true;
			}
			if (hasBoundary) mesh.mergePeriodBoundary();
			// Clean up isolated vertices
			{
				mesh.request_vertex_status();
				bool cleaned = false;
				for (auto v_it = mesh.vertices_begin(); v_it != mesh.vertices_end(); ++v_it)
					if (mesh.valence(*v_it) == 0) { mesh.delete_vertex(*v_it, false); cleaned = true; }
				if (cleaned) mesh.garbage_collection();
			}
		}

		// [2.5] sanity check after surgery/remesh
		// At iter=0, do not check nf upper limit; accept high-density initial seed
		{
			int nfLimit = (iter == 0) ? -1 : opts.nfLimit;
			auto err = meshSanityCheck("after remesh", nfLimit);
			if (!err.empty()) {
				std::cerr << "tailorADC abort: " << err << " at iter " << iter << "\n";
				restoreBest();
				break;
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

		// [5.5] check for NaN in solution
		if (kA.hasNaN() || ulist.hasNaN()) {
			std::cerr << "tailorADC abort: NaN in ADC solution at iter " << iter << "\n";
			restoreBest();
			break;
		}

		// [6] objective
		ADCObjective obj = evaluateADCObjective(opts.objectiveType, kA);
		if (!std::isnan(obj.value) && obj.value > 0) {
			saveBest(obj.value, kA);
		}
		if (std::isnan(obj.value) || obj.value <= 0) {
			std::cerr << "tailorADC abort: invalid objective (" << obj.value << ") at iter " << iter << "\n";
			restoreBest();
			break;
		}

		// [7] sensitivity
		auto sens = computeSensitivity(mesh, geom, ulist);

		// [8] gradient (consistent with minsurf: first divide by vertex area, then divide by As)
		double As = geom.vertexAreas.sum();
		// Divide by vertex area (from area density to per-vertex value)
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
		double c = 1.0 / conv.estimatePrecondition(20.0);
		auto L = assembleLaplacian(mesh, geom.cotWeights);
		Eigen::SparseMatrix<double> G = -c * L;
		for (int i = 0; i < nv; ++i) G.coeffRef(i, i) += geom.vertexAreas[i];

		Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> precSolver(G);
		double fweight = c + 1.0;
		Eigen::VectorXd dn = -precSolver.solve(geom.vertexAreas.cwiseProduct(fweight * dfdvn).eval());

		// Sanity check: clamp dn to prevent numerical explosion
		if (dn.hasNaN() || !dn.allFinite()) {
			std::cerr << "tailorADC abort: NaN/Inf in preconditioned gradient at iter " << iter << "\n";
			restoreBest();
			break;
		}
		// Compute average edge length once per iteration (used for dn clamping)
		double avgEdgeLen = 0;
		{
			int nEdges = 0;
			for (auto e_it = mesh.edges_begin(); e_it != mesh.edges_end(); ++e_it) {
				avgEdgeLen += mesh.calc_edge_length(*e_it);
				++nEdges;
			}
			avgEdgeLen = (nEdges > 0) ? avgEdgeLen / nEdges : 0.1;
			double dnMax = 2.0 * avgEdgeLen;
			for (int i = 0; i < nv; ++i) {
				if (std::abs(dn[i]) > dnMax) dn[i] = (dn[i] > 0 ? dnMax : -dnMax);
			}
		}


		// [10] Build per-vertex displacement: ADC gradient (normal) + mean curvature flow
		std::vector<Eigen::Vector3d> stepVec(static_cast<std::size_t>(nv), Eigen::Vector3d::Zero());
		double maxMcf = 0;
		for (int i = 0; i < nv; ++i) {
			stepVec[static_cast<std::size_t>(i)] = dn[i] * geom.vertexNormals[static_cast<std::size_t>(i)];
			if (opts.mcfWeight > 0) {
				Eigen::Vector3d mcf = opts.mcfWeight * geom.vrings[static_cast<std::size_t>(i)].Lx;
				double mcfNorm = mcf.norm();
				maxMcf = std::max(maxMcf, mcfNorm);
				if (mcfNorm > 1.0) mcf *= 1.0 / mcfNorm;
				stepVec[static_cast<std::size_t>(i)] += mcf;
			}
		}

		// maximal unflip step
		double tbar = conv.estimateNextStep(opts.maxStep);
		double step = tbar;
		// maximal unflip step: per-face search, cos(60°) threshold
		const Vec3d hp = mesh.halfPeriod();
		const double cosThres = 0.5; // cos(60°)
		for (auto fit = mesh.faces_begin(); fit != mesh.faces_end(); ++fit) {
			double faceStep = step;
			int fidx[3];
			getFaceVertexIdx(mesh, *fit, fidx);
			Eigen::Vector3d oldE1 = toEig(makePeriod(mesh.point(VH(fidx[1])) - mesh.point(VH(fidx[0])), hp));
			Eigen::Vector3d oldE2 = toEig(makePeriod(mesh.point(VH(fidx[2])) - mesh.point(VH(fidx[0])), hp));
			Eigen::Vector3d oldN = oldE1.cross(oldE2);
			double oldLen = oldN.norm();
			if (oldLen < 1e-20) continue;
			oldN /= oldLen;
			while (faceStep > 1e-10) {
				Eigen::Vector3d np0 = toEig(mesh.point(VH(fidx[0]))) + faceStep * stepVec[static_cast<std::size_t>(fidx[0])];
				Eigen::Vector3d np1 = toEig(mesh.point(VH(fidx[1]))) + faceStep * stepVec[static_cast<std::size_t>(fidx[1])];
				Eigen::Vector3d np2 = toEig(mesh.point(VH(fidx[2]))) + faceStep * stepVec[static_cast<std::size_t>(fidx[2])];
				Eigen::Vector3d newE1 = toEig(makePeriod(toOM(np1) - toOM(np0), hp));
				Eigen::Vector3d newE2 = toEig(makePeriod(toOM(np2) - toOM(np0), hp));
				Eigen::Vector3d newN = newE1.cross(newE2);
				double newLen = newN.norm();
				if (newLen > 1e-20 && newN.dot(oldN) / newLen > cosThres) {
					break;
				}
				faceStep *= 0.7;
			}
			step = std::min(step, faceStep);
		}

		// [11] Backtracking line search: try step sizes, solve ADC, check if objective improves
		{
			// Save current vertex positions
			std::vector<Vec3d> savedPos(static_cast<std::size_t>(nv));
			for (auto v_it = mesh.vertices_begin(); v_it != mesh.vertices_end(); ++v_it)
				savedPos[static_cast<std::size_t>((*v_it).idx())] = mesh.point(*v_it);

			for (int ls = 0; ls < 15; ++ls) {
				// Trial displacement
				for (auto v_it = mesh.vertices_begin(); v_it != mesh.vertices_end(); ++v_it) {
					VH vh = *v_it;
					Eigen::Vector3d p = toEig(savedPos[static_cast<std::size_t>(vh.idx())]);
					p += step * stepVec[static_cast<std::size_t>(vh.idx())];
					mesh.set_point(vh, toOM(p));
				}
				// Evaluate objective
				auto trialGeom = computeVertexGeometry(mesh);
				Eigen::MatrixX3d trialU;
				Eigen::Matrix3d trialKA = solveAsymptoticConductivity(mesh, trialGeom, trialU);
				ADCObjective trialObj = evaluateADCObjective(opts.objectiveType, trialKA);

				if (!std::isnan(trialObj.value) && trialObj.value > obj.value) {
					// Accept this step size
					break;
				}
				// Restore and reduce step size
				for (auto v_it = mesh.vertices_begin(); v_it != mesh.vertices_end(); ++v_it)
					mesh.set_point(*v_it, savedPos[static_cast<std::size_t>((*v_it).idx())]);
				if (ls >= 10) break; // give up, keep original positions
				step *= 0.7;
			}
		}

		// [11.5] post-displacement sanity check (iter=0 has no nf limit; accept high-density initial mesh)
		{
			int nfLim = (iter == 0) ? -1 : opts.nfLimit;
			auto err = meshSanityCheck("after displacement", nfLim);
			if (!err.empty()) {
				std::cerr << "tailorADC abort: " << err << " at iter " << iter << "\n";
				restoreBest();
				break;
			}
		}

		// Displacement statistics (maxMcf already computed in step [10])
		double maxDisp = 0, maxDn = dn.cwiseAbs().maxCoeff();
		for (int i = 0; i < nv; ++i)
			maxDisp = std::max(maxDisp, step * stepVec[static_cast<std::size_t>(i)].norm());
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
