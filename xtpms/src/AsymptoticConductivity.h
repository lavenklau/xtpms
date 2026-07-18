#pragma once

// Asymptotic conductivity (ADC): FEM solve, sensitivity, shape optimization.
// Depends on VertexGeometry for 1-ring curvature and Laplacian assembly,
// and PeriodicRemesh for remeshing during optimization.

#include "VertexGeometry.h"
#include "PeriodicRemesh.h"

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <string>
#include <vector>

namespace xtpms {

// ── ADC differential geometry helpers ──────────────────────

// Face local coordinate frame [e1, e2, n] (column vectors), where n = (v1-v0)x(v2-v0)/2
Eigen::Matrix3d faceFrame(const Eigen::Matrix3d& tri);

// Second fundamental form (edge form): be_ij = -(n_j - n_i) . (v_j - v_i)
Eigen::Vector3d secondFundamentalFormEdge(const Eigen::Matrix3d& tri,
										  const Compile1ring& v0,
										  const Compile1ring& v1,
										  const Compile1ring& v2);

// Edge stretch to strain matrix Bn: {eps11, eps22, 2*eps12} = Bn * {be12, be23, be31}
Eigen::Matrix3d strainMatrixEdgeStretch(const Eigen::Matrix3d& tri,
										const Eigen::Vector3d& e1,
										const Eigen::Vector3d& e2);

// Area shape derivative (derivative w.r.t. normal displacement), returns contribution from each of
// 3 vertices
Eigen::Vector3d
areaShapeDerivative(const Eigen::Matrix3d& tri, const Eigen::Matrix3d& fr, const Compile1ring v[3]);

// Voigt notation conversions
Eigen::Vector<double, 6> toVoigt(const Eigen::Matrix3d& M);
Eigen::Matrix3d fromVoigt6(const Eigen::Vector<double, 6>& v);
Eigen::Vector3d toVoigt2(const Eigen::Matrix2d& M);
Eigen::Matrix2d fromVoigt3(const Eigen::Vector3d& v);

// 2x3 scalar gradient matrix (in local coordinate frame)
Eigen::Matrix<double, 2, 3> scalarGradientMatrix(const Eigen::Matrix3d& tri,
												 const Eigen::Matrix3d& fr);

// Assemble RHS
Eigen::MatrixX3d assembleRHS(const PeriodicTriMesh& mesh,
							 const std::vector<double>& cotWeights,
							 const std::vector<Eigen::Vector3d>& edgeVectors);

// Solve ADC tensor
Eigen::Matrix3d solveAsymptoticConductivity(PeriodicTriMesh& mesh,
											const VertexGeometry& geom,
											Eigen::MatrixX3d& outU); // output solution vectors

// Evaluate ADC tensor
Eigen::Matrix3d evaluateConductivityTensor(const PeriodicTriMesh& mesh,
										   const Eigen::MatrixX3d& blist,
										   const Eigen::MatrixX3d& ulist,
										   double totalArea);

// Shape sensitivity: returns (n_vertices x 6) Voigt sensitivity and area sensitivity
struct SensitivityResult {
	Eigen::MatrixXd vSens; // n_vertices x 6 (Voigt)
	Eigen::VectorXd aSens; // n_vertices
};

SensitivityResult computeSensitivity(const PeriodicTriMesh& mesh,
									 const VertexGeometry& geom,
									 const Eigen::MatrixX3d& ulist);

// ADC objective function
struct ADCObjective {
	double value;
	Eigen::Vector<double, 6> gradient; // derivative w.r.t. kA Voigt components
};

ADCObjective evaluateADCObjective(const std::string& type, const Eigen::Matrix3d& kA);

// Convergence checker
struct ConvergenceChecker {
	int histSize = 50;
	double objTol;
	double stepTol;
	double c0;
	std::vector<double> objHistory;
	std::vector<double> stepHistory;

	ConvergenceChecker(double objTol_, double stepTol_, double c0_)
		: objTol(objTol_), stepTol(stepTol_), c0(c0_) {}

	bool operator()(double obj, double step);
	double estimatePrecondition(double cmax) const;
	double estimateNextStep(double tmax) const;
};

// ADC optimization options
struct TailorADCOptions {
	int maxIter{200};
	double convergeTol{1e-3};
	double maxStep{0.3};
	double stepTol{1e-3};
	double preconditionStrength{0.1};
	std::string objectiveType{"apac"};
	double mcfWeight{0.1}; // mean curvature flow (area regularization) weight

	bool enableRemesh{true};
	RemeshOptions remeshOpts;

	bool enableSurgery{true};
	int surgeryInterval{4};	 // check every N iterations
	int surgeryStartIter{0}; // iteration from which surgery is allowed
	SurgeryOptions surgeryOpts;

	int nfLimit{100000}; // face count limit (abort if exceeded), <= 0 means no limit

	std::string outputDir; // output intermediate results when non-empty
	int saveInterval{50};  // save every N iterations
};

// ADC shape optimization main loop
void tailorADC(PeriodicTriMesh& mesh, const TailorADCOptions& opts = TailorADCOptions{});

} // namespace xtpms
