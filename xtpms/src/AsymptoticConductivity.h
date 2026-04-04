#pragma once

#include "PeriodicMesh.h"
#include "PeriodicRemesh.h"

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <string>
#include <vector>

namespace xtpms {

// 顶点 1-ring 编译后的几何量
struct Compile1ring {
	double As = 0;              // Voronoi area
	Eigen::Vector3d nv{0, 0, 0}; // angle-weighted vertex normal
	Eigen::Vector3d Lx{0, 0, 0}; // mean curvature vector
	double H = 0;               // mean curvature (scalar)
	double K = 0;               // Gaussian curvature

	Compile1ring() = default;
	Compile1ring(const Eigen::Vector3d& o, const std::vector<Eigen::Vector3d>& ring);
};

// 面局部坐标系 [e1, e2, n]（列向量），其中 n = (v1-v0)×(v2-v0)/2
Eigen::Matrix3d faceFrame(const Eigen::Matrix3d& tri);

// 第二基本形式（边形式）：be_ij = -(n_j - n_i) · (v_j - v_i)
Eigen::Vector3d secondFundamentalFormEdge(
	const Eigen::Matrix3d& tri,
	const Compile1ring& v0, const Compile1ring& v1, const Compile1ring& v2);

// 边拉伸 → 应变矩阵 Bn：{eps11, eps22, 2*eps12} = Bn * {be12, be23, be31}
Eigen::Matrix3d strainMatrixEdgeStretch(
	const Eigen::Matrix3d& tri,
	const Eigen::Vector3d& e1, const Eigen::Vector3d& e2);

// 面积形状导数（对法向位移的导数），返回 3 个顶点各自的贡献
Eigen::Vector3d areaShapeDerivative(
	const Eigen::Matrix3d& tri, const Eigen::Matrix3d& fr,
	const Compile1ring v[3]);

// Voigt 记法
Eigen::Vector<double, 6> toVoigt(const Eigen::Matrix3d& M);
Eigen::Matrix3d fromVoigt6(const Eigen::Vector<double, 6>& v);
Eigen::Vector3d toVoigt2(const Eigen::Matrix2d& M);
Eigen::Matrix2d fromVoigt3(const Eigen::Vector3d& v);

// 2x3 标量梯度矩阵（局部坐标系下）
Eigen::Matrix<double, 2, 3> scalarGradientMatrix(
	const Eigen::Matrix3d& tri, const Eigen::Matrix3d& fr);

// 离散微分几何数据
struct VertexGeometry {
	std::vector<double> cotWeights;             // per-edge cotangent weight
	std::vector<Eigen::Vector3d> edgeVectors;   // per-edge period-wrapped edge vector
	Eigen::VectorXd vertexAreas;                // per-vertex dual area
	std::vector<Eigen::Vector3d> vertexNormals; // per-vertex angle-weighted normal
	std::vector<Compile1ring> vrings;           // per-vertex compiled 1-ring
};

VertexGeometry computeVertexGeometry(const PeriodicTriMesh& mesh);

// 组装余切 Laplacian
Eigen::SparseMatrix<double> assembleLaplacian(
	const PeriodicTriMesh& mesh,
	const std::vector<double>& cotWeights);

// 组装 RHS
Eigen::MatrixX3d assembleRHS(
	const PeriodicTriMesh& mesh,
	const std::vector<double>& cotWeights,
	const std::vector<Eigen::Vector3d>& edgeVectors);

// 求解 ADC 张量
Eigen::Matrix3d solveAsymptoticConductivity(
	PeriodicTriMesh& mesh,
	const VertexGeometry& geom,
	Eigen::MatrixX3d& outU);  // 输出解向量

// 评估 ADC 张量
Eigen::Matrix3d evaluateConductivityTensor(
	const PeriodicTriMesh& mesh,
	const Eigen::MatrixX3d& blist,
	const Eigen::MatrixX3d& ulist,
	double totalArea);

// 形状敏感度：返回 (n_vertices x 6) Voigt 敏感度和面积敏感度
struct SensitivityResult {
	Eigen::MatrixXd vSens;   // n_vertices x 6 (Voigt)
	Eigen::VectorXd aSens;   // n_vertices
};

SensitivityResult computeSensitivity(
	const PeriodicTriMesh& mesh,
	const VertexGeometry& geom,
	const Eigen::MatrixX3d& ulist);

// ADC 目标函数
struct ADCObjective {
	double value;
	Eigen::Vector<double, 6> gradient; // 对 kA Voigt 分量的导数
};

ADCObjective evaluateADCObjective(const std::string& type, const Eigen::Matrix3d& kA);

// 收敛检查
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

// ADC 优化选项
struct TailorADCOptions {
	int maxIter{200};
	double convergeTol{1e-3};
	double maxStep{0.3};
	double stepTol{1e-3};
	double preconditionStrength{0.1};
	std::string objectiveType{"apac"};
	double mcfWeight{0.1};           // 均曲率流（area regularization）权重

	bool enableRemesh{true};
	RemeshOptions remeshOpts;

	bool enableSurgery{true};
	int surgeryInterval{4};   // 每 N 步检查一次
	int surgeryStartIter{0};  // 从第几步开始允许 surgery
	PeriodicTriMesh::SurgeryOptions surgeryOpts;

	std::string outputDir;    // 非空时输出中间结果
	int saveInterval{50};     // 每 N 步保存一次
};

// ADC 形状优化主循环
void tailorADC(PeriodicTriMesh& mesh, const TailorADCOptions& opts = {});

} // namespace xtpms
