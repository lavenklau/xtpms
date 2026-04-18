#include <CLI/CLI.hpp>
#include <OpenMesh/Core/IO/MeshIO.hh>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <random>

#include "PeriodicMesh.h"
#include "MarchingCubes.h"
#include "PeriodicRemesh.h"
#include "AsymptoticConductivity.h"

#include <CGAL/Simple_cartesian.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Polygon_mesh_processing/remesh.h>

extern "C" {
#include "tinyexpr.h"
}

// ──────────────────────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────────────────────

using Vec3d = xtpms::PeriodicTriMesh::Vec3d;

static Vec3d parseVec3(const std::string& s) {
	double x = 0, y = 0, z = 0;
	char sep;
	std::istringstream iss(s);
	iss >> x >> sep >> y >> sep >> z;
	return Vec3d(
		static_cast<xtpms::DefaultTriMesh::Scalar>(x),
		static_cast<xtpms::DefaultTriMesh::Scalar>(y),
		static_cast<xtpms::DefaultTriMesh::Scalar>(z));
}

// CGAL isotropic remesh on OpenMesh (via conversion)
static void cgalIsotropicRemesh(xtpms::DefaultTriMesh& omesh, double targetEdgeLength, int nIter = 3) {
	using K = CGAL::Simple_cartesian<double>;
	using CGALMesh = CGAL::Surface_mesh<K::Point_3>;
	namespace PMP = CGAL::Polygon_mesh_processing;

	// OpenMesh → CGAL
	CGALMesh cmesh;
	std::vector<CGALMesh::Vertex_index> vmap;
	vmap.reserve(omesh.n_vertices());
	for (auto v = omesh.vertices_begin(); v != omesh.vertices_end(); ++v) {
		auto p = omesh.point(*v);
		vmap.push_back(cmesh.add_vertex(K::Point_3(p[0], p[1], p[2])));
	}
	for (auto f = omesh.faces_begin(); f != omesh.faces_end(); ++f) {
		auto fv = omesh.cfv_iter(*f);
		int a = (*fv).idx(); ++fv; int b = (*fv).idx(); ++fv; int c = (*fv).idx();
		cmesh.add_face(vmap[a], vmap[b], vmap[c]);
	}

	// 固定边界顶点位置 + 保护边界边
	auto vcmap = cmesh.add_property_map<CGALMesh::Vertex_index, bool>("v:constrained", false).first;
	for (auto v : cmesh.vertices())
		if (CGAL::is_border(v, cmesh)) vcmap[v] = true;

	PMP::isotropic_remeshing(cmesh.faces(), targetEdgeLength, cmesh,
		CGAL::parameters::number_of_iterations(nIter)
		.protect_constraints(true)
		.vertex_is_constrained_map(vcmap));

	cmesh.collect_garbage();

	// CGAL → OpenMesh
	omesh.clear();
	for (auto v : cmesh.vertices()) {
		auto p = cmesh.point(v);
		omesh.add_vertex(xtpms::DefaultTriMesh::Point(
			static_cast<xtpms::DefaultTriMesh::Scalar>(p.x()),
			static_cast<xtpms::DefaultTriMesh::Scalar>(p.y()),
			static_cast<xtpms::DefaultTriMesh::Scalar>(p.z())));
	}
	for (auto f : cmesh.faces()) {
		std::vector<xtpms::DefaultTriMesh::VertexHandle> fverts;
		for (auto v : cmesh.vertices_around_face(cmesh.halfedge(f)))
			fverts.push_back(xtpms::DefaultTriMesh::VertexHandle(static_cast<int>(v)));
		if (fverts.size() == 3)
			omesh.add_face(fverts[0], fverts[1], fverts[2]);
	}
	std::cout << "CGAL isotropic remesh: nv=" << omesh.n_vertices()
			  << " nf=" << omesh.n_faces() << " targetLen=" << targetEdgeLength << "\n";
}

// Load mesh and periodize.
// hpStr: "" or "auto" → auto-detect from bbox
//        "x,y,z"      → scale mesh bbox to match specified half-period
static bool loadPeriodicMesh(xtpms::PeriodicTriMesh& mesh,
							 const std::string& inputFile, const std::string& hpStr) {
	xtpms::DefaultTriMesh src;
	if (!OpenMesh::IO::read_mesh(src, inputFile)) {
		std::cerr << "Error: cannot read " << inputFile << "\n";
		return false;
	}

	// 检测输入是否已经是闭合周期网格（bnd=0）
	{
		bool hasBnd = false;
		for (auto e = src.edges_begin(); e != src.edges_end(); ++e)
			if (src.is_boundary(*e)) { hasBnd = true; break; }
		if (!hasBnd) {
			// 已经闭合：直接读入，不做 remesh/merge/scale
			// 周期边界点已 merge，bbox 无法推断实际周期，必须指定 --half-period
			if (hpStr.empty() || hpStr == "auto") {
				std::cerr << "Error: closed periodic mesh requires --half-period\n";
				return false;
			}
			Vec3d hp = parseVec3(hpStr);
			std::cout << "Input is closed periodic mesh, skipping remesh/merge\n";
			std::cout << "Half-period: " << hp[0] << ", " << hp[1] << ", " << hp[2] << "\n";
			for (auto v = src.vertices_begin(); v != src.vertices_end(); ++v)
				mesh.add_vertex(src.point(*v));
			for (auto f = src.faces_begin(); f != src.faces_end(); ++f) {
				auto fv = src.cfv_iter(*f);
				int a = (*fv).idx(); ++fv; int b = (*fv).idx(); ++fv; int c = (*fv).idx();
				mesh.add_face(xtpms::PeriodicTriMesh::VertexHandle(a),
							  xtpms::PeriodicTriMesh::VertexHandle(b),
							  xtpms::PeriodicTriMesh::VertexHandle(c));
			}
			mesh.setHalfPeriod(hp);
			return true;
		}
	}

	// 以下处理有边界（需要 remesh + merge）的输入

	// Compute bbox
	Vec3d bmin, bmax;
	{
		auto v0 = src.vertices_begin();
		bmin = bmax = src.point(*v0);
		for (auto v = v0; v != src.vertices_end(); ++v) {
			const auto& p = src.point(*v);
			for (int i = 0; i < 3; ++i) {
				if (p[i] < bmin[i]) bmin[i] = p[i];
				if (p[i] > bmax[i]) bmax[i] = p[i];
			}
		}
	}

	// Step 1: Clamp boundary vertices to bbox
	// Use tight tolerance: only snap vertices that are very close to bbox
	// (saveUnitCell produces vertices at exact boundary positions)
	{
		double tol = 1e-4 * std::min({
			static_cast<double>(bmax[0] - bmin[0]),
			static_cast<double>(bmax[1] - bmin[1]),
			static_cast<double>(bmax[2] - bmin[2])});
		int clamped = 0;
		for (auto v = src.vertices_begin(); v != src.vertices_end(); ++v) {
			auto p = src.point(*v);
			bool changed = false;
			for (int i = 0; i < 3; ++i) {
				if (std::abs(static_cast<double>(p[i] - bmin[i])) < tol) {
					p[i] = bmin[i]; changed = true;
				} else if (std::abs(static_cast<double>(p[i] - bmax[i])) < tol) {
					p[i] = bmax[i]; changed = true;
				}
			}
			if (changed) { src.set_point(*v, p); ++clamped; }
		}
		if (clamped > 0) std::cout << "Clamped " << clamped << " boundary vertices to bbox\n";
	}

	// Step 2: Determine target half-period (scale deferred to after merge)
	Vec3d hp;
	bool specified = !hpStr.empty() && hpStr != "auto";
	if (specified) {
		hp = parseVec3(hpStr);
		std::cout << "Target half-period: " << hp[0] << ", " << hp[1] << ", " << hp[2] << "\n";
	} else {
		for (int i = 0; i < 3; ++i)
			hp[i] = (bmax[i] - bmin[i]) / static_cast<xtpms::DefaultTriMesh::Scalar>(2.0);
		std::cout << "Auto half-period: " << hp[0] << ", " << hp[1] << ", " << hp[2] << "\n";
	}
	// Shift src to [bmin, bmax] → [0, bbox_size] (original proportions)
	for (auto v = src.vertices_begin(); v != src.vertices_end(); ++v) {
		auto p = src.point(*v);
		for (int i = 0; i < 3; ++i) p[i] -= bmin[i];
		src.set_point(*v, p);
	}

	// Step 3: Build PeriodicTriMesh with bbox half-period, then merge
	{
		Vec3d bboxHp;
		for (int i = 0; i < 3; ++i)
			bboxHp[i] = (bmax[i] - bmin[i]) / static_cast<xtpms::DefaultTriMesh::Scalar>(2.0);
		mesh.setHalfPeriod(bboxHp);
		for (auto v = src.vertices_begin(); v != src.vertices_end(); ++v)
			mesh.add_vertex(src.point(*v));
		for (auto f = src.faces_begin(); f != src.faces_end(); ++f) {
			auto fv = src.cfv_iter(*f);
			int a = (*fv).idx(); ++fv; int b = (*fv).idx(); ++fv; int c = (*fv).idx();
			mesh.add_face(xtpms::PeriodicTriMesh::VertexHandle(a),
						  xtpms::PeriodicTriMesh::VertexHandle(b),
						  xtpms::PeriodicTriMesh::VertexHandle(c));
		}
		bool hasBoundary = false;
		for (auto e = mesh.edges_begin(); e != mesh.edges_end(); ++e)
			if (mesh.is_boundary(*e)) { hasBoundary = true; break; }
		if (hasBoundary) {
			mesh.mergePeriodBoundary();
		} else {
			std::cout << "Mesh already closed, skipping merge\n";
		}
	}

	// Step 6: Scale from [0, bbox_size] to [0, 2*hp]
	{
		Vec3d targetHp = specified ? hp : Vec3d(
			(bmax[0]-bmin[0])/static_cast<xtpms::DefaultTriMesh::Scalar>(2.0),
			(bmax[1]-bmin[1])/static_cast<xtpms::DefaultTriMesh::Scalar>(2.0),
			(bmax[2]-bmin[2])/static_cast<xtpms::DefaultTriMesh::Scalar>(2.0));
		if (specified) {
			for (auto v = mesh.vertices_begin(); v != mesh.vertices_end(); ++v) {
				auto p = mesh.point(*v);
				for (int i = 0; i < 3; ++i) {
					double bsize = static_cast<double>(bmax[i] - bmin[i]);
					double t = (bsize > 1e-15) ? static_cast<double>(p[i]) / bsize : 0.0;
					p[i] = static_cast<xtpms::DefaultTriMesh::Scalar>(t * 2.0 * static_cast<double>(hp[i]));
				}
				mesh.set_point(*v, p);
			}
		}
		mesh.setHalfPeriod(targetHp);
	}
	return true;
}

static void saveMesh(xtpms::PeriodicTriMesh& mesh,
					 const std::string& outputFile, bool noSplit = false) {
	mesh.saveUnitCell(outputFile, !noSplit);
}

// ──────────────────────────────────────────────────────────
// Custom objective via tinyexpr
// ──────────────────────────────────────────────────────────

struct ExprObjective {
	std::string expr;
	double k00=0, k01=0, k02=0, k10=0, k11=0, k12=0, k20=0, k21=0, k22=0;
	te_expr* compiled = nullptr;

	bool compile() {
		te_variable vars[] = {
			{"k00",&k00}, {"k01",&k01}, {"k02",&k02},
			{"k10",&k10}, {"k11",&k11}, {"k12",&k12},
			{"k20",&k20}, {"k21",&k21}, {"k22",&k22},
		};
		int err = 0;
		compiled = te_compile(expr.c_str(), vars, 9, &err);
		if (!compiled) {
			std::cerr << "Error: cannot parse expression '" << expr
					  << "' at position " << err << "\n";
			return false;
		}
		return true;
	}

	void setKA(const Eigen::Matrix3d& kA) {
		k00=kA(0,0); k01=kA(0,1); k02=kA(0,2);
		k10=kA(1,0); k11=kA(1,1); k12=kA(1,2);
		k20=kA(2,0); k21=kA(2,1); k22=kA(2,2);
	}

	double eval() { return te_eval(compiled); }

	// Numerical gradient w.r.t. Voigt kA components
	Eigen::Vector<double,6> gradient(const Eigen::Matrix3d& kA) {
		const double eps = 1e-7;
		Eigen::Vector<double,6> grad;
		for (int i = 0; i < 3; ++i) {
			Eigen::Matrix3d kp=kA, km=kA;
			kp(i,i)+=eps; km(i,i)-=eps;
			setKA(kp); double fp=eval(); setKA(km); double fm=eval();
			grad[i] = (fp-fm)/(2*eps);
		}
		int vm[][2] = {{1,2},{0,2},{0,1}};
		for (int vi = 0; vi < 3; ++vi) {
			int r=vm[vi][0], c=vm[vi][1];
			Eigen::Matrix3d kp=kA, km=kA;
			kp(r,c)+=eps; kp(c,r)+=eps; km(r,c)-=eps; km(c,r)-=eps;
			setKA(kp); double fp=eval(); setKA(km); double fm=eval();
			grad[3+vi] = (fp-fm)/(2*eps);
		}
		setKA(kA);
		return grad;
	}

	~ExprObjective() { if (compiled) te_free(compiled); }
};

// ──────────────────────────────────────────────────────────
// Subcommand: periodize
// ──────────────────────────────────────────────────────────

int cmdPeriodize(const std::string& input, const std::string& output,
				 const std::string& hpStr, bool noSplit) {
	xtpms::PeriodicTriMesh mesh;
	if (!loadPeriodicMesh(mesh, input, hpStr)) return 1;
	std::cout << "Periodized: nv=" << mesh.n_vertices()
			  << " nf=" << mesh.n_faces() << "\n";
	saveMesh(mesh, output, noSplit);
	std::cout << "Saved: " << output << "\n";
	return 0;
}

// ──────────────────────────────────────────────────────────
// Subcommand: compute
// ──────────────────────────────────────────────────────────

int cmdCompute(const std::string& input, const std::string& hpStr) {
	xtpms::PeriodicTriMesh mesh;
	if (!loadPeriodicMesh(mesh, input, hpStr)) return 1;
	auto geom = xtpms::computeVertexGeometry(mesh);
	Eigen::MatrixX3d u;
	Eigen::Matrix3d kA = xtpms::solveAsymptoticConductivity(mesh, geom, u);
	std::cout << "nv=" << mesh.n_vertices() << " nf=" << mesh.n_faces()
			  << " As=" << geom.vertexAreas.sum() << "\n";
	std::cout << "kA =\n" << kA << "\n";
	std::cout << "APAC = " << kA.trace() / 3.0 << "\n";
	Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(kA);
	std::cout << "eigenvalues: " << eig.eigenvalues().transpose() << "\n";
	return 0;
}

// ──────────────────────────────────────────────────────────
// Subcommand: optimize
// ──────────────────────────────────────────────────────────

int cmdOptimize(const std::string& input, const std::string& output,
				const std::string& hpStr, const std::string& objective,
				int maxIter, double maxStep, double mcfWeight,
				double precondStrength, bool enableSurgery,
				int surgeryStart, int surgeryInterval, double surgeryTol,
				bool noSplit, const std::string& outputDir) {
	xtpms::PeriodicTriMesh mesh;
	if (!loadPeriodicMesh(mesh, input, hpStr)) return 1;
	std::cout << "Input: nv=" << mesh.n_vertices() << " nf=" << mesh.n_faces() << "\n";

	bool isBuiltin = (objective == "apac" || objective == "k00" ||
					  objective == "k11" || objective == "k22" || objective == "iso");

	if (isBuiltin) {
		xtpms::TailorADCOptions opts;
		opts.objectiveType = objective;
		opts.maxIter = maxIter;
		opts.maxStep = maxStep;
		opts.mcfWeight = mcfWeight;
		opts.preconditionStrength = precondStrength;
		opts.enableRemesh = true;
		opts.remeshOpts.adaptiveEps = 0.6;
		opts.enableSurgery = enableSurgery;
		opts.surgeryStartIter = surgeryStart;
		opts.surgeryInterval = surgeryInterval;
		opts.surgeryOpts.singularityTol = surgeryTol;
		opts.outputDir = outputDir;
		if (!outputDir.empty()) opts.saveInterval = 1;
		xtpms::tailorADC(mesh, opts);
	} else {
		// Custom expression objective
		ExprObjective exprObj;
		exprObj.expr = objective;
		if (!exprObj.compile()) return 1;
		std::cout << "Custom objective: " << objective << "\n";

		xtpms::RemeshOptions remeshOpts;
		remeshOpts.adaptiveEps = 0.6;

		for (int iter = 0; iter < maxIter; ++iter) {
			if (iter > 0) {
				xtpms::delaunayRemesh(mesh, remeshOpts);
				bool hasBnd = false;
				for (auto e = mesh.edges_begin(); e != mesh.edges_end() && !hasBnd; ++e)
					if (mesh.is_boundary(*e)) hasBnd = true;
				if (hasBnd) mesh.mergePeriodBoundary();
			}

			auto geom = xtpms::computeVertexGeometry(mesh);
			int nv = static_cast<int>(mesh.n_vertices());
			Eigen::MatrixX3d ulist;
			Eigen::Matrix3d kA = xtpms::solveAsymptoticConductivity(mesh, geom, ulist);
			exprObj.setKA(kA);
			double objVal = exprObj.eval();

			auto sens = xtpms::computeSensitivity(mesh, geom, ulist);
			double As = geom.vertexAreas.sum();
			for (int i = 0; i < nv; ++i) {
				double ai = geom.vertexAreas[i];
				if (ai > 1e-15) { sens.vSens.row(i) /= ai; sens.aSens[i] /= ai; }
			}
			auto kAv = xtpms::toVoigt(kA); kAv.tail<3>() /= 2.0;
			auto grad6 = exprObj.gradient(kA);
			Eigen::VectorXd dfdvn = (sens.vSens / As - sens.aSens * kAv.transpose() / As) * (-grad6);

			auto L = xtpms::assembleLaplacian(mesh, geom.cotWeights);
			Eigen::SparseMatrix<double> G = -precondStrength * L;
			for (int i = 0; i < nv; ++i) G.coeffRef(i,i) += geom.vertexAreas[i];
			Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver(G);
			Eigen::VectorXd dn = -solver.solve(
				geom.vertexAreas.cwiseProduct((precondStrength+1.0) * dfdvn).eval());

			for (auto v = mesh.vertices_begin(); v != mesh.vertices_end(); ++v) {
				int vi = (*v).idx();
				Eigen::Vector3d disp = dn[vi] * Eigen::Vector3d(
					geom.vertexNormals[vi][0], geom.vertexNormals[vi][1], geom.vertexNormals[vi][2]);
				if (mcfWeight > 0)
					disp += mcfWeight * Eigen::Vector3d(
						geom.vrings[vi].Lx[0], geom.vrings[vi].Lx[1], geom.vrings[vi].Lx[2]);
				auto p = mesh.point(*v);
				mesh.set_point(*v, Vec3d(
					p[0]+static_cast<xtpms::DefaultTriMesh::Scalar>(maxStep*disp[0]),
					p[1]+static_cast<xtpms::DefaultTriMesh::Scalar>(maxStep*disp[1]),
					p[2]+static_cast<xtpms::DefaultTriMesh::Scalar>(maxStep*disp[2])));
			}

			std::cout << "iter=" << iter << " obj=" << objVal
					  << " nv=" << nv << " nf=" << mesh.n_faces() << "\n";

			if (!outputDir.empty())
				mesh.saveUnitCell(outputDir + "/iter_" + std::to_string(iter) + ".obj");
		}
	}

	auto geom = xtpms::computeVertexGeometry(mesh);
	Eigen::MatrixX3d u;
	Eigen::Matrix3d kA = xtpms::solveAsymptoticConductivity(mesh, geom, u);
	std::cout << "\nFinal kA =\n" << kA << "\n";
	std::cout << "APAC = " << kA.trace() / 3.0 << "\n";
	saveMesh(mesh, output, noSplit);
	std::cout << "Saved: " << output << "\n";
	return 0;
}

// ──────────────────────────────────────────────────────────
// Marching cubes helper
// ──────────────────────────────────────────────────────────

static xtpms::DefaultTriMesh marchingCubesFromLevelSet(
	const std::function<double(double,double,double)>& levelSet,
	const Vec3d& hp, int resolution) {
	double Lx=2.0*hp[0], Ly=2.0*hp[1], Lz=2.0*hp[2];
	int nx=resolution, ny=resolution, nz=resolution;
	double dx=Lx/nx, dy=Ly/ny, dz=Lz/nz;
	// 周期 MC：nx^3 个基础节点 + 边界 phantom 节点
	// phantom 节点的函数值 = 对面基础节点的值，坐标 = 实际位置
	// 这保证边界 cell 和对面 cell 使用相同函数值 → MC 三角化完全匹配
	int Np = nx; // 基础节点数
	auto baseIdx = [Np](int i, int j, int k) {
		return ((i%Np+Np)%Np) + Np * (((j%Np+Np)%Np) + Np * ((k%Np+Np)%Np));
	};
	// 基础节点
	std::vector<xtpms::LevelSetNode> nodes(static_cast<std::size_t>(Np*Np*Np));
	for (int k=0; k<nz; ++k)
		for (int j=0; j<ny; ++j)
			for (int i=0; i<nx; ++i)
				nodes[static_cast<std::size_t>(baseIdx(i,j,k))] =
					{i*dx, j*dy, k*dz, levelSet(i*dx, j*dy, k*dz)};
	// 为边界 cell 创建 phantom 节点（坐标在 [L, L+dx]，函数值 = 对面）
	std::map<std::tuple<int,int,int>, int> phantomMap;
	auto getNode = [&](int i, int j, int k) -> int {
		bool phantom = (i >= nx || j >= ny || k >= nz);
		if (!phantom) return baseIdx(i, j, k);
		auto key = std::make_tuple(i, j, k);
		if (phantomMap.count(key)) return phantomMap[key];
		int idx = static_cast<int>(nodes.size());
		double val = nodes[static_cast<std::size_t>(baseIdx(i,j,k))].value;
		nodes.push_back({i*dx, j*dy, k*dz, val});
		phantomMap[key] = idx;
		return idx;
	};
	xtpms::SparseVoxelCornerMap voxels;
	for (int k=0; k<nz; ++k)
		for (int j=0; j<ny; ++j)
			for (int i=0; i<nx; ++i) {
				xtpms::VoxelCornerNodeIndices c{};
				c[0]=getNode(i,j,k);       c[1]=getNode(i+1,j,k);
				c[2]=getNode(i+1,j+1,k);   c[3]=getNode(i,j+1,k);
				c[4]=getNode(i,j,k+1);     c[5]=getNode(i+1,j,k+1);
				c[6]=getNode(i+1,j+1,k+1); c[7]=getNode(i,j+1,k+1);
				voxels[{i,j,k}] = c;
			}
	xtpms::DefaultTriMesh src;
	xtpms::ExtractMarchingCubesOptions mcOpts;
	mcOpts.weldSharedEdges = true;
	xtpms::marchingCubesExtractToTriMesh(voxels, nodes, 0.0, src, mcOpts);
	return src;
}

static xtpms::PeriodicTriMesh periodizeMesh(const xtpms::DefaultTriMesh& src, const Vec3d& hp) {
	// MC 输出的周期边界顶点完全匹配（平移一个周期后重合）
	// 直接 wrap 到 [0, L) + 哈希去重即可，不需要投影/split
	const double L[3] = {
		2.0 * static_cast<double>(hp[0]),
		2.0 * static_cast<double>(hp[1]),
		2.0 * static_cast<double>(hp[2])
	};

	const std::size_t nv = src.n_vertices();
	// wrap 顶点到 [0, L)
	std::vector<std::array<double, 3>> wpos(nv);
	for (auto v = src.vertices_begin(); v != src.vertices_end(); ++v) {
		auto p = src.point(*v);
		auto& w = wpos[static_cast<std::size_t>((*v).idx())];
		for (int i = 0; i < 3; ++i) {
			double pi = static_cast<double>(p[i]);
			while (pi < -1e-8) pi += L[i];
			while (pi >= L[i] - 1e-8) pi -= L[i];
			if (pi < 0) pi = 0;
			w[static_cast<std::size_t>(i)] = pi;
		}
	}

	// 哈希去重：量化坐标到 grid，合并相同 cell 的顶点
	const double res = 1e-5; // 量化分辨率
	struct GridKey {
		int64_t x, y, z;
		bool operator==(const GridKey& o) const { return x==o.x && y==o.y && z==o.z; }
	};
	struct GridHash {
		std::size_t operator()(const GridKey& k) const {
			return std::hash<int64_t>()(k.x * 1000003LL + k.y) * 1000003LL + k.z;
		}
	};
	auto quantize = [res](double v) -> int64_t { return static_cast<int64_t>(std::round(v / res)); };

	std::unordered_map<GridKey, int, GridHash> grid;
	std::vector<std::array<double, 3>> uniqueVerts;
	std::vector<int> vmap(nv);

	for (std::size_t i = 0; i < nv; ++i) {
		GridKey key{quantize(wpos[i][0]), quantize(wpos[i][1]), quantize(wpos[i][2])};
		auto it = grid.find(key);
		if (it != grid.end()) {
			vmap[i] = it->second;
		} else {
			int idx = static_cast<int>(uniqueVerts.size());
			grid[key] = idx;
			uniqueVerts.push_back(wpos[i]);
			vmap[i] = idx;
		}
	}

	// 构建 PeriodicTriMesh
	xtpms::PeriodicTriMesh mesh;
	mesh.setHalfPeriod(hp);
	std::vector<xtpms::PeriodicTriMesh::VertexHandle> vh(uniqueVerts.size());
	for (std::size_t i = 0; i < uniqueVerts.size(); ++i) {
		vh[i] = mesh.add_vertex(Vec3d(
			static_cast<xtpms::DefaultTriMesh::Scalar>(uniqueVerts[i][0]),
			static_cast<xtpms::DefaultTriMesh::Scalar>(uniqueVerts[i][1]),
			static_cast<xtpms::DefaultTriMesh::Scalar>(uniqueVerts[i][2])));
	}
	// 去重复面：weld 后不同原始面可能映射到相同三顶点
	std::set<std::array<int,3>> faceSet;
	for (auto f = src.faces_begin(); f != src.faces_end(); ++f) {
		auto fv = src.cfv_iter(*f);
		int a = vmap[static_cast<std::size_t>((*fv).idx())]; ++fv;
		int b = vmap[static_cast<std::size_t>((*fv).idx())]; ++fv;
		int c = vmap[static_cast<std::size_t>((*fv).idx())];
		if (a == b || b == c || a == c) continue;
		// 规范化面（最小顶点在前）用于去重
		std::array<int,3> key = {a, b, c};
		std::sort(key.begin(), key.end());
		if (!faceSet.insert(key).second) continue; // 重复面
		auto fh = mesh.add_face(vh[static_cast<std::size_t>(a)],
								vh[static_cast<std::size_t>(b)],
								vh[static_cast<std::size_t>(c)]);
		if (!fh.is_valid())
			mesh.add_face(vh[static_cast<std::size_t>(a)],
						  vh[static_cast<std::size_t>(c)],
						  vh[static_cast<std::size_t>(b)]);
	}

	int bnd = 0;
	for (auto e = mesh.edges_begin(); e != mesh.edges_end(); ++e)
		if (mesh.is_boundary(*e)) ++bnd;
	std::cerr << "[merge] after weld: nv=" << mesh.n_vertices()
			  << " nf=" << mesh.n_faces() << " bnd=" << bnd << "\n";
	return mesh;
}

// Built-in level set functions
static std::function<double(double,double,double)> getBuiltinLevelSet(
	const std::string& type, double Lx, double Ly, double Lz) {
	if (type == "gyroid") {
		return [Lx,Ly,Lz](double x, double y, double z) {
			double px=2*M_PI*x/Lx, py=2*M_PI*y/Ly, pz=2*M_PI*z/Lz;
			return sin(px)*cos(py) + sin(py)*cos(pz) + sin(pz)*cos(px);
		};
	} else if (type == "schwarzp" || type == "schwarz-p") {
		return [Lx,Ly,Lz](double x, double y, double z) {
			return cos(2*M_PI*x/Lx) + cos(2*M_PI*y/Ly) + cos(2*M_PI*z/Lz);
		};
	} else if (type == "diamond" || type == "schwarzd" || type == "schwarz-d") {
		return [Lx,Ly,Lz](double x, double y, double z) {
			double px=2*M_PI*x/Lx, py=2*M_PI*y/Ly, pz=2*M_PI*z/Lz;
			return sin(px)*sin(py)*sin(pz) + sin(px)*cos(py)*cos(pz)
				 + cos(px)*sin(py)*cos(pz) + cos(px)*cos(py)*sin(pz);
		};
	}
	return nullptr;
}

// ──────────────────────────────────────────────────────────
// Subcommand: sample (extract isosurface from level set)
// ──────────────────────────────────────────────────────────

int cmdSample(const std::string& expression, const std::string& output,
			  const std::string& hpStr, int resolution, bool noSplit, bool randomMode,
			  int kmax, double decay) {
	Vec3d hp = parseVec3(hpStr);
	double Lx=2.0*hp[0], Ly=2.0*hp[1], Lz=2.0*hp[2];

	std::function<double(double,double,double)> levelSet;

	if (randomMode) {
		// Random triperiodic Fourier series:
		// f(x,y,z) = Σ_k [ c_k cos(k·ω) + s_k sin(k·ω) ]
		// where ω = (2πx/Lx, 2πy/Ly, 2πz/Lz), k = (k1,k2,k3), |ki| ≤ kmax
		// Coefficients decay as 1/|k|^decay to ensure smoothness
		// kmax and decay are now parameters

		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_real_distribution<double> dist(-1.0, 1.0);

		// Enumerate all frequency vectors and generate coefficients
		struct FourierTerm { int k[3]; double c, s; };
		std::vector<FourierTerm> terms;
		for (int k0 = -kmax; k0 <= kmax; ++k0)
			for (int k1 = -kmax; k1 <= kmax; ++k1)
				for (int k2 = -kmax; k2 <= kmax; ++k2) {
					if (k0 == 0 && k1 == 0 && k2 == 0) continue; // skip DC
					double knorm = std::sqrt(k0*k0 + k1*k1 + k2*k2);
					double weight = 1.0 / std::pow(knorm, decay);
					terms.push_back({{k0,k1,k2}, dist(gen)*weight, dist(gen)*weight});
				}

		std::cout << "Random Fourier series: " << terms.size() << " terms, kmax=" << kmax
				  << ", decay=" << decay << "\n";

		levelSet = [=](double x, double y, double z) {
			double w[3] = {2*M_PI*x/Lx, 2*M_PI*y/Ly, 2*M_PI*z/Lz};
			double val = 0;
			for (const auto& t : terms) {
				double phase = t.k[0]*w[0] + t.k[1]*w[1] + t.k[2]*w[2];
				val += t.c * cos(phase) + t.s * sin(phase);
			}
			return val;
		};
	} else {
		// Try built-in name first
		levelSet = getBuiltinLevelSet(expression, Lx, Ly, Lz);
		if (!levelSet) {
			// Parse as tinyexpr expression with variables x, y, z
			// Expression assumes 2π-periodic: e.g. cos(x)+cos(y)+cos(z)
			// Auto-scale: x_expr = 2π * x_physical / period
			double vx=0, vy=0, vz=0;
			double pi_val = M_PI;
			te_variable vars[] = {
				{"x", &vx}, {"y", &vy}, {"z", &vz}, {"pi", &pi_val},
			};
			int err = 0;
			te_expr* compiled = te_compile(expression.c_str(), vars, 4, &err);
			if (!compiled) {
				std::cerr << "Error: cannot parse expression '" << expression
						  << "' at position " << err << "\n";
				return 1;
			}
			levelSet = [compiled, &vx, &vy, &vz, Lx, Ly, Lz](double x, double y, double z) mutable {
				// Scale physical [0, L] → expression [0, 2π]
				vx = 2*M_PI*x/Lx; vy = 2*M_PI*y/Ly; vz = 2*M_PI*z/Lz;
				return te_eval(compiled);
			};
			std::cout << "Expression: " << expression
					  << " (2pi-periodic, scaled to period " << Lx << "x" << Ly << "x" << Lz << ")\n";
		}
	}

	auto src = marchingCubesFromLevelSet(levelSet, hp, resolution);
	std::cout << "Isosurface: nv=" << src.n_vertices() << " nf=" << src.n_faces() << "\n";

	if (src.n_faces() == 0) {
		std::cerr << "Warning: no isosurface found (try different expression or resolution)\n";
		return 1;
	}

	auto mesh = periodizeMesh(src, hp);
	std::cout << "Periodized: nv=" << mesh.n_vertices() << " nf=" << mesh.n_faces() << "\n";
	saveMesh(mesh, output, noSplit);
	std::cout << "Saved: " << output << "\n";
	return 0;
}

// ──────────────────────────────────────────────────────────
// Subcommand: generate (seed mesh → optimize → TPMS)
// ──────────────────────────────────────────────────────────

int cmdGenerate(const std::string& input, const std::string& output,
				const std::string& hpStr, int maxIter, bool noSplit,
				const std::string& outputDir = "") {
	xtpms::PeriodicTriMesh mesh;
	if (!loadPeriodicMesh(mesh, input, hpStr)) return 1;
	std::cout << "Seed: nv=" << mesh.n_vertices() << " nf=" << mesh.n_faces() << "\n";

	// Pre-smoothing: MCF + remesh to improve mesh quality before optimization
	std::cout << "Pre-smoothing...\n";
	for (int pre = 0; pre < 5; ++pre) {
		auto geomPre = xtpms::computeVertexGeometry(mesh);
		double maxLx = 0;
		for (auto v = mesh.vertices_begin(); v != mesh.vertices_end(); ++v) {
			int vi = (*v).idx();
			auto& lx = geomPre.vrings[static_cast<std::size_t>(vi)].Lx;
			maxLx = std::max(maxLx, std::sqrt(lx[0]*lx[0]+lx[1]*lx[1]+lx[2]*lx[2]));
			auto p = mesh.point(*v);
			double step = 0.05;
			mesh.set_point(*v, Vec3d(
				p[0] + static_cast<xtpms::DefaultTriMesh::Scalar>(step * lx[0]),
				p[1] + static_cast<xtpms::DefaultTriMesh::Scalar>(step * lx[1]),
				p[2] + static_cast<xtpms::DefaultTriMesh::Scalar>(step * lx[2])));
		}
		auto ropts = xtpms::defaultRemeshOptions(mesh);
		xtpms::delaunayRemesh(mesh, ropts);
		bool hasBnd = false;
		for (auto e = mesh.edges_begin(); e != mesh.edges_end() && !hasBnd; ++e)
			if (mesh.is_boundary(*e)) hasBnd = true;
		if (hasBnd) mesh.mergePeriodBoundary();
		std::cout << "  pre " << pre << ": nv=" << mesh.n_vertices()
				  << " nf=" << mesh.n_faces() << " maxLx=" << maxLx << "\n";
	}

	// Optimize toward TPMS (maximize APAC)
	xtpms::TailorADCOptions opts;
	opts.objectiveType = "apac";
	opts.maxIter = maxIter;
	opts.maxStep = 1.0;
	opts.mcfWeight = 0.1;
	opts.enableRemesh = true;
	opts.remeshOpts.adaptiveEps = 0.6;
	opts.enableSurgery = true;
	opts.surgeryStartIter = std::max(maxIter / 3, 20);
	opts.surgeryInterval = 10;
	// 对齐 minsurf 主流: surgery-tol=25 surgery-type=2
	opts.surgeryOpts.singularityTol = 25.0;
	opts.surgeryOpts.singularityRatio = 0.0;
	opts.surgeryOpts.surgeryType = 2;
	opts.outputDir = outputDir;
	if (!outputDir.empty()) opts.saveInterval = 1;

	xtpms::tailorADC(mesh, opts);

	auto geom = xtpms::computeVertexGeometry(mesh);
	Eigen::MatrixX3d u;
	Eigen::Matrix3d kA = xtpms::solveAsymptoticConductivity(mesh, geom, u);
	std::cout << "Final APAC = " << kA.trace() / 3.0 << "\n";
	std::cout << "kA =\n" << kA << "\n";

	saveMesh(mesh, output, noSplit);
	std::cout << "Saved: " << output << "\n";
	return 0;
}

// ──────────────────────────────────────────────────────────
// Main
// ──────────────────────────────────────────────────────────

int main(int argc, char** argv) {
	CLI::App app{"xtpms - Computational design of triply periodic minimal surfaces"};
	app.require_subcommand(1);

	std::string hpStr;  // empty = auto-detect from bbox

	// ── periodize ──
	auto* cmdP = app.add_subcommand("periodize", "Merge periodic boundary of a mesh");
	std::string pz_in, pz_out; bool pz_nosplit = false;
	cmdP->add_option("-i,--input", pz_in, "Input OBJ")->required();
	cmdP->add_option("-o,--output", pz_out, "Output OBJ")->required();
	cmdP->add_option("--half-period", hpStr, "Half-period x,y,z");
	cmdP->add_flag("--no-split", pz_nosplit, "Skip split (use raw saveUnitCell instead)");

	// ── compute ──
	auto* cmdC = app.add_subcommand("compute", "Compute effective conductivity tensor");
	std::string cp_in;
	cmdC->add_option("-i,--input", cp_in, "Input OBJ")->required();
	cmdC->add_option("--half-period", hpStr, "Half-period x,y,z");

	// ── optimize ──
	auto* cmdO = app.add_subcommand("optimize", "Optimize surface conductivity");
	std::string o_in, o_out, o_obj="apac", o_dir;
	int o_iter=100; double o_step=1, o_mcf=0.1, o_prec=0.1;
	bool o_surg=false, o_nosplit=false;
	int o_surgStart=40, o_surgInt=20; double o_surgTol=50;
	cmdO->add_option("-i,--input", o_in, "Input OBJ")->required();
	cmdO->add_option("-o,--output", o_out, "Output OBJ")->required();
	cmdO->add_option("--half-period", hpStr);
	cmdO->add_option("--objective", o_obj,
		"apac|k00|k11|k22|iso or expression e.g. '(k00+k11+k22)/3'")->default_val("apac");
	cmdO->add_option("--max-iter", o_iter)->default_val(100);
	cmdO->add_option("--max-step", o_step)->default_val(1.0);
	cmdO->add_option("--mcf-weight", o_mcf)->default_val(0.1);
	cmdO->add_option("--precondition", o_prec)->default_val(0.1);
	cmdO->add_flag("--surgery", o_surg, "Enable singularity surgery");
	cmdO->add_option("--surgery-start", o_surgStart)->default_val(40);
	cmdO->add_option("--surgery-interval", o_surgInt)->default_val(20);
	cmdO->add_option("--surgery-tol", o_surgTol)->default_val(50);
	cmdO->add_flag("--no-split", o_nosplit);
	cmdO->add_option("--output-dir", o_dir, "Save intermediate meshes");

	// ── sample ──
	auto* cmdS = app.add_subcommand("sample", "Sample isosurface from level set expression");
	std::string s_expr, s_out, s_hpStr = "1,1,1";
	int s_res=20; bool s_nosplit=false, s_random=false;
	int s_kmax=2; double s_decay=2.0;
	cmdS->add_option("-e,--expression", s_expr,
		"Level set expression (x,y,z,pi) or built-in: gyroid|schwarzp|diamond");
	cmdS->add_option("-o,--output", s_out, "Output OBJ")->required();
	cmdS->add_option("--half-period", s_hpStr, "Half-period x,y,z")->default_val("1,1,1");
	cmdS->add_option("-r,--resolution", s_res)->default_val(20);
	cmdS->add_flag("--random", s_random, "Random triperiodic function (ignore -e)");
	cmdS->add_option("--kmax", s_kmax, "Max frequency index for --random")->default_val(2);
	cmdS->add_option("--decay", s_decay, "Coefficient decay exponent for --random")->default_val(2.0);
	cmdS->add_flag("--no-split", s_nosplit);

	// ── generate ──
	auto* cmdG = app.add_subcommand("generate", "Generate TPMS from seed mesh (auto period from bbox)");
	std::string g_in, g_out;
	int g_iter=100; bool g_nosplit=false;
	cmdG->add_option("-i,--input", g_in, "Seed mesh OBJ")->required();
	std::string g_hpStr;
	cmdG->add_option("-o,--output", g_out, "Output OBJ")->required();
	cmdG->add_option("--half-period", g_hpStr, "Target half-period (scale bbox to match)");
	std::string g_dir;
	cmdG->add_option("--max-iter", g_iter)->default_val(100);
	cmdG->add_option("--output-dir", g_dir, "Save intermediate meshes");
	cmdG->add_flag("--no-split", g_nosplit);

	CLI11_PARSE(app, argc, argv);

	if (cmdP->parsed()) return cmdPeriodize(pz_in, pz_out, hpStr, pz_nosplit);
	if (cmdC->parsed()) return cmdCompute(cp_in, hpStr);
	if (cmdO->parsed()) return cmdOptimize(o_in, o_out, hpStr, o_obj, o_iter, o_step,
										   o_mcf, o_prec, o_surg, o_surgStart, o_surgInt,
										   o_surgTol, o_nosplit, o_dir);
	if (cmdS->parsed()) return cmdSample(s_expr, s_out, s_hpStr, s_res, s_nosplit, s_random, s_kmax, s_decay);
	if (cmdG->parsed()) return cmdGenerate(g_in, g_out, g_hpStr, g_iter, g_nosplit, g_dir);
	return 0;
}
