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

static bool loadPeriodicMesh(xtpms::PeriodicTriMesh& mesh,
							 const std::string& inputFile, const Vec3d& hp) {
	xtpms::DefaultTriMesh src;
	if (!OpenMesh::IO::read_mesh(src, inputFile)) {
		std::cerr << "Error: cannot read " << inputFile << "\n";
		return false;
	}
	mesh.setHalfPeriod(hp);
	for (auto v = src.vertices_begin(); v != src.vertices_end(); ++v)
		mesh.add_vertex(src.point(*v));
	for (auto f = src.faces_begin(); f != src.faces_end(); ++f) {
		auto fv = src.cfv_iter(*f);
		int a = (*fv).idx(); ++fv; int b = (*fv).idx(); ++fv; int c = (*fv).idx();
		mesh.add_face(xtpms::PeriodicTriMesh::VertexHandle(a),
					  xtpms::PeriodicTriMesh::VertexHandle(b),
					  xtpms::PeriodicTriMesh::VertexHandle(c));
	}
	mesh.mergePeriodBoundary();
	return true;
}

static void saveMesh(xtpms::PeriodicTriMesh& mesh,
					 const std::string& outputFile, bool split) {
	if (split) {
		mesh.splitUnitCell();
		OpenMesh::IO::write_mesh(mesh, outputFile);
	} else {
		mesh.saveUnitCell(outputFile);
	}
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
				 const std::string& hpStr, bool split) {
	xtpms::PeriodicTriMesh mesh;
	if (!loadPeriodicMesh(mesh, input, parseVec3(hpStr))) return 1;
	std::cout << "Periodized: nv=" << mesh.n_vertices()
			  << " nf=" << mesh.n_faces() << "\n";
	saveMesh(mesh, output, split);
	std::cout << "Saved: " << output << "\n";
	return 0;
}

// ──────────────────────────────────────────────────────────
// Subcommand: compute
// ──────────────────────────────────────────────────────────

int cmdCompute(const std::string& input, const std::string& hpStr) {
	xtpms::PeriodicTriMesh mesh;
	if (!loadPeriodicMesh(mesh, input, parseVec3(hpStr))) return 1;
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
				bool split, const std::string& outputDir) {
	xtpms::PeriodicTriMesh mesh;
	if (!loadPeriodicMesh(mesh, input, parseVec3(hpStr))) return 1;
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
		opts.remeshOpts.adaptiveEps = 1.0;
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
		remeshOpts.adaptiveEps = 1.0;

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
	saveMesh(mesh, output, split);
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
	int nx=resolution, ny=resolution, nz=resolution, Np=resolution+1;
	double dx=Lx/nx, dy=Ly/ny, dz=Lz/nz;
	std::vector<xtpms::LevelSetNode> nodes(Np*Np*Np);
	for (int k=0; k<=nz; ++k)
		for (int j=0; j<=ny; ++j)
			for (int i=0; i<=nx; ++i)
				nodes[i+Np*(j+Np*k)] = {i*dx, j*dy, k*dz, levelSet(i*dx, j*dy, k*dz)};
	xtpms::SparseVoxelCornerMap voxels;
	for (int k=0; k<nz; ++k)
		for (int j=0; j<ny; ++j)
			for (int i=0; i<nx; ++i) {
				xtpms::VoxelCornerNodeIndices c{};
				c[0]=i+Np*(j+Np*k);       c[1]=(i+1)+Np*(j+Np*k);
				c[2]=(i+1)+Np*((j+1)+Np*k); c[3]=i+Np*((j+1)+Np*k);
				c[4]=i+Np*(j+Np*(k+1));    c[5]=(i+1)+Np*(j+Np*(k+1));
				c[6]=(i+1)+Np*((j+1)+Np*(k+1)); c[7]=i+Np*((j+1)+Np*(k+1));
				voxels[{i,j,k}] = c;
			}
	xtpms::DefaultTriMesh src;
	xtpms::ExtractMarchingCubesOptions mcOpts;
	mcOpts.weldSharedEdges = true;
	xtpms::marchingCubesExtractToTriMesh(voxels, nodes, 0.0, src, mcOpts);
	return src;
}

static xtpms::PeriodicTriMesh periodizeMesh(const xtpms::DefaultTriMesh& src, const Vec3d& hp) {
	xtpms::PeriodicTriMesh mesh;
	mesh.setHalfPeriod(hp);
	for (auto v = src.vertices_begin(); v != src.vertices_end(); ++v)
		mesh.add_vertex(src.point(*v));
	for (auto f = src.faces_begin(); f != src.faces_end(); ++f) {
		auto fv = src.cfv_iter(*f);
		int a=(*fv).idx(); ++fv; int b=(*fv).idx(); ++fv; int c=(*fv).idx();
		mesh.add_face(xtpms::PeriodicTriMesh::VertexHandle(a),
					  xtpms::PeriodicTriMesh::VertexHandle(b),
					  xtpms::PeriodicTriMesh::VertexHandle(c));
	}
	mesh.mergePeriodBoundary();
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
			  const std::string& hpStr, int resolution, bool split, bool randomMode) {
	Vec3d hp = parseVec3(hpStr);
	double Lx=2.0*hp[0], Ly=2.0*hp[1], Lz=2.0*hp[2];

	std::function<double(double,double,double)> levelSet;

	if (randomMode) {
		// Random triperiodic function:
		// f = Σ c_cos[i]*cos(2π x_i/L_i) + c_sin[i]*sin(2π x_i/L_i)
		//   + Σ c_cc[i,j]*cos*cos + c_sc[i,j]*sin*cos + c_ss[i,j]*sin*sin
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_real_distribution<double> dist(-1.0, 1.0);

		double c_cos[3], c_sin[3], c_cc[9], c_sc[9], c_ss[9];
		for (int i = 0; i < 3; ++i) {
			c_cos[i] = dist(gen); c_sin[i] = dist(gen);
			for (int j = 0; j < 3; ++j) {
				c_cc[i*3+j] = dist(gen);
				c_sc[i*3+j] = dist(gen);
				c_ss[i*3+j] = dist(gen);
			}
		}

		std::cout << "Random coefficients:\n";
		std::cout << "  cos: " << c_cos[0] << " " << c_cos[1] << " " << c_cos[2] << "\n";
		std::cout << "  sin: " << c_sin[0] << " " << c_sin[1] << " " << c_sin[2] << "\n";

		levelSet = [=](double x, double y, double z) {
			double cx[3] = {cos(2*M_PI*x/Lx), cos(2*M_PI*y/Ly), cos(2*M_PI*z/Lz)};
			double sx[3] = {sin(2*M_PI*x/Lx), sin(2*M_PI*y/Ly), sin(2*M_PI*z/Lz)};
			double val = 0;
			for (int i = 0; i < 3; ++i) {
				val += c_cos[i]*cx[i] + c_sin[i]*sx[i];
				for (int j = 0; j < 3; ++j)
					val += c_cc[i*3+j]*cx[i]*cx[j] + c_sc[i*3+j]*sx[i]*cx[j] + c_ss[i*3+j]*sx[i]*sx[j];
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
	saveMesh(mesh, output, split);
	std::cout << "Saved: " << output << "\n";
	return 0;
}

// ──────────────────────────────────────────────────────────
// Subcommand: generate (seed mesh → optimize → TPMS)
// ──────────────────────────────────────────────────────────

int cmdGenerate(const std::string& input, const std::string& output,
				int maxIter, bool split) {
	// Load raw mesh
	xtpms::DefaultTriMesh src;
	if (!OpenMesh::IO::read_mesh(src, input)) {
		std::cerr << "Error: cannot read " << input << "\n";
		return 1;
	}

	// Auto-detect half-period from bounding box
	Vec3d bmin, bmax;
	{
		auto v = src.vertices_begin();
		bmin = bmax = src.point(*v);
		for (++v; v != src.vertices_end(); ++v) {
			const auto& p = src.point(*v);
			for (int i = 0; i < 3; ++i) {
				if (p[i] < bmin[i]) bmin[i] = p[i];
				if (p[i] > bmax[i]) bmax[i] = p[i];
			}
		}
	}
	Vec3d hp;
	for (int i = 0; i < 3; ++i)
		hp[i] = (bmax[i] - bmin[i]) / static_cast<xtpms::DefaultTriMesh::Scalar>(2.0);

	std::cout << "Seed: nv=" << src.n_vertices() << " nf=" << src.n_faces() << "\n";
	std::cout << "BBox: [" << bmin[0] << "," << bmin[1] << "," << bmin[2] << "] - ["
			  << bmax[0] << "," << bmax[1] << "," << bmax[2] << "]\n";
	std::cout << "Half-period: " << hp[0] << ", " << hp[1] << ", " << hp[2] << "\n";

	// Clamp boundary vertices to bbox faces
	{
		double tol = 0.02 * std::min({static_cast<double>(hp[0]),
									   static_cast<double>(hp[1]),
									   static_cast<double>(hp[2])});
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
		std::cout << "Clamped " << clamped << " boundary vertices to bbox\n";
	}

	// Shift to [0, 2*hp] domain
	for (auto v = src.vertices_begin(); v != src.vertices_end(); ++v) {
		auto p = src.point(*v);
		for (int i = 0; i < 3; ++i)
			p[i] -= bmin[i];
		src.set_point(*v, p);
	}

	// Periodize
	xtpms::PeriodicTriMesh mesh;
	mesh.setHalfPeriod(hp);
	for (auto v = src.vertices_begin(); v != src.vertices_end(); ++v)
		mesh.add_vertex(src.point(*v));
	for (auto f = src.faces_begin(); f != src.faces_end(); ++f) {
		auto fv = src.cfv_iter(*f);
		int a = (*fv).idx(); ++fv; int b = (*fv).idx(); ++fv; int c = (*fv).idx();
		mesh.add_face(xtpms::PeriodicTriMesh::VertexHandle(a),
					  xtpms::PeriodicTriMesh::VertexHandle(b),
					  xtpms::PeriodicTriMesh::VertexHandle(c));
	}
	mesh.mergePeriodBoundary();
	std::cout << "Periodized: nv=" << mesh.n_vertices() << " nf=" << mesh.n_faces() << "\n";

	// Optimize toward TPMS (maximize APAC)
	xtpms::TailorADCOptions opts;
	opts.objectiveType = "apac";
	opts.maxIter = maxIter;
	opts.maxStep = 1.0;
	opts.mcfWeight = 0.1;
	opts.enableRemesh = true;
	opts.remeshOpts.adaptiveEps = 1.0;
	opts.enableSurgery = true;
	opts.surgeryStartIter = std::max(maxIter / 3, 10);
	opts.surgeryInterval = 20;
	opts.surgeryOpts.singularityTol = 50.0;

	xtpms::tailorADC(mesh, opts);

	auto geom = xtpms::computeVertexGeometry(mesh);
	Eigen::MatrixX3d u;
	Eigen::Matrix3d kA = xtpms::solveAsymptoticConductivity(mesh, geom, u);
	std::cout << "Final APAC = " << kA.trace() / 3.0 << "\n";
	std::cout << "kA =\n" << kA << "\n";

	saveMesh(mesh, output, split);
	std::cout << "Saved: " << output << "\n";
	return 0;
}

// ──────────────────────────────────────────────────────────
// Main
// ──────────────────────────────────────────────────────────

int main(int argc, char** argv) {
	CLI::App app{"xtpms - Computational design of triply periodic minimal surfaces"};
	app.require_subcommand(1);

	std::string hpStr = "1,1,1";

	// ── periodize ──
	auto* cmdP = app.add_subcommand("periodize", "Merge periodic boundary of a mesh");
	std::string pz_in, pz_out; bool pz_split = false;
	cmdP->add_option("-i,--input", pz_in, "Input OBJ")->required();
	cmdP->add_option("-o,--output", pz_out, "Output OBJ")->required();
	cmdP->add_option("--half-period", hpStr, "Half-period x,y,z")->default_val("1,1,1");
	cmdP->add_flag("--split", pz_split, "Split unit cell at period boundaries");

	// ── compute ──
	auto* cmdC = app.add_subcommand("compute", "Compute effective conductivity tensor");
	std::string cp_in;
	cmdC->add_option("-i,--input", cp_in, "Input OBJ")->required();
	cmdC->add_option("--half-period", hpStr, "Half-period x,y,z")->default_val("1,1,1");

	// ── optimize ──
	auto* cmdO = app.add_subcommand("optimize", "Optimize surface conductivity");
	std::string o_in, o_out, o_obj="apac", o_dir;
	int o_iter=100; double o_step=1, o_mcf=0.1, o_prec=0.1;
	bool o_surg=false, o_split=false;
	int o_surgStart=40, o_surgInt=20; double o_surgTol=50;
	cmdO->add_option("-i,--input", o_in, "Input OBJ")->required();
	cmdO->add_option("-o,--output", o_out, "Output OBJ")->required();
	cmdO->add_option("--half-period", hpStr)->default_val("1,1,1");
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
	cmdO->add_flag("--split", o_split);
	cmdO->add_option("--output-dir", o_dir, "Save intermediate meshes");

	// ── sample ──
	auto* cmdS = app.add_subcommand("sample", "Sample isosurface from level set expression");
	std::string s_expr, s_out;
	int s_res=20; bool s_split=false, s_random=false;
	cmdS->add_option("-e,--expression", s_expr,
		"Level set expression (x,y,z,pi) or built-in: gyroid|schwarzp|diamond");
	cmdS->add_option("-o,--output", s_out, "Output OBJ")->required();
	cmdS->add_option("--half-period", hpStr)->default_val("1,1,1");
	cmdS->add_option("-r,--resolution", s_res)->default_val(20);
	cmdS->add_flag("--random", s_random, "Random triperiodic function (ignore -e)");
	cmdS->add_flag("--split", s_split);

	// ── generate ──
	auto* cmdG = app.add_subcommand("generate", "Generate TPMS from seed mesh (auto period from bbox)");
	std::string g_in, g_out;
	int g_iter=100; bool g_split=false;
	cmdG->add_option("-i,--input", g_in, "Seed mesh OBJ")->required();
	cmdG->add_option("-o,--output", g_out, "Output OBJ")->required();
	cmdG->add_option("--max-iter", g_iter)->default_val(100);
	cmdG->add_flag("--split", g_split);

	CLI11_PARSE(app, argc, argv);

	if (cmdP->parsed()) return cmdPeriodize(pz_in, pz_out, hpStr, pz_split);
	if (cmdC->parsed()) return cmdCompute(cp_in, hpStr);
	if (cmdO->parsed()) return cmdOptimize(o_in, o_out, hpStr, o_obj, o_iter, o_step,
										   o_mcf, o_prec, o_surg, o_surgStart, o_surgInt,
										   o_surgTol, o_split, o_dir);
	if (cmdS->parsed()) return cmdSample(s_expr, s_out, hpStr, s_res, s_split, s_random);
	if (cmdG->parsed()) return cmdGenerate(g_in, g_out, g_iter, g_split);
	return 0;
}
