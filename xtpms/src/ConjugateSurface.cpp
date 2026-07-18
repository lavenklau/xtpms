#include "ConjugateSurface.h"

#include <OpenMesh/Core/IO/MeshIO.hh>

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <queue>
#include <set>
#include <vector>

namespace xtpms {

namespace {

Eigen::Vector3d toEigen(const DefaultTriMesh::Point& p) {
	return Eigen::Vector3d(p[0], p[1], p[2]);
}

DefaultTriMesh::Point toOM(const Eigen::Vector3d& v) {
	return DefaultTriMesh::Point(v[0], v[1], v[2]);
}

// cot(angle at v_k) for halfedge he: v_i → v_j, face (v_i, v_j, v_k)
double cotHalfedge(DefaultTriMesh& mesh, DefaultTriMesh::HalfedgeHandle he) {
	auto ph = mesh.prev_halfedge_handle(he);
	auto v_i = mesh.from_vertex_handle(he);
	auto v_j = mesh.to_vertex_handle(he);
	auto v_k = mesh.from_vertex_handle(ph);
	Eigen::Vector3d a = toEigen(mesh.point(v_i)) - toEigen(mesh.point(v_k));
	Eigen::Vector3d b = toEigen(mesh.point(v_j)) - toEigen(mesh.point(v_k));
	double cosA = a.dot(b);
	double sinA = a.cross(b).norm();
	return (sinA > 1e-15) ? (cosA / sinA) : 0.0;
}

// Edge cotangent: cot α + cot β (sum of two opposite angles)
double cotEdge(DefaultTriMesh& mesh, DefaultTriMesh::EdgeHandle eh) {
	auto he0 = mesh.halfedge_handle(eh, 0);
	auto he1 = mesh.halfedge_handle(eh, 1);
	double sum = 0.0;
	if (!mesh.is_boundary(he0))
		sum += cotHalfedge(mesh, he0);
	if (!mesh.is_boundary(he1))
		sum += cotHalfedge(mesh, he1);
	return sum;
}

Eigen::Vector3d edgeVector(DefaultTriMesh& mesh, DefaultTriMesh::HalfedgeHandle he) {
	return toEigen(mesh.point(mesh.to_vertex_handle(he))) -
		   toEigen(mesh.point(mesh.from_vertex_handle(he)));
}

} // anonymous namespace

bool computeConjugateSurface(DefaultTriMesh& mesh, const std::string& outputFile, double thetaDeg) {
	using H = DefaultTriMesh::HalfedgeHandle;
	using F = DefaultTriMesh::FaceHandle;
	using E = DefaultTriMesh::EdgeHandle;

	const int nf = static_cast<int>(mesh.n_faces());

	// ── Step 1: Compute dual face points via BFS ────────────
	// (matches minsurf: dV* = cot(edge) * edge_vector, NO rotation)
	std::vector<Eigen::Vector3d> Vdual(static_cast<std::size_t>(nf));

	F f0 = *mesh.faces_begin();
	H he0;
	for (auto fh = mesh.fh_begin(f0); fh != mesh.fh_end(f0); ++fh) {
		H he = *fh;
		if (!mesh.is_boundary(he) && !mesh.is_boundary(mesh.opposite_halfedge_handle(he))) {
			he0 = he;
			break;
		}
	}
	if (!he0.is_valid()) {
		std::cerr << "Error: no interior halfedge in seed face\n";
		return false;
	}

	Vdual[static_cast<std::size_t>(f0.idx())] = Eigen::Vector3d::Zero();

	std::set<E> visitedEdges;
	std::queue<H> front;
	front.push(he0);
	visitedEdges.insert(mesh.edge_handle(he0));

	while (!front.empty()) {
		H fr = front.front();
		front.pop();
		if (mesh.is_boundary(fr))
			continue;
		H fropp = mesh.opposite_halfedge_handle(fr);
		if (mesh.is_boundary(fropp))
			continue;

		// minsurf formula: cot(edge) * edge_vector (no rotation)
		Eigen::Vector3d ev = edgeVector(mesh, fr) * cotEdge(mesh, mesh.edge_handle(fr));
		F fOpp = mesh.face_handle(fropp);
		Vdual[static_cast<std::size_t>(fOpp.idx())] =
			Vdual[static_cast<std::size_t>(mesh.face_handle(fr).idx())] + ev;

		for (auto fh = mesh.fh_begin(fOpp); fh != mesh.fh_end(fOpp); ++fh) {
			H fhe = *fh;
			if (fhe == fropp)
				continue;
			E e = mesh.edge_handle(fhe);
			if (visitedEdges.insert(e).second)
				front.push(fhe);
		}
	}

	// ── Step 2: Set conjugate vertex = average of surrounding dual points ─
	// (matches minsurf: conj.set_point(vh, cdual) — no LSQ solve)
	DefaultTriMesh conj = mesh;

	for (auto vh = mesh.vertices_begin(); vh != mesh.vertices_end(); ++vh) {
		Eigen::Vector3d cdual = Eigen::Vector3d::Zero();
		int counter = 0;

		for (auto vf = mesh.vf_begin(*vh); vf != mesh.vf_end(*vh); ++vf) {
			cdual += Vdual[static_cast<std::size_t>((*vf).idx())];
			counter++;
		}

		if (counter > 0) {
			cdual /= static_cast<double>(counter);
			conj.set_point(*vh, toOM(cdual));
		}
	}

	// ── Step 3: Bonnet rotation (associate family) ────────
	if (std::abs(thetaDeg - 90.0) > 1e-9) {
		double rad = thetaDeg * M_PI / 180.0;
		double c = std::cos(rad);
		double s = std::sin(rad);
		for (auto vh = mesh.vertices_begin(); vh != mesh.vertices_end(); ++vh) {
			Eigen::Vector3d vOrig = toEigen(mesh.point(*vh));
			Eigen::Vector3d vConj = toEigen(conj.point(*vh));
			conj.set_point(*vh, toOM(c * vOrig + s * vConj));
		}
	}

	// ── Validation: cotangent Laplacian ──────────────────────
	{
		double maxLap = 0.0, sumLap = 0.0;
		int nInterior = 0;
		for (auto vh = conj.vertices_begin(); vh != conj.vertices_end(); ++vh) {
			if (conj.is_boundary(*vh))
				continue;
			nInterior++;
			Eigen::Vector3d laplacian = Eigen::Vector3d::Zero();
			for (auto voh = conj.voh_begin(*vh); voh != conj.voh_end(*vh); ++voh) {
				auto he = *voh;
				if (conj.is_boundary(he))
					continue;
				auto opp = conj.opposite_halfedge_handle(he);
				if (conj.is_boundary(opp))
					continue;
				auto vw = conj.to_vertex_handle(he);
				double w = cotHalfedge(conj, he) + cotHalfedge(conj, opp);
				laplacian += w * (toEigen(conj.point(vw)) - toEigen(conj.point(*vh)));
			}
			double L = laplacian.norm();
			if (L > maxLap)
				maxLap = L;
			sumLap += L;
		}
		double avgLap = (nInterior > 0) ? sumLap / nInterior : 0.0;
		std::cout << "validation: |cot-Laplacian| max=" << maxLap << " avg=" << avgLap << " ("
				  << nInterior << " interior vertices)\n";
	}

	// ── Step 4: Write output ────────────────────────────────
	std::string outFile = outputFile.empty() ? "conjugate_surface.obj" : outputFile;
	if (!OpenMesh::IO::write_mesh(conj, outFile)) {
		std::cerr << "Error: failed to write " << outFile << "\n";
		return false;
	}
	std::cout << "Conjugate surface saved: " << outFile << " (nv=" << conj.n_vertices()
			  << " nf=" << conj.n_faces() << ")\n";
	return true;
}

} // namespace xtpms
