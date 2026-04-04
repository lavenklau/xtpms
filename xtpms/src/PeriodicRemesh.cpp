#include "PeriodicRemesh.h"
#include "AsymptoticConductivity.h"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <queue>
#include <vector>

namespace xtpms {

namespace {

using Vec3d = PeriodicTriMesh::Vec3d;
using VH = PeriodicTriMesh::VertexHandle;
using EH = PeriodicTriMesh::EdgeHandle;
using HH = PeriodicTriMesh::HalfedgeHandle;
using FH = PeriodicTriMesh::FaceHandle;

// 周期包装：将差向量映射到 [-hp, hp)
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

// 周期边长
double periodEdgeLength(const PeriodicTriMesh& m, EH eh) {
	HH he = m.halfedge_handle(eh, 0);
	Vec3d ev = makePeriod(m.point(m.to_vertex_handle(he)) - m.point(m.from_vertex_handle(he)), m.halfPeriod());
	return ev.norm();
}

// 周期边中点（minsurf: eval_period_edge(he, 0.5)）
Vec3d periodMidpoint(const PeriodicTriMesh& m, HH he) {
	const Vec3d& p0 = m.point(m.from_vertex_handle(he));
	Vec3d ev = makePeriod(m.point(m.to_vertex_handle(he)) - p0, m.halfPeriod());
	return p0 + ev * 0.5;
}

// 周期扇形角度（minsurf: period_sector_angle）
double periodSectorAngle(const PeriodicTriMesh& m, HH he) {
	const Vec3d& a = m.point(m.from_vertex_handle(he));
	const Vec3d hp = m.halfPeriod();
	Vec3d ab = makePeriod(m.point(m.to_vertex_handle(he)) - a, hp);
	Vec3d ac = makePeriod(m.point(m.to_vertex_handle(m.next_halfedge_handle(he))) - a, hp);
	double la = ab.norm(), lb = ac.norm();
	if (la < 1e-15 || lb < 1e-15) return 0.0;
	double cosA = std::clamp(static_cast<double>(ab | ac) / (la * lb), -1.0, 1.0);
	return std::acos(cosA);
}

// 周期面面积
double periodFaceArea(const PeriodicTriMesh& m, FH fh) {
	auto fv = m.cfv_iter(fh);
	const Vec3d& p0 = m.point(*fv); ++fv;
	const Vec3d hp = m.halfPeriod();
	Vec3d e1 = makePeriod(m.point(*fv) - p0, hp); ++fv;
	Vec3d e2 = makePeriod(m.point(*fv) - p0, hp);
	return 0.5 * (e1 % e2).norm();
}

// 顶点法向
Vec3d vertexNormal(const PeriodicTriMesh& m, VH vh) {
	Vec3d n(0, 0, 0);
	const Vec3d hp = m.halfPeriod();
	for (auto voh = m.cvoh_iter(vh); voh.is_valid(); ++voh) {
		if (m.is_boundary(*voh)) continue;
		double angle = periodSectorAngle(m, *voh);
		const Vec3d& p0 = m.point(m.from_vertex_handle(*voh));
		Vec3d e1 = makePeriod(m.point(m.to_vertex_handle(*voh)) - p0, hp);
		Vec3d e2 = makePeriod(m.point(m.to_vertex_handle(m.next_halfedge_handle(*voh))) - p0, hp);
		Vec3d fn = e1 % e2;
		double fnl = fn.norm();
		if (fnl > 1e-15) fn /= static_cast<DefaultTriMesh::Scalar>(fnl);
		n += fn * static_cast<DefaultTriMesh::Scalar>(angle);
	}
	double nl = n.norm();
	if (nl > 1e-15) n /= static_cast<DefaultTriMesh::Scalar>(nl);
	return n;
}

// Delaunay 条件：把菱形四个顶点 unwrap 到同一位置再算角度
bool isDelaunay(const PeriodicTriMesh& m, EH eh) {
	if (m.is_boundary(eh)) return true;
	HH he0 = m.halfedge_handle(eh, 0);
	HH he1 = m.halfedge_handle(eh, 1);
	const Vec3d hp = m.halfPeriod();

	// 菱形: edge = (v0, v1), opposite vertices = (v2, v3)
	// v0 = from(he0), v1 = to(he0)
	// v2 = to(next(he0)),  v3 = to(next(he1))
	const Vec3d& p0 = m.point(m.from_vertex_handle(he0));
	Vec3d p1 = p0 + makePeriod(m.point(m.to_vertex_handle(he0)) - p0, hp);
	Vec3d p2 = p0 + makePeriod(m.point(m.to_vertex_handle(m.next_halfedge_handle(he0))) - p0, hp);
	Vec3d p3 = p0 + makePeriod(m.point(m.to_vertex_handle(m.next_halfedge_handle(he1))) - p0, hp);

	// angle at v2 in triangle (v0, v1, v2)
	Vec3d e20 = p0 - p2, e21 = p1 - p2;
	double cos2 = static_cast<double>(e20 | e21) / (e20.norm() * e21.norm());
	double a2 = std::acos(std::clamp(cos2, -1.0, 1.0));

	// angle at v3 in triangle (v1, v0, v3)
	Vec3d e30 = p0 - p3, e31 = p1 - p3;
	double cos3 = static_cast<double>(e30 | e31) / (e30.norm() * e31.norm());
	double a3 = std::acos(std::clamp(cos3, -1.0, 1.0));

	return (a2 + a3) <= M_PI + 1e-8;
}

// 外接圆心
Vec3d circumcenter(const Vec3d& p1, const Vec3d& p2, const Vec3d& p3) {
	double a = (p3 - p2).norm(), b = (p3 - p1).norm(), c = (p2 - p1).norm();
	double a2 = a * a, b2 = b * b, c2 = c * c;
	double w0 = a2 * (b2 + c2 - a2);
	double w1 = b2 * (c2 + a2 - b2);
	double w2 = c2 * (a2 + b2 - c2);
	double wsum = w0 + w1 + w2;
	if (std::abs(wsum) < 1e-30) return (p1 + p2 + p3) * (1.0 / 3.0);
	return p1 * (w0 / wsum) + p2 * (w1 / wsum) + p3 * (w2 / wsum);
}

// period_shift: 将所有顶点 wrap 回 [0, 2*hp)
void periodShift(PeriodicTriMesh& m) {
	const Vec3d hp = m.halfPeriod();
	for (auto v = m.vertices_begin(); v != m.vertices_end(); ++v) {
		Vec3d p = m.point(*v);
		for (int i = 0; i < 3; ++i) {
			double period = 2.0 * static_cast<double>(hp[i]);
			double pi = static_cast<double>(p[i]);
			while (pi < -1e-5) pi += period;
			while (pi > period + 1e-5) pi -= period;
			p[i] = static_cast<DefaultTriMesh::Scalar>(pi);
		}
		m.set_point(*v, p);
	}
}

// collapse foldover 检查（minsurf: shouldCollapse）
// 检查 collapse he 到 midpoint 后，保留顶点周围面是否翻转
bool shouldCollapse(const PeriodicTriMesh& m, HH he, const Vec3d& midpoint) {
	VH vKeep = m.to_vertex_handle(he);
	VH vRemove = m.from_vertex_handle(he);
	const Vec3d hp = m.halfPeriod();

	for (auto voh = m.cvoh_iter(vKeep); voh.is_valid(); ++voh) {
		if (m.is_boundary(*voh)) continue;
		VH va = m.to_vertex_handle(*voh);
		VH vb = m.to_vertex_handle(m.next_halfedge_handle(*voh));
		if (va == vRemove || vb == vRemove) continue;

		// 所有点都 make_period 到 midpoint 附近
		Vec3d a = midpoint + makePeriod(m.point(va) - midpoint, hp);
		Vec3d b = midpoint + makePeriod(m.point(vb) - midpoint, hp);
		Vec3d n = (a - midpoint) % (b - midpoint);
		if (n.sqrnorm() < 1e-20) return false; // degenerate → 不 collapse
	}
	return true;
}

// 平均边长
double estimateAvgEdgeLength(const PeriodicTriMesh& m) {
	double total = 0;
	int cnt = 0;
	for (auto e = m.edges_begin(); e != m.edges_end(); ++e) {
		total += periodEdgeLength(m, *e);
		++cnt;
	}
	return (cnt > 0) ? total / cnt : 0.1;
}

// 曲率自适应目标边长（和 minsurf findTotalCurvatureTargetL 完全对齐）
// averageK = mean of (4H²-2K) at two endpoints
// L = flatLen * eps / (fabs(sqrt(averageK)) + eps)
double adaptiveTargetLength(const PeriodicTriMesh& m, EH eh,
							double flatLength, double epsilon) {
	const Vec3d hp = m.halfPeriod();
	HH he0 = m.halfedge_handle(eh, 0);
	VH v[2] = { m.from_vertex_handle(he0), m.to_vertex_handle(he0) };

	double averageK = 0;
	for (int s = 0; s < 2; ++s) {
		// 构建 1-ring → Compile1ring（和 minsurf 完全一致）
		Eigen::Vector3d center(
			static_cast<double>(m.point(v[s])[0]),
			static_cast<double>(m.point(v[s])[1]),
			static_cast<double>(m.point(v[s])[2]));
		std::vector<Eigen::Vector3d> ring;
		if (m.is_boundary(v[s])) {
			for (auto voh = m.cvoh_iter(v[s]); voh.is_valid(); ++voh) {
				Vec3d ev = makePeriod(m.point(m.to_vertex_handle(*voh)) - m.point(v[s]), hp);
				ring.push_back(center + Eigen::Vector3d(ev[0], ev[1], ev[2]));
			}
		} else {
			HH he_start = m.halfedge_handle(v[s]);
			HH he = he_start;
			do {
				Vec3d ev = makePeriod(m.point(m.to_vertex_handle(he)) - m.point(v[s]), hp);
				ring.push_back(center + Eigen::Vector3d(ev[0], ev[1], ev[2]));
				he = m.opposite_halfedge_handle(m.prev_halfedge_handle(he));
			} while (he != he_start && ring.size() < 30);
		}
		if (ring.size() >= 3) {
			xtpms::Compile1ring vring(center, ring);
			averageK += (4.0 * vring.H * vring.H - 2.0 * vring.K);
		}
	}
	averageK /= 2.0;

	// 和 minsurf 完全一致：fabs(sqrt(averageK))
	double sqrtK = (averageK >= 0) ? std::sqrt(averageK) : std::sqrt(-averageK);
	double L = flatLength * epsilon / (sqrtK + epsilon);
	return L;
}

} // namespace

// ══════════════════════════════════════════════════════════════
// delaunayRemesh（严格按 minsurf 流程）
// ══════════════════════════════════════════════════════════════

void delaunayRemesh(PeriodicTriMesh& mesh, const RemeshOptions& opts) {
	mesh.request_vertex_status();
	mesh.request_edge_status();
	mesh.request_halfedge_status();
	mesh.request_face_status();

	periodShift(mesh);

	if (opts.innerIter <= 0) return;

	double flatLen = opts.targetLength;
	if (flatLen < 0) flatLen = estimateAvgEdgeLength(mesh);
	double minLen = opts.minLength;
	if (minLen < 0 || minLen > flatLen) minLen = flatLen / 4.0;

	const double adaptEps = (opts.adaptiveEps > 0) ? (1.0 / opts.adaptiveEps) : 0;
	const bool useAdaptive = (adaptEps > 0);

	// 每条边的目标边长（自适应或全局）
	auto edgeTargetLen = [&](EH eh) -> double {
		if (useAdaptive)
			return adaptiveTargetLength(mesh, eh, flatLen, adaptEps);
		return flatLen;
	};

	const bool dbg = !opts.debugOutputDir.empty();
	auto dbgSave = [&](const std::string& tag) {
		if (dbg) mesh.saveUnitCell(opts.debugOutputDir + "/remesh_" + tag + ".obj");
	};

	dbgSave("0_input");

	for (int outer = 0; outer < opts.outerIter; ++outer) {

		// ── Phase 1: split long edges + collapse short edges ──
		{
			// split（单 pass，遍历快照）
			std::vector<EH> toSplit;
			for (auto e = mesh.edges_begin(); e != mesh.edges_end(); ++e) {
				double len = periodEdgeLength(mesh, *e);
				double tgt = edgeTargetLen(*e);
				if (len > tgt * opts.splitRatio && len > minLen)
					toSplit.push_back(*e);
			}
			for (EH eh : toSplit) {
				if (!eh.is_valid() || mesh.status(eh).deleted()) continue;
				HH he = mesh.halfedge_handle(eh, 0);
				Vec3d mid = periodMidpoint(mesh, he);
				mesh.split(eh, mid);
			}
			periodShift(mesh);
			dbgSave("1_after_split");

			// collapse（单 pass）
			std::vector<EH> toCollapse;
			for (auto e = mesh.edges_begin(); e != mesh.edges_end(); ++e) {
				if (mesh.status(*e).deleted()) continue;
				double len = periodEdgeLength(mesh, *e);
				double tgt = edgeTargetLen(*e);
				if (len < tgt * opts.collapseRatio)
					toCollapse.push_back(*e);
			}
			int collapsed = 0;
			for (EH eh : toCollapse) {
				if (!eh.is_valid() || mesh.status(eh).deleted()) continue;
				HH he = mesh.halfedge_handle(eh, 0);
				if (!mesh.is_collapse_ok(he)) continue;
				Vec3d mid = periodMidpoint(mesh, he);
				if (!shouldCollapse(mesh, he, mid)) continue;
				VH vTo = mesh.to_vertex_handle(he);
				mesh.collapse(he);
				mesh.set_point(vTo, mid);
				++collapsed;
			}
			mesh.garbage_collection();
			periodShift(mesh);
			dbgSave("2_after_collapse");
		}

		// ── Phase 2: fixDelaunay ──
		auto flipDelaunay = [&]() {
			std::queue<int> q;
			const std::size_t ne = mesh.n_edges();
			std::vector<bool> inQ(ne, true);
			for (std::size_t i = 0; i < ne; ++i) q.push(static_cast<int>(i));
			int maxFlips = static_cast<int>(mesh.n_vertices()) * 100;
			int flips = 0;
			while (!q.empty() && flips < maxFlips) {
				int eid = q.front(); q.pop();
				if (static_cast<std::size_t>(eid) >= mesh.n_edges()) continue;
				inQ[static_cast<std::size_t>(eid)] = false;
				EH eh(eid);
				if (!eh.is_valid() || mesh.status(eh).deleted()) continue;
				if (mesh.is_boundary(eh)) continue;
				if (!isDelaunay(mesh, eh)) {
					if (!mesh.is_flip_ok(eh)) continue;
					HH h0 = mesh.halfedge_handle(eh, 0);
					HH h1 = mesh.halfedge_handle(eh, 1);
					int nb[4] = {
						mesh.edge_handle(mesh.next_halfedge_handle(h0)).idx(),
						mesh.edge_handle(mesh.prev_halfedge_handle(h0)).idx(),
						mesh.edge_handle(mesh.next_halfedge_handle(h1)).idx(),
						mesh.edge_handle(mesh.prev_halfedge_handle(h1)).idx()
					};
					mesh.flip(eh);
					++flips;
					for (int n : nb) {
						if (n >= 0 && static_cast<std::size_t>(n) < mesh.n_edges() && !inQ[static_cast<std::size_t>(n)]) {
							inQ[static_cast<std::size_t>(n)] = true;
							q.push(n);
						}
					}
				}
			}
		};

		flipDelaunay();
		dbgSave("3_after_flip");

		// ── Phase 3: smooth + flip ──
		for (int inner = 0; inner < opts.innerIter; ++inner) {
			const std::size_t nv = mesh.n_vertices();
			std::vector<Vec3d> newPos(nv);
			const Vec3d hp = mesh.halfPeriod();

			for (auto vi = mesh.vertices_begin(); vi != mesh.vertices_end(); ++vi) {
				VH vh = *vi;
				newPos[static_cast<std::size_t>(vh.idx())] = mesh.point(vh);
				if (mesh.is_boundary(vh)) continue;

				Vec3d updateDir(0, 0, 0);
				double totalArea = 0;
				const Vec3d& pv = mesh.point(vh);

				for (auto voh = mesh.cvoh_iter(vh); voh.is_valid(); ++voh) {
					if (mesh.is_boundary(*voh)) continue;
					FH fh = mesh.face_handle(*voh);
					double fArea = periodFaceArea(mesh, fh);

					// 面三个顶点 unwrap 到 pv 附近
					auto fvi = mesh.cfv_iter(fh);
					Vec3d fp0 = pv + makePeriod(mesh.point(*fvi) - pv, hp); ++fvi;
					Vec3d fp1 = pv + makePeriod(mesh.point(*fvi) - pv, hp); ++fvi;
					Vec3d fp2 = pv + makePeriod(mesh.point(*fvi) - pv, hp);

					Vec3d center = mesh.is_boundary(fh)
						? (fp0 + fp1 + fp2) * (1.0 / 3.0)
						: circumcenter(fp0, fp1, fp2);

					updateDir += (center - pv) * static_cast<DefaultTriMesh::Scalar>(fArea);
					totalArea += fArea;
				}

				if (totalArea > 1e-15) {
					updateDir /= static_cast<DefaultTriMesh::Scalar>(totalArea);
					Vec3d n = vertexNormal(mesh, vh);
					updateDir -= n * static_cast<DefaultTriMesh::Scalar>(static_cast<double>(updateDir | n));
					newPos[static_cast<std::size_t>(vh.idx())] = pv + updateDir * 0.5;
				}
			}
			for (auto vi = mesh.vertices_begin(); vi != mesh.vertices_end(); ++vi) {
				mesh.set_point(*vi, newPos[static_cast<std::size_t>((*vi).idx())]);
			}
			periodShift(mesh);
			dbgSave("4_smooth" + std::to_string(inner));
			flipDelaunay();
			dbgSave("5_flip" + std::to_string(inner));
		}
	}

	// 删除度为 3 的顶点（和 minsurf delete_degree3_faces 对齐）
	for (auto v_it = mesh.vertices_begin(); v_it != mesh.vertices_end(); ++v_it) {
		if (mesh.status(*v_it).deleted()) continue;
		if (mesh.valence(*v_it) == 3 && !mesh.is_boundary(*v_it)) {
			VH neighbors[3];
			int cnt = 0;
			for (auto voh = mesh.cvoh_iter(*v_it); voh.is_valid() && cnt < 3; ++voh)
				neighbors[cnt++] = mesh.to_vertex_handle(*voh);
			if (cnt == 3) {
				mesh.delete_vertex(*v_it, true);
				mesh.add_face(neighbors[0], neighbors[1], neighbors[2]);
			}
		}
	}

	mesh.garbage_collection();
	periodShift(mesh);

	// 检查
	int bndEdges = 0;
	for (auto e = mesh.edges_begin(); e != mesh.edges_end(); ++e)
		if (mesh.is_boundary(*e)) ++bndEdges;
	if (bndEdges > 0)
		std::cerr << "WARNING: delaunayRemesh produced " << bndEdges << " boundary edges\n";
}

} // namespace xtpms
