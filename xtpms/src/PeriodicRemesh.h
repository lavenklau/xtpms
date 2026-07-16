#pragma once

#include "PeriodicMesh.h"

namespace xtpms {

struct RemeshOptions {
	int outerIter{1};
	int innerIter{5};
	double targetLength{-1.0};	// <0 means auto-estimate
	double splitRatio{1.5};		// split edge when length > targetLen * splitRatio
	double collapseRatio{0.5};	// collapse edge when length < targetLen * collapseRatio
	double minLength{-1.0};		// <0 means auto-set to targetLength/4
	double adaptiveEps{0.6};	// curvature-adaptive parameter (>0 to enable)
								// epsilon = 1/adaptiveEps
								// L_target = flatLen * eps / (sqrt(|K_total|) + eps)
								// K_total = 4H² - 2K
	std::string debugOutputDir; // if non-empty, output the mesh at each sub-step
};

// Generate recommended remesh parameters based on the period size
inline RemeshOptions defaultRemeshOptions(const PeriodicTriMesh& mesh) {
	// Use geometric mean of periods to avoid over-refinement on non-uniform periods
	double hp0 = static_cast<double>(mesh.halfPeriod()[0]);
	double hp1 = static_cast<double>(mesh.halfPeriod()[1]);
	double hp2 = static_cast<double>(mesh.halfPeriod()[2]);
	double meanPeriod = 2.0 * std::cbrt(hp0 * hp1 * hp2);
	RemeshOptions opts;
	opts.targetLength = meanPeriod * 0.1;
	opts.minLength = opts.targetLength * 0.1;
	opts.adaptiveEps = 1.0;
	opts.outerIter = 1;
	opts.innerIter = 5;
	return opts;
}

// Periodic Delaunay remesh: adjustEdgeLengths + fixDelaunay + smoothByCircumcenter
void delaunayRemesh(PeriodicTriMesh& mesh, const RemeshOptions& opts = RemeshOptions{});

// Intrinsic Delaunay: iteratively flip non-Delaunay interior edges (vertices fixed).
// Returns the number of flips performed. Used before curvature / singularity measure.
int makeIntrinsicDelaunay(PeriodicTriMesh& mesh);

} // namespace xtpms
