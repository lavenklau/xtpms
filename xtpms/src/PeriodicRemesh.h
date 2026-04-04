#pragma once

#include "PeriodicMesh.h"

namespace xtpms {

struct RemeshOptions {
	int outerIter{1};
	int innerIter{5};
	double targetLength{-1.0};  // <0 自动估算
	double splitRatio{1.5};     // 边长 > targetLen * splitRatio 时分裂
	double collapseRatio{0.5};  // 边长 < targetLen * collapseRatio 时折叠
	double minLength{-1.0};     // <0 时自动设为 targetLength/4
	double adaptiveEps{0.6};    // 曲率自适应参数 (>0 启用)
	                            // epsilon = 1/adaptiveEps
	                            // L_target = flatLen * eps / (sqrt(|K_total|) + eps)
	                            // K_total = 4H² - 2K
	std::string debugOutputDir; // 非空时输出每个子步骤的网格
};

// 周期 Delaunay remesh：adjustEdgeLengths + fixDelaunay + smoothByCircumcenter
void delaunayRemesh(PeriodicTriMesh& mesh, const RemeshOptions& opts = {});

} // namespace xtpms
