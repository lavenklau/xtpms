#pragma once

// 周期网格：跨周期边判定、周期同步邻域访问；periodizeFrom 为 AABB 窄带 SDF + 周期对称 + Marching Cubes。
#include "MeshTypes.h"

#include <array>
#include <cmath>
#include <tuple>
#include <vector>

namespace xtpms {

// 顶点/边周期边界标记位：bit0=x_min, bit1=x_max, bit2=y_min, bit3=y_max, bit4=z_min, bit5=z_max
struct BoundaryFlag {
	static constexpr int kMinBit[3] = {0, 2, 4};
	static constexpr int kMaxBit[3] = {1, 3, 5};
	static constexpr int kPeriodMask = 0x3F;
	static constexpr int kMinMask = 0x15; // 010101
	static constexpr int kMaxMask = 0x2A; // 101010

	int flag = 0;

	void classify(double px, double py, double pz,
				  double xmin, double xmax, double ymin, double ymax,
				  double zmin, double zmax, double tol) {
		flag = 0;
		const double p[3] = {px, py, pz};
		const double lo[3] = {xmin, ymin, zmin};
		const double hi[3] = {xmax, ymax, zmax};
		for (int i = 0; i < 3; ++i) {
			if (p[i] - lo[i] < tol) flag |= (1 << kMinBit[i]);
			if (hi[i] - p[i] < tol) flag |= (1 << kMaxBit[i]);
		}
	}

	bool isMinBoundary(int axis) const { return (flag >> kMinBit[axis]) & 1; }
	bool isMaxBoundary(int axis) const { return (flag >> kMaxBit[axis]) & 1; }
	bool isMinBoundary() const { return flag & kMinMask; }
	bool isMaxBoundary() const { return flag & kMaxMask; }
	bool isBoundary() const { return flag & kPeriodMask; }
	int getMask() const { return flag & kPeriodMask; }
};

// 周期边界合并选项
struct MergeBoundaryOptions {
	double shortEdgeTol{2e-5};      // 折叠短边阈值
	double projectionTol{0.01};     // 投影距离警告阈值（平方距离）
	double vertexWeldTol{1e-6};     // 顶点合并容差
};

// periodizeFrom：平移网格、周期复制 + AABB 窄带体素 + 有符号距离 + 周期对称 + Marching Cubes 生成周期网格。
struct PeriodizeOptions {
	double bboxPaddingWorld{0.0};
	double bboxPaddingFraction{0.0};
	// >0 时：对边界上短于该长度的边尝试 collapse（需 OpenMesh is_collapse_ok）。0 表示不折叠。
	double collapseShortBoundaryEdgesBelow{0.0};

	/// 基本域 \([0,2\cdot\text{halfPeriod}]\)^3 的 AABB 填充（写入 resolvedBBox*）。
	/// 每轴体素单元数 \(N\)（格点 \(N+1\)），域为 \([0,L_x]\times[0,L_y]\times[0,L_z]\)。
	int mcCellsPerAxis{32};
	/// 与表面相交的体素集合在 26-邻域上扩张的层数（补齐窄带、不对称角点）。
	int mcVoxelDilateLayers{2};
	double mcIsoValue{0.0};
	bool mcWeldVertices{true};

	std::array<double, 3> resolvedBBoxMin{};
	std::array<double, 3> resolvedBBoxMax{};
};

// 拓扑手术选项
struct SurgeryOptions {
	// 对齐 minsurf 的主流设置（surgery-tol=25 surgery-type=2，8/14 脚本使用）
	double singularityTol{25.0};      // 奇异度绝对阈值
	double singularityRatio{0.0};     // 自适应比例（0 = 禁用，仅用绝对阈值）
	                                  // max(tol, ratio * avgH)
	double maxCutAreaFraction{0.3};   // 切除区域最大占比，超过则放弃 surgery（安全阀）
	int surgeryType{2};               // 1: |H|; 2: max(|κ₁|,|κ₂|)（minsurf 主流）
	double islandCullRatio{0.1};      // 面数 < 最大连通分量 * ratio 的孤岛被删除
};

class PeriodicTriMesh : public DefaultTriMesh {
public:
	using Base = DefaultTriMesh;
	using VertexHandle = typename Base::VertexHandle;
	using EdgeHandle = typename Base::EdgeHandle;
	using HalfedgeHandle = typename Base::HalfedgeHandle;
	using Vec3d = typename Base::Point;

public:
	PeriodicTriMesh();

	void setHalfPeriod(const Vec3d& halfPeriod);
	Vec3d halfPeriod() const;

	Vec3d wrapVector(const Vec3d& v) const;

	Vec3d shift2origin(const Vec3d& p) const;

	void shift2originInPlace();

	bool isPeriodicEdge(VertexHandle center, VertexHandle neighbor) const;

	bool isPeriodicEdge(EdgeHandle eh) const;

	std::tuple<std::vector<HalfedgeHandle>, std::vector<Vec3d>> e1ring(VertexHandle center) const;

	// 周期化：拷贝 →（可选）折短边界边 → AABB min 平移到原点 → 3×3×3 周期复制 → AABB 树 → 窄带体素 →
	// 格点 SDF（最近面法向定号）→ 周期格点组加权平均 → Marching Cubes 写回本网格。
	// 若 &naiveMesh == this，会先拷贝再处理。
	void periodizeFrom(const DefaultTriMesh& naiveMesh, PeriodizeOptions& options);

	// 折叠长度 < max(shortTol, 1e-12) 的边界边（若 is_collapse_ok）。
	void mergePeriodEdges(double shortEdgeTol);

	// 周期边界合并：边匹配 + 分裂对齐 + 顶点合并，使拓扑非周期网格变为真正周期网格。
	void mergePeriodBoundary(const MergeBoundaryOptions& options = MergeBoundaryOptions{});

	// 将顶点 wrap 到基本域 [0, 2*halfPeriod) 内。
	void periodShift();

	// 检测并删除不贯穿周期的孤岛连通分量
	// 从每个连通分量的种子顶点出发，沿 3 个轴做 BFS 累加周期包装后的坐标差
	// 如果某轴方向累积距离 >= 一个完整周期，说明该分量贯穿了该轴
	// 只要贯穿任意一个轴就保留，否则删除
	// 返回删除的面数
	int removeNonPeriodicIslands();

	// 仅保留面数最大的连通分量，返回删除的分量数。
	// 用于 surgery 后清理残留孤岛，避免 FEM Laplacian 零空间多维化。
	int keepLargestComponent();

	// 在周期边界处截断跨周期边（和 minsurf split_unit_cell 对齐）
	// 截断后网格不再是周期闭合的（有边界边在周期面上）
	void splitUnitCell();

	// 保存单元胞。split=true 时先复制网格做 splitUnitCell 再保存（默认），
	// split=false 时用原始 unwrap 逻辑保存（不修改原网格）。
	bool saveUnitCell(const std::string& filename, bool split = true) const;

	// 拓扑手术：检测颈部奇异（高曲率），删除周围面，填洞，去除孤岛。
	// 返回 true 表示执行了手术。
	bool surgery(const SurgeryOptions& opts = SurgeryOptions{});

private:
	Vec3d halfPeriod_{};
};

} // namespace xtpms
