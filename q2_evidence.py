# ============================================================================
# 【学习注释】问题二：结果落盘与绘图（q2_evidence.py，约 480 行）
# ----------------------------------------------------------------------------
# 负责把候选指标写成 CSV、把可靠性权衡曲线画出来（含中文字体配置）。
# 值得学的点：**证据留痕**——凡进入论文的图表都由脚本可复现地生成，
# 而不是手工绘图；这也是支撑材料应有的样子。
# ============================================================================

""" 问题 2 结果与图表输出。"""
from __future__ import annotations
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
import numpy as np
if TYPE_CHECKING:
    from q2 import Q2Result
@dataclass(frozen=True)
class ReliabilityTradeoffPoint :
    allowed_loss: float
    actual_loss: float
    robust_diameter: float
    relative_diameter: float
    safe_coverage: float
    point: tuple[float, float]
def build_reliability_tradeoff(
    result: Q2Result,
) -> tuple[ReliabilityTradeoffPoint, ...]:
    if not result.reliability_tradeoff:
        raise ValueError (" 没有可用的可靠性权衡数据")
    baseline = result.reliability_tradeoff[0].robust_diameter
    return tuple(
        ReliabilityTradeoffPoint(
            allowed_loss=envelope.allowed_loss,
            actual_loss=envelope.actual_loss,
            robust_diameter=envelope.robust_diameter,
            relative_diameter=(
                1.0
                if baseline <= 1e-12
                else envelope.robust_diameter / baseline
            ),
            safe_coverage=result.validated_candidates[
                envelope.best_index
            ].safe_coverage,
            point=result.validated_candidates[envelope.best_index].point,
        )
        for envelope in result.reliability_tradeoff
    )
def write_candidate_csv(result: Q2Result, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok= True)
    fieldnames = [
        "x 坐标 _ 米",
        "y 坐标 _ 米",
        " 沿首次示向位移 u_ 米",
        " 侧向位移 v_ 米",
        " 安全覆盖率",
        " 不确定覆盖率",
        " 必然失效覆盖率",
        " 交会角惩罚",
        " 粗筛最坏直径 _ 米",
        " 粗筛加权评分",
        " 是否粗筛帕累托点",
        " 局部加密层级",
        " 验证最坏测向直径 _ 米",
        " 过强信号后验直径 _ 米",
        " 无信号后验直径 _ 米",
        " 验证稳健直径 _ 米",
        " 验证加权评分",
        " 是否验证帕累托点",
        " 是否可靠性可行区域",
        " 是否主候选区域",
        " 是否最高可靠覆盖近优区域",
        " 是否帕累托加权近优点",
        " 数值稳定性检查角色",
        " 粗配置稳健直径 _ 米",
        " 加密配置稳健直径 _ 米",
        " 默认与加密相对差",
    ]
    near_points = {candidate.point for candidate in result.near_optimal_candidates}
    safe_near_points = {
        candidate.point for candidate in result.safe_near_optimal_candidates
    }
    pareto_near_points = {
        candidate.point for candidate in result.pareto_near_optimal_candidates
    }
    feasible_points = {
        candidate.point for candidate in result.reliability_feasible_candidates
    }
    validated_by_point = {
        candidate.point: candidate for candidate in result.validated_candidates
    }
    stability_by_point = {
        record.point: record for record in result.numerical_stability_records
    }
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for candidate in result.candidates:
            validated = validated_by_point.get(candidate.point)
            stability = stability_by_point.get(candidate.point)
            writer.writerow(
                {
                    "x 坐标 _ 米": candidate.point[0],
                    "y 坐标 _ 米": candidate.point[1],
                    " 沿首次示向位移 u_ 米": candidate.local_u,
                    " 侧向位移 v_ 米": candidate.local_v,
                    " 安全覆盖率": candidate.safe_coverage,
                    " 不确定覆盖率": candidate.uncertain_coverage,
                    " 必然失效覆盖率": candidate.certain_failure,
                    " 交会角惩罚": candidate.angle_penalty,
                    " 粗筛最坏直径 _ 米": candidate.proxy_diameter,
                    " 粗筛加权评分": candidate.proxy_score,
                    " 是否粗筛帕累托点": int(candidate.pareto),
                    " 局部加密层级": candidate.refinement_level,
                    " 验证最坏测向直径 _ 米": (
                        "" if validated is None else validated.validated_bearing_worst
                    ),
                    " 过强信号后验直径 _ 米": (
                        "" if validated is None else validated.strong_signal_diameter
                    ),
                    " 无信号后验直径 _ 米": (
                        "" if validated is None else validated.no_signal_diameter
                    ),
                    " 验证稳健直径 _ 米": (
                        "" if validated is None else validated.validated_robust_diameter
                    ),
                    " 验证加权评分": (
                        "" if validated is None else validated.validated_score
                    ),
                    " 是否验证帕累托点": (
                        0 if validated is None else int(validated.validated_pareto)
                    ),
                    " 是否可靠性可行区域": int(candidate.point in feasible_points),
                    " 是否主候选区域": int(candidate.point in near_points),
                    " 是否最高可靠覆盖近优区域": int(
                        candidate.point in safe_near_points
                    ),
                    " 是否帕累托加权近优点": int(
                        candidate.point in pareto_near_points
                    ),
                    " 数值稳定性检查角色": (
                        "" if stability is None else stability.role
                    ),
                    " 粗配置稳健直径 _ 米": (
                        "" if stability is None else stability.coarse_robust_diameter
                    ),
                    " 加密配置稳健直径 _ 米": (
                        "" if stability is None else stability.fine_robust_diameter
                    ),
                    " 默认与加密相对差": (
                        ""
                        if stability is None
                        else stability.default_fine_relative_difference
                    ),
                }
            )
def write_reliability_tradeoff_csv(result: Q2Result, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok= True)
    fieldnames = [
        " 允许可靠性下降 _ 百分点",
        " 实际可靠性下降 _ 百分点",
        " 最优稳健定位直径 _ 米",
        " 相对稳健定位直径",
        " 定位直径改善 _ 百分比",
        " 对应安全覆盖率",
        " 对应点 x_ 米",
        " 对应点 y_ 米",
        " 是否自动选中",
        " 定位改善数值容差 _ 米",
        " 数值误差标定比例",
        " 是否执行数值加密标定",
        " 精验证候选总数",
    ]
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for point in build_reliability_tradeoff(result):
            writer.writerow(
                {
                    " 允许可靠性下降 _ 百分点": 100.0 * point.allowed_loss,
                    " 实际可靠性下降 _ 百分点": 100.0 * point.actual_loss,
                    " 最优稳健定位直径 _ 米": point.robust_diameter,
                    " 相对稳健定位直径": point.relative_diameter,
                    " 定位直径改善 _ 百分比": 100.0
                    * (1.0 - point.relative_diameter),
                    " 对应安全覆盖率": point.safe_coverage,
                    " 对应点 x_ 米": point.point[0],
                    " 对应点 y_ 米": point.point[1],
                    " 是否自动选中": int(
                        abs(point.allowed_loss - result.selected_reliability_loss)
                        <= 1e-12
                    ),
                    " 定位改善数值容差 _ 米": (
                        result.tradeoff_improvement_tolerance
                    ),
                    " 数值误差标定比例": result.numerical_error_ratio,
                    " 是否执行数值加密标定": int(
                        result.numerical_calibration_applied
                    ),
                    " 精验证候选总数": len(result.validated_candidates),
                }
            )
def _configure_chinese_font(plt: Any) -> None:
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
def plot_result(result: Q2Result, output_path: Path) -> None:
    import matplotlib.pyplot as plt
    from scipy.interpolate import griddata
    from scipy.spatial import ConvexHull
    _configure_chinese_font(plt)
    candidates = result.candidates
    x = np.array([candidate.point[0] for candidate in candidates])
    y = np.array([candidate.point[1] for candidate in candidates])
    safe = np.array([candidate.safe_coverage for candidate in candidates])
    validated = result.validated_candidates
    validated_safe = np.array([candidate.safe_coverage for candidate in validated])
    validated_diameter = np.array(
        [float(candidate.validated_robust_diameter) for candidate in validated]
    )
    validated_pareto = np.array(
        [candidate.validated_pareto for candidate in validated], dtype=bool
    )
    figure, axes = plt.subplots(1, 2, figsize=(14, 6))
    display_x = np.linspace(float(x.min()), float(x.max()), 180)
    display_y = np.linspace(float(y.min()), float(y.max()), 180)
    grid_x, grid_y = np.meshgrid(display_x, display_y)
    display_safe = griddata((x, y), safe, (grid_x, grid_y), method="linear")
    missing = ~np.isfinite(display_safe)
    if np.any(missing):
        nearest = griddata((x, y), safe, (grid_x, grid_y), method="nearest")
        display_safe[missing] = nearest[missing]
    safe_min = float(safe.min())
    safe_max = float(safe.max())
    if np.isclose(safe_min, safe_max):
        color_levels = np.linspace(safe_min - 1e-6, safe_max + 1e-6, 3)
    else:
        color_levels = np.linspace(safe_min, safe_max, 13)
    spatial = axes[0].contourf(
        grid_x,
        grid_y,
        display_safe,
        levels=color_levels,
        cmap="Blues",
        alpha=0.88,
    )
    figure.colorbar(
        spatial,
        ax=axes[0],
        label=" 安全覆盖面积比例",
    )
    if safe.min() < result.accepted_safe_coverage_floor < safe.max():
        axes[0].contour(
            grid_x,
            grid_y,
            display_safe,
            levels=[result.accepted_safe_coverage_floor],
            colors=["tab:blue"],
            linewidths=1.8,
        )
    feasible = result.reliability_feasible_candidates
    axes[0].scatter(
        [item.point[0] for item in feasible],
        [item.point[1] for item in feasible],
        s=8,
        facecolors="none",
        edgecolors="tab:blue",
        linewidths=0.35,
        alpha=0.45,
        label=r" 可靠性可行域 $A_{\varepsilon^*}$ 节点",
    )
    axes[0].scatter(
        *result.first_station,
        marker="s",
        s=80,
        color="black",
        label="S1",
    )
    axes[0].scatter(
        *result.best_candidate.point,
        marker="*",
        s=220,
        color="red",
        edgecolor="black",
        label=" 主策略最优 S2",
    )
    near = result.near_optimal_candidates
    near_coordinates = np.asarray([item.point for item in near], dtype=float)
    if len(near_coordinates) >= 3:
        unassigned = set(range(len(near_coordinates)))
        connection_radius = max(5.0 * result.refinement_resolution, 1e-8)
        clusters: list[list[int]] = []
        while unassigned:
            seed = unassigned.pop()
            cluster = [seed]
            frontier = [seed]
            while frontier:
                current = frontier.pop()
                neighbors = [
                    index
                    for index in tuple(unassigned)
                    if np.linalg.norm(
                        near_coordinates[index] - near_coordinates[current]
                    )
                    <= connection_radius
                ]
                for index in neighbors:
                    unassigned.remove(index)
                    cluster.append(index)
                    frontier.append(index)
            clusters.append(cluster)
        region_label_used = False
        for cluster in clusters:
            if len(cluster) < 3:
                continue
            cluster_points = near_coordinates[cluster]
            try:
                hull = ConvexHull(cluster_points)
            except Exception :
                continue
            polygon = cluster_points[hull.vertices]
            axes[0].fill(
                polygon[:, 0],
                polygon[:, 1],
                facecolor="tab:red",
                edgecolor="tab:red",
                alpha=0.13,
                linewidth=1.0,
                label=(r" 近优区域 $C^*$ 数值近似" if not region_label_used else None ),
            )
            region_label_used = True
    axes[0].scatter(
        [item.point[0] for item in near],
        [item.point[1] for item in near],
        facecolors="none",
        edgecolors="red",
        s=45,
        linewidths=0.8,
        label=r" 近优区域 $C^*$ 的精验证节点",
    )
    safe_near = result.safe_near_optimal_candidates
    if safe_near:
        axes[0].scatter(
            [item.point[0] for item in safe_near],
            [item.point[1] for item in safe_near],
            facecolors="none",
            edgecolors="tab:blue",
            marker="s",
            s=55,
            linewidths=0.9,
            label=" 最高可靠覆盖近优区域",
        )
    pareto_best = result.pareto_best_candidate
    axes[0].scatter(
        *pareto_best.point,
        marker="D",
        s=70,
        color="tab:purple",
        edgecolor="black",
        label=" 辅助帕累托折中点",
    )
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_title(
        " 主结论：可靠性可行域、最优点与近优区域 \n"
        f" 自动选择可靠性损失 {100.0 * result.selected_reliability_loss :.2f} 个百分点"
    )
    axes[0].set_xlabel("x / m")
    axes[0].set_ylabel("y / m")
    axes[0].grid(alpha=0.2)
    axes[0].legend(loc="best")
    axes[1].scatter(
        validated_safe[~validated_pareto],
        validated_diameter[~validated_pareto],
        c="lightgray",
        s=25,
        label=" 验证后被支配点",
    )
    pareto_failure = np.array(
        [candidate.certain_failure for candidate in validated]
    )[validated_pareto]
    if len(pareto_failure) and np.ptp(pareto_failure) > 1e-12:
        pareto_scatter = axes[1].scatter(
            validated_safe[validated_pareto],
            validated_diameter[validated_pareto],
            c=pareto_failure,
            cmap="plasma",
            s=38,
            label=" 验证后四目标帕累托点",
        )
        figure.colorbar(
            pareto_scatter, ax=axes[1], label=" 必然失效覆盖比例"
        )
    else:
        constant_failure = float(pareto_failure[0]) if len(pareto_failure) else 0.0
        axes[1].scatter(
            validated_safe[validated_pareto],
            validated_diameter[validated_pareto],
            color="#cf3f72",
            s=38,
            label=(
                " 验证后四目标帕累托点"
                f"（q_fail={constant_failure:.3f}）"
            ),
        )
    axes[1].scatter(
        result.best_candidate.safe_coverage,
        result.best_candidate.validated_robust_diameter,
        marker="*",
        s=220,
        color="red",
        edgecolor="black",
        label=" 主策略最优 S2",
    )
    axes[1].scatter(
        pareto_best.safe_coverage,
        pareto_best.validated_robust_diameter,
        marker="D",
        s=70,
        color="tab:purple",
        edgecolor="black",
        label=" 辅助帕累托折中点",
    )
    axes[1].axvline(
        result.accepted_safe_coverage_floor,
        color="tab:blue",
        linestyle="--",
        linewidth=1.1,
        label=" 主策略可靠性下限",
    )
    axes[1].set_title(" 辅助：精验证后的四目标帕累托二维投影")
    axes[1].set_xlabel(" 安全覆盖率")
    axes[1].set_ylabel(r" 稳健后验直径 $J_{\rm rob}$ / m")
    axes[1].grid(alpha=0.2)
    axes[1].legend(loc="best")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok= True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
def plot_reliability_tradeoff(result: Q2Result, output_path: Path) -> None:
    import matplotlib.pyplot as plt
    _configure_chinese_font(plt)
    points = build_reliability_tradeoff(result)
    x = np.asarray([100.0 * point.allowed_loss for point in points])
    y = np.asarray([point.relative_diameter for point in points])
    selected_loss = 100.0 * result.selected_reliability_loss
    selected_index = int(np.argmin(np.abs(x - selected_loss)))
    figure, axis = plt.subplots(figsize=(8, 5.5))
    axis.step(
        x,
        y,
        where="post",
        color="tab:blue",
        linewidth=2.0,
        label=" 最佳可达相对定位直径",
    )
    axis.scatter(x, y, s=28, color="tab:blue")
    axis.axhline(1.0, color="gray", linestyle="--", linewidth=1.0)
    axis.axvline(
        selected_loss,
        color="tab:red",
        linestyle="--",
        linewidth=1.2,
        label=f" 自动选择 {selected_loss:.2f} 个百分点",
    )
    axis.scatter(
        x[selected_index],
        y[selected_index],
        marker="*",
        s=180,
        color="tab:red",
        edgecolor="black",
        zorder=4,
        label=" 最终决策位置",
    )
    axis.set_title(
        " 可靠性与定位精度权衡曲线 \n"
        f" 基于 {len(result.validated_candidates)} 个分层及局部精验证候选"
    )
    axis.set_xlabel(" 允许安全覆盖率下降 / 百分点")
    axis.set_ylabel(" 相对稳健定位直径（不降低可靠性时 =1）")
    axis.grid(alpha=0.25)
    axis.legend(loc="best")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok= True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
