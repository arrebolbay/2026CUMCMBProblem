"""生成论文图件（results/figs/*.png）。

包含：
  fig1_problem1.png  问题1：两站交会定位区域 + 直径圆 + 最小覆盖圆；锐角三角形反例
  fig2_problem2.png  问题2：面积/det(FIM) 关于基线 b 的曲线（最优点 b*=d1，theta*=45°）+ 蝶形候选区
  fig3_problem3.png  问题3：经典 7 点覆盖网（说明"最少 7 点"的下界）
  fig3b_problem3_scheme.png  问题3 **实际运行方案**：中心+14 点加密环（8 个全频道扫描点 + 7 个几何复测点）与真实巡游路径
  fig4_problem4.png  问题4：7 点方案定向失败 vs 25 点定向安全双环网成功

中文显示：优先使用微软雅黑，缺失时自动回退。
"""

from __future__ import annotations

import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from matplotlib.figure import Figure

# Windows 下保存 png 偶发 OSError(22)（杀软扫描 / 图片预览器占用），加重试更稳。
_orig_savefig = Figure.savefig


def _retrying_savefig(self, fname, *args, **kwargs):
    import time

    last = None
    for _ in range(6):
        try:
            return _orig_savefig(self, fname, *args, **kwargs)
        except OSError as exc:          # pragma: no cover - 环境相关
            last = exc
            time.sleep(0.8)
    raise last


Figure.savefig = _retrying_savefig



from src.config import EPS_DEG, REGION_RADIUS, R_MIN
from src.coverage import (
    directional_coverage_ok,
    directional_survey_design,
    seven_point_design,
)
from src.geometry import (
    bearing_between,
    diameter_circle_covers,
    localization_region,
    unit_vector,
    wedge_polygon,
)

FIG_DIR = os.path.join(ROOT, "results", "figs")
os.makedirs(FIG_DIR, exist_ok=True)

for font in ("Microsoft YaHei", "SimHei", "DejaVu Sans"):
    plt.rcParams["font.sans-serif"] = [font]
    plt.rcParams["font.family"] = "sans-serif"
    break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130


def circle(ax, center, radius, **kw):
    th = np.linspace(0, 2 * math.pi, 400)
    ax.plot(center[0] + radius * np.cos(th), center[1] + radius * np.sin(th), **kw)


def fig_problem1():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 5.2))
    jammer = (620.0, 430.0)
    s1, s2 = (0.0, 0.0), (900.0, -200.0)
    b1, b2 = bearing_between(s1, jammer), bearing_between(s2, jammer)
    poly = localization_region([s1, s2], [b1, b2])
    info = diameter_circle_covers(poly)

    for s, b, name in ((s1, b1, "S1"), (s2, b2, "S2")):
        tri = wedge_polygon(s, b, EPS_DEG, radius=1600.0)
        ax1.fill([p[0] for p in tri], [p[1] for p in tri], color="tab:blue", alpha=0.12)
        for sgn in (-1, 1):
            ux, uy = unit_vector(b + sgn * EPS_DEG)
            ax1.plot([s[0], s[0] + 1600 * ux], [s[1], s[1] + 1600 * uy],
                     color="tab:blue", lw=0.8)
        ax1.plot(*s, "ko")
        ax1.annotate(name, s, textcoords="offset points", xytext=(6, 6))
    ax1.fill([p[0] for p in poly], [p[1] for p in poly], color="tab:red", alpha=0.35,
             label="定位区域 P（楔形交）")
    circle(ax1, info["circle_center"], info["circle_radius"], color="tab:red",
           ls="--", lw=1.2, label="以直径为直径的圆（覆盖）")
    circle(ax1, info["mec_center"], info["mec_radius"], color="tab:green", ls=":",
           lw=1.2, label="最小覆盖圆")
    ax1.plot(*jammer, "r*", ms=12, label="干扰源真值")
    ax1.set_title("问题1：两站交会定位区域（平行四边形必被直径圆覆盖）")
    ax1.legend(fontsize=8, loc="upper left")
    ax1.set_aspect("equal")
    ax1.grid(alpha=0.3)

    acute = [(0.0, 0.0), (1.0, 0.0), (0.5, math.sqrt(3) / 2)]
    info2 = diameter_circle_covers(acute)
    ax2.fill([p[0] for p in acute], [p[1] for p in acute], color="tab:orange", alpha=0.35,
             label="锐角三角形定位区域")
    circle(ax2, info2["circle_center"], info2["circle_radius"], color="tab:red", ls="--",
           label="直径圆（覆盖失败）")
    circle(ax2, info2["mec_center"], info2["mec_radius"], color="tab:green", ls=":",
           label="最小覆盖圆（可覆盖）")
    ax2.set_title("问题1：锐角三角形反例（直径圆不能覆盖）")
    ax2.legend(fontsize=8, loc="upper right")
    ax2.set_aspect("equal")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig1_problem1.png")
    fig.savefig(out)
    plt.close(fig)
    return out

def fig_problem2():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.8))
    d1 = 1000.0
    eps = math.radians(EPS_DEG)
    bs = np.linspace(0.2 * d1, 3.0 * d1, 800)
    d2 = np.sqrt(d1 ** 2 + bs ** 2)
    sin_th = bs / d2
    area = 4.0 * (math.tan(eps) ** 2) * d1 * d2 / sin_th
    det_fim = (sin_th / (d1 * d2)) ** 2
    ax1.plot(bs / d1, area / area.min(), label="定位区域面积（归一化）")
    ax1.plot(bs / d1, det_fim / det_fim.max(), label="det(FIM)（归一化）")
    ax1.axvline(1.0, color="tab:red", ls="--", lw=1.0)
    ax1.annotate("最优基线 b* = d1\n（交会角 45°）", xy=(1.0, 0.55),
                 xytext=(1.35, 0.45), arrowprops=dict(arrowstyle="->", lw=0.9), fontsize=8)
    ax1.set_xlabel("垂直基线 b / d1")
    ax1.set_ylabel("归一化指标")
    ax1.set_title("问题2：面积与 Fisher 信息的最优基线一致（b*=d1）")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    s1 = (0.0, 0.0)
    b1 = 30.0
    ux, uy = unit_vector(b1)
    ax2.plot([0, 1800 * ux], [0, 1800 * uy], color="tab:blue", lw=1.0, label="示向度方向")
    for sgn in (-1, 1):
        nx, ny = unit_vector(b1 + sgn * 90.0)
        for frac in (0.5, 1.0, 1.5):
            cx, cy = frac * d1 * nx, frac * d1 * ny
            ax2.plot([cx - 350 * ux, cx + 350 * ux], [cy - 350 * uy, cy + 350 * uy],
                     color="tab:orange", lw=0.8, alpha=0.9)
    for sgn in (-1, 1):
        nx, ny = unit_vector(b1 + sgn * 90.0)
        ax2.plot([0], [0], marker="o", color="k")
        ax2.annotate("S1", (0, 0), textcoords="offset points", xytext=(6, 4))
        ax2.annotate("候选区域（蝶形）" if sgn > 0 else "",
                     (d1 * nx, d1 * ny), textcoords="offset points", xytext=(6, 6), fontsize=8)
    g = (d1 * ux, d1 * uy)
    ax2.plot(*g, "r*", ms=10, label="目标（未知距离 d1）")
    ax2.plot([0, d1 * unit_vector(b1 + 90.0)[0]],
             [0, d1 * unit_vector(b1 + 90.0)[1]], "k:", lw=0.8)
    ax2.set_title("问题2：第二检测点候选区域（示向度两侧垂直带）")
    ax2.legend(fontsize=8, loc="lower right")
    ax2.set_aspect("equal")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig2_problem2.png")
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_problem3():
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    circle(ax, (0, 0), REGION_RADIUS, color="k", lw=1.4, label="目标区域 R=1800 m")
    pts = seven_point_design(1400.0, 6)
    for p in pts:
        circle(ax, p, R_MIN, color="tab:blue", lw=0.4, alpha=0.35)
    ax.plot([p[0] for p in pts], [p[1] for p in pts], "o", color="tab:red", ms=6,
            label="7 点检测网（中心+环1400m）")
    from src.coverage import worst_case_radius
    wc = worst_case_radius(pts, REGION_RADIUS, 6000, 6000)
    ax.set_title("问题3：7 点覆盖网（6 点不可行，7 点最优）\n"
                 "实测最坏覆盖距离 %.1f m < %.0f m" % (wc, R_MIN), fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig3_problem3.png")
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_problem4():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 6.0))
    # 左：7 点方案在定向情形失败
    circle(ax1, (0, 0), REGION_RADIUS, color="k", lw=1.2)
    pts7 = seven_point_design(1400.0, 6)
    ax1.plot([p[0] for p in pts7], [p[1] for p in pts7], "o", color="tab:red", ms=5,
             label="7 点网")
    g = (1750.0, 0.0)
    for p in pts7:
        if math.dist(p, g) <= R_MIN:
            circle(ax1, p, R_MIN, color="tab:blue", lw=0.4, alpha=0.3)
    ax1.plot(*g, "k*", ms=13, label="定向源 G")
    ux, uy = unit_vector(180.0)          # 背向唯一检测点的朝向
    ax1.annotate("", xy=(g[0] + 500 * ux, g[1] + 500 * uy), xytext=g,
                 arrowprops=dict(arrowstyle="-|>", lw=1.4, color="tab:purple"))
    ax1.annotate("定向方向（背向检测点）\n=> 所有点都 no_signal", xy=(g[0] - 420, g[1]),
                 fontsize=8, color="tab:purple")
    r7 = directional_coverage_ok(pts7, n_samples=2000, seed=3)
    ax1.set_title("问题4：7 点方案定向覆盖通过率 %.1f%%（不可用）" % (100 * r7["ok_ratio"]),
                  fontsize=10)
    ax1.legend(fontsize=8, loc="upper left")
    ax1.set_aspect("equal")
    ax1.grid(alpha=0.3)

    # 右：25 点定向安全双环网满足定向覆盖
    circle(ax2, (0, 0), REGION_RADIUS, color="k", lw=1.2, label="目标区域")
    design = directional_survey_design()
    ax2.plot([p[0] for p in design], [p[1] for p in design], "^", color="tab:red", ms=4.5,
             label="25 点定向安全双环网（r=950/1875 m，证书 fail=0）")
    g2 = (500.0, 300.0)
    near = [p for p in design if math.dist(p, g2) <= R_MIN]
    for p in near:
        circle(ax2, p, R_MIN, color="tab:blue", lw=0.35, alpha=0.25)
    for p in near:
        ax2.plot([p[0], g2[0]], [p[1], g2[1]], color="tab:green", lw=0.7, alpha=0.8)
    ax2.plot(*g2, "k*", ms=13, label="定向源 G（任取）")
    r4 = directional_coverage_ok(design, n_samples=4000, seed=3)
    ax2.set_title("问题4：双环网定向覆盖抽样通过率 %.1f%%（近旁 %d 点包围 G）"
                  % (100 * r4["ok_ratio"], len(near)), fontsize=10)
    ax2.legend(fontsize=8, loc="upper left")
    ax2.set_aspect("equal")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig4_problem4.png")
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_problem3_scheme():
    """问题3 **当前运行方案**示意图：覆盖骨干 + 几何复测点 + 实际巡游。

    与 fig3（经典"7 点覆盖网"，说明最少 7 点的下界结论）配套保留：
    fig3 讲下界，本图讲实际落地方案（中心 + 14 点加密环，其中 8 个全频道
    扫描点承担"不漏测"保证，另 7 个只复测"进行中"频道以改善交会几何）。
    """
    from src.coverage import seven_point_design, worst_case_radius
    from src.routing import tour_length, two_opt_tour
    from src.strategy_p3 import StrategyP3

    strat = StrategyP3()
    ring = seven_point_design(strat.cover_ring_radius, strat.cover_ring_count)
    sweep = [ring[i] for i in range(0, len(ring), strat.cover_probe_stride)]
    sweep_set = set(sweep)
    probes = [p for p in ring[1:] if p not in sweep_set]
    tour = two_opt_tour(ring)
    tl = tour_length(tour)
    worst = worst_case_radius(sweep, REGION_RADIUS, 6000, 6000)

    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    circle(ax, (0, 0), REGION_RADIUS, color="k", lw=1.4, label="目标区域 R=1800 m")
    for p in sweep:
        circle(ax, p, R_MIN, color="tab:red", lw=0.45, alpha=0.30)
    ax.plot([p[0] for p in tour], [p[1] for p in tour], "-", color="0.55", lw=1.0,
            zorder=1, label="实际巡游 %.2f km" % (tl / 1000.0))
    ax.plot([p[0] for p in probes], [p[1] for p in probes], "o", color="tab:blue",
            ms=6.5, zorder=4, label="几何复测点 ×7（只测进行中频道）")
    ax.plot([p[0] for p in sweep], [p[1] for p in sweep], "*", color="tab:red",
            ms=13, zorder=5, label="全频道扫描点 ×8（中心 + 7，保证不漏测）")
    ax.annotate("中心 (0,0)", xy=(0, 0), xytext=(120, -260), fontsize=8,
                arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.annotate("环半径 1000 m", xy=(707, 707), xytext=(950, 1250), fontsize=8,
                arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.set_title("问题3 运行方案：中心 + 14 点加密环\n"
                 "覆盖保证只依赖 8 个扫描点：解析最坏 %.2f m < %.0f m（实测 %.1f m）"
                 % (998.2545, R_MIN, worst), fontsize=9.5)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig3b_problem3_scheme.png")
    fig.savefig(out)
    plt.close(fig)
    return out


def main():
    for fn in (fig_problem1, fig_problem2, fig_problem3, fig_problem3_scheme,
               fig_problem4):
        print("已保存:", fn())


if __name__ == "__main__":
    main()
