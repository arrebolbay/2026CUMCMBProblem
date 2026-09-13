"""问题4 检测点网设计：定向安全双环网 + 确定性证书 + 巡游最短化搜索。

理论条件（问题4 的核心）
------------------------
对圆域内任意 G 与任意定向朝向 u，必须存在检测点 p 满足
        |p - G| <= R_min  且  (p - G)·u >= 0
等价条件：G 严格位于集合 { p : |p - G| <= R_min } 的凸包内部
        <=> 近旁点相对 G 的方位角最大角隙 < 180°。

**当前方案（25 点双环网）的解析证书**
    中心 1 点 + 内环 r=950 m ×12 + 外环 r=1875 m ×12（错相 15°）。
      * 外环正十二边形内切半径 1875·cos15° = 1811.11 m > 1800 m
        ⇒ 目标圆域整体含于"中心—内环—外环"三角剖分内部；
      * 该剖分全部边长 < R_min = 1000 m：
        中心—内环 950.0、内环相邻边 491.76、外环相邻边 970.57、交叉边 988.44（最长）。
    ⇒ 域内任意 G 落在某个三角形内，其三顶点都在接收半径内且包围 G
      ⇒ 任意朝向的定向源必被至少一个检测点看到（**严格保证，非抽样**）。

**为什么是双环而不是 31 点三角格点**
    开放巡游长度 ≈ 各环多段线之和 ≈ 2π(r_in + r_out)：
    双环 2π(950+1875) = 17.75 km（实测 2-opt 18.02 km）；
    31 点格点必须铺到 r ≈ 2800 m ⇒ 巡游约 30 km。双环缩短约 40%。

**本脚本做什么**
    1. 用确定性证书 + 蒙特卡洛复核当前方案；
    2. 在双环参数族 (r_in, m_in, r_out, m_out, phase) 中搜索"巡游最短且证书通过"
       的候选，输出对照表，说明当前方案的巡游已接近该族下界。

用法：python scripts/design_p4_survey.py
输出：results/p4_survey_design.json
"""

from __future__ import annotations

import io
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.config import REGION_RADIUS, R_MIN                       # noqa: E402
from src.coverage import (                                        # noqa: E402
    directional_certificate,
    directional_coverage_ok,
    directional_survey_design,
    two_ring_design,
)
from src.strategy_p4 import two_opt_tour                          # noqa: E402


def tour_length(points) -> float:
    """从原点出发的最近邻 + 2-opt 开放巡游长度 (m)。"""
    tour = two_opt_tour([(0.0, 0.0)] + list(points))
    return sum(math.dist(a, b) for a, b in zip(tour, tour[1:]))


def main() -> None:
    current = directional_survey_design()
    cert = directional_certificate(current)
    mc = directional_coverage_ok(current, n_samples=20000, seed=7)
    length = tour_length(current)
    edges = {
        "center_inner": 950.0,
        "inner_edge": 2.0 * 950.0 * math.sin(math.pi / 12.0),
        "outer_edge": 2.0 * 1875.0 * math.sin(math.pi / 12.0),
        "cross_edge": math.sqrt(950.0 ** 2 + 1875.0 ** 2
                                - 2.0 * 950.0 * 1875.0 * math.cos(math.pi / 12.0)),
    }
    print(f"当前方案：{len(current)} 点，巡游 {length / 1000:.2f} km")
    print(f"  确定性证书 ok={cert['ok']} fail={cert['fail']} "
          f"最大角隙 {cert['max_gap_deg']:.2f}°（裕量 {180 - cert['max_gap_deg']:.2f}°）")
    print(f"  蒙特卡洛 20000 次通过率 {mc['ok_ratio']:.4f}")
    print(f"  三角剖分最长边 {max(edges.values()):.2f} m（< {R_MIN:.0f}），"
          f"外环内切半径 {1875.0 * math.cos(math.pi / 12.0):.2f} m（> {REGION_RADIUS:.0f}）")

    # -------- 双环参数族搜索：巡游最短且证书通过 --------
    candidates = []
    for r_in in range(300, 1001, 50):
        for m_in in (6, 8, 10, 12):
            for r_out in range(1800, 1951, 25):
                for m_out in (12, 14, 16):
                    for phase in (0.0, 0.5):
                        pts = two_ring_design(r_in, m_in, r_out, m_out, phase)
                        # 先用中等密度证书筛（快速），通过的才计算巡游长度
                        quick = directional_certificate(
                            pts, n_radial=25, n_angular=72, n_boundary=720)
                        if not quick["ok"]:
                            continue
                        dense = directional_certificate(pts)
                        if not dense["ok"]:
                            continue
                        candidates.append({
                            "inner_radius": float(r_in), "inner_count": m_in,
                            "outer_radius": float(r_out), "outer_count": m_out,
                            "phase": phase, "n_points": len(pts),
                            "tour_m": tour_length(pts),
                            "max_gap_deg": dense["max_gap_deg"],
                        })
    candidates.sort(key=lambda c: c["tour_m"])
    print(f"\n双环族中通过确定性证书的构型共 {len(candidates)} 个；巡游最短 5 个：")
    for c in candidates[:5]:
        print(f"  巡游 {c['tour_m'] / 1000:6.2f} km  最大角隙 {c['max_gap_deg']:6.2f}°  "
              f"点数 {c['n_points']:2d}  r_in={c['inner_radius']:.0f}×{c['inner_count']}  "
              f"r_out={c['outer_radius']:.0f}×{c['outer_count']}  phase={c['phase']}")

    best = candidates[0] if candidates else None
    if best:
        gain = length - best["tour_m"]
        print(f"\n结论：最优候选比现方案省 {gain / 1000:.2f} km "
              f"({100.0 * gain / length:.1f}%)，但角隙裕量 "
              f"{180 - best['max_gap_deg']:.2f}° vs 现方案 {180 - cert['max_gap_deg']:.2f}°；"
              f"收益不足 3% 时保留现方案。")

    out = os.path.join(ROOT, "results", "p4_survey_design.json")
    io.open(out, "w", encoding="utf-8").write(json.dumps({
        "current": {
            "n_points": len(current), "tour_m": length,
            "certificate": cert, "directional_pass_ratio": mc["ok_ratio"],
            "triangulation_longest_edge_m": max(edges.values()),
            "outer_inradius_m": 1875.0 * math.cos(math.pi / 12.0),
            "points": current,
        },
        "feasible_two_ring_candidates": len(candidates),
        "shortest_candidates": candidates[:10],
    }, ensure_ascii=False, indent=2))
    print("\n已保存:", out)


if __name__ == "__main__":
    main()
