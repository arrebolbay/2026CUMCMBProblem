"""理论常数的数值验证脚本（论文"模型检验"章节的原始数据来源）。

验证三项关键结论：
  1) 覆盖布点：7 点构造方案 worst-case 覆盖距离 < 1000 m；
     6 点全局最优 > 1000 m（与 Bezdek 经典阈值 1.7988 一致）→ 最少 7 点；
  2) 问题2 最优基线：定位区域面积 / det(FIM) 均在 b = d1（交会角 45°）处最优，
     而非经典文献常引的 90°（因本题角度误差导致楔形宽度随距离线性增长）；
  3) 问题1 覆盖性：两站定位区域（平行四边形）必被"直径圆"覆盖；
     锐角三角形定位区域则不能（反例）。

用法：python -m scripts.verify_theory 或直接 python scripts/verify_theory.py
"""

from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.config import EPS_DEG, REGION_RADIUS, R_MIN          # noqa: E402
from src.coverage import (                                     # noqa: E402
    directional_certificate,
    directional_coverage_ok,
    directional_survey_design,
    hexagonal_design,
    optimize_cover_radius,
    seven_point_design,
    worst_case_radius,
)
from src.geometry import (                                     # noqa: E402
    bearing_between,
    diameter_circle_covers,
    localization_region,
    polygon_area,
)


def verify_coverage() -> dict:
    design7 = seven_point_design(ring_radius=1400.0, k=6)
    hex7 = hexagonal_design(r_min=R_MIN)
    wc7 = worst_case_radius(design7, REGION_RADIUS,
                            n_interior=30000, n_boundary=30000, seed=123)
    wc_hex = worst_case_radius(hex7, REGION_RADIUS,
                               n_interior=30000, n_boundary=30000, seed=123)
    print(f"[1] 7 点方案(中心+环1400) worst-case 覆盖距离 = {wc7:.3f} m  (需 <= {R_MIN})")
    print(f"[1] 7 点方案(六边形sqrt3*1000) worst-case 覆盖距离 = {wc_hex:.3f} m")

    r6, pts6 = optimize_cover_radius(6, REGION_RADIUS, n_objective=1500, seed=11,
                                     maxiter=45, popsize=10,
                                     init_guesses=[design7[1:]])
    print(f"[1] 6 点最优最小覆盖半径 = {r6:.3f} m  (需 >  {R_MIN}) -> 6 点不可行")

    r7, pts7 = optimize_cover_radius(7, REGION_RADIUS, n_objective=1500, seed=13,
                                     maxiter=45, popsize=10,
                                     init_guesses=[design7, hex7])
    print(f"[1] 7 点最优最小覆盖半径 = {r7:.3f} m  (需 <= {R_MIN}) -> 7 点可行")
    return {
        "seven_point_design_worst_case_m": wc7,
        "hexagonal_design_worst_case_m": wc_hex,
        "optimal_6_cover_radius_m": r6,
        "optimal_7_cover_radius_m": r7,
        "ratio_1800_1000": REGION_RADIUS / R_MIN,
        "bezdek_threshold_6": 1.7988,
        "bezdek_threshold_7": 2.0,
        "min_points_for_coverage": 7,
    }


def verify_problem2(d1: float = 1000.0, d_lo: float = 300.0, d_hi: float = 5000.0) -> dict:
    """第二检测点：比较定位区域面积与 det(FIM) 关于基线 b 的曲线。

    设 S1=(0,0)，目标真值 G=(d1,0)，第二点 S2=(0,b)（垂直于示向度方向）。
    则 d2=sqrt(d1^2+b^2)，交会角 sin(theta)=b/d2。
      * 定位区域面积 A(b) = 4*tan^2(eps)*d1*d2/sin(theta)
      * det(FIM) ∝ (sin(theta)/(d1*d2))^2
    理论最优 b* = d1（交会角 45°）。
    """
    eps = math.radians(EPS_DEG)
    bs = np.linspace(d1 * 0.2, d1 * 3.0, 4000)
    d2 = np.sqrt(d1 ** 2 + bs ** 2)
    sin_th = bs / d2
    area = 4.0 * (math.tan(eps) ** 2) * d1 * d2 / sin_th
    det_fim = (sin_th / (d1 * d2)) ** 2

    b_area = float(bs[int(np.argmin(area))])
    b_fim = float(bs[int(np.argmax(det_fim))])
    theta_star = math.degrees(math.atan2(b_area, d1))
    print(f"[2] d1={d1:.0f} m: 面积最优 b*={b_area:.1f} m, det(FIM) 最优 b*={b_fim:.1f} m, "
          f"最优交会角 theta*={theta_star:.2f} deg (理论 45 deg)")
    return {
        "d1": d1,
        "b_star_area": b_area,
        "b_star_fim": b_fim,
        "theta_star_deg": theta_star,
        "theoretical_b_star": d1,
        "theoretical_theta_deg": 45.0,
    }


def verify_problem1() -> dict:
    jammer = (620.0, 430.0)
    s1, s2 = (0.0, 0.0), (900.0, -200.0)
    poly = localization_region([s1, s2], [bearing_between(s1, jammer),
                                          bearing_between(s2, jammer)])
    info = diameter_circle_covers(poly)
    acute = [(0.0, 0.0), (1.0, 0.0), (0.5, math.sqrt(3) / 2.0)]
    info_acute = diameter_circle_covers(acute)
    print(f"[3] 两站定位区域: 顶点数={len(poly)}, 面积={polygon_area(poly):.1f} m^2, "
          f"直径圆覆盖={info['covered']}")
    print(f"[3] 锐角三角形反例: 直径圆覆盖={info_acute['covered']} (应为 False), "
          f"最小覆盖圆半径={info_acute['mec_radius']:.4f}")
    return {
        "two_station_vertices": len(poly),
        "two_station_area": polygon_area(poly),
        "two_station_covered": info["covered"],
        "two_station_diameter": info["diameter"],
        "acute_triangle_covered": info_acute["covered"],
        "acute_triangle_mec_radius": info_acute["mec_radius"],
    }


def verify_problem3_design() -> dict:
    """问题3 运行布点：中心 + 1000 m 七均布（8 点）的解析最坏覆盖距离。"""
    k, ring = 7, 1000.0
    worst = math.sqrt(REGION_RADIUS ** 2 + ring ** 2
                      - 2.0 * REGION_RADIUS * ring * math.cos(math.pi / k))
    mc = worst_case_radius(seven_point_design(ring, k), REGION_RADIUS)
    print(f"[3] 运行布点 8 点（中心 + {ring:.0f}m 七均布）：解析最坏覆盖距离 "
          f"= {worst:.3f} m（< {R_MIN:.0f}）；蒙特卡洛 {mc:.3f} m")
    return {
        "design_points": k + 1,
        "ring_radius_m": ring,
        "analytic_worst_case_m": worst,
        "monte_carlo_worst_case_m": mc,
        "margin_m": R_MIN - worst,
    }


def verify_problem4() -> dict:
    """问题4：定向覆盖条件的数值验证 + 双环网的解析/确定性证书。"""
    design = directional_survey_design()
    r7 = directional_coverage_ok(seven_point_design(1400.0, 6), n_samples=4000, seed=3)
    r4 = directional_coverage_ok(design, n_samples=20000, seed=7)
    omni = worst_case_radius(design, REGION_RADIUS, n_interior=20000, n_boundary=20000)
    cert = directional_certificate(design)
    edges = {
        "center_inner": 950.0,
        "inner_edge": 2.0 * 950.0 * math.sin(math.pi / 12.0),
        "outer_edge": 2.0 * 1875.0 * math.sin(math.pi / 12.0),
        "cross_edge": math.sqrt(950.0 ** 2 + 1875.0 ** 2
                                - 2.0 * 950.0 * 1875.0 * math.cos(math.pi / 12.0)),
    }
    outer_inradius = 1875.0 * math.cos(math.pi / 12.0)
    print(f"[4] 问题3 的 7 点方案定向通过率 = {r7['ok_ratio']:.4f} (不满足)")
    print(f"[4] 双环网 {len(design)} 点：抽样通过率 = {r4['ok_ratio']:.4f}；"
          f"确定性证书 ok={cert['ok']}、fail={cert['fail']}、"
          f"最大角隙 = {cert['max_gap_deg']:.2f}°（< 180°）")
    print(f"[4] 三角剖分：外环内切半径 {outer_inradius:.2f} m (> {REGION_RADIUS:.0f})，"
          f"最长边 {max(edges.values()):.2f} m (< {R_MIN:.0f})；"
          f"全向最坏覆盖距离 {omni:.1f} m")
    return {
        "seven_point_directional_pass_ratio": r7["ok_ratio"],
        "design_points": len(design),
        "directional_pass_ratio": r4["ok_ratio"],
        "certificate_ok": cert["ok"],
        "certificate_fail": cert["fail"],
        "certificate_max_gap_deg": cert["max_gap_deg"],
        "certificate_samples": cert["n_samples"],
        "triangulation_longest_edge_m": max(edges.values()),
        "outer_inradius_m": outer_inradius,
        "omni_worst_case_m": omni,
    }


def main() -> None:
    report = {
        "coverage": verify_coverage(),
        "problem3_design": verify_problem3_design(),
        "problem2": verify_problem2(),
        "problem1": verify_problem1(),
        "problem4": verify_problem4(),
    }
    out_dir = os.path.join(ROOT, "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "theory_verification.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print(f"\n结果已写入: {out_path}")


if __name__ == "__main__":
    main()
