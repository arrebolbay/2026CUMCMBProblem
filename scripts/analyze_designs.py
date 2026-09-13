"""问题1/2/3 的定量分析与设计优化验证。

一、布站准则对照（回应"45° vs 54.74°"）
    模型：S1=(0,0)，目标 G=(d1,0)，第二点 S2=(0,b)（垂直于示向度），
    则 d2=sqrt(d1²+b²)，交会角 theta=arctan(b/d1)。
    设第 i 个测向的**横向误差** e_i = c·d_i^α：
      * α=1  ：纯角度误差（本题附件给的有界 ±1°，横向误差 = d·tanε）→ b*=d1，θ*=45°
      * α=0.5：角度误差部分随距离加权 → b*=√2·d1，θ*=54.74°
      * α=2  ：强距离加权 → b*=d1/√2，θ*=35.26°
    推导：定位区域面积 A ∝ e1·e2/sinθ = c²·d1^α·d2^(α+1)/b，
    对 b 求导得 b* = d1/√α ⇒ θ* = arctan(1/√α)。**两个数字同属一个族**，
    只是横向误差模型不同；本题附件明确"误差范围 ±1°"（有界）⇒ 取 α=1、b*=d1。
    再做极小极大（α 不确定）与机动成本两种稳健化，给出推荐区间。

二、问题1：蒙特卡洛验证"直径圆覆盖定位区域"的失败率（论文表格）
三、问题3：覆盖环设计优化（回应"把交点放到 1800 圆周上、让环更靠近圆心"）
"""

from __future__ import annotations

import json
import math
import os
import random
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.config import EPS_DEG, REGION_RADIUS, R_MIN                # noqa: E402
from src.coverage import seven_point_design, worst_case_radius       # noqa: E402
from src.geometry import (                                           # noqa: E402
    bearing_between,
    diameter_circle_covers,
    localization_region,
)
from src.routing import tour_length, two_opt_tour                    # noqa: E402

D1 = 1000.0
EPS = math.radians(EPS_DEG)


def area_factor(b: float, d1: float = D1, alpha: float = 1.0) -> float:
    """定位区域面积的比例因子（去掉与 ε 无关的常数）：d1^α·d2^(α+1)/b。"""
    d2 = math.hypot(d1, b)
    return (d1 ** alpha) * (d2 ** (alpha + 1.0)) / b


def optimal_b_analytic(d1: float = D1, alpha: float = 1.0) -> float:
    """解析最优基线 b* = d1/√α（α<=0 表示纯横向误差，随 b 单调变优 → 退化）。"""
    if alpha <= 0.0:
        return float("inf")
    return d1 / math.sqrt(alpha)


def optimal_b_numeric(d1: float = D1, alpha: float = 1.0,
                      b_hi: float = 8000.0) -> float:
    bs = np.linspace(d1 * 0.05, b_hi, 40000)
    vals = np.array([area_factor(float(b), d1, alpha) for b in bs])
    return float(bs[int(np.argmin(vals))])


def analyse_criteria() -> dict:
    alphas = (0.5, 2.0 / 3.0, 0.8, 1.0, 1.25, 1.5, 2.0)
    rows = []
    for a in alphas:
        b_an = optimal_b_analytic(D1, a)
        b_num = optimal_b_numeric(D1, a)
        rows.append({"alpha": a, "b_star": b_an, "b_star_numeric": b_num,
                     "theta_star_deg": math.degrees(math.atan2(b_an, D1)),
                     "numeric_error_m": abs(b_an - b_num)})
    print("=== 布站准则对照（横向误差 e ∝ d^α）===")
    print(" alpha | 解析 b* | 数值 b* | θ* (deg) | 数值误差(m)")
    for r in rows:
        print(f" {r['alpha']:5.2f} | {r['b_star']:7.1f} | {r['b_star_numeric']:7.1f} | "
              f"{r['theta_star_deg']:8.2f} | {r['numeric_error_m']:8.1f}")
    key = {a: next(r for r in rows if abs(r["alpha"] - a) < 1e-9) for a in (0.5, 1.0)}
    print(f"\n  alpha=1.0（本题：有界 ±1°） → b*={key[1.0]['b_star']:.1f} m, "
          f"θ*={key[1.0]['theta_star_deg']:.2f}°  （本方案采用）")
    print(f"  alpha=0.5（误差随距离加权） → b*={key[0.5]['b_star']:.1f} m, "
          f"θ*={key[0.5]['theta_star_deg']:.2f}°  （= √2·d1，与参考思路一致）")

    print("\n  面积惩罚（相对各自最优，1.000 为最优）：")
    print("   设计\\真值 " + " ".join(f"α={a:<4.2f}" for a in (0.5, 1.0, 2.0)))
    penalty = {}
    for design_alpha in (0.5, 1.0, 2.0):
        b_design = optimal_b_analytic(D1, design_alpha)
        row = [area_factor(b_design, D1, t)
               / area_factor(optimal_b_analytic(D1, t), D1, t)
               for t in (0.5, 1.0, 2.0)]
        penalty[design_alpha] = row
        print(f"   α={design_alpha:<4.2f}   " + " ".join(f"{v:6.3f}" for v in row))

    cand = np.linspace(0.5 * D1, 2.2 * D1, 3000)
    worst = [max(area_factor(float(b), D1, a)
                 / area_factor(optimal_b_analytic(D1, a), D1, a)
                 for a in (0.5, 0.75, 1.0, 1.25, 1.5)) for b in cand]
    i = int(np.argmin(worst))
    b_minimax, worst_minimax = float(cand[i]), float(worst[i])
    print(f"\n  极小极大稳健设计（α∈[0.5,1.5]）：b={b_minimax:.1f} m "
          f"({b_minimax / D1:.2f}·d1)，θ={math.degrees(math.atan2(b_minimax, D1)):.2f}°，"
          f"最坏面积惩罚={worst_minimax:.3f}")

    print("\n  计入机动成本（总代价 = 归一面积惩罚 + w·b/d1）：")
    base = area_factor(optimal_b_analytic(D1, 1.0), D1, 1.0)
    move_rows = []
    for w in (0.0, 0.02, 0.05, 0.10, 0.20):
        vals = [area_factor(float(b), D1, 1.0) / base + w * (b / D1) for b in cand]
        b_star = float(cand[int(np.argmin(vals))])
        move_rows.append({"w_move": w, "b_star": b_star,
                          "theta_star_deg": math.degrees(math.atan2(b_star, D1)),
                          "area_penalty_vs_pure": area_factor(b_star, D1, 1.0) / base})
        print(f"   w={w:5.2f}: b*={b_star:7.1f} m, "
              f"θ*={move_rows[-1]['theta_star_deg']:5.2f}°, "
              f"面积惩罚={move_rows[-1]['area_penalty_vs_pure']:.3f}")
    return {"alpha_family": rows, "penalty": penalty,
            "minimax": {"b": b_minimax, "penalty": worst_minimax,
                        "theta_deg": math.degrees(math.atan2(b_minimax, D1))},
            "move_cost": move_rows}


def monte_carlo_problem1(n_cases: int = 5000, seed: int = 7) -> dict:
    """问题1 蒙特卡洛：不同站数下"以区域直径为直径的圆"能否覆盖定位区域。

    采样设定与题目一致：站点与目标都落在半径 1800 m 的目标区域内，
    示向度误差 ±1° 均匀分布。统计三类结果：
      * **空区域**：±1° 楔形互不相交（真值仍在内，但交会失败）→ 必须增加观测站，
        这正是问题3/4 采用"三观测/多站交会"的直接依据；
      * **直径圆覆盖**：能否用"区域直径为直径的圆"覆盖整个定位区域；
      * 2 站时由平行四边形定理**必然覆盖**（对角线互相平分）；
        ≥3 站不保证（锐角三角形反例），失败时的半径超出量是米级。
    """
    rng = random.Random(seed)
    out = {}
    for k in (2, 3, 4):
        covered = empty = total = 0
        excess = []
        for _ in range(n_cases):
            stations = []
            for _ in range(k):
                r = 1800.0 * math.sqrt(rng.random())
                th = rng.random() * 2.0 * math.pi
                stations.append((r * math.cos(th), r * math.sin(th)))
            r = 1800.0 * math.sqrt(rng.random())
            th = rng.random() * 2.0 * math.pi
            target = (r * math.cos(th), r * math.sin(th))
            bearings = []
            skip = False
            for s in stations:
                if math.dist(s, target) < 1.0:
                    skip = True
                    break
                bearings.append(bearing_between(s, target)
                                + rng.uniform(-EPS_DEG, EPS_DEG))
            if skip:
                continue
            total += 1
            poly = localization_region(stations, bearings)
            if not poly or len(poly) < 3:
                empty += 1
                continue
            info = diameter_circle_covers(poly)
            if info["covered"]:
                covered += 1
            else:
                excess.append(info["mec_radius"] - info["circle_radius"])
        valid = total - empty
        rate = covered / valid if valid else float("nan")
        out[k] = {"cases": total, "empty": empty,
                  "empty_ratio": empty / total if total else float("nan"),
                  "coverage_ratio": rate, "fails": valid - covered,
                  "excess_median_m": float(np.median(excess)) if excess else 0.0,
                  "excess_max_m": max(excess) if excess else 0.0}
        print(f"  站数 K={k}: 空区域率 {out[k]['empty_ratio']:.4f}；"
              f"直径圆覆盖率 {rate:.4f}（失败 {out[k]['fails']}/{valid}）；"
              f"半径超出量中位 {out[k]['excess_median_m']:.4f} m "
              f"最大 {out[k]['excess_max_m']:.4f} m")
    print("  结论：2 站（平行四边形）必然覆盖；≥3 站不保证，失效构型的半径超出量为米级；")
    print("        空区域的存在说明『两站交会』在 ±1° 误差下并不总可用，多站/多次观测是必要的。")
    return out


def ring_radius_min(k: int, region_radius: float = REGION_RADIUS,
                    r_min: float = R_MIN) -> float:
    """k 个均布环点 + 中心点覆盖圆域所需的**最小环半径**（解析）。

    约束：边界"角平分点"恰好落在覆盖圆周上（即相邻两覆盖圆的交点正好在 R 圆周上）：
        sqrt(R² + a² − 2Ra·cos(π/k)) = r_min
    ⇒ a² − 2R·cos(π/k)·a + (R² − r_min²) = 0，取较小根。
    这就是"把交点放到 1800 圆周上"的精确数学表达。
    """
    c = math.cos(math.pi / k)
    disc = (region_radius * c) ** 2 - (region_radius ** 2 - r_min ** 2)
    return region_radius * c - math.sqrt(max(disc, 0.0))


def analyse_p3_ring() -> dict:
    print("\n=== 问题3 覆盖环：点数 k 与环半径/巡游/检测代价 ===")
    print("  k | 最小环半径 a_min(m) | 数值复核最坏最近距离 | 2-opt巡游(km) |"
          " 全频道扫描点数 | N=10代价(s) | N=16代价(s)")
    rows = []
    for k in (6, 7, 8, 10, 12, 14, 16, 20):
        a = ring_radius_min(k)
        design = seven_point_design(a, k)
        wc = worst_case_radius(design, REGION_RADIUS, 6000, 6000)
        tour = tour_length(two_opt_tour(design))
        sweeps = k + 1                      # 中心 + k 个环点都要做全频道扫描
        row = {"k": k, "a_min": a, "worst_case_m": wc, "tour_m": tour,
               "sweeps": sweeps}
        for n in (10, 16):
            row[f"cost_N{n}"] = tour / 5.0 + sweeps * (20 - n) * 6.0
        rows.append(row)
        print(f" {k:2d} | {a:16.1f} | {wc:19.1f} | {tour / 1000:11.2f} |"
              f" {sweeps:12d} | {row['cost_N10']:12.0f} | {row['cost_N16']:12.0f}")
    best10 = min(rows, key=lambda r: r["cost_N10"])
    best16 = min(rows, key=lambda r: r["cost_N16"])
    print(f"\n  最小总代价：N=10 → k={best10['k']}（{best10['cost_N10']:.0f} s）；"
          f"N=16 → k={best16['k']}（{best16['cost_N16']:.0f} s）")
    print("  说明：环半径随 k 减小（k=7→999 m、k=14→839 m、极限 800 m = 1800−1000），"
          "巡游也随之变短；\n  但每个环点都要做一次全频道扫描（空频道也要测），"
          "每多点约多 (20−N)×6 s，而巡游每多点只省约 20 s。")
    return {"rows": rows, "best_N10": best10["k"], "best_N16": best16["k"]}


def analyse_intersection_variant() -> dict:
    """验证"让相邻覆盖圆的外侧交点落在 R=1800 圆周上"这一构造。

    几何：环上相邻两点（半径 a、夹角 2π/k）的覆盖圆（半径 R_min）有两个交点，
    其中**外侧交点**位于角平分线方向，到原点距离为
        ρ_outer(a, k) = a·cos(π/k) + sqrt(R_min² − (a·sin(π/k))²)
    令 ρ_outer = R 即得到"外侧交点正好落在 1800 圆周上"的环半径。

    注意：它与"边界角平分点恰好被覆盖"是**同一条件**
        √(R² + a² − 2Ra·cos(π/k)) = R_min，
    因此也等于 `ring_radius_min(k)` 的解析解。
    """
    print("\n=== 构造验证：相邻覆盖圆外侧交点落在 R=1800 圆周上 ===")
    rows = {}
    for k in (6, 7, 8):
        a_analytic = ring_radius_min(k)
        # 直接按"外侧交点"定义做数值求根，与解析式互证
        lo, hi = 100.0, 2600.0
        for _ in range(200):
            mid = (lo + hi) / 2.0
            rho = (mid * math.cos(math.pi / k)
                   + math.sqrt(max(R_MIN ** 2 - (mid * math.sin(math.pi / k)) ** 2, 0.0)))
            if rho < REGION_RADIUS:
                lo = mid
            else:
                hi = mid
        a_numeric = (lo + hi) / 2.0
        design = seven_point_design(a_analytic, k)
        wc = worst_case_radius(design, REGION_RADIUS, 8000, 8000)
        tour = tour_length(two_opt_tour(design))
        rows[k] = {"a_analytic": a_analytic, "a_numeric": a_numeric,
                   "worst_case_m": wc, "tour_m": tour, "sweeps": k + 1}
        print(f"  k={k}: 环半径 a={a_analytic:.1f} m（数值求根 {a_numeric:.1f}）"
              f"  最坏最近距离={wc:.1f} m  巡游={tour / 1000:.2f} km")
    a6, a7 = rows[6]["a_analytic"], rows[7]["a_analytic"]
    print(f"\n  ⇒ k=6 的『外侧交点落在 1800 圆周上』要求 a={a6:.1f} m，"
          f"比 k=7 的 {a7:.1f} m **更远**：")
    print(f"     巡游也随之更长（{rows[6]['tour_m'] / 1000:.2f} km vs "
          f"{rows[7]['tour_m'] / 1000:.2f} km）—— 与『更靠近圆心』的直觉相反。")
    print("     原因：点数越少，相邻角隙 2π/k 越大，边界角平分点离环点越远；")
    print("     要让半径 1000 的圆仍能够到它，环只能往外挪。")

    print("\n  验证不等式方向（环点到边界角平分点的距离，须 <= R_min=1000）：")
    print("    a(m) | k=6 (30°) | k=7 (25.71°) | k=8 (22.5°)")
    for a in (800.0, 900.0, 997.0, 1050.0, 1122.9):
        cells = []
        for k in (6, 7, 8):
            d = math.sqrt(REGION_RADIUS ** 2 + a ** 2
                          - 2 * REGION_RADIUS * a * math.cos(math.pi / k))
            cells.append(f"{d:6.1f}{'✓' if d <= R_MIN else '✗'}")
        print(f"   {a:6.1f} | " + " | ".join(cells))
    print("   ⇒ 半径越小越难够到边界：a 的下界由角隙决定（k=6→1122.9、k=7→997.2、k=8→938.1）。")
    return rows


def main() -> None:
    report = {"criteria": analyse_criteria(),
              "problem1_mc": monte_carlo_problem1(),
              "problem3_ring": analyse_p3_ring(),
              "intersection_variant": analyse_intersection_variant()}
    out = os.path.join(ROOT, "results", "design_analysis.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print(f"\n结果已写入: {out}")


if __name__ == "__main__":
    main()

