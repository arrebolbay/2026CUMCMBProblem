"""问题1/2/3 的设计分析测试（对应 scripts/analyze_designs.py）。"""

import importlib.util
import math
import os

import pytest

from src.config import REGION_RADIUS, R_MIN
from src.coverage import seven_point_design, worst_case_radius


def _load():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, "scripts", "analyze_designs.py")
    spec = importlib.util.spec_from_file_location("analyze_designs_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


AD = _load()


# ------------------------- 布站准则（45° vs 54.74°） ------------------------- #
def test_alpha_family_recovers_both_criteria():
    """b* = d1/√α：α=1 → 45°（本题有界误差模型）；α=1/2 → 54.74°（=√2·d1）。"""
    b1 = AD.optimal_b_analytic(1000.0, 1.0)
    assert b1 == pytest.approx(1000.0)
    assert math.degrees(math.atan2(b1, 1000.0)) == pytest.approx(45.0, abs=1e-9)

    b2 = AD.optimal_b_analytic(1000.0, 0.5)
    assert b2 == pytest.approx(1000.0 * math.sqrt(2.0), rel=1e-12)
    assert math.degrees(math.atan2(b2, 1000.0)) == pytest.approx(54.7356, abs=1e-3)


def test_optimal_b_analytic_matches_numeric_scan():
    for alpha in (0.5, 2.0 / 3.0, 1.0, 1.5, 2.0):
        analytic = AD.optimal_b_analytic(1000.0, alpha)
        numeric = AD.optimal_b_numeric(1000.0, alpha)
        assert numeric == pytest.approx(analytic, abs=2.0), alpha


def test_minimax_baseline_is_robust_across_alpha():
    """极小极大稳健基线应落在两种准则之间，且最坏面积惩罚 < 1.06。"""
    report = AD.analyse_criteria()
    b = report["minimax"]["b"]
    assert 1000.0 <= b <= 1414.3
    assert report["minimax"]["penalty"] < 1.06
    # 机动成本越大，最优基线越短（与"移动代价变高则基线缩短"一致）
    rows = report["move_cost"]
    assert rows[0]["b_star"] >= rows[-1]["b_star"]


# ------------------------------ 问题3 覆盖环 ------------------------------ #
def test_ring_radius_min_matches_known_cases():
    """k=6 时最小环半径 1123 m（交点恰在 1800 圆周上），k=7 时 997 m。

    这正是"把相邻覆盖圆的交点放到 1800 m 圆周上"的解析解：环点数越少，
    交点约束要求环半径越大（6 点必须放到 1123 m，比 7 点的 997 m 更远）。
    """
    a6 = AD.ring_radius_min(6)
    a7 = AD.ring_radius_min(7)
    assert a6 == pytest.approx(1122.5, abs=1.0)
    assert a7 == pytest.approx(997.2, abs=1.0)
    assert a6 > a7, "点数越少，环半径必须越大（与直觉相反）"


def test_ring_radius_min_is_monotone_and_bounded():
    """a_min(k) 随 k 单调减小，下界为 1800−1000 = 800 m（对准边界点的极限）。"""
    values = [AD.ring_radius_min(k) for k in (6, 7, 8, 10, 12, 14, 16, 20, 40)]
    assert all(a > 800.0 for a in values)
    assert all(b <= a + 1e-9 for a, b in zip(values, values[1:]))
    assert AD.ring_radius_min(200) == pytest.approx(800.0, abs=2.0)


def test_outer_intersection_variant_equals_analytic_min_radius():
    """『相邻覆盖圆外侧交点落在 R=1800 圆周上』与 a_min(k) 解析式等价。"""
    rows = AD.analyse_intersection_variant()
    for k in (6, 7, 8):
        assert rows[k]["a_numeric"] == pytest.approx(rows[k]["a_analytic"], abs=0.5)
        # 该构造下覆盖恰好"卡"在 1000 m（零裕量）
        assert rows[k]["worst_case_m"] == pytest.approx(R_MIN, abs=0.05)


def test_six_ring_points_must_be_farther_out_and_tour_is_longer():
    """关键结论：6 点的外侧交点构造要求环半径 1123 m（> 7 点的 997 m），巡游更长。

    即"让外侧交点落在 1800 圆周上"这一约束把环**往外推**，而不是拉近圆心。
    """
    rows = AD.analyse_intersection_variant()
    assert rows[6]["a_analytic"] > rows[7]["a_analytic"] > rows[8]["a_analytic"]
    assert rows[6]["tour_m"] > rows[7]["tour_m"] > rows[8]["tour_m"]


def test_boundary_reach_grows_with_angle_gap():
    """环点到边界角平分点的距离随角隙 π/k 增大而增大——这就是"点数越少环越远"的原因。"""
    def reach(a, k):
        return math.sqrt(REGION_RADIUS ** 2 + a ** 2
                         - 2 * REGION_RADIUS * a * math.cos(math.pi / k))

    for a in (900.0, 1000.0, 1100.0):
        assert reach(a, 6) > reach(a, 7) > reach(a, 8)
    # 当前采用 a=1000、k=7：边界 998.25 m 有裕量，且中心点也覆盖原点
    assert reach(1000.0, 7) == pytest.approx(998.2545, abs=1e-3)
    assert reach(1000.0, 7) < R_MIN

    """取 a_min(k) 的环 + 中心点，数值复核最坏最近距离均 <= R_min。"""
    from src.coverage import seven_point_design as design

    for k in (6, 7, 8, 12):
        pts = design(AD.ring_radius_min(k), k)
        wc = worst_case_radius(pts, REGION_RADIUS, 4000, 4000)
        assert wc <= R_MIN + 1e-6, (k, wc)


def test_analytic_min_radius_matches_numeric_coverage():
    """解析 a_min 处最坏最近距离≈1000：略小一点就不再覆盖（紧致性）。"""
    a = AD.ring_radius_min(7)
    tight = seven_point_design(a, 7)
    loose = seven_point_design(a * 0.96, 7)
    assert worst_case_radius(tight, REGION_RADIUS, 6000, 6000) <= R_MIN
    assert worst_case_radius(loose, REGION_RADIUS, 6000, 6000) > R_MIN


# ------------------------------ 问题1 蒙特卡洛 ------------------------------ #
def test_monte_carlo_reports_coverage_by_station_count():
    mc = AD.monte_carlo_problem1(n_cases=600, seed=11)
    assert set(mc) == {2, 3, 4}
    # 2 站（纯楔形交为平行四边形）覆盖率最高，站数越多越可能失效
    assert mc[2]["coverage_ratio"] >= mc[3]["coverage_ratio"]
    assert mc[2]["coverage_ratio"] > 0.95
    # 失效构型的半径超出量是米级
    assert mc[3]["excess_median_m"] < 1.0
