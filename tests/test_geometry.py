"""几何库测试：交会定位区域、直径算法、最小覆盖圆、问题1 的覆盖性结论。"""

import math
import random

import pytest

from src.geometry import (
    angle_diff,
    bearing_between,
    circle_covers_polygon,
    convex_hull,
    diameter_circle_covers,
    distance,
    localization_region,
    min_enclosing_circle,
    normalize_deg,
    polygon_area,
    polygon_diameter_bruteforce,
    polygon_diameter_calipers,
    wedge_halfplanes,
)
from src.config import EPS_DEG


# --------------------------------------------------------------------------- #
# 角度工具
# --------------------------------------------------------------------------- #
def test_angle_tools():
    assert normalize_deg(-30.0) == pytest.approx(330.0)
    assert normalize_deg(370.0) == pytest.approx(10.0)
    assert angle_diff(1.0, 359.0) == pytest.approx(2.0)
    assert angle_diff(359.0, 1.0) == pytest.approx(-2.0)
    assert bearing_between((0.0, 0.0), (1.0, 0.0)) == pytest.approx(0.0)
    assert bearing_between((0.0, 0.0), (0.0, 1.0)) == pytest.approx(90.0)


# --------------------------------------------------------------------------- #
# 楔形与半平面
# --------------------------------------------------------------------------- #
def test_wedge_halfplanes_contain_bearing_direction():
    station = (100.0, -50.0)
    bearing = 37.0
    for ang_off in (-0.9, 0.0, 0.9):
        rad = math.radians(bearing + ang_off)
        p = (station[0] + 800.0 * math.cos(rad), station[1] + 800.0 * math.sin(rad))
        for hp in wedge_halfplanes(station, bearing, EPS_DEG):
            assert hp.contains(p), "楔形内部点应满足两个半平面"

    for ang_off in (-1.5, 1.5):
        rad = math.radians(bearing + ang_off)
        p = (station[0] + 800.0 * math.cos(rad), station[1] + 800.0 * math.sin(rad))
        assert not all(hp.contains(p) for hp in wedge_halfplanes(station, bearing, EPS_DEG))


# --------------------------------------------------------------------------- #
# 辅助：凸多边形内部判定
# --------------------------------------------------------------------------- #
def _point_in_convex(poly, p, tol=1e-9):
    n = len(poly)
    sign = 0
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        cross = (x2 - x1) * (p[1] - y1) - (y2 - y1) * (p[0] - x1)
        cur = 0 if abs(cross) <= tol else (1 if cross > 0 else -1)
        if cur == 0:
            continue
        if sign == 0:
            sign = cur
        elif sign != cur:
            return False
    return True


def _circle2(a, b):
    c = ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
    return c, distance(a, b) / 2.0


# --------------------------------------------------------------------------- #
# 问题1 主结论：两站定位区域是平行四边形，且"直径圆"必能覆盖
# --------------------------------------------------------------------------- #
def test_two_station_region_is_parallelogram_covered_by_diameter_circle():
    jammer = (620.0, 430.0)
    s1, s2 = (0.0, 0.0), (900.0, -200.0)
    b1 = bearing_between(s1, jammer)
    b2 = bearing_between(s2, jammer)
    poly = localization_region([s1, s2], [b1, b2])

    assert 3 <= len(poly) <= 4, "两站定位区域应为平行四边形"
    assert polygon_area(poly) > 0.0
    assert _point_in_convex(poly, jammer), "真值必落在定位区域内"

    info = diameter_circle_covers(poly)
    assert info["covered"] is True, "平行四边形情形下直径圆应能覆盖定位区域"
    assert info["diameter"] == pytest.approx(
        polygon_diameter_bruteforce(poly)[0], rel=1e-9)
    assert info["mec_radius"] <= info["circle_radius"] + 1e-9
    assert info["mec_radius"] >= info["circle_radius"] * (math.sqrt(3) / 2.0) - 1e-6


def test_three_station_region_counterexample_acute_triangle():
    """n>=3 时定位区域可以是锐角三角形，此时"直径圆"不能覆盖。"""
    acute = [(0.0, 0.0), (1.0, 0.0), (0.5, math.sqrt(3) / 2.0)]  # 等边三角形
    info = diameter_circle_covers(acute)
    assert info["diameter"] == pytest.approx(1.0)
    assert info["covered"] is False, "锐角三角形：直径圆无法覆盖"
    assert info["mec_radius"] == pytest.approx(1.0 / math.sqrt(3.0), rel=1e-6)
    assert circle_covers_polygon(info["mec_center"], info["mec_radius"], acute)


def test_obtuse_and_right_triangle_covered_by_diameter_circle():
    """钝角/直角三角形：直径圆必覆盖（Thales 定理的直接推论）。"""
    for tri in ([(0, 0), (4, 0), (1, 1)], [(0, 0), (3, 0), (3, 4)]):
        poly = [(float(a), float(b)) for a, b in tri]
        assert diameter_circle_covers(poly)["covered"] is True


# --------------------------------------------------------------------------- #
# 直径算法：旋转卡壳 vs 暴力
# --------------------------------------------------------------------------- #
def test_calipers_matches_bruteforce_on_random_polygons():
    rng = random.Random(42)
    for _ in range(20):
        pts = [(rng.uniform(-500, 500), rng.uniform(-500, 500)) for _ in range(12)]
        hull = convex_hull(pts)
        if len(hull) < 3:
            continue
        d1, _, _ = polygon_diameter_calipers(hull)
        d2, _, _ = polygon_diameter_bruteforce(hull)
        assert d1 == pytest.approx(d2, rel=1e-9, abs=1e-9)


# --------------------------------------------------------------------------- #
# 最小覆盖圆
# --------------------------------------------------------------------------- #
def test_min_enclosing_circle_covers_and_is_tight():
    rng = random.Random(7)
    pts = [(rng.uniform(-300, 300), rng.uniform(-300, 300)) for _ in range(30)]
    c, r = min_enclosing_circle(pts)
    assert all(distance(c, p) <= r + 1e-9 for p in pts)

    best = float("inf")
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            cc, rr = _circle2(pts[i], pts[j])
            if all(distance(cc, p) <= rr + 1e-9 for p in pts):
                best = min(best, rr)
    assert best >= r - 1e-6, "Welzl 结果不应大于任何可行覆盖圆"


# --------------------------------------------------------------------------- #
# 随机压力测试：真值必须落在交会定位区域内
# --------------------------------------------------------------------------- #
def test_localization_region_contains_true_jammer_stress():
    rng = random.Random(2026)
    tested = 0
    while tested < 20:
        jammer = (rng.uniform(-1600, 1600), rng.uniform(-1600, 1600))
        if distance((0, 0), jammer) > 1700:
            continue
        stations, bearings = [], []
        for _ in range(2):
            while True:
                s = (rng.uniform(-1000, 1000), rng.uniform(-1000, 1000))
                if distance(s, jammer) > 50:
                    break
            stations.append(s)
            bearings.append(bearing_between(s, jammer) + rng.uniform(-1.0, 1.0))
        poly = localization_region(stations, bearings)
        assert poly, "交会区域不应为空"
        assert _point_in_convex(poly, jammer), "真值应落在定位区域内"
        tested += 1
