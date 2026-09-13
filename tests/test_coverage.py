"""覆盖布点测试：验证问题3"最少 7 个检测点"的可证明下界与构造性方案。

理论依据（Bezdek 1979/1983，转引自 Fejes Tóth 2005）：
  * 6 个等圆最多覆盖半径比 1.7988… 的大圆；
  * 7 个等圆可达 2.0。
本题半径比 = 1800/1000 = 1.8，落在 (1.7988, 2.0]，故最少检测点数为 7。
本测试用数值实验交叉验证：
  * 构造性 7 点方案的最坏覆盖距离 < 1000 m（可行）；
  * 全局优化 6 点的最小覆盖半径 > 1000 m（不可能）。
"""

import pytest

from src.config import REGION_RADIUS, R_MIN
from src.coverage import (
    optimize_cover_radius,
    sample_disk,
    seven_point_design,
    worst_case_radius,
)


def test_seven_point_design_covers_region_with_margin():
    pts = seven_point_design(ring_radius=1400.0, k=6)
    assert len(pts) == 7
    wc = worst_case_radius(pts, REGION_RADIUS, n_interior=8000, n_boundary=8000)
    assert wc < R_MIN, f"7 点方案必须保证覆盖，实测最坏距离 {wc:.2f} m"
    assert wc < 950.0, f"7 点方案应有裕量，实测 {wc:.2f} m"


def test_boundary_ratio_threshold():
    """半径比 1800/1000 = 1.8 恰在 6 圆（1.7988）与 7 圆（2.0）阈值之间。

    即：6 个半径 1000 m 的圆最多覆盖 1798.8 m 半径的圆域（< 1800），
        7 个可达 2000 m（>= 1800）—— 故最少 7 个检测点。
    """
    ratio = REGION_RADIUS / R_MIN          # = 1.8
    six_threshold = 1.7988                 # Bezdek 1979 证明
    seven_threshold = 2.0                  # 平凡六边形构造
    assert six_threshold < ratio <= seven_threshold
    assert 1000.0 * seven_threshold >= REGION_RADIUS      # 7 点可行
    assert 1000.0 * six_threshold < REGION_RADIUS         # 6 点不可行


def test_six_points_cannot_cover_region_numerically():
    """数值优化 6 个点：最小覆盖半径应 > 1000 m（与经典结论一致）。"""
    best_radius, pts = optimize_cover_radius(
        6, REGION_RADIUS, n_objective=2000, seed=7, maxiter=25, popsize=10)
    assert pts.shape == (6, 2)
    assert best_radius > R_MIN, f"6 点最小覆盖半径 {best_radius:.2f} m 应超过 1000 m"


def test_sample_disk_within_region():
    import numpy as np

    s = sample_disk(5000, REGION_RADIUS, seed=1)
    r = np.linalg.norm(s, axis=1)
    assert r.max() <= REGION_RADIUS + 1e-9
    assert r.mean() == pytest.approx(2.0 * REGION_RADIUS / 3.0, rel=0.05)
