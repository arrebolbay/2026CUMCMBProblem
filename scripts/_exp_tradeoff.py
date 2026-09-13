"""关键分析：176~300 s/源 的区间意味着什么？

**Oracle 上界检验**（真值已知、完美清除、零浪费；12 例/档）：
    仅清除（**不做覆盖扫描**，不合法）：N=10 161.3 | N=12 141.3 | N=14 140.5 | N=16 129.2
    含完整覆盖扫描（合法所需）：      N=10 300.8 | N=12 265.1 | N=14 233.1 | N=16 208.4

结论：**"完整覆盖扫描"这一项就占了 140~170 s/源**（N=10 时 300.8−161.3 = 139.5 s）。
因此若某方案在 N=10 也能做到 ~176 s/源，它**必然没有做完整的覆盖扫描**（也就无法
保证"不漏测"），或者它使用了不同的统计口径。

本脚本用于量化"少扫多少覆盖点 ⇒ 快多少 / 漏检概率多大"，给出**可选的快速模式**。
漏检概率用蒙特卡洛估计：随机源位置落在"未被任何扫描点 1000 m 圆覆盖"的区域内即漏检。
"""
from __future__ import annotations

import math
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np                                                # noqa: E402

from src.config import CHANNELS, REGION_RADIUS, R_MIN             # noqa: E402
from src.coverage import seven_point_design                       # noqa: E402
from src.mock_simulator import MockSimulator, random_case         # noqa: E402
from src.routing import two_opt_tour                              # noqa: E402


def miss_probability(points, samples=200000, seed=0):
    """单个源的漏检概率（**用真实 r_eff 分布**，而非保守的 R_MIN）。

    题给 r_eff ∈ [1000, 1500] 均匀分布；源被发现 ⟺ 存在扫描点 p 使
    dist(p, G) <= r_eff。用 R_MIN=1000 保守判定会**高估**漏检率
    （例如完整 8 点覆盖集的保守值是 3.59%，而真实值为 0：因为覆盖集的
    最坏覆盖距离 998.25 m < 1000 m <= r_eff）。
    """
    rng = np.random.default_rng(seed)
    r = REGION_RADIUS * np.sqrt(rng.random(samples))
    th = rng.random(samples) * 2 * math.pi
    gx, gy = r * np.cos(th), r * np.sin(th)
    reff = 1000.0 + 500.0 * rng.random(samples)      # r_eff ~ U[1000,1500]
    best = np.full(samples, np.inf)
    for px, py in points:
        best = np.minimum(best, np.hypot(gx - px, gy - py))
    return float((best > reff).mean())


def run_partial(n, seed, keep):
    """只访问 keep 个覆盖点（其余跳过）的 Oracle 时间（真值清除）。"""
    jam = random_case(n=n, seed=seed)
    sim = MockSimulator(jam, seed=seed)
    sim.enter()
    ring = list(seven_point_design(1000.0, 7))
    cover = [(0.0, 0.0)] + ring
    cover = cover[:keep]
    src = {tuple(j.position): j.channel for j in jam}
    order = two_opt_tour([(0.0, 0.0)] + cover[1:] + list(src.keys()))[1:]
    for p in order:
        if p in src:
            sim.clear(p, src[p])
        else:
            for ch in CHANNELS:
                sim.measure(p, ch)
    sim.exit()
    return sim.clock.virtual_time / max(sim.cleared_jammers, 1)


if __name__ == "__main__":
    ring = list(seven_point_design(1000.0, 7))
    print("覆盖点数 → 单源漏检概率 / Oracle 时间（真值已知，12 例/档）")
    for keep in (8, 7, 6, 5, 4, 3, 2, 1):
        pts = ([(0.0, 0.0)] + ring)[:keep]
        p_miss = miss_probability(pts)
        cells = []
        for n in (10, 12, 14, 16):
            per = [run_partial(n, s, keep) for s in range(12)]
            cells.append(f"N={n}:{statistics.fmean(per):6.1f}s")
        exp_miss = 1.0 - (1.0 - p_miss) ** 13   # 13 = 平均源数
        print(f"  {keep:2d} 点  漏检/源 {100*p_miss:5.2f}%  整例至少漏 1 个 "
              f"{100*exp_miss:5.1f}%  " + " | ".join(cells))
