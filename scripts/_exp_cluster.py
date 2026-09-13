"""实验：把"首击失败"的代价降到最低（问题3 最后的可压空间）。

**实测**：首击误差中位 14.3~16.6 m（清除半径 20 m），**无系统偏置**
（偏远占比 47.5%/53.1%），故无法靠"偏置修正"提高命中率。
清除阶段行程构成（12 例/档）：去估计点的主行程 `first` 5.38/8.23 km 是主体，
近场收口仅 0.4 km。因此**唯一可压的是"首击失败后的补救效率"**。

**变体**：
  R 首击失败后立即用**极密微网格**（步长 14 m、5×5，覆盖 ±28 m）
  S 首击前不动，先在**估计点 20 m 内布 4 个偏移点**依次尝试（等价于把
    一次 clear 变成"小簇 clear"，每次失败只花 3 s + 极短行程）
  T R + 关闭昂贵的 rayscan/shift（仅保留网格类收口）
"""
from __future__ import annotations

import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.geometry import distance                                 # noqa: E402
from src.mock_simulator import MockSimulator, random_case         # noqa: E402
from src.strategy_p3 import StrategyP3                            # noqa: E402


class V(StrategyP3):
    def __init__(self, cluster=0, step=14.0, no_ray=False, no_shift=False):
        super().__init__()
        self.cluster = int(cluster)       # 首击点周围额外尝试的点数（0=关闭）
        self.cluster_step = float(step)
        if no_ray:
            self.corridor_min_diameter = 1e9   # 实际关闭 rayscan
            self.corridor_ratio = 1e9
        if no_shift:
            self.tighten_shift = False

    def _fast_clear(self, transport, channel, estimate, records, result):
        if self.cluster:
            # 小簇清除：估计点 + 周围 cluster 个点（半径 cluster_step）
            # 误差中位 15 m 且无偏置 ⇒ 在 ±14~20 m 处补几个点命中率提升明显，
            # 而每次失败只付 3 s + 十几米行程（远比走 200 m 收口便宜）。
            import math
            pts = [tuple(estimate)]
            for k in range(self.cluster):
                ang = 2.0 * math.pi * k / self.cluster
                pts.append((estimate[0] + self.cluster_step * math.cos(ang),
                            estimate[1] + self.cluster_step * math.sin(ang)))
            here = self._current_position(transport)
            pts.sort(key=lambda q: distance(here, q))
            for q in pts:
                if self._try_clear(transport, channel, q, result):
                    return True
        return super()._fast_clear(transport, channel, estimate, records, result)


def bench(n, seeds=12, **kw):
    t, mv, ms, c, ur = [], [], [], [], 0
    for seed in range(seeds):
        sim = MockSimulator(random_case(n=n, seed=seed), seed=seed)
        res = V(**kw).run(sim)
        t.append(res.mean_clear_time)
        mv.append(res.move_distance / 1000.0)
        ms.append(res.measures)
        c.append(res.clears + res.failed_clears)
        ur += len(res.unresolved)
    return (statistics.fmean(t), statistics.fmean(mv), statistics.fmean(ms),
            statistics.fmean(c), ur)


if __name__ == "__main__":
    print("目标 176~300；下界 272.6/235.6/205.4/184.0")
    print("格式：时间s / 行程km / 检测 / clear次数")
    for label, kw in [
        ("A 基线", {}),
        ("R 簇4@14m", dict(cluster=4, step=14.0)),
        ("R2 簇6@16m", dict(cluster=6, step=16.0)),
        ("R3 簇8@18m", dict(cluster=8, step=18.0)),
        ("T 簇6+关ray", dict(cluster=6, step=16.0, no_ray=True)),
        ("T2 簇6+关ray+关shift", dict(cluster=6, step=16.0, no_ray=True,
                                     no_shift=True)),
    ]:
        cells, bad = [], 0
        for n in (10, 12, 14, 16):
            a, b, c, d, u = bench(n, **kw)
            cells.append(f"{a:6.1f}/{b:5.2f}/{c:4.0f}/{d:5.1f}")
            bad += u
        print(f"{label:22s} " + " | ".join(cells) + f"  未清={bad}")
