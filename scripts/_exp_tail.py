"""实验：消除"收尾折返"—— 问题3 的真正瓶颈。

**账本（12 例/档）**：
    N=10：主阶段 8.77 km + **收尾折返 4.23 km**（≈846 s ≈ 85 s/源），清除失败 14.7 次
    N=16：主阶段 11.49 km + **收尾折返 3.34 km**，清除失败 21.7 次

收尾折返之所以贵：这些源要么只有 1 条示向度（`insert_min_observations=2` 不给排程）、
要么估计质量差被 `pursue_mec_limit=80 m` 挡住，于是全部堆到最后，形成一趟额外 TSP。

**变体**：
  G 放宽 `pursue_mec_limit`（允许带较差估计前往，就地用 1-D 射线扫描收口）
  H 允许**单条示向度**排程（`insert_min_observations=1`，用切向复测点定位）
  I G+H
  J I + 在最后一个覆盖点前**强制清空**所有已发现频道（禁止留到收尾）
"""
from __future__ import annotations

import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.mock_simulator import MockSimulator, random_case         # noqa: E402
from src.strategy_p3 import StrategyP3                            # noqa: E402


class V(StrategyP3):
    def __init__(self, mec=None, min_obs=None, insert=None):
        super().__init__()
        if mec is not None:
            self.pursue_mec_limit = mec
        if min_obs is not None:
            self.insert_min_observations = min_obs
        if insert is not None:
            self.insert_limit = insert


def bench(n, seeds=12, **kw):
    times, moves, meas, ur, fail = [], [], [], 0, []
    for seed in range(seeds):
        sim = MockSimulator(random_case(n=n, seed=seed), seed=seed)
        res = V(**kw).run(sim)
        times.append(res.mean_clear_time)
        moves.append(res.move_distance / 1000.0)
        meas.append(res.measures)
        fail.append(res.failed_clears)
        ur += len(res.unresolved)
    return (statistics.fmean(times), statistics.fmean(moves),
            statistics.fmean(meas), statistics.fmean(fail), ur)


if __name__ == "__main__":
    print("目标 176~300；下界 272.6/235.6/205.4/184.0（N=10/12/14/16）")
    print("格式：时间s / 行程km / 检测 / 清除失败")
    for label, kw in [
        ("A 基线", {}),
        ("G mec=200", dict(mec=200.0)),
        ("G2 mec=400", dict(mec=400.0)),
        ("H min_obs=1", dict(min_obs=1)),
        ("I mec=400+obs1", dict(mec=400.0, min_obs=1)),
        ("I2 mec=1e9+obs1", dict(mec=1e9, min_obs=1)),
        ("J I2+insert=1e9", dict(mec=1e9, min_obs=1, insert=1e9)),
    ]:
        cells, bad = [], 0
        for n in (10, 12, 14, 16):
            t, mv, ms, f, u = bench(n, **kw)
            cells.append(f"{t:6.1f}/{mv:5.2f}/{ms:5.0f}/{f:4.1f}")
            bad += u
        print(f"{label:17s} " + " | ".join(cells) + f"  未清={bad}")
