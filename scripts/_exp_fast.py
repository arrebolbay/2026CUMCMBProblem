"""实验：问题3 逼近理论下界的两个真实杠杆（早清除 + 自适应观测数）。

**理论下界**（20 例/档，离线已知真值 + 动作下界）：
    N=10 272.6 s/源 | N=12 235.6 | N=14 205.4 | N=16 184.0
**现状**：N=10 362.2 | N=12 329.9 | N=14 290.9 | N=16 244.2

归因（10 例/档实测）：N=10 共 155.8 次检测、13.00 km，下界是 110 次、9.93 km
  ⇒ 检测超 45 次（≈270 s）、行程超 3.07 km（≈614 s）。

**注意"空频道检测"不是浪费**：频道 20 个、源 10~16 个，有 20−N 个空频道；
要断言某频道无源，必须在**一个完整覆盖集**（8 点）上都测到 no_signal（覆盖判据），
故 8×(20−N) 次检测是**硬下界**（N=10 时 80 次），已计入上表。

**两个真实杠杆**：
  1. **早清除**：源在清除前，会在后续每个覆盖点被重复检测；越早清掉越省。
  2. **自适应观测数**：固定 3 条示向度，但 2 条几何足够好时第 3 条纯浪费，
     还推迟了清除时机。
"""
from __future__ import annotations

import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.geometry import (                                        # noqa: E402
    distance, localization_region, min_enclosing_circle,
)
from src.mock_simulator import MockSimulator, random_case         # noqa: E402
from src.strategy_p3 import StrategyP3                            # noqa: E402


class Fast(StrategyP3):
    """可开关的提速变体（覆盖保证不变：8 个全频道扫描点全部走到）。"""

    def __init__(self, adaptive_obs=False, obs_mec_limit=60.0,
                 clear_now=False, clear_now_limit=600.0):
        super().__init__()
        self.adaptive_obs = adaptive_obs
        self.obs_mec_limit = obs_mec_limit
        self.clear_now = clear_now
        self.clear_now_limit = clear_now_limit

    def _observations_complete(self, records):
        if len(records) >= self.observations_target:
            return True
        if self.adaptive_obs and len(records) == 2:
            poly = localization_region([p for p, _ in records],
                                       [b for _, b in records])
            if poly:
                _, radius = min_enclosing_circle(poly)
                return radius <= self.obs_mec_limit
        return False

    def _schedule_sources(self, route, index, channels, bearings, cleared,
                          attempted, scheduled) -> None:
        if self.clear_now:
            # 先把"很便宜就能马上清掉"的源插到**下一站**，尽早停止重复检测
            while True:
                here = route[index][0]
                nxt = route[index + 1][0] if index + 1 < len(route) else None
                best = None
                for ch in channels:
                    if ch in cleared or ch in attempted or ch in scheduled:
                        continue
                    if len(bearings[ch]) < self.insert_min_observations:
                        continue
                    est = self._source_target(bearings[ch])
                    if est is None:
                        continue
                    detour = distance(here, est)
                    if nxt is not None:
                        detour += distance(est, nxt) - distance(here, nxt)
                    if detour <= self.clear_now_limit and (
                            best is None or detour < best[0]):
                        best = (detour, ch, est)
                if best is None:
                    break
                _, ch, est = best
                route.insert(index + 1, (est, ch, False))
                scheduled.add(ch)
        super()._schedule_sources(route, index, channels, bearings, cleared,
                                  attempted, scheduled)


def bench(n, seeds=12, **kw):
    times, moves, meas, ur = [], [], [], 0
    for seed in range(seeds):
        sim = MockSimulator(random_case(n=n, seed=seed), seed=seed)
        res = Fast(**kw).run(sim)
        times.append(res.mean_clear_time)
        moves.append(res.move_distance / 1000.0)
        meas.append(res.measures)
        ur += len(res.unresolved)
    return (statistics.fmean(times), statistics.fmean(moves),
            statistics.fmean(meas), ur)


if __name__ == "__main__":
    print("目标 176~300 s/源；下界 N=10/12/14/16 = 272.6/235.6/205.4/184.0")
    print("格式：时间s / 行程km / 检测次数")
    for label, kw in [
        ("A 基线", {}),
        ("B 自适应obs 60m", dict(adaptive_obs=True)),
        ("C 自适应obs 120m", dict(adaptive_obs=True, obs_mec_limit=120.0)),
        ("D 早清除 600m", dict(clear_now=True)),
        ("E B+D", dict(adaptive_obs=True, clear_now=True)),
        ("F C+早清除1200m", dict(adaptive_obs=True, obs_mec_limit=120.0,
                                clear_now=True, clear_now_limit=1200.0)),
    ]:
        cells, bad = [], 0
        for n in (10, 12, 14, 16):
            t, mv, ms, u = bench(n, **kw)
            cells.append(f"{t:6.1f}/{mv:5.2f}/{ms:5.1f}")
            bad += u
        print(f"{label:17s} " + " | ".join(cells) + f"  未清={bad}")
