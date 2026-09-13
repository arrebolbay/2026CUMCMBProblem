"""实验：问题3「滚动重优化（receding-horizon）」新模型 vs 现方案。

思路：把问题看成**带强制点的在线 TSP**——
  强制点 = 尚未到访的覆盖扫描点（保证不漏测，必须全部走到）
  动态点 = 已定位但未清除的源（发现后才出现）
每走完一步，用"贪心最近邻 + 2-opt + Or-opt"对 (当前位置 ∪ 剩余强制点 ∪ 待清源)
**从头重解**一条开放路径，只提交第一站（滚动时域）。
比"最便宜插入"更接近理想合并巡游：插入法一旦把源放错段就无法整体重排。
"""
from __future__ import annotations

import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.coverage import seven_point_design                       # noqa: E402
from src.geometry import (                                        # noqa: E402
    distance, localization_region, min_enclosing_circle,
)
from src.mock_simulator import MockSimulator, random_case         # noqa: E402
from src.strategy_p3 import CHANNELS, StrategyP3, StrategyResult  # noqa: E402


def open_tour(start, points, rounds: int = 6):
    """贪心最近邻 + 2-opt + Or-opt 的开放路径（固定起点，不回到起点）。"""
    remaining = list(points)
    tour = [start]
    while remaining:
        nxt = min(remaining, key=lambda q: distance(tour[-1], q))
        remaining.remove(nxt)
        tour.append(nxt)
    if len(tour) < 4:
        return tour
    for _ in range(rounds):
        improved = False
        n = len(tour)
        for i in range(n - 2):
            for j in range(i + 2, n - 1):
                a, b, c, d = tour[i], tour[i + 1], tour[j], tour[j + 1]
                if (distance(a, b) + distance(c, d)
                        > distance(a, c) + distance(b, d) + 1e-9):
                    tour[i + 1:j + 1] = tour[i + 1:j + 1][::-1]
                    improved = True
        for seg in (1, 2, 3):
            for s in range(1, len(tour) - seg):
                block = tour[s:s + seg]
                prev = tour[s - 1]
                nxt = tour[s + seg] if s + seg < len(tour) else None
                out = distance(prev, block[0])
                if nxt is not None:
                    out += distance(block[-1], nxt) - distance(prev, nxt)
                rest = tour[:s] + tour[s + seg:]
                best = None
                for t in range(len(rest) - 1):
                    u, v = rest[t], rest[t + 1]
                    add = (distance(u, block[0]) + distance(block[-1], v)
                           - distance(u, v))
                    if best is None or add < best[0]:
                        best = (add, t + 1)
                tail = distance(rest[-1], block[0])
                if best is None or tail < best[0]:
                    best = (tail, len(rest))
                if best[0] < out - 1e-9:
                    tour = rest[:best[1]] + block + rest[best[1]:]
                    improved = True
        if not improved:
            break
    return tour


class StrategyReplan(StrategyP3):
    """滚动重优化版问题3 策略（覆盖保证与 StrategyP3 完全一致：7 均布环全走到）。"""

    def __init__(self, adaptive_obs: bool = True, obs_mec_limit: float = 45.0):
        super().__init__()
        self.adaptive_obs = adaptive_obs
        self.obs_mec_limit = obs_mec_limit

    def _observations_complete(self, records):
        """自适应观测目标：2 条示向度已把定位区域压得足够小就不必要第 3 条。

        原实现固定要 3 条（省下长尾误差），但当两条示向度**视差已经很好**时，
        第 3 条纯属浪费（一次检测 6 s，且会推迟"顺路插入"的时机）。
        """
        if len(records) >= self.observations_target:
            return True
        if self.adaptive_obs and len(records) == 2:
            poly = localization_region([p for p, _ in records],
                                       [b for _, b in records])
            if poly:
                _, radius = min_enclosing_circle(poly)
                return radius <= self.obs_mec_limit
        return False

    def run(self, transport, channel_order=None, do_enter=True):
        result = StrategyResult()
        if do_enter:
            transport.enter()
        channels = list(channel_order) if channel_order else list(CHANNELS)
        bearings = {ch: [] for ch in channels}
        cleared: set = set()
        attempted: set = set()

        origin = self._current_position(transport)
        # 覆盖保证：半径 1000 m 的 7 均布环（解析最坏 998.25 m < R_min）
        mandatory = list(seven_point_design(self.cover_ring_radius, 7))

        self._sweep(transport, origin, channels, bearings, cleared, result)

        while len(cleared) < self.max_jammers:
            here = self._current_position(transport)
            targets = {}
            for ch in channels:
                if ch in cleared or ch in attempted:
                    continue
                if len(bearings[ch]) < self.insert_min_observations:
                    continue
                est = self._source_target(bearings[ch])
                if est is not None:
                    targets[tuple(est)] = ch
            nodes = list(mandatory) + list(targets.keys())
            if not nodes:
                break
            nxt = open_tour(here, nodes)[1]
            if nxt in targets:                       # 下一站是源：清除
                ch = targets[nxt]
                attempted.add(ch)
                if self._fast_clear(transport, ch, nxt, bearings[ch], result):
                    cleared.add(ch)
            else:                                     # 下一站是覆盖点：全频道扫描
                mandatory = [p for p in mandatory if p != nxt]
                self._sweep(transport, nxt, channels, bearings, cleared, result)

        self._finalize_channel_clearing(transport, channels, bearings, cleared, result)
        transport.exit()
        result.cleared = len(cleared)
        result.total = self._evaluate_total(transport)
        result.virtual_time = self._virtual_time(transport)
        result.move_distance = self._move_distance(transport)
        return result


def bench(cls, n, seeds=12, **kw):
    times, moves, meas, ur = [], [], [], 0
    for seed in range(seeds):
        sim = MockSimulator(random_case(n=n, seed=seed), seed=seed)
        res = cls(**kw).run(sim)
        times.append(res.mean_clear_time)
        moves.append(res.move_distance / 1000.0)
        meas.append(res.measures)
        ur += len(res.unresolved)
    return (statistics.fmean(times), statistics.fmean(moves),
            statistics.fmean(meas), ur)


if __name__ == "__main__":
    for label, cls, kw in (
        ("现方案 StrategyP3", StrategyP3, {}),
        ("滚动重优化+obs自适应", StrategyReplan, {}),
        ("滚动重优化+obs固定3", StrategyReplan, dict(adaptive_obs=False)),
    ):
        cells, bad = [], 0
        for n in (10, 12, 14, 16):
            t, mv, ms, u = bench(cls, n, **kw)
            cells.append(f"N={n}:{t:6.1f}s/{mv:5.2f}km/{ms:5.1f}")
            bad += u
        print(f"{label:22s} " + " ".join(cells) + f"  未清={bad}")
