"""实验：一次性联合规划（信息早已齐备 ⇒ 不需要滚动插入）。

**关键实测**：所有源平均在第 **2.04 个**覆盖点扫描后就集齐 2 条示向度
（0% 的源需要等到第 6 个覆盖点之后）。也就是说走完 2~3 个覆盖点，
几乎**全部源的位置都已知**；此后的问题退化为一个**离线 TSP**：
    起点 = 当前位置；必访点 = 剩余覆盖点 ∪ 所有已定位源估计点
于是不该再"逐个最便宜插入 + 收尾折返"，而应**一次性联合求解**并按序执行。

变体：
  K 走完 ``warmup`` 个覆盖点后，对 (剩余覆盖点 ∪ 全部已定位源) 做一次全局
    开放路径优化（贪心 + 2-opt + Or-opt），之后**严格按该序执行**（不再重排）。
  L 同 K，但每清除一个源后允许再做一次全局重优化（吸收新发现的源）。
"""
from __future__ import annotations

import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.config import CHANNELS, REGION_RADIUS                    # noqa: E402
from src.coverage import seven_point_design                       # noqa: E402
from src.geometry import (                                        # noqa: E402
    distance, localization_region, min_enclosing_circle,
)
from src.mock_simulator import MockSimulator, random_case         # noqa: E402
from src.strategy_p3 import StrategyP3, StrategyResult            # noqa: E402


def open_tour(start, points, rounds: int = 8):
    """贪心最近邻 + 2-opt + Or-opt 的开放路径（起点固定，不回起点）。"""
    rest = list(points)
    tour = [start]
    while rest:
        nxt = min(rest, key=lambda q: distance(tour[-1], q))
        rest.remove(nxt)
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
            s = 1
            while s + seg <= len(tour) - 1:
                block = tour[s:s + seg]
                prev = tour[s - 1]
                nxt = tour[s + seg] if s + seg < len(tour) else None
                out = distance(prev, block[0])
                if nxt is not None:
                    out += distance(block[-1], nxt) - distance(prev, nxt)
                rest2 = tour[:s] + tour[s + seg:]
                best = None
                for t in range(len(rest2) - 1):
                    u, v = rest2[t], rest2[t + 1]
                    add = (distance(u, block[0]) + distance(block[-1], v)
                           - distance(u, v))
                    if best is None or add < best[0]:
                        best = (add, t + 1)
                tail = distance(rest2[-1], block[0])
                if best is None or tail < best[0]:
                    best = (tail, len(rest2))
                if best[0] < out - 1e-9:
                    tour = rest2[:best[1]] + block + rest2[best[1]:]
                    improved = True
                else:
                    s += 1
        if not improved:
            break
    return tour



class Joint(StrategyP3):
    """两阶段：短暖机（拿信息）+ 一次性联合 TSP（覆盖点与源共用一条路）。"""

    def __init__(self, warmup: int = 3, requick: bool = False,
                 plan_mec_limit: float = 300.0):
        super().__init__()
        self.warmup = int(warmup)
        self.requick = bool(requick)
        self.plan_mec_limit = float(plan_mec_limit)

    def _plan_target(self, records):
        """联合规划用的目标点：**必须有界可信**，否则不纳入本轮 TSP。

        `_source_target` 在"两条示向度近共线"时会返回**细长无界区域的形心**，
        它可能落在几千米之外（实测导致联合规划出现 6350 km 的荒谬行程）。
        这里只接受"最小覆盖圆半径 <= plan_mec_limit 且落在目标区域内"的估计，
        其余源留给后续覆盖点补足示向度（或收尾处理）。
        """
        if len(records) < 2:
            return None
        poly = localization_region([p for p, _ in records],
                                   [b for _, b in records])
        if not poly:
            return None
        center, radius = min_enclosing_circle(poly)
        if radius > self.plan_mec_limit:
            return None
        if distance((0.0, 0.0), center) > REGION_RADIUS + 50.0:
            return None
        return (float(center[0]), float(center[1]))

    def run(self, transport, channel_order=None, do_enter=True):
        result = StrategyResult()
        if do_enter:
            transport.enter()
        channels = list(channel_order) if channel_order else list(CHANNELS)
        bearings = {ch: [] for ch in channels}
        cleared: set = set()
        attempted: set = set()

        origin = self._current_position(transport)
        ring = list(seven_point_design(self.cover_ring_radius, 7))
        # --- 阶段1 暖机：按最近邻走前 warmup 个覆盖点，快速积累示向度 ---
        self._sweep(transport, origin, channels, bearings, cleared, result)
        pending_cov = list(ring)
        for _ in range(min(self.warmup, len(ring))):
            here = self._current_position(transport)
            nxt = min(pending_cov, key=lambda q: distance(here, q))
            pending_cov.remove(nxt)
            self._sweep(transport, nxt, channels, bearings, cleared, result)

        # --- 阶段2 联合规划：剩余覆盖点 ∪ 所有已定位源，一次性求最短开放路径 ---
        while True:
            here = self._current_position(transport)
            targets = {}
            for ch in channels:
                if ch in cleared or ch in attempted:
                    continue
                if len(bearings[ch]) < self.insert_min_observations:
                    continue
                est = self._plan_target(bearings[ch])
                if est is not None:
                    targets[tuple(est)] = ch
            nodes = list(pending_cov) + list(targets.keys())
            if not nodes:
                break
            plan = open_tour(here, nodes)[1:]
            for point in plan:
                if point in targets:
                    ch = targets[point]
                    if ch in cleared or ch in attempted:
                        continue
                    attempted.add(ch)
                    if self._fast_clear(transport, ch, point, bearings[ch], result):
                        cleared.add(ch)
                else:
                    if point not in pending_cov:
                        continue
                    pending_cov.remove(point)
                    got = self._sweep(transport, point, channels, bearings,
                                      cleared, result)
                    # 发现了新的可定位源 ⇒ 打断当前计划，重做联合规划
                    if self.requick and got:
                        fresh = any(ch not in cleared and ch not in attempted
                                    and self._plan_target(bearings[ch]) is not None
                                    and tuple(self._plan_target(bearings[ch]))
                                    not in targets
                                    for ch in channels)
                        if fresh:
                            break
            else:
                continue
            if not self.requick:
                break

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
    print("目标 176~300；下界 272.6/235.6/205.4/184.0（N=10/12/14/16）")
    for label, cls, kw in [
        ("A 基线", StrategyP3, {}),
        ("K 暖机2+联合", Joint, dict(warmup=2)),
        ("K 暖机3+联合", Joint, dict(warmup=3)),
        ("K 暖机4+联合", Joint, dict(warmup=4)),
        ("L 暖机3+联合+重规划", Joint, dict(warmup=3, requick=True)),
    ]:
        cells, bad = [], 0
        for n in (10, 12, 14, 16):
            t, mv, ms, u = bench(cls, n, **kw)
            cells.append(f"{t:6.1f}/{mv:5.2f}/{ms:5.0f}")
            bad += u
        print(f"{label:20s} " + " | ".join(cells) + f"  未清={bad}")
