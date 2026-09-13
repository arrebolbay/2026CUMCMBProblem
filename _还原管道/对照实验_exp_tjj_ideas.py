"""对照实验：把 tjj 组的关键思路放进我们的框架实测。

对照项
------
A. 骨架半径 1000 → 1010（他们的 8 点骨架取值）：
   最坏覆盖从 998.25 m 降到 992.06 m，余量 1.75 m → 8.1 m。
B. 最便宜插入阈值 insert_limit：我们默认 2500 m（很宽松，容易绕远路），
   他们的调度是"距离 − 等效奖励"择优，等价于**只在绕行很小时才插入**。
   故扫 2500 / 800 / 400 / 200 / 100 m。
C. 问题四的 insert_threshold（默认 400 m）同法扫。
"""
from __future__ import annotations

import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.coverage import seven_point_design                 # noqa: E402
from src.mock_simulator import MockSimulator, random_case    # noqa: E402
from src.strategy_p3 import StrategyP3                       # noqa: E402
from src.strategy_p4 import StrategyP4                       # noqa: E402


def worst_case(ring_radius: float, ring_count: int = 14) -> float:
    """覆盖集（中心 + 每隔一点取一个环点）的最坏覆盖距离（解析式）。"""
    k = ring_count // 2                       # 实际参与覆盖的环点数 = 7
    a = ring_radius
    return math.sqrt(1800.0 ** 2 + a * a - 2 * 1800.0 * a * math.cos(math.pi / k))


def bench(maker, n: int, ratio: float, seeds: int):
    times, moves, meas, missed = [], [], [], 0
    for s in range(seeds):
        jammers = random_case(n=n, seed=s, directional_ratio=ratio)
        sim = MockSimulator(jammers, seed=s)
        res = maker().run(sim)
        times.append(res.mean_clear_time)
        moves.append(res.move_distance / 1000.0)
        meas.append(res.measures)
        missed += max(0, n - res.cleared)
    return (statistics.fmean(times), statistics.fmean(moves),
            statistics.fmean(meas), missed)


def p3(insert_limit=None, ring_radius=None, probe_keep=None):
    """构造一个按需覆写类属性的 P3 实例工厂。"""
    strat = StrategyP3()
    if insert_limit is not None:
        strat.insert_limit = float(insert_limit)
    if ring_radius is not None:
        strat.cover_ring_radius = float(ring_radius)
    if probe_keep is not None:
        strat.probe_keep = int(probe_keep)
    return strat


def _p3_probe_keep(keep, ring_radius=None):
    return p3(probe_keep=keep, ring_radius=ring_radius)


def p4(threshold=None):
    strat = StrategyP4(insert_threshold=float(threshold)) \
        if threshold is not None else StrategyP4()
    return strat


def main() -> None:
    print("=== A. 骨架半径对最坏覆盖距离的影响 ===")
    for r in (1000.0, 1005.0, 1010.0, 1015.0):
        wc = worst_case(r)
        print("   环半径 %6.1f m  ⇒  最坏覆盖 %8.3f m   余量 %5.2f m   %s"
              % (r, wc, 1000.0 - wc, "可行" if wc < 1000.0 else "不可行"))

    print("\n=== B. 问题三：最便宜插入阈值 insert_limit（N=16，20 例）===")
    print("   %-12s %8s %9s %8s %6s" % ("insert_limit", "s/源", "行程km", "测量", "漏清"))
    for lim in (2500, 800, 400, 200, 100):
        t, mv, ms, miss = bench(lambda lim=lim: p3(insert_limit=lim), 16, 0.0, 20)
        print("   %-12d %8.1f %9.2f %8.1f %6d" % (lim, t, mv, ms, miss))

    print("\n=== C. 问题三：骨架半径 1000 → 1010（N=16，20 例）===")
    print("   %-12s %8s %9s %8s %6s" % ("环半径", "s/源", "行程km", "测量", "漏清"))
    for r in (1000.0, 1010.0):
        t, mv, ms, miss = bench(lambda r=r: p3(ring_radius=r), 16, 0.0, 20)
        print("   %-12.0f %8.1f %9.2f %8.1f %6d" % (r, t, mv, ms, miss))

    print("\n=== D. 问题四：机会清除阈值（全定向 N=16，12 例）===")
    print("   %-12s %8s %9s %8s %6s" % ("insert_threshold", "s/源", "行程km", "测量", "漏清"))
    for th in (400.0, 250.0, 150.0, 80.0):
        t, mv, ms, miss = bench(lambda th=th: p4(th), 16, 1.0, 12)
        print("   %-12.0f %8.1f %9.2f %8.1f %6d" % (th, t, mv, ms, miss))

    print("\n=== E. 问题三：几何复测点个数 probe_keep（他们=0，我们默认 4）===")
    print("   %-12s %8s %9s %8s %6s" % ("probe_keep", "s/源", "行程km", "测量", "漏清"))
    for keep in (0, 2, 4, 5):
        t, mv, ms, miss = bench(
            lambda keep=keep: _p3_probe_keep(keep), 16, 0.0, 20)
        print("   %-12d %8.1f %9.2f %8.1f %6d" % (keep, t, mv, ms, miss))

    print("\n=== F. 组合：probe_keep=0 + 环半径 1010（最贴近他们的骨架设计）===")
    print("   %-12s %8s %9s %8s %6s" % ("组合", "s/源", "行程km", "测量", "漏清"))
    t, mv, ms, miss = bench(lambda: _p3_probe_keep(0, ring_radius=1010.0),
                            16, 0.0, 20)
    print("   %-12s %8.1f %9.2f %8.1f %6d" % ("keep0+r1010", t, mv, ms, miss))




if __name__ == "__main__":
    main()
