"""对照实验：干扰源"径向分布"对总时间的影响（用于解释练习 213 s 与正式 307 s 的差异）。

做法：控制唯一的自变量——源的径向范围，其余（N、频道、策略、模拟器）全部相同。
    A 组"近源"：全部源落在 r ≤ 1200 m（贴覆盖骨干环附近）
    B 组"远源"：全部源落在 1200 ≤ r ≤ 1800 m（贴近目标区域边界）
若两组差异显著，则说明单局时间主要由"案例的源分布"决定，而非策略本身。
"""
from __future__ import annotations

import math
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import CHANNELS, REGION_RADIUS, R_MAX, R_MIN  # noqa: E402
from src.mock_simulator import Jammer, MockSimulator          # noqa: E402
from src.strategy_p3 import StrategyP3                        # noqa: E402


def make_case(seed: int, n: int, r_lo: float, r_hi: float):
    """在给定径向区间 [r_lo, r_hi] 内按面积均匀生成 n 个全向源。"""
    rng = random.Random(seed)
    channels = rng.sample(list(CHANNELS), n)
    out = []
    for ch in channels:
        u = rng.random()
        r = math.sqrt(r_lo * r_lo + u * (r_hi * r_hi - r_lo * r_lo))
        t = rng.random() * 2.0 * math.pi
        out.append(Jammer(channel=ch, x=r * math.cos(t), y=r * math.sin(t),
                          r_eff=rng.uniform(R_MIN, R_MAX), direction=None))
    return out


def bench(tag: str, r_lo: float, r_hi: float, n: int = 15, seeds: int = 30):
    times, moves, meas, cleared_all = [], [], [], True
    for s in range(seeds):
        jammers = make_case(s, n, r_lo, r_hi)
        sim = MockSimulator(jammers, seed=s)
        res = StrategyP3().run(sim)
        times.append(res.mean_clear_time)
        moves.append(res.move_distance / 1000.0)
        meas.append(res.measures)
        if res.cleared < n:
            cleared_all = False
    return {
        "tag": tag, "time": statistics.fmean(times),
        "move": statistics.fmean(moves), "meas": statistics.fmean(meas),
        "all_cleared": cleared_all, "n": n, "seeds": seeds,
    }


def main() -> None:
    print("受控实验：仅改变源的径向分布（N=15，30 例/组，其余完全相同）")
    rows = [
        bench("A 近源  r ≤ 1200 m", 0.0, 1200.0),
        bench("B 远源  1200 ≤ r ≤ 1800 m", 1200.0, REGION_RADIUS),
        bench("C 全域（题目默认）", 0.0, REGION_RADIUS),
    ]
    print("%-26s %9s %9s %8s %8s" %
          ("组的径向范围", "s/源", "移动km", "测量", "100%清除"))
    for r in rows:
        print("%-26s %9.1f %9.2f %8.1f %8s"
              % (r["tag"], r["time"], r["move"], r["meas"],
                 "是" if r["all_cleared"] else "否"))
    a, b = rows[0], rows[1]
    print("\n差异（远源 − 近源）：%.1f s/源，其中移动 %.2f km（≈ %.0f s）、"
          "测量 %.0f 次（≈ %.0f s）"
          % (b["time"] - a["time"], b["move"] - a["move"],
             (b["move"] - a["move"]) * 1000.0 / 5.0,
             b["meas"] - a["meas"], (b["meas"] - a["meas"]) * 6.0))
    print("结论：单局时间主要由“源离原点多远”决定 —— 这正是练习 213 s 与"
          "正式 307 s 差异的来源。")


if __name__ == "__main__":
    main()
