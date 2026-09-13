"""上界检验：**真值已知**（作弊）时能做到多少 s/源？

用途：区分"策略不够好"与"下界本身不可达"。
  Oracle-A：直接走 (原点 → 所有真源) 的最优 TSP，每源一次成功 clear，
            **完全不做覆盖扫描**（不合法，但给出绝对下界）
  Oracle-B：覆盖扫描 + 真值清除（合法所需的扫描 + 完美清除），
            即"策略层完美"时的极限
"""
from __future__ import annotations

import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.config import CHANNELS                                   # noqa: E402
from src.coverage import seven_point_design                       # noqa: E402
from src.mock_simulator import MockSimulator, random_case         # noqa: E402
from src.routing import tour_length, two_opt_tour                 # noqa: E402


def oracle(n, seed, with_cover=True):
    """按真值执行：覆盖扫描（可选）+ 每源一次 clear。返回虚拟总时间。"""
    jam = random_case(n=n, seed=seed)
    sim = MockSimulator(jam, seed=seed)
    sim.enter()
    ring = list(seven_point_design(1000.0, 7)) if with_cover else []
    src = [(tuple(j.position), j.channel) for j in jam]
    nodes = list(ring) + [p for p, _ in src]
    order = two_opt_tour([(0.0, 0.0)] + nodes)[1:]
    ch_of = {p: c for p, c in src}
    pending = set(CHANNELS)
    for p in order:
        if p in ch_of:
            sim.clear(p, ch_of[p])
        else:
            for ch in sorted(pending):
                r = sim.measure(p, ch)
                if r["measure_result"] == "no_signal":
                    pass
    sim.exit()
    return sim.clock.virtual_time, sim.cleared_jammers, len(jam)


if __name__ == "__main__":
    print("Oracle（真值已知）上界检验：")
    for with_cover in (False, True):
        tag = "含覆盖扫描" if with_cover else "仅清除(不合法)"
        cells = []
        for n in (10, 12, 14, 16):
            per = []
            for seed in range(12):
                t, c, tot = oracle(n, seed, with_cover)
                per.append(t / max(c, 1))
            cells.append(f"N={n}:{statistics.fmean(per):6.1f}s")
        print(f"  {tag:16s} " + " | ".join(cells))
