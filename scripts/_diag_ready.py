"""诊断：源在"第几个覆盖点扫描后"才集齐 2 条示向度（决定能否顺路清除）。"""
from __future__ import annotations

import collections
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.coverage import seven_point_design                       # noqa: E402
from src.mock_simulator import MockSimulator, random_case         # noqa: E402
from src.strategy_p3 import StrategyP3                            # noqa: E402

RING = {(round(p[0], 3), round(p[1], 3)) for p in seven_point_design(1000.0, 7)}

for n in (10, 16):
    ready_at, late, nsrc = [], [], []
    for seed in range(12):
        jam = random_case(n=n, seed=seed)
        sim = MockSimulator(jam, seed=seed)
        StrategyP3().run(sim)
        seen = collections.Counter()
        order, ncov = [], 0
        seen_cov = set()
        for e in sim.log:
            if e["path"] != "/measure":
                continue
            key = (round(e["position"][0], 3), round(e["position"][1], 3))
            if key in RING and key not in seen_cov:
                seen_cov.add(key)
                ncov += 1
            if e["measure_result"] == "direction":
                seen[e["channel"]] += 1
                if seen[e["channel"]] == 2:
                    order.append(ncov)
        if order:
            ready_at.append(statistics.fmean(order))
            late.append(sum(1 for k in order if k >= 6) / len(order))
        nsrc.append(len(order))
    print(f"N={n}: 平均在第 {statistics.fmean(ready_at):.2f} 个覆盖点后集齐 2 条示向度；"
          f"其中 {100*statistics.fmean(late):.0f}% 的源要等到第 6 个覆盖点之后"
          f"（可交会源 {statistics.fmean(nsrc):.1f} 个）")
