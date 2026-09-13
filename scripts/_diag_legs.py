"""诊断：问题3 N=10 单例逐动作行程账本（找出 3.07 km 超支的确切位置）。"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.coverage import seven_point_design                       # noqa: E402
from src.geometry import distance                                 # noqa: E402
from src.mock_simulator import MockSimulator, random_case         # noqa: E402
from src.routing import tour_length, two_opt_tour                 # noqa: E402
from src.strategy_p3 import StrategyP3                            # noqa: E402

SEED = 0
N = 10
jam = random_case(n=N, seed=SEED)
sim = MockSimulator(jam, seed=SEED)
res = StrategyP3().run(sim)

ring = seven_point_design(1000.0, 7)
ideal = tour_length(two_opt_tour([(0.0, 0.0)] + list(ring)
                                + [tuple(j.position) for j in jam]))
print(f"N={N} seed={SEED}: 实际 {res.move_distance/1000:.2f} km | "
      f"理想合并 {ideal/1000:.2f} km | 时间 {res.mean_clear_time:.1f} s/源")
print(f"检测 {res.measures} 次、清除成功 {res.clears}、失败 {res.failed_clears}\n")

# 逐动作账本：把移动 > 50 m 的动作按"目的"归类打印
pos = (0.0, 0.0)
ring_set = {(round(p[0], 3), round(p[1], 3)) for p in ring}
src_pos = {j.channel: tuple(j.position) for j in jam}
legs = []
for e in sim.log:
    if e["path"] not in ("/measure", "/clear"):
        continue
    p = tuple(e["position"])
    d = distance(pos, p)
    if d > 50.0:
        key = (round(p[0], 3), round(p[1], 3))
        if key in ring_set:
            kind = "覆盖点(全频道扫描)"
        elif e["path"] == "/clear":
            g = src_pos.get(e["channel"])
            kind = f"清除ch{e['channel']}(距真值{distance(p, g):.0f}m)" if g else "清除"
        else:
            g = src_pos.get(e["channel"])
            kind = (f"复测ch{e['channel']}(距真值{distance(p, g):.0f}m)" if g
                    else f"探测点ch{e['channel']}(空频道)")
        legs.append((d, kind, e.get("measure_result") or e.get("clear_result")))
    pos = p

print(f"{'行程m':>8}  {'目的':<34} 结果")
for d, kind, r in legs:
    print(f"{d:8.0f}  {kind:<34} {r}")
print(f"\n>50 m 的移动共 {len(legs)} 段、合计 {sum(d for d, _, _ in legs)/1000:.2f} km")
