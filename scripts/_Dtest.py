"""实验D-端到端：新候选网（21/25 点）vs 基准 25 点网的 Mock 实测。"""
import math
import statistics
import sys

sys.path.insert(0, r"G:\C\CUMCM B题")

from src.mock_simulator import MockSimulator, random_case  # noqa: E402
from src.strategy_p4 import StrategyP4  # noqa: E402

START = (0.0, 0.0)


def build(r1, n1, r2, n2, phase, center=True):
    pts = [START] if center else []
    for i in range(n1):
        a = 2 * math.pi * i / n1
        pts.append((r1 * math.cos(a), r1 * math.sin(a)))
    for i in range(n2):
        a = 2 * math.pi * (i + phase) / n2
        pts.append((r2 * math.cos(a), r2 * math.sin(a)))
    return pts


NETS = {
    "基准 25 点(950x12+1875x12)": build(950.0, 12, 1875.0, 12, 0.5),
    "21 点(1000x8+1875x12)": build(1000.0, 8, 1875.0, 12, 0.0),
    "21 点(1000x8+1875x12,ph.5)": build(1000.0, 8, 1875.0, 12, 0.5),
    "25 点(950x10+1860x14)": build(950.0, 10, 1860.0, 14, 0.5),
}


class V(StrategyP4):
    def __init__(self, pts, **kw):
        super().__init__(survey_points=pts, **kw)


def bench(pts, ratio, n, seeds=25):
    T, M, U, A = [], [], 0, 0
    for s in range(seeds):
        sim = MockSimulator(random_case(n=n, seed=s, directional_ratio=ratio),
                            seed=s)
        r = V(pts).run(sim)
        T.append(r.mean_clear_time)
        M.append(r.move_distance / 1000)
        A += len(sim.log)
        U += len(r.unresolved)
    return statistics.fmean(T), statistics.fmean(M), U, A / seeds


for label, pts in NETS.items():
    row = [f"{label:28s}"]
    for ratio, tag in ((1.0, "全定向"), (0.0, "全向")):
        for n in (16, 10):
            t, m, u, a = bench(pts, ratio, n)
            row.append(f"{tag}N{n}:{t:6.1f}/{m:5.2f}km/动作{a:4.0f}/未清{u}")
    print("  ".join(row))
