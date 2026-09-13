"""实验：预测性牵引（1 条示向度就插入复测点）的参数扫描。"""
import statistics
import sys

sys.path.insert(0, r"G:\C\CUMCM B题")

from src.mock_simulator import MockSimulator, random_case
from src.strategy_p3 import StrategyP3


class V(StrategyP3):
    def __init__(self, minobs, step, side):
        super().__init__()
        self.insert_min_observations = minobs
        self.pursuit_step = step
        self.pursuit_side = side


def bench(n, seeds=20, **kw):
    T, M, C, U = [], [], [], 0
    for seed in range(seeds):
        sim = MockSimulator(random_case(n=n, seed=seed), seed=seed)
        r = V(**kw).run(sim)
        T.append(r.mean_clear_time)
        M.append(r.move_distance / 1000)
        C.append(r.clears + r.failed_clears)
        U += len(r.unresolved)
    return (statistics.fmean(T), statistics.fmean(M), statistics.fmean(C), U)


print("预测性牵引参数扫描（20 例/档）")
for label, kw in [
    ("基线 obs=2", dict(minobs=2, step=150, side=300)),
    ("obs=1 step600 side300", dict(minobs=1, step=600, side=300)),
    ("obs=1 step700 side300", dict(minobs=1, step=700, side=300)),
    ("obs=1 step800 side400", dict(minobs=1, step=800, side=400)),
    ("obs=1 step600 side200", dict(minobs=1, step=600, side=200)),
]:
    cells, bad = [], 0
    for n in (10, 16):
        t, m, c, u = bench(n, **kw)
        cells.append(f"N{n}:{t:6.1f}/{m:5.2f}/{c:5.0f}")
        bad += u
    print(f"{label:24s} " + " | ".join(cells) + f"  未清={bad}")
