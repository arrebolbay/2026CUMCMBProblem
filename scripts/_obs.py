"""组合杠杆：观测数 / 追击兜底 对 N=10/16 的影响（当前追击代码已就位）。"""
import statistics
import sys

sys.path.insert(0, r"G:\C\CUMCM B题")

from src.geometry import localization_region, min_enclosing_circle
from src.mock_simulator import MockSimulator, random_case
from src.strategy_p3 import StrategyP3


class V(StrategyP3):
    def __init__(self, obs=None, adapt=None, mec=None):
        super().__init__()
        if obs is not None:
            self.observations_target = obs
        self.adapt = adapt
        if mec is not None:
            self.pursue_mec_limit = mec

    def _observations_complete(self, records):
        if len(records) >= self.observations_target:
            return True
        if self.adapt and len(records) == 2:
            poly = localization_region([p for p, _ in records],
                                       [b for _, b in records])
            if poly:
                _, r = min_enclosing_circle(poly)
                if r <= self.adapt:
                    return True
        return False


def bench(n, seeds=20, **kw):
    T, M, Me, C, U = [], [], [], [], 0
    for seed in range(seeds):
        sim = MockSimulator(random_case(n=n, seed=seed), seed=seed)
        r = V(**kw).run(sim)
        T.append(r.mean_clear_time)
        M.append(r.move_distance / 1000)
        Me.append(r.measures)
        C.append(r.clears + r.failed_clears)
        U += len(r.unresolved)
    return (statistics.fmean(T), statistics.fmean(M), statistics.fmean(Me),
            statistics.fmean(C), U)


for label, kw in [
    ("基线 obs=3", {}),
    ("obs=2", dict(obs=2)),
    ("obs=2+adapt40", dict(obs=2, adapt=40.0)),
    ("obs=3+adapt60", dict(adapt=60.0)),
    ("obs=3+adapt120", dict(adapt=120.0)),
]:
    cells, bad = [], 0
    for n in (10, 16):
        t, m, me, c, u = bench(n, **kw)
        cells.append(f"N{n}:{t:6.1f}/{m:5.2f}/{me:5.0f}/{c:5.1f}")
        bad += u
    print(f"{label:16s} " + " | ".join(cells) + f"  未清={bad}")
