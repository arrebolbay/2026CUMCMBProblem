"""实验：问题4 优化 —— obs 自适应 + insert 阈值 + 检测跳过。"""
import statistics
import sys

sys.path.insert(0, r"G:\C\CUMCM B题")

from src.geometry import localization_region, min_enclosing_circle, distance
from src.mock_simulator import MockSimulator, random_case
from src.strategy_p4 import StrategyP4


class V(StrategyP4):
    def __init__(self, insert=400.0, adapt=None, obs=None):
        super().__init__(insert_threshold=insert)
        self.adapt = adapt
        if obs is not None:
            self.observations_target = obs

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


def bench(ratio, n, seeds=15, **kw):
    T, M, Me, U = [], [], [], 0
    for seed in range(seeds):
        sim = MockSimulator(random_case(n=n, seed=seed, directional_ratio=ratio),
                            seed=seed)
        r = V(**kw).run(sim)
        T.append(r.mean_clear_time)
        M.append(r.move_distance / 1000)
        Me.append(r.measures)
        U += len(r.unresolved)
    return (statistics.fmean(T), statistics.fmean(M), statistics.fmean(Me), U)


print("问题4 优化扫描（全定向 N=16，15 例/档）")
for label, kw in [
    ("基线 obs=3", {}),
    ("obs=2", dict(obs=2)),
    ("obs=3+adapt60", dict(adapt=60.0)),
    ("obs=3+adapt120", dict(adapt=120.0)),
    ("obs=3+adapt200", dict(adapt=200.0)),
]:
    t, m, me, u = bench(1.0, 16, **kw)
    print(f"{label:16s} {t:6.1f}s 行程{m:5.2f}km 检测{me:4.0f}次 未清{u}")
