"""实验A：early=1（16 源都 ≥1 bearing 即停）实测收益 vs early=2（当前）。"""
import statistics
import sys

sys.path.insert(0, r"G:\C\CUMCM B题")

from src.mock_simulator import MockSimulator, random_case  # noqa: E402
from src.strategy_p4 import StrategyP4  # noqa: E402


class V(StrategyP4):
    def __init__(self, early=2, **kw):
        super().__init__(**kw)
        self.early_stop_bearings = early


def bench(ratio, n, early, seeds=30, **kw):
    T, M, U, DD = [], [], 0, 0
    for s in range(seeds):
        sim = MockSimulator(random_case(n=n, seed=s, directional_ratio=ratio),
                            seed=s)
        r = V(early=early, **kw).run(sim)
        T.append(r.mean_clear_time)
        M.append(r.move_distance / 1000)
        DD += r.total
        U += len(r.unresolved)
    return statistics.fmean(T), statistics.fmean(M), U, DD / seeds


print("early=1 vs early=2（30 例/档，全定向比例 1.0 / 0.5 / 0.0，N=16）")
for ratio, tag in [(1.0, "全定向"), (0.5, "半定向"), (0.0, "全向")]:
    for early in (2, 1):
        t, m, u, dd = bench(ratio, 16, early)
        print(f"  {tag} early={early}: {t:6.1f} s/源  行程{m:5.2f}km  "
              f"未清{u}  总耗时{dd:7.0f}s")
