"""实验：问题4 更早停止（early_stop_bearings=1）+ 收尾 1-bearing 用追击。"""
import statistics
import sys

sys.path.insert(0, r"G:\C\CUMCM B题")

from src.mock_simulator import MockSimulator, random_case
from src.strategy_p4 import StrategyP4


class V(StrategyP4):
    def __init__(self, insert=400.0, early=2, pursuit=False):
        super().__init__(insert_threshold=insert)
        self.early_stop_bearings = early
        self.use_pursuit = pursuit

    def _resolve_channel(self, transport, channel, observations, result):
        """1 条示向度：优先沿示向度追击（定向源沿轴走必在朝向半平面内）。"""
        if self.use_pursuit and len(observations) == 1:
            b, br = observations[-1]
            if self._pursuit_clear(transport, channel, b, br, result):
                return True
        return super()._resolve_channel(transport, channel, observations, result)


def bench(ratio, n, seeds=20, **kw):
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


print("问题4 更早停止扫描（全定向 N=16，20 例/档）")
for label, kw in [
    ("基线 early=2", dict(early=2)),
    ("early=1 无追击", dict(early=1)),
    ("early=1 + 追击", dict(early=1, pursuit=True)),
]:
    t, m, me, u = bench(1.0, 16, **kw)
    print(f"{label:16s} {t:6.1f}s 行程{m:5.2f}km 检测{me:4.0f}次 未清{u}")
# 半定向也测
for label, kw in [
    ("半定向 early=2", dict(early=2)),
    ("半定向 early=1+追击", dict(early=1, pursuit=True)),
]:
    t, m, me, u = bench(0.5, 16, **kw)
    print(f"{label:16s} {t:6.1f}s 未清{u}")
