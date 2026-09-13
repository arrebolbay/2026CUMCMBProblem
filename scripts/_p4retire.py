"""实验：问题4 E1 频道退休（第一次 direction 后从主扫描退休）+ 收尾补测方式。"""
import statistics
import sys

sys.path.insert(0, r"G:\C\CUMCM B题")

from src.mock_simulator import MockSimulator, random_case
from src.strategy_p4 import StrategyP4


class V(StrategyP4):
    """E1：主扫描只测零观测频道；已发现源退休，收尾补第二条示向度。"""

    def __init__(self, insert=400.0, retire=True, pursuit=False, early=2):
        super().__init__(insert_threshold=insert)
        self.retire = retire
        self.use_pursuit = pursuit
        self.early_stop_bearings = early

    def _sweep(self, transport, point, channels, bearings, cleared, result):
        done = 0
        for ch in channels:
            if ch in cleared:
                continue
            # 频道退休：第一次 direction 后，从主扫描集合退休（不再承担"发现"任务）
            if self.retire and bearings[ch]:
                continue
            if bearings[ch] and self._beyond_reachable(point, bearings[ch]):
                continue
            resp = transport.measure(point, ch)
            result.measures += 1
            done += 1
            kind = resp.get("measure_result")
            if kind == "direction":
                bearings[ch].append((point, float(resp["svd_deg"])))
            elif kind == "near" and self._try_clear(transport, ch, point, result):
                cleared.add(ch)
        return done

    def _resolve_channel(self, transport, channel, observations, result):
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


print("E1 频道退休（全定向 N=16，20 例/档）")
for label, kw in [
    ("基线(不退休)", dict(retire=False)),
    ("E1退休+扇形探测", dict(retire=True)),
    ("E1退休+追击", dict(retire=True, pursuit=True)),
    ("E1+追击+early1", dict(retire=True, pursuit=True, early=1)),
]:
    t, m, me, u = bench(1.0, 16, **kw)
    print(f"{label:18s} {t:6.1f}s 行程{m:5.2f}km 检测{me:4.0f}次 未清{u}")
