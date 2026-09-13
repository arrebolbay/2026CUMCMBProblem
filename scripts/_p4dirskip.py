"""实验：问题4 定向源"朝向半平面外"检测跳过（减少 no_signal 浪费）。"""
import math
import statistics
import sys

sys.path.insert(0, r"G:\C\CUMCM B题")

from src.geometry import distance
from src.mock_simulator import MockSimulator, random_case
from src.strategy_p4 import StrategyP4, bearing_least_squares


class V(StrategyP4):
    def __init__(self, insert=400.0, dir_skip=False, deg=95.0):
        super().__init__(insert_threshold=insert)
        self.dir_skip = dir_skip
        self.dir_deg = deg

    def _directional_invisible(self, point, records):
        """定向源在 point 处是否**一定 no_signal**（保守：只用检测点正约束）。

        源被检测点 d_i 看到 ⇒ (d_i - G)·u >= 0，即源的朝向 u 落在"所有检测点方向
        的 ±90° 交集"内。若 point 到估计 G 的方向与**每个**检测点方向都 >dir_deg°，
        则对任何可行朝向 u 都有 (point-G)·u < 0 ⇒ point 一定 no_signal，可跳过。
        只用正约束 ⇒ 保守（只会跳过"确定 no_signal"的，不会漏）。
        """
        if len(records) < 2:
            return False
        est = bearing_least_squares(records)
        if est is None:
            return False
        dirs = [math.atan2(est[1] - p[1], est[0] - p[0]) for p, _ in records]
        tp = math.atan2(est[1] - point[1], est[0] - point[0])
        for d in dirs:
            diff = abs((tp - d + math.pi) % (2 * math.pi) - math.pi)
            if diff <= math.radians(self.dir_deg):
                return False
        return True

    def _sweep(self, transport, point, channels, bearings, cleared, result):
        done = 0
        for ch in channels:
            if ch in cleared or self._observations_complete(bearings[ch]):
                continue
            if bearings[ch] and self._beyond_reachable(point, bearings[ch]):
                continue
            if self.dir_skip and bearings[ch] and \
                    self._directional_invisible(point, bearings[ch]):
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


print("问题4 朝向跳过扫描（全定向 N=16，15 例/档）")
for label, kw in [
    ("基线(不跳过)", {}),
    ("朝向跳过95°", dict(dir_skip=True, deg=95.0)),
    ("朝向跳过100°", dict(dir_skip=True, deg=100.0)),
    ("朝向跳过110°", dict(dir_skip=True, deg=110.0)),
    ("朝向跳过120°", dict(dir_skip=True, deg=120.0)),
]:
    t, m, me, u = bench(1.0, 16, **kw)
    print(f"{label:16s} {t:6.1f}s 行程{m:5.2f}km 检测{me:4.0f}次 未清{u}")
