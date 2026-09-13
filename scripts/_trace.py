"""追踪：主循环每次访问的点，以及 route 里 full 覆盖点是否齐全。"""
import sys

sys.path.insert(0, r"G:\C\CUMCM B题")

from src.geometry import distance
from src.mock_simulator import MockSimulator, random_case
from src.strategy_p3 import StrategyP3

TARGET = (-623.5, 781.8)


class Rec(StrategyP3):
    def _sweep(self, t, p, c, b, cl, r):
        tag = "  <== 目标点!" if distance(p, TARGET) < 5 else ""
        print(f"  sweep ({p[0]:6.0f},{p[1]:6.0f}){tag}")
        return super()._sweep(t, p, c, b, cl, r)

    def _probe(self, t, p, c, b, cl, r):
        tag = "  <== 目标点!" if distance(p, TARGET) < 5 else ""
        print(f"  probe ({p[0]:6.0f},{p[1]:6.0f}){tag}")
        return super()._probe(t, p, c, b, cl, r)

    def _fast_clear(self, t, ch, est, rec, r):
        tag = "  <== 目标点!" if distance(est, TARGET) < 5 else ""
        print(f"  clear ch{ch} @ ({est[0]:6.0f},{est[1]:6.0f}){tag}")
        return super()._fast_clear(t, ch, est, rec, r)

    def _schedule_sources(self, route, index, *a):
        has = any(distance(p, TARGET) < 5 for p, _, _ in route)
        super()._schedule_sources(route, index, *a)
        has2 = any(distance(p, TARGET) < 5 for p, _, _ in route)
        if has and not has2:
            print(f"  [schedule] index={index} 目标点消失!")

    def _prune_coverage(self, route, index, swept):
        has = any(distance(p, TARGET) < 5 for p, _, _ in route)
        super()._prune_coverage(route, index, swept)
        has2 = any(distance(p, TARGET) < 5 for p, _, _ in route)
        if has and not has2:
            print(f"  [prune] index={index} 目标点被剪!")


jam = random_case(seed=2)
sim = MockSimulator(jam, seed=2)
st = Rec()
res = st.run(sim)
print(f"\ncleared={res.cleared}/{res.total}")
