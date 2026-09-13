"""诊断：seed=2 漏清频道 11 —— 它距所有 swept 点(起点+sweep+discharge) 的真实距离。"""
import sys

sys.path.insert(0, r"G:\C\CUMCM B题")

from src.geometry import distance
from src.mock_simulator import MockSimulator, random_case
from src.strategy_p3 import StrategyP3


class Rec(StrategyP3):
    def __init__(self):
        super().__init__()
        self.swept_all = []
        self.discharge = []
        self.pruned = []

    def _sweep(self, transport, point, channels, bearings, cleared, result):
        n = super()._sweep(transport, point, channels, bearings, cleared, result)
        self.swept_all.append(("sweep", tuple(point)))
        return n

    def _discharge_sweep(self, transport, point, channels, bearings, cleared, result):
        n = super()._discharge_sweep(transport, point, channels, bearings, cleared, result)
        self.discharge.append(tuple(point))
        self.swept_all.append(("discharge", tuple(point)))
        return n

    def _prune_coverage(self, route, index, swept):
        before = [(p, ch, f) for p, ch, f in route if f]
        super()._prune_coverage(route, index, swept)
        after = [(p, ch, f) for p, ch, f in route if f]
        removed = [p for p, _, _ in before if p not in [q for q, _, _ in after]]
        for p in removed:
            self.pruned.append(p)


jam = random_case(seed=2)
sim = MockSimulator(jam, seed=2)
st = Rec()
res = st.run(sim)

src11 = [tuple(j.position) for j in jam if j.channel == 11][0]
print("源11", tuple(round(v, 1) for v in src11))
print(f"\n剪掉的环点数: {len(st.pruned)}")
for p in st.pruned:
    print(f"  剪掉 {tuple(round(v,1) for v in p)} 距源11 {distance(p, src11):.0f} m")
print(f"\ndischarge 点数: {len(st.discharge)}")
dmin = min((distance(p, src11) for p in st.discharge), default=float("inf"))
print(f"  源11 距最近 discharge 点 {dmin:.1f} m")
print(f"\n全部 swept 点 {len(st.swept_all)} 个，源11 距最近 swept 点 "
      f"{min(distance(p, src11) for _, p in st.swept_all):.1f} m")
