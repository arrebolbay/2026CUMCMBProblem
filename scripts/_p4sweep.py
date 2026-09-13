"""诊断：问题4 清满 16 时实际扫了多少覆盖点（巡游能否提前结束）。"""
import statistics
import sys

sys.path.insert(0, r"G:\C\CUMCM B题")

from src.coverage import directional_survey_design
from src.mock_simulator import MockSimulator, random_case
from src.strategy_p4 import StrategyP4


class Rec(StrategyP4):
    def __init__(self, insert=400.0):
        super().__init__(insert_threshold=insert)
        self.nswept = 0
        self.sweep_pts = []

    def _sweep(self, t, p, c, b, cl, r):
        self.sweep_pts.append((round(p[0], 1), round(p[1], 1)))
        return super()._sweep(t, p, c, b, cl, r)


pts = [(0.0, 0.0)] + list(directional_survey_design())
print(f"覆盖网点数: {len(pts)}")
for ratio, tag in [(1.0, "全定向"), (0.0, "全向")]:
    for n in (10, 16):
        ns = []
        T = []
        U = 0
        for seed in range(15):
            sim = MockSimulator(random_case(n=n, seed=seed, directional_ratio=ratio),
                                seed=seed)
            st = Rec()
            r = st.run(sim)
            # 统计扫描的"覆盖点"数量（去重后与设计点匹配的）
            swept = {(round(p[0], 1), round(p[1], 1)) for p in st.sweep_pts}
            design = {(round(p[0], 1), round(p[1], 1)) for p in pts}
            ns.append(len(swept & design))
            T.append(r.mean_clear_time)
            U += len(r.unresolved)
        print(f"{tag} N={n}: 扫{statistics.fmean(ns):.1f}个覆盖点/{len(pts)} 时间{statistics.fmean(T):.1f}s 未清{U}")
