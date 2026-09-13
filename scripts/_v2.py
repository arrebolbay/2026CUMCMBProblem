"""问题3 v2 验证：追击模型 + 无复测点（8 点覆盖骨干）。"""
import statistics
import sys

sys.path.insert(0, r"G:\C\CUMCM B题")

from src.mock_simulator import MockSimulator, random_case
from src.strategy_p3 import StrategyP3


class V2(StrategyP3):
    """v2：8 个覆盖扫描点、无复测点，追击处理 1 示向度/细长源。"""

    def __init__(self):
        super().__init__()
        self.cover_ring_count = 7          # center + 7 环 = 8 点
        self.cover_probe_stride = 1        # 全部作为覆盖扫描点
        self.probe_keep = 0                # 无几何复测点


def bench(cls, n, seeds=20, **kw):
    T, M, Me, C, U = [], [], [], [], 0
    for seed in range(seeds):
        sim = MockSimulator(random_case(n=n, seed=seed), seed=seed)
        r = cls(**kw).run(sim)
        T.append(r.mean_clear_time)
        M.append(r.move_distance / 1000)
        Me.append(r.measures)
        C.append(r.clears + r.failed_clears)
        U += len(r.unresolved)
    return (statistics.fmean(T), statistics.fmean(M), statistics.fmean(Me),
            statistics.fmean(C), U)


print("格式：时间s / 行程km / 检测 / 清除动作 / 未清")
for label, cls in (("v1 基线", StrategyP3), ("v2 追击无复测", V2)):
    cells, bad = [], 0
    for n in (10, 12, 14, 16):
        t, m, me, c, u = bench(cls, n)
        cells.append(f"N{n}:{t:6.1f}/{m:5.2f}/{me:5.0f}/{c:5.1f}")
        bad += u
    print(f"{label:14s} " + " | ".join(cells) + f"  未清={bad}")
