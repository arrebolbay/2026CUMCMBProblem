"""精确行程账本：把问题3 总行程拆到"骨干/复测/源访问/近场收口/收尾"五类。"""
import collections
import statistics
import sys

sys.path.insert(0, r"G:\C\CUMCM B题")

from src.coverage import seven_point_design
from src.geometry import distance
from src.mock_simulator import MockSimulator, random_case
from src.strategy_p3 import StrategyP3


def ledger(n, seed):
    jam = random_case(n=n, seed=seed)
    sim = MockSimulator(jam, seed=seed)
    src = {j.channel: tuple(j.position) for j in jam}
    ring = seven_point_design(1000.0, 7)
    sweep_pts = {(round(p[0], 1), round(p[1], 1)) for p in ring}
    st = StrategyP3()
    res = st.run(sim)

    acc = collections.Counter()
    pos = (0.0, 0.0)
    last_kind = "other"
    for e in sim.log:
        if e["path"] not in ("/measure", "/clear"):
            continue
        p = tuple(e["position"])
        d = distance(pos, p)
        key = (round(p[0], 1), round(p[1], 1))
        if key in sweep_pts:
            kind = "cover"      # 到覆盖扫描点
        elif e["path"] == "/measure" and e.get("channel") in src:
            kind = "src_meas"   # 到源附近的复测
        elif e["path"] == "/clear" and e.get("channel") in src:
            g = src[e["channel"]]
            if distance(p, g) <= 20:
                kind = "src_hit"   # 到真源(命中)
            else:
                kind = "src_miss"  # 到估计点(未命中)
        else:
            kind = "other"
        acc[kind] += d
        pos = p
    return acc, res


def main():
    for n in (10, 16):
        A = collections.Counter()
        T, M, Me, C, U = [], [], [], [], 0
        for seed in range(20):
            acc, res = ledger(n, seed)
            A.update(acc)
            T.append(res.mean_clear_time)
            M.append(res.move_distance / 1000)
            Me.append(res.measures)
            C.append(res.clears + res.failed_clears)
            U += len(res.unresolved)
        tot = sum(A.values()) / 20 / 1000
        print(f"N={n}: {statistics.fmean(T):6.1f} s/源  总行程 {tot:5.2f} km")
        for k in ("cover", "src_hit", "src_miss", "src_meas", "other"):
            v = A[k] / 20 / 1000
            print(f"    {k:10s} {v:5.2f} km")
        print(f"    检测 {statistics.fmean(Me):5.1f} 次  清除动作 {statistics.fmean(C):5.1f}  未清 {U}")


main()
