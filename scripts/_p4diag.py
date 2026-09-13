"""诊断：问题4 的扫描点数、源发现时机、检测构成、收尾返程。"""
import collections
import statistics
import sys

sys.path.insert(0, r"G:\C\CUMCM B题")

from src.geometry import distance
from src.mock_simulator import MockSimulator, random_case
from src.strategy_p4 import StrategyP4


def diag(ratio, n, seeds=15):
    scanned = []       # 清满 16 时扫了多少个覆盖点
    first_obs = []     # 每个源在第几个扫描点首次被发现
    meas_kind = collections.Counter()
    tail_move = []
    T = []
    for seed in range(seeds):
        jam = random_case(n=n, seed=seed, directional_ratio=ratio)
        sim = MockSimulator(jam, seed=seed)
        res = StrategyP4(insert_threshold=400.0).run(sim)
        T.append(res.mean_clear_time)
        # 统计扫描点（sweep 是全频道扫描；P4 里每个 survey point 都 sweep）
        pts = [(0.0, 0.0)] + list(StrategyP4().survey_points)
        # 遍历日志：统计 measure 的 result 分布 + 每个频道的首次 direction 时机
        ncov = 0
        seen_cov = set()
        first = {}
        last_cover_idx = 0
        for i, e in enumerate(sim.log):
            if e["path"] != "/measure":
                continue
            meas_kind[e.get("measure_result")] += 1
            key = (round(e["position"][0], 1), round(e["position"][1], 1))
            if key not in seen_cov:
                seen_cov.add(key)
                ncov += 1
            if e.get("measure_result") == "direction" and e["channel"] not in first:
                first[e["channel"]] = ncov
        first_obs += list(first.values())
        # 收尾返程：最后一个覆盖点之后的行程
        ring = {(round(p[0], 1), round(p[1], 1)) for p in pts}
        last = 0
        for i, e in enumerate(sim.log):
            if e["path"] == "/measure" and (round(e["position"][0], 1), round(e["position"][1], 1)) in ring:
                last = i
        p = (0.0, 0.0)
        tm = 0.0
        for i, e in enumerate(sim.log):
            if e["path"] not in ("/measure", "/clear"):
                continue
            d = distance(p, tuple(e["position"]))
            if i > last:
                tm += d
            p = tuple(e["position"])
        tail_move.append(tm / 1000)
        scanned.append(ncov)
    print(f"{ratio*100:.0f}%定向 N={n}: 平均 {statistics.fmean(T):.1f}s "
          f"扫{statistics.fmean(scanned):.1f}点 源首现第{statistics.fmean(first_obs):.1f}点 "
          f"收尾返程{statistics.fmean(tail_move):.2f}km")
    return meas_kind


mk = diag(1.0, 16)
print(f"  检测结果分布: {dict(mk)}")
