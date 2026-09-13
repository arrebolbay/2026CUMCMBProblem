"""诊断：多少源被收尾处理、收尾行程多大、哪些源进收尾。"""
import collections
import statistics
import sys

sys.path.insert(0, r"G:\C\CUMCM B题")

from src.coverage import seven_point_design
from src.geometry import distance
from src.mock_simulator import MockSimulator, random_case
from src.strategy_p3 import StrategyP3

for n in (10, 16):
    main_clear = tail_clear = 0
    tail_travel = []
    tail_bearings = []
    for seed in range(15):
        jam = random_case(n=n, seed=seed)
        sim = MockSimulator(jam, seed=seed)
        st = StrategyP3()
        res = st.run(sim)
        ring = {(round(p[0], 1), round(p[1], 1)) for p in seven_point_design(1000.0, 7)}
        # 找最后一个覆盖点扫描的日志位置
        last_cov = 0
        for i, e in enumerate(sim.log):
            if e["path"] == "/measure" and (round(e["position"][0], 1), round(e["position"][1], 1)) in ring:
                last_cov = i
        # 统计：主路线（<= last_cov）清除的频道 vs 收尾清除的频道
        cleared_in_main = set()
        cleared_in_tail = set()
        tail_move = 0.0
        pos = None
        # 重新走一遍日志累计行程
        p = (0.0, 0.0)
        for i, e in enumerate(sim.log):
            if e["path"] not in ("/measure", "/clear"):
                continue
            d = distance(p, tuple(e["position"]))
            if i > last_cov:
                tail_move += d
            p = tuple(e["position"])
            if e["path"] == "/clear" and e.get("clear_result") == "success":
                if i <= last_cov:
                    cleared_in_main.add(e["channel"])
                else:
                    cleared_in_tail.add(e["channel"])
        main_clear += len(cleared_in_main)
        tail_clear += len(cleared_in_tail)
        tail_travel.append(tail_move / 1000)
        # 尾程源的示向度条数
        bearings = collections.defaultdict(int)
        for i, e in enumerate(sim.log):
            if e["path"] == "/measure" and e.get("measure_result") == "direction" and i <= last_cov:
                bearings[e["channel"]] += 1
        tail_bearings += [bearings[c] for c in cleared_in_tail]
    print(f"N={n}: 主路线清除 {main_clear/15:.1f} 个源 | 收尾清除 {tail_clear/15:.1f} 个源 | "
          f"收尾行程 {statistics.fmean(tail_travel):.2f} km | 尾程源在覆盖扫描结束时的示向度数 "
          f"{statistics.fmean(tail_bearings) if tail_bearings else 0:.1f}")
