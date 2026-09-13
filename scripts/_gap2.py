"""精确 gap 分析：实际行程 vs 同点集最优开放巡游 vs 理想点集最优开放巡游。"""
import statistics
import sys

sys.path.insert(0, r"G:\C\CUMCM B题")

from src.coverage import seven_point_design
from src.geometry import distance
from src.mock_simulator import MockSimulator, random_case
from src.strategy_p3 import StrategyP3
from src.routing import tour_length


def open_tour_len(start, points, rounds=15):
    """开放巡游长度（最近邻 + 2-opt + Or-opt，多起点取最短）。"""
    best = None
    for s in [start] + list(points)[:0] + [None]:
        tour = [start]
        rest = list(points)
        while rest:
            nxt = min(rest, key=lambda q: distance(tour[-1], q))
            rest.remove(nxt)
            tour.append(nxt)
        n = len(tour)
        for _ in range(rounds):
            imp = False
            for i in range(n - 2):
                for j in range(i + 2, n - 1):
                    a, b, c, d = tour[i], tour[i + 1], tour[j], tour[j + 1]
                    if distance(a, b) + distance(c, d) > distance(a, c) + distance(b, d) + 1e-9:
                        tour[i + 1:j + 1] = tour[i + 1:j + 1][::-1]
                        imp = True
            for seg in (1, 2, 3):
                s2 = 1
                while s2 + seg < len(tour):
                    blk = tour[s2:s2 + seg]
                    prev = tour[s2 - 1]
                    nxt = tour[s2 + seg] if s2 + seg < len(tour) else None
                    out = distance(prev, blk[0])
                    if nxt is not None:
                        out += distance(blk[-1], nxt) - distance(prev, nxt)
                    rest2 = tour[:s2] + tour[s2 + seg:]
                    best2 = None
                    for t in range(len(rest2) - 1):
                        u, v = rest2[t], rest2[t + 1]
                        add = distance(u, blk[0]) + distance(blk[-1], v) - distance(u, v)
                        if best2 is None or add < best2[0]:
                            best2 = (add, t + 1)
                    tail = distance(rest2[-1], blk[0])
                    if best2 is None or tail < best2[0]:
                        best2 = (tail, len(rest2))
                    if best2[0] < out - 1e-9:
                        tour = rest2[:best2[1]] + blk + rest2[best2[1]:]
                        imp = True
                    else:
                        s2 += 1
            if not imp:
                break
        L = sum(distance(tour[i], tour[i + 1]) for i in range(n - 1))
        if best is None or L < best:
            best = L
    return best


for n in (10, 16):
    actual, same, ideal = [], [], []
    for seed in range(8):
        jam = random_case(n=n, seed=seed)
        sim = MockSimulator(jam, seed=seed)
        st = StrategyP3()
        res = st.run(sim)
        actual.append(res.move_distance / 1000)
        # 同点集：收集到访点（从日志）
        pts = set()
        for e in sim.log:
            if e["path"] in ("/measure", "/clear"):
                pts.add((round(e["position"][0], 1), round(e["position"][1], 1)))
        same.append(open_tour_len((0.0, 0.0), list(pts)) / 1000)
        # 理想点集：8 覆盖点 + 真源
        cover = list(seven_point_design(1000.0, 7))
        ideal.append(open_tour_len((0.0, 0.0),
                                   cover + [tuple(j.position) for j in jam]) / 1000)
    a, s, i = (statistics.fmean(x) for x in (actual, same, ideal))
    print(f"N={n}: 实际 {a:5.2f} km | 同点集最优 {s:5.2f} km | 理想点集最优 {i:5.2f} km")
    print(f"   排序损失 {a-s:5.2f} km ({(a-s)*1000/5:.0f} s) | 点集损失 {s-i:5.2f} km ({(s-i)*1000/5:.0f} s)")
