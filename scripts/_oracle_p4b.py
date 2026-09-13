"""Oracle2：问题4「覆盖+清除合并巡游」vs「覆盖巡游 + 收尾清除」（分离）。

问题：把 25 点覆盖与 16 个源清除合成一条 TSP，理论上比"先扫完 25 点、再收尾
清源"省多少？（这就是"清源路程反哺覆盖"的上限；源位置用真值，仅离线估计）
"""
import statistics
import sys

sys.path.insert(0, r"G:\C\CUMCM B题")

from src.coverage import directional_survey_design
from src.geometry import distance
from src.mock_simulator import random_case


def tour_len(start, points, rounds=8, warm=3):
    """开放巡游长度（NN + 2-opt + Or-opt，多起点取最短）。"""
    pts = list(points)
    if not pts:
        return 0.0
    best = None
    for k in range(min(warm, len(pts))):
        head = pts[k]
        tour = [start, head]
        pool = pts[:k] + pts[k + 1:]
        while pool:
            nxt = min(pool, key=lambda q: distance(tour[-1], q))
            pool.remove(nxt)
            tour.append(nxt)
        n = len(tour)
        for _ in range(rounds):
            imp = False
            for i in range(n - 2):
                for j in range(i + 2, n - 1):
                    a, b, c, d = tour[i], tour[i + 1], tour[j], tour[j + 1]
                    if (distance(a, b) + distance(c, d)
                            > distance(a, c) + distance(b, d) + 1e-9):
                        tour[i + 1:j + 1] = tour[i + 1:j + 1][::-1]
                        imp = True
            for seg in (1, 2, 3):
                s = 1
                while s + seg < len(tour):
                    blk = tour[s:s + seg]
                    prev = tour[s - 1]
                    nxt = tour[s + seg] if s + seg < len(tour) else None
                    out = distance(prev, blk[0])
                    if nxt is not None:
                        out += distance(blk[-1], nxt) - distance(prev, nxt)
                    rest = tour[:s] + tour[s + seg:]
                    b2 = None
                    for t in range(len(rest) - 1):
                        u, v = rest[t], rest[t + 1]
                        add = (distance(u, blk[0]) + distance(blk[-1], v)
                               - distance(u, v))
                        if b2 is None or add < b2[0]:
                            b2 = (add, t + 1)
                    tail = distance(rest[-1], blk[0])
                    if b2 is None or tail < b2[0]:
                        b2 = (tail, len(rest))
                    if b2[0] < out - 1e-9:
                        tour = rest[:b2[1]] + blk + rest[b2[1]:]
                        imp = True
                    else:
                        s += 1
            if not imp:
                break
        L = sum(distance(tour[i], tour[i + 1]) for i in range(len(tour) - 1))
        if best is None or L < best:
            best = L
    return best


design = [(0.0, 0.0)] + list(directional_survey_design())
for ratio, tag in [(1.0, "全定向"), (0.0, "全向")]:
    base_all, merged_all = [], []
    for seed in range(15):
        jam = random_case(n=16, seed=seed, directional_ratio=ratio)
        src = [tuple(j.position) for j in jam]
        # 分离式（当前范式）：25 点巡游 + 从终点收尾清源
        sep = tour_len((0.0, 0.0), design) + tour_len(design[-1], src)
        # 合并式（Oracle）：25 点 + 16 源 一条 TSP
        mer = tour_len((0.0, 0.0), design + src)
        base_all.append(sep / 1000)
        merged_all.append(mer / 1000)
    a, b = statistics.fmean(base_all), statistics.fmean(merged_all)
    print(f"{tag} N=16: 分离 {a:5.2f} km | 合并(理想) {b:5.2f} km | "
          f"可回收 {a-b:5.2f} km = {(a-b)*1000/5/16:5.1f} s/源")
