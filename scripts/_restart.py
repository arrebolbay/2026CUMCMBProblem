"""测试：插入后对剩余路线做"最近邻重启 + 2-opt + Or-opt"全局重排，回收排序损失。"""
import statistics
import sys

sys.path.insert(0, r"G:\C\CUMCM B题")

from src.geometry import distance
from src.mock_simulator import MockSimulator, random_case
from src.strategy_p3 import StrategyP3


class V(StrategyP3):
    """用全局重排替换局部 2-opt/Or-opt。"""

    def _improve_remaining(self, route, index, rounds=6):
        m = len(route)
        if m - index < 4:
            return
        # 提取剩余点（保持 (point, channel, full) 三元组）
        seg = route[index:]
        start = seg[0][0]
        rest = seg[1:]
        # 最近邻重启
        tour = [seg[0]]
        pool = list(rest)
        while pool:
            nxt = min(pool, key=lambda t: distance(tour[-1][0], t[0]))
            pool.remove(nxt)
            tour.append(nxt)
        n = len(tour)
        def tot_len(t):
            return sum(distance(t[i][0], t[i + 1][0]) for i in range(len(t) - 1))
        # 2-opt + Or-opt（多轮，只接受变短）
        for _ in range(rounds):
            imp = False
            for i in range(n - 2):
                for j in range(i + 2, n - 1):
                    a, b, c, d = tour[i][0], tour[i + 1][0], tour[j][0], tour[j + 1][0]
                    if distance(a, b) + distance(c, d) > distance(a, c) + distance(b, d) + 1e-9:
                        tour[i + 1:j + 1] = tour[i + 1:j + 1][::-1]
                        imp = True
            for seg_len in (1, 2, 3):
                s = 1
                while s + seg_len < len(tour):
                    blk = tour[s:s + seg_len]
                    prev = tour[s - 1][0]
                    nxt = tour[s + seg_len][0] if s + seg_len < len(tour) else None
                    out = distance(prev, blk[0][0])
                    if nxt is not None:
                        out += distance(blk[-1][0], nxt) - distance(prev, nxt)
                    rest2 = tour[:s] + tour[s + seg_len:]
                    best = None
                    for t in range(len(rest2) - 1):
                        u, v = rest2[t][0], rest2[t + 1][0]
                        add = distance(u, blk[0][0]) + distance(blk[-1][0], v) - distance(u, v)
                        if best is None or add < best[0]:
                            best = (add, t + 1)
                    tail = distance(rest2[-1][0], blk[0][0])
                    if best is None or tail < best[0]:
                        best = (tail, len(rest2))
                    if best[0] < out - 1e-9:
                        tour = rest2[:best[1]] + blk + rest2[best[1]:]
                        imp = True
                    else:
                        s += 1
            if not imp:
                break
        route[index:] = tour


def bench(n, seeds=20, **kw):
    T, M, Me, U = [], [], [], 0
    for seed in range(seeds):
        sim = MockSimulator(random_case(n=n, seed=seed), seed=seed)
        r = V(**kw).run(sim)
        T.append(r.mean_clear_time)
        M.append(r.move_distance / 1000)
        Me.append(r.measures)
        U += len(r.unresolved)
    return (statistics.fmean(T), statistics.fmean(M), statistics.fmean(Me), U)


for label, cls in (("基线(局部2opt+OrOpt)", StrategyP3), ("全局重排(NN重启)", V)):
    cells, bad = [], 0
    for n in (10, 12, 14, 16):
        t, m, me, u = bench(n)
        cells.append(f"N{n}:{t:6.1f}/{m:5.2f}/{me:5.0f}")
        bad += u
    print(f"{label:20s} " + " | ".join(cells) + f"  未清={bad}")
