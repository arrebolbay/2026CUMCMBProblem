"""实验：问题4 扫描顺序 / 巡游 TSP 的变体（影响"源发现时机"与总距离）。"""
import math
import statistics
import sys

sys.path.insert(0, r"G:\C\CUMCM B题")

from src.geometry import distance
from src.mock_simulator import MockSimulator, random_case
from src.routing import two_opt_tour
from src.strategy_p4 import StrategyP4


def strong_tour(start, pts, rounds=10, warm=4):
    """NN + 2-opt + Or-opt，多起点取最短（比 two_opt_tour 更强）。"""
    pts = list(pts)
    if not pts:
        return [start]
    best, bestL = None, None
    for k in range(min(warm, len(pts))):
        tour = [start, pts[k]]
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
                    prev, nxt = tour[s - 1], (tour[s + seg]
                                              if s + seg < len(tour) else None)
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
        if bestL is None or L < bestL:
            best, bestL = tour, L
    return best


def by_ring(pts):
    """按半径分组：中心 / 内环 / 外环。"""
    c = [p for p in pts if math.hypot(*p) < 1.0]
    mid = [p for p in pts if 1.0 <= math.hypot(*p) < 1400.0]
    out = [p for p in pts if math.hypot(*p) >= 1400.0]
    key = lambda q: math.atan2(q[1], q[0])
    return c, sorted(mid, key=key), sorted(out, key=key)


class V(StrategyP4):
    def __init__(self, order="tsp", **kw):
        super().__init__(**kw)
        self.order = order

    def _tour_order(self):
        pts = list(self.survey_points)
        if self.order == "tsp":
            return two_opt_tour([(0.0, 0.0)] + pts)
        if self.order == "strong":
            return strong_tour((0.0, 0.0), pts)
        c, mid, out = by_ring(pts)
        if self.order == "inner_first":
            return [(0.0, 0.0)] + mid + out
        if self.order == "outer_first":
            return [(0.0, 0.0)] + out + mid
        if self.order == "inner_first_strong":
            return strong_tour((0.0, 0.0), mid + out)
        return two_opt_tour([(0.0, 0.0)] + pts)


def bench(ratio, n, seeds=15, **kw):
    T, M, U = [], [], 0
    for seed in range(seeds):
        sim = MockSimulator(random_case(n=n, seed=seed, directional_ratio=ratio),
                            seed=seed)
        r = V(**kw).run(sim)
        T.append(r.mean_clear_time)
        M.append(r.move_distance / 1000)
        U += len(r.unresolved)
    return statistics.fmean(T), statistics.fmean(M), U


print("扫描顺序 / 巡游变体（全定向 N=16，15 例/档）")
for label, kw in [
    ("tsp(当前)", dict(order="tsp")),
    ("strong(NN+2opt+OrOpt)", dict(order="strong")),
    ("inner_first", dict(order="inner_first")),
    ("outer_first", dict(order="outer_first")),
    ("inner_first_strong", dict(order="inner_first_strong")),
]:
    t, m, u = bench(1.0, 16, **kw)
    print(f"{label:22s} {t:6.1f}s 行程{m:5.2f}km 未清{u}")
