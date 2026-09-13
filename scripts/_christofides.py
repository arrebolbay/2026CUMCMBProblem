"""验证：2-opt 的 TSP 是否接近最优（对比 Christofides 图论方法）。

若 2-opt 与 Christofides 差距很小 ⇒ "排序损失"是信息论代价而非 TSP 次优性；
若差距大 ⇒ 换更强的图论 TSP 求解器可回收一部分。
"""
import sys
import statistics
import itertools

sys.path.insert(0, r"G:\C\CUMCM B题")

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csr_matrix

from src.coverage import seven_point_design
from src.geometry import distance
from src.mock_simulator import random_case
from src.routing import two_opt_tour, tour_length


def christofides(points):
    """Christofides 近似 TSP（图论：MST + 奇度点最小权完美匹配 + 欧拉回路 + 捷径）。"""
    pts = [tuple(p) for p in points]
    n = len(pts)
    if n < 3:
        return sum(distance(pts[i], pts[(i + 1) % n]) for i in range(n))
    D = np.array([[distance(pts[i], pts[j]) for j in range(n)] for i in range(n)])
    # MST
    mst = minimum_spanning_tree(csr_matrix(D)).toarray()
    # 奇度点
    deg = (mst > 0).sum(axis=1)
    odd = [i for i in range(n) if deg[i] % 2 == 1]
    # 贪心完美匹配（近似最小权匹配）
    matched = [False] * n
    for i in odd:
        if matched[i]:
            continue
        j = min((k for k in odd if k != i and not matched[k]),
                key=lambda k: D[i, k])
        matched[i] = matched[j] = True
        mst[i, j] = mst[j, i] = 1.0
    # Hierholzer 欧拉回路（多重边 → 邻接列表）
    adj = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            if mst[i, j] > 0:
                adj[i].append(j)
                adj[j].append(i)
    # 找欧拉回路（迭代）
    stack = [0]
    circuit = []
    while stack:
        v = stack[-1]
        if adj[v]:
            w = adj[v].pop()
            adj[w].remove(v)
            stack.append(w)
        else:
            circuit.append(stack.pop())
    # 捷径
    seen = set()
    order = []
    for v in circuit:
        if v not in seen:
            seen.add(v)
            order.append(v)
    return sum(distance(pts[order[i]], pts[order[(i + 1) % len(order)]])
               for i in range(len(order)))


for n in (10, 16):
    t2 = tc = []
    t2, tc = [], []
    for seed in range(12):
        jam = random_case(n=n, seed=seed)
        cover = list(seven_point_design(1000.0, 7))
        pts = [(0.0, 0.0)] + cover + [tuple(j.position) for j in jam]
        # 闭合 2-opt
        l2 = tour_length(two_opt_tour(pts)) + distance(pts[-1], pts[0])
        # Christofides
        lc = christofides(pts)
        t2.append(l2 / 1000)
        tc.append(lc / 1000)
    a2, ac = statistics.fmean(t2), statistics.fmean(tc)
    print(f"N={n}: 2-opt 闭合 {a2:.2f} km | Christofides {ac:.2f} km | "
          f"2-opt 超出 {(a2/ac-1)*100:.1f}%")
