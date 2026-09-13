"""开放巡游（路径）规划工具：最近邻构造 + 2-opt 改进。

问题3/问题4 都要在"当前点 → 若干目标点"之间规划路线：
  * 问题4：固定的 25 点定向安全网巡游；
  * 问题3：自适应覆盖过程中反复重规划"已定位干扰源 + 覆盖缺口点"。
故抽出为共享模块，避免 strategy_p3 与 strategy_p4 之间循环导入。
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

Point = Tuple[float, float]

__all__ = ["two_opt_tour", "tour_length"]


def tour_length(order: Sequence[Point]) -> float:
    """开放巡游总长度 (m)。"""
    total = 0.0
    for a, b in zip(order, order[1:]):
        total += ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
    return total


def two_opt_tour(points: Sequence[Point], max_rounds: int = 40) -> List[Point]:
    """最近邻 + 2-opt 的开放式巡游（从第一个点出发，不回到起点）。"""
    pts = [(float(p[0]), float(p[1])) for p in points]
    if len(pts) <= 2:
        return pts
    unvisited = pts[1:]
    tour = [pts[0]]
    while unvisited:
        last = tour[-1]
        nxt = min(unvisited, key=lambda q: (last[0] - q[0]) ** 2 + (last[1] - q[1]) ** 2)
        tour.append(nxt)
        unvisited.remove(nxt)

    improved = True
    rounds = 0
    while improved and rounds < max_rounds:
        improved = False
        rounds += 1
        n = len(tour)
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                a, b, c = tour[i - 1], tour[i], tour[j]
                if j + 1 >= n:
                    continue
                d = tour[j + 1]
                delta = (((a[0] - c[0]) ** 2 + (a[1] - c[1]) ** 2) ** 0.5
                         + ((b[0] - d[0]) ** 2 + (b[1] - d[1]) ** 2) ** 0.5
                         - ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
                         - ((c[0] - d[0]) ** 2 + (c[1] - d[1]) ** 2) ** 0.5)
                if delta < -1e-9:
                    tour[i:j + 1] = reversed(tour[i:j + 1])
                    improved = True
    return tour
