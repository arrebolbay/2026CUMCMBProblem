"""问题3 v5：覆盖义务驱动（Coverage-obligation-driven）在线规划。

与 v1（固定 8 环点骨干 + 最便宜插入 + 收尾 TSP）的核心区别：

  v1 把"发现顺序"和"覆盖顺序"绑定：先走固定 7 环点，途中把源插进剩余路线。
     这天然产生"源被发现时其最优位置已走过"的结构性损失。

  v5 只保证"未发现源的可能位置 U 被已扫点集覆盖"：每一步动态生成下一扫描点
     （贪心集合覆盖），而源清除点（discharge）本身就是覆盖点，能替代环点。
     于是"清除行程"与"覆盖行程"合流，逼近 Oracle（Oracle 正是靠走到源旁边
     就完成覆盖而省掉绕环）。

保留不动：_sweep/_discharge_sweep/_fast_clear/_grid_clear/_covers/几何定位。
只重写主循环 run()：覆盖义务 + 贪心补洞 + 顺路清除。
"""
from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np

from .config import CHANNELS, REGION_RADIUS, R_MIN
from .geometry import distance, unit_vector
from .strategy_p3 import StrategyP3, StrategyResult

Point = Tuple[float, float]


class StrategyP3V5(StrategyP3):
    """覆盖义务驱动版：动态覆盖集 = 起点 + 贪心补洞点 + 源清除点(discharge)。"""

    def __init__(self):
        super().__init__()
        self.cover_duty_limit = 998.0   # 覆盖义务"满足"判定：最坏覆盖距离 <= 该值

    # ------------------------------------------------------------------ #
    def _worst_angle(self, swept) -> Tuple[float, float]:
        """边界上距 swept 最远的点：返回 (角, 最坏距离)。确定性采样。"""
        pts = np.asarray(list(swept), dtype=float)
        th = np.linspace(0.0, 2.0 * math.pi, 720, endpoint=False)
        bnd = REGION_RADIUS * np.column_stack([np.cos(th), np.sin(th)])
        d = np.linalg.norm(bnd[:, None, :] - pts[None, :, :], axis=2)
        md = d.min(axis=1)
        idx = int(md.argmax())
        return float(th[idx]), float(md[idx])

    def _next_cover(self, swept, here):
        """贪心补洞：在"最坏边界点"方向、半径 R_MIN 处放下一覆盖点。"""
        ang, worst = self._worst_angle(swept)
        if worst <= self.cover_duty_limit:
            return None
        return (R_MIN * math.cos(ang), R_MIN * math.sin(ang))

    def _covers_duty(self, swept) -> bool:
        """覆盖义务是否满足：swept 对圆域最坏覆盖距离 <= cover_duty_limit。"""
        pts = np.asarray(list(swept), dtype=float)
        th = np.linspace(0.0, 2.0 * math.pi, 14400, endpoint=False)
        bnd = REGION_RADIUS * np.column_stack([np.cos(th), np.sin(th)])
        d = np.linalg.norm(bnd[:, None, :] - pts[None, :, :], axis=2)
        return float(d.min(axis=1).max()) <= self.cover_duty_limit

    def _open_tour(self, start, nodes, rounds=6):
        """最近邻 + 2-opt 开放巡游（从 start 出发，访问所有 nodes）。"""
        tour = [start]
        pool = list(nodes)
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
            if not imp:
                break
        return tour[1:]

    def _generate_covers(self, swept):
        """贪心生成满足当前覆盖义务所需的全部覆盖点。"""
        covers = []
        tmp = list(swept)
        for _ in range(9):
            ang, worst = self._worst_angle(tmp)
            if worst <= self.cover_duty_limit:
                break
            p = (R_MIN * math.cos(ang), R_MIN * math.sin(ang))
            covers.append(p)
            tmp.append(p)
        return covers

    def _covers_duty(self, swept) -> bool:
        """覆盖义务是否满足：swept 对圆域最坏覆盖距离 <= cover_duty_limit。"""
        pts = np.asarray(list(swept), dtype=float)
        th = np.linspace(0.0, 2.0 * math.pi, 14400, endpoint=False)
        bnd = REGION_RADIUS * np.column_stack([np.cos(th), np.sin(th)])
        d = np.linalg.norm(bnd[:, None, :] - pts[None, :, :], axis=2)
        return float(d.min(axis=1).max()) <= self.cover_duty_limit

    # ------------------------------------------------------------------ #
    def run(self, transport, channel_order=None, do_enter=True):
        result = StrategyResult()
        if do_enter:
            transport.enter()
        channels = list(channel_order) if channel_order else list(CHANNELS)
        bearings = {ch: [] for ch in channels}
        cleared: set = set()
        attempted: set = set()

        origin = self._current_position(transport)
        swept: List[Point] = []

        self._sweep(transport, origin, channels, bearings, cleared, result)
        swept.append(tuple(self._current_position(transport)))

        while len(cleared) < self.max_jammers:
            here = tuple(self._current_position(transport))

            nodes = []          # (point, channel or None)
            ch_of_point = {}

            # 覆盖点：批量贪心生成（当前 swept 所需的全部）
            for cover in self._generate_covers(swept):
                nodes.append(cover)

            # 源清除点
            for ch in channels:
                if ch in cleared or ch in attempted:
                    continue
                if len(bearings[ch]) < self.insert_min_observations:
                    continue
                est = self._source_target(bearings[ch], here)
                if est is not None:
                    k = (round(est[0], 1), round(est[1], 1))
                    if k not in ch_of_point:
                        nodes.append(est)
                        ch_of_point[k] = ch

            if not nodes:
                pending = [ch for ch in channels
                           if ch not in cleared and ch not in attempted
                           and bearings[ch]]
                if pending:
                    ch = min(pending, key=lambda c: distance(
                        here, self._resolve_anchor(bearings[c])))
                    if self._resolve_channel(transport, ch, bearings[ch], result):
                        cleared.add(ch)
                        swept.append(tuple(self._current_position(transport)))
                    continue
                break

            order = self._open_tour(here, nodes)
            point = order[0]
            k = (round(point[0], 1), round(point[1], 1))
            ch = ch_of_point.get(k)
            if ch is not None:
                attempted.add(ch)
                if self._fast_clear(transport, ch, point, bearings[ch], result):
                    cleared.add(ch)
                    here2 = tuple(self._current_position(transport))
                    if min((distance(here2, q) for q in swept),
                           default=float("inf")) >= self.inplace_sweep_min_gain:
                        self._discharge_sweep(transport, here2, channels,
                                              bearings, cleared, result)
                    swept.append(here2)
            else:
                self._sweep(transport, point, channels, bearings, cleared, result)
                swept.append(tuple(self._current_position(transport)))

        self._finalize_channel_clearing(transport, channels, bearings,
                                        cleared, result)
        transport.exit()
        result.cleared = len(cleared)
        result.total = self._evaluate_total(transport)
        result.virtual_time = self._virtual_time(transport)
        result.move_distance = self._move_distance(transport)
        return result

