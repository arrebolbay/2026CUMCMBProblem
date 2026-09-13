"""访问优先的极简对照策略：把"覆盖巡游"与"追源行程"融合成一条路线。

设计动机（时间分解见 ``scripts/diag_time_split.py``）
----------------------------------------------------
主策略的时间有 **76~78% 花在移动**上，而这笔移动由**两段几乎不重叠**的行程组成：

  * 保证性覆盖骨干巡游 —— 问题三 ``中心+14 点环`` 开放巡游 **6.79 km = 1357 s**；
  * 逐个访问已定位干扰源的 TSP —— N=16 的理论下界 **≈ 9.09 km = 1818 s**。

两者相加 ≈ 15.9 km，与实测 14.84 km 吻合。这就是"保证性"的代价：必须先走完
覆盖集才算"探测完备"，而"访问源"（清除必须走到源 20 m 内）又要另走一趟。

本策略把两段**融合**成一条路线：候选目标 = {已定位源} ∪ {最近的未探索格}，
每一步取"等效距离"最小者前行（``等效距离 = 实际距离 × access_weight``），
于是"追源"的行程同时承担"探索"使命，行程目标 ≈ max(骨干, 访问) ≈ 9~10 km
——正是 160 s/源 所需的量级：

    9.5 km / 5 m·s⁻¹ = 1900 s；+ 测量 ~650 s + 清除 ~90 s ≈ 2640 s ⇒ 165 s/源

⚠ **代价**：不再存在"域内任意位置都被某个扫描点看到"的证书 —— 探索只按
   450 m 粗网格推进，且判"已探索"用的是"格心距扫描点 ≤ R_MIN"这种**格尺度**
   保守判据，因此**存在漏测**。本模块**只作对照实验**，正式测试仍走
   ``StrategyP3`` / ``StrategyP4``；导入本模块不改变任何主策略行为。
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Set, Tuple

from .config import CHANNELS, REGION_RADIUS, R_MIN
from .geometry import distance
from .strategy_p3 import (StrategyP3, StrategyResult,
                          bearing_least_squares)

Point = Tuple[float, float]

__all__ = ["FastAccessMixin", "StrategyP3Fast", "StrategyP4Fast"]


class FastAccessMixin:
    """"访问优先 + 按需探索"的融合路线（可混入 StrategyP3 / StrategyP4）。"""

    sweep_cell: float = 450.0     # 探索网格边长 (m)
    access_weight: float = 0.6    # 已定位源的等效距离权重（<1 表示优先追源）
    fast_max_steps: int = 300     # 主循环步数上限
    fast_stall_limit: int = 40    # 连续多少步无新示向度就收尾
    fast_max_total: float = 16.0  # 清满即停（与主策略同口径）
    ring_width: float = 600.0     # 探索推进的"环"宽度 (m)：由内向外，避免横跳
    fast_resolve: bool = True     # 收尾是否走严格兜底（False ⇒ 放弃难源，更快）
    explore_budget: int = 10_000  # 最多做多少次"探索性扫描"（越小越快、越可能漏）
    explore_first: bool = False   # True ⇒ 先把预算内的探索走完（建骨架）再追源

    # ---------------------------------------------------------------- 探索网格
    def _reset_swept_grid(self) -> None:
        """域内粗网格（格心代表整格），并用 R_MIN 作为"一点可探多大范围"。"""
        cell = float(self.sweep_cell)
        self._swept: Set[Tuple[int, int]] = set()
        cells: Dict[Tuple[int, int], Point] = {}
        nc = int(math.ceil(REGION_RADIUS / cell)) + 1
        for i in range(-nc, nc + 1):
            for j in range(-nc, nc + 1):
                cx, cy = (i + 0.5) * cell, (j + 0.5) * cell
                if math.hypot(cx, cy) <= REGION_RADIUS:
                    cells[(i, j)] = (cx, cy)
        self._cells = cells
        self._cover_radius = max(1.0, float(R_MIN))

    def _mark_swept(self, p: Point) -> None:
        """把"距 p ≤ R_MIN"的域内格标记为已探索。"""
        cell = float(self.sweep_cell)
        rr = self._cover_radius
        nc = int(rr / cell) + 1
        ci, cj = int(math.floor(p[0] / cell)), int(math.floor(p[1] / cell))
        cells, swept = self._cells, self._swept
        for di in range(-nc, nc + 1):
            for dj in range(-nc, nc + 1):
                k = (ci + di, cj + dj)
                pt = cells.get(k)
                if pt is None or k in swept:
                    continue
                if math.hypot(pt[0] - p[0], pt[1] - p[1]) <= rr:
                    swept.add(k)

    def _nearest_unswept(self, cur: Point) -> Optional[Point]:
        """由内向外推进：先取"环号最小"的未探索格，环内再取最近的。

        若直接取"全局最近的未探索格"，路线会在内外圈之间来回横跳
        （实测行程膨胀到 36 km）。改为按到原点的距离分环、每 ``ring_width``
        一圈，**取环号最小者**，推进方向便稳定"由内到外"，等价于螺旋爬升。
        """
        best: Optional[Point] = None
        best_key: Optional[Tuple[int, float]] = None
        swept = self._swept
        w = float(self.ring_width)
        for k, pt in self._cells.items():
            if k in swept:
                continue
            ring = int(math.hypot(pt[0], pt[1]) / w)
            d = distance(cur, pt)
            key = (ring, d)
            if best_key is None or key < best_key:
                best_key, best = key, pt
        return best

    # ---------------------------------------------------------------- 目标选择
    def _route_target(self, cur: Point, channels, bearings, cleared, attempted=None):
        """返回 (kind, channel, point)：已定位源与最近未探索格中"等效距离"最小者。

        ``attempted`` 记录"已经追过一次但没清掉"的频道：不再重复追击
        （否则会在同一目标上来回跑，实测行程膨胀到 36 km），改由收尾统一处理。
        """
        best_score = float("inf")
        best: Optional[Tuple[str, int, Point]] = None
        # 骨架优先：预算内的探索没走完之前不追源（保证源的定位由多点远距离交会
        # 给出，避免"边探边追"时定位过差导致近场收口极贵）
        if self.explore_first and self._explored < int(self.explore_budget):
            pt = self._nearest_unswept(cur)
            return None if pt is None else ("sweep", -1, pt)
        skip = attempted or ()
        for ch in channels:
            if ch in cleared or ch in skip or not bearings[ch]:
                continue
            est = self._source_target(bearings[ch], cur)
            if est is None:
                continue
            score = distance(cur, est) * float(self.access_weight)
            if score < best_score:
                best_score, best = score, ("clear", ch, est)
        pt = None
        if self._explored < int(self.explore_budget):
            pt = self._nearest_unswept(cur)
        if pt is not None:
            score = distance(cur, pt)
            if score < best_score:
                best = ("sweep", -1, pt)
        return best

    # ---------------------------------------------------------------- 主循环
    def _run_fast(self, transport, channel_order=None, do_enter=True):
        """走得最省的"边走边扫边清"：清满即停，未决频道由严格兜底收尾。"""
        result = StrategyResult()
        if do_enter:
            transport.enter()
        channels = list(channel_order) if channel_order else list(CHANNELS)
        bearings = {ch: [] for ch in channels}
        cleared: Set[int] = set()
        self.early_stop_bearings = getattr(self, "early_stop_bearings", 2)
        self._reset_swept_grid()
        max_total = float(getattr(self, "max_jammers", 16))

        # 起点（原点）先做一次全频道探测：成本 120 s，但立刻锁定中心附近的源
        self._sweep(transport, tuple(self._current_position(transport)),
                    channels, bearings, cleared, result)
        self._mark_swept(tuple(self._current_position(transport)))

        stall = 0
        attempted: Set[int] = set()
        self._explored = 0
        for _ in range(int(self.fast_max_steps)):
            if len(cleared) >= max_total:
                break
            cur = tuple(self._current_position(transport))
            nxt = self._route_target(cur, channels, bearings, cleared, attempted)
            if nxt is None:
                break
            kind, channel, target = nxt
            before = sum(len(v) for v in bearings.values())
            if kind == "clear":
                if self._fast_clear(transport, channel, target,
                                    bearings[channel], result):
                    cleared.add(channel)
                else:
                    attempted.add(channel)
            else:
                self._sweep(transport, target, channels, bearings, cleared, result)
                self._explored += 1
            self._mark_swept(tuple(self._current_position(transport)))
            after = sum(len(v) for v in bearings.values())
            stall = stall + 1 if after == before else 0
            if stall >= int(self.fast_stall_limit):
                break

        if self.fast_resolve:
            self._finalize_channel_clearing(transport, channels, bearings,
                                            cleared, result)
        else:
            self._fast_finalize(transport, channels, bearings, cleared, result)
        transport.exit()
        result.cleared = len(cleared)
        result.total = self._evaluate_total(transport)
        result.virtual_time = self._virtual_time(transport)
        result.move_distance = self._move_distance(transport)
        return result

    def _fast_finalize(self, transport, channels, bearings, cleared, result):
        """收尾但**不兜底**：只对"已有 ≥2 条示向度"的频道做一次快速清除。

        ``_resolve_channel`` 的扇形探测 + 27 m 网格是"100% 清除保证"的最后
        一道闸门，但它对"从未被发现"的源毫无帮助（连一条示向度都没有），
        却对"发现了但定位差"的源消耗大量行程。对照组把它关掉，用更短的行程
        换更低的清除率 —— 这正是要量化的取舍。
        """
        for ch in channels:
            if ch in cleared or len(bearings[ch]) < 2:
                continue
            center = bearing_least_squares(bearings[ch])
            if center is None:
                center = self._source_target(
                    bearings[ch], self._current_position(transport))
            if center is None:
                continue
            if self._fast_clear(transport, ch, center, bearings[ch], result):
                cleared.add(ch)

    def run(self, transport, channel_order=None, do_enter=True):
        return self._run_fast(transport, channel_order, do_enter)


class StrategyP3Fast(FastAccessMixin, StrategyP3):
    """问题3 的快速对照策略（**无覆盖率证书，可能漏测**）。"""

    def __init__(self, sweep_cell: float = 450.0, access_weight: float = 0.6,
                 ring_width: float = 600.0, fast_resolve: bool = True,
                 explore_budget: int = 10_000, explore_first: bool = False,
                 fast_max_steps: int = 300, fast_stall_limit: int = 40,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.sweep_cell = float(sweep_cell)
        self.access_weight = float(access_weight)
        self.ring_width = float(ring_width)
        self.fast_resolve = bool(fast_resolve)
        self.explore_budget = int(explore_budget)
        self.explore_first = bool(explore_first)
        self.fast_max_steps = int(fast_max_steps)
        self.fast_stall_limit = int(fast_stall_limit)


class StrategyP4Fast(StrategyP3Fast):
    """问题4 的快速对照策略（沿用"面积探索"判据 ⇒ 对定向源漏测更明显）。"""

    __slots__ = ()

