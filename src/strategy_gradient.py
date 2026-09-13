"""面积梯度引导的"边走边扫边清"策略（对照策略，**无全覆盖保证**）。

设计来源：竞赛社区流传的一类"贪心/梯度"思路 —— 把作业域离散成网格，
用"每格被扫描到的次数"刻画覆盖热度，每一步只往"没扫过或扫得少"的地方走
（相当于对"未覆盖面积"做自然梯度上升），路线每轮都不同。

与主策略（结构化覆盖网 + 严格证书）的差别
------------------------------------------
* 主策略先算好**覆盖证书**（保证域内任意点都被至少一个网点看到），再按
  开放巡游走网点 —— **零漏测**，但必须付完整巡游。
* 本策略不预置布点，按"覆盖热度梯度"滚动作走；一旦"域内每个格子都被覆盖
  过"就收尾 —— 省掉结构化网的重复扫描，但格子级判据只有"格尺度"的
  保守性（见 ``_fully_covered``），故**存在小概率漏测**（社区版"扫不全"
  的来源即在此）。

本模块只作**对照实验**，正式测试仍走 ``StrategyP3`` / ``StrategyP4``；
导入本模块不会改变任何主策略行为。
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Set, Tuple

from .config import CHANNELS, R_MIN, REGION_RADIUS
from .geometry import distance
from .strategy_p3 import StrategyP3, StrategyResult, bearing_least_squares

Point = Tuple[float, float]
Cell = Tuple[int, int]

__all__ = ["CoverageGradientMixin", "StrategyP3Gradient"]


class CoverageGradientMixin:
    """覆盖热度网格 + 梯度选点（可混入 StrategyP3 / StrategyP4）。"""

    cell: float = 75.0          # 网格边长 (m)
    step: float = 250.0         # 每步候选前进距离 (m)
    n_dirs: int = 16            # 候选方向数
    max_steps: int = 400        # 主循环步数上限
    stall_limit: int = 25       # 连续多少步无新示向度就收尾
    clear_detour: float = 400.0  # 顺路清源的最大绕行距离 (m)

    def _reset_grid(self) -> None:
        cell = float(self.cell)
        self._count: Dict[Cell, int] = {}
        nc = int(math.ceil(REGION_RADIUS / cell)) + 1
        self._domain: Set[Cell] = set()
        for i in range(-nc, nc + 1):
            for j in range(-nc, nc + 1):
                cx, cy = (i + 0.5) * cell, (j + 0.5) * cell
                if math.hypot(cx, cy) <= REGION_RADIUS:
                    self._domain.add((i, j))
        # 判"已覆盖"时把检测半径收掉 0.8 个格对角线，抵消"用格心代表整格"的误差
        self._cover_radius = max(1.0, R_MIN - 0.8 * cell * math.sqrt(2.0))

    def _mark_covered(self, p: Point) -> None:
        """把"距 p ≤ 覆盖半径"的域内格子计数 +1（覆盖热度）。"""
        cell = float(self.cell)
        rr = self._cover_radius
        nc = int(rr / cell) + 1
        ci, cj = int(math.floor(p[0] / cell)), int(math.floor(p[1] / cell))
        cnt = self._count
        for di in range(-nc, nc + 1):
            for dj in range(-nc, nc + 1):
                k = (ci + di, cj + dj)
                if k not in self._domain:
                    continue
                cx, cy = (k[0] + 0.5) * cell, (k[1] + 0.5) * cell
                if math.hypot(cx - p[0], cy - p[1]) <= rr:
                    cnt[k] = cnt.get(k, 0) + 1

    def _gain(self, p: Point) -> float:
        """候选点 p 的覆盖增益 = Σ 1/(1+热度)（未扫过的格子权重大）。"""
        cell = float(self.cell)
        rr = self._cover_radius
        nc = int(rr / cell) + 1
        ci, cj = int(math.floor(p[0] / cell)), int(math.floor(p[1] / cell))
        cnt = self._count
        g = 0.0
        for di in range(-nc, nc + 1):
            for dj in range(-nc, nc + 1):
                k = (ci + di, cj + dj)
                if k not in self._domain:
                    continue
                cx, cy = (k[0] + 0.5) * cell, (k[1] + 0.5) * cell
                if math.hypot(cx - p[0], cy - p[1]) <= rr:
                    g += 1.0 / (1.0 + cnt.get(k, 0))
        return g

    def _fully_covered(self) -> bool:
        """域内所有格子都至少被覆盖过一次（面积级全覆盖判据）。"""
        return len(self._count) >= len(self._domain)

    def _coldest_point(self) -> Optional[Point]:
        """全局最冷格子中心（未覆盖优先，其次热度最低）。"""
        cell = float(self.cell)
        best_key = None
        best_p = None
        for k in self._domain:
            v = self._count.get(k, 0)
            key = (v, k)
            if best_key is None or key < best_key:
                best_key = key
                best_p = ((k[0] + 0.5) * cell, (k[1] + 0.5) * cell)
        return best_p

    def _gradient_target(self, cur: Point) -> Optional[Point]:
        """局部梯度上升：在 cur 周围取方向候选，取覆盖增益最大者。

        若"局部最优"仍不如"全局最冷格子"，则直接跳到最冷格子
        （对应优化里的"跳出局部最优"）。
        """
        best_g, best_p = -1.0, None
        n = max(4, int(self.n_dirs))
        for i in range(n):
            ang = 2.0 * math.pi * i / n
            p = (cur[0] + self.step * math.cos(ang),
                 cur[1] + self.step * math.sin(ang))
            if math.hypot(p[0], p[1]) > REGION_RADIUS + R_MIN:
                continue          # 域外无源，不必去
            g = self._gain(p)
            if g > best_g:
                best_g, best_p = g, p
        if best_p is None:
            return None
        cold = self._coldest_point()
        if cold is not None and self._gain(best_p) <= self._gain(cold) * 1.02:
            return cold
        return best_p

    def _next_target(self, cur: Point, channels, bearings, cleared: Set[int]):
        """选下一站：优先"顺路清源"（延迟捕获），否则走覆盖梯度。"""
        best = None
        for ch in channels:
            if ch in cleared:
                continue
            records = bearings[ch]
            if len(records) < getattr(self, "early_stop_bearings", 2):
                continue
            est = bearing_least_squares(records)
            if est is None:
                continue
            d = distance(cur, est)
            if d <= self.clear_detour and (best is None or d < best[0]):
                best = (d, ch, est)
        if best is not None:
            return ("clear", best[1], best[2])
        target = self._gradient_target(cur)
        if target is None:
            return None
        return ("sweep", -1, target)

    def run(self, transport, channel_order=None, do_enter=True):
        """面积梯度主循环：边走边扫边清，直到"清满"或"全域覆盖"。"""
        result = StrategyResult()
        if do_enter:
            transport.enter()
        channels = list(channel_order) if channel_order else list(CHANNELS)
        bearings = {ch: [] for ch in channels}
        cleared: Set[int] = set()
        state: Dict[str, object] = {}
        self.early_stop_bearings = getattr(self, "early_stop_bearings", 2)
        self._reset_grid()

        cur = tuple(self._current_position(transport))
        self._sweep(transport, cur, channels, bearings, cleared, result)
        self._mark_covered(tuple(self._current_position(transport)))

        stall = 0
        for _ in range(int(self.max_steps)):
            if len(cleared) >= self.max_total:
                break
            if self._fully_covered():
                break
            cur = tuple(self._current_position(transport))
            nxt = self._next_target(cur, channels, bearings, cleared)
            if nxt is None:
                break
            kind, channel, target = nxt
            before = sum(len(v) for v in bearings.values())
            if kind == "clear":
                if self._fast_clear(transport, channel, target,
                                    bearings[channel], result):
                    cleared.add(channel)
            else:
                self._sweep(transport, target, channels, bearings, cleared, result)
                self._mark_covered(tuple(self._current_position(transport)))
            after = sum(len(v) for v in bearings.values())
            stall = stall + 1 if after == before else 0
            if stall >= int(self.stall_limit):
                break

        self._finalize_channel_clearing(transport, channels, bearings, cleared,
                                        result)
        transport.exit()
        result.cleared = len(cleared)
        result.total = self._evaluate_total(transport)
        result.virtual_time = self._virtual_time(transport)
        result.move_distance = self._move_distance(transport)
        return result


class StrategyP3Gradient(CoverageGradientMixin, StrategyP3):
    """问题3 的面积梯度对照策略（**无全覆盖保证**）。"""

    def __init__(self, cell: float = 75.0, step: float = 250.0,
                 n_dirs: int = 16, max_steps: int = 400,
                 stall_limit: int = 25, clear_detour: float = 400.0,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.cell = float(cell)
        self.step = float(step)
        self.n_dirs = int(n_dirs)
        self.max_steps = int(max_steps)
        self.stall_limit = int(stall_limit)
        self.clear_detour = float(clear_detour)
        # 问题3/4 的干扰源总数上限为 16，清满即必定清空（与主策略同口径）
        self.max_total = float(self.max_jammers)


class StrategyP4Gradient(StrategyP3Gradient):
    """问题4 的面积梯度对照策略（与问题3 同代码，仅用于定向源场景）。

    **注意**：本类沿用"面积覆盖"判据，而定向源的可见与否还取决于朝向
    （半平面），因此它**不保证**"任意朝向的源至少被一个网点看到" ——
    这正是要量化的"牺牲覆盖率换速度"代价。
    """

    __slots__ = ()

