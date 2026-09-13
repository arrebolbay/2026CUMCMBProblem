# -*- coding: utf-8 -*-
"""小算子 A/B：把"只有 1 条示向度"的频道纳入**顺路插入**（射线—路径几何）。

背景（原策略行为，不改动）
--------------------------
``src/strategy_p3.py`` 的 ``_schedule_sources`` 只把"已有 >= insert_min_observations
(=2) 条示向度"的源插进剩余路线；**只有 1 条示向度**的频道一律留到收尾阶段，
由 ``probe_steps_m = (450,225,110,55,25)`` 沿射线阶梯追击（射线最长 1500 m）
或 ``_resolve_channel`` 兜底，这段"1-bearing 尾程"是高成本来源。

本文件用**子类**只覆盖调度环节，测试两种插入口径：
  * V1（射线—路径最近点）：若剩余路线某处离该射线足够近（插入绕行 <= ray_insert_limit），
    就把"射线上离该处最近的点"作为顺路访问点，就地测第 2 条示向度。
  * V2（视差保护版）：同上，但要求访问点与射线的**横向偏移 h ∈ [h_min, h_max]**，
    保证第 2 条示向度与第 1 条有足够视差（h 很小 ⇒ 两线近平行 ⇒ 交会退化）。
命中后就地 ``_fast_clear``（先测后清）；失败时该频道已升为 2 条示向度，
收尾成本远低于原来的 1 条示向度兜底。

设计约束
--------
* **不修改任何原文件**：只继承 ``StrategyP3`` 并覆盖 ``_schedule_sources``；
* 覆盖保证不受影响：插入点只**增加**访问点，骨干点一个都不会被删除；
* 每次插入都受 ``ray_insert_limit``（小阈值）与 ``max_ray_insertions``（次数上限）约束。
"""

from __future__ import annotations

import os
import sys
from typing import List, Optional, Sequence, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.config import R_MAX                       # noqa: E402
from src.geometry import distance, unit_vector     # noqa: E402
from src.strategy_p3 import StrategyP3             # noqa: E402

Point = Tuple[float, float]

__all__ = ["StrategyP3RayInsert"]


class StrategyP3RayInsert(StrategyP3):
    """允许"恰好 1 条示向度"的频道参与顺路插入（射线—路径几何）。"""

    def __init__(
        self,
        ray_variant: int = 2,
        ray_insert_limit: float = 200.0,
        ray_min_lateral: float = 150.0,
        ray_max_lateral: float = 400.0,
        max_ray_insertions: int = 6,
        ray_step_h: float = 50.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.ray_variant = int(ray_variant)          # 0=关闭, 1=最近点, 2=视差保护
        self.ray_insert_limit = float(ray_insert_limit)
        self.ray_min_lateral = float(ray_min_lateral)
        self.ray_max_lateral = float(ray_max_lateral)
        self.max_ray_insertions = int(max_ray_insertions)
        self.ray_step_h = float(ray_step_h)
        self.ray_insert_fired = 0                    # 本次运行实际插入次数

    # ------------------------------------------------------------------ #
    # 几何：求该射线对剩余路线的最佳顺路插入点
    # ------------------------------------------------------------------ #
    def _best_insertion(self, seq: Sequence[Point], base: Point, bearing: float):
        """返回 (cost, segment_index, X)：把 X 插在 seq[k] 与 seq[k+1] 之间的最小额外路程。

        X = base + t·u + s·h·n（u 为射线方向、n 为其法向），t ∈ [0, R_MAX]，
        h 取 0（V1）或 [h_min, h_max]（V2，双侧都试）。
        """
        ux, uy = unit_vector(bearing)
        nx, ny = -uy, ux
        if self.ray_variant == 1:
            hs = [0.0]
        else:
            span = max(0.0, self.ray_max_lateral - self.ray_min_lateral)
            n_h = int(span / max(self.ray_step_h, 1e-9)) + 1
            hs = [self.ray_min_lateral + k * self.ray_step_h for k in range(n_h)]
        best = None
        for k in range(len(seq) - 1):
            a, b = seq[k], seq[k + 1]
            ab = distance(a, b)
            for h in hs:
                sides = (1.0,) if h <= 0.0 else (1.0, -1.0)
                for s in sides:
                    def cost_at(t: float) -> float:
                        x = base[0] + t * ux + s * h * nx
                        y = base[1] + t * uy + s * h * ny
                        return distance(a, (x, y)) + distance(b, (x, y)) - ab
                    lo, hi = 0.0, float(R_MAX)
                    for _ in range(45):                       # 三分搜索（cost 对 t 凸）
                        m1 = lo + (hi - lo) / 3.0
                        m2 = hi - (hi - lo) / 3.0
                        if cost_at(m1) < cost_at(m2):
                            hi = m2
                        else:
                            lo = m1
                    t = 0.5 * (lo + hi)
                    cost = cost_at(t)
                    if cost <= self.ray_insert_limit and (best is None or cost < best[0]):
                        X = (base[0] + t * ux + s * h * nx, base[1] + t * uy + s * h * ny)
                        best = (cost, k, X)
        return best

    # ------------------------------------------------------------------ #
    # 调度：先按原策略插入已定位源，再尝试 1-bearing 频道的射线顺路插入
    # ------------------------------------------------------------------ #
    def _schedule_sources(self, route, index, channels, bearings, cleared,
                          attempted, scheduled) -> None:
        super()._schedule_sources(route, index, channels, bearings, cleared,
                                  attempted, scheduled)
        if self.ray_variant == 0 or self.ray_insert_limit <= 0:
            return
        while self.ray_insert_fired < self.max_ray_insertions:
            seq = [route[index][0]] + [p for p, _, _ in route[index + 1:]]
            if len(seq) < 2:
                return
            best = None
            for ch in channels:
                if ch in cleared or ch in attempted or ch in scheduled:
                    continue
                records = bearings[ch]
                if len(records) != 1:                 # 只处理"恰好 1 条示向度"
                    continue
                cand = self._best_insertion(seq, records[0][0], records[0][1])
                if cand is None:
                    continue
                cost, k, X = cand
                if best is None or cost < best[0]:
                    best = (cost, k, ch, X)
            if best is None:
                return
            _, k, channel, X = best
            route.insert(index + 1 + k, (X, channel, False))
            scheduled.add(channel)
            self.ray_insert_fired += 1
            self._improve_remaining(route, index)

    # ------------------------------------------------------------------ #
    def run(self, transport, channel_order=None, do_enter=True):
        self.ray_insert_fired = 0
        return super().run(transport, channel_order, do_enter)
