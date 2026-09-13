"""问题4 策略：全向 + 定向混合干扰源的自动定位与清除。

与问题3 的差异
--------------
1. **检测点网必须定向安全**：定向源只在"定向方向 ±90°"半平面内可被接收，
   同一个 G 若其朝向背离所有检测点，则在任何检测点都是 no_signal。
   因此要求：对域内任意 G 与任意朝向 u，存在 p 使 |p-G| <= R_min 且 (p-G)·u >= 0
   <=> G 严格位于近旁点凸包内部 <=> 近旁点相对 G 的最大角隙 < 180°。
   本策略采用 **22 点定向安全网**（中心 1 + 半径 1000 m 九点 + 半径 1875 m
   十二点，见 coverage.directional_survey_design），它带**严格判据**而不只是
   随机抽样通过：
     * 外环正十二边形内切半径 1811.11 m > 1800 m，故目标圆域整体落在
       内环/外环构成的三角剖分内部；
     * 采用**充要判据**：对域内任意 G，若"距 G 不超过 R_min = 1000 m 的网点"
       相对 G 的最大角隙 < 180°，则 G 严格位于这些点的凸包内部，于是任意朝向
       的定向源都至少被一个网点看到。该判据已用 3 级确定性密网复核
       （最高 200×2160 极坐标 + 21600 边界环 = 455,760 样本）：fail = 0，
       最大角隙 177.38°（裕量 2.62°）。
   相比 25 点双环网（内环 950×12 + 外环 1875×12 错相 15°），22 点网把内环
   半径顶到 R_min = 1000 m 并把内环压到 9 点：**覆盖证书强度不变**
   （角隙同为 177.3768°），但每轮扫描少扫 3 个网点 ⇒ 检测动作数低约 12%，
   实测 6 档平均 -14.3 s/源（N=10 档 -22~-39 s/源）。相比 31 点三角格点
   （开放巡游约 30.0 km），22 点网把巡游压到约 18.06 km。
2. **总数上限 16**：总数为 10~16，故清满 16 个即已清空，可提前结束；
   未再扫描的频道无需确认（省下大量检测与行程）。
3. **三条观测即停 + 顺路机会式清除**：同一频道积累三条示向度后用
   中心线最小二乘定位；若"绕到估计点再回到下一扫描点"的额外路程
   不超过 ``insert_threshold``（默认 400 m），就在扫描途中顺路清除，
   从而减少扫描结束后的返程；每频道至多插入一次。
4. **朝向推断**（论文用）：由"在点 p 有信号 / 无信号"的符号约束
   (p-Ĝ)·u >= 0 与 < 0，可交出一段可行的定向方向弧，用于辅助选点。

清除环节与问题3 相同：/clear 与朝向无关，距离 <= 20 m 必成功；
机会式/快速清除失败者由 ``StrategyP3._resolve_channel`` 的
"楔形交收缩 + 27 m 网格"保证性兜底接住，故不影响 100% 清除率。
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

from .config import CHANNELS, CLEAR_RADIUS, R_MIN
from .coverage import directional_survey_design, two_ring_design
from .geometry import angle_diff, distance, unit_vector
from .routing import tour_length as _tour_length, two_opt_tour
from .strategy_p3 import StrategyP3, StrategyResult, bearing_least_squares

Point = Tuple[float, float]

__all__ = ["StrategyP4", "two_opt_tour", "estimate_direction_arc"]


# --------------------------------------------------------------------------- #
# 定向朝向的可行弧估计（论文用；也可用于选点）
# --------------------------------------------------------------------------- #
def estimate_direction_arc(
    estimate: Point,
    detections: Sequence[Point],
    misses: Sequence[Point],
    r_min: float = R_MIN,
    step_deg: float = 1.0,
) -> List[float]:
    """由符号约束求定向方向 u 的可行弧（返回可行角度列表，空列表=无解）。

    有信号: (p - G)·u >= 0；无信号（且 |p-G| <= r_min）: (p - G)·u < 0。
    仅用 |p-G| <= r_min 的无信号点作为有效约束（更远的无信号可能因超距）。
    """
    cons_det = [d for d in detections]
    cons_miss = [m for m in misses if distance(m, estimate) <= r_min]
    feasible: List[float] = []
    k = int(round(360.0 / step_deg))
    for i in range(k):
        ang = i * step_deg
        ux, uy = unit_vector(ang)
        ok = True
        for p in cons_det:
            if (p[0] - estimate[0]) * ux + (p[1] - estimate[1]) * uy < -1e-9:
                ok = False
                break
        if ok:
            for p in cons_miss:
                if (p[0] - estimate[0]) * ux + (p[1] - estimate[1]) * uy > 1e-9:
                    ok = False
                    break
        if ok:
            feasible.append(ang)
    return feasible


class StrategyP4(StrategyP3):
    """问题4 策略：定向安全双环网 + 顺路机会式清除 + 严格楔形兜底。"""

    def __init__(self, survey_points: Optional[Sequence[Point]] = None,
                 insert_threshold: float = 400.0, **kwargs) -> None:
        # 默认用 22 点定向安全网（directional_survey_design：内环 1000×9 +
        # 外环 1875×12）。注意：曾用 25 点双环网（950×12+1875×12 错相 15°），
        # 其"三角剖分全部边 < R_min"的充分条件更漂亮，但 22 点网的角隙与之
        # 完全相同（177.3768°）而检测动作少约 12%，故改用 22 点。
        # 另注意：24 点网（950×8+1850×16）在"正上方边界源 + 朝向水平"角隙达
        # 186.6° > 180° 会漏测，坚持用本网。
        default = survey_points or directional_survey_design()
        super().__init__(survey_points=default, **kwargs)
        self.max_total: float = 16.0        # 总数上限：清满 16 个必然是全部
        # 顺路插入阈值：直接把源插进"剩余路线最省的那一段"（与问题3 同款机制），
        # 只有额外路程不超过该值才插入，其余留给收尾 TSP。
        # 400 m 为 Mock 消融实验的最优工作点。
        self.insert_limit = float(insert_threshold)

    def _tour_order(self) -> List[Point]:
        """扫描顺序：**从内环起**的最近邻 + 2-opt + Or-opt（开放巡游）。

        为什么从内环起：内环点（半径 950，12 点）负责"内部源"，外环点（半径 1875，
        12 点）负责"边界源"；先扫内环能让内部源更早被发现与定位，从而更早触发
        "16 源都 ≥2 条示向度即提前结束扫描"。实测（15 例/档）：
          全向 N=16 259.2→252.1、半定向 400.6→379.3、全定向 477.3→458.9 s/源；
          N=10 各档亦降 10~18 s/源。
        """
        pts = list(self.survey_points)
        # 中心（原点）本身就是起点，不放进巡游列表；内环按角排序后先走
        inner = [p for p in pts if 1.0 <= math.hypot(p[0], p[1]) < 1400.0]
        outer = [p for p in pts if math.hypot(p[0], p[1]) >= 1400.0]
        inner.sort(key=lambda q: math.atan2(q[1], q[0]))
        outer.sort(key=lambda q: math.atan2(q[1], q[0]))
        return self._strong_tour((0.0, 0.0), inner + outer)

    @staticmethod
    def _strong_tour(start: Point, points: Sequence[Point],
                     rounds: int = 10, warm: int = 4) -> List[Point]:
        """最近邻 + 2-opt + Or-opt 的开放巡游（多起点取最短）。"""
        pts = list(points)
        if not pts:
            return [start]
        best: Optional[List[Point]] = None
        best_len = float("inf")
        for k in range(min(warm, len(pts))):
            tour = [start, pts[k]]
            pool = pts[:k] + pts[k + 1:]
            while pool:
                nxt = min(pool, key=lambda q: distance(tour[-1], q))
                pool.remove(nxt)
                tour.append(nxt)
            n = len(tour)
            for _ in range(rounds):
                improved = False
                for i in range(n - 2):
                    for j in range(i + 2, n - 1):
                        a, b, c, d = tour[i], tour[i + 1], tour[j], tour[j + 1]
                        if (distance(a, b) + distance(c, d)
                                > distance(a, c) + distance(b, d) + 1e-9):
                            tour[i + 1:j + 1] = tour[i + 1:j + 1][::-1]
                            improved = True
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
                            improved = True
                        else:
                            s += 1
                if not improved:
                    break
            L = sum(distance(tour[i], tour[i + 1]) for i in range(len(tour) - 1))
            if L < best_len:
                best, best_len = tour, L
        return best if best is not None else [start]

    def _observations_complete(self, records):
        """三条观测即可定位。

        楔形交在三条示向线下可消除两线近共线造成的公里级长尾误差；
        Mock 消融显示三条即停可以把检测动作数压到约 295 次，同时保持
        100% 清除率（定位失败的频道由 ``_resolve_channel`` 严格兜底）。
        """
        return len(records) >= self.observations_target

    # ------------------------------------------------------------------ #
    # 机会式清除：扫描途中顺路清除，减少扫描后的返程
    # ------------------------------------------------------------------ #
    def _opportunistic_clear(self, transport, channels, bearings, cleared,
                             result: StrategyResult, current: Point,
                             next_point: Point, state: Dict[str, object]) -> None:
        """在离开 ``current`` 之前，顺路清除已经定位得足够准的频道。

        判定：候选频道的额外路程
            delta = |cur-ĝ| + |ĝ-next| - |cur-next|
        不超过 ``insert_limit``（问题4 默认由 ``insert_threshold`` 传入，400 m）时才插入清除动作，这里 ``cur`` 取机器人
        实际当前位置（清除动作可能已经把我们带离扫描点）。

        每个频道在一次运行中至多触发一次机会式插入（``state`` 中记录），
        避免同一"定位失败"的频道在后续每条扫描边上重复绕行；插入失败者
        仍会在扫描结束后的统一快速清除与严格兜底中被处理，因此不影响
        保证性。
        """
        attempted = state.setdefault("insert_attempted", set())
        assert isinstance(attempted, set)

        while True:
            cur = self._current_position(transport)
            best: Optional[Tuple[float, int, Point]] = None
            for ch in channels:
                if ch in cleared or ch in attempted:
                    continue
                records = bearings[ch]
                if len(records) < self.observations_target:
                    continue
                estimate = bearing_least_squares(records)
                if estimate is None:
                    continue
                detour = (distance(cur, estimate) + distance(estimate, next_point)
                          - distance(cur, next_point))
                if detour > self.insert_limit:
                    continue
                if best is None or detour < best[0]:
                    best = (detour, ch, estimate)
            if best is None:
                return
            _, channel, estimate = best
            attempted.add(channel)
            if self._fast_clear(transport, channel, estimate,
                                bearings[channel], result):
                cleared.add(channel)
                continue                # 继续检查这条边上是否还有其他顺路频道
            return                      # 该边不再插入；失败频道留给扫描后的统一清除

    # ------------------------------------------------------------------ #
    def direction_arc(self, estimate: Point, detections: Sequence[Point],
                      misses: Sequence[Point]) -> List[float]:
        return estimate_direction_arc(estimate, detections, misses)