#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ============================================================================
# 【学习注释】问题四：含定向源的方向完备搜索与清除（q4.py，约 2980 行）
# ----------------------------------------------------------------------------
# 与问题三的根本差别：定向源只在"定向方向 ±90°"的半平面内辐射，因此
# "某频道在所有检测点都无信号"**不再能推出**"该频道为空" —— 完全可能所有
# 检测点都落在它的辐射半平面之外！于是完备性要求从"覆盖位置"加强为
# "覆盖位置 × 方向"的乘积空间。
#
# 【核心等价命题】
#   一组检测点能对位于 G 的**任意朝向**定向源保证可见
#     ⇔  G 落在"这些点中距 G 不超过 R_min 的那些点"的**凸包内部**
#     ⇔  这些近旁点相对 G 的**最大方位角隙 < 180°**。
# 由此构造 中心 1 + 990 m 内环 7 + 外接正十四边形外环 14 = **22 点方向完备骨架**，
# 并用 2 m 网格**离线证书**验证（距离余量 1.586 m、凸包余量 7.187 m），恢复严格判空。
# 他们还证明了：**只用目标区域内部的点不可能方向完备，外环是必需的**。
#
# 【值得学的工程细节】
#   · 定向源失联时用"前向对 + 侧向对"四点恢复扇（recovery_candidates）；
#   · 后验收缩不到 20 m 时，用步长 27.72 m 的**确定性网格**覆盖清除（保证命中）；
#   · **SAFE 层**：只有当"替换点仍严格保证清除、且两动作总移动严格更短"时才挪动
#     清除点，并强制锚定原调度的下一动作 —— 保证"同一前状态下决策不变"。
#
# 主要函数：validate_direction_complete_backbone（22 点证书）/
#          run_geometry_self_test（2 m 网格离线证书）/ is_strict_guaranteed_clear_action /
#          best_safe_clear_point_between / safe_transform_selected_action（SAFE 层）
# ============================================================================

"""B 题问题 4 程序。"""
from __future__ import annotations
import copy
import itertools
import json
import math
import random
import socket
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
# 接口配置
BASE_URL = "http://127.0.0.1:2026"
ROBOT_ID = "************" # 提交版隐藏队号
ARENA_ID = "default"
HTTP_TIMEOUT_S = 10
HTTP_RETRIES = 3
# 每轮单独保存日志。
RUN_STAMP = (
    time.strftime("%Y%m%d_%H%M%S")
    + f"_ {time.time_ns() % 1_000_000_000 :09d}"
)
OUTPUT_DIR = Path(__file__).resolve().parent / " 第四问输出" / RUN_STAMP
TRACE_PATH = OUTPUT_DIR / "q4_trace.jsonl"
SUMMARY_PATH = OUTPUT_DIR / "q4_summary.json"
# 题目常量
TARGET_RADIUS = 1800.0
MIN_RECEIVE_RADIUS = 1000.0
MAX_RECEIVE_RADIUS = 1500.0
BEARING_ERROR_DEG = 1.0
NEAR_RADIUS = 5.0
CLEAR_RADIUS = 20.0
SPEED = 5.0
MEASURE_SECONDS = 5.0
SWITCH_SECONDS = 1.0
CLEAR_SUCCESS_SECONDS = 5.0
CLEAR_FAILURE_SECONDS = 3.0
CHANNEL_COUNT = 20
SOURCE_MAX = 16
# 策略参数
# 搜索点：中心 1 点、内环 7 点、外环 14 点。
# 外环给目标圆留 10m 余量。
INNER_SEARCH_COUNT = 7
INNER_SEARCH_RADIUS_M = 990.0
OUTER_SEARCH_COUNT = 14
OUTER_HULL_MARGIN_M = 10.0
OUTER_PHASE_DEG = 0.7
OUTER_SEARCH_RADIUS_M = (TARGET_RADIUS + OUTER_HULL_MARGIN_M) / math.cos(
    math.pi / OUTER_SEARCH_COUNT
)
SEARCH_POINT_COUNT = 1 + INNER_SEARCH_COUNT + OUTER_SEARCH_COUNT
# 离线搜索覆盖证书。
CERTIFICATE_GRID_STEP_M = 2.0
CERTIFICATE_NEIGHBOR_RADIUS_M = 997.0
CERTIFICATE_MIN_HULL_MARGIN_M = 8.601631088485874
CERTIFICATE_VERSION = "q4-22-local-hull-2m-v1"
SCHEDULER_VERSION = "q4-v5-safe-final-local-dominance-v1"
# 第一次 direction 后的侧移量。
Q2_FAST_SIDE_M = 80.0
# 定向源失联后的四点恢复。
RECOVERY_FORWARD_M = 200.0
RECOVERY_SIDE_M = 350.0
# Probe clear 使用工作半径。
STRONG_PROBE_RADIUS_M = 80.0
WEAK_PROBE_RADIUS_M = 180.0
# 后验较小时切换到网格清除。
# 网格最远覆盖约 19.6m。
SWEEP_TRIGGER_DIRECTION_COUNT = 6
SWEEP_TRIGGER_WORK_RADIUS_M = 120.0
SWEEP_GRID_STEP_M = CLEAR_RADIUS * math.sqrt(2.0) * 0.98
# 调度奖励（等价米）
GUARANTEED_CLEAR_BONUS_M = 320.0
STRONG_PROBE_BONUS_M = 150.0
WEAK_PROBE_BONUS_M = 60.0
SWEEP_CLEAR_BONUS_M = 210.0
EW_PROGRESS_BONUS_M = 300.0
INNER_EW_PROGRESS_BONUS_M = 500.0
Q2_FAST_INFO_BONUS_M = 45.0
# 2-opt 首步允许的最大额外代价。
POST_ROUTE_FIRST_STEP_REGRET_M = 100.0
# EW 顺路复测上限。
MAX_OPPORTUNISTIC_PER_EW = SOURCE_MAX
MIN_OPPORTUNISTIC_BASELINE_M = 220.0
MIN_OPPORTUNISTIC_SIN = 0.15
MAX_OPPORTUNISTIC_CENTER_DISTANCE_M = 1200.0
# 两条示向线的几何 probe。
# 失败后原地补测。
GEOM_PROBE_MIN_SIN = 0.45
GEOM_PROBE_MAX_ERROR_M = 50.0
GEOM_PROBE_BONUS_M = 260.0
# 演练数据得到的调度可靠度。
# 只参与调度评分。
GEOM_PROBE_RELIABILITY = 0.88
STRONG_PROBE_RELIABILITY = 0.65
WEAK_PROBE_RELIABILITY = 0.50
Q2_MEASURE_RELIABILITY = 0.45
RECOVERY_RELIABILITY = 0.70
# 避免重复 clear 同一点。
FAILED_CLEAR_AVOID_RADIUS_M = 8.0
# SAFE 层参数。
# 清除半径留 5cm 余量。
SAFE_CLEAR_MARGIN_M = 0.05
# 小于 1cm 的收益忽略。
SAFE_MIN_ROUTE_GAIN_M = 0.01
MAX_LOGICAL_ACTIONS = 20000
Point = tuple[float, float]
# 几何工具
def distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])
def open_route_length(
    start: Point,
    points: list[Point],
) -> float:
    """ 从 start 出发且不返回起点的开放路线长度。"""
    total = 0.0
    current = start
    for point in points:
        total += distance(current, point)
        current = point
    return total
# 【学习注释】开放（不回到起点）巡游的排序优化：
# 最近邻构造 + 局部改进。注意与 q3 的分工 —— q3 对小规模强制任务集用**精确 DP**，
# 这里对搜索环这种较长序列用启发式，是"精度/规模"的权衡。

def optimized_open_order(
    start: Point,
    points: list[Point],
) -> list[int]:
    """ 最近邻生成初解，再用开放路线 2-opt 消除明显折返。"""
    if not points:
        return []
    pending = list(range(len(points)))
    order: list[int] = []
    current = start
    while pending:
        chosen = min(
            pending,
            key=lambda index: (
                distance(current, points[index]),
                index,
            ),
        )
        pending.remove(chosen)
        order.append(chosen)
        current = points[chosen]
    # 2-opt 只比较反转段两端。
    improved = True
    while improved:
        improved = False
        for i in range(len(order) - 1):
            previous = start if i == 0 else points[order[i - 1]]
            for j in range(i + 1, len(order)):
                first = points[order[i]]
                last = points[order[j]]
                following = (
                    points[order[j + 1]]
                    if j + 1 < len(order)
                    else None
                )
                old_length = distance(previous, first)
                new_length = distance(previous, last)
                if following is not None :
                    old_length += distance(last, following)
                    new_length += distance(first, following)
                if new_length + 1e-9 < old_length:
                    order[i:j + 1] = reversed(order[i:j + 1])
                    improved = True
                    break
            if improved:
                break
    return order
def cross(a: Point, b: Point) -> float:
    return a[0] * b[1] - a[1] * b[0]
def sub(a: Point, b: Point) -> Point:
    return a[0] - b[0], a[1] - b[1]
def clip_left(
    polygon: list[Point],
    line_point: Point,
    line_direction: Point,
    eps: float = 1e-9,
) -> list[Point]:
    if not polygon:
        return []
    def value(p: Point) -> float:
        return cross(line_direction, sub(p, line_point))
    result: list[Point] = []
    prev = polygon[-1]
    prev_value = value(prev)
    prev_in = prev_value >= -eps
    for cur in polygon:
        cur_value = value(cur)
        cur_in = cur_value >= -eps
        if prev_in != cur_in:
            denom = prev_value - cur_value
            if abs(denom) > 1e-15:
                t = prev_value / denom
                t = min(1.0, max(0.0, t))
                result.append(
                    (
                        prev[0] + t * (cur[0] - prev[0]),
                        prev[1] + t * (cur[1] - prev[1]),
                    )
                )
        if cur_in:
            result.append(cur)
        prev = cur
        prev_value = cur_value
        prev_in = cur_in
    clean: list[Point] = []
    for p in result:
        if not clean or distance(clean[-1], p) > 1e-8:
            clean.append(p)
    if len(clean) > 1 and distance(clean[0], clean[-1]) <= 1e-8:
        clean.pop()
    return clean
def circumscribed_disk(
    center: Point,
    radius: float,
    sides: int,
) -> list[Point]:
    """ 圆盘的外接正多边形；只会保守放大。"""
    polygon_radius = radius / math.cos(math.pi / sides)
    phase = math.pi / sides
    return [
        (
            center[0]
            + polygon_radius * math.cos(phase + 2.0 * math.pi * i / sides),
            center[1]
            + polygon_radius * math.sin(phase + 2.0 * math.pi * i / sides),
        )
        for i in range(sides)
    ]
def clip_by_convex_polygon(
    subject: list[Point],
    clipper: list[Point],
) -> list[Point]:
    poly = subject
    for i in range(len(clipper)):
        a = clipper[i]
        b = clipper[(i + 1) % len(clipper)]
        poly = clip_left(
            poly,
            a,
            (b[0] - a[0], b[1] - a[1]),
        )
        if not poly:
            break
    return poly
TARGET_POLY = circumscribed_disk(
    (0.0, 0.0),
    TARGET_RADIUS,
    72,
)
def update_polygon_with_direction(
    polygon: Optional[list[Point]],
    station: Point,
    bearing_deg: float,
) -> list[Point]:
    """ 根据目标圆、测向角域和 1500m 接收范围更新保守后验。"""
    poly = TARGET_POLY.copy() if polygon is None else list(polygon)
    lower = math.radians(bearing_deg - BEARING_ERROR_DEG)
    upper = math.radians(bearing_deg + BEARING_ERROR_DEG)
    poly = clip_left(
        poly,
        station,
        (math.cos(lower), math.sin(lower)),
    )
    poly = clip_left(
        poly,
        station,
        (-math.cos(upper), -math.sin(upper)),
    )
    poly = clip_by_convex_polygon(
        poly,
        circumscribed_disk(
            station,
            MAX_RECEIVE_RADIUS,
            32,
        ),
    )
    return poly
def polygon_diameter(
    polygon: list[Point],
) -> tuple[float, tuple[Point, Point]]:
    if not polygon:
        raise ValueError (" 空多边形没有直径")
    if len(polygon) == 1:
        return 0.0, (polygon[0], polygon[0])
    best_sq = -1.0
    best_pair = (polygon[0], polygon[1])
    for i, p in enumerate(polygon):
        for q in polygon[i + 1:]:
            sq = (
                (p[0] - q[0]) ** 2
                + (p[1] - q[1]) ** 2
            )
            if sq > best_sq:
                best_sq = sq
                best_pair = (p, q)
    return math.sqrt(best_sq), best_pair
@dataclass(frozen=True)
class Circle :
    center: Point
    radius: float
def circle_from_two(a: Point, b: Point) -> Circle:
    center = (
        (a[0] + b[0]) / 2.0,
        (a[1] + b[1]) / 2.0,
    )
    return Circle(center, distance(center, a))
def circle_from_three(
    a: Point,
    b: Point,
    c: Point,
) -> Optional[Circle]:
    ax, ay = a
    bx, by = b
    cx, cy = c
    det = 2.0 * (
        ax * (by - cy)
        + bx * (cy - ay)
        + cx * (ay - by)
    )
    if abs(det) <= 1e-12:
        return None
    a2 = ax * ax + ay * ay
    b2 = bx * bx + by * by
    c2 = cx * cx + cy * cy
    ux = (
        a2 * (by - cy)
        + b2 * (cy - ay)
        + c2 * (ay - by)
    ) / det
    uy = (
        a2 * (cx - bx)
        + b2 * (ax - cx)
        + c2 * (bx - ax)
    ) / det
    center = (ux, uy)
    return Circle(center, distance(center, a))
def circle_contains(circle: Circle, point: Point) -> bool:
    return distance(circle.center, point) <= circle.radius + 1e-7
def minimum_enclosing_circle(points: list[Point]) -> Circle:
    if not points:
        raise ValueError (" 空点集没有最小包围圆")
    unique = list(dict.fromkeys(points))
    random.Random(20260912).shuffle(unique)
    circle: Optional[Circle] = None
    for i, p in enumerate(unique):
        if circle is not None and circle_contains(circle, p):
            continue
        circle = Circle(p, 0.0)
        for j in range(i):
            q = unique[j]
            if circle_contains(circle, q):
                continue
            circle = circle_from_two(p, q)
            for k in range(j):
                r = unique[k]
                if circle_contains(circle, r):
                    continue
                candidate = circle_from_three(p, q, r)
                if candidate is not None :
                    circle = candidate
                else:
                    options = [
                        circle_from_two(p, q),
                        circle_from_two(p, r),
                        circle_from_two(q, r),
                    ]
                    valid = [
                        item
                        for item in options
                        if circle_contains(item, p)
                        and circle_contains(item, q)
                        and circle_contains(item, r)
                    ]
                    circle = min(
                        valid,
                        key=lambda item: item.radius,
                    )
    assert circle is not None
    return Circle(
        circle.center,
        max(distance(circle.center, p) for p in unique),
    )
@dataclass(frozen=True)
class TargetInfo :
    center: Point
    work_radius: float
    mec_center: Point
    mec_radius: float
def target_info_from_polygon(polygon: list[Point]) -> TargetInfo:
    diameter, pair = polygon_diameter(polygon)
    # 工作中心取直径端点中点。
    center = (
        (pair[0][0] + pair[1][0]) / 2.0,
        (pair[0][1] + pair[1][1]) / 2.0,
    )
    work_radius = diameter / 2.0
    mec = minimum_enclosing_circle(polygon)
    return TargetInfo(
        center=center,
        work_radius=work_radius,
        mec_center=mec.center,
        mec_radius=mec.radius,
    )
@dataclass(frozen=True)
class GeometricProbe :
    point: Point
    sin_phi: float
    error_proxy_m: float
def point_in_convex_polygon(
    point: Point,
    polygon: list[Point],
    eps: float = 1e-7,
) -> bool:
    """ 判断点是否在凸多边形内。"""
    if not polygon:
        return False
    sign = 0
    for i in range(len(polygon)):
        a = polygon[i]
        b = polygon[(i + 1) % len(polygon)]
        value = cross(
            (b[0] - a[0], b[1] - a[1]),
            (point[0] - a[0], point[1] - a[1]),
        )
        if abs(value) <= eps:
            continue
        current = 1 if value > 0 else -1
        if sign == 0:
            sign = current
        elif current != sign:
            return False
    return True
def point_segment_distance(
    point: Point,
    a: Point,
    b: Point,
) -> float:
    """ 点到闭线段的欧氏距离。"""
    vx = b[0] - a[0]
    vy = b[1] - a[1]
    vv = vx * vx + vy * vy
    if vv <= 1e-18:
        return distance(point, a)
    t = (
        (point[0] - a[0]) * vx
        + (point[1] - a[1]) * vy
    ) / vv
    t = min(1.0, max(0.0, t))
    projection = (
        a[0] + t * vx,
        a[1] + t * vy,
    )
    return distance(point, projection)
def point_to_convex_polygon_distance(
    point: Point,
    polygon: list[Point],
) -> float:
    """ 计算点到凸多边形的最短距离。"""
    if not polygon:
        return math.inf
    if point_in_convex_polygon(point, polygon):
        return 0.0
    return min(
        point_segment_distance(
            point,
            polygon[i],
            polygon[(i + 1) % len(polygon)],
        )
        for i in range(len(polygon))
    )
def segment_disk_parameter_interval(
    start: Point,
    end: Point,
    center: Point,
    radius: float,
) -> Optional[tuple[float, float]]:
    """ 返回线段落在给定圆盘内的参数区间。"""
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    fx = start[0] - center[0]
    fy = start[1] - center[1]
    aa = dx * dx + dy * dy
    cc = fx * fx + fy * fy - radius * radius
    if aa <= 1e-18:
        if cc <= 1e-12:
            return 0.0, 1.0
        return None
    bb = 2.0 * (fx * dx + fy * dy)
    discriminant = bb * bb - 4.0 * aa * cc
    if discriminant < -1e-10:
        return None
    discriminant = max(0.0, discriminant)
    root = math.sqrt(discriminant)
    t1 = (-bb - root) / (2.0 * aa)
    t2 = (-bb + root) / (2.0 * aa)
    lo = max(0.0, min(t1, t2))
    hi = min(1.0, max(t1, t2))
    if lo > hi + 1e-12:
        return None
    return lo, hi
def bearing_pair_probe(
    first: "DirectionObservation",
    second: "DirectionObservation",
) -> Optional[GeometricProbe]:
    """ 由两条中心示向线生成几何试探点和误差代理。"""
    a = math.radians(first.bearing_deg)
    b = math.radians(second.bearing_deg)
    d1 = (math.cos(a), math.sin(a))
    d2 = (math.cos(b), math.sin(b))
    denominator = cross(d1, d2)
    sin_phi = abs(denominator)
    if sin_phi < GEOM_PROBE_MIN_SIN:
        return None
    delta = (
        second.station[0] - first.station[0],
        second.station[1] - first.station[1],
    )
    t1 = cross(delta, d2) / denominator
    t2 = cross(delta, d1) / denominator
    if t1 <= 0.0 or t2 <= 0.0:
        return None
    point = (
        first.station[0] + t1 * d1[0],
        first.station[1] + t1 * d1[1],
    )
    error_proxy = (
        math.tan(math.radians(BEARING_ERROR_DEG))
        * (abs(t1) + abs(t2))
        / sin_phi
    )
    return GeometricProbe(
        point=point,
        sin_phi=sin_phi,
        error_proxy_m=error_proxy,
    )
# 搜索骨架
def build_backbone() -> list[tuple[int, Point]]:
    """ 返回中心、内环 7 点、外环 14 点；编号固定用于严格判空。"""
    points: list[Point] = [(0.0, 0.0)]
    for i in range(INNER_SEARCH_COUNT):
        angle = 2.0 * math.pi * i / INNER_SEARCH_COUNT
        points.append(
            (
                INNER_SEARCH_RADIUS_M * math.cos(angle),
                INNER_SEARCH_RADIUS_M * math.sin(angle),
            )
        )
    for i in range(OUTER_SEARCH_COUNT):
        angle = (
            math.radians(OUTER_PHASE_DEG)
            + 2.0 * math.pi * i / OUTER_SEARCH_COUNT
        )
        points.append(
            (
                OUTER_SEARCH_RADIUS_M * math.cos(angle),
                OUTER_SEARCH_RADIUS_M * math.sin(angle),
            )
        )
    return list(enumerate(points))
def search_phase(ew_index: int) -> int:
    """ 先完成中心/内环快速筛查，再开放外环完备搜索。"""
    return 0 if ew_index <= INNER_SEARCH_COUNT else 1
def phase_a_omni_cover_radius() -> float:
    """ 中心 + 内环 7 点的距离覆盖半径，仅用于诊断。"""
    alpha = math.pi / INNER_SEARCH_COUNT
    inner_seam = INNER_SEARCH_RADIUS_M / (2.0 * math.cos(alpha))
    boundary_seam = math.sqrt(
        TARGET_RADIUS**2
        + INNER_SEARCH_RADIUS_M**2
        - 2.0
        * TARGET_RADIUS
        * INNER_SEARCH_RADIUS_M
        * math.cos(alpha)
    )
    return max(inner_seam, boundary_seam)
# 【学习注释】**22 点方向完备骨架的证书校验**。
# 判据（充要）：对目标圆内任一点 G，取其"近旁点"（距 G ≤ R_min 的点），
# 若这些点相对 G 的**最大方位角隙 < 180°**，则 G 严格位于这些点的凸包内部，
# 于是**任意朝向**的定向源都至少被一个点看到。
# 骨架 = 中心 1 + 990 m 内环 7 + 外接正十四边形外环 14 = 22 点。
# 他们还证明了：只用区域**内部**的点不可能方向完备 ⇒ 外环是必需的。

def validate_direction_complete_backbone() -> dict[str, float]:
    """ 检查当前 22 点搜索骨架是否仍满足离线证书参数。"""
    # 参数变动后需要重新做覆盖认证。
    cert_params = (
        7,
        990.0,
        14,
        10.0,
        0.7,
        2.0,
        997.0,
    )
    params_now = (
        INNER_SEARCH_COUNT,
        INNER_SEARCH_RADIUS_M,
        OUTER_SEARCH_COUNT,
        OUTER_HULL_MARGIN_M,
        OUTER_PHASE_DEG,
        CERTIFICATE_GRID_STEP_M,
        CERTIFICATE_NEIGHBOR_RADIUS_M,
    )
    if params_now != cert_params:
        raise RuntimeError (
            " 搜索骨架参数已变化，22 点局部凸包证书失效；"
            " 请重新执行离线全网格认证"
        )
    if SEARCH_POINT_COUNT != 22:
        raise RuntimeError (" 方向完备搜索点数应为 22")
    outer_apothem = OUTER_SEARCH_RADIUS_M * math.cos(
        math.pi / OUTER_SEARCH_COUNT
    )
    omni_cover = phase_a_omni_cover_radius()
    cell_radius = CERTIFICATE_GRID_STEP_M / math.sqrt(2.0)
    distance_slack = (
        MIN_RECEIVE_RADIUS
        - CERTIFICATE_NEIGHBOR_RADIUS_M
        - cell_radius
    )
    hull_slack = CERTIFICATE_MIN_HULL_MARGIN_M - cell_radius
    if outer_apothem + 1e-6 < TARGET_RADIUS:
        raise RuntimeError (" 外环多边形没有包含目标圆")
    if distance_slack <= 0.0 or hull_slack <= 0.0:
        raise RuntimeError ("22 点方向完备证书没有正安全余量")
    return {
        "outer_apothem_m": outer_apothem,
        "phase_a_omni_cover_radius_m": omni_cover,
        "certificate_cell_radius_m": cell_radius,
        "certificate_distance_slack_m": distance_slack,
        "certificate_hull_slack_m": hull_slack,
    }
# 【学习注释】用 **2 m 网格离线证书**验证上面的判据（不是随机抽样！）：
# 对每个网格点算最近旁点集的凸包余量与距离余量，
# 实测最小余量：距离 1.586 m、凸包 7.187 m，**严格为正** ⇒ 证书成立。
# 这种"确定性证书 + 独立验证网"的写法，可直接作为论文结论。

def run_geometry_self_test(samples: int = 20000) -> None:
    """ 不连接模拟器，随机复核搜索完备性和四点恢复命题。"""
    metrics = validate_direction_complete_backbone()
    points = [point for _, point in build_backbone()]
    rng = random.Random(20260912)
    for _ in range(samples):
        radius = TARGET_RADIUS * math.sqrt(rng.random())
        angle = 2.0 * math.pi * rng.random()
        source = (radius * math.cos(angle), radius * math.sin(angle))
        emission = 2.0 * math.pi * rng.random()
        direction = (math.cos(emission), math.sin(emission))
        detected = any(
            distance(point, source) <= MIN_RECEIVE_RADIUS + 1e-7
            and (
                (point[0] - source[0]) * direction[0]
                + (point[1] - source[1]) * direction[1]
            ) >= -1e-7
            for point in points
        )
        if not detected:
            raise AssertionError ((source, emission))
    # 在局部坐标系中检查恢复点。
    recovery = (
        (RECOVERY_FORWARD_M, RECOVERY_SIDE_M),
        (RECOVERY_FORWARD_M, -RECOVERY_SIDE_M),
        (0.0, RECOVERY_SIDE_M),
        (0.0, -RECOVERY_SIDE_M),
    )
    for r_i in range(6, 1501, 3):
        for error_i in range(-100, 101, 5):
            error = math.radians(error_i / 100.0)
            source = (r_i * math.cos(error), r_i * math.sin(error))
            worst_radius = max(MIN_RECEIVE_RADIUS, float(r_i))
            for beta_i in range(0, 360, 3):
                beta = math.radians(beta_i)
                emission = (math.cos(beta), math.sin(beta))
                station_visible = (
                    -source[0] * emission[0]
                    - source[1] * emission[1]
                ) >= -1e-10
                if not station_visible:
                    continue
                recovered = any(
                    distance(point, source) <= worst_radius + 1e-7
                    and (
                        (point[0] - source[0]) * emission[0]
                        + (point[1] - source[1]) * emission[1]
                    ) >= -1e-7
                    for point in recovery
                )
                if not recovered:
                    raise AssertionError ((source, beta_i))
    test_pts = [
        (10.0, 0.0),
        (1.0, 0.0),
        (2.0, 0.0),
    ]
    route_order = optimized_open_order(
        (0.0, 0.0),
        test_pts,
    )
    if sorted(route_order) != [0, 1, 2]:
        raise AssertionError ("2-opt 路线遗漏或重复节点")
    if open_route_length(
        (0.0, 0.0),
        [test_pts[index] for index in route_order],
    ) > 10.0 + 1e-9:
        raise AssertionError (" 开放路线优化产生了额外折返")
    scheduler = Q4Agent()
    remaining_route = scheduler.remaining_search_route()
    if (
        len(remaining_route) != SEARCH_POINT_COUNT
        or len({index for index, _ in remaining_route})
        != SEARCH_POINT_COUNT
    ):
        raise AssertionError (" 未来搜索路线遗漏或重复 EW")
    # 简单检查搜索期机会成本。
    scheduler.channels[1].state = "LOCALIZING"
    scheduler.channels[1].strong_position = remaining_route[1][1]
    if scheduler.choose_action()[1][0] != "EW":
        raise AssertionError (" 未来路线机会成本未能阻止提前折返")
    scheduler.channels[1].strong_position = scheduler.position
    if scheduler.choose_action()[1][0] != "CLEAR":
        raise AssertionError (" 当前位置保证清除被错误延后")
    marginal = scheduler.post_search_marginal_costs(
        [
            (0.0, ("CLEAR", 1, (10.0, 0.0), " 测试", False)),
            (0.0, ("CLEAR", 2, (20.0, 0.0), " 测试", False)),
        ],
        (0.0, 0.0),
    )
    if marginal[1] > 1e-9 or abs(marginal[2] - 10.0) > 1e-9:
        raise AssertionError (" 联合任务路线边际成本计算错误")
    # SAFE 层自检。
    # 新 clear 点必须保持严格可清除且路线不变长。
    safe_agent = Q4Agent()
    safe_agent.pending_ew.clear()
    safe_agent.planned_ew.clear()
    safe_agent.channels[1].state = "LOCALIZING"
    safe_agent.channels[1].polygon = [
        (9.0, -1.0),
        (11.0, -1.0),
        (11.0, 1.0),
        (9.0, 1.0),
    ]
    safe_info = safe_agent.target_info(1)
    old_target = safe_info.mec_center
    start = (0.0, 0.0)
    anchor = (20.0, 0.0)
    candidate, _, _ = safe_agent.best_safe_clear_point_between(
        1,
        start,
        old_target,
        anchor,
    )
    if not safe_agent.point_is_strict_guaranteed_clear(
        1,
        candidate,
    ):
        raise AssertionError ("SAFE clear 点失去严格 20m 保证")
    old_path = (
        distance(start, old_target)
        + distance(old_target, anchor)
    )
    new_path = (
        distance(start, candidate)
        + distance(candidate, anchor)
    )
    if new_path > old_path + 1e-8:
        raise AssertionError ("SAFE 局部支配产生了更长路径")
    print("Q4 geometry self-test passed")
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")
# HTTP 客户端
class OfficialClient :
    def __init__(self) -> None:
        self.request_counter = 0
    def next_request_id(self, prefix: str) -> str:
        self.request_counter += 1
        return (
            f"{prefix}-"
            f"{int(time.time() * 1000) }-"
            f"{self.request_counter}"
        )
    def base(self, request_id: str) -> dict:
        return {
            "arena_id": ARENA_ID,
            "robot_id": ROBOT_ID,
            "request_id": request_id,
        }
    def action_payload(
        self,
        request_id: str,
        position: Point,
        channel: int,
    ) -> dict:
        payload = self.base(request_id)
        payload["position"] = {
            "x": float(position[0]),
            "y": float(position[1]),
        }
        payload["channel"] = int(channel)
        return payload
    def post(self, path: str, payload: dict) -> dict:
        """ 发送请求；重试时保持原 request_id 和请求体。"""
        raw_body = json.dumps(
            payload,
            ensure_ascii=False,
        ).encode("utf-8")
        last_error: Optional[ Exception] = None
        for attempt in range(1, HTTP_RETRIES + 1):
            request = Request(
                BASE_URL + path,
                data=raw_body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urlopen(
                    request,
                    timeout=HTTP_TIMEOUT_S,
                ) as response:
                    return json.loads(
                        response.read().decode("utf-8")
                    )
            except HTTPError as exc:
                text = ""
                try:
                    text = exc.read().decode(
                        "utf-8",
                        errors="replace",
                    )
                except Exception :
                    pass
                raise RuntimeError (
                    f"HTTP {exc.code}: {text}"
                ) from exc
            except (
                URLError,
                TimeoutError,
                socket.timeout,
                ConnectionError,
            ) as exc:
                last_error = exc
                print(
                    f"[网络异常] attempt= {attempt}, "
                    f" 复用同一 request_id: {exc}"
                )
        assert last_error is not None
        raise last_error
    def enter(self) -> dict:
        rid = self.next_request_id("enter")
        return self.post("/enter", self.base(rid))
    def measure(
        self,
        position: Point,
        channel: int,
    ) -> dict:
        rid = self.next_request_id("measure")
        return self.post(
            "/measure",
            self.action_payload(
                rid,
                position,
                channel,
            ),
        )
    def clear(
        self,
        position: Point,
        channel: int,
    ) -> dict:
        rid = self.next_request_id("clear")
        return self.post(
            "/clear",
            self.action_payload(
                rid,
                position,
                channel,
            ),
        )
    def exit(self) -> dict:
        rid = self.next_request_id("exit")
        return self.post("/exit", self.base(rid))
# 状态记录
@dataclass
class DirectionObservation :
    station: Point
    bearing_deg: float
@dataclass
class ChannelRecord :
    state: str = "UNKNOWN"
    # UNKNOWN / LOCALIZING / CLEARED / CERTIFIED_EMPTY
    directions: list[DirectionObservation] = field(
        default_factory=list
    )
    polygon: Optional[list[Point]] = None
    strong_position: Optional[Point] = None
    measured_positions: list[Point] = field(
        default_factory=list
    )
    no_signal_ew_indices: set[int] = field(
        default_factory=set
    )
    lost: bool = False
    # 四点恢复队列。
    recovery_generation: int = -1
    recovery_pending: list[Point] = field(default_factory=list)
    sweep_pending: list[Point] = field(default_factory=list)
    sweep_polygon_version: int = -1
    failed_clear_points: list[Point] = field(
        default_factory=list
    )
@dataclass
class Ledger :
    move_distance: float = 0.0
    move_seconds: float = 0.0
    measure_count: int = 0
    measure_seconds: float = 0.0
    switch_count: int = 0
    switch_seconds: float = 0.0
    clear_success: int = 0
    clear_fail: int = 0
    clear_seconds: float = 0.0
    @property
    def total_seconds(self) -> float:
        return (
            self.move_seconds
            + self.measure_seconds
            + self.switch_seconds
            + self.clear_seconds
        )
# 主控制器
class Q4Agent :
    def __init__(self) -> None:
        self.client = OfficialClient()
        self.position: Point = (0.0, 0.0)
        self.current_measure_channel = 1
        self.channels = {
            ch: ChannelRecord()
            for ch in range(1, CHANNEL_COUNT + 1)
        }
        self.pending_ew = build_backbone()
        self.planned_ew: list[tuple[int, Point]] = []
        self.ledger = Ledger()
        self.trace_sequence = 0
        self.remaining_real_duration_s: Optional[float] = None
        self.server_virtual_time_s: Optional[float] = None
        self.action_count_by_category: dict[str, int] = {}
        self.move_distance_by_category_m: dict[str, float] = {}
        # 搜索统计
        self.early_stop_search_by_found16 = False
        self.geom_probe_attempts = 0
        self.geom_probe_success = 0
        self.search_visit_count = 0
        self.search_completed_virtual_time_s: Optional[float] = None
        self.opportunistic_measure_count = 0
        self.opportunistic_receive_count = 0
        self.recovery_attempts = 0
        self.recovery_receive_count = 0
        self.first_discovery_virtual_time_s: dict[int, float] = {}
        # 路线统计
        self.search_insertion_action_count = 0
        self.search_insertion_detour_m = 0.0
        self.search_insertion_future_cost_m = 0.0
        self.search_insertion_count_by_category: dict[str, int] = {}
        self.future_ew_deferred_states: set[tuple[int, int, bool]] = set()
        self.post_search_route_plan_count = 0
        # SAFE 层统计
        # 保存原 V5 下一动作，便于下一步汇合。
        self.safe_forced_action: Optional[tuple[float, tuple]] = None
        self.safe_forced_channels_patch = None
        self.safe_forced_planned_ew_patch = None
        self.safe_clear_shift_count = 0
        self.safe_clear_shift_saved_m = 0.0
        self.safe_clear_segment_hit_count = 0
    # 日志与统计
    @property
    def cleared_count(self) -> int:
        return sum(
            record.state == "CLEARED"
            for record in self.channels.values()
        )
    @property
    def empty_count(self) -> int:
        return sum(
            record.state == "CERTIFIED_EMPTY"
            for record in self.channels.values()
        )
    @property
    def found_count(self) -> int:
        """ 统计已经实际发现的有源频道数。"""
        return sum(
            record.state in {"LOCALIZING", "CLEARED"}
            for record in self.channels.values()
        )
    def active_channels(self) -> list[int]:
        return [
            ch
            for ch, record in self.channels.items()
            if record.state == "LOCALIZING"
        ]
    def trace(self, event: dict) -> None:
        self.trace_sequence += 1
        row = {
            "seq": self.trace_sequence,
            "virtual_time_est": self.ledger.total_seconds,
            "position": {
                "x": self.position[0],
                "y": self.position[1],
            },
            "cleared": self.cleared_count,
            "empty": self.empty_count,
            "ew_left": len(self.pending_ew),
            **event,
        }
        with TRACE_PATH.open(
            "a",
            encoding="utf-8",
        ) as handle:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + " \n"
            )
    def account_move(self, target: Point) -> float:
        moved = distance(self.position, target)
        self.ledger.move_distance += moved
        self.ledger.move_seconds += moved / SPEED
        self.position = target
        return moved
    @staticmethod
    def reason_category(reason: str) -> str:
        for prefix, category in (
            ("EW", "search_waypoint"),
            ("Q2", "q2_measure"),
            (" 直接", "direct_refine"),
            (" 定向源四点", "directional_recovery"),
            ("MEC", "mec_clear"),
            (" 两线", "geometric_probe_clear"),
            (" 强 Probe", "strong_probe_clear"),
            (" 弱 Probe", "weak_probe_clear"),
            (" 后验网格", "posterior_sweep_clear"),
            ("near", "near_clear"),
            ("clear 失败", "post_clear_measure"),
        ):
            if reason.startswith(prefix):
                return category
        return "other"
    def account_action_category(
        self,
        reason: str,
        moved: float,
    ) -> None:
        category = self.reason_category(reason)
        self.action_count_by_category[category] = (
            self.action_count_by_category.get(category, 0) + 1
        )
        self.move_distance_by_category_m[category] = (
            self.move_distance_by_category_m.get(category, 0.0) + moved
        )
    # 后验更新
    def target_info(self, channel: int) -> TargetInfo:
        record = self.channels[channel]
        if record.strong_position is not None :
            return TargetInfo(
                center=record.strong_position,
                work_radius=0.0,
                mec_center=record.strong_position,
                mec_radius=5.0,
            )
        if not record.polygon:
            raise RuntimeError (
                f" 频道 {channel} 没有可用后验"
            )
        return target_info_from_polygon(record.polygon)
    # measure
    def measure_at(
        self,
        position: Point,
        channel: int,
        *,
        ew_index: Optional[int] = None,
        mark_lost_on_no_signal: bool = True,
        reason: str = "",
    ) -> tuple[str, Optional[float]]:
        before = self.position
        response = self.client.measure(
            position,
            channel,
        )
        if response.get("accepted") is not True :
            raise RuntimeError (
                f"/measure rejected: {response}"
            )
        if "virtual_time_s" in response:
            self.server_virtual_time_s = float(response["virtual_time_s"])
        moved = self.account_move(position)
        self.account_action_category(reason, moved)
        switched = (
            self.current_measure_channel
            != channel
        )
        if switched:
            self.ledger.switch_count += 1
            self.ledger.switch_seconds += SWITCH_SECONDS
        self.current_measure_channel = channel
        self.ledger.measure_count += 1
        self.ledger.measure_seconds += MEASURE_SECONDS
        record = self.channels[channel]
        was_unknown = record.state == "UNKNOWN"
        record.measured_positions.append(position)
        result = response["measure_result"]
        bearing: Optional[float] = None
        if result == "direction":
            bearing = float(response["svd_deg"])
            record.state = "LOCALIZING"
            record.lost = False
            record.recovery_pending.clear()
            record.directions.append(
                DirectionObservation(
                    station=position,
                    bearing_deg=bearing,
                )
            )
            polygon = update_polygon_with_direction(
                record.polygon,
                position,
                bearing,
            )
            if not polygon:
                raise RuntimeError (
                    f" 频道 {channel} 后验意外为空"
                )
            record.polygon = polygon
        elif result == "near":
            record.state = "LOCALIZING"
            record.lost = False
            record.recovery_pending.clear()
            record.strong_position = position
        elif result == "no_signal":
            if (
                record.state == "UNKNOWN"
                and ew_index is not None
            ):
                record.no_signal_ew_indices.add(
                    ew_index
                )
            if (
                record.state == "LOCALIZING"
                and mark_lost_on_no_signal
            ):
                record.lost = True
        else:
            raise RuntimeError (
                f" 未知 measure_result: {result!r}"
            )
        if was_unknown and result in {"direction", "near"}:
            self.first_discovery_virtual_time_s[channel] = (
                self.ledger.total_seconds
            )
        diagnostic = {}
        if (
            record.state == "LOCALIZING"
            and record.polygon
        ):
            info = self.target_info(channel)
            diagnostic = {
                "work_radius": info.work_radius,
                "mec_radius": info.mec_radius,
                "direction_count": len(
                    record.directions
                ),
            }
        self.trace(
            {
                "action": "measure",
                "channel": channel,
                "reason": reason,
                "from": {
                    "x": before[0],
                    "y": before[1],
                },
                "to": {
                    "x": position[0],
                    "y": position[1],
                },
                "move_m": moved,
                "switch": switched,
                "result": result,
                "bearing": bearing,
                **diagnostic,
            }
        )
        # near 后直接 clear。
        if result == "near":
            self.clear_at(
                position,
                channel,
                reason="near 后原地保证清除",
                measure_after_failure=False,
            )
        self.apply_found16_shortcut()
        return result, bearing
    # clear
    def clear_at(
        self,
        position: Point,
        channel: int,
        *,
        reason: str,
        measure_after_failure: bool,
    ) -> bool:
        before = self.position
        response = self.client.clear(
            position,
            channel,
        )
        if response.get("accepted") is not True :
            raise RuntimeError (
                f"/clear rejected: {response}"
            )
        if "virtual_time_s" in response:
            self.server_virtual_time_s = float(response["virtual_time_s"])
        moved = self.account_move(position)
        self.account_action_category(reason, moved)
        success = (
            response.get("clear_result")
            == "success"
        )
        record = self.channels[channel]
        if success:
            self.ledger.clear_success += 1
            self.ledger.clear_seconds += CLEAR_SUCCESS_SECONDS
            record.state = "CLEARED"
            record.lost = False
        else:
            # SAFE clear 意外失败时取消缓存动作。
            self.safe_forced_action = None
            self.safe_forced_channels_patch = None
            self.safe_forced_planned_ew_patch = None
            self.ledger.clear_fail += 1
            self.ledger.clear_seconds += CLEAR_FAILURE_SECONDS
            record.failed_clear_points.append(
                position
            )
        self.trace(
            {
                "action": "clear",
                "channel": channel,
                "reason": reason,
                "from": {
                    "x": before[0],
                    "y": before[1],
                },
                "to": {
                    "x": position[0],
                    "y": position[1],
                },
                "move_m": moved,
                "result": (
                    "success"
                    if success
                    else response.get(
                        "clear_result",
                        "fail",
                    )
                ),
            }
        )
        if (
            not success
            and measure_after_failure
        ):
            self.measure_at(
                position,
                channel,
                mark_lost_on_no_signal=True,
                reason="clear 失败后原地补测",
            )
        self.apply_found16_shortcut()
        return success
    # Q2-fast 与恢复
    def q2_fast_candidates(
        self,
        channel: int,
    ) -> tuple[Point, Point]:
        info = self.target_info(channel)
        observation = self.channels[channel].directions[-1]
        theta = math.radians(
            observation.bearing_deg
        )
        normal = (
            -math.sin(theta),
            math.cos(theta),
        )
        center = info.center
        return (
            (
                center[0]
                + Q2_FAST_SIDE_M * normal[0],
                center[1]
                + Q2_FAST_SIDE_M * normal[1],
            ),
            (
                center[0]
                - Q2_FAST_SIDE_M * normal[0],
                center[1]
                - Q2_FAST_SIDE_M * normal[1],
            ),
        )
    def q2_fast_information(
        self,
        channel: int,
        point: Point,
    ) -> float:
        info = self.target_info(channel)
        last = self.channels[channel].directions[-1]
        a = (
            info.center[0] - last.station[0],
            info.center[1] - last.station[1],
        )
        b = (
            info.center[0] - point[0],
            info.center[1] - point[1],
        )
        na = math.hypot(a[0], a[1])
        nb = math.hypot(b[0], b[1])
        if na <= 1e-9 or nb <= 1e-9:
            return 1.0
        return min(
            1.0,
            abs(cross(a, b)) / (na * nb),
        )
    def recovery_candidates(
        self,
        channel: int,
    ) -> tuple[Point, Point, Point, Point]:
        """ 根据最后一次成功示向生成四个恢复测点。"""
        last = self.channels[channel].directions[-1]
        theta = math.radians(
            last.bearing_deg
        )
        forward = (
            math.cos(theta),
            math.sin(theta),
        )
        normal = (
            -math.sin(theta),
            math.cos(theta),
        )
        s = last.station
        forward_pair = (
            (
                s[0]
                + RECOVERY_FORWARD_M * forward[0]
                + RECOVERY_SIDE_M * normal[0],
                s[1]
                + RECOVERY_FORWARD_M * forward[1]
                + RECOVERY_SIDE_M * normal[1],
            ),
            (
                s[0]
                + RECOVERY_FORWARD_M * forward[0]
                - RECOVERY_SIDE_M * normal[0],
                s[1]
                + RECOVERY_FORWARD_M * forward[1]
                - RECOVERY_SIDE_M * normal[1],
            ),
        )
        lateral_pair = (
            (
                s[0] + RECOVERY_SIDE_M * normal[0],
                s[1] + RECOVERY_SIDE_M * normal[1],
            ),
            (
                s[0] - RECOVERY_SIDE_M * normal[0],
                s[1] - RECOVERY_SIDE_M * normal[1],
            ),
        )
        return forward_pair + lateral_pair
    def apply_found16_shortcut(self) -> None:
        """ 已发现 16 个频道时，直接结束剩余未知频道搜索。"""
        if self.found_count < SOURCE_MAX:
            return
        changed = False
        for record in self.channels.values():
            if record.state == "UNKNOWN":
                record.state = "CERTIFIED_EMPTY"
                changed = True
        if self.pending_ew:
            self.pending_ew.clear()
            self.planned_ew.clear()
            changed = True
        if changed:
            self.early_stop_search_by_found16 = True
            self.trace(
                {
                    "event": "found16_cancel_search",
                    "found_count": self.found_count,
                }
            )
    def best_geometric_probe(
        self,
        channel: int,
    ) -> Optional[GeometricProbe]:
        record = self.channels[channel]
        if len(record.directions) < 2 or not record.polygon:
            return None
        best: Optional[GeometricProbe] = None
        # 观测数很少，直接枚举组合。
        for first, second in itertools.combinations(
            record.directions,
            2,
        ):
            candidate = bearing_pair_probe(
                first,
                second,
            )
            if candidate is None :
                continue
            if (
                candidate.error_proxy_m
                > GEOM_PROBE_MAX_ERROR_M
            ):
                continue
            # 交点必须在当前后验内。
            if not point_in_convex_polygon(
                candidate.point,
                record.polygon,
            ):
                continue
            if (
                candidate.point[0] ** 2
                + candidate.point[1] ** 2
                > TARGET_RADIUS ** 2 + 1e-6
            ):
                continue
            if (
                best is None
                or candidate.error_proxy_m
                < best.error_proxy_m
            ):
                best = candidate
        return best
    # EW 顺路复测
    def already_measured_near(
        self,
        channel: int,
        point: Point,
        radius: float = 1.0,
    ) -> bool:
        return any(
            distance(point, old) <= radius
            for old
            in self.channels[channel].measured_positions
        )
    def opportunistic_score(
        self,
        channel: int,
        ew: Point,
    ) -> Optional[float]:
        record = self.channels[channel]
        if (
            record.state != "LOCALIZING"
            or not record.directions
            or record.strong_position is not None
        ):
            return None
        if self.already_measured_near(
            channel,
            ew,
        ):
            return None
        # lost 频道优先重捕获。
        if record.lost:
            return 100.0
        info = self.target_info(channel)
        # 已接近清除时不再额外 EW 复测。
        if (
            info.mec_radius <= CLEAR_RADIUS
            or (
                len(record.directions) >= 2
                and info.work_radius <= STRONG_PROBE_RADIUS_M
            )
            or self.best_geometric_probe(channel) is not None
        ):
            return None
        if (
            distance(
                ew,
                info.center,
            )
            > MAX_OPPORTUNISTIC_CENTER_DISTANCE_M
        ):
            return None
        last = record.directions[-1]
        baseline = distance(
            ew,
            last.station,
        )
        if baseline < MIN_OPPORTUNISTIC_BASELINE_M:
            return None
        v1 = (
            info.center[0] - last.station[0],
            info.center[1] - last.station[1],
        )
        v2 = (
            info.center[0] - ew[0],
            info.center[1] - ew[1],
        )
        n1 = math.hypot(v1[0], v1[1])
        n2 = math.hypot(v2[0], v2[1])
        if n1 <= 1e-9 or n2 <= 1e-9:
            sin_phi = 1.0
        else:
            sin_phi = min(
                1.0,
                abs(cross(v1, v2)) / (n1 * n2),
            )
        if sin_phi < MIN_OPPORTUNISTIC_SIN:
            return None
        # 先看交会角，再看基线。
        return (
            10.0 * sin_phi
            + min(baseline, 1500.0) / 1500.0
        )
    def execute_ew(
        self,
        ew_index: int,
        ew: Point,
    ) -> None:
        self.search_visit_count += 1
        unknown = [
            ch
            for ch, record in self.channels.items()
            if record.state == "UNKNOWN"
        ]
        active_scores: list[tuple[float, int]] = []
        for ch in self.active_channels():
            score = self.opportunistic_score(
                ch,
                ew,
            )
            if score is not None :
                active_scores.append(
                    (score, ch)
                )
        active_scores.sort(
            reverse=True
        )
        opportunistic = [
            ch
            for _, ch
            in active_scores[
                :MAX_OPPORTUNISTIC_PER_EW
            ]
        ]
        todo = list(
            dict.fromkeys(
                unknown
                + opportunistic
            )
        )
        # 当前频道优先。
        if self.current_measure_channel in todo:
            todo.remove(
                self.current_measure_channel
            )
            todo.insert(
                0,
                self.current_measure_channel,
            )
        if not todo:
            return
        for ch in todo:
            is_unknown = (
                self.channels[ch].state
                == "UNKNOWN"
            )
            result, _ = self.measure_at(
                ew,
                ch,
                ew_index=(
                    ew_index
                    if is_unknown
                    else None
                ),
                # EW 上的 no_signal 不改变 active 状态。
                mark_lost_on_no_signal=is_unknown,
                reason=(
                    "EW 存在性扫描"
                    if is_unknown
                    else "EW 零移动成本顺路定位"
                ),
            )
            if not is_unknown:
                self.opportunistic_measure_count += 1
                if result in {"direction", "near"}:
                    self.opportunistic_receive_count += 1
            # 找到 16 个源后停止剩余 UNKNOWN 扫描。
            if self.found_count >= SOURCE_MAX:
                return
    # 空频道判定
    def certify_empty(self) -> None:
        if self.pending_ew:
            return
        required = set(range(SEARCH_POINT_COUNT))
        for record in self.channels.values():
            if (
                record.state == "UNKNOWN"
                and required.issubset(
                    record.no_signal_ew_indices
                )
            ):
                record.state = "CERTIFIED_EMPTY"
    # clear 去重
    def too_close_to_failed_clear(
        self,
        channel: int,
        point: Point,
    ) -> bool:
        return any(
            distance(point, old)
            <= FAILED_CLEAR_AVOID_RADIUS_M
            for old
            in self.channels[channel].failed_clear_points
        )
    # 【学习注释】由"有信号 / 无信号"的观测构造**方向后验**：
    # 有信号 ⇒ (p − G)·u ≥ 0；无信号且 |p − G| ≤ R_min ⇒ (p − G)·u < 0。
    # 两类半平面约束求交，就把未知的定向方向 u 收缩成一段可行弧。

    def build_posterior_sweep(
        self,
        channel: int,
    ) -> list[Point]:
        """ 后验足够小时生成固定步长的清除网格。"""
        polygon = self.channels[channel].polygon
        if not polygon:
            return []
        xmin = min(point[0] for point in polygon)
        xmax = max(point[0] for point in polygon)
        ymin = min(point[1] for point in polygon)
        ymax = max(point[1] for point in polygon)
        nx = max(1, math.ceil((xmax - xmin) / SWEEP_GRID_STEP_M))
        ny = max(1, math.ceil((ymax - ymin) / SWEEP_GRID_STEP_M))
        xs = [xmin + i * SWEEP_GRID_STEP_M for i in range(nx + 1)]
        ys = [ymin + i * SWEEP_GRID_STEP_M for i in range(ny + 1)]
        points: list[Point] = []
        for row, y in enumerate(ys):
            row_points = [(x, y) for x in xs]
            if row % 2:
                row_points.reverse()
            points.extend(row_points)
        # Sweep 保持原 V5 顺序。
        return points
    # SAFE 局部路径优化
    # 【学习注释】判断一个清除动作是否**严格保证成功**：
    # 要求目标点落在"由当前后验给出的、保证包含真值的区域"内，
    # 且该区域能被 20 m 清除半径稳定覆盖（必要时布 27.72 m 间距的确定性网格）。
    # 只有这种动作才允许进入 SAFE 层的替换候选。

    def is_strict_guaranteed_clear_action(
        self,
        action: tuple,
    ) -> bool:
        if action[0] != "CLEAR":
            return False
        reason = action[3]
        return (
            reason.startswith("MEC<=20m 保证清除")
            or reason.startswith("near 保证清除")
        )
    def point_is_strict_guaranteed_clear(
        self,
        channel: int,
        point: Point,
    ) -> bool:
        """ 检查给定位置是否仍满足严格清除条件。"""
        record = self.channels[channel]
        limit = CLEAR_RADIUS - SAFE_CLEAR_MARGIN_M
        if record.strong_position is not None :
            return (
                distance(
                    point,
                    record.strong_position,
                )
                + NEAR_RADIUS
                <= CLEAR_RADIUS - SAFE_CLEAR_MARGIN_M
            )
        if not record.polygon:
            return False
        return all(
            distance(point, vertex) <= limit + 1e-9
            for vertex in record.polygon
        )
    def clear_segment_feasible_point(
        self,
        channel: int,
        start: Point,
        end: Point,
        preferred: Point,
    ) -> Optional[Point]:
        """ 若路径线段穿过严格清除区域，返回可用清除点。"""
        record = self.channels[channel]
        limit = CLEAR_RADIUS - SAFE_CLEAR_MARGIN_M
        lo = 0.0
        hi = 1.0
        constraints: list[tuple[Point, float]] = []
        if record.strong_position is not None :
            constraints.append(
                (
                    record.strong_position,
                    CLEAR_RADIUS
                    - NEAR_RADIUS
                    - SAFE_CLEAR_MARGIN_M,
                )
            )
        elif record.polygon:
            constraints.extend(
                (vertex, limit)
                for vertex in record.polygon
            )
        else:
            return None
        for center, radius in constraints:
            interval = segment_disk_parameter_interval(
                start,
                end,
                center,
                radius,
            )
            if interval is None :
                return None
            lo = max(lo, interval[0])
            hi = min(hi, interval[1])
            if lo > hi + 1e-12:
                return None
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        dd = dx * dx + dy * dy
        if dd <= 1e-18:
            candidate = start
        else:
            preferred_t = (
                (preferred[0] - start[0]) * dx
                + (preferred[1] - start[1]) * dy
            ) / dd
            t = min(hi, max(lo, preferred_t))
            candidate = (
                start[0] + t * dx,
                start[1] + t * dy,
            )
        if self.point_is_strict_guaranteed_clear(
            channel,
            candidate,
        ):
            return candidate
        return None
    def clear_mec_safe_radius(
        self,
        channel: int,
        original_target: Point,
    ) -> float:
        """ 计算以原 MEC 中心为圆心的安全移动半径。"""
        record = self.channels[channel]
        if record.strong_position is not None :
            if distance(
                original_target,
                record.strong_position,
            ) <= 1e-6:
                return max(
                    0.0,
                    CLEAR_RADIUS
                    - NEAR_RADIUS
                    - SAFE_CLEAR_MARGIN_M,
                )
            return 0.0
        if not record.polygon:
            return 0.0
        info = self.target_info(channel)
        if distance(
            original_target,
            info.mec_center,
        ) > 1e-5:
            return 0.0
        return max(
            0.0,
            CLEAR_RADIUS
            - info.mec_radius
            - SAFE_CLEAR_MARGIN_M,
        )
    # 【学习注释】在"当前点 → 原清除点"这一段上，找一个**仍然严格保证清除、
    # 且使两段总移动严格更短**的替代点：等于把清除点挪到更顺路的位置，
    # 但不牺牲任何保证性。

    def best_safe_clear_point_between(
        self,
        channel: int,
        start: Point,
        original_target: Point,
        next_target: Optional[Point],
    ) -> tuple[Point, float, bool]:
        """ 在保证清除成功的前提下寻找更短的清除位置。"""
        if next_target is None :
            # 最后一动作尽量把 clear 点往当前位置移动。
            radius = self.clear_mec_safe_radius(
                channel,
                original_target,
            )
            old_length = distance(
                start,
                original_target,
            )
            if radius <= 1e-12 or old_length <= 1e-12:
                return original_target, 0.0, False
            shift = min(radius, old_length)
            candidate = (
                original_target[0]
                + (
                    start[0] - original_target[0]
                )
                * shift
                / old_length,
                original_target[1]
                + (
                    start[1] - original_target[1]
                )
                * shift
                / old_length,
            )
            if not self.point_is_strict_guaranteed_clear(
                channel,
                candidate,
            ):
                return original_target, 0.0, False
            new_length = distance(start, candidate)
            saved = old_length - new_length
            if saved <= SAFE_MIN_ROUTE_GAIN_M:
                return original_target, 0.0, False
            return candidate, saved, False
        old_length = (
            distance(start, original_target)
            + distance(original_target, next_target)
        )
        # 优先找路径线段上的可清除点。
        seg_point = self.clear_segment_feasible_point(
            channel,
            start,
            next_target,
            original_target,
        )
        if seg_point is not None :
            new_length = (
                distance(start, seg_point)
                + distance(seg_point, next_target)
            )
            saved = old_length - new_length
            if saved > SAFE_MIN_ROUTE_GAIN_M:
                return seg_point, saved, True
        # 否则在 MEC 安全圆上找更短点。
        radius = self.clear_mec_safe_radius(
            channel,
            original_target,
        )
        if radius <= 1e-12:
            return original_target, 0.0, False
        best_point = original_target
        best_length = old_length
        # 这里只需找到严格更短的可行点。
        # 找不到就回退原目标。
        for index in range(180):
            angle = 2.0 * math.pi * index / 180.0
            candidate = (
                original_target[0]
                + radius * math.cos(angle),
                original_target[1]
                + radius * math.sin(angle),
            )
            if not self.point_is_strict_guaranteed_clear(
                channel,
                candidate,
            ):
                continue
            cand_len = (
                distance(start, candidate)
                + distance(candidate, next_target)
            )
            if cand_len + 1e-9 < best_length:
                best_length = cand_len
                best_point = candidate
        saved = old_length - best_length
        if saved <= SAFE_MIN_ROUTE_GAIN_M:
            return original_target, 0.0, False
        return best_point, saved, False
    @staticmethod
    def shadow_apply_guaranteed_clear_success(
        shadow: "Q4Agent",
        action: tuple,
    ) -> None:
        """ 在副本状态中模拟一次必成功清除，不发送 HTTP 请求。"""
        _, channel, point, _, _ = action
        shadow.position = point
        record = shadow.channels[channel]
        record.state = "CLEARED"
        record.lost = False
        record.recovery_pending.clear()
        record.sweep_pending.clear()
        shadow.apply_found16_shortcut()
        shadow.certify_empty()
    # 【学习注释】**SAFE 层**（收尾优化）：只接受"严格局部支配"的收益 ——
    #   ① 替换点仍严格保证清除；② 两动作总移动严格更短；
    # 并强制**锚定原调度的下一个动作**，从而保证"同一前状态下决策不变"，
    # 不会因为优化而引入路线抖动。

    def safe_transform_selected_action(
        self,
        chosen: tuple[float, tuple],
    ) -> tuple[float, tuple]:
        """ 保持原 V5 动作顺序，只尝试缩短已选保证清除动作的路径。"""
        score, action = chosen
        if not self.is_strict_guaranteed_clear_action(action):
            return chosen
        _, channel, original_target, reason, measure_after_failure = action
        # 先复核原 clear 点。
        if not self.point_is_strict_guaranteed_clear(
            channel,
            original_target,
        ):
            return chosen
        shadow = copy.deepcopy(self)
        self.shadow_apply_guaranteed_clear_success(
            shadow,
            action,
        )
        next_chosen = shadow._choose_action_v5()
        next_target = (
            next_chosen[1][2]
            if next_chosen is not None
            else None
        )
        new_target, saved_m, segment_hit = (
            self.best_safe_clear_point_between(
                channel,
                self.position,
                original_target,
                next_target,
            )
        )
        if saved_m <= SAFE_MIN_ROUTE_GAIN_M:
            return chosen
        # 新点再核验一次。
        if not self.point_is_strict_guaranteed_clear(
            channel,
            new_target,
        ):
            return chosen
        # 保存 shadow 中的下一动作和计划。
        if next_chosen is not None :
            self.safe_forced_action = next_chosen
            self.safe_forced_channels_patch = copy.deepcopy(
                shadow.channels
            )
            self.safe_forced_planned_ew_patch = copy.deepcopy(
                shadow.planned_ew
            )
        self.safe_clear_shift_count += 1
        self.safe_clear_shift_saved_m += saved_m
        if segment_hit:
            self.safe_clear_segment_hit_count += 1
        self.trace(
            {
                "event": "safe_clear_local_dominance",
                "channel": channel,
                "reason": reason,
                "old_target": {
                    "x": original_target[0],
                    "y": original_target[1],
                },
                "new_target": {
                    "x": new_target[0],
                    "y": new_target[1],
                },
                "next_anchor": (
                    {
                        "x": next_target[0],
                        "y": next_target[1],
                    }
                    if next_target is not None
                    else None
                ),
                "saved_m_est": saved_m,
                "segment_zero_detour": segment_hit,
            }
        )
        new_action = (
            "CLEAR",
            channel,
            new_target,
            reason + " [SAFE 局部支配点]",
            measure_after_failure,
        )
        # 保留原 V5 分数。
        return score, new_action
    # 任务选择
    def search_route_information_key(
        self,
        route: list[tuple[int, Point]],
    ) -> tuple[int, float]:
        """ 等长环路中，让可顺路形成大交会角的 EW 尽量提前。"""
        total_delay = 0
        total_quality = 0.0
        for ch in self.active_channels():
            opportunities = []
            for step, (_, point) in enumerate(route):
                score = self.opportunistic_score(ch, point)
                if score is not None :
                    opportunities.append((step, -score))
            if opportunities:
                first_step, negative_score = min(opportunities)
                total_delay += first_step
                total_quality -= negative_score
        return total_delay, -total_quality
    def plan_current_search_phase(self) -> None:
        """ 为当前搜索环生成固定开放路线。"""
        if not self.pending_ew:
            self.planned_ew.clear()
            return
        current_phase = min(
            search_phase(index)
            for index, _ in self.pending_ew
        )
        phase_entries = sorted(
            (
                item
                for item in self.pending_ew
                if search_phase(item[0]) == current_phase
            ),
            key=lambda item: item[0],
        )
        if len(phase_entries) <= 1:
            self.planned_ew = phase_entries
            return
        routes: list[list[tuple[int, Point]]] = []
        count = len(phase_entries)
        for start in range(count):
            routes.append(
                [
                    phase_entries[(start + offset) % count]
                    for offset in range(count)
                ]
            )
            routes.append(
                [
                    phase_entries[(start - offset) % count]
                    for offset in range(count)
                ]
            )
        lengths = [
            open_route_length(
                self.position,
                [point for _, point in route],
            )
            for route in routes
        ]
        shortest_length = min(lengths)
        shortest_routes = [
            route
            for route, length in zip(routes, lengths)
            if length <= shortest_length + 1e-7
        ]
        self.planned_ew = min(
            shortest_routes,
            key=self.search_route_information_key,
        )
    def next_search_waypoint(self) -> Optional[tuple[int, Point]]:
        if not self.pending_ew:
            return None
        pending_indices = {
            index
            for index, _ in self.pending_ew
        }
        self.planned_ew = [
            item
            for item in self.planned_ew
            if item[0] in pending_indices
        ]
        if not self.planned_ew:
            self.plan_current_search_phase()
        return self.planned_ew[0]
    def mark_search_waypoint_done(self, ew_index: int) -> None:
        self.planned_ew = [
            item
            for item in self.planned_ew
            if item[0] != ew_index
        ]
    def remaining_search_route(self) -> list[tuple[int, Point]]:
        """ 返回当前固定环之后的完整剩余搜索路线，不改变实际计划。"""
        if not self.pending_ew:
            return []
        # 确保当前搜索环已有路线。
        self.next_search_waypoint()
        pending_indices = {
            index
            for index, _ in self.pending_ew
        }
        route = [
            item
            for item in self.planned_ew
            if item[0] in pending_indices
        ]
        used_indices = {
            index
            for index, _ in route
        }
        cursor = route[-1][1] if route else self.position
        phases_left = sorted(
            {
                search_phase(index)
                for index, _ in self.pending_ew
                if index not in used_indices
            }
        )
        for phase in phases_left:
            entries = sorted(
                (
                    item
                    for item in self.pending_ew
                    if item[0] not in used_indices
                    and search_phase(item[0]) == phase
                ),
                key=lambda item: item[0],
            )
            if len(entries) <= 1:
                best_route = entries
            else:
                count = len(entries)
                cyclic_routes = []
                for start in range(count):
                    cyclic_routes.append(
                        [
                            entries[(start + offset) % count]
                            for offset in range(count)
                        ]
                    )
                    cyclic_routes.append(
                        [
                            entries[(start - offset) % count]
                            for offset in range(count)
                        ]
                    )
                best_route = min(
                    cyclic_routes,
                    key=lambda candidate: (
                        open_route_length(
                            cursor,
                            [point for _, point in candidate],
                        ),
                        self.search_route_information_key(candidate),
                    ),
                )
            route.extend(best_route)
            used_indices.update(index for index, _ in best_route)
            if best_route:
                cursor = best_route[-1][1]
        return route
    def best_future_search_insertion_detour(
        self,
        target: Point,
        route: Optional[list[tuple[int, Point]]] = None,
    ) -> float:
        """ 估计目标点插入后续搜索路线的最小绕路。"""
        if route is None :
            route = self.remaining_search_route()
        if not route:
            return 0.0
        previous = route[0][1]
        best = math.inf
        for _, next_point in route[1:]:
            detour = (
                distance(previous, target)
                + distance(target, next_point)
                - distance(previous, next_point)
            )
            best = min(best, max(0.0, detour))
            previous = next_point
        # 最后一个 EW 后直接接目标点。
        best = min(best, distance(previous, target))
        return best
    def post_search_marginal_costs(
        self,
        candidates: list[tuple[float, tuple]],
        start: Point,
    ) -> dict[int, float]:
        """ 估计搜索结束后各频道任务的联合路线边际成本。"""
        per_channel: dict[int, tuple[float, tuple]] = {}
        for item in candidates:
            action = item[1]
            if action[0] == "EW":
                continue
            channel = action[1]
            old = per_channel.get(channel)
            if old is None or item[0] < old[0]:
                per_channel[channel] = item
        channels = list(per_channel)
        if not channels:
            return {}
        points = [
            per_channel[channel][1][2]
            for channel in channels
        ]
        full_order = optimized_open_order(start, points)
        full_length = open_route_length(
            start,
            [points[index] for index in full_order],
        )
        marginal: dict[int, float] = {}
        for removed_index, channel in enumerate(channels):
            reduced_points = [
                point
                for index, point in enumerate(points)
                if index != removed_index
            ]
            reduced_order = optimized_open_order(start, reduced_points)
            reduced_length = open_route_length(
                start,
                [reduced_points[index] for index in reduced_order],
            )
            marginal[channel] = max(
                0.0,
                full_length - reduced_length,
            )
        return marginal
    def remaining_ew_can_refine(self, channel: int) -> bool:
        """ 是否存在无需额外移动且满足原顺路门槛的未来 EW。"""
        return any(
            self.opportunistic_score(channel, point) is not None
            for _, point in self.pending_ew
        )
    def search_insertion_detour(self, target: Point) -> float:
        next_ew = self.next_search_waypoint()
        if next_ew is None :
            return 0.0
        _, next_point = next_ew
        return max(
            0.0,
            distance(self.position, target)
            + distance(target, next_point)
            - distance(self.position, next_point),
        )
    def action_travel_cost(self, target: Point) -> float:
        """ 计算当前动作的移动代价。"""
        if not self.pending_ew:
            return distance(self.position, target)
        return self.search_insertion_detour(target)
    def search_action_priority(
        self,
        item: tuple[float, tuple],
        current_detour: float,
        future_detour: float,
        can_refine_on_ew: bool,
    ) -> Optional[tuple[float, float, float, float]]:
        """ 给出搜索期联合动作优先级；None 表示先继续搜索路线。"""
        if can_refine_on_ew:
            future_detour = 0.0
        action = item[1]
        reason = action[3]
        reliability = 1.0
        if reason.startswith(" 两线"):
            reliability = GEOM_PROBE_RELIABILITY
        elif reason.startswith(" 强 Probe"):
            reliability = STRONG_PROBE_RELIABILITY
        elif reason.startswith(" 弱 Probe"):
            reliability = WEAK_PROBE_RELIABILITY
        elif reason.startswith("Q2"):
            reliability = Q2_MEASURE_RELIABILITY
        elif action[0] == "RECOVERY":
            reliability = RECOVERY_RELIABILITY
        future_detour *= reliability
        # score = 绕路 - 收益。
        route_regret = item[0] - future_detour
        if route_regret > 1e-9:
            return None
        return (
            route_regret,
            item[0],
            current_detour,
            future_detour,
        )
    def choose_post_search_action(
        self,
        candidates: list[tuple[float, tuple]],
    ) -> tuple[float, tuple]:
        """ 搜索结束后按当前任务池生成开放路线并选第一步。"""
        per_channel: dict[int, tuple[float, tuple]] = {}
        for item in candidates:
            action = item[1]
            channel = action[1]
            old = per_channel.get(channel)
            if old is None or item[0] < old[0]:
                per_channel[channel] = item
        tasks = list(per_channel.values())
        points = [item[1][2] for item in tasks]
        order = optimized_open_order(self.position, points)
        if not order:
            raise RuntimeError (" 搜索后仍有活动频道但任务路线为空")
        self.post_search_route_plan_count += 1
        planned = tasks[order[0]]
        greedy = min(candidates, key= lambda item: item[0])
        plan_dist = distance(
            self.position,
            planned[1][2],
        )
        greedy_distance = distance(
            self.position,
            greedy[1][2],
        )
        if (
            planned[0]
            > greedy[0] + POST_ROUTE_FIRST_STEP_REGRET_M
            or plan_dist
            > greedy_distance + POST_ROUTE_FIRST_STEP_REGRET_M
        ):
            return greedy
        return planned
    def _choose_action_v5(self):
        candidates = []
        # 每个搜索环固定路线。
        next_ew = self.next_search_waypoint()
        if next_ew is not None :
            ew_index, ew = next_ew
            current_search_phase = search_phase(ew_index)
            progress_bonus = (
                INNER_EW_PROGRESS_BONUS_M
                if current_search_phase == 0
                else EW_PROGRESS_BONUS_M
            )
            candidates.append(
                (
                    distance(
                        self.position,
                        ew,
                    )
                    - progress_bonus,
                    (
                        "EW",
                        ew_index,
                        ew,
                        " 完备搜索",
                    ),
                )
            )
        # LOCALIZING 频道
        for ch in self.active_channels():
            record = self.channels[ch]
            if record.strong_position is not None :
                candidates.append(
                    (
                        self.action_travel_cost(record.strong_position)
                        - GUARANTEED_CLEAR_BONUS_M,
                        (
                            "CLEAR",
                            ch,
                            record.strong_position,
                            "near 保证清除",
                            False,
                        ),
                    )
                )
                continue
            info = self.target_info(ch)
            # lost 频道按四点恢复。
            if record.lost:
                generation = len(record.directions)
                if record.recovery_generation != generation:
                    record.recovery_generation = generation
                    recovery_points = self.recovery_candidates(ch)
                    # 先走前向恢复点，再补侧向点。
                    forward_pair = sorted(
                        recovery_points[:2],
                        key=lambda point: distance(self.position, point),
                    )
                    lateral_pair = sorted(
                        recovery_points[2:],
                        key=lambda point: distance(self.position, point),
                    )
                    record.recovery_pending = forward_pair + lateral_pair
                if not record.recovery_pending:
                    raise RuntimeError (
                        f" 频道 {ch} 四点恢复扇全部失败；"
                        " 这与 180 度覆盖及 1000m 最小半径约束矛盾"
                    )
                recovery = record.recovery_pending[0]
                candidates.append(
                    (
                        self.action_travel_cost(recovery),
                        (
                            "RECOVERY",
                            ch,
                            recovery,
                            " 定向源四点恢复扇",
                        ),
                    )
                )
                continue
            # 保证 clear
            if (
                info.mec_radius
                <= CLEAR_RADIUS - 1e-6
                and not self.too_close_to_failed_clear(
                    ch,
                    info.mec_center,
                )
            ):
                candidates.append(
                    (
                        self.action_travel_cost(info.mec_center)
                        - GUARANTEED_CLEAR_BONUS_M,
                        (
                            "CLEAR",
                            ch,
                            info.mec_center,
                            "MEC<=20m 保证清除",
                            False,
                        ),
                    )
                )
                continue
            n_direction = len(
                record.directions
            )
            # 几何 probe
            geometric_probe = self.best_geometric_probe(ch)
            if (
                geometric_probe is not None
                and not self.too_close_to_failed_clear(
                    ch,
                    geometric_probe.point,
                )
            ):
                candidates.append(
                    (
                        self.action_travel_cost(geometric_probe.point)
                        - GEOM_PROBE_BONUS_M,
                        (
                            "CLEAR",
                            ch,
                            geometric_probe.point,
                            (
                                " 两线交点 Probe "
                                f"err≈{geometric_probe.error_proxy_m:.1f}m, "
                                f"sin={geometric_probe.sin_phi:.2f}"
                            ),
                            True,
                        ),
                    )
                )
            # probe clear
            if (
                n_direction >= 2
                and info.work_radius
                <= STRONG_PROBE_RADIUS_M
                and not self.too_close_to_failed_clear(
                    ch,
                    info.center,
                )
            ):
                candidates.append(
                    (
                        self.action_travel_cost(info.center)
                        - STRONG_PROBE_BONUS_M,
                        (
                            "CLEAR",
                            ch,
                            info.center,
                            (
                                " 强 Probe Clear "
                                f"workR={info.work_radius:.1f}"
                            ),
                            True,
                        ),
                    )
                )
                continue
            if (
                n_direction >= 2
                and info.work_radius
                <= WEAK_PROBE_RADIUS_M
                and not self.too_close_to_failed_clear(
                    ch,
                    info.center,
                )
            ):
                candidates.append(
                    (
                        self.action_travel_cost(info.center)
                        - WEAK_PROBE_BONUS_M,
                        (
                            "CLEAR",
                            ch,
                            info.center,
                            (
                                " 弱 Probe Clear "
                                f"workR={info.work_radius:.1f}"
                            ),
                            True,
                        ),
                    )
                )
                # 同时保留普通 refine。
            # 后验较小时改用 Sweep。
            if (
                n_direction >= SWEEP_TRIGGER_DIRECTION_COUNT
                and info.work_radius <= SWEEP_TRIGGER_WORK_RADIUS_M
            ):
                if record.sweep_polygon_version != n_direction:
                    record.sweep_polygon_version = n_direction
                    record.sweep_pending = self.build_posterior_sweep(ch)
                if record.sweep_pending:
                    sweep_point = min(
                        record.sweep_pending,
                        key=lambda point: distance(self.position, point),
                    )
                    candidates.append(
                        (
                            self.action_travel_cost(sweep_point)
                            - SWEEP_CLEAR_BONUS_M,
                            (
                                "SWEEP_CLEAR",
                                ch,
                                sweep_point,
                                " 后验网格保证清除",
                                False,
                            ),
                        )
                    )
            # 首次 direction 后加入 Q2-fast。
            if n_direction == 1:
                for point in self.q2_fast_candidates(ch):
                    info_gain = self.q2_fast_information(
                        ch,
                        point,
                    )
                    candidates.append(
                        (
                            self.action_travel_cost(point)
                            - Q2_FAST_INFO_BONUS_M
                            * info_gain,
                            (
                                "MEASURE",
                                ch,
                                point,
                                (
                                    "Q2-fast 第二测点 "
                                    f"sin≈{info_gain:.2f}"
                                ),
                            ),
                        )
                    )
            # 始终保留后验中心候选。
            candidates.append(
                (
                    self.action_travel_cost(info.center),
                    (
                        "MEASURE",
                        ch,
                        info.center,
                        (
                            " 直接逼近后验中心 "
                            f"workR={info.work_radius:.1f}, "
                            f"mecR={info.mec_radius:.1f}"
                        ),
                    ),
                )
            )
        if not candidates:
            return None
        if self.pending_ew:
            ew_candidate = next(
                item
                for item in candidates
                if item[1][0] == "EW"
            )
            due_actions = []
            remaining_route = self.remaining_search_route()
            refine_cache: dict[int, bool] = {}
            route_end = remaining_route[-1][1]
            marginal_cost = self.post_search_marginal_costs(
                candidates,
                route_end,
            )
            for item in candidates:
                action = item[1]
                if action[0] == "EW":
                    continue
                current_detour = self.search_insertion_detour(action[2])
                future_detour = self.best_future_search_insertion_detour(
                    action[2],
                    remaining_route,
                )
                future_detour = min(
                    future_detour,
                    marginal_cost.get(
                        action[1],
                        future_detour,
                    ),
                )
                # 有顺路 EW 时，measure/recovery 的未来移动代价按 0。
                can_refine_on_ew = False
                if action[0] in {"MEASURE", "RECOVERY"}:
                    channel = action[1]
                    if channel not in refine_cache:
                        refine_cache[channel] = (
                            self.remaining_ew_can_refine(channel)
                        )
                    can_refine_on_ew = refine_cache[channel]
                # score = 绕路 - 收益。
                # 当前净代价不高于未来插入代价时才提前做。
                priority = self.search_action_priority(
                    item,
                    current_detour,
                    future_detour,
                    can_refine_on_ew,
                )
                if priority is not None :
                    due_actions.append(
                        (*priority, item)
                    )
                elif can_refine_on_ew:
                    record = self.channels[action[1]]
                    self.future_ew_deferred_states.add(
                        (
                            action[1],
                            len(record.directions),
                            record.lost,
                        )
                    )
            if not due_actions:
                return ew_candidate
            (
                _,
                _,
                current_detour,
                future_detour,
                chosen,
            ) = min(
                due_actions,
                key=lambda entry: entry[:3],
            )
            self.search_insertion_action_count += 1
            self.search_insertion_detour_m += current_detour
            self.search_insertion_future_cost_m += future_detour
            category = self.reason_category(chosen[1][3])
            self.search_insertion_count_by_category[category] = (
                self.search_insertion_count_by_category.get(
                    category,
                    0,
                )
                + 1
            )
            return chosen
        return self.choose_post_search_action(candidates)
    def choose_action(self):
        # SAFE 层有缓存动作时先执行缓存。
        if self.safe_forced_action is not None :
            if self.safe_forced_channels_patch is not None :
                self.channels = copy.deepcopy(
                    self.safe_forced_channels_patch
                )
            if self.safe_forced_planned_ew_patch is not None :
                self.planned_ew = copy.deepcopy(
                    self.safe_forced_planned_ew_patch
                )
            chosen = self.safe_forced_action
            self.safe_forced_action = None
            self.safe_forced_channels_patch = None
            self.safe_forced_planned_ew_patch = None
            return chosen
        # 先按原 V5 选动作，再由 SAFE 层尝试缩短 clear 路径。
        chosen = self._choose_action_v5()
        if chosen is None :
            return None
        return self.safe_transform_selected_action(chosen)
    # 运行
    def run(self) -> None:
        # 墙钟时间只做统计。
        wall_start = time.perf_counter()
        for path in (TRACE_PATH, SUMMARY_PATH):
            try:
                path.unlink()
            except FileNotFoundError :
                pass
        validation = validate_direction_complete_backbone()
        print("=" * 80)
        print("Q4 DIRECTION-COMPLETE SEARCH")
        print(
            f"{SEARCH_POINT_COUNT} 点局部凸包方向完备搜索"
        )
        print(
            " 阶段 A 诊断覆盖半径 = "
            f"{validation['phase_a_omni_cover_radius_m']:.3f} m"
        )
        print(
            " 外环正十四边形内切半径 = "
            f"{validation['outer_apothem_m']:.3f} m"
        )
        print(
            " 证书距离/凸包安全余量 = "
            f"{validation['certificate_distance_slack_m']:.3f} / "
            f"{validation['certificate_hull_slack_m']:.3f} m"
        )
        print("=" * 80)
        enter = self.client.enter()
        if enter.get("accepted") is not True :
            raise RuntimeError (
                f"/enter rejected: {enter}"
            )
        OUTPUT_DIR.mkdir(parents=True, exist_ok= False)
        self.remaining_real_duration_s = float(
            enter["remaining_real_duration_s"]
        )
        if "virtual_time_s" in enter:
            self.server_virtual_time_s = float(enter["virtual_time_s"])
        self.trace(
            {
                "action": "enter",
                "remaining_real_duration_s": (
                    self.remaining_real_duration_s
                ),
            }
        )
        for logical_step in range(
            1,
            MAX_LOGICAL_ACTIONS + 1,
        ):
            self.apply_found16_shortcut()
            self.certify_empty()
            if self.cleared_count >= SOURCE_MAX:
                print(" 已清除 16 个源，提前结束。")
                break
            if (
                self.cleared_count
                + self.empty_count
                == CHANNEL_COUNT
            ):
                print(
                    " 全部 20 频道已清除或严格判空。"
                )
                break
            chosen = self.choose_action()
            if chosen is None :
                raise RuntimeError (
                    " 无动作但尚未满足终止条件"
                )
            score, action = chosen
            kind = action[0]
            if kind == "EW":
                _, ew_index, ew, reason = action
                print(
                    f"\n[{logical_step}] EW {ew_index} "
                    f"score={score:.1f} "
                    f"({ew[0]:.1f},{ew[1]:.1f})"
                )
                self.pending_ew.remove(
                    (ew_index, ew)
                )
                self.mark_search_waypoint_done(ew_index)
                self.execute_ew(
                    ew_index,
                    ew,
                )
            elif kind in {"MEASURE", "RECOVERY"}:
                _, ch, point, reason = action
                print(
                    f"\n[{logical_step}] MEASURE "
                    f"ch={ch} score={score:.1f} "
                    f"{reason}"
                )
                if kind == "RECOVERY":
                    record = self.channels[ch]
                    record.recovery_pending.remove(point)
                if kind == "RECOVERY":
                    self.recovery_attempts += 1
                result, _ = self.measure_at(
                    point,
                    ch,
                    mark_lost_on_no_signal=True,
                    reason=reason,
                )
                if (
                    kind == "RECOVERY"
                    and result in {"direction", "near"}
                ):
                    self.recovery_receive_count += 1
            elif kind in {"CLEAR", "SWEEP_CLEAR"}:
                (
                    _,
                    ch,
                    point,
                    reason,
                    measure_after_failure,
                ) = action
                if kind == "SWEEP_CLEAR":
                    self.channels[ch].sweep_pending.remove(point)
                print(
                    f"\n[{logical_step}] CLEAR "
                    f"ch={ch} score={score:.1f} "
                    f"{reason}"
                )
                is_geom_probe = (
                    reason.startswith(" 两线交点 Probe")
                )
                if is_geom_probe:
                    self.geom_probe_attempts += 1
                clear_ok = self.clear_at(
                    point,
                    ch,
                    reason=reason,
                    measure_after_failure=(
                        measure_after_failure
                    ),
                )
                if is_geom_probe and clear_ok:
                    self.geom_probe_success += 1
            else:
                raise RuntimeError (
                    f" 未知动作: {kind}"
                )
            if (
                not self.pending_ew
                and self.search_completed_virtual_time_s is None
            ):
                self.search_completed_virtual_time_s = (
                    self.ledger.total_seconds
                )
            print(
                f" cleared= {self.cleared_count}, "
                f"found={self.found_count}, "
                f"empty={self.empty_count}, "
                f"active={len(self.active_channels())}, "
                f"EW_left={len(self.pending_ew)}, "
                f"T≈{self.ledger.total_seconds:.1f}s"
            )
        else:
            raise RuntimeError (
                " 超过最大逻辑动作数，疑似循环"
            )
        exit_response = self.client.exit()
        if "virtual_time_s" in exit_response:
            self.server_virtual_time_s = float(
                exit_response["virtual_time_s"]
            )
        # 统计到/exit 完成为止。
        wall_runtime_s = (
            time.perf_counter() - wall_start
        )
        summary = {
            "cleared": self.cleared_count,
            "found": self.found_count,
            "certified_empty": self.empty_count,
            "early_stop_search_by_found16": (
                self.early_stop_search_by_found16
            ),
            "geom_probe_attempts": self.geom_probe_attempts,
            "geom_probe_success": self.geom_probe_success,
            "search_certificate_version": CERTIFICATE_VERSION,
            "scheduler_version": SCHEDULER_VERSION,
            "search_visit_count": self.search_visit_count,
            "search_completed_virtual_time_s": (
                self.search_completed_virtual_time_s
            ),
            "opportunistic_measure_count": (
                self.opportunistic_measure_count
            ),
            "opportunistic_receive_count": (
                self.opportunistic_receive_count
            ),
            "recovery_attempts": self.recovery_attempts,
            "recovery_receive_count": self.recovery_receive_count,
            "search_insertion_action_count": (
                self.search_insertion_action_count
            ),
            "search_insertion_detour_m": (
                self.search_insertion_detour_m
            ),
            "search_insertion_future_cost_m": (
                self.search_insertion_future_cost_m
            ),
            "search_insertion_count_by_category": (
                self.search_insertion_count_by_category
            ),
            "future_ew_deferred_state_count": len(
                self.future_ew_deferred_states
            ),
            "post_search_route_plan_count": (
                self.post_search_route_plan_count
            ),
            "safe_clear_shift_count": (
                self.safe_clear_shift_count
            ),
            "safe_clear_shift_saved_m_est": (
                self.safe_clear_shift_saved_m
            ),
            "safe_clear_shift_saved_s_est": (
                self.safe_clear_shift_saved_m / SPEED
            ),
            "safe_clear_segment_hit_count": (
                self.safe_clear_segment_hit_count
            ),
            "first_discovery_virtual_time_s": (
                self.first_discovery_virtual_time_s
            ),
            "move_distance_m": self.ledger.move_distance,
            "movement_seconds": self.ledger.move_seconds,
            "measure_count": self.ledger.measure_count,
            "measure_seconds": self.ledger.measure_seconds,
            "switch_count": self.ledger.switch_count,
            "switch_seconds": self.ledger.switch_seconds,
            "clear_success": self.ledger.clear_success,
            "clear_fail": self.ledger.clear_fail,
            "clear_seconds": self.ledger.clear_seconds,
            "estimated_virtual_time": self.ledger.total_seconds,
            "simulator_virtual_time_s": self.server_virtual_time_s,
            "program_wall_runtime_s": wall_runtime_s,
            "program_wall_runtime_min": (
                wall_runtime_s / 60.0
            ),
            "estimated_time_error_s": (
                self.ledger.total_seconds - self.server_virtual_time_s
                if self.server_virtual_time_s is not None
                else None
            ),
            "action_count_by_category": self.action_count_by_category,
            "move_distance_by_category_m": (
                self.move_distance_by_category_m
            ),
            "average_seconds_per_cleared": (
                self.ledger.total_seconds
                / self.cleared_count
                if self.cleared_count
                else None
            ),
            "search_point_count": SEARCH_POINT_COUNT,
            **validation,
            "exit_reason": exit_response.get(
                "exit_reason"
            ),
        }
        SUMMARY_PATH.write_text(
            json.dumps(
                summary,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        self.trace(
            {
                "action": "exit",
                "exit_reason": exit_response.get(
                    "exit_reason"
                ),
                "summary": summary,
            }
        )
        print("\n" + "=" * 80)
        print("Q4 RESULT")
        print("=" * 80)
        for key, value in summary.items():
            print(f"{key:28s}: {value}")
        print(
            "\n 程序真实运行时间: "
            f"{wall_runtime_s:.3f} s "
            f"({wall_runtime_s / 60.0 :.3f} min)"
        )
        print(
            f"\n 明文轨迹: {TRACE_PATH.resolve()}"
        )
        print(
            f" 明文汇总: {SUMMARY_PATH.resolve()}"
        )
def main() -> None:
    if "--self-test" in sys.argv:
        run_geometry_self_test()
        return
    agent = Q4Agent()
    try:
        agent.run()
    except KeyboardInterrupt :
        print("\n 程序被手动中止。若已经成功进入测试，请在模拟器中确认测试状态。")
        raise SystemExit (130)
    except Exception as exc:
        print("\n" + "=" * 80)
        print(" 第四问程序未能继续执行")
        print("=" * 80)
        print(str(exc))
        print(
            "\n 若错误发生在 /enter：这不是搜索算法错误，而是模拟器接口尚未建立。 \n"
            " 检查顺序：模拟器进程仍在运行 -> 界面显示接口已就绪 -> "
            "BASE_URL 为 127.0.0.1:2026 -> 防火墙/代理未拦截本机端口。"
        )
        raise SystemExit (1)
if __name__ == "__main__":
    main()
