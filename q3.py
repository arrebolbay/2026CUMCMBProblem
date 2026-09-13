#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ============================================================================
# 【学习注释】问题三：全向干扰源的在线搜索与清除（q3.py，约 1760 行）
# ----------------------------------------------------------------------------
# 核心思路：**把"完备性"和"效率"分开处理** ——
#   完备性由可证明的几何/逻辑保证（覆盖骨架 + 认证空频道），
#   效率由在线启发式逼近（统一排序 + 精确 DP + 试探清除）。
#
# 【关键常量对照】（学代码先看这张表，能省很多时间）
#   EW_OUTER_COUNT = 7, EW_RING_RADIUS = 1010.0
#       → 搜索骨架 = 原点 + 半径 1010 m 上 7 个等角点（共 8 点）。
#         取 1010 而非 1000 是算过的：最坏覆盖距离
#         √(1800² + 1010² − 2·1800·1010·cos(π/7)) = 992.057 m < 1000 m，
#         比取 1000 m 时（998.25 m）多出约 6 m 余量。于是
#         "某频道在 8 点全部无信号 ⇒ 该频道必为空"成为严格结论（见 certify_empty）。
#   CLEAR_RADIUS = 20.0 / NEAR_RADIUS = 5.0 / MIN_RECEIVE_RADIUS = 1000.0
#       → 题给物理量：清除半径、near 阈值、有效接收半径下界。
#   STRONG_PROBE_RADIUS_M = 80 / WEAK_PROBE_RADIUS_M = 180
#       → 两类"试探清除"的作用半径（见 opportunistic_score）。
#   GUARANTEED_CLEAR_BONUS_M = 320 / STRONG_PROBE_BONUS_M = 150 / WEAK_PROBE_BONUS_M = 60
#       → 调度时把动作折算成"距离 − 等效奖励"的奖励值，越大越优先做。
#   GEOM_PROBE_MIN_SIN = 0.65 / GEOM_PROBE_MAX_ERROR_M = 24.0
#       → 几何 Probe 门槛：楔形交张角够大、定位误差够小才值得顺手清。
#   MANDATORY_ROUTE_MAX_NODES = 12 / MANDATORY_ROUTE_MIN_REGRET_M = 250.0
#       → 强制任务集最多 12 点，可用 bitmask 精确 DP 求开放最短路径；
#         只有精确解比启发式省 ≥ 250 m 时才覆盖原决策（避免抖动）。
#   DENSE_FOUND_THRESHOLD = 13
#       → 已发现 13 个源即视为"源很密"，放宽机会清除次数上限（4 → 5）。
#
# 【为什么"试探清除"划算】（论文式(24)，很重要的一个判断）
#   清除成功耗 5 s、失败只耗 3 s，差 ΔT。若某处有源的概率为 p，
#   试探一次再决定是否绕过去清，只要 p > 3/(ΔT+3) 就比"直接绕过去清"划算。
#   ΔT 较大时门槛很低（约 6%），所以可以**大胆地顺路试清**。
#
# 主要函数：build_backbone（骨架）/ analytic_cover_radius（992 m 解析证明）/
#          certify_empty（认证空频道）/ _mandatory_open_tsp_first（bitmask DP）/
#          opportunistic_score（统一排序 + 试探清除）/ choose_action（总调度）
# ============================================================================

from __future__ import annotations
import itertools
import json
import math
import random
import socket
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
BASE_URL = "http://127.0.0.1:2026"
ROBOT_ID = "********"
ARENA_ID = "default"
HTTP_TIMEOUT_S = 10
HTTP_RETRIES = 3
TRACE_PATH = Path("q3_v3_trace.jsonl")
SUMMARY_PATH = Path("q3_v3_summary.json")
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
EW_OUTER_COUNT = 7
EW_RING_RADIUS = 1010.0
Q2_FAST_SIDE_M = 80.0
RECOVERY_FORWARD_M = 200.0
RECOVERY_SIDE_M = 350.0
STRONG_PROBE_RADIUS_M = 80.0
WEAK_PROBE_RADIUS_M = 180.0
GUARANTEED_CLEAR_BONUS_M = 320.0
STRONG_PROBE_BONUS_M = 150.0
WEAK_PROBE_BONUS_M = 60.0
EW_PROGRESS_BONUS_M = 15.0
Q2_FAST_INFO_BONUS_M = 45.0
MAX_OPPORTUNISTIC_PER_EW = 4
MIN_OPPORTUNISTIC_BASELINE_M = 220.0
MIN_OPPORTUNISTIC_SIN = 0.20
MAX_OPPORTUNISTIC_CENTER_DISTANCE_M = 1650.0
DENSE_FOUND_THRESHOLD = 13
DENSE_MAX_OPPORTUNISTIC_PER_EW = 5
GEOM_PROBE_MIN_SIN = 0.65
GEOM_PROBE_MAX_ERROR_M = 24.0
GEOM_PROBE_BONUS_M = 170.0
FAILED_CLEAR_AVOID_RADIUS_M = 8.0
MAX_LOGICAL_ACTIONS = 2500
MANDATORY_ROUTE_MAX_NODES = 12
MANDATORY_ROUTE_MIN_REGRET_M = 250.0
Point = tuple[float, float]
def distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])
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
    out: list[Point] = []
    last = polygon[-1]
    last_value = value(last)
    last_inside = last_value >= -eps
    for point in polygon:
        point_value = value(point)
        point_inside = point_value >= -eps
        if last_inside != point_inside:
            span = last_value - point_value
            if abs(span) > 1e-15:
                t = last_value / span
                t = min(1.0, max(0.0, t))
                out.append(
                    (
                        last[0] + t * (point[0] - last[0]),
                        last[1] + t * (point[1] - last[1]),
                    )
                )
        if point_inside:
            out.append(point)
        last = point
        last_value = point_value
        last_inside = point_inside
    trimmed: list[Point] = []
    for p in out:
        if not trimmed or distance(trimmed[-1], p) > 1e-8:
            trimmed.append(p)
    if len(trimmed) > 1 and distance(trimmed[0], trimmed[-1]) <= 1e-8:
        trimmed.pop()
    return trimmed
def circumscribed_disk(
    center: Point,
    radius: float,
    sides: int,
) -> list[Point]:
    outer_radius = radius / math.cos(math.pi / sides)
    phase = math.pi / sides
    return [
        (
            center[0]
            + outer_radius * math.cos(phase + 2.0 * math.pi * i / sides),
            center[1]
            + outer_radius * math.sin(phase + 2.0 * math.pi * i / sides),
        )
        for i in range(sides)
    ]
def clip_by_convex_polygon(
    subject: list[Point],
    clipper: list[Point],
) -> list[Point]:
    out = subject
    for i in range(len(clipper)):
        a = clipper[i]
        b = clipper[(i + 1) % len(clipper)]
        out = clip_left(
            out,
            a,
            (b[0] - a[0], b[1] - a[1]),
        )
        if not out:
            break
    return out
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
    max_sq = -1.0
    farthest = (polygon[0], polygon[1])
    for i, p in enumerate(polygon):
        for q in polygon[i + 1:]:
            sq = (
                (p[0] - q[0]) ** 2
                + (p[1] - q[1]) ** 2
            )
            if sq > max_sq:
                max_sq = sq
                farthest = (p, q)
    return math.sqrt(max_sq), farthest
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
    denom = 2.0 * (
        ax * (by - cy)
        + bx * (cy - ay)
        + cx * (ay - by)
    )
    if abs(denom) <= 1e-12:
        return None
    a2 = ax * ax + ay * ay
    b2 = bx * bx + by * by
    c2 = cx * cx + cy * cy
    ux = (
        a2 * (by - cy)
        + b2 * (cy - ay)
        + c2 * (ay - by)
    ) / denom
    uy = (
        a2 * (cx - bx)
        + b2 * (ax - cx)
        + c2 * (bx - ax)
    ) / denom
    center = (ux, uy)
    return Circle(center, distance(center, a))
def circle_contains(circle: Circle, point: Point) -> bool:
    return distance(circle.center, point) <= circle.radius + 1e-7
def minimum_enclosing_circle(points: list[Point]) -> Circle:
    if not points:
        raise ValueError (" 空点集没有最小包围圆")
    pts = list(dict.fromkeys(points))
    random.Random(20260912).shuffle(pts)
    mec: Optional[Circle] = None
    for i, p in enumerate(pts):
        if mec is not None and circle_contains(mec, p):
            continue
        mec = Circle(p, 0.0)
        for j in range(i):
            q = pts[j]
            if circle_contains(mec, q):
                continue
            mec = circle_from_two(p, q)
            for k in range(j):
                r = pts[k]
                if circle_contains(mec, r):
                    continue
                circle3 = circle_from_three(p, q, r)
                if circle3 is not None :
                    mec = circle3
                else:
                    pairs = [
                        circle_from_two(p, q),
                        circle_from_two(p, r),
                        circle_from_two(q, r),
                    ]
                    valid_pairs = [
                        item
                        for item in pairs
                        if circle_contains(item, p)
                        and circle_contains(item, q)
                        and circle_contains(item, r)
                    ]
                    mec = min(
                        valid_pairs,
                        key=lambda item: item.radius,
                    )
    assert mec is not None
    return Circle(
        mec.center,
        max(distance(mec.center, p) for p in pts),
    )
@dataclass(frozen=True)
class TargetInfo :
    center: Point
    work_radius: float
    mec_center: Point
    mec_radius: float
def target_info_from_polygon(polygon: list[Point]) -> TargetInfo:
    diam, ends = polygon_diameter(polygon)
    center = (
        (ends[0][0] + ends[1][0]) / 2.0,
        (ends[0][1] + ends[1][1]) / 2.0,
    )
    work_radius = diam / 2.0
    enclosing = minimum_enclosing_circle(polygon)
    return TargetInfo(
        center=center,
        work_radius=work_radius,
        mec_center=enclosing.center,
        mec_radius=enclosing.radius,
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
    if not polygon:
        return False
    orientation = 0
    for i in range(len(polygon)):
        a = polygon[i]
        b = polygon[(i + 1) % len(polygon)]
        value = cross(
            (b[0] - a[0], b[1] - a[1]),
            (point[0] - a[0], point[1] - a[1]),
        )
        if abs(value) <= eps:
            continue
        edge_sign = 1 if value > 0 else -1
        if orientation == 0:
            orientation = edge_sign
        elif edge_sign != orientation:
            return False
    return True
def bearing_pair_probe(
    first: "DirectionObservation",
    second: "DirectionObservation",
) -> Optional[GeometricProbe]:
    angle1 = math.radians(first.bearing_deg)
    angle2 = math.radians(second.bearing_deg)
    ray1 = (math.cos(angle1), math.sin(angle1))
    ray2 = (math.cos(angle2), math.sin(angle2))
    denom = cross(ray1, ray2)
    sin_phi = abs(denom)
    if sin_phi < GEOM_PROBE_MIN_SIN:
        return None
    offset = (
        second.station[0] - first.station[0],
        second.station[1] - first.station[1],
    )
    dist1 = cross(offset, ray2) / denom
    dist2 = cross(offset, ray1) / denom
    if dist1 <= 0.0 or dist2 <= 0.0:
        return None
    point = (
        first.station[0] + dist1 * ray1[0],
        first.station[1] + dist1 * ray1[1],
    )
    err_est = (
        math.tan(math.radians(BEARING_ERROR_DEG))
        * (abs(dist1) + abs(dist2))
        / sin_phi
    )
    return GeometricProbe(
        point=point,
        sin_phi=sin_phi,
        error_proxy_m=err_est,
    )
# 【学习注释】构造**搜索骨架**：原点 + 半径 EW_RING_RADIUS(=1010 m) 上
# EW_OUTER_COUNT(=7) 个**等角**点，共 8 点。
# 为什么 7 个环点是最少：要让 360° 边界全覆盖，环半径需 ≥ ~995 m；
# 若只用 6 个环点则需 ≥ ~1123 m，余量被吃光。故 8 点即**最小骨架**。
# 为什么半径取 1010：见 analytic_cover_radius —— 它把最坏覆盖压到 992.057 m。

def build_backbone() -> list[tuple[int, Point]]:
    sites = [(0.0, 0.0)]
    for i in range(EW_OUTER_COUNT):
        theta = 2.0 * math.pi * i / EW_OUTER_COUNT
        sites.append(
            (
                EW_RING_RADIUS * math.cos(theta),
                EW_RING_RADIUS * math.sin(theta),
            )
        )
    return list(enumerate(sites))
# 【学习注释】骨架最坏覆盖距离的**解析式**（问题三完备性的基石）：
#   d_max = √(R² + a² − 2·R·a·cos(π/k))，R = 1800（区域半径）、a = 环半径、k = 环点数。
# 代入 a = 1010、k = 7 ⇒ **992.057 m < 1000 m**（题给最小有效接收半径）；
# 若取 a = 1000 ⇒ 998.25 m，只剩 1.75 m 余量。
# 结论：任何全向源都必被 8 点中至少一个"看到"，于是"8 点全无信号 ⇒ 该频道为空"。

def analytic_cover_radius() -> float:
    half_gap = math.pi / EW_OUTER_COUNT
    center_gap = EW_RING_RADIUS / (2.0 * math.cos(half_gap))
    boundary_gap = math.sqrt(
        TARGET_RADIUS**2
        + EW_RING_RADIUS**2
        - 2.0
        * TARGET_RADIUS
        * EW_RING_RADIUS
        * math.cos(half_gap)
    )
    return max(center_gap, boundary_gap)
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
        body = json.dumps(
            payload,
            ensure_ascii=False,
        ).encode("utf-8")
        last_exc: Optional[ Exception] = None
        for retry_no in range(1, HTTP_RETRIES + 1):
            req = Request(
                BASE_URL + path,
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urlopen(
                    req,
                    timeout=HTTP_TIMEOUT_S,
                ) as resp:
                    return json.loads(
                        resp.read().decode("utf-8")
                    )
            except HTTPError as exc:
                message = ""
                try:
                    message = exc.read().decode(
                        "utf-8",
                        errors="replace",
                    )
                except Exception :
                    pass
                raise RuntimeError (
                    f"HTTP {exc.code}: {message}"
                ) from exc
            except (
                URLError,
                TimeoutError,
                socket.timeout,
                ConnectionError,
            ) as exc:
                last_exc = exc
                print(
                    f"[网络异常] attempt= {retry_no}, "
                    f" 复用同一 request_id: {exc}"
                )
        assert last_exc is not None
        raise last_exc
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
@dataclass
class DirectionObservation :
    station: Point
    bearing_deg: float
@dataclass
class ChannelRecord :
    state: str = "UNKNOWN"
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
class Q3V3Agent :
    def __init__(self) -> None:
        self.client = OfficialClient()
        self.position: Point = (0.0, 0.0)
        self.current_measure_channel = 1
        self.channels = {
            ch: ChannelRecord()
            for ch in range(1, CHANNEL_COUNT + 1)
        }
        self.pending_ew = build_backbone()
        self.ledger = Ledger()
        self.trace_sequence = 0
        self.remaining_real_duration_s: Optional[float] = None
        self.early_stop_search_by_found16 = False
        self.geom_probe_attempts = 0
        self.geom_probe_success = 0
        self.route_guard_overrides = 0
        self.route_guard_regret_saved_m_est = 0.0
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
    def measure_at(
        self,
        position: Point,
        channel: int,
        *,
        ew_index: Optional[int] = None,
        mark_lost_on_no_signal: bool = True,
        reason: str = "",
    ) -> tuple[str, Optional[float]]:
        start = self.position
        resp = self.client.measure(
            position,
            channel,
        )
        if resp.get("accepted") is not True :
            raise RuntimeError (
                f"/measure rejected: {resp}"
            )
        travelled = self.account_move(position)
        channel_changed = (
            self.current_measure_channel
            != channel
        )
        if channel_changed:
            self.ledger.switch_count += 1
            self.ledger.switch_seconds += SWITCH_SECONDS
        self.current_measure_channel = channel
        self.ledger.measure_count += 1
        self.ledger.measure_seconds += MEASURE_SECONDS
        rec = self.channels[channel]
        rec.measured_positions.append(position)
        measure_result = resp["measure_result"]
        svd: Optional[float] = None
        if measure_result == "direction":
            svd = float(resp["svd_deg"])
            rec.state = "LOCALIZING"
            rec.lost = False
            rec.directions.append(
                DirectionObservation(
                    station=position,
                    bearing_deg=svd,
                )
            )
            polygon = update_polygon_with_direction(
                rec.polygon,
                position,
                svd,
            )
            if not polygon:
                raise RuntimeError (
                    f" 频道 {channel} 后验意外为空"
                )
            rec.polygon = polygon
        elif measure_result == "near":
            rec.state = "LOCALIZING"
            rec.lost = False
            rec.strong_position = position
        elif measure_result == "no_signal":
            if (
                rec.state == "UNKNOWN"
                and ew_index is not None
            ):
                rec.no_signal_ew_indices.add(
                    ew_index
                )
            if (
                rec.state == "LOCALIZING"
                and mark_lost_on_no_signal
            ):
                rec.lost = True
        else:
            raise RuntimeError (
                f" 未知 measure_result: {measure_result!r}"
            )
        extra_log = {}
        if (
            rec.state == "LOCALIZING"
            and rec.polygon
        ):
            target = self.target_info(channel)
            extra_log = {
                "work_radius": target.work_radius,
                "mec_radius": target.mec_radius,
                "direction_count": len(
                    rec.directions
                ),
            }
        self.trace(
            {
                "action": "measure",
                "channel": channel,
                "reason": reason,
                "from": {
                    "x": start[0],
                    "y": start[1],
                },
                "to": {
                    "x": position[0],
                    "y": position[1],
                },
                "move_m": travelled,
                "switch": channel_changed,
                "result": measure_result,
                "bearing": svd,
                **extra_log,
            }
        )
        if measure_result == "near":
            self.clear_at(
                position,
                channel,
                reason="near 后原地保证清除",
                measure_after_failure=False,
            )
        self.apply_found16_shortcut()
        return measure_result, svd
    def clear_at(
        self,
        position: Point,
        channel: int,
        *,
        reason: str,
        measure_after_failure: bool,
    ) -> bool:
        start = self.position
        resp = self.client.clear(
            position,
            channel,
        )
        if resp.get("accepted") is not True :
            raise RuntimeError (
                f"/clear rejected: {resp}"
            )
        travelled = self.account_move(position)
        cleared = (
            resp.get("clear_result")
            == "success"
        )
        rec = self.channels[channel]
        if cleared:
            self.ledger.clear_success += 1
            self.ledger.clear_seconds += CLEAR_SUCCESS_SECONDS
            rec.state = "CLEARED"
            rec.lost = False
        else:
            self.ledger.clear_fail += 1
            self.ledger.clear_seconds += CLEAR_FAILURE_SECONDS
            rec.failed_clear_points.append(
                position
            )
        self.trace(
            {
                "action": "clear",
                "channel": channel,
                "reason": reason,
                "from": {
                    "x": start[0],
                    "y": start[1],
                },
                "to": {
                    "x": position[0],
                    "y": position[1],
                },
                "move_m": travelled,
                "result": (
                    "success"
                    if cleared
                    else resp.get(
                        "clear_result",
                        "fail",
                    )
                ),
            }
        )
        if (
            not cleared
            and measure_after_failure
        ):
            self.measure_at(
                position,
                channel,
                mark_lost_on_no_signal=True,
                reason="clear 失败后原地补测",
            )
        self.apply_found16_shortcut()
        return cleared
    def q2_fast_candidates(
        self,
        channel: int,
    ) -> tuple[Point, Point]:
        target = self.target_info(channel)
        obs = self.channels[channel].directions[-1]
        angle = math.radians(
            obs.bearing_deg
        )
        side = (
            -math.sin(angle),
            math.cos(angle),
        )
        center = target.center
        return (
            (
                center[0]
                + Q2_FAST_SIDE_M * side[0],
                center[1]
                + Q2_FAST_SIDE_M * side[1],
            ),
            (
                center[0]
                - Q2_FAST_SIDE_M * side[0],
                center[1]
                - Q2_FAST_SIDE_M * side[1],
            ),
        )
    def q2_fast_information(
        self,
        channel: int,
        point: Point,
    ) -> float:
        target = self.target_info(channel)
        obs = self.channels[channel].directions[-1]
        v1 = (
            target.center[0] - obs.station[0],
            target.center[1] - obs.station[1],
        )
        v2 = (
            target.center[0] - point[0],
            target.center[1] - point[1],
        )
        len1 = math.hypot(v1[0], v1[1])
        len2 = math.hypot(v2[0], v2[1])
        if len1 <= 1e-9 or len2 <= 1e-9:
            return 1.0
        return min(
            1.0,
            abs(cross(v1, v2)) / (len1 * len2),
        )
    # 【学习注释】"恢复候选点"：当某频道已定位但还不够可靠时，
    # 用**前向对 + 侧向对**在附近取补充测量点，既避免路线被拖长，
    # 又能把楔形交的几何改善到可清除的程度。

    def recovery_candidates(
        self,
        channel: int,
    ) -> tuple[Point, Point]:
        obs = self.channels[channel].directions[-1]
        angle = math.radians(
            obs.bearing_deg
        )
        heading = (
            math.cos(angle),
            math.sin(angle),
        )
        side = (
            -math.sin(angle),
            math.cos(angle),
        )
        base = obs.station
        return (
            (
                base[0]
                + RECOVERY_FORWARD_M * heading[0]
                + RECOVERY_SIDE_M * side[0],
                base[1]
                + RECOVERY_FORWARD_M * heading[1]
                + RECOVERY_SIDE_M * side[1],
            ),
            (
                base[0]
                + RECOVERY_FORWARD_M * heading[0]
                - RECOVERY_SIDE_M * side[0],
                base[1]
                + RECOVERY_FORWARD_M * heading[1]
                - RECOVERY_SIDE_M * side[1],
            ),
        )
    # 【学习注释】**"已发现 16 个源"即提早收工**的逻辑：
    # 题目保证源总数 N ∈ [10,16] 且每频道至多一源，故一旦**确认发现 16 个源**，
    # 就不必再扫描剩余频道，可立即转入攻击/清除。这是"高 N 案例"的免费加速。

    def apply_found16_shortcut(self) -> None:
        if self.found_count < SOURCE_MAX:
            return
        updated = False
        for record in self.channels.values():
            if record.state == "UNKNOWN":
                record.state = "CERTIFIED_EMPTY"
                updated = True
        if self.pending_ew:
            self.pending_ew.clear()
            updated = True
        if updated:
            self.early_stop_search_by_found16 = True
            self.trace(
                {
                    "event": "found16_cancel_search",
                    "found_count": self.found_count,
                }
            )
    # 【学习注释】**几何 Probe**：当楔形交的几何形状足够好时，
    # 直接用交会区域内的点试着清除，不必再补测量。门槛是两个无量纲量：
    #   GEOM_PROBE_MIN_SIN = 0.65（两条示向度夹角够大，交会不过于拉长）
    #   GEOM_PROBE_MAX_ERROR_M = 24.0（估计误差有界，略放宽于 20 m 清除半径）
    # 问题四统计中 67.5% 的清除（27/40）就是这样完成的。

    def best_geometric_probe(
        self,
        channel: int,
    ) -> Optional[GeometricProbe]:
        rec = self.channels[channel]
        if len(rec.directions) < 2 or not rec.polygon:
            return None
        best_probe: Optional[GeometricProbe] = None
        for first, second in itertools.combinations(
            rec.directions,
            2,
        ):
            probe = bearing_pair_probe(
                first,
                second,
            )
            if probe is None :
                continue
            if (
                probe.error_proxy_m
                > GEOM_PROBE_MAX_ERROR_M
            ):
                continue
            if not point_in_convex_polygon(
                probe.point,
                rec.polygon,
            ):
                continue
            if (
                probe.point[0] ** 2
                + probe.point[1] ** 2
                > TARGET_RADIUS ** 2 + 1e-6
            ):
                continue
            if (
                best_probe is None
                or probe.error_proxy_m
                < best_probe.error_proxy_m
            ):
                best_probe = probe
        return best_probe
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
    # 【学习注释】**把所有动作折算成"移动距离 − 等效奖励"统一排序**（调度核心）。
    # 奖励项：保证能清 320 m、强 Probe 150 m、弱 Probe 60 m、推进骨架 15 m 等。
    # 为什么敢这么激进地"顺路试清"：清除成功 5 s、失败仅 3 s，成本高度不对称，
    # 由 p > 3/(ΔT+3) 得门槛仅约 6% —— 成功率很低也不亏。
    # 这正是他们比我们少走很多路的直接原因（我们只在绕行 ≤ 400 m 时才顺手清）。

    def opportunistic_score(
        self,
        channel: int,
        ew: Point,
    ) -> Optional[float]:
        rec = self.channels[channel]
        if (
            rec.state != "LOCALIZING"
            or not rec.directions
            or rec.strong_position is not None
        ):
            return None
        if self.already_measured_near(
            channel,
            ew,
        ):
            return None
        if rec.lost:
            return 100.0
        target = self.target_info(channel)
        if (
            target.mec_radius <= CLEAR_RADIUS
            or (
                len(rec.directions) >= 2
                and target.work_radius <= WEAK_PROBE_RADIUS_M
            )
            or self.best_geometric_probe(channel) is not None
        ):
            return None
        if (
            distance(
                ew,
                target.center,
            )
            > MAX_OPPORTUNISTIC_CENTER_DISTANCE_M
        ):
            return None
        obs = rec.directions[-1]
        base_len = distance(
            ew,
            obs.station,
        )
        if base_len < MIN_OPPORTUNISTIC_BASELINE_M:
            return None
        first_vec = (
            target.center[0] - obs.station[0],
            target.center[1] - obs.station[1],
        )
        second_vec = (
            target.center[0] - ew[0],
            target.center[1] - ew[1],
        )
        first_len = math.hypot(first_vec[0], first_vec[1])
        second_len = math.hypot(second_vec[0], second_vec[1])
        if first_len <= 1e-9 or second_len <= 1e-9:
            sin_phi = 1.0
        else:
            sin_phi = min(
                1.0,
                abs(cross(first_vec, second_vec)) / (first_len * second_len),
            )
        if sin_phi < MIN_OPPORTUNISTIC_SIN:
            return None
        return (
            10.0 * sin_phi
            + min(base_len, 1500.0) / 1500.0
        )
    def execute_ew(
        self,
        ew_index: int,
        ew: Point,
    ) -> None:
        unknown_channels = [
            ch
            for ch, record in self.channels.items()
            if record.state == "UNKNOWN"
        ]
        ranked: list[tuple[float, int]] = []
        for ch in self.active_channels():
            score = self.opportunistic_score(
                ch,
                ew,
            )
            if score is not None :
                ranked.append(
                    (score, ch)
                )
        ranked.sort(
            reverse=True
        )
        limit = (
            DENSE_MAX_OPPORTUNISTIC_PER_EW
            if self.found_count >= DENSE_FOUND_THRESHOLD
            else MAX_OPPORTUNISTIC_PER_EW
        )
        extra_channels = [
            ch
            for _, ch
            in ranked[
                :limit
            ]
        ]
        channels_to_measure = list(
            dict.fromkeys(
                unknown_channels
                + extra_channels
            )
        )
        if self.current_measure_channel in channels_to_measure:
            channels_to_measure.remove(
                self.current_measure_channel
            )
            channels_to_measure.insert(
                0,
                self.current_measure_channel,
            )
        if not channels_to_measure:
            return
        for ch in channels_to_measure:
            is_new = (
                self.channels[ch].state
                == "UNKNOWN"
            )
            self.measure_at(
                ew,
                ch,
                ew_index=(
                    ew_index
                    if is_new
                    else None
                ),
                mark_lost_on_no_signal=is_new,
                reason=(
                    "EW 存在性扫描"
                    if is_new
                    else "EW 零移动成本顺路定位"
                ),
            )
            if self.cleared_count >= SOURCE_MAX:
                return
    # 【学习注释】**认证空频道**：某频道在全部 8 个骨架点都 no_signal 时，
    # 依上面的覆盖证明可判定该频道**不存在干扰源**。
    # 于是"清除数 + 认证空频道数 = 20（频道总数）"成为**可自检的完备性证书**，
    # 这也是他们三次正式测试都敢声称清除比例 100% 的底气。
    # 对比：我们只在"有示向度却清不掉"时记录 unresolved，对"从未被发现"的源无感知。

    def certify_empty(self) -> None:
        if self.pending_ew:
            return
        all_ew = set(
            range(EW_OUTER_COUNT + 1)
        )
        for record in self.channels.values():
            if (
                record.state == "UNKNOWN"
                and all_ew.issubset(
                    record.no_signal_ew_indices
                )
            ):
                record.state = "CERTIFIED_EMPTY"
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
    @staticmethod
    def mandatory_action_id(action):
        kind = action[0]
        if kind == "EW":
            return ("EW", int(action[1]))
        if kind == "CLEAR":
            _, channel, _, reason, _ = action
            if (
                reason == "MEC<=20m 保证清除"
                or reason == "near 保证清除"
            ):
                return ("CLEAR", int(channel))
        return None
    @staticmethod
    def mandatory_action_point(action) -> Point:
        kind = action[0]
        if kind == "EW":
            return action[2]
        if kind == "CLEAR":
            return action[2]
        raise ValueError (" 非强制动作没有 mandatory point")
    # 【学习注释】**bitmask 动态规划求精确开放最短 Hamilton 路径**。
    # 只用于"强制任务集"，且 MANDATORY_ROUTE_MAX_NODES = 12 限制了规模，
    # 使 O(2ⁿ·n²) 的 DP 轻松可承受。
    # 相比启发式（最近邻 + 2-opt）：点数 ≤12 时精确解更省心；
    # 更大规模才退回启发式 —— 该精确的地方精确、该启发的地方启发。

    def _mandatory_open_tsp_first(
        self,
        nodes: list[tuple[tuple[str, int], Point]],
    ) -> tuple[
        Optional[tuple[str, int]],
        float,
        dict[tuple[str, int], float],
    ]:
        n = len(nodes)
        if n == 0:
            return None , 0.0, {}
        points = [point for _, point in nodes]
        pair_dist = [
            [
                distance(points[i], points[j])
                for j in range(n)
            ]
            for i in range(n)
        ]
        start_dist = [
            distance(self.position, point)
            for point in points
        ]
        memo: dict[tuple[int, int], float] = {}
        def tail_cost(
            last: int,
            remaining_mask: int,
        ) -> float:
            if remaining_mask == 0:
                return 0.0
            key = (last, remaining_mask)
            cached = memo.get(key)
            if cached is not None :
                return cached
            best = float("inf")
            mask = remaining_mask
            while mask:
                bit = mask & -mask
                nxt = bit.bit_length() - 1
                candidate = (
                    pair_dist[last][nxt]
                    + tail_cost(
                        nxt,
                        remaining_mask ^ bit,
                    )
                )
                if candidate < best:
                    best = candidate
                mask ^= bit
            memo[key] = best
            return best
        # 状态压缩 DP
        full_mask = (1 << n) - 1
        forced_first_cost: dict[
            tuple[str, int],
            float,
        ] = {}
        best_id: Optional[tuple[str, int]] = None
        best_cost = float("inf")
        for first in range(n):
            remaining = full_mask ^ (1 << first)
            total = (
                start_dist[first]
                + tail_cost(
                    first,
                    remaining,
                )
            )
            node_id = nodes[first][0]
            forced_first_cost[node_id] = total
            if total < best_cost:
                best_cost = total
                best_id = node_id
        return (
            best_id,
            best_cost,
            forced_first_cost,
        )
    # 【学习注释】"精确 DP 结果要不要采用"的**门控**：
    # 只有精确路径比当前方案省 MANDATORY_ROUTE_MIN_REGRET_M(=250 m) 以上，
    # 才覆盖原决策。作用：① 避免为几米收益频繁改路线（抖动反而更慢）；
    # ② 保证"同一前状态 → 同一决策"的可复现性。

    def apply_mandatory_route_guard(
        self,
        selected,
        candidates,
    ):
        if selected is None :
            return None
        selected_score, selected_action = selected
        selected_id = self.mandatory_action_id(
            selected_action
        )
        if selected_id is None :
            return selected
        if self.found_count >= 15:
            return selected
        mandatory_candidates = []
        seen_ids = set()
        for item in candidates:
            _, action = item
            node_id = self.mandatory_action_id(
                action
            )
            if (
                node_id is None
                or node_id in seen_ids
            ):
                continue
            seen_ids.add(node_id)
            mandatory_candidates.append(
                (
                    node_id,
                    self.mandatory_action_point(
                        action
                    ),
                    item,
                )
            )
        if (
            len(mandatory_candidates) <= 1
            or len(mandatory_candidates)
            > MANDATORY_ROUTE_MAX_NODES
        ):
            return selected
        nodes = [
            (node_id, point)
            for node_id, point, _
            in mandatory_candidates
        ]
        (
            best_first_id,
            best_total_m,
            forced_first_cost,
        ) = self._mandatory_open_tsp_first(
            nodes
        )
        if (
            best_first_id is None
            or best_first_id == selected_id
        ):
            return selected
        selected_total_m = forced_first_cost.get(
            selected_id
        )
        if selected_total_m is None :
            return selected
        regret_m = (
            selected_total_m
            - best_total_m
        )
        if (
            regret_m
            < MANDATORY_ROUTE_MIN_REGRET_M
        ):
            return selected
        replacement = next(
            (
                item
                for node_id, _, item
                in mandatory_candidates
                if node_id == best_first_id
            ),
            None,
        )
        if replacement is None :
            return selected
        self.route_guard_overrides += 1
        self.route_guard_regret_saved_m_est += (
            regret_m
        )
        self.trace(
            {
                "event": "mandatory_route_guard_override",
                "from_task": list(selected_id),
                "to_task": list(best_first_id),
                "route_regret_m": regret_m,
                "route_regret_seconds_equiv": (
                    regret_m / SPEED
                ),
                "mandatory_node_count": len(
                    mandatory_candidates
                ),
            }
        )
        return replacement
    # 【学习注释】总调度：每一步在"去扫描点 / 去试探清除 / 去保证清除 /
    # 修方向后验"等候选动作中选一个，判据就是 opportunistic_score 的
    # "距离 − 等效奖励"最小者（等价于单位时间收益最大者）。

    def choose_action(self):
        pool = []
        for ew_index, ew in self.pending_ew:
            pool.append(
                (
                    distance(
                        self.position,
                        ew,
                    )
                    - EW_PROGRESS_BONUS_M,
                    (
                        "EW",
                        ew_index,
                        ew,
                        " 完备搜索",
                    ),
                )
            )
        for ch in self.active_channels():
            rec = self.channels[ch]
            if rec.strong_position is not None :
                pool.append(
                    (
                        distance(
                            self.position,
                            rec.strong_position,
                        )
                        - GUARANTEED_CLEAR_BONUS_M,
                        (
                            "CLEAR",
                            ch,
                            rec.strong_position,
                            "near 保证清除",
                            False,
                        ),
                    )
                )
                continue
            target = self.target_info(ch)
            if rec.lost:
                if self.pending_ew:
                    continue
                recovery_points = self.recovery_candidates(ch)
                recovery_point = min(
                    recovery_points,
                    key=lambda p: distance(
                        self.position,
                        p,
                    ),
                )
                pool.append(
                    (
                        distance(
                            self.position,
                            recovery_point,
                        ),
                        (
                            "MEASURE",
                            ch,
                            recovery_point,
                            " 最后 EW 后的保证接收恢复点",
                        ),
                    )
                )
                continue
            if (
                target.mec_radius
                <= CLEAR_RADIUS - 1e-6
                and not self.too_close_to_failed_clear(
                    ch,
                    target.mec_center,
                )
            ):
                pool.append(
                    (
                        distance(
                            self.position,
                            target.mec_center,
                        )
                        - GUARANTEED_CLEAR_BONUS_M,
                        (
                            "CLEAR",
                            ch,
                            target.mec_center,
                            "MEC<=20m 保证清除",
                            False,
                        ),
                    )
                )
                continue
            direction_count = len(
                rec.directions
            )
            probe = self.best_geometric_probe(ch)
            if (
                probe is not None
                and not self.too_close_to_failed_clear(
                    ch,
                    probe.point,
                )
            ):
                pool.append(
                    (
                        distance(
                            self.position,
                            probe.point,
                        )
                        - GEOM_PROBE_BONUS_M,
                        (
                            "CLEAR",
                            ch,
                            probe.point,
                            (
                                " 两线交点 Probe "
                                f"err≈{probe.error_proxy_m:.1f}m, "
                                f"sin={probe.sin_phi:.2f}"
                            ),
                            True,
                        ),
                    )
                )
            if (
                direction_count >= 2
                and target.work_radius
                <= STRONG_PROBE_RADIUS_M
                and not self.too_close_to_failed_clear(
                    ch,
                    target.center,
                )
            ):
                pool.append(
                    (
                        distance(
                            self.position,
                            target.center,
                        )
                        - STRONG_PROBE_BONUS_M,
                        (
                            "CLEAR",
                            ch,
                            target.center,
                            (
                                " 强 Probe Clear "
                                f"workR={target.work_radius:.1f}"
                            ),
                            True,
                        ),
                    )
                )
                continue
            if (
                direction_count >= 2
                and target.work_radius
                <= WEAK_PROBE_RADIUS_M
                and not self.too_close_to_failed_clear(
                    ch,
                    target.center,
                )
            ):
                pool.append(
                    (
                        distance(
                            self.position,
                            target.center,
                        )
                        - WEAK_PROBE_BONUS_M,
                        (
                            "CLEAR",
                            ch,
                            target.center,
                            (
                                " 弱 Probe Clear "
                                f"workR={target.work_radius:.1f}"
                            ),
                            True,
                        ),
                    )
                )
            if direction_count == 1:
                for point in self.q2_fast_candidates(ch):
                    gain = self.q2_fast_information(
                        ch,
                        point,
                    )
                    pool.append(
                        (
                            distance(
                                self.position,
                                point,
                            )
                            - Q2_FAST_INFO_BONUS_M
                            * gain,
                            (
                                "MEASURE",
                                ch,
                                point,
                                (
                                    "Q2-fast 第二测点 "
                                    f"sin≈{gain:.2f}"
                                ),
                            ),
                        )
                    )
            pool.append(
                (
                    distance(
                        self.position,
                        target.center,
                    ),
                    (
                        "MEASURE",
                        ch,
                        target.center,
                        (
                            " 直接逼近后验中心 "
                            f"workR={target.work_radius:.1f}, "
                            f"mecR={target.mec_radius:.1f}"
                        ),
                    ),
                )
            )
        if not pool:
            return None
        chosen = min(
            pool,
            key=lambda item: item[0],
        )
        return self.apply_mandatory_route_guard(
            chosen,
            pool,
        )
    def run(self) -> None:
        for path in (TRACE_PATH, SUMMARY_PATH):
            try:
                path.unlink()
            except FileNotFoundError :
                pass
        cover = analytic_cover_radius()
        print("=" * 80)
        print("Q3 V3.2 FINAL - MANDATORY ROUTE GUARD")
        print(
            f"8 点连续覆盖半径 = "
            f"{cover:.3f} m"
        )
        print("=" * 80)
        if cover > MIN_RECEIVE_RADIUS:
            raise RuntimeError (
                " 当前骨架不满足 1000m 完备覆盖"
            )
        enter = self.client.enter()
        if enter.get("accepted") is not True :
            raise RuntimeError (
                f"/enter rejected: {enter}"
            )
        self.remaining_real_duration_s = float(
            enter["remaining_real_duration_s"]
        )
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
                self.execute_ew(
                    ew_index,
                    ew,
                )
            elif kind == "MEASURE":
                _, ch, point, reason = action
                print(
                    f"\n[{logical_step}] MEASURE "
                    f"ch={ch} score={score:.1f} "
                    f"{reason}"
                )
                self.measure_at(
                    point,
                    ch,
                    mark_lost_on_no_signal=True,
                    reason=reason,
                )
            elif kind == "CLEAR":
                (
                    _,
                    ch,
                    point,
                    reason,
                    measure_after_failure,
                ) = action
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
        summary = {
            "cleared": self.cleared_count,
            "found": self.found_count,
            "certified_empty": self.empty_count,
            "early_stop_search_by_found16": (
                self.early_stop_search_by_found16
            ),
            "geom_probe_attempts": self.geom_probe_attempts,
            "geom_probe_success": self.geom_probe_success,
            "route_guard_overrides": self.route_guard_overrides,
            "route_guard_regret_saved_m_est": (
                self.route_guard_regret_saved_m_est
            ),
            "route_guard_regret_saved_s_est": (
                self.route_guard_regret_saved_m_est
                / SPEED
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
            "average_seconds_per_cleared": (
                self.ledger.total_seconds
                / self.cleared_count
                if self.cleared_count
                else None
            ),
            "backbone_cover_radius": cover,
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
        print("Q3 V3.2 FINAL RESULT")
        print("=" * 80)
        for key, value in summary.items():
            print(f"{key:28s}: {value}")
        print(
            f"\n 明文轨迹: {TRACE_PATH.resolve()}"
        )
        print(
            f" 明文汇总: {SUMMARY_PATH.resolve()}"
        )
def main() -> None:
    agent = Q3V3Agent()
    agent.run()
if __name__ == "__main__":
    main()
