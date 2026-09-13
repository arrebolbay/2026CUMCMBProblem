# ============================================================================
# 【学习注释】问题二：第二检测点的稳健选址（q2.py，约 1790 行）
# ----------------------------------------------------------------------------
# 难点在于"目标函数依赖未知量"：一次测向只给方向、不给距离，
# 而定位直径 D 同时依赖交会角 γ 与两个距离 r1、r2。
# 他们的处理（很值得学）：
#   1) 把源位置的不确定性写成**先验可行集 U**——一条长约 1480 m、
#      末端半宽仅 26 m 的极细扇形（这就是集员估计的"保证集"）；
#   2) 目标改为**最坏源位置下的定位直径 J_rob**，约束是"U 中每一点都可被保证接收"；
#   3) 于是问题变成**可靠性约束下的稳健优化**，用三重几何约束交出候选区域：
#      保证接收透镜、领结形交会禁区、稳健直径的近优水平集；
#   4) 结论：约束由**近端**源决定，目标由**远端**源决定，最优点落在近端接收圆边界弧上。
# 主要函数：build_first_feasible_polygon（先验可行集）/ generate_candidate_points /
#          _compute_raw_metrics（J_rob）/ solve_second_station（主入口）
# ============================================================================

"""B 题问题 2：第二检测点选择与候选区域。"""
from __future__ import annotations
import argparse
from dataclasses import dataclass, replace
from math import atan2, cos, degrees, isfinite, pi, radians, sin
from pathlib import Path
from typing import Sequence
import numpy as np
from scipy.spatial import ConvexHull, distance
from q1 import clip_polygon_by_left_halfplane, polygon_diameter
from q2_decision import (
    TradeoffEnvelopePoint,
    make_final_decision,
    pareto_mask as _pareto_mask,
    select_validation_indices,
)
from q2_evaluation import calculate_weighted_scores, evaluate_candidate_score
from q2_evidence import (
    plot_reliability_tradeoff,
    plot_result,
    write_candidate_csv,
    write_reliability_tradeoff_csv,
)
Point = tuple[float, float]
@dataclass(frozen=True)
class Q2Config :
    target_radius: float = 1800.0
    min_receive_radius: float = 1000.0
    max_receive_radius: float = 1500.0
    bearing_error_deg: float = 1.0
    too_strong_radius: float = 5.0
    source_radial_samples: int = 180
    source_angular_samples: int = 41
    candidate_axis_samples: int = 55
    disk_sides: int = 144
    validated_candidate_count: int = 24
    bearing_error_samples: int = 5
    reliability_loss_levels: tuple[float, ...] = (
        0.0,
        0.005,
        0.01,
        0.02,
        0.03,
        0.05,
    )
    perform_numerical_calibration: bool = True
    numerical_calibration_factor: int = 2
    tradeoff_improvement_tolerance_ratio: float = 0.01
    local_refinement_axis_samples: int = 3
    local_refinement_passes: int = 4
    max_refinement_centers: int = 4
    position_convergence_tolerance: float = 10.0
    diameter_convergence_tolerance_ratio: float = 0.01
    score_tolerance: float = 0.03
    diameter_tolerance_ratio: float = 0.03
    weight_not_safe: float = 0.24
    weight_certain_failure: float = 0.31
    weight_diameter: float = 0.35
    weight_angle: float = 0.10
    def validate(self) -> None:
        positive_values = {
            "target_radius": self.target_radius,
            "min_receive_radius": self.min_receive_radius,
            "max_receive_radius": self.max_receive_radius,
            "bearing_error_deg": self.bearing_error_deg,
            "too_strong_radius": self.too_strong_radius,
            "score_tolerance": self.score_tolerance,
            "diameter_tolerance_ratio": self.diameter_tolerance_ratio,
            "position_convergence_tolerance": self.position_convergence_tolerance,
        }
        for name, value in positive_values.items():
            if not isfinite(value) or value <= 0.0:
                raise ValueError (f"{name} 必须为有限正数")
        if self.min_receive_radius > self.max_receive_radius:
            raise ValueError ("min_receive_radius 不能大于 max_receive_radius")
        if not 0.0 < self.bearing_error_deg < 90.0:
            raise ValueError ("bearing_error_deg 必须位于 (0, 90) 度")
        if (
            not isfinite(self.tradeoff_improvement_tolerance_ratio)
            or self.tradeoff_improvement_tolerance_ratio < 0.0
        ):
            raise ValueError ("tradeoff_improvement_tolerance_ratio 必须为有限非负数")
        if not isinstance(self.perform_numerical_calibration, bool):
            raise ValueError ("perform_numerical_calibration 必须为布尔值")
        if (
            not isinstance(self.numerical_calibration_factor, int)
            or self.numerical_calibration_factor < 2
        ):
            raise ValueError ("numerical_calibration_factor 必须为不小于 2 的整数")
        if (
            not isfinite(self.diameter_convergence_tolerance_ratio)
            or self.diameter_convergence_tolerance_ratio < 0.0
        ):
            raise ValueError ("diameter_convergence_tolerance_ratio 必须为有限非负数")
        loss_levels = tuple(sorted({float(x) for x in self.reliability_loss_levels}))
        if not loss_levels or abs(loss_levels[0]) > 1e-12:
            raise ValueError ("reliability_loss_levels 必须从 0 开始")
        if any(not isfinite(x) or not 0.0 <= x < 1.0 for x in loss_levels):
            raise ValueError ("reliability_loss_levels 必须位于 [0, 1)")
        integer_values = {
            "source_radial_samples": self.source_radial_samples,
            "source_angular_samples": self.source_angular_samples,
            "candidate_axis_samples": self.candidate_axis_samples,
            "disk_sides": self.disk_sides,
            "validated_candidate_count": self.validated_candidate_count,
            "bearing_error_samples": self.bearing_error_samples,
            "local_refinement_axis_samples": self.local_refinement_axis_samples,
        }
        for name, value in integer_values.items():
            if not isinstance(value, int) or value < 3:
                raise ValueError (f"{name} 必须为不小于 3 的整数")
        if (
            not isinstance(self.local_refinement_passes, int)
            or self.local_refinement_passes < 0
        ):
            raise ValueError ("local_refinement_passes 必须为非负整数")
        if (
            not isinstance(self.max_refinement_centers, int)
            or self.max_refinement_centers <= 0
        ):
            raise ValueError ("max_refinement_centers 必须为正整数")
        weights = np.array(
            [
                self.weight_not_safe,
                self.weight_certain_failure,
                self.weight_diameter,
                self.weight_angle,
            ],
            dtype=float,
        )
        if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError (" 评分权重必须为有限非负数")
        if not np.isclose(float(weights.sum()), 1.0, atol=1e-12):
            raise ValueError (" 四项评分权重之和必须等于 1")
@dataclass(frozen=True)
class CandidateMetrics :
    point: Point
    local_u: float
    local_v: float
    safe_coverage: float
    uncertain_coverage: float
    certain_failure: float
    angle_penalty: float
    proxy_diameter: float
    proxy_score: float
    pareto: bool
    refinement_level: int = 0
    validated_bearing_worst: float | None = None
    strong_signal_diameter: float | None = None
    no_signal_diameter: float | None = None
    validated_robust_diameter: float | None = None
    validated_score: float | None = None
    validated_pareto: bool = False
@dataclass(frozen=True)
class NumericalStabilityRecord :
    role: str
    point: Point
    coarse_robust_diameter: float
    default_robust_diameter: float
    fine_robust_diameter: float
    default_fine_relative_difference: float
@dataclass(frozen=True)
class Q2Result :
    first_station: Point
    first_bearing_deg: float
    first_region_polygon: tuple[Point, ...]
    first_region_diameter: float
    source_samples: np.ndarray
    candidates: tuple[CandidateMetrics, ...]
    validated_candidates: tuple[CandidateMetrics, ...]
    primary_safe_coverage: float
    selected_reliability_loss: float
    accepted_safe_coverage_floor: float
    tradeoff_improvement_tolerance: float
    reliability_tradeoff: tuple[TradeoffEnvelopePoint, ...]
    reliability_feasible_candidates: tuple[CandidateMetrics, ...]
    refinement_converged: bool
    refinement_resolution: float
    refinement_position_change: float
    refinement_diameter_relative_change: float
    numerical_calibration_applied: bool
    numerical_error_ratio: float
    numerical_stability_records: tuple[NumericalStabilityRecord, ...]
    best_candidate: CandidateMetrics
    near_optimal_candidates: tuple[CandidateMetrics, ...]
    safe_near_optimal_candidates: tuple[CandidateMetrics, ...]
    pareto_best_candidate: CandidateMetrics
    pareto_near_optimal_candidates: tuple[CandidateMetrics, ...]
    config: Q2Config
def _validate_station_and_bearing(station: Point, bearing_deg: float) -> Point:
    clean_station = float(station[0]), float(station[1])
    values = (*clean_station, float(bearing_deg))
    if any(not isfinite(value) for value in values):
        raise ValueError (" 检测点坐标和示向度必须为有限数")
    return clean_station
def _directions(bearing_deg: float) -> tuple[np.ndarray, np.ndarray]:
    angle = radians(bearing_deg)
    forward = np.array([cos(angle), sin(angle)], dtype=float)
    normal = np.array([-sin(angle), cos(angle)], dtype=float)
    return forward, normal
def _regular_circle_polygon(
    center: Point, radius: float, sides: int, circumscribed: bool = True
) -> list[Point]:
    r_poly = radius / cos(pi / sides) if circumscribed else radius
    return [
        (
            center[0] + r_poly * cos(2.0 * pi * i / sides),
            center[1] + r_poly * sin(2.0 * pi * i / sides),
        )
        for i in range(sides)
    ]
def _bearing_halfplanes(
    station: Point, bearing_deg: float, error_deg: float
) -> tuple[tuple[Point, Point], tuple[Point, Point]]:
    low = radians(bearing_deg - error_deg)
    high = radians(bearing_deg + error_deg)
    d_low = cos(low), sin(low)
    d_high = cos(high), sin(high)
    return (
        (station, d_low),
        (station, (-d_high[0], -d_high[1])),
    )
def _clip_by_bearing(
    polygon: Sequence[Point],
    station: Point,
    bearing_deg: float,
    error_deg: float,
    eps: float = 1e-8,
) -> list[Point]:
    out = list(polygon)
    for p0, direction in _bearing_halfplanes(
        station, bearing_deg, error_deg
    ):
        out = clip_polygon_by_left_halfplane(
            out, p0, direction, eps
        )
        if not out:
            break
    return out
def _clip_by_disk(
    polygon: Sequence[Point],
    center: Point,
    radius: float,
    sides: int,
    eps: float = 1e-8,
) -> list[Point]:
    out = list(polygon)
    disk = _regular_circle_polygon(center, radius, sides, True)
    for i, a in enumerate(disk):
        b = disk[(i + 1) % len(disk)]
        edge = b[0] - a[0], b[1] - a[1]
        edge_len = float(np.hypot(*edge))
        unit = edge[0] / edge_len, edge[1] / edge_len
        out = clip_polygon_by_left_halfplane(
            out, a, unit, eps
        )
        if not out:
            break
    return out
# 【学习注释】构造**先验可行集 U**：只凭一条示向度，源必然落在
# "以 S1 为顶点、张角 2° 的角域"与"半径 1000~1500 m 圆环"的交内 ——
# 这是一个长约 1480 m、末端半宽仅 26 m 的**极细扇形**。
# 它就是集员估计里的"保证集"，后续所有最坏情形分析都基于它。

def build_first_feasible_polygon(
    first_station: Point,
    first_bearing_deg: float,
    config: Q2Config,
) -> list[Point]:
    polygon = _regular_circle_polygon(
        (0.0, 0.0), config.target_radius, config.disk_sides, True
    )
    polygon = _clip_by_bearing(
        polygon,
        first_station,
        first_bearing_deg,
        config.bearing_error_deg,
    )
    polygon = _clip_by_disk(
        polygon,
        first_station,
        config.max_receive_radius,
        config.disk_sides,
    )
    if not polygon:
        raise ValueError (" 给定 S1 和 theta1 在目标圆域内没有可行干扰源位置")
    return polygon
def sample_first_feasible_region(
    first_station: Point,
    first_bearing_deg: float,
    config: Q2Config,
) -> np.ndarray:
    radial_t = (
        np.arange(config.source_radial_samples, dtype=float) + 0.5
    ) / config.source_radial_samples
    r_squared = config.too_strong_radius**2 + radial_t * (
        config.max_receive_radius**2 - config.too_strong_radius**2
    )
    radii = np.sqrt(r_squared)
    angle_delta = np.linspace(
        -config.bearing_error_deg,
        config.bearing_error_deg,
        config.source_angular_samples,
    )
    angles = np.deg2rad(first_bearing_deg + angle_delta)
    r_grid, a_grid = np.meshgrid(radii, angles, indexing="ij")
    sample_grid = np.column_stack(
        [
            first_station[0] + (r_grid * np.cos(a_grid)).ravel(),
            first_station[1] + (r_grid * np.sin(a_grid)).ravel(),
        ]
    )
    inside = np.einsum("ij,ij->i", sample_grid, sample_grid) <= (
        config.target_radius**2 + 1e-7
    )
    points = sample_grid[inside]
    s1 = np.asarray(first_station, dtype=float)
    boundary: list[np.ndarray] = []
    for angle in angles:
        ray = np.array([np.cos(angle), np.sin(angle)], dtype=float)
        span = _intersect_interval(
            (config.too_strong_radius, config.max_receive_radius),
            _ray_disk_interval(
                s1,
                ray,
                np.zeros(2, dtype=float),
                config.target_radius,
            ),
        )
        if span is None :
            continue
        edge_radii = [
            max(span[0], config.too_strong_radius + 1e-7),
            span[1],
        ]
        if span[0] <= config.min_receive_radius <= span[1]:
            edge_radii.append(config.min_receive_radius)
        for r in edge_radii:
            boundary.append(s1 + r * ray)
    if boundary:
        points = np.vstack([points, np.asarray(boundary)])
        points = np.unique(np.round(points, decimals=10), axis=0)
    if len(points) < 20:
        raise ValueError (
            " 第一次可行区域的有效样本过少，请提高 source_radial_samples 和 "
            "source_angular_samples"
        )
    return points
def _ray_disk_interval(
    origin: np.ndarray,
    direction: np.ndarray,
    center: np.ndarray,
    radius: float,
    eps: float = 1e-12,
) -> tuple[float, float] | None:
    delta = origin - center
    proj = float(delta @ direction)
    disc = proj**2 - (float(delta @ delta) - radius**2)
    if disc < -eps:
        return None
    root = float(np.sqrt(max(0.0, disc)))
    low = -proj - root
    high = -proj + root
    if high < -eps:
        return None
    return max(0.0, low), max(0.0, high)
def _intersect_interval(
    first: tuple[float, float],
    second: tuple[float, float] | None,
) -> tuple[float, float] | None:
    if second is None :
        return None
    low = max(first[0], second[0])
    high = min(first[1], second[1])
    if high <= low + 1e-12:
        return None
    return low, high
def _radial_area(interval: tuple[float, float] | None) -> float:
    if interval is None :
        return 0.0
    return 0.5 * max(0.0, interval[1] ** 2 - interval[0] ** 2)
def _coverage_angular_quadrature(
    first_station: Point,
    first_bearing_deg: float,
    config: Q2Config,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xg, wg = np.polynomial.legendre.leggauss(
        config.source_angular_samples
    )
    mid = radians(first_bearing_deg)
    half = radians(config.bearing_error_deg)
    angles = mid + half * xg
    quad_w = half * wg
    rays = np.column_stack([np.cos(angles), np.sin(angles)])
    s1 = np.asarray(first_station, dtype=float)
    origin = np.zeros(2, dtype=float)
    low = np.empty(len(rays), dtype=float)
    high = np.empty(len(rays), dtype=float)
    for i, ray in enumerate(rays):
        hit = _ray_disk_interval(
            s1,
            ray,
            origin,
            config.target_radius,
        )
        span = _intersect_interval(
            (config.too_strong_radius, config.max_receive_radius),
            hit,
        )
        if span is None :
            low[i] = 0.0
            high[i] = 0.0
        else:
            low[i], high[i] = span
    return rays, quad_w, low, high
def _continuous_coverages_for_candidate(
    second_station: np.ndarray,
    first_station: np.ndarray,
    directions: np.ndarray,
    angular_weights: np.ndarray,
    lower_radius: np.ndarray,
    upper_radius: np.ndarray,
    config: Q2Config,
) -> tuple[float, float, float]:
    total = 0.0
    safe_sum = 0.0
    fail_sum = 0.0
    delta = first_station - second_station
    delta2 = float(delta @ delta)
    for ray, w, low, high in zip(
        directions,
        angular_weights,
        lower_radius,
        upper_radius,
    ):
        span = float(low), float(high)
        area = _radial_area(span)
        if area <= 0.0:
            continue
        total += float(w) * area
        near_span = (
            float(low), min(float(high), config.min_receive_radius)
        )
        near_safe = None
        if near_span[1] > near_span[0] + 1e-12:
            near_safe = _intersect_interval(
                near_span,
                _ray_disk_interval(
                    first_station,
                    ray,
                    second_station,
                    config.min_receive_radius,
                ),
            )
        far_low = max(float(low), config.min_receive_radius)
        far_high = float(high)
        far_safe: tuple[float, float] | None = None
        if far_high > far_low + 1e-12:
            proj = float(ray @ delta)
            if delta2 <= 1e-18:
                far_safe = far_low, far_high
            elif proj < -1e-15:
                cut = -delta2 / (2.0 * proj)
                start = max(far_low, cut)
                if far_high > start + 1e-12:
                    far_safe = start, far_high
        safe_sum += float(w) * (
            _radial_area(near_safe) + _radial_area(far_safe)
        )
        received = _intersect_interval(
            span,
            _ray_disk_interval(
                first_station,
                ray,
                second_station,
                config.max_receive_radius,
            ),
        )
        fail_sum += float(w) * (
            area - _radial_area(received)
        )
    if total <= 0.0:
        raise ValueError (" 第一次示向可行域面积为零")
    safe = float(np.clip(safe_sum / total, 0.0, 1.0))
    fail = float(np.clip(fail_sum / total, 0.0, 1.0))
    uncertain = float(np.clip(1.0 - safe - fail, 0.0, 1.0))
    uncertain = 1.0 - safe - fail
    if uncertain < -1e-10:
        raise RuntimeError (" 连续覆盖积分得到相互重叠的结果分类")
    return safe, max(0.0, uncertain), fail
def _point_set_diameter(points: np.ndarray) -> float:
    if len(points) <= 1:
        return 0.0
    if len(points) == 2:
        return float(np.linalg.norm(points[0] - points[1]))
    try:
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]
    except Exception :
        hull_points = points
    if len(hull_points) <= 1:
        return 0.0
    return float(np.max(distance.pdist(hull_points)))
def _point_in_convex_polygon(
    point: np.ndarray, polygon: Sequence[Point], eps: float = 1e-8
) -> bool:
    if not polygon:
        return False
    for index, first in enumerate(polygon):
        second = polygon[(index + 1) % len(polygon)]
        edge_x = second[0] - first[0]
        edge_y = second[1] - first[1]
        cross = edge_x * (point[1] - first[1]) - edge_y * (
            point[0] - first[0]
        )
        if cross < -eps:
            return False
    return True
def _segment_circle_intersections(
    first: np.ndarray,
    second: np.ndarray,
    center: np.ndarray,
    radius: float,
    eps: float = 1e-10,
) -> list[np.ndarray]:
    segment = second - first
    delta = first - center
    a = float(segment @ segment)
    if a <= eps:
        return []
    b = 2.0 * float(delta @ segment)
    c = float(delta @ delta) - radius**2
    disc = b**2 - 4.0 * a * c
    if disc < -eps:
        return []
    root = np.sqrt(max(0.0, disc))
    hits: list[np.ndarray] = []
    for t in ((-b - root) / (2.0 * a), (-b + root) / (2.0 * a)):
        if -eps <= t <= 1.0 + eps:
            p = first + min(1.0, max(0.0, t)) * segment
            if not hits or np.linalg.norm(p - hits[-1]) > eps:
                hits.append(p)
    return hits
def _circle_circle_intersections(
    first_center: np.ndarray,
    first_radius: float,
    second_center: np.ndarray,
    second_radius: float,
    eps: float = 1e-10,
) -> list[np.ndarray]:
    delta = second_center - first_center
    d = float(np.linalg.norm(delta))
    if d <= eps:
        return []
    if d > first_radius + second_radius + eps:
        return []
    if d < abs(first_radius - second_radius) - eps:
        return []
    x = (
        first_radius**2 - second_radius**2 + d**2
    ) / (2.0 * d)
    h2 = first_radius**2 - x**2
    if h2 < -eps:
        return []
    base = first_center + x * delta / d
    normal = np.array([-delta[1], delta[0]], dtype=float)
    normal /= d
    h = np.sqrt(max(0.0, h2))
    if h <= eps:
        return [base]
    return [base + h * normal, base - h * normal]
def _polygon_diameter_excluding_disks(
    polygon: Sequence[Point],
    excluded_disks: Sequence[tuple[Point, float]],
    circle_samples: int,
    eps: float = 1e-8,
) -> float:
    if not polygon:
        return 0.0
    if not excluded_disks:
        return polygon_diameter(polygon)[0]
    clean_disks = [
        (np.asarray(center, dtype=float), float(radius))
        for center, radius in excluded_disks
    ]
    def allowed(point: np.ndarray) -> bool:
        return all(
            np.linalg.norm(point - center) >= radius - eps
            for center, radius in clean_disks
        )
    outer_diameter, outer_pair = polygon_diameter(polygon)
    if all(allowed(np.asarray(point, dtype=float)) for point in outer_pair):
        return outer_diameter
    candidates: list[np.ndarray] = []
    polygon_arrays = [np.asarray(point, dtype=float) for point in polygon]
    candidates.extend(point for point in polygon_arrays if allowed(point))
    for index, first in enumerate(polygon_arrays):
        second = polygon_arrays[(index + 1) % len(polygon_arrays)]
        for center, radius in clean_disks:
            for point in _segment_circle_intersections(
                first, second, center, radius, eps
            ):
                if allowed(point):
                    candidates.append(point)
    angles = np.linspace(0.0, 2.0 * pi, circle_samples, endpoint= False)
    circle_directions = np.column_stack([np.cos(angles), np.sin(angles)])
    for center, radius in clean_disks:
        for point in center + radius * circle_directions:
            if _point_in_convex_polygon(point, polygon, eps) and allowed(point):
                candidates.append(point)
    for first_index, (first_center, first_radius) in enumerate(clean_disks):
        for second_center, second_radius in clean_disks[first_index + 1 :]:
            for point in _circle_circle_intersections(
                first_center,
                first_radius,
                second_center,
                second_radius,
                eps,
            ):
                if _point_in_convex_polygon(point, polygon, eps) and allowed(point):
                    candidates.append(point)
    if not candidates:
        return 0.0
    return _point_set_diameter(np.asarray(candidates, dtype=float))
def _clip_by_distance_comparison(
    polygon: Sequence[Point],
    first_station: Point,
    second_station: Point,
    eps: float = 1e-8,
) -> list[Point]:
    s1 = np.asarray(first_station, dtype=float)
    s2 = np.asarray(second_station, dtype=float)
    normal = 2.0 * (s1 - s2)
    normal2 = float(normal @ normal)
    if normal2 <= eps**2:
        return []
    rhs = float(s1 @ s1 - s2 @ s2)
    p0 = rhs * normal / normal2
    line_dir = np.array(
        [normal[1], -normal[0]], dtype=float
    )
    line_dir /= np.linalg.norm(line_dir)
    return clip_polygon_by_left_halfplane(
        polygon,
        (float(p0[0]), float(p0[1])),
        (float(line_dir[0]), float(line_dir[1])),
        eps,
    )
# 【学习注释】候选点怎么生成：不是随便撒点，而是由**三重几何约束**交出范围 ——
# 保证接收透镜（源必须在第二点有效半径内）、领结形交会禁区（两线不能太平行）、
# 以及稳健直径的近优水平集；只在可行范围内布点，效率高且不漏掉最优解。

def generate_candidate_points(
    source_samples: np.ndarray,
    first_station: Point,
    first_bearing_deg: float,
    config: Q2Config,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u_dir, v_dir = _directions(first_bearing_deg)
    relative = source_samples - np.asarray(first_station, dtype=float)
    u0 = relative @ u_dir
    v0 = relative @ v_dir
    pad = config.max_receive_radius
    u_values = np.linspace(
        float(u0.min() - pad),
        float(u0.max() + pad),
        config.candidate_axis_samples,
    )
    v_values = np.linspace(
        float(v0.min() - pad),
        float(v0.max() + pad),
        config.candidate_axis_samples,
    )
    u_grid, v_grid = np.meshgrid(u_values, v_values, indexing="xy")
    local = np.column_stack([u_grid.ravel(), v_grid.ravel()])
    points = (
        np.asarray(first_station, dtype=float)
        + local[:, :1] * u_dir
        + local[:, 1:] * v_dir
    )
    keep = np.linalg.norm(
        points - np.asarray(first_station, dtype=float), axis=1
    ) > 1e-8
    return points[keep], local[keep, 0], local[keep, 1]
# 【学习注释】计算候选第二检测点对应的 **J_rob = 最坏源位置下的定位直径**。
# 注意：不是先做点估计再评优，而是对先验可行集 U 内**所有可能位置取最坏** ——
# 这才与题面"误差与地点绑定、不能靠重复测量平均掉"的设定自洽。

def _compute_raw_metrics(
    candidate_points: np.ndarray,
    candidate_u: np.ndarray,
    candidate_v: np.ndarray,
    source_samples: np.ndarray,
    first_station: Point,
    first_bearing_deg: float,
    config: Q2Config,
    first_region_diameter: float,
) -> list[dict[str, float | Point]]:
    s1 = np.asarray(first_station, dtype=float)
    d1 = source_samples - s1
    r1 = np.linalg.norm(d1, axis=1)
    u1 = d1 / r1[:, None]
    tan_err = np.tan(np.deg2rad(config.bearing_error_deg))
    rays, quad_w, r_low, r_high = (
        _coverage_angular_quadrature(
            first_station,
            first_bearing_deg,
            config,
        )
    )
    rows: list[dict[str, float | Point]] = []
    for s2, u, v in zip(candidate_points, candidate_u, candidate_v):
        d2 = source_samples - s2
        r2 = np.linalg.norm(d2, axis=1)
        has_bearing = (
            (r2 > config.too_strong_radius)
            & (r2 <= config.max_receive_radius)
        )
        safe, uncertain, failure = (
            _continuous_coverages_for_candidate(
                s2,
                s1,
                rays,
                quad_w,
                r_low,
                r_high,
                config,
            )
        )
        if np.any(has_bearing):
            r2_sel = r2[has_bearing]
            u2 = d2[has_bearing] / r2_sel[:, None]
            cos_phi = np.abs(np.einsum("ij,ij->i", u1[has_bearing], u2))
            cos_phi = np.clip(cos_phi, 0.0, 1.0)
            sin_phi = np.sqrt(np.maximum(0.0, 1.0 - cos_phi**2))
            angle_cost = float(np.mean(1.0 - sin_phi))
            sin_safe = np.maximum(sin_phi, 1e-8)
            r1_sel = r1[has_bearing]
            linear_d = (
                2.0
                * tan_err
                / sin_safe
                * np.sqrt(
                    r1_sel**2
                    + r2_sel**2
                    + 2.0 * r1_sel * r2_sel * cos_phi
                )
            )
            proxy_d = min(
                first_region_diameter,
                float(np.max(linear_d)),
            )
        else:
            angle_cost = 1.0
            proxy_d = first_region_diameter
        rows.append(
            {
                "point": (float(s2[0]), float(s2[1])),
                "local_u": float(u),
                "local_v": float(v),
                "safe_coverage": safe,
                "uncertain_coverage": uncertain,
                "certain_failure": failure,
                "angle_penalty": angle_cost,
                "proxy_diameter": proxy_d,
            }
        )
    return rows
def _polygon_after_second_bearing(
    candidate_base_polygon: Sequence[Point],
    second_station: Point,
    measured_bearing_deg: float,
    config: Q2Config,
) -> list[Point]:
    return _clip_by_bearing(
        candidate_base_polygon,
        second_station,
        measured_bearing_deg,
        config.bearing_error_deg,
    )
def _validate_candidate_on_grid(
    record: dict[str, float | Point],
    first_polygon: Sequence[Point],
    source_samples: np.ndarray,
    first_station: Point,
    config: Q2Config,
    first_region_diameter: float,
) -> tuple[float, float, float, float]:
    point = np.asarray(record["point"], dtype=float)
    station = np.asarray(first_station, dtype=float)
    r1 = np.linalg.norm(source_samples - station, axis=1)
    source_from_second = source_samples - point
    r2 = np.linalg.norm(source_from_second, axis=1)
    second_station = float(point[0]), float(point[1])
    strong_polygon = _clip_by_disk(
        first_polygon,
        second_station,
        config.too_strong_radius,
        config.disk_sides,
    )
    strong_signal_diameter = _polygon_diameter_excluding_disks(
        strong_polygon,
        [(first_station, config.too_strong_radius)],
        config.disk_sides,
    )
    no_signal_outer = _clip_by_distance_comparison(
        first_polygon, first_station, second_station
    )
    no_signal_diameter = _polygon_diameter_excluding_disks(
        no_signal_outer,
        [
            (first_station, config.too_strong_radius),
            (second_station, config.min_receive_radius),
        ],
        config.disk_sides,
    )
    candidate_base_polygon = _clip_by_disk(
        first_polygon,
        second_station,
        config.max_receive_radius,
        config.disk_sides,
    )
    possible = (
        (r2 > config.too_strong_radius)
        & (r2 <= config.max_receive_radius)
    )
    possible_indices = np.flatnonzero(possible)
    bearing_grid_worst = 0.0
    if len(possible_indices) and candidate_base_polygon:
        error_values = np.linspace(
            -config.bearing_error_deg,
            config.bearing_error_deg,
            config.bearing_error_samples,
        )
        for source_index in possible_indices:
            source = source_samples[source_index]
            true_bearing = degrees(
                atan2(source[1] - point[1], source[0] - point[0])
            )
            for error in error_values:
                polygon = _polygon_after_second_bearing(
                    candidate_base_polygon,
                    second_station,
                    true_bearing + float(error),
                    config,
                )
                if polygon:
                    diameter_value = _polygon_diameter_excluding_disks(
                        polygon,
                        [
                            (first_station, config.too_strong_radius),
                            (second_station, config.too_strong_radius),
                        ],
                        config.disk_sides,
                    )
                    bearing_grid_worst = max(
                        bearing_grid_worst, diameter_value
                    )
    robust_diameter = min(
        first_region_diameter,
        max(
            bearing_grid_worst,
            strong_signal_diameter,
            no_signal_diameter,
        ),
    )
    return (
        bearing_grid_worst,
        strong_signal_diameter,
        no_signal_diameter,
        robust_diameter,
    )
def _build_candidate_metrics(
    raw_records: Sequence[dict[str, float | Point]],
    first_region_diameter: float,
    score_weights: tuple[float, float, float, float],
) -> tuple[
    tuple[CandidateMetrics, ...],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    safe = np.array([record["safe_coverage"] for record in raw_records])
    failure = np.array([record["certain_failure"] for record in raw_records])
    diameter = np.array([record["proxy_diameter"] for record in raw_records])
    angle = np.array([record["angle_penalty"] for record in raw_records])
    score = calculate_weighted_scores(
        safe,
        failure,
        diameter,
        angle,
        first_region_diameter,
        score_weights,
    )
    nondominated = _pareto_mask(
        safe,
        failure,
        np.clip(diameter / max(first_region_diameter, 1e-9), 0.0, 1.0),
        angle,
    )
    candidates = tuple(
        CandidateMetrics(
            point=record["point"],
            local_u=float(record["local_u"]),
            local_v=float(record["local_v"]),
            safe_coverage=float(record["safe_coverage"]),
            uncertain_coverage=float(record["uncertain_coverage"]),
            certain_failure=float(record["certain_failure"]),
            angle_penalty=float(record["angle_penalty"]),
            proxy_diameter=float(record["proxy_diameter"]),
            proxy_score=float(score[index]),
            pareto=bool(nondominated[index]),
            refinement_level=int(record.get("refinement_level", 0)),
        )
        for index, record in enumerate(raw_records)
    )
    return candidates, safe, failure, diameter, angle, score, nondominated
def _validate_selected_candidates(
    selected_indices: Sequence[int],
    candidates: Sequence[CandidateMetrics],
    raw_records: Sequence[dict[str, float | Point]],
    first_polygon: Sequence[Point],
    source_samples: np.ndarray,
    first_station: Point,
    config: Q2Config,
    first_region_diameter: float,
    score_weights: tuple[float, float, float, float],
) -> list[CandidateMetrics]:
    validated: list[CandidateMetrics] = []
    for index in selected_indices:
        candidate = candidates[index]
        bearing_worst, strong_signal, no_signal, robust = (
            _validate_candidate_on_grid(
                raw_records[index],
                first_polygon,
                source_samples,
                first_station,
                config,
                first_region_diameter,
            )
        )
        validated_score = evaluate_candidate_score(
            candidate.safe_coverage,
            candidate.certain_failure,
            robust,
            candidate.angle_penalty,
            first_region_diameter,
            score_weights,
        ).total
        validated.append(
            replace(
                candidate,
                validated_bearing_worst=bearing_worst,
                strong_signal_diameter=strong_signal,
                no_signal_diameter=no_signal,
                validated_robust_diameter=robust,
                validated_score=float(validated_score),
            )
        )
    return validated
def _make_decision_from_candidates(
    validated: Sequence[CandidateMetrics],
    maximum_safe_coverage: float,
    first_station: Point,
    config: Q2Config,
    improvement_tolerance_ratio: float | None = None,
):
    if not validated:
        raise RuntimeError (" 没有可用于离散几何验证的候选点")
    return make_final_decision(
        np.array([item.safe_coverage for item in validated]),
        np.array([item.certain_failure for item in validated]),
        np.array([float(item.validated_robust_diameter) for item in validated]),
        np.array([item.angle_penalty for item in validated]),
        np.array([float(item.validated_score) for item in validated]),
        np.array(
            [
                np.hypot(
                    item.point[0] - first_station[0],
                    item.point[1] - first_station[1],
                )
                for item in validated
            ]
        ),
        config.reliability_loss_levels,
        (
            config.tradeoff_improvement_tolerance_ratio
            if improvement_tolerance_ratio is None
            else improvement_tolerance_ratio
        ),
        config.diameter_tolerance_ratio,
        config.score_tolerance,
        maximum_safe_coverage=maximum_safe_coverage,
    )
def _candidate_grid_steps(
    source_samples: np.ndarray,
    first_station: Point,
    first_bearing_deg: float,
    config: Q2Config,
) -> tuple[float, float]:
    forward, normal = _directions(first_bearing_deg)
    relative_sources = source_samples - np.asarray(first_station, dtype=float)
    source_u = relative_sources @ forward
    source_v = relative_sources @ normal
    denominator = config.candidate_axis_samples - 1
    return (
        float(source_u.max() - source_u.min() + 2.0 * config.max_receive_radius)
        / denominator,
        float(source_v.max() - source_v.min() + 2.0 * config.max_receive_radius)
        / denominator,
    )
def _generate_local_refinement_points(
    center: CandidateMetrics,
    first_station: Point,
    first_bearing_deg: float,
    step_u: float,
    step_v: float,
    axis_samples: int,
    known_points: set[tuple[float, float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    offsets_u = np.linspace(-0.5 * step_u, 0.5 * step_u, axis_samples)
    offsets_v = np.linspace(-0.5 * step_v, 0.5 * step_v, axis_samples)
    uu, vv = np.meshgrid(
        center.local_u + offsets_u,
        center.local_v + offsets_v,
        indexing="xy",
    )
    local_u = uu.ravel()
    local_v = vv.ravel()
    forward, normal = _directions(first_bearing_deg)
    points = (
        np.asarray(first_station, dtype=float)
        + local_u[:, None] * forward
        + local_v[:, None] * normal
    )
    keep = []
    for point in points:
        key = (round(float(point[0]), 8), round(float(point[1]), 8))
        keep.append(
            key not in known_points
            and np.linalg.norm(point - np.asarray(first_station, dtype=float))
            > 1e-8
        )
    mask = np.asarray(keep, dtype=bool)
    return points[mask], local_u[mask], local_v[mask]
def _select_refinement_centers(
    validated: Sequence[CandidateMetrics],
    near_optimal_indices: Sequence[int],
    step_u: float,
    step_v: float,
    maximum_centers: int,
) -> tuple[CandidateMetrics, ...]:
    ordered = sorted(
        (validated[index] for index in near_optimal_indices),
        key=lambda item: (
            float(item.validated_robust_diameter),
            item.certain_failure,
            item.angle_penalty,
            np.hypot(item.local_u, item.local_v),
        ),
    )
    separation = 0.75 * max(step_u, step_v)
    selected: list[CandidateMetrics] = []
    for candidate in ordered:
        if all(
            np.hypot(
                candidate.point[0] - center.point[0],
                candidate.point[1] - center.point[1],
            )
            > separation
            for center in selected
        ):
            selected.append(candidate)
        if len(selected) >= maximum_centers:
            break
    return tuple(selected)
def _select_decision_refinement_centers(
    validated: Sequence[CandidateMetrics],
    near_optimal_indices: Sequence[int],
    safe_near_optimal_indices: Sequence[int],
    step_u: float,
    step_v: float,
    maximum_centers: int,
) -> tuple[CandidateMetrics, ...]:
    main_budget = max(1, (maximum_centers + 1) // 2)
    safe_budget = max(0, maximum_centers - main_budget)
    main_centers = _select_refinement_centers(
        validated,
        near_optimal_indices,
        step_u,
        step_v,
        main_budget,
    )
    safe_centers = (
        _select_refinement_centers(
            validated,
            safe_near_optimal_indices,
            step_u,
            step_v,
            safe_budget,
        )
        if safe_budget
        else ()
    )
    selected: list[CandidateMetrics] = list(main_centers)
    selected_points = {candidate.point for candidate in selected}
    for candidate in safe_centers:
        if candidate.point not in selected_points:
            selected.append(candidate)
            selected_points.add(candidate.point)
    if len(selected) < maximum_centers:
        combined_indices = tuple(
            dict.fromkeys((*near_optimal_indices, *safe_near_optimal_indices))
        )
        extras = _select_refinement_centers(
            validated,
            combined_indices,
            step_u,
            step_v,
            maximum_centers,
        )
        for candidate in extras:
            if candidate.point not in selected_points:
                selected.append(candidate)
                selected_points.add(candidate.point)
            if len(selected) >= maximum_centers:
                break
    return tuple(selected)
def _numerical_resolution_config(
    config: Q2Config,
    *,
    fine: bool,
) -> Q2Config:
    factor = config.numerical_calibration_factor
    if fine:
        return replace(
            config,
            source_radial_samples=config.source_radial_samples * factor,
            source_angular_samples=(config.source_angular_samples - 1) * factor + 1,
            bearing_error_samples=(config.bearing_error_samples - 1) * factor + 1,
            disk_sides=config.disk_sides * factor,
            perform_numerical_calibration=False,
            local_refinement_passes=0,
        )
    return replace(
        config,
        source_radial_samples=max(3, config.source_radial_samples // factor),
        source_angular_samples=max(
            3, (config.source_angular_samples - 1) // factor + 1
        ),
        bearing_error_samples=max(
            3, (config.bearing_error_samples - 1) // factor + 1
        ),
        disk_sides=max(12, config.disk_sides // factor),
        perform_numerical_calibration=False,
        local_refinement_passes=0,
    )
def _build_numerical_validation_context(
    first_station: Point,
    first_bearing_deg: float,
    config: Q2Config,
) -> tuple[Q2Config, tuple[Point, ...], np.ndarray, float]:
    polygon = build_first_feasible_polygon(
        first_station, first_bearing_deg, config
    )
    first_diameter = _polygon_diameter_excluding_disks(
        polygon,
        [(first_station, config.too_strong_radius)],
        config.disk_sides,
    )
    samples = sample_first_feasible_region(
        first_station, first_bearing_deg, config
    )
    return config, tuple(polygon), samples, float(first_diameter)
def _evaluate_stability_records(
    candidates_with_roles: Sequence[tuple[int, str]],
    validated: Sequence[CandidateMetrics],
    first_station: Point,
    coarse_context: tuple[Q2Config, tuple[Point, ...], np.ndarray, float],
    fine_context: tuple[Q2Config, tuple[Point, ...], np.ndarray, float],
) -> tuple[NumericalStabilityRecord, ...]:
    def evaluate(
        candidate: CandidateMetrics,
        context: tuple[Q2Config, tuple[Point, ...], np.ndarray, float],
    ) -> float:
        resolution_config, polygon, samples, first_diameter = context
        return float(
            _validate_candidate_on_grid(
                {"point": candidate.point},
                polygon,
                samples,
                first_station,
                resolution_config,
                first_diameter,
            )[3]
        )
    records: list[NumericalStabilityRecord] = []
    for index, role in candidates_with_roles:
        candidate = validated[index]
        default_diameter = float(candidate.validated_robust_diameter)
        coarse_diameter = evaluate(candidate, coarse_context)
        fine_diameter = evaluate(candidate, fine_context)
        relative_difference = abs(fine_diameter - default_diameter) / max(
            abs(fine_diameter), 1e-12
        )
        records.append(
            NumericalStabilityRecord(
                role=role,
                point=candidate.point,
                coarse_robust_diameter=coarse_diameter,
                default_robust_diameter=default_diameter,
                fine_robust_diameter=fine_diameter,
                default_fine_relative_difference=float(relative_difference),
            )
        )
    return tuple(records)
# 【学习注释】问题二主入口：**可靠性约束下的稳健优化**。
# 流程：构造 U → 生成候选点 → 逐点算 J_rob 与保证接收比例 →
#       Pareto 筛选 → 给出最优与近优候选区域（关于示向线对称的两支窄带）。
# 结论：最优点落在**近端接收圆的边界弧**上，距 S1 约 1016 m、与示向方向成 34.7°。

def solve_second_station(
    first_station: Point,
    first_bearing_deg: float,
    config: Q2Config | None = None,
) -> Q2Result:
    active_config = config or Q2Config()
    active_config.validate()
    clean_station = _validate_station_and_bearing(first_station, first_bearing_deg)
    clean_bearing = float(first_bearing_deg) % 360.0
    first_polygon = build_first_feasible_polygon(
        clean_station, clean_bearing, active_config
    )
    first_region_diameter = _polygon_diameter_excluding_disks(
        first_polygon,
        [(clean_station, active_config.too_strong_radius)],
        active_config.disk_sides,
    )
    source_samples = sample_first_feasible_region(
        clean_station, clean_bearing, active_config
    )
    candidate_points, candidate_u, candidate_v = generate_candidate_points(
        source_samples, clean_station, clean_bearing, active_config
    )
    raw_records = _compute_raw_metrics(
        candidate_points,
        candidate_u,
        candidate_v,
        source_samples,
        clean_station,
        clean_bearing,
        active_config,
        first_region_diameter,
    )
    score_weights = (
        active_config.weight_not_safe,
        active_config.weight_certain_failure,
        active_config.weight_diameter,
        active_config.weight_angle,
    )
    (
        candidates,
        safe_array,
        failure_array,
        diameter_array,
        angle_array,
        _,
        pareto,
    ) = _build_candidate_metrics(raw_records, first_region_diameter, score_weights)
    selected_indices = select_validation_indices(
        safe_array,
        diameter_array,
        np.asarray([candidate.point for candidate in candidates], dtype=float),
        active_config.validated_candidate_count,
        active_config.reliability_loss_levels,
    )
    validated = _validate_selected_candidates(
        selected_indices,
        candidates,
        raw_records,
        first_polygon,
        source_samples,
        clean_station,
        active_config,
        first_region_diameter,
        score_weights,
    )
    step_u, step_v = _candidate_grid_steps(
        source_samples,
        clean_station,
        clean_bearing,
        active_config,
    )
    known_points = {
        (round(item.point[0], 8), round(item.point[1], 8)) for item in candidates
    }
    all_records = list(raw_records)
    all_candidates = list(candidates)
    maximum_safe = max(item.safe_coverage for item in all_candidates)
    decision = _make_decision_from_candidates(
        validated,
        maximum_safe,
        clean_station,
        active_config,
    )
    previous_near = [
        validated[index] for index in decision.near_optimal_indices
    ]
    previous_best_diameter = float(
        validated[decision.best_index].validated_robust_diameter
    )
    refinement_converged = False
    refinement_resolution = max(step_u, step_v)
    refinement_position_change = float("inf")
    refinement_diameter_relative_change = float("inf")
    for refinement_level in range(1, active_config.local_refinement_passes + 1):
        centers = _select_decision_refinement_centers(
            validated,
            decision.near_optimal_indices,
            decision.safe_near_optimal_indices,
            step_u,
            step_v,
            active_config.max_refinement_centers,
        )
        point_parts: list[np.ndarray] = []
        u_parts: list[np.ndarray] = []
        v_parts: list[np.ndarray] = []
        temporary_known = set(known_points)
        for center in centers:
            points_part, u_part, v_part = _generate_local_refinement_points(
                center,
                clean_station,
                clean_bearing,
                step_u,
                step_v,
                active_config.local_refinement_axis_samples,
                temporary_known,
            )
            if len(points_part):
                point_parts.append(points_part)
                u_parts.append(u_part)
                v_parts.append(v_part)
                temporary_known.update(
                    (round(float(point[0]), 8), round(float(point[1]), 8))
                    for point in points_part
                )
        if not point_parts:
            break
        local_points = np.vstack(point_parts)
        local_u = np.concatenate(u_parts)
        local_v = np.concatenate(v_parts)
        local_records = _compute_raw_metrics(
            local_points,
            local_u,
            local_v,
            source_samples,
            clean_station,
            clean_bearing,
            active_config,
            first_region_diameter,
        )
        for record in local_records:
            record["refinement_level"] = float(refinement_level)
        local_candidates, *_ = _build_candidate_metrics(
            local_records,
            first_region_diameter,
            score_weights,
        )
        all_records.extend(local_records)
        all_candidates.extend(local_candidates)
        local_indices = tuple(range(len(local_records)))
        validated.extend(
            _validate_selected_candidates(
                local_indices,
                local_candidates,
                local_records,
                first_polygon,
                source_samples,
                clean_station,
                active_config,
                first_region_diameter,
                score_weights,
            )
        )
        for item in local_candidates:
            known_points.add((round(item.point[0], 8), round(item.point[1], 8)))
        maximum_safe = max(
            maximum_safe,
            max(item.safe_coverage for item in local_candidates),
        )
        decision = _make_decision_from_candidates(
            validated,
            maximum_safe,
            clean_station,
            active_config,
        )
        divisor = active_config.local_refinement_axis_samples - 1
        step_u /= divisor
        step_v /= divisor
        refinement_resolution = max(step_u, step_v)
        current_best = validated[decision.best_index]
        refinement_position_change = min(
            float(
                np.hypot(
                    current_best.point[0] - previous.point[0],
                    current_best.point[1] - previous.point[1],
                )
            )
            for previous in previous_near
        )
        refinement_diameter_relative_change = abs(
            float(current_best.validated_robust_diameter) - previous_best_diameter
        ) / max(previous_best_diameter, 1e-12)
        previous_best_diameter = float(current_best.validated_robust_diameter)
        previous_near = [
            validated[index] for index in decision.near_optimal_indices
        ]
        if (
            refinement_level >= 2
            and refinement_resolution
            <= active_config.position_convergence_tolerance
            and refinement_position_change
            <= active_config.position_convergence_tolerance
            and refinement_diameter_relative_change
            <= active_config.diameter_convergence_tolerance_ratio
        ):
            refinement_converged = True
            break
    candidates, *_ = _build_candidate_metrics(
        all_records,
        first_region_diameter,
        score_weights,
    )
    candidate_by_point = {candidate.point: candidate for candidate in candidates}
    validated = [
        replace(
            item,
            proxy_score=candidate_by_point[item.point].proxy_score,
            pareto=candidate_by_point[item.point].pareto,
        )
        for item in validated
    ]
    maximum_safe = max(item.safe_coverage for item in candidates)
    decision = _make_decision_from_candidates(
        validated,
        maximum_safe,
        clean_station,
        active_config,
    )
    numerical_calibration_applied = False
    numerical_error_ratio = active_config.tradeoff_improvement_tolerance_ratio
    numerical_stability_records: tuple[NumericalStabilityRecord, ...] = ()
    if active_config.perform_numerical_calibration:
        coarse_config = _numerical_resolution_config(active_config, fine= False)
        fine_config = _numerical_resolution_config(active_config, fine= True)
        coarse_context = _build_numerical_validation_context(
            clean_station, clean_bearing, coarse_config
        )
        fine_context = _build_numerical_validation_context(
            clean_station, clean_bearing, fine_config
        )
        baseline_index = decision.tradeoff_points[0].best_index
        minimum_index = min(
            decision.tradeoff_points,
            key=lambda point: point.robust_diameter,
        ).best_index
        provisional_index = decision.best_index
        role_by_index: dict[int, list[str]] = {}
        for index, role in (
            (baseline_index, " 零可靠性损失最优点"),
            (minimum_index, " 最大探索范围内最小直径点"),
            (provisional_index, " 标定前推荐点"),
        ):
            role_by_index.setdefault(index, []).append(role)
        calibration_targets = tuple(
            (index, "、".join(roles))
            for index, roles in role_by_index.items()
        )
        numerical_stability_records = _evaluate_stability_records(
            calibration_targets,
            validated,
            clean_station,
            coarse_context,
            fine_context,
        )
        numerical_error_ratio = max(
            record.default_fine_relative_difference
            for record in numerical_stability_records
        )
        numerical_calibration_applied = True
        decision = _make_decision_from_candidates(
            validated,
            maximum_safe,
            clean_station,
            active_config,
            improvement_tolerance_ratio=numerical_error_ratio,
        )
        calibrated_indices = set(role_by_index)
        while decision.best_index not in calibrated_indices:
            extra_records = _evaluate_stability_records(
                ((decision.best_index, " 标定后推荐点"),),
                validated,
                clean_station,
                coarse_context,
                fine_context,
            )
            numerical_stability_records += extra_records
            calibrated_indices.add(decision.best_index)
            numerical_error_ratio = max(
                numerical_error_ratio,
                extra_records[0].default_fine_relative_difference,
            )
            decision = _make_decision_from_candidates(
                validated,
                maximum_safe,
                clean_station,
                active_config,
                improvement_tolerance_ratio=numerical_error_ratio,
            )
    validated = [
        replace(item, validated_pareto=bool(decision.pareto_mask[index]))
        for index, item in enumerate(validated)
    ]
    best = validated[decision.best_index]
    near_optimal = tuple(
        validated[index] for index in decision.near_optimal_indices
    )
    safe_near_optimal = tuple(
        validated[index] for index in decision.safe_near_optimal_indices
    )
    pareto_best = validated[decision.weighted_best_index]
    pareto_near_optimal = tuple(
        validated[index] for index in decision.weighted_near_optimal_indices
    )
    reliability_feasible = tuple(
        item
        for item in candidates
        if item.safe_coverage >= decision.accepted_safe_coverage_floor - 1e-12
    )
    return Q2Result(
        first_station=clean_station,
        first_bearing_deg=clean_bearing,
        first_region_polygon=tuple(first_polygon),
        first_region_diameter=float(first_region_diameter),
        source_samples=source_samples,
        candidates=candidates,
        validated_candidates=tuple(validated),
        primary_safe_coverage=decision.maximum_safe_coverage,
        selected_reliability_loss=decision.selected_reliability_loss,
        accepted_safe_coverage_floor=decision.accepted_safe_coverage_floor,
        tradeoff_improvement_tolerance=decision.improvement_tolerance,
        reliability_tradeoff=decision.tradeoff_points,
        reliability_feasible_candidates=reliability_feasible,
        refinement_converged=refinement_converged,
        refinement_resolution=float(refinement_resolution),
        refinement_position_change=float(refinement_position_change),
        refinement_diameter_relative_change=float(
            refinement_diameter_relative_change
        ),
        numerical_calibration_applied=numerical_calibration_applied,
        numerical_error_ratio=float(numerical_error_ratio),
        numerical_stability_records=numerical_stability_records,
        best_candidate=best,
        near_optimal_candidates=near_optimal,
        safe_near_optimal_candidates=safe_near_optimal,
        pareto_best_candidate=pareto_best,
        pareto_near_optimal_candidates=pareto_near_optimal,
        config=active_config,
    )
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="B 题问题 2 参数化第二检测点求解器")
    parser.add_argument("--s1-x", type=float, default=0.0, help=" 第一检测点 x 坐标")
    parser.add_argument("--s1-y", type=float, default=0.0, help=" 第一检测点 y 坐标")
    parser.add_argument("--bearing", type=float, default=0.0, help=" 第一次示向度/度")
    parser.add_argument(
        "--candidate-samples",
        type=int,
        default=55,
        help=" 候选网格每个坐标轴的采样数",
    )
    parser.add_argument(
        "--source-radial-samples",
        type=int,
        default=180,
        help=" 第一次可行域的径向样本数",
    )
    parser.add_argument(
        "--source-angular-samples",
        type=int,
        default=41,
        help=" 第一次可行域的角向样本数",
    )
    parser.add_argument(
        "--disk-sides",
        type=int,
        default=144,
        help=" 圆边界及排除圆弧的离散边数",
    )
    parser.add_argument(
        "--validated-candidates",
        type=int,
        default=24,
        help=" 按可靠性分层、代理直径和空间多样性选出的初始精验证数",
    )
    parser.add_argument(
        "--bearing-error-samples",
        type=int,
        default=5,
        help="[-1°, 1°] 内的确定性误差网格点数",
    )
    parser.add_argument(
        "--reliability-loss-levels",
        type=float,
        nargs="+",
        default=list(Q2Config().reliability_loss_levels),
        help=" 精验证探索分层及最大损失范围，例如 0 .005 .01 .02 .03 .05",
    )
    parser.add_argument(
        "--skip-numerical-calibration",
        action="store_true",
        help=" 跳过粗/默认/加密复算，仅供快速调试",
    )
    parser.add_argument(
        "--numerical-calibration-factor",
        type=int,
        default=2,
        help=" 关键候选点数值加密倍数，默认 2",
    )
    parser.add_argument(
        "--tradeoff-improvement-tolerance-ratio",
        type=float,
        default=0.01,
        help=" 关闭数值标定时使用的后备相对容差",
    )
    parser.add_argument(
        "--local-refinement-samples",
        type=int,
        default=3,
        help=" 每轮局部加密网格的单轴点数",
    )
    parser.add_argument(
        "--local-refinement-passes",
        type=int,
        default=4,
        help=" 围绕当前最优点逐层缩小网格的轮数",
    )
    parser.add_argument(
        "--max-refinement-centers",
        type=int,
        default=4,
        help=" 每轮同时跟踪的相互分离近优区域数量",
    )
    parser.add_argument(
        "--position-convergence-tolerance",
        type=float,
        default=10.0,
        help=" 最优点坐标与局部网格分辨率的收敛阈值/米",
    )
    parser.add_argument(
        "--diameter-convergence-tolerance-ratio",
        type=float,
        default=0.01,
        help=" 相邻局部加密轮次稳健直径的相对收敛阈值",
    )
    parser.add_argument("--weight-not-safe", type=float, default=0.24)
    parser.add_argument("--weight-certain-failure", type=float, default=0.31)
    parser.add_argument("--weight-diameter", type=float, default=0.35)
    parser.add_argument("--weight-angle", type=float, default=0.10)
    parser.add_argument(
        "--score-tolerance",
        type=float,
        default=0.03,
        help=" 验证后无量纲综合评分的近优绝对容差",
    )
    parser.add_argument(
        "--diameter-tolerance-ratio",
        type=float,
        default=0.03,
        help=" 可靠接收候选区域相对最小直径的容差",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/q2"),
        help=" 结果输出目录",
    )
    return parser
def main() -> None:
    args = _build_parser().parse_args()
    config = Q2Config(
        candidate_axis_samples=args.candidate_samples,
        source_radial_samples=args.source_radial_samples,
        source_angular_samples=args.source_angular_samples,
        disk_sides=args.disk_sides,
        validated_candidate_count=args.validated_candidates,
        bearing_error_samples=args.bearing_error_samples,
        reliability_loss_levels=tuple(args.reliability_loss_levels),
        perform_numerical_calibration=not args.skip_numerical_calibration,
        numerical_calibration_factor=args.numerical_calibration_factor,
        tradeoff_improvement_tolerance_ratio=(
            args.tradeoff_improvement_tolerance_ratio
        ),
        local_refinement_axis_samples=args.local_refinement_samples,
        local_refinement_passes=args.local_refinement_passes,
        max_refinement_centers=args.max_refinement_centers,
        position_convergence_tolerance=args.position_convergence_tolerance,
        diameter_convergence_tolerance_ratio=(
            args.diameter_convergence_tolerance_ratio
        ),
        score_tolerance=args.score_tolerance,
        diameter_tolerance_ratio=args.diameter_tolerance_ratio,
        weight_not_safe=args.weight_not_safe,
        weight_certain_failure=args.weight_certain_failure,
        weight_diameter=args.weight_diameter,
        weight_angle=args.weight_angle,
    )
    result = solve_second_station(
        first_station=(args.s1_x, args.s1_y),
        first_bearing_deg=args.bearing,
        config=config,
    )
    output_dir: Path = args.output_dir
    candidate_csv_path = output_dir / " 候选点评价.csv"
    selection_figure_path = output_dir / " 选点结果图.png"
    tradeoff_csv_path = output_dir / " 可靠性权衡数据.csv"
    tradeoff_figure_path = output_dir / " 可靠性与定位精度权衡曲线.png"
    write_candidate_csv(result, candidate_csv_path)
    plot_result(result, selection_figure_path)
    write_reliability_tradeoff_csv(result, tradeoff_csv_path)
    plot_reliability_tradeoff(result, tradeoff_figure_path)
    best = result.best_candidate
    print(" 问题 2 参数化求解完成")
    print(f" 输入 S1: {result.first_station}")
    print(f" 输入示向度: {result.first_bearing_deg:.6f}°")
    print(f" 第一次可行区域直径: {result.first_region_diameter:.6f} m")
    print(f" 候选点数量: {len(result.candidates)}")
    print(f" 粗筛帕累托点数量: {sum(item.pareto for item in result.candidates)}")
    print(f" 主策略最大可靠覆盖率: {result.primary_safe_coverage:.6%}")
    print(
        " 自动选择的可靠性损失上限: "
        f"{100.0 * result.selected_reliability_loss :.3f} 个百分点"
    )
    print(
        " 主策略可接受覆盖率下限: "
        f"{result.accepted_safe_coverage_floor:.6%}"
    )
    print(f" 主策略最优候选点数量: {len(result.near_optimal_candidates)}")
    print(
        " 最高可靠覆盖近优候选点数量: "
        f"{len(result.safe_near_optimal_candidates)}"
    )
    print(
        " 可靠性可行区域网格点数量: "
        f"{len(result.reliability_feasible_candidates)}"
    )
    print(
        " 局部加密收敛: "
        f"{'是' if result.refinement_converged else '否'}，"
        f" 最终分辨率 {result.refinement_resolution:.3f} m，"
        f" 最优点变化 {result.refinement_position_change:.3f} m，"
        " 稳健直径相对变化 "
        f"{result.refinement_diameter_relative_change:.3%}"
    )
    if result.numerical_calibration_applied:
        print(
            " 数值误差标定比例: "
            f"{result.numerical_error_ratio:.4%}（默认与加密配置最大相对差）"
        )
        for record in result.numerical_stability_records:
            print(
                f" {record.role}: S2=( {record.point[0]:.3f}, "
                f"{record.point[1]:.3f})，J 粗/默认/加密 ="
                f"{record.coarse_robust_diameter:.3f}/"
                f"{record.default_robust_diameter:.3f}/"
                f"{record.fine_robust_diameter:.3f} m"
            )
    else:
        print(
            " 数值误差标定: 已跳过；使用后备容差比例 "
            f"{result.numerical_error_ratio:.4%}"
        )
    print(f" 主策略最优 S2: ( {best.point[0]:.6f}, {best.point[1]:.6f})")
    print(f" 局部坐标 (u, v): ( {best.local_u:.6f}, {best.local_v:.6f})")
    print(f" 安全覆盖率: {best.safe_coverage:.6%}")
    print(f" 接收不确定覆盖率: {best.uncertain_coverage:.6%}")
    print(f" 必然失效覆盖率: {best.certain_failure:.6%}")
    print(f" 角度惩罚: {best.angle_penalty:.6f}")
    print(f" 离散几何验证最坏测向直径: {best.validated_bearing_worst:.6f} m")
    print(f" 过强信号后验直径: {best.strong_signal_diameter:.6f} m")
    print(f" 无信号后验直径: {best.no_signal_diameter:.6f} m")
    print(f" 离散验证后的稳健后验直径: {best.validated_robust_diameter:.6f} m")
    pareto_best = result.pareto_best_candidate
    print(
        " 辅助帕累托加权折中 S2: "
        f"({pareto_best.point[0]:.6f}, {pareto_best.point[1]:.6f})"
    )
    print(f" 辅助折中点综合评分: {pareto_best.validated_score:.6f}")
    print(f" 辅助帕累托近优点数量: {len(result.pareto_near_optimal_candidates)}")
    print(f" 候选点评价数据: {candidate_csv_path.resolve()}")
    print(f" 选点结果图: {selection_figure_path.resolve()}")
    print(f" 可靠性权衡数据: {tradeoff_csv_path.resolve()}")
    print(f" 可靠性权衡曲线: {tradeoff_figure_path.resolve()}")
if __name__ == "__main__":
    main()
