# ============================================================================
# 【学习注释】问题二：候选筛选与最终决策（q2_decision.py，约 320 行）
# ----------------------------------------------------------------------------
# 把"多目标权衡"做成可解释的三层：
#   pareto_mask            —— 只保留不被任何其它候选**严格支配**的点（Pareto 前沿）；
#   select_validation_indices —— 从前沿里挑**代表点**去高保真复算（省算力）；
#   make_final_decision    —— 在"保证接收比例损失"的一维权衡上取折中，
#                             并对评分/直径设**近优容差**，避免为微小改善牺牲稳健性。
# 值得学的点：把"保证性指标"和"工程评分"分开，且只让**严格支配**的改进生效。
# ============================================================================

""" 问题 2 的候选筛选与最终决策。"""
from __future__ import annotations
from dataclasses import dataclass
from math import isfinite
from typing import Sequence
import numpy as np
@dataclass(frozen=True)
class TradeoffEnvelopePoint :
    allowed_loss: float
    actual_loss: float
    robust_diameter: float
    best_index: int
@dataclass(frozen=True)
class FinalDecision :
    pareto_mask: np.ndarray
    maximum_safe_coverage: float
    selected_reliability_loss: float
    accepted_safe_coverage_floor: float
    improvement_tolerance: float
    tradeoff_points: tuple[TradeoffEnvelopePoint, ...]
    best_index: int
    near_optimal_indices: tuple[int, ...]
    safe_near_optimal_indices: tuple[int, ...]
    weighted_best_index: int
    weighted_near_optimal_indices: tuple[int, ...]
def _validate_same_length(*arrays: np.ndarray) -> int:
    sizes = {len(np.asarray(array)) for array in arrays}
    if len(sizes) != 1:
        raise ValueError (" 候选点指标数组长度必须一致")
    n = sizes.pop()
    if n == 0:
        raise ValueError (" 候选点集不能为空")
    return n
def _clean_reliability_loss_levels(
    levels: Sequence[float],
) -> tuple[float, ...]:
    values = tuple(sorted({float(level) for level in levels}))
    if not values or abs(values[0]) > 1e-12:
        raise ValueError (" 可靠性损失层级必须从 0 开始")
    if any(not isfinite(level) or not 0.0 <= level < 1.0 for level in values):
        raise ValueError (" 可靠性损失层级必须为 [0, 1) 内的有限数")
    return values
def pareto_mask(
    safe_coverage: np.ndarray,
    certain_failure: np.ndarray,
    diameter_score: np.ndarray,
    angle_penalty: np.ndarray,
) -> np.ndarray:
    safe = np.asarray(safe_coverage, dtype=float)
    failure = np.asarray(certain_failure, dtype=float)
    diameter = np.asarray(diameter_score, dtype=float)
    angle = np.asarray(angle_penalty, dtype=float)
    n = _validate_same_length(safe, failure, diameter, angle)
    if any(
        np.any(~np.isfinite(array))
        for array in (safe, failure, diameter, angle)
    ):
        raise ValueError (" 帕累托指标不能包含非有限数")
    mask = np.ones(n, dtype=bool)
    eps = 1e-12
    for i in range(n):
        not_worse = (
            (safe >= safe[i] - eps)
            & (failure <= failure[i] + eps)
            & (diameter <= diameter[i] + eps)
            & (angle <= angle[i] + eps)
        )
        better = (
            (safe > safe[i] + eps)
            | (failure < failure[i] - eps)
            | (diameter < diameter[i] - eps)
            | (angle < angle[i] - eps)
        )
        if np.any(not_worse & better):
            mask[i] = False
    return mask
def select_validation_indices(
    safe_coverage: np.ndarray,
    proxy_diameter: np.ndarray,
    candidate_points: np.ndarray,
    validation_count: int,
    reliability_loss_levels: Sequence[float],
) -> tuple[int, ...]:
    safe = np.asarray(safe_coverage, dtype=float)
    diameter = np.asarray(proxy_diameter, dtype=float)
    points = np.asarray(candidate_points, dtype=float)
    n = _validate_same_length(safe, diameter, points)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError (" 候选点坐标必须是 n×2 数组")
    if any(np.any(~np.isfinite(array)) for array in (safe, diameter, points)):
        raise ValueError (" 验证分层指标不能包含非有限数")
    if not isinstance(validation_count, int) or validation_count <= 0:
        raise ValueError (" 验证候选数必须为正整数")
    levels = _clean_reliability_loss_levels(reliability_loss_levels)
    take = min(validation_count, n)
    max_safe = float(np.max(safe))
    losses = np.maximum(0.0, max_safe - safe)
    bands: list[np.ndarray] = []
    prev = 0.0
    for level_i, level in enumerate(levels):
        if level_i == 0:
            group = np.flatnonzero(losses <= level + 1e-12)
        else:
            group = np.flatnonzero(
                (losses > prev + 1e-12) & (losses <= level + 1e-12)
            )
        if not len(group):
            group = np.flatnonzero(losses <= level + 1e-12)
        bands.append(group)
        prev = level
    chosen: list[int] = []
    chosen_set: set[int] = set()
    visits = [0 for _ in bands]
    def choose(pool: np.ndarray, prefer_proxy: bool) -> int | None:
        free = [int(idx) for idx in pool if int(idx) not in chosen_set]
        if not free:
            return None
        pool_ids = {int(idx) for idx in pool}
        chosen_here = [idx for idx in chosen if idx in pool_ids]
        if prefer_proxy or not chosen_here:
            return min(free, key= lambda idx: (diameter[idx], idx))
        chosen_points = points[np.asarray(chosen_here, dtype=int)]
        return max(
            free,
            key=lambda idx: (
                float(
                    np.min(
                        np.linalg.norm(chosen_points - points[idx], axis=1)
                    )
                ),
                -float(diameter[idx]),
                -idx,
            ),
        )
    while len(chosen) < take:
        changed = False
        for band_i, group in enumerate(bands):
            seen = visits[band_i]
            prefer_proxy = seen < 2 or seen % 2 == 1
            idx = choose(group, prefer_proxy)
            if idx is None :
                continue
            chosen.append(idx)
            chosen_set.add(idx)
            visits[band_i] += 1
            changed = True
            if len(chosen) >= take:
                break
        if not changed:
            break
    if len(chosen) < take:
        all_ids = np.arange(n, dtype=int)
        while len(chosen) < take:
            idx = choose(all_ids, prefer_proxy= False)
            if idx is None :
                break
            chosen.append(idx)
            chosen_set.add(idx)
    return tuple(chosen)
def make_final_decision(
    safe_coverage: np.ndarray,
    certain_failure: np.ndarray,
    robust_diameter: np.ndarray,
    angle_penalty: np.ndarray,
    weighted_score: np.ndarray,
    movement_distance: np.ndarray,
    reliability_loss_levels: Sequence[float],
    improvement_tolerance_ratio: float,
    diameter_tolerance_ratio: float,
    score_tolerance: float,
    maximum_safe_coverage: float | None = None,
) -> FinalDecision:
    safe = np.asarray(safe_coverage, dtype=float)
    failure = np.asarray(certain_failure, dtype=float)
    diameter = np.asarray(robust_diameter, dtype=float)
    angle = np.asarray(angle_penalty, dtype=float)
    score = np.asarray(weighted_score, dtype=float)
    movement = np.asarray(movement_distance, dtype=float)
    _validate_same_length(safe, failure, diameter, angle, score, movement)
    if any(
        np.any(~np.isfinite(array))
        for array in (safe, failure, diameter, angle, score, movement)
    ):
        raise ValueError (" 决策指标不能包含非有限数")
    levels = _clean_reliability_loss_levels(reliability_loss_levels)
    if (
        not isfinite(improvement_tolerance_ratio)
        or improvement_tolerance_ratio < 0.0
    ):
        raise ValueError (" 定位改善数值容差比例必须为有限非负数")
    if not isfinite(diameter_tolerance_ratio) or diameter_tolerance_ratio < 0.0:
        raise ValueError (" 直径容差比例必须为有限非负数")
    if not isfinite(score_tolerance) or score_tolerance < 0.0:
        raise ValueError (" 评分容差必须为有限非负数")
    observed_maximum = float(np.max(safe))
    maximum_safe = (
        observed_maximum
        if maximum_safe_coverage is None
        else float(maximum_safe_coverage)
    )
    if (
        not isfinite(maximum_safe)
        or maximum_safe < observed_maximum - 1e-10
        or not 0.0 <= maximum_safe <= 1.0 + 1e-10
    ):
        raise ValueError (" 最大安全覆盖率必须覆盖全部验证候选且位于 [0, 1]")
    losses = np.maximum(0.0, maximum_safe - safe)
    maximum_allowed_loss = levels[-1]
    actual_levels = sorted(
        {
            round(float(loss), 12)
            for loss in losses
            if loss <= maximum_allowed_loss + 1e-12
        }
    )
    if not actual_levels or actual_levels[0] > 1e-12:
        raise RuntimeError (" 最高可靠覆盖层没有经过精验证的候选点")
    tradeoff_points: list[TradeoffEnvelopePoint] = []
    for allowed_loss in actual_levels:
        floor = max(0.0, maximum_safe - allowed_loss)
        eligible = [
            index
            for index in range(len(safe))
            if safe[index] >= floor - 1e-12
        ]
        if not eligible:
            continue
        best_index = min(
            eligible,
            key=lambda index: (
                diameter[index],
                failure[index],
                angle[index],
                movement[index],
            ),
        )
        tradeoff_points.append(
            TradeoffEnvelopePoint(
                allowed_loss=float(allowed_loss),
                actual_loss=float(maximum_safe - safe[best_index]),
                robust_diameter=float(diameter[best_index]),
                best_index=int(best_index),
            )
        )
    baseline_diameter = tradeoff_points[0].robust_diameter
    improvement_tolerance = improvement_tolerance_ratio * baseline_diameter
    minimum_diameter = min(point.robust_diameter for point in tradeoff_points)
    if baseline_diameter - minimum_diameter <= improvement_tolerance + 1e-12:
        selected_index = tradeoff_points[0].best_index
    else:
        selected_index = min(
            (
                index
                for index in range(len(safe))
                if losses[index] <= maximum_allowed_loss + 1e-12
                and diameter[index]
                <= minimum_diameter + improvement_tolerance + 1e-12
            ),
            key=lambda index: (
                losses[index],
                diameter[index],
                failure[index],
                angle[index],
                movement[index],
            ),
        )
    selected_loss = float(losses[selected_index])
    safe_floor = max(0.0, maximum_safe - selected_loss)
    eligible = [
        index for index in range(len(safe)) if safe[index] >= safe_floor - 1e-12
    ]
    best_index = min(
        eligible,
        key=lambda index: (
            diameter[index],
            failure[index],
            angle[index],
            movement[index],
        ),
    )
    best_diameter = float(diameter[best_index])
    near_optimal = tuple(
        index
        for index in eligible
        if diameter[index]
        <= (1.0 + diameter_tolerance_ratio) * best_diameter + 1e-12
    )
    maximum_reliability = [
        index for index in range(len(safe)) if safe[index] >= maximum_safe - 1e-12
    ]
    safe_best_diameter = min(float(diameter[index]) for index in maximum_reliability)
    safe_near_optimal = tuple(
        index
        for index in maximum_reliability
        if diameter[index]
        <= (1.0 + diameter_tolerance_ratio) * safe_best_diameter + 1e-12
    )
    nondominated = pareto_mask(safe, failure, diameter, angle)
    pareto_indices = [int(index) for index in np.flatnonzero(nondominated)]
    weighted_best_index = min(pareto_indices, key= lambda index: score[index])
    weighted_best_score = float(score[weighted_best_index])
    weighted_near_optimal = tuple(
        index
        for index in pareto_indices
        if score[index] <= weighted_best_score + score_tolerance + 1e-12
    )
    return FinalDecision(
        pareto_mask=nondominated,
        maximum_safe_coverage=maximum_safe,
        selected_reliability_loss=float(selected_loss),
        accepted_safe_coverage_floor=safe_floor,
        improvement_tolerance=float(improvement_tolerance),
        tradeoff_points=tuple(tradeoff_points),
        best_index=best_index,
        near_optimal_indices=near_optimal,
        safe_near_optimal_indices=safe_near_optimal,
        weighted_best_index=weighted_best_index,
        weighted_near_optimal_indices=weighted_near_optimal,
    )
