# ============================================================================
# 【学习注释】问题二：候选评分（q2_evaluation.py，约 70 行）
# ----------------------------------------------------------------------------
# 把定位直径、角度惩罚、移动距离等**量纲不同**的指标加权成无量纲综合评分。
# 关键做法：先各自归一化，再加权求和——这样"评分"可解释、可做灵敏度分析。
# ============================================================================

""" 问题 2 候选点评分。"""
from __future__ import annotations
from dataclasses import dataclass
from math import isfinite
import numpy as np
@dataclass(frozen=True)
class ScoreBreakdown :
    not_safe: float
    certain_failure: float
    relative_diameter: float
    angle_penalty: float
    total: float
def _validate_weights(weights: tuple[float, float, float, float]) -> None:
    w = np.asarray(weights, dtype=float)
    if np.any(~np.isfinite(w)) or np.any(w < 0.0):
        raise ValueError (" 评分权重必须为有限非负数")
    if not np.isclose(float(w.sum()), 1.0, atol=1e-12):
        raise ValueError (" 四项评分权重之和必须等于 1")
def calculate_weighted_scores(
    safe_coverage: np.ndarray,
    certain_failure: np.ndarray,
    robust_diameter: np.ndarray,
    angle_penalty: np.ndarray,
    reference_diameter: float,
    weights: tuple[float, float, float, float],
) -> np.ndarray:
    parts = [
        np.asarray(safe_coverage, dtype=float),
        np.asarray(certain_failure, dtype=float),
        np.asarray(robust_diameter, dtype=float),
        np.asarray(angle_penalty, dtype=float),
    ]
    if len({array.shape for array in parts}) != 1:
        raise ValueError (" 四项评价指标必须具有相同形状")
    if not isfinite(reference_diameter) or reference_diameter <= 0.0:
        raise ValueError (" 参考直径必须为有限正数")
    if any(np.any(~np.isfinite(array)) for array in parts):
        raise ValueError (" 评价指标不能包含非有限数")
    _validate_weights(weights)
    s, f, d, a = parts
    relative_diameter = np.clip(d / reference_diameter, 0.0, 1.0)
    return (
        weights[0] * (1.0 - s)
        + weights[1] * f
        + weights[2] * relative_diameter
        + weights[3] * a
    )
def evaluate_candidate_score(
    safe_coverage: float,
    certain_failure: float,
    robust_diameter: float,
    angle_penalty: float,
    reference_diameter: float,
    weights: tuple[float, float, float, float],
) -> ScoreBreakdown:
    total = calculate_weighted_scores(
        np.asarray([safe_coverage], dtype=float),
        np.asarray([certain_failure], dtype=float),
        np.asarray([robust_diameter], dtype=float),
        np.asarray([angle_penalty], dtype=float),
        reference_diameter,
        weights,
    )[0]
    return ScoreBreakdown(
        not_safe=1.0 - float(safe_coverage),
        certain_failure=float(certain_failure),
        relative_diameter=float(
            np.clip(robust_diameter / reference_diameter, 0.0, 1.0)
        ),
        angle_penalty=float(angle_penalty),
        total=float(total),
    )
