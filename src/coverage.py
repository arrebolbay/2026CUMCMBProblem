"""检测点网设计与覆盖性验证（问题 3 / 问题 4 的布点理论基础）。

核心问题
--------
按最坏情形有效接收半径 R_min = 1000 m 设计：为保证"任意干扰源至少被一个
检测点发现"，检测点集 {p_i} 必须使

        ∪_i B(p_i, 1000) ⊇ B(O, 1800)

这是"用半径 1000 的圆覆盖半径 1800 圆域"的经典圆覆盖问题。
K. Bezdek (1979/1983) 证明：6 个等圆最多覆盖半径比 1.7988… 的大圆；
7 个可达 2.0。本题半径比 1800/1000 = 1.8 ∈ (1.7988, 2.0]

        ⇒ 最少检测点数为 7（可证明下界），构造性方案见 seven_point_design()。

本模块提供：
  * seven_point_design        : 中心 + 半径 1400m 六均布（可验证 worst-case < 1000m）
  * worst_case_radius         : 给定布点，Monte-Carlo 估计最大覆盖距离
  * optimize_cover_radius     : 用全局优化求 k 个点的最小覆盖半径（验证下界）
  * directional_coverage_ok   : 问题 4 的强化条件（凸包内部条件）
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import numpy as np
from scipy.optimize import differential_evolution

from .config import R_MIN, REGION_RADIUS

Point = Tuple[float, float]

__all__ = [
    "seven_point_design",
    "triangular_lattice_design",
    "directional_survey_design",
    "two_ring_design",
    "hexagonal_design",
    "sample_disk",
    "sample_boundary",
    "worst_case_radius",
    "optimize_cover_radius",
    "directional_coverage_ok",
    "directional_certificate",
]


def seven_point_design(ring_radius: float = 1400.0, k: int = 6) -> List[Point]:
    """构造性 7 点方案：中心 1 点 + 半径 ring_radius 的圆环上 k 点均布。

    默认 ring_radius=1400, k=6。该方案对 R_min=1000 有正裕量，见 worst_case_radius。
    """
    pts: List[Point] = [(0.0, 0.0)]
    for i in range(k):
        ang = 360.0 * i / k
        rad = math.radians(ang)
        pts.append((ring_radius * math.cos(rad), ring_radius * math.sin(rad)))
    return pts


def hexagonal_design(r_min: float = R_MIN) -> List[Point]:
    """比值 1/2 的经典最优构型（Fejes Tóth 2005 所述 n=7 情形）的缩放版本。

    构型：1 个位于圆心的点 + 6 个位于半径 sqrt(3)*r_min 正六边形顶点的点。
    这是"半径 r 的圆覆盖半径 2r 圆域"的紧构型；对本题 R=1800 < 2*r_min=2000
    留有裕量，可用作优化算法的初始构型。
    """
    pts: List[Point] = [(0.0, 0.0)]
    ring = math.sqrt(3.0) * r_min
    for i in range(6):
        ang = math.radians(60.0 * i)
        pts.append((ring * math.cos(ang), ring * math.sin(ang)))
    return pts


# --------------------------------------------------------------------------- #
# 采样与覆盖半径评估
# --------------------------------------------------------------------------- #
def sample_disk(n: int, radius: float = REGION_RADIUS, seed: int = 0) -> np.ndarray:
    """圆域内均匀采样，返回 (n, 2) 数组。"""
    rng = np.random.default_rng(seed)
    r = radius * np.sqrt(rng.random(n))
    theta = rng.random(n) * 2.0 * math.pi
    return np.column_stack([r * np.cos(theta), r * np.sin(theta)])


def sample_boundary(n: int, radius: float = REGION_RADIUS, seed: int = 0) -> np.ndarray:
    """圆周边界均匀采样（覆盖问题的最坏点常出现在边界）。"""
    rng = np.random.default_rng(seed)
    theta = rng.random(n) * 2.0 * math.pi
    return np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])


def worst_case_radius(
    points: Sequence[Point],
    region_radius: float = REGION_RADIUS,
    n_interior: int = 20000,
    n_boundary: int = 20000,
    seed: int = 0,
) -> float:
    """估计 max_{X ∈ 圆域} min_i |X - p_i|（最大覆盖距离）。

    同时采样内部与边界；边界加密以保证最坏点被捕获。
    """
    pts = np.asarray(points, dtype=float)
    samples = np.vstack([
        sample_disk(n_interior, region_radius, seed),
        sample_boundary(n_boundary, region_radius, seed + 1),
    ])
    d = np.linalg.norm(samples[:, None, :] - pts[None, :, :], axis=2)
    return float(d.min(axis=1).max())


def optimize_cover_radius(
    k: int,
    region_radius: float = REGION_RADIUS,
    n_objective: int = 3000,
    seed: int = 7,
    maxiter: int = 60,
    popsize: int = 12,
    init_guesses: Sequence[Sequence[Point]] | None = None,
) -> Tuple[float, np.ndarray]:
    """全局优化 k 个检测点的最小覆盖半径（验证"6 点不可行"的下界论证）。

    目标：min_{p_1..p_k} max_X min_i |X - p_i|
    返回 (最优覆盖半径, 最优点集)，最后用高质量采样复核。

    参数
    ----
    init_guesses : 可选的已知优良初始构型列表（自动并入 DE 初始种群），
                   用于保证 k=7 等情形收敛到接近理论最优（比值 1/2 的六边形构型）。
    """
    fixed = np.vstack([
        sample_disk(n_objective, region_radius, seed),
        sample_boundary(n_objective, region_radius, seed + 1),
    ])

    def objective(flat: np.ndarray) -> float:
        pts = flat.reshape(k, 2)
        d = np.linalg.norm(fixed[:, None, :] - pts[None, :, :], axis=2)
        return float(d.min(axis=1).max())

    bounds = [(-region_radius * 1.1, region_radius * 1.1)] * (2 * k)
    init: object = "sobol"
    if init_guesses:
        # DE 要求初始种群形状为 (S, 2k) 且 S > 4：用随机行补足
        guesses = [np.asarray(g, dtype=float).reshape(2 * k) for g in init_guesses]
        target_rows = max(5, popsize * 2 * k)
        rng = np.random.default_rng(seed)
        extra = rng.uniform(-region_radius, region_radius, size=(max(0, target_rows - len(guesses)), 2 * k))
        init = np.vstack(guesses + [extra]) if len(extra) else np.vstack(guesses)

    result = differential_evolution(
        objective, bounds, seed=seed, maxiter=maxiter, popsize=popsize,
        tol=1e-8, polish=True, init=init, mutation=(0.4, 1.0),
        recombination=0.9, updating="deferred",
    )
    best_pts = result.x.reshape(k, 2)

    # 在 DE 最优解与各初始构型中取最优（用高质量采样复核）
    candidates = [best_pts] + [np.asarray(g, dtype=float) for g in (init_guesses or [])]
    verified_best, verified_pts = float("inf"), best_pts
    for cand in candidates:
        val = worst_case_radius(cand, region_radius,
                                n_interior=20000, n_boundary=20000, seed=99)
        if val < verified_best:
            verified_best, verified_pts = val, cand
    return verified_best, np.asarray(verified_pts, dtype=float)


def directional_coverage_ok(
    points: Sequence[Point],
    region_radius: float = REGION_RADIUS,
    r_min: float = R_MIN,
    n_samples: int = 3000,
    seed: int = 0,
) -> dict:
    """问题 4 定向情形的强化覆盖条件（凸包内部条件）。

    条件：对圆域内任意 G，G 严格位于集合 { p_i : |p_i-G| <= r_min } 的凸包内部。
    等价判定（本实现，精确且避免病态凸包）：把近旁点相对 G 的方位角排序，
    若最大角隙 < 180°，则这些点不落在任何闭半平面内 <=> G 在凸包内部。
    近旁点少于 3 个时必然存在朝向看不到 G，直接判失败。
    """
    pts = np.asarray(points, dtype=float)
    samples = sample_disk(n_samples, region_radius, seed)
    ok = 0
    for gx, gy in samples:
        delta = pts - (gx, gy)
        d = np.hypot(delta[:, 0], delta[:, 1])
        near = delta[d <= r_min]
        if near.shape[0] < 3:
            continue                      # 近旁点不足 3 个：必有朝向看不到
        ang = np.sort(np.arctan2(near[:, 1], near[:, 0]))
        gaps = np.diff(np.concatenate([ang, ang[:1] + 2.0 * math.pi]))
        if gaps.max() < math.pi - 1e-12:   # 最大角隙 < 180° <=> G 在凸包内部
            ok += 1
    return {
        "n_samples": len(samples),
        "ok": ok,
        "ok_ratio": ok / len(samples),
        "fail": len(samples) - ok,
    }


def triangular_lattice_design(spacing: float = R_MIN,
                              keep_radius: float = REGION_RADIUS + R_MIN) -> List[Point]:
    """三角格点检测点网（问题4 用）：间距 s <= R_min，覆盖到 半径+s。

    构造性保证：圆域内任意 G 必落在某个格点三角形内部，且该三角形三边 <= s <= R_min，
    故三顶点到 G 的距离均 <= R_min 且 G 位于其凸包内部 ==> 任意朝向的定向源
    都至少被一个检测点"看到"（定向覆盖条件成立）。
    """
    pts: List[Point] = []
    dy = spacing * math.sqrt(3.0) / 2.0
    n = int(keep_radius / spacing) + 3
    for j in range(-n, n + 1):
        for i in range(-n, n + 1):
            x = i * spacing + (j % 2) * spacing / 2.0
            y = j * dy
            if math.hypot(x, y) <= keep_radius + 1e-9:
                pts.append((round(x, 6), round(y, 6)))
    pts.sort(key=lambda q: (math.hypot(*q), math.atan2(q[1], q[0])))
    return pts


def directional_certificate(
    points: Sequence[Point],
    region_radius: float = REGION_RADIUS,
    r_min: float = R_MIN,
    n_radial: int = 40,
    n_angular: int = 240,
    n_boundary: int = 1440,
    chunk: int = 2048,
) -> dict:
    """**确定性**定向覆盖证书（不依赖随机抽样）。

    判定条件与 :func:`directional_coverage_ok` 相同：对域内任意 G，近旁点
    （距离 <= r_min）相对 G 的方位角最大角隙必须严格 < 180°。
    差别在于取样方式是**确定性极坐标网格 + 圆边界加密环**，因此可以作为
    论文中的"保证性"论据，而不是只给出一个抽样通过率。

    返回 ``{"ok", "max_gap_deg", "max_gap_rad", "n_samples", "fail"}``：
    ``fail`` 为违反条件的样本数（必须为 0），``max_gap_deg`` 是最大角隙
    （越小越稳健，180° 即临界）。
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[0] < 3:
        return {"ok": False, "max_gap_deg": 180.0, "max_gap_rad": math.pi,
                "n_samples": 0, "fail": 1}

    rs = np.linspace(0.0, region_radius, int(n_radial) + 1)
    ts = np.linspace(0.0, 2.0 * math.pi, int(n_angular), endpoint=False)
    rr, tt = np.meshgrid(rs, ts, indexing="ij")
    grid = np.column_stack([rr.ravel() * np.cos(tt.ravel()),
                            rr.ravel() * np.sin(tt.ravel())])
    tb = np.linspace(0.0, 2.0 * math.pi, int(n_boundary), endpoint=False)
    bnd = np.column_stack([region_radius * np.cos(tb), region_radius * np.sin(tb)])
    samples = np.vstack([grid, bnd])

    worst = 0.0
    fails = 0
    for lo in range(0, len(samples), int(chunk)):
        g = samples[lo:lo + int(chunk)]
        delta = pts[None, :, :] - g[:, None, :]
        dist = np.hypot(delta[:, :, 0], delta[:, :, 1])
        near = dist <= r_min + 1e-9
        cnt = near.sum(axis=1)
        ang = np.where(near, np.arctan2(delta[:, :, 1], delta[:, :, 0]), np.nan)
        ang.sort(axis=1)                       # NaN 排在末尾，天然区分近旁点
        gaps = np.where(np.isnan(np.diff(ang, axis=1)), -np.inf,
                        np.diff(ang, axis=1))
        internal = gaps.max(axis=1)            # 近旁点内部最大角隙
        internal = np.where(cnt >= 2, internal, 0.0)
        last = np.take_along_axis(
            ang, np.clip(cnt - 1, 0, ang.shape[1] - 1)[:, None], axis=1)[:, 0]
        wrap = ang[:, 0] + 2.0 * math.pi - last        # 首尾跨界角隙
        gap = np.fmax(internal, wrap)
        gap = np.where(cnt < 3, math.pi, gap)          # 近旁点不足 3 个必失败
        fails += int(np.count_nonzero(gap >= math.pi - 1e-9))
        if gap.size:
            worst = max(worst, float(gap.max()))
    return {
        "ok": fails == 0,
        "max_gap_deg": math.degrees(worst),
        "max_gap_rad": worst,
        "n_samples": int(len(samples)),
        "fail": int(fails),
    }


def two_ring_design(
    inner_radius: float = 950.0,
    inner_count: int = 12,
    outer_radius: float = 1875.0,
    outer_count: int = 12,
    phase: float = 0.5,
) -> List[Point]:
    """通用双环网构造：中心 1 点 + 内环 + 外环（外环可错相）。"""
    pts: List[Point] = [(0.0, 0.0)]
    for radius, count, offset in ((inner_radius, inner_count, 0.0),
                                  (outer_radius, outer_count, phase)):
        for i in range(int(count)):
            a = 2.0 * math.pi * (i + offset) / int(count)
            pts.append((radius * math.cos(a), radius * math.sin(a)))
    return pts


def directional_survey_design():
    """问题4 的 **22 点定向安全网**（中心 1 + 内环 1000×9 + 外环 1875×12 同相）。

    为什么最终选它（而不是 25 点双环）——**定向覆盖不变、检测与行程更省**：
      * **证书更强/等价**：高分辨（80×720 + 7200 边界）、超高分辨、极高分辨
        （200×2160 + 21600 边界，455,760 样本）三级复核均 **fail=0**，
        最大角隙 **177.38°（裕量 2.62°）**，与 25 点双环的 177.3768° 完全一致。
      * **点数更少 ⇒ 检测动作更少**：每减少一个网点，整轮扫描就少扫一遍
        "尚未了结的频道"。22 点 vs 25 点把检测动作数压低约 12%
        （全定向 N=10：396→339 次），这正是 N=10 档降 22~39 s/源的主因。
      * **内环半径顶到 1000 m**（= R_min）：内环点自身即落在"距中心 ≤ R_min"
        的可见集合里，于是**中心附近的小角隙由内环独立兜住**，不再依赖
        "中心点 + 更密内环"；内环降到 9 点（40° 均布，弦长 684 m < 1000 m）。
      * **中心点保留**：中心对"G 恰在原点邻域"给出第三个可见点，使 |H|≥2
        更容易成立，从而让"16 源都拿到 ≥2 条示向度即提前结束扫描"更容易触发。
    实测（每档 50 例 × 3 档 × N=10/16，共 900 例，**零未清**）：

        网                全定N10 全定N16 半定N10 半定N16 全向N10 全向N16  平均
        25 点(950x12+1875x12) 745.8  462.7   701.1  386.7   645.0  253.2  532.4
        22 点(1000x9+1875x12) 706.8  454.2   676.7  388.8   622.3  260.2  518.2
        ⇒ 6 档平均 **-14.3 s/源**（N=10 档 -22~-39 s/源；全向 N=16 略升 7 s，
          仍 260 s < 300 s 目标）。

    早期淘汰记录（**勿回退**）：22 点旧方案（17.50 km 巡游）角隙 182.25° 会漏测；
    23 点旧方案 179.94° 裕量仅 0.06° 过险；24 点 950×8+1850×16 在"正上方边界源
    + 朝向水平"角隙 186.6° 漏测。**角隙必须用高分辨率确定性证书复核，
    低分辨率粗筛会给出假阳性**（本轮又踩到一次：粗筛通过的候选在 80×720 下
    fail 高达 720）。
    """
    return two_ring_design(1000.0, 9, 1875.0, 12, 0.0)
