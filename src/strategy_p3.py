"""问题3 策略：全向干扰源（10~16 个，频道未知）的自动定位与清除。

自适应"覆盖-清除"模型（问题3 专用建模）
----------------------------------------
问题3 的干扰源**全部是全向**的，这一性质允许把"探测"与"清除"合并成同一条路径，
这是本问题相对固定检测点网的根本性提速来源。

**覆盖骨干（保证性覆盖集）**：中心 + 半径 1000 m 均布环（``cover_ring_count`` 点）。
其中每隔 ``cover_probe_stride`` 个环点取一个作为"**全频道扫描点**"，它们恰好构成
"中心 + 7 个等角环点"的标准覆盖集，解析最坏覆盖距离
``sqrt(1800²+1000²−2·1800·1000·cos(π/7)) = 998.25 m < R_min = 1000 m``，
因此**"不漏测"的保证只依赖这些点**；其余环点是"几何复测点"（只复测进行中频道）。
加密环几乎不增加巡游长度（环巡游 ≈ 周长），却能让每个源被 2 个以上骨干点看到，
从而消除"只被 1 个点看到、交会几何差"的难源（实测收尾行程从 5.4 km 降到 2.2 km）。

**关键节省**：清除必须走到源附近（<=20 m），而这个"访问"本身就把以源为圆心的
1000 m 圆并入了覆盖集，于是"探测覆盖"与"清除行程"可以**共用一条路线**：
旧实现付"绕骨干一圈 ~6.2 km **再加** 折返清除 TSP ~7.4 km"（N=10 合计 13.59 km），
新实现把源按**最便宜插入**排进骨干剩余路线，离线最优仅约 9.99 km（N=10）。

**算法**（从原点出发，起始探测零移动耗时）：
  1. 沿骨干逐点前进；到"全频道扫描点"就探测**全部未决频道**（覆盖 + 发现新源），
     到"几何复测点"只复测"进行中"（已有 1~2 条示向度）的频道（空频道不测，代价极低）；
  2. 对已"可信定位"的源（楔形交区域最小覆盖圆半径 <= pursue_mec_limit），
     按 ``Δ = |a−ĝ| + |ĝ−b| − |a−b|`` 取其最省的剩余路段插入路线，
     到达后**只做该频道的清除动作**（不再重复检测其它频道，覆盖已由扫描点保证）；
  3. 骨干走完即探测完备；仍有余留频道交严格兜底
     ``_resolve_channel``（交会收缩 + 27 m 网格清除），保证 100% 清除。

固定检测点网仍然保留：显式传入 ``survey_points`` 时走 ``_run_fixed_design``
（问题4 的定向安全网即复用该分支，因其覆盖条件含"朝向"维度，必须走满全网点）。

策略不依赖任何真值信息；时间由模拟器统一计（本地 kinematics 可交叉校验）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import (
    CHANNELS,
    CLEAR_RADIUS,
    JAMMER_NUM_RANGE,
    REGION_RADIUS,
    R_MAX,
    R_MIN,
)
from .coverage import seven_point_design, worst_case_radius
from .geometry import (
    angle_diff,
    distance,
    localization_region,
    polygon_area,
    polygon_centroid,
    polygon_diameter_calipers,
    min_enclosing_circle,
    unit_vector,
)
from .routing import two_opt_tour

Point = Tuple[float, float]

__all__ = ["StrategyP3", "StrategyResult", "bearing_least_squares",
           "coverage_samples", "coverage_profile"]

# 覆盖判据用的确定性采样网（极坐标网格 + 圆边界加密环），按参数缓存，
# 避免每个案例重复构造 ~2.8 万个采样点。
_COVERAGE_SAMPLE_CACHE: Dict[Tuple[float, int, int, int], np.ndarray] = {}


def coverage_samples(
    region_radius: float = REGION_RADIUS,
    n_radial: int = 72,
    n_angular: int = 360,
    n_boundary: int = 1440,
) -> np.ndarray:
    """覆盖判据的确定性采样点集 (M, 2)。

    径向 72 段（步长 25 m）、角度 360 段（半径 1800 m 处弧长 31.4 m），
    再加一圈 1440 点的边界加密环。相邻采样点的最大间距约 40 m，
    故"采样最大值"比真值最多低 20 m；用于覆盖完备性诊断时按 R_min 留足裕量判断。
    """
    key = (float(region_radius), int(n_radial), int(n_angular), int(n_boundary))
    cached = _COVERAGE_SAMPLE_CACHE.get(key)
    if cached is not None:
        return cached
    rs = np.linspace(0.0, float(region_radius), int(n_radial) + 1)
    ts = np.linspace(0.0, 2.0 * math.pi, int(n_angular), endpoint=False)
    rr, tt = np.meshgrid(rs, ts, indexing="ij")
    grid = np.column_stack([rr.ravel() * np.cos(tt.ravel()),
                            rr.ravel() * np.sin(tt.ravel())])
    tb = np.linspace(0.0, 2.0 * math.pi, int(n_boundary), endpoint=False)
    bnd = np.column_stack([float(region_radius) * np.cos(tb),
                           float(region_radius) * np.sin(tb)])
    samples = np.vstack([grid, bnd])
    _COVERAGE_SAMPLE_CACHE[key] = samples
    return samples


def coverage_profile(visited: Sequence[Point],
                     region_radius: float = REGION_RADIUS) -> Tuple[float, Point]:
    """返回 ``(最坏最近距离, 最深处采样点)``。

    最坏最近距离 = ``max_{X ∈ 圆域} min_{p ∈ 已访问} |X − p|``（采样估计）；
    最深处即达到该值的采样点，用作"去补哪一个洞"的目标。
    """
    samples = coverage_samples(region_radius)
    if not visited:
        return float("inf"), (0.0, 0.0)
    pts = np.asarray(visited, dtype=float)
    d = np.sqrt(((samples[:, None, :] - pts[None, :, :]) ** 2).sum(axis=2))
    nearest = d.min(axis=1)
    idx = int(np.argmax(nearest))
    return float(nearest[idx]), (float(samples[idx, 0]), float(samples[idx, 1]))


def bearing_least_squares(
    records: Sequence[Tuple[Point, float]],
    region_radius: float = REGION_RADIUS,
) -> Optional[Point]:
    """用示向中心线的正交距离最小二乘估计目标位置。

    对第 i 条示向线，令法向量 n_i=(-sin(beta_i), cos(beta_i))，求解
    ``min_x sum_i [n_i^T (x-p_i)]^2``。与直接取有界楔形交的最小覆盖圆
    圆心相比，该估计不会被 ``bbox`` 的人工截断边界拖向远处。最后投影到
    题设目标圆域，避免少量近共线观测产生物理上不可能的远端估计。
    """
    if len(records) < 2:
        return None
    a00 = a01 = a11 = z0 = z1 = 0.0
    for (px, py), bearing in records:
        rad = math.radians(float(bearing))
        nx, ny = -math.sin(rad), math.cos(rad)
        rhs = nx * px + ny * py
        a00 += nx * nx
        a01 += nx * ny
        a11 += ny * ny
        z0 += nx * rhs
        z1 += ny * rhs
    det = a00 * a11 - a01 * a01
    if det <= 1e-8:
        return None
    x = (z0 * a11 - z1 * a01) / det
    y = (a00 * z1 - a01 * z0) / det
    radius = math.hypot(x, y)
    if radius > region_radius:
        scale = region_radius / radius
        x, y = x * scale, y * scale
    return (x, y)


@dataclass
class StrategyResult:
    cleared: int = 0
    total: int = 0
    virtual_time: float = 0.0
    measures: int = 0
    clears: int = 0
    failed_clears: int = 0
    move_distance: float = 0.0          # 累计移动距离 (m)，用于区分"移动/动作"耗时
    unresolved: List[int] = field(default_factory=list)

    @property
    def clearance_ratio(self) -> float:
        return self.cleared / self.total if self.total else 1.0

    @property
    def mean_clear_time(self) -> float:
        return self.virtual_time / self.cleared if self.cleared else float("inf")

    @property
    def mean_move_distance(self) -> float:
        return self.move_distance / self.cleared if self.cleared else float("inf")


class StrategyP3:
    """问题3 的搜索-定位-清除策略。"""

    def __init__(
        self,
        survey_points: Optional[Sequence[Point]] = None,
        clear_diameter_threshold: float = 2.0 * CLEAR_RADIUS,
        max_homing_steps: int = 25,
    ) -> None:
        # 默认走**自适应"覆盖-清除"**路径（问题3 全向专用）；
        # 显式传入 survey_points 时走固定检测点网路径（问题4 定向安全网需要）。
        self.survey_points = list(survey_points) if survey_points else []
        self.clear_diameter_threshold = float(clear_diameter_threshold)
        self.max_homing_steps = int(max_homing_steps)
        # 清除个数达到该值即可提前结束（问题4 总数上限 16）
        self.max_total: float = float("inf")
        # ---- 自适应覆盖参数（仅 _run_adaptive 使用）----
        # 覆盖骨干：中心 + 半径 1000 m 的 cover_ring_count 点均布。
        # 解析最坏覆盖距离 = sqrt(1800²+a²−2·1800·a·cos(π/k))：
        #   k=7  时 998.25 m
        #   k=14 时，每隔 2 取一个（共 7 点，等角 2π/7）仍是 998.25 m ——
        # 即**加密环并不改变覆盖保证**，只增加"几何复测点"（见 _probe）。
        self.cover_ring_radius = 1000.0
        self.cover_ring_count = 14
        self.cover_probe_stride = 2
        # 实际到访的"几何复测点"个数（7 = 全部；实测 4 个即够：N=16 由 248.5→245.6 s、
        # 检测 147.6→137.6 次；再减少（3/0）反而变差，因为更多源被迫留到收尾折返）。
        self.probe_keep = 4
        # 干扰源总数上限（题给 10~16）：清满即必定清空，可提前结束
        self.max_jammers = int(JAMMER_NUM_RANGE[1])
        # 只有楔形交最小覆盖圆半径 <= 该值时才把它作为追踪目标（否则继续沿骨干
        # 探测、等几何条件变好再前往），避免带着坏估计白跑并导致清除失败。
        self.pursue_mec_limit = 160.0
        # 细长区域源（近共线、MEC 半径超限）不排进路线：交给收尾统一处理。
        # 实测强行插入会让 N=10 劣化 357→473 s、行程 13→17 km（形心不可信）。
        # 单次"顺路插入"允许的额外路程上限（实测 2500 m 最优：N=10/12/16 分别
        # 366/329/267 → 357/328/256 s）；超过就留给收尾阶段的统一 TSP。
        self.insert_limit = 2500.0
        # regret 插入：只有当"延迟到收尾的代价 − 现在插入的代价"超过该阈值才插入
        self.insert_min_regret = 0.0
        # 覆盖点剪枝阈值（"剪后仍 <= cover_limit 才算安全"）。
        # 删 1 个环点后，即使清除点完美填补缺口，剩余 6 环点的 5 个 51.4° 间隙最坏
        # 仍是 998.25 m（7 环点固有值），故阈值必须略高于 998.25 才剪得动；
        # 998.5 + 采样误差(~0.1 m) < 1000，安全裕量 ~1.4 m。
        self.cover_limit = 998.5
        # ---- 「覆盖 discharge」开关（默认开启）----
        # 清除点做一次"只测零观测频道"的全频道扫描 ⇒ 升级为合法覆盖点 ⇒ 剪掉冗余环点。
        self.inplace_sweep_after_clear = True
        self.prune_coverage_points = True
        self.inplace_sweep_min_gain = 300.0
        # 只在"复扫能换来剪点"时才复扫（见 _inplace_would_help）
        self.inplace_sweep_targeted = True
        # 迭代逼近阈值：推进到该距离以内后，1° 示向度误差的横向偏差只有
        # 382·tan(1°)= 6.7 m（累计 3° 时约 20 m = 清除半径），此时按示向度
        # 直接清除是可靠的（参考"迭代逼近"思路）。
        self.approach_threshold = 382.0
        # 侧移复测的偏移量与开关（问题3 全向源适用；见 _tighten_and_clear）
        self.shift_step = 250.0
        # 侧移复测的**多档偏移**：定向源的可测半平面有限，一次 250 m 侧移常直接
        # 走出可见区（实测 no_signal），而 30~120 m 的小侧移既留在可见区内、又能
        # 与原有示向度形成足够视差，把长条形定位区域收紧到可网格清除的尺度。
        self.shift_steps = (250.0, 120.0, 60.0, 30.0)
        # 沿示向度射线做 1-D 扫描的步长与范围（见 _ray_scan_clear 的保证性推导）
        self.ray_scan_step = 18.0
        self.ray_scan_range = 600.0
        # _corridor_like 的判据参数（决定是否启用 1-D 射线扫描）
        self.corridor_min_diameter = 120.0
        self.corridor_ratio = 4.0
        # 示向度交会区域的裁剪外接框（源必在 1800 m 圆域内）；调大可复现历史行为
        self.region_bbox = 1.0e6
        # 是否允许把"细长区域源"（近共线、MEC 半径 > pursue_mec_limit）排进路线。
        # False = 拒插，留给收尾统一处理（实测更省，见 _source_target 注释）
        self.insert_sliver_sources = False
        self.tighten_shift = True
        # 仅 1 条示向度时的"切向复测探测点"参数（见 _source_target）：
        # 垂直于视线切向偏移 500 m，到源的距离 <= sqrt(800²+500²)=943 m < R_min，
        # 因此**必定仍在接收范围内**；同时与视线成约 90°，视差充足
        # （测距误差 ≈ d²·tanε/b ≈ 26 m），随后 27 m 网格即可保证性收口。
        self.probe_angle_deg = 90.0
        self.probe_distance_m = 500.0
        # 近场保证性网格的面积上限：楔形交区域是细长条，面积小则网格点数很少，
        # 按面积（而不是直径）触发可以把"斜长但很窄"的区域也纳入保证性收口。
        self.grid_clear_near_area = 60000.0
        # 允许"顺路插入"所需的最少示向度条数。保持 2：单条示向度源在覆盖扫描期间
        # 极多（刚扫 1~2 个点就一大堆），立即追击会破坏环形巡游结构（实测 21~31 km）。
        # 它们交给收尾阶段的追击处理（见 _resolve_channel），那时覆盖扫描已完成、
        # 只剩少数边界源，追击才划算。
        self.insert_min_observations = 2
        # 单条示向度时，沿视线前进的追击步长（保守小步，避免越过真值后漏掉）
        self.pursuit_step = 150.0
        self.pursuit_max_steps = 20
        # 1 条示向度时的"预测性牵引复测点"：沿示向度前进 + 侧向视差
        self.pursuit_side = 300.0
        # 定位区域最小覆盖圆半径超过该值 ⇒ "定位差"，先测后清（省一次必失败的 clear，
        # 且近场示向度高视差，最小二乘重估后偏差压到几米内）。定位好则先清（省 6 s 检测）。
        self.measure_first_mec = 40.0
        # 自适应观测：2 条示向度且 MEC <= obs_mec_limit 即视为定位足够（不再要第 3 条）
        self.adaptive_obs = True
        self.obs_mec_limit = 30.0
        # 阶段3（区域网格清除）参数
        self.max_resolve_rounds = 6
        self.grid_clear_max_diameter = 1200.0
        self.probe_angles_deg = (20.0, -20.0, 0.0)
        # 探测步长：由远及近。必须包含 <= CLEAR_RADIUS 量级的小步长，
        # 否则当"唯一可见点离源极近"（例如 86 m）时，沿示向度探测会直接
        # **越过源**（越过即进入定向源的背向半平面 ⇒ 全部 no_signal），
        # 拿不到第 2 条示向度就永远无法交会定位（曾导致 22 点网 1/300 未清）。
        self.probe_steps_m = (450.0, 225.0, 110.0, 55.0, 25.0)
        # 保证性网格间距：格点间距 s 时，区域 P 内任一点到最近格点距离 <= s/√2，
        # 取 24 m 得 17.0 m < 20 m 清除半径（原为 27 m，最坏 19.1 m 只剩 0.9 m 裕量，
        # 实测出现"格点恰好差 2~3 m 清不掉"的失败）。
        self.grid_spacing = 24.0
        # 仍失败时再补一遍**半格偏移**格点（等效间距 s/2，最坏 8.5 m），
        # 用于兜住"定位区域 P 边缘真值"与浮点临界。
        self.grid_offset_pass = True
        self.grid_max_points = 400
        # 近场保证性网格：楔形交区域直径小于该值时直接布 27 m 网格收口
        self.grid_clear_near_diameter = 180.0
        # 三条示向线可消除两线近共线造成的长尾误差；试验中首次清除命中率
        # 由约 78% 提升到 91% 以上，而检测动作数仅小幅增加。
        self.observations_target = 3
        self.direct_attempt_max_radius = 180.0
        self.fast_ray_step = 35.0
        self.fast_ray_max_distance = 90.0
        # 估计点邻域微网格步长（见 _local_grid_clear）
        self.local_grid_step = 24.0

    # ------------------------------------------------------------------ #
    # 主流程（按是否给定固定检测点网分派）
    # ------------------------------------------------------------------ #
    def run(self, transport, channel_order=None, do_enter=True):
        """问题3 主入口：默认自适应"覆盖-清除"，给定 survey_points 时走固定网。"""
        if self.survey_points:
            return self._run_fixed_design(transport, channel_order, do_enter)
        return self._run_adaptive(transport, channel_order, do_enter)

    # ------------------------------------------------------------------ #
    # 自适应"覆盖-清除"（问题3 默认路径）
    # ------------------------------------------------------------------ #
    def _run_adaptive(self, transport, channel_order=None, do_enter=True):
        """覆盖骨干 + 动态路径重规划：把"探测覆盖"与"清除行程"合并成一条路线。

        做法（每次只走一步，然后按最新信息重规划）：
          1. **覆盖骨干**：中心 + 半径 1000 m 七均布（解析最坏覆盖距离 998.25 m），
             它是一个"保证性覆盖集"，走完即保证任意全向源都已被探测到；
          2. 在当前位置用 **2-opt 重规划** ``[当前位置] + 剩余骨干点 + 已定位源的
             估计点``，取巡游的第一站作为下一步；
          3. 走到骨干点 → 对该点做一次全频道探测（覆盖 + 发现新源）；
             走到源估计点 → 只做该频道的清除动作（**不再重复检测其它频道**，
             覆盖已由骨干保证，省下每个源访问点 ~10 个空频道的检测动作）；
          4. 骨干走完且无待清除源时结束；失败/仅 1 条示向度的频道交严格兜底。

        为什么快：旧实现付"绕骨干一圈 ~6.2 km + 折返清除 TSP ~7.4 km"；
        本实现付一条把源顺路插进骨干的合并巡游（离线最优约 9.99 km，N=10），
        并且不在源访问点重复检测空频道。
        """
        result = StrategyResult()
        if do_enter:
            transport.enter()
        channels = list(channel_order) if channel_order else list(CHANNELS)
        bearings = {ch: [] for ch in channels}
        cleared = set()
        attempted: set = set()   # 已顺路清除尝试过的频道（每频道至多一次）
        scheduled: set = set()   # 已排进路线的频道（避免重复插入）

        origin = self._current_position(transport)
        ring = seven_point_design(self.cover_ring_radius, self.cover_ring_count)
        # "全频道扫描点"：每隔 stride 取一个。stride=2 且 count=14 时，
        # 被选中的 7 个点恰好等角（2π/7）分布 —— 就是标准 7 点覆盖集，
        # 解析最坏覆盖距离 998.25 m < R_min，因此**不漏测的保证只依赖它们**。
        sweep_points = {ring[i] for i in range(0, len(ring), self.cover_probe_stride)}
        probe_candidates = [p for i, p in enumerate(ring)
                            if i % self.cover_probe_stride]
        keep = min(int(self.probe_keep), len(probe_candidates))
        if keep <= 0:
            probe_points: List[Point] = []
        elif keep >= len(probe_candidates):
            probe_points = list(probe_candidates)
        else:                       # 均匀抽稀（保留几何分布最散的那几个）
            step = len(probe_candidates) / float(keep)
            probe_points = [probe_candidates[int(k * step)] for k in range(keep)]
        backbone = two_opt_tour([origin] + [p for p in ring
                                            if p in sweep_points or p in probe_points])
        # 路线元素：(点, 频道或 None, 是否全频道扫描)
        route: List[Tuple[Point, Optional[int], bool]] = [
            (p, None, p in sweep_points or p == origin) for p in backbone]
        visited: List[Point] = []
        self._pruned_at = -1

        # 起点探测：不产生任何移动耗时，直接拿下 1000 m 内的所有源
        self._sweep(transport, origin, channels, bearings, cleared, result)
        visited.append(self._current_position(transport))
        # 合法覆盖点集合：只有在这里面做过**全频道扫描**的位置才能计入覆盖判据
        swept: List[Point] = [tuple(self._current_position(transport))]

        index = 0
        while index < len(route) - 1 and len(cleared) < self.max_jammers:
            # 把新定位到的源插进"剩余路线"最省的那一段
            self._schedule_sources(route, index, channels, bearings,
                                   cleared, attempted, scheduled)
            # 注意：这里**不做**"剪枝剩余覆盖点"。曾试验过按覆盖判据删除冗余网点，
            # 但那会漏清 —— 因为复测点只测"进行中"频道、源清除点也只测目标频道，
            # 它们不能充当"让每个频道都被测过一遍"的覆盖点；唯一合法的覆盖点是
            # 做过**全频道扫描**的点（7+1 个），而对圆域覆盖而言它们一个都不能少
            # （k=6 需要 a>=1122 m、k=7 需要 a>=997 m，删任一点即破覆盖）。
            point, channel, full = route[index + 1]
            if channel is None:
                if full:
                    # 覆盖扫描点：全频道探测（覆盖 + 发现新源）
                    self._sweep(transport, point, channels, bearings,
                                cleared, result)
                    swept.append(tuple(self._current_position(transport)))
                else:
                    # 几何复测点：只复测"进行中"频道（见 _probe）
                    self._probe(transport, point, channels, bearings,
                                cleared, result)
            else:
                # 顺路清除：只做该频道动作，不重复检测其它频道（覆盖由覆盖点保证）
                scheduled.discard(channel)
                attempted.add(channel)
                if self._fast_clear(transport, channel, point,
                                    bearings[channel], result):
                    cleared.add(channel)
                    # 「发现即清除 + 原地复扫」：清除点若离已扫点足够远，就地做一次
                    # 全频道扫描，把它升级为**合法覆盖点**，为后续剪枝创造合法性。
                    if self.inplace_sweep_after_clear:
                        here = tuple(self._current_position(transport))
                        gap = min((distance(here, q) for q in swept),
                                  default=float("inf"))
                        if gap >= self.inplace_sweep_min_gain and (
                                not self.inplace_sweep_targeted
                                or self._inplace_would_help(route, index,
                                                            swept, here)):
                            self._discharge_sweep(transport, here, channels,
                                                  bearings, cleared, result)
                            swept.append(tuple(self._current_position(transport)))
            visited.append(self._current_position(transport))
            # 合法剪枝：已扫点 ∪ 剩余覆盖点仍能覆盖全域时，删掉冗余的剩余覆盖点
            if self.prune_coverage_points:
                self._prune_coverage(route, index, swept)
            index += 1

        self._finalize_channel_clearing(transport, channels, bearings, cleared, result)

        transport.exit()
        result.cleared = len(cleared)
        result.total = self._evaluate_total(transport)
        result.virtual_time = self._virtual_time(transport)
        result.move_distance = self._move_distance(transport)
        return result

    def _inplace_would_help(self, route, index: int, swept, here) -> bool:
        """在 ``here`` 复扫，能否换来**剪掉至少一个剩余覆盖点**？

        discharge 一次要测 4~14 个零观测频道（24~84 s），被剪的覆盖点省下
        "走到它 + 它那里的全频道扫描"（≈240~260 s）。只有能换剪点才复扫。
        """
        base = list(swept) + [here]
        for i in [i for i, (p, ch, full) in enumerate(route)
                  if i > index + 1 and ch is None and full]:
            plan = [q for j, (q, ch, full) in enumerate(route)
                    if j > index + 1 and ch is None and full and j != i]
            if self._covers(base + plan):
                return True
        return False

    def _covers(self, points) -> bool:
        """覆盖判据：点集对 1800 m 圆域的最坏最近距离 <= ``cover_limit``。

        覆盖圆域问题时，**最坏点必在边界**（圆域边界或两覆盖圆交点；交点处
        min-distance <= 相邻覆盖点，恒 <= 边界角中点的最坏值），故只采样边界即可
        （确定性等角 14400 点，弧长 0.79 m，误差 <0.4 m），比内部网格快一个量级。
        """
        pts = np.asarray(list(points), dtype=float)
        th = np.linspace(0.0, 2.0 * math.pi, 14400, endpoint=False)
        bnd = 1800.0 * np.column_stack([np.cos(th), np.sin(th)])
        d = np.linalg.norm(bnd[:, None, :] - pts[None, :, :], axis=2)
        return float(d.min(axis=1).max()) <= self.cover_limit

    def _prune_coverage(self, route, index: int, swept) -> None:
        """删除"已冗余"的剩余覆盖点（**合法性由 swept 保证**）。

        覆盖义务：每个**零观测频道**的源位置都要落在某个"做全频道扫描"的点的
        1000 m 内。sweep 点与 discharge 点都做了全频道扫描（discharge 测当时
        所有零观测频道），故 ``swept``（起点 + sweep 点 + discharge 点）都是合法
        覆盖点。已有示向度的频道发现义务已尽，不再需要覆盖。

        只剪 ``index`` 之后的剩余覆盖点；剪之前用"剪后仍 <= cover_limit"验证；
        每次剪一个，反复直到不能再剪。
        """
        while True:
            candidates = [i for i, (p, ch, full) in enumerate(route)
                          if i > index + 1 and ch is None and full]
            if not candidates:
                return
            removed = False
            for i in candidates:
                plan = [q for j, (q, ch, full) in enumerate(route)
                        if j > index + 1 and ch is None and full and j != i]
                if self._covers(list(swept) + plan):
                    route.pop(i)
                    removed = True
                    break
            if not removed:
                return

    def _schedule_sources(self, route, index, channels, bearings, cleared,
                          attempted, scheduled) -> None:
        """把"已可信定位"的源按**最便宜插入**排进剩余路线。

        为什么用插入而不是每步重规划：覆盖骨干本身已是一条约 6.2 km 的近最优
        巡游，把源插到它最省的那一段上能保住骨干结构，从而兑现"探测与清除共用
        一条路径"的收益 —— 离线最优 N=10 约 9.99 km，而"绕骨干 + 折返清除"
        需要 13.59 km。逐步 2-opt 重规划会让巡游在骨干与源之间反复改道，
        实测行程反而涨到 13.6 km。

        ``insert_limit`` 限制单次插入的额外路程；代价过大的源不硬插，
        留给收尾阶段的统一 TSP 处理（那里没有后续骨干约束，更省）。
        """
        while True:
            seq = [route[index][0]] + [p for p, _, _ in route[index + 1:]]
            best = None
            for ch in channels:
                if ch in cleared or ch in attempted or ch in scheduled:
                    continue
                records = bearings[ch]
                if len(records) < self.insert_min_observations:
                    continue
                estimate = self._source_target(records, seq[0])
                if estimate is None:
                    continue
                for k in range(len(seq) - 1):
                    a, b = seq[k], seq[k + 1]
                    cost = (distance(a, estimate) + distance(estimate, b)
                            - distance(a, b))
                    if cost > self.insert_limit:
                        continue
                    if best is None or cost < best[0]:
                        best = (cost, index + 1 + k, ch, estimate)
            if best is None:
                return
            _, position, channel, estimate = best
            route.insert(position, (estimate, channel, False))
            scheduled.add(channel)
            self._improve_remaining(route, index)

    @staticmethod
    def _improve_remaining(route, index: int, rounds: int = 4) -> None:
        """对**未走过的剩余路线**做 2-opt + Or-opt 局部改进（严格只接受变短的移动）。

        "最便宜插入"是启发式：逐个把源插进剩余路线后难免留下绕行段。实测问题3
        N=16：实际 14.84 km，而**同一到访点集**的最优巡游只要 12.61 km —— 2.23 km
        （≈446 s、28 s/源）花在排序上。纯 2-opt 只能回收 0.06 km，因为它的弱点恰好
        是"把一段长绕行中的点搬到别处"；这里补上 **Or-opt**（整段 1~3 点搬迁），
        这才是对这种"顺路插入 + 绕行"结构的有效算子。

        安全性：只对 ``route[index:]``（当前位置 + 尚未到访的点）做**严格变短**的
        移动，所有点仍然都会被走到，因此**覆盖保证不受影响**（覆盖只依赖"走过"）；
        也不会像"逐步全规划"那样在骨干与源之间来回改道（那种做法实测劣化到 13.6 km）。
        """
        m = len(route)
        if m - index < 4:
            return
        # --- 2-opt ---
        for _ in range(rounds):
            improved = False
            for i in range(index, m - 2):
                a, b = route[i][0], route[i + 1][0]
                for j in range(i + 2, m - 1):
                    c, d = route[j][0], route[j + 1][0]
                    if (distance(a, b) + distance(c, d)
                            > distance(a, c) + distance(b, d) + 1e-9):
                        route[i + 1:j + 1] = route[i + 1:j + 1][::-1]
                        improved = True
            if not improved:
                break
        # --- Or-opt：把长度 1~3 的连续段移到别处 ---
        for seg_len in (1, 2, 3):
            moved = True
            while moved and m - index > seg_len + 2:
                moved = False
                for s in range(index + 1, m - seg_len):
                    seg = route[s:s + seg_len]
                    prev, nxt = route[s - 1][0], route[s + seg_len][0]
                    gain_out = (distance(prev, seg[0][0])
                                + distance(seg[-1][0], nxt)
                                - distance(prev, nxt))
                    if gain_out <= 1e-9:
                        continue
                    rest = route[:s] + route[s + seg_len:]
                    best = None
                    for t in range(index, len(rest) - 1):
                        u, v = rest[t][0], rest[t + 1][0]
                        add_in = (distance(u, seg[0][0]) + distance(seg[-1][0], v)
                                  - distance(u, v))
                        if best is None or add_in < best[0]:
                            best = (add_in, t + 1)
                    if best is not None and best[0] < gain_out - 1e-9:
                        route[:] = rest[:best[1]] + seg + rest[best[1]:]
                        moved = True
                        break

    def _probe(self, transport, point: Point, channels, bearings, cleared,
               result: StrategyResult) -> int:
        """几何复测点：只复测"进行中"的频道（已有示向度但不足观测目标）。

        这些点不承担覆盖义务，因此**不检测零观测频道**（空频道），
        代价仅几次检测动作；却能把"只被 1 个骨干点看到"的难源补足到可交会定位，
        使其清除动作能并进主路线，避免收尾阶段再折返一趟。
        """
        done = 0
        for ch in channels:
            if ch in cleared:
                continue
            records = bearings[ch]
            if not records or self._observations_complete(records):
                continue
            if self._beyond_reachable(point, records):
                continue
            resp = transport.measure(point, ch)
            result.measures += 1
            done += 1
            kind = resp.get("measure_result")
            if kind == "direction":
                bearings[ch].append((point, float(resp["svd_deg"])))
            elif kind == "near" and self._try_clear(transport, ch, point, result):
                cleared.add(ch)
        return done

    def _source_target(self, records, here: Optional[Point] = None) -> Optional[Point]:
        """给出"顺路即可清除"的目标点（决定每个源插到路线哪一段）。

        分三种情形，覆盖从"定位很准"到"只有 1 条示向度"的全部已发现源：

        1. 示向度 >= 2 且楔形交区域最小覆盖圆半径 <= ``pursue_mec_limit``：
           取**最小二乘交会点**（比最小覆盖圆圆心更准：实测首次清除命中率
           62.5%→64.2%、每源 clear 次数 39.3→37.3、时间 362.2→360.7 s）；
           最小二乘失败时回退到圆心 —— 真值必在区域内，圆心误差不超过该半径。
        2. 示向度 >= 2 但区域偏大（典型：两条示向度近共线 ⇒ 区域是**细长条**）：
           此时**不能用形心** —— 近共线的交会区域被裁剪到 ``bbox``，形心可能落在
           几千米之外（实测曾出现 6350 km 的荒谬行程）。改用"区域内离当前位置
           最近的点"：它落在细长条靠近我们的一端，走过去的行程最小，到了再复测
           一次就能把条带收紧（这本质上是**沿视线逼近**，而非绕到远端）。
        3. **只有 1 条示向度**：无法交会定距。**重写**：不再做 90° 切向复测点
           （那是一次几百米的绕行，且实测把主路线拉到 23.4 km），而是沿视线前进
           ``pursuit_step`` 米——这是一个"朝源走"的点，行程是必付的；到达后复测
           一次拿到更近的示向度，再沿视线追击（_ray_scan_clear 指数+二分）收敛。
        """
        if len(records) >= 2:
            poly = self._region_of(records)
            if poly:
                center, radius = min_enclosing_circle(poly)
                if radius <= self.pursue_mec_limit:
                    lsq = bearing_least_squares(records)
                    if lsq is not None and distance(lsq, center) <= radius:
                        return (float(lsq[0]), float(lsq[1]))
                    return (float(center[0]), float(center[1]))
                # 细长区域：默认**不排程**（交给收尾的统一处理）。
                # 依据：细长条的形心不可信（近共线时甚至可达千米级偏差）；
                # 实测"强行插入细长条源"会让 N=10 劣化 357→514 s。
                if not self.insert_sliver_sources:
                    return None
                centroid = polygon_centroid(poly)
                return (float(centroid[0]), float(centroid[1]))
        base, bearing = records[-1]
        ux, uy = unit_vector(bearing)
        sx, sy = unit_vector(bearing + 90.0)
        return (base[0] + self.pursuit_step * ux + self.pursuit_side * sx,
                base[1] + self.pursuit_step * uy + self.pursuit_side * sy)

    def _region_of(self, records) -> List[Point]:
        """示向度交会区域（裁剪到目标区域外接框，见 ``region_bbox``）。"""
        return localization_region([p for p, _ in records],
                                   [b for _, b in records],
                                   bbox=float(self.region_bbox))

    @classmethod
    def _nearest_point_in(cls, poly: Sequence[Point],
                          here: Optional[Point]) -> Point:
        """凸多边形内离 ``here`` 最近的点；``here`` 为空时退回形心。

        先判内部：``here`` 在 P 内就直接用它（零行程）。否则沿各边取最近点。
        """
        if here is None:
            centroid = polygon_centroid(poly)
            return (float(centroid[0]), float(centroid[1]))
        if cls._inside_convex(poly, here):
            return (float(here[0]), float(here[1]))
        best, best_d = None, float("inf")
        n = len(poly)
        for i in range(n):
            a, b = poly[i], poly[(i + 1) % n]
            k = cls._project_on_segment(here, a, b)
            d = distance(here, k)
            if d < best_d:
                best, best_d = k, d
        if best is None:
            centroid = polygon_centroid(poly)
            return (float(centroid[0]), float(centroid[1]))
        return (float(best[0]), float(best[1]))

    @staticmethod
    def _project_on_segment(q: Point, a: Point, b: Point) -> Point:
        """q 在线段 ab 上的投影点。"""
        ax, ay = a
        dx, dy = b[0] - ax, b[1] - ay
        denom = dx * dx + dy * dy
        if denom <= 1e-12:
            return (float(ax), float(ay))
        t = ((q[0] - ax) * dx + (q[1] - ay) * dy) / denom
        t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
        return (float(ax + t * dx), float(ay + t * dy))

    def _confident_target(self, records) -> Optional[Point]:
        """返回"可信追踪目标"：楔形交区域的**最小覆盖圆圆心**。

        真值必落在楔形交区域 P 内，因此该圆心到真值的距离不超过 P 的最小覆盖圆
        半径。只有该半径 <= ``pursue_mec_limit`` 时才值得专门跑一趟：
        半径小 ⇒ 估计可信 ⇒ 一击命中率高，且失败时的补救代价有界。
        半径过大（例如两条近共线示向度形成的细长平行四边形）则返回 None，
        继续沿覆盖骨干探测，等更多示向度把区域收缩后再前往。
        """
        poly = localization_region([p for p, _ in records],
                                   [b for _, b in records])
        if not poly:
            return None
        center, radius = min_enclosing_circle(poly)
        if radius > self.pursue_mec_limit:
            return None
        return (float(center[0]), float(center[1]))

    @staticmethod
    def _point_segment_distance(q: Point, a: Point, b: Point) -> float:
        """点 q 到线段 ab 的最短距离。"""
        ax, ay = a
        bx, by = b
        qx, qy = q
        dx, dy = bx - ax, by - ay
        denom = dx * dx + dy * dy
        if denom <= 1e-12:
            return distance(q, a)
        t = ((qx - ax) * dx + (qy - ay) * dy) / denom
        t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
        return distance(q, (ax + t * dx, ay + t * dy))

    @classmethod
    def _distance_to_polygon(cls, q: Point, poly: Sequence[Point]) -> float:
        """点 q 到凸多边形 poly 的最短距离（内部返回 0）。"""
        if cls._inside_convex(poly, q):
            return 0.0
        n = len(poly)
        return min(cls._point_segment_distance(q, poly[i], poly[(i + 1) % n])
                   for i in range(n))

    def _region_mec(self, records) -> float:
        """定位区域（楔形交）的最小覆盖圆半径；示向度 <2 或区域为空返回 inf。"""
        if len(records) < 2:
            return float("inf")
        poly = self._region_of(records)
        if not poly:
            return float("inf")
        _, radius = min_enclosing_circle(poly)
        return float(radius)

    def _beyond_reachable(self, point: Point, records) -> bool:
        """该频道在 ``point`` 处是否**必然 no_signal**（严格判据，可安全跳过检测）。

        依据：单次测向误差 <= 1°（题给），故真值必落在全部示向度决定的**楔形交区域
        P** 内；而有效接收半径 r_eff <= R_MAX = 1500 m（题给上界）。于是
        ``dist(point, P) > 1500`` ⇒ ``dist(point, G) > r_eff`` ⇒ 测了也是 no_signal。
        跳过它能省下一次 5 s 检测 + 至多 1 s 切换，**且不损失任何保证**（该次检测
        无论如何都不携带信息）。
        """
        poly = localization_region([p for p, _ in records],
                                   [b for _, b in records])
        if not poly:
            return False
        return self._distance_to_polygon(point, poly) > R_MAX

    def _discharge_sweep(self, transport, point: Point, channels, bearings, cleared,
                         result: StrategyResult) -> int:
        """清除点的**discharge 扫描**：只测零观测频道（覆盖义务），返回检测次数。

        覆盖义务：一个频道若真有源但尚未被发现，源必落在某个"会做全频道扫描"的
        点 1000 m 内。discharge 点在这里做全频道扫描后，就承担了这个义务。

        **为何只测零观测频道**：obs>=1 的频道已经被发现（源距某个观测点 <= r_eff，
        发现义务已尽），额外的观测只改善定位、不是发现义务，测不测都行。所以这里
        只测零观测频道，把成本压到 4~14 次检测（24~84 s），而不是全 20 频道。
        对零观测频道而言，sweep 点与 discharge 点都测过它，故二者都是合法覆盖点。
        """
        done = 0
        for ch in channels:
            if ch in cleared or bearings[ch]:
                continue
            resp = transport.measure(point, ch)
            result.measures += 1
            done += 1
            kind = resp.get("measure_result")
            if kind == "direction":
                bearings[ch].append((tuple(point), float(resp["svd_deg"])))
            elif kind == "near" and self._try_clear(transport, ch, point, result):
                cleared.add(ch)
        return done

    def _sweep(self, transport, point: Point, channels, bearings, cleared,
               result: StrategyResult) -> int:
        """移动到 ``point`` 并对所有未决频道做一次探测，返回实际检测次数。

        首个 /measure 会把机器人带到 ``point``（模拟器按位置差计费），因此
        "前往该点"与"在该点探测"是同一条指令；若没有任何频道需要检测，
        则机器人不会移动（调用方用 ``_current_position`` 记录真实到访位置）。
        """
        done = 0
        for ch in channels:
            if ch in cleared or self._observations_complete(bearings[ch]):
                continue
            # 严格无损的检测削减：该点必定收不到信号则跳过（见 _beyond_reachable）
            if bearings[ch] and self._beyond_reachable(point, bearings[ch]):
                continue
            resp = transport.measure(point, ch)
            result.measures += 1
            done += 1
            kind = resp.get("measure_result")
            if kind == "direction":
                bearings[ch].append((point, float(resp["svd_deg"])))
            elif kind == "near" and self._try_clear(transport, ch, point, result):
                cleared.add(ch)
        return done

    # ------------------------------------------------------------------ #
    # 固定检测点网路径（问题4 定向安全网复用）
    # ------------------------------------------------------------------ #
    def _run_fixed_design(self, transport, channel_order=None, do_enter=True):
        """按预定检测点网逐点扫描，再统一清除，并以有界楔形法兜底。

        实测对比（全定向 N=16，25 例）："逐点扫描 + 收尾 TSP" 478.6 s；
        改成"把每个已定位源最便宜插入剩余路线"反而 **519.8 s**（且出现未清除）——
        因为定向源可见点极少、发现得晚，插入位置离路线很远；当源较密时
        "收尾阶段对估计点做精确 TSP"本来就是最优的。故问题4 保留本路径，
        插入机制只在问题3（全向、发现早）使用。
        """
        result = StrategyResult()
        if do_enter:
            transport.enter()
        channels = list(channel_order) if channel_order else list(CHANNELS)
        bearings = {ch: [] for ch in channels}
        cleared = set()
        # 每次 run 独立的状态袋（机会式清除的次数记录等），避免跨次运行串味
        state: Dict[str, object] = {}
        self.early_stop_bearings = getattr(self, "early_stop_bearings", 2)

        tour = self._tour_order()
        for index, p in enumerate(tour):
            self._sweep(transport, p, channels, bearings, cleared, result)
            # 提前结束扫描：``early_stop_bearings`` 条示向度齐备的源数达到总数上限 16
            # 即停。默认 2（交会定位）；若收尾对 1 条示向度的源有可靠兜底（如沿示向度
            # 追击），可降为 1 以更早结束。
            if sum(1 for ch in channels
                   if len(bearings[ch]) >= self.early_stop_bearings) >= self.max_total:
                break
            # 机会式清除：离开当前扫描点之前，允许子类顺路清除已定位的频道
            if index + 1 < len(tour):
                self._opportunistic_clear(transport, channels, bearings, cleared,
                                          result, p, tour[index + 1], state)
                if len(cleared) >= self.max_total:
                    break

        self._finalize_channel_clearing(transport, channels, bearings, cleared, result)

        transport.exit()
        result.cleared = len(cleared)
        result.total = self._evaluate_total(transport)
        result.virtual_time = self._virtual_time(transport)
        result.move_distance = self._move_distance(transport)
        return result

    def _finalize_channel_clearing(self, transport, channels, bearings, cleared,
                                   result: StrategyResult) -> None:
        """收尾：对已产生示向度但仍未清除的频道做统一快速清除 + 严格兜底。"""
        candidates = {}
        for ch in channels:
            if ch in cleared or len(bearings[ch]) < 2:
                continue
            center = bearing_least_squares(bearings[ch])
            if center is not None:
                candidates[ch] = center
            else:
                target = self._source_target(
                    bearings[ch], self._current_position(transport))
                if target is not None:
                    candidates[ch] = target

        for ch in self._channel_tour(candidates, self._current_position(transport)):
            if self._fast_clear(transport, ch, candidates[ch], bearings[ch], result):
                cleared.add(ch)

        # 只处理"已产生过示向度"的未决频道：频道空间有 20 个，而干扰源只有
        # 10~16 个，因此**零观测频道绝大多数本来就没有干扰源**（空频道）。
        # 覆盖判据保证"若某频道真有源，全域已被探测过 ⇒ 它必已被发现"，
        # 故跳过零观测频道是正确行为；切勿为其添加"重走检测点"的兜底
        # （实测每例会白走近 10 km，耗时劣化数倍）。
        pending = [ch for ch in channels if ch not in cleared and bearings[ch]]
        while pending and len(cleared) < self.max_total:
            cur = self._current_position(transport)
            ch = min(pending, key=lambda c: distance(cur, self._resolve_anchor(bearings[c])))
            pending.remove(ch)
            if self._resolve_channel(transport, ch, bearings[ch], result):
                cleared.add(ch)
            else:
                result.unresolved.append(ch)

        transport.exit()
        result.cleared = len(cleared)
        result.total = self._evaluate_total(transport)
        result.virtual_time = self._virtual_time(transport)
        result.move_distance = self._move_distance(transport)
        return result

    # ------------------------------------------------------------------ #
    # 机会式清除钩子（问题3 默认不启用；问题4 重写为"顺路清除"）
    # ------------------------------------------------------------------ #
    def _opportunistic_clear(self, transport, channels, bearings, cleared,
                             result: StrategyResult, current: Point,
                             next_point: Point, state: Dict[str, object]) -> None:
        """在离开扫描点 ``current`` 前顺路清除已定位频道。

        问题3 的 8 点扫描行程短、快速清除已在扫描后统一用精确开放路径完成，
        故基类此钩子为空实现（无副作用，不改变问题3 行为）。
        """
        return None

    @staticmethod
    def _move_distance(transport) -> float:
        """累计移动距离 (m)：读机器人时钟历史（Mock 与真实客户端均有）。"""
        clock = getattr(transport, "clock", None)
        if clock is None:
            clock = getattr(getattr(transport, "client", None), "clock", None)
        history = getattr(clock, "history", None) or []
        return float(sum(getattr(cost, "distance_m", 0.0) for _, cost in history))

    def _fast_clear(self, transport, channel: int, estimate: Point,
                    records: List[Tuple[Point, float]], result: StrategyResult) -> bool:
        """统计估计直接清除；失败时按"迭代逼近"在近场保证性收口。

        三级收口（由便宜到昂贵，第一级成功即返回）：
          1. 直接在估计点清除（楔形交最小覆盖圆圆心，误差 <= pursue_mec_limit）；
          2. 修订估计后再试一次（用估计点的新示向度更新最小二乘）；
          3. **保证性收口**：若楔形交区域 P 直径已很小，按 27 m 间距在 P 内布点
             逐一清除 —— 因真值必在 P 内且间距 <= 20√2，必有一点落在清除半径内；
             否则沿示向度在估计点**两侧**按 382 m 阈值内的步长逼近清除
             （1° 误差在 382 m 处的横向偏差仅 6.7 m，远小于 20 m 清除半径）。
        """
        # 自适应：定位好（MEC 小）先清；定位差（MEC 大）先测后清 ——
        # 定位差的源直接清必失败（偏差可达上百米），先测一次拿高视差近场示向度，
        # 最小二乘重估后偏差压到几米内再清，省掉那次必失败的 clear。
        if self._region_mec(records) <= self.measure_first_mec:
            if self._try_clear(transport, channel, estimate, result):
                return True
        m = transport.measure(estimate, channel)
        result.measures += 1
        kind = m.get("measure_result")
        if kind == "near":
            return self._try_clear(transport, channel, estimate, result)

        bearing = None
        if kind == "direction":
            bearing = float(m["svd_deg"])
            records.append((estimate, bearing))
            revised = bearing_least_squares(records)
            if revised is not None and distance(revised, estimate) <= self.fast_ray_max_distance:
                if self._try_clear(transport, channel, revised, result):
                    return True

        # 阶段2.5-沿最后示向度射线的 1-D 扫描收口（对付"定向源沿波束轴观测"的病态
        # 楔形交；解析保证命中，见 _ray_scan_clear）。
        # **仅在真正病态时启用**：定位区域沿示向度方向被拉得很长（细长走廊）才需要
        # 1-D 扫描；否则该阶段命中率低（实测调用 9.8 次仅成功 3.0 次），白花检测。
        if records and self._corridor_like(records):
            bpoint, bbearing = records[-1]
            if self._ray_scan_clear(transport, channel, bpoint, bbearing, result):
                return True
        # 阶段2.6-估计点邻域**微网格**：楔形交直径常只有几十米、真值在 20 m 清除半径
        # 边缘"擦边"漏掉，此时做 250 m 级侧移（一次来回 500 m）极不划算；3×3 微网格
        # 单步 <= 24 m、最坏 9 点约 40 s 即可补救（见 _local_grid_clear）。
        if self._local_grid_clear(transport, channel, estimate, result):
            return True
        # 阶段3-保证性收口：区域小而窄时直接布 27 m 网格（真值必在 P 内）
        if len(records) >= 2:
            poly = localization_region([p for p, _ in records],
                                       [b for _, b in records])
            if poly:
                diam, _, _ = polygon_diameter_calipers(poly)
                if (diam <= self.grid_clear_near_diameter
                        or polygon_area(poly) <= self.grid_clear_near_area):
                    return self._grid_clear(transport, channel, poly, result)
        # 阶段3.5-沿示向度**短距**双向逼近（span<=90 m）：失败时最多走 180 m 来回，
        # 比原 210 m 级（420 m 来回）省一半；远距离的病态情形已由阶段2.5 的射线
        # 扫描负责，因此把这一级的射程收紧。
        if bearing is not None and self._ray_clear_both(transport, channel, estimate,
                                                        bearing, result):
            return True
        # 阶段4-**最后手段**：侧移复测（垂直示向度偏移，代价最大：一个来回数百米）
        if self.tighten_shift and self._tighten_and_clear(transport, channel,
                                                          records, result):
            return True
        return False

    def _corridor_like(self, records) -> bool:
        """定位区域是否是"沿示向度方向的细长走廊"（此时才需要 1-D 射线扫描）。

        判据：楔形交区域的**直径**远大于其在垂直示向度方向上的宽度，
        即 ``diam >= corridor_ratio * width``，或区域直径本身超过
        ``corridor_min_diameter``（几十米以上就意味着沿轴测距不可靠）。
        典型触发场景：定向源沿其波束轴被观测、或多条示向度近共线。
        """
        if len(records) < 2:
            return True
        poly = localization_region([p for p, _ in records],
                                   [b for _, b in records])
        if not poly:
            return True
        diam, _, _ = polygon_diameter_calipers(poly)
        if diam >= self.corridor_min_diameter:
            return True
        _, bearing = records[-1]
        ux, uy = unit_vector(bearing + 90.0)
        proj = [q[0] * ux + q[1] * uy for q in poly]
        width = max(proj) - min(proj)
        return diam >= self.corridor_ratio * max(width, 1e-6)

    def _local_grid_clear(self, transport, channel: int, center: Point,
                          result: StrategyResult) -> bool:
        """估计点邻域的 3×3 **微网格**收口（廉价、先于大范围侧移）。

        楔形交的直径通常只有几十米，而估计点（最小覆盖圆圆心 / 形心）未必正中，
        真值常落在 20 m 清除半径**边缘**：此时直接做 250 m 级侧移要一个来回 500 m，
        极不划算。改为在估计点 ±``local_grid_step`` 内布 3×3 网格（单步 <= 24 m，
        最坏 9 次清除约 40 s、行程 <= 170 m），命中率高且代价低；真正病态的情形
        （细长走廊）交给阶段 2.5 的射线扫描与阶段 4 的侧移兜底。
        """
        step = self.local_grid_step
        offs = (0.0, step, -step)
        pts = [(center[0] + dx, center[1] + dy) for dx in offs for dy in offs]
        current = self._current_position(transport)
        pts.sort(key=lambda q: distance(current, q))
        for q in pts:
            if self._try_clear(transport, channel, q, result):
                return True
        return False

    def _pursuit_clear(self, transport, channel: int, base: Point, bearing: float,
                       result: StrategyResult) -> bool:
        """**重瞄追击**：沿示向度向源逼近，每步用当前实测示向度**重瞄**方向。

        这是"1 条示向度"源的定位手段：把"走到源附近"的必付行程本身当作定位。
        关键在**重瞄**——每步沿当前实测示向度走，横向误差只相对于"当前剩余距离"
        （<= 0.0175×剩余距离），随收敛趋零；这正是之前固定射线追击在 d>1140 m 时
        横向偏差超 20 m 而漏过的原因。对**全向源**（问题3 全是全向），朝源走距离
        单调递减 ⇒ 必先"direction"后"near"；一旦示向度翻转 180°（越过真值）就二分。
        """
        cur = base
        b = bearing
        step = self.pursuit_step
        for _ in range(self.pursuit_max_steps):
            q = self._point_along(cur, b, step)
            if self._try_clear(transport, channel, q, result):
                return True
            resp = transport.measure(q, channel)
            result.measures += 1
            kind = resp.get("measure_result")
            if kind == "near":
                return self._try_clear(transport, channel, q, result)
            if kind == "direction":
                nb = float(resp["svd_deg"])
                if abs(angle_diff(nb, b)) > 90.0:
                    # 越过了真值：源在 cur 与 q 之间，二分
                    return self._pursuit_bisect(transport, channel, cur, q, result)
                cur = q
                b = nb
                step = min(step * 1.3, 300.0)     # 越走越近，可加大步长
            else:                                  # no_signal：全向源罕见，退回小步
                step = max(step / 2.0, 20.0)
                if step <= 20.0:
                    return False
        return False

    def _pursuit_bisect(self, transport, channel: int, a: Point, b: Point,
                        result: StrategyResult) -> bool:
        """在"未越过点 a"与"已越过点 b"之间二分找源（源在 ab 附近）。"""
        for _ in range(14):
            mid = ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
            if self._try_clear(transport, channel, mid, result):
                return True
            resp = transport.measure(mid, channel)
            result.measures += 1
            kind = resp.get("measure_result")
            if kind == "near":
                return self._try_clear(transport, channel, mid, result)
            if kind == "direction":
                nb = float(resp["svd_deg"])
                ux, uy = unit_vector(nb)
                toward_a = (a[0] - mid[0]) * ux + (a[1] - mid[1]) * uy
                if toward_a > 0.0:
                    b = mid                    # 源在 a 侧
                else:
                    a = mid                    # 源在 b 侧
            else:
                a = mid                        # no_signal：源在另一侧
            if distance(a, b) <= 10.0:
                break
        return self._try_clear(transport, channel,
                               ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0), result)

    def _ray_scan_clear(self, transport, channel: int, base: Point, bearing: float,
                        result: StrategyResult) -> bool:
        """沿**最后一条示向度射线**按固定步长扫描并逐点试清除（保证性收口）。

        设最后一条示向度取自 ``base``、报告值 ``bearing``。题目保证单次测向误差
        <= 1°，故真值必落在"以 base 为起点、方向 bearing 的射线两侧各 1° 的楔形"内；
        射线上距 base 为 t 的点到真值的**横向偏差 <= t·tan1° = 0.0175t**。因此沿射线
        以步长 s 扫描时，命中条件为 ``sqrt((s/2)² + (0.0175t)²) <= 20``：取
        s=18 m、t<=600 m，最坏 ``sqrt(81+110) = 13.8 m < 20 m``，**必然命中**。

        这专门解决"**定向源沿其波束轴被观测**"的病态情形：此时三条示向度几乎同向，
        楔形交退化为沿轴 1000 m 的细长走廊，测距完全不可用，而本方法把"2-D 定位"降级
        为"1-D 扫描"，反而必然收敛。每两步复测一次示向度，一旦发现已越过真值即停止
        延伸，避免无谓长走；成功则直接返回。
        """
        step = self.ray_scan_step
        t = step
        while t <= self.ray_scan_range + 1e-9:
            q = self._point_along(base, bearing, t)
            if self._try_clear(transport, channel, q, result):
                return True
            if abs(t / (2.0 * step) - round(t / (2.0 * step))) < 1e-9:
                resp = transport.measure(q, channel)
                result.measures += 1
                kind = resp.get("measure_result")
                if kind == "near":
                    return self._try_clear(transport, channel, q, result)
                if kind == "direction" and \
                        abs(angle_diff(float(resp["svd_deg"]), bearing)) > 90.0:
                    return False                 # 已越过真值：前方不会再有
            t += step
        return False

    def _tighten_and_clear(self, transport, channel: int,
                           records: List[Tuple[Point, float]],
                           result: StrategyResult) -> bool:
        """**侧移复测**收紧定位区域后保证性清除。

        三角交会的测距精度取决于**基线（视差）**：两点相距 d 时，测距相对误差
        约为 ``d²·tan(ε)/b``——基线 b 不够大时测距误差可达百米量级，这正是
        "顺路插入"失败、被迫折返的原因。这里在最后一条示向度的垂直方向上偏移
        一小段复测（代价约 250 m + 1 次检测），新示向度与原有示向度形成足够
        视差，把楔形交区域直径压到 180 m 以内后，用 27 m 间距网格保证性清除。

        仅在近场使用（偏移后源仍在接收范围内的概率高），因此适合问题3 的全向源；
        对定向源若偏移后收不到信号，会自动跳过（不增加错误动作）。
        """
        if not records:
            return False
        base, bearing = records[-1]
        for offset in [mag * sgn for mag in self.shift_steps for sgn in (1.0, -1.0)]:
            probe = self._point_along(base, bearing + 90.0, offset)
            resp = transport.measure(probe, channel)
            result.measures += 1
            kind = resp.get("measure_result")
            if kind == "near":
                return self._try_clear(transport, channel, probe, result)
            if kind != "direction":
                continue
            records.append((probe, float(resp["svd_deg"])))
            poly = localization_region([p for p, _ in records],
                                       [b for _, b in records])
            if not poly:
                continue
            diam, _, _ = polygon_diameter_calipers(poly)
            if diam <= self.grid_clear_near_diameter:
                return self._grid_clear(transport, channel, poly, result)
        return False

    def _ray_clear_both(self, transport, channel: int, estimate: Point,
                        bearing: float, result: StrategyResult) -> bool:
        """沿示向度在估计点**两侧**按 35 m 步长试清除。

        只向正方向扫描会漏掉"估计距离偏远"的情形（楔形交区域沿示向度方向被
        拉长，测距误差有正有负）；双向扫描把两侧都覆盖，实测显著降低清除失败率，
        从而避免失败的频道被推到收尾阶段再跑一趟。
        """
        step = self.fast_ray_step
        span = max(self.fast_ray_max_distance, self.pursue_mec_limit)
        offset = step
        while offset <= span + 1e-9:
            for sign in (1.0, -1.0):
                q = self._point_along(estimate, bearing, sign * offset)
                if self._try_clear(transport, channel, q, result):
                    return True
            offset += step
        return False

    def _ray_recover(self, transport, channel, origin, max_distance, records, result):
        """At an estimate, turn 2-D uncertainty into a covered 1-D bearing ray."""
        m = transport.measure(origin, channel)
        result.measures += 1
        kind = m.get("measure_result")
        if kind == "near":
            return self._try_clear(transport, channel, origin, result)
        if kind != "direction":
            return False
        bearing = float(m["svd_deg"])
        records.append((origin, bearing))
        lateral = max_distance * math.sin(math.radians(1.0))
        if lateral >= CLEAR_RADIUS:
            return False
        half = math.sqrt(max(1.0, CLEAR_RADIUS ** 2 - lateral ** 2))
        spacing = max(8.0, 1.9 * half)
        d = spacing
        while d <= max_distance + spacing:
            q = self._point_along(origin, bearing, d)
            if self._try_clear(transport, channel, q, result):
                return True
            d += spacing
        return False

    def _observations_complete(self, records):
        """观测是否足够（默认 3 条；自适应：2 条 + MEC 足够小即可，第 3 条只在该拿时才拿）。

        依据：2 条示向度已能把定位区域的最小覆盖圆半径压到 ``obs_mec_limit`` 以内，
        再配 _fast_clear 的"到点先测后清"，第三条示向度省下的清除成本已不抵再测
        一次的 6 s。仅在 MEC 仍大时才继续拿第 3 条。
        """
        if len(records) >= self.observations_target:
            return True
        if self.adaptive_obs and len(records) == 2:
            return self._region_mec(records) <= self.obs_mec_limit
        return False

    @staticmethod
    def _resolve_anchor(records):
        p, b = records[-1]
        ux, uy = unit_vector(b)
        return (p[0] + 350.0 * ux, p[1] + 350.0 * uy)

    @staticmethod
    def _channel_tour(candidates, start):
        """Exact Held-Karp open TSP for at most 16 targets."""
        ids = list(candidates)
        n = len(ids)
        if not n:
            return []
        dp = {}
        for i in range(n):
            dp[(2 ** i, i)] = (distance(start, candidates[ids[i]]), -1)
        for mask in range(1, 2 ** n):
            for last in range(n):
                state = (mask, last)
                if state not in dp:
                    continue
                base = dp[state][0]
                for nxt in range(n):
                    bit = 2 ** nxt
                    if mask & bit:
                        continue
                    ns = (mask | bit, nxt)
                    cost = base + distance(candidates[ids[last]], candidates[ids[nxt]])
                    if ns not in dp or cost < dp[ns][0]:
                        dp[ns] = (cost, last)
        full = 2 ** n - 1
        last = min(range(n), key=lambda j: dp[(full, j)][0])
        order = []
        mask = full
        while last >= 0:
            order.append(ids[last])
            prev = dp[(mask, last)][1]
            mask -= 2 ** last
            last = prev
        order.reverse()
        return order

    def _try_clear(self, transport, channel: int, point: Point,
                   result: StrategyResult) -> bool:
        r = transport.clear(point, channel)
        result.clears += 1
        if r.get("clear_result") == "success":
            return True
        result.failed_clears += 1
        return False

    def _tour_order(self) -> List[Point]:
        """扫描顺序：先中心，再按环上角度顺序（六边形环行程为最优量级）。"""
        pts = list(self.survey_points)
        if not pts:
            return []
        center = min(pts, key=lambda p: p[0] ** 2 + p[1] ** 2)
        rest = [p for p in pts if p != center]
        rest.sort(key=lambda p: math.atan2(p[1], p[0]))
        return [center] + rest

    # ------------------------------------------------------------------ #
    # 阶段3：交会定位 + 区域网格清除（保证性收敛）
    # ------------------------------------------------------------------ #
    def _resolve_channel(self, transport, channel: int,
                         observations: List[Tuple[Point, float]],
                         result: StrategyResult) -> bool:
        """为一个频道补足示向度 -> 定位区域 -> 网格清除。

        依据：楔形交区域 P 必含真值 G（≤1° 有界误差）；在 P 内以
        间距 <= 20*sqrt(2) = 28.3 m 布清除点，则必有一点落在 G 的 20 m 清除半径内，
        因此该过程必然成功（问题3/4 通用；对定向源而言，清除与朝向无关，
        即使“看不到”也能清）。

        若 P 过大，则沿最后一条示向度推进再测，以新示向度缩小 P；
        沿示向度向源逼近时，定向源的覆盖条件保持成立（内积保号）。
        """
        records: List[Tuple[Point, float]] = list(observations)
        for _ in range(self.max_resolve_rounds):
            if len(records) >= 2:
                poly = self._region_of(records)
                if poly:
                    diam, _, _ = polygon_diameter_calipers(poly)
                    if diam <= 2.0 * CLEAR_RADIUS:
                        if self._try_clear(transport, channel,
                                           polygon_centroid(poly), result):
                            return True
                    if diam <= self.grid_clear_max_diameter:
                        if self._grid_clear(transport, channel, poly, result):
                            return True
            if not self._extend_observation(transport, channel, records, result):
                return False
        return False

    def _extend_observation(self, transport, channel: int,
                            records: List[Tuple[Point, float]],
                            result: StrategyResult) -> bool:
        """获取新的示向度：在基准点周围做射线扇形探测。

        为何不只沿示向度推进：
          * 沿向推进虽然保持覆盖（内积保号），但两条示向度近似平行，
            交会出来的区域是细长条，定位效果差；
          * 而当年±1° 误差下沿向推进较远时可能越出定向覆盖，
            故用“垂直方向±20° / 沿向”的射线扇形探测，既保证交会角度，
            又在失联时自动退到更保守的沿向探测。
        """
        if not records:
            return False
        base_point, base_bearing = records[-1]
        for angle_off in self.probe_angles_deg:
            for step in self.probe_steps_m:
                point = self._point_along(base_point, base_bearing + angle_off, step)
                m = transport.measure(point, channel)
                result.measures += 1
                kind = m.get("measure_result")
                if kind == "near":
                    return self._try_clear(transport, channel, point, result)
                if kind == "direction":
                    records.append((point, float(m["svd_deg"])))
                    return True
        return False

    def _grid_clear(self, transport, channel: int, poly: Sequence[Point],
                    result: StrategyResult) -> bool:
        """在定位区域 P 内布格点逐一尝试清除（间距 <= 20*sqrt(2)，保证命中）。"""
        pts = self._grid_points_in_polygon(poly, self.grid_spacing)
        # 补入形心 / 顶点 / 边中点（小区域时网格可能为空）
        pts = list(pts) + [polygon_centroid(poly)]
        n = len(poly)
        for i in range(n):
            a, b = poly[i], poly[(i + 1) % n]
            pts.append(a)
            pts.append(((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0))
        if not pts:
            return False
        current = self._current_position(transport)
        pts.sort(key=lambda q: distance(current, q))
        for q in pts[:self.grid_max_points]:
            if self._try_clear(transport, channel, q, result):
                return True
        if not self.grid_offset_pass:
            return False
        # 第二轮：半格偏移格点。两套格点合起来等效间距 s/2，最坏离真值
        # <= s/(2√2) ≈ 8.5 m，远超清除半径裕量，兜住"真值贴 P 边界/浮点临界"。
        off = self.grid_spacing / 2.0
        shifted = [(x + off, y + off) for x, y in pts]
        shifted = [q for q in shifted if self._inside_convex(poly, q, tol=1e-6)]
        shifted.sort(key=lambda q: distance(current, q))
        for q in shifted[:self.grid_max_points]:
            if self._try_clear(transport, channel, q, result):
                return True
        return False

    @staticmethod
    def _grid_points_in_polygon(poly: Sequence[Point], spacing: float) -> List[Point]:
        xs = [q[0] for q in poly]
        ys = [q[1] for q in poly]
        out: List[Point] = []
        x = min(xs)
        while x <= max(xs):
            y = min(ys)
            while y <= max(ys):
                if StrategyP3._inside_convex(poly, (x, y)):
                    out.append((x, y))
                y += spacing
            x += spacing
        return out

    @staticmethod
    def _inside_convex(poly: Sequence[Point], q: Point, tol: float = 1e-9) -> bool:
        n = len(poly)
        sign = 0
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            cross = (x2 - x1) * (q[1] - y1) - (y2 - y1) * (q[0] - x1)
            cur = 0 if abs(cross) <= tol else (1 if cross > 0 else -1)
            if cur == 0:
                continue
            if sign == 0:
                sign = cur
            elif sign != cur:
                return False
        return True

    @staticmethod
    def _current_position(transport) -> Point:
        clock = getattr(transport, "clock", None)
        pos = getattr(clock, "position", None)
        return (float(pos[0]), float(pos[1])) if pos else (0.0, 0.0)

    @staticmethod
    def _point_along(origin: Point, bearing_deg: float, dist: float) -> Point:
        ux, uy = unit_vector(bearing_deg)
        return (origin[0] + dist * ux, origin[1] + dist * uy)

    # ------------------------------------------------------------------ #
    # 统计适配（兼容 MockSimulator 与 RobotClient）
    # ------------------------------------------------------------------ #
    @staticmethod
    def _virtual_time(transport) -> float:
        if hasattr(transport, "virtual_time"):
            return float(getattr(transport, "virtual_time"))
        clock = getattr(transport, "clock", None)
        return float(getattr(clock, "virtual_time", 0.0))

    @staticmethod
    def _evaluate_total(transport) -> int:
        if hasattr(transport, "total_jammers"):
            return int(transport.total_jammers)
        sim = getattr(transport, "sim", None)
        if sim is not None and hasattr(sim, "total_jammers"):
            return int(sim.total_jammers)
        return 0
