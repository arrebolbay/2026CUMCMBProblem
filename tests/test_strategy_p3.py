"""问题3 策略测试：在 Mock 模拟器上的端到端清除能力。"""

import math

import pytest

from src.config import CHANNELS, REGION_RADIUS, R_MIN
from src.coverage import seven_point_design
from src.mock_simulator import MockSimulator, random_case
from src.strategy_p3 import StrategyP3


# ------------------------------ 默认布点 ------------------------------ #
def test_default_design_is_dense_ring_with_covering_subset():
    """问题3 默认走自适应模型：骨干是"加密环"，其中每 stride 个点取一个，
    这些"全频道扫描点"必须恰好构成标准 7 点覆盖集（不漏测保证只依赖它们）；"""
    strategy = StrategyP3()
    assert strategy.survey_points == [], "默认应为自适应路径（不给定固定检测点网）"
    ring = seven_point_design(strategy.cover_ring_radius, strategy.cover_ring_count)
    assert len(ring) == strategy.cover_ring_count + 1, "seven_point_design 含中心点"
    sweep = [ring[i] for i in range(0, len(ring), strategy.cover_probe_stride)]
    assert len(sweep) == 8, "全频道扫描点 = 中心 + 7 个等角环点（覆盖集）"
    assert sweep[0] == (0.0, 0.0)
    radii = [math.hypot(x, y) for x, y in sweep[1:]]
    assert all(abs(r - 1000.0) < 1e-9 for r in radii)
    angles = sorted(math.degrees(math.atan2(y, x)) % 360.0 for x, y in sweep[1:])
    gaps = [b - a for a, b in zip(angles, angles[1:])] + [360.0 - angles[-1] + angles[0]]
    assert all(abs(g - 360.0 / 7.0) < 1e-9 for g in gaps), "必须是等角 2π/7 分布"


def test_adaptive_backbone_covers_region_analytically_and_numerically():
    """覆盖性：解析 998.25 m + 高密极坐标网格复核，均 < R_min。"""
    from src.coverage import worst_case_radius

    design = seven_point_design(1000.0, 7)
    worst = math.sqrt(REGION_RADIUS ** 2 + 1000.0 ** 2
                      - 2.0 * REGION_RADIUS * 1000.0 * math.cos(math.pi / 7))
    assert worst == pytest.approx(998.2545144156829, rel=1e-12)
    assert worst < R_MIN
    assert worst_case_radius(design, REGION_RADIUS) < R_MIN


def test_schedule_sources_uses_cheapest_insertion():
    """最便宜插入：源应被插到剩余路线中"额外路程最小"的那一段。"""
    strategy = StrategyP3()
    route = [((0.0, 0.0), None, True), ((0.0, 1000.0), None, True)]
    # 三条示向度交于 (0, 500) —— 正好在 (0,0)→(0,1000) 这一段的中点
    bearings = {7: [((500.0, 500.0), 180.0), ((-500.0, 500.0), 0.0),
                    ((0.0, 1000.0), 270.0)]}
    strategy._schedule_sources(route, 0, [7], bearings, set(), set(), set())
    assert len(route) == 3
    assert route[1][1] == 7, "应插在 0→1 段中间"
    assert math.dist(route[1][0], (0.0, 500.0)) < 5.0, route[1][0]


def test_schedule_sources_skips_when_insert_limit_exceeded():
    """插入代价超过阈值时不得硬插（留给收尾 TSP）。"""
    strategy = StrategyP3()
    strategy.insert_limit = 100.0
    # 估计点在路线外 999 m 处，插入代价远大于 100 m
    route = [((0.0, 0.0), None, True), ((0.0, 1000.0), None, True)]
    bearings = {7: [((500.0, 1500.0), 180.0), ((-500.0, 1500.0), 0.0),
                    ((0.0, 2000.0), 270.0)]}
    strategy._schedule_sources(route, 0, [7], bearings, set(), set(), set())
    assert len(route) == 2
    assert all(ch is None for _, ch, _ in route)


def test_adaptive_run_visits_backbone_without_jammers():
    """无干扰源时：走完覆盖骨干、无清除、无未解决、仍正常退出。"""
    sim = MockSimulator([], seed=1)
    strategy = StrategyP3()
    res = strategy.run(sim)
    assert res.cleared == 0
    assert res.unresolved == []
    # 8 个骨干点各做一次全频道探测
    assert res.measures >= (1 + strategy.cover_ring_count)
    assert res.virtual_time > 0.0


def test_strategy_p3_inserts_sources_on_route():
    """端到端：自适应模型应把大部分源"顺路"清除（插入率不低于 40%）。"""
    inserted_total = 0
    cases = 6
    for seed in range(cases):
        jammers = random_case(n=12, seed=seed)
        sim = MockSimulator(jammers, seed=seed)
        strategy = StrategyP3()
        original = strategy._schedule_sources

        counter = {"n": 0}

        def counting(route, index, channels, bearings, cleared, attempted,
                     scheduled, _c=counter):
            before = len(route)
            original(route, index, channels, bearings, cleared, attempted,
                     scheduled)
            _c["n"] += len(route) - before

        strategy._schedule_sources = counting
        res = strategy.run(sim)
        assert res.cleared == 12
        inserted_total += counter["n"]
    assert inserted_total >= 0.4 * 12 * cases, "顺路插入率过低，合并收益会丢失"


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_strategy_clears_all_omni_jammers(seed):
    jammers = random_case(seed=seed)
    sim = MockSimulator(jammers, seed=seed)
    res = StrategyP3().run(sim)

    assert res.total == sim.total_jammers
    assert res.cleared == sim.total_jammers, f"存在未清除: {res.unresolved}"
    assert res.clearance_ratio == 1.0
    assert res.unresolved == []
    assert res.virtual_time > 0.0
    assert res.mean_clear_time > 0.0
    assert res.move_distance > 0.0
    assert res.mean_move_distance > 0.0


def test_empty_channels_are_not_reported_unresolved():
    """频道空间 20 个而干扰源仅若干个：空频道不得被当成"未清除"或反复重扫。

    回归测试：曾错误地为"零观测频道"加了重走检测点的兜底，导致每例白走
    约 6×18 km（耗时劣化 4 倍）。自适应模型下每个骨干点做一次全频道探测，
    因此检测次数必须与"骨干点数 × 频道数"同量级。
    """
    jammers = random_case(n=10, seed=3)
    sim = MockSimulator(jammers, seed=3)
    strategy = StrategyP3()
    res = strategy.run(sim)

    assert res.cleared == 10
    assert res.unresolved == []
    empty_channels = [ch for ch in CHANNELS
                      if ch not in {j.channel for j in jammers}]
    assert len(empty_channels) == len(CHANNELS) - 10
    backbone_points = 1 + strategy.cover_ring_count      # 中心 + 环点
    sweep_points = len(range(0, strategy.cover_ring_count,
                             strategy.cover_probe_stride)) + 1
    # 覆盖扫描点做全频道探测；复测点只测"进行中"频道；不应出现"整圈重扫"的量级
    assert res.measures <= sweep_points * len(CHANNELS) + 60
    # 行程应与"骨干巡游 + 各源清除"同量级（不会出现每空频道数公里的白跑）
    assert res.move_distance < 25_000.0


def test_strategy_handles_empty_channel_space():
    """无干扰源（极端情形）：策略应正常退出且无异常。"""
    sim = MockSimulator([], seed=0)
    res = StrategyP3().run(sim)
    assert res.cleared == 0
    assert res.unresolved == []
    assert res.clearance_ratio == 1.0


def test_strategy_handles_far_and_near_jammers():
    """边界情形：干扰源在区域边缘（远）与原点附近（近）。"""
    from src.mock_simulator import Jammer

    jammers = [
        Jammer(channel=1, x=1750.0, y=0.0, r_eff=1000.0),    # 边缘 + 最小接收半径
        Jammer(channel=2, x=8.0, y=0.0, r_eff=1500.0),       # 距原点 8m
        Jammer(channel=3, x=-900.0, y=-1200.0, r_eff=1100.0),
    ]
    sim = MockSimulator(jammers, seed=11)
    res = StrategyP3().run(sim)
    assert res.cleared == 3, f"未清除: {res.unresolved}"
    assert sim.cleared_jammers == 3
