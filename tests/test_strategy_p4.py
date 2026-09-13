"""问题4 策略测试：含完全定向干扰源的端到端清除能力与定向布点条件。"""

import math

import pytest

from src.config import REGION_RADIUS, R_MIN
from src.coverage import (
    directional_certificate,
    directional_coverage_ok,
    directional_survey_design,
    seven_point_design,
    triangular_lattice_design,
    worst_case_radius,
)
from src.mock_simulator import Jammer, MockSimulator, random_case
from src.strategy_p4 import StrategyP4, estimate_direction_arc, two_opt_tour


# ------------------------------ 布点条件 ------------------------------ #
def test_seven_point_design_fails_directional_condition():
    """问题3 的 7 点方案不能满足定向覆盖（定向源可能背向所有检测点）。"""
    r = directional_coverage_ok(seven_point_design(1400.0, 6), n_samples=800, seed=1)
    assert r["ok_ratio"] < 1.0


# ------------------------------ 定向安全证书 ------------------------------ #
def test_two_ring_design_has_analytic_certificate():
    """22 点定向安全网的证书：外环含域 + 角隙充要判据零失败。"""
    design = directional_survey_design()
    assert len(design) == 22
    outer_inradius = 1875.0 * math.cos(math.pi / 12.0)   # 正十二边形内切半径
    assert outer_inradius > REGION_RADIUS, "目标圆域必须整体含于外环正十二边形内"
    # 「三角剖分全部边 < R_min」是**充分**条件：中心—内环 1000、内环边 684.04、
    # 外环边 970.57、最近交叉边 906.98 均 < 1000；但同相构型下"四边形对角线"
    # 可达 1126 m，故充分条件不再整体成立，正式保证改用**充要判据**
    # （G ∈ conv(距 G ≤ R_min 的网点) ⟺ 这些点相对 G 的最大角隙 < 180°）。
    edges = {
        "center_inner": 1000.0,
        "inner_edge": 2.0 * 1000.0 * math.sin(math.pi / 9.0),
        "outer_edge": 2.0 * 1875.0 * math.sin(math.pi / 12.0),
        "cross_edge": math.dist(
            (1000.0 * math.cos(2 * math.pi / 9), 1000.0 * math.sin(2 * math.pi / 9)),
            (1875.0 * math.cos(math.pi / 6), 1875.0 * math.sin(math.pi / 6))),
    }
    assert edges["inner_edge"] < R_MIN and edges["outer_edge"] < R_MIN, edges
    assert edges["cross_edge"] < R_MIN, edges
    # 证书函数（确定性极坐标密网 + 边界加密）必须零失败
    cert = directional_certificate(design)
    assert cert["ok"] is True, cert
    assert cert["fail"] == 0
    assert cert["max_gap_deg"] < 179.0, cert


def test_certificate_rejects_incomplete_design():
    """去掉外环后（仅中心 + 内环）定向条件必须被证书判否。"""
    design = directional_survey_design()[:10]
    cert = directional_certificate(design)
    assert cert["ok"] is False
    assert cert["fail"] > 0


def test_lattice_design_satisfies_directional_condition():
    """定向安全网满足定向覆盖（用 20000 次抽样验证）。"""
    design = directional_survey_design()
    assert len(design) == 22
    r = directional_coverage_ok(design, n_samples=20000, seed=7)
    assert r["ok_ratio"] == 1.0, r


def test_lattice_design_also_covers_omni():
    """定向安全布点同时满足全向覆盖（最坏距离 < R_min）。"""
    design = directional_survey_design()
    wc = worst_case_radius(design, REGION_RADIUS, n_interior=8000, n_boundary=8000)
    assert wc < R_MIN


def test_spacing_must_not_exceed_r_min():
    """间距超过 R_min 时定向条件不再成立（理论边界的数值印证）。"""
    coarse = triangular_lattice_design(spacing=1500.0)
    r = directional_coverage_ok(coarse, n_samples=3000, seed=11)
    assert r["ok_ratio"] < 1.0


# ------------------------------ 巡游与方向弧 ------------------------------ #
def test_two_opt_tour_is_complete_and_shorter_than_naive():
    design = directional_survey_design()
    tour = two_opt_tour([(0.0, 0.0)] + design)
    assert len(tour) == len(design) + 1
    assert set(tour) == set([(0.0, 0.0)] + design)

    def length(seq):
        return sum(math.dist(seq[i], seq[i + 1]) for i in range(len(seq) - 1))

    assert length(tour) <= length([(0.0, 0.0)] + design) + 1e-9


def test_direction_arc_estimation():
    """由有信号/无信号约束交出的可行方向弧应包含真实朝向。"""
    g = (0.0, 0.0)
    true_dir = 90.0
    detections = [(math.cos(math.radians(a)) * 500, math.sin(math.radians(a)) * 500)
                  for a in (40.0, 90.0, 140.0)]      # 在朝北半平面内
    misses = [(500.0, -300.0)]                      # 背面点：无信号
    arc = estimate_direction_arc(g, detections, misses)
    assert arc, "可行弧不应为空"
    assert any(abs(((a - true_dir + 180) % 360) - 180) <= 2.0 for a in arc)


# ------------------------------ 机会式清除（顺路清除） ------------------------------ #
class _FakeClock:
    def __init__(self, position):
        self.position = position
        self.history = []
        self.virtual_time = 0.0


class _FakeTransport:
    """最小传输桩：只实现策略实际调用的协议动作，用于确定性单测。"""

    def __init__(self, position=(0.0, 0.0), clear_success=True,
                 measure_kind="no_signal"):
        self.clock = _FakeClock((float(position[0]), float(position[1])))
        self.clear_calls = []
        self.measure_calls = []
        self._clear_success = clear_success
        self._measure_kind = measure_kind

    def measure(self, position, channel):
        self.measure_calls.append((position, channel))
        self.clock.position = (float(position[0]), float(position[1]))
        return {"measure_result": self._measure_kind, "svd_deg": None}

    def clear(self, position, channel):
        self.clear_calls.append((position, channel))
        self.clock.position = (float(position[0]), float(position[1]))
        return {"clear_result": ("success" if self._clear_success
                                 else "no_target_in_range")}

    def enter(self):
        return {"accepted": True}

    def exit(self):
        return {"accepted": True}


# 三条示向线均精确指向原点 ⇒ 最小二乘估计恰为 (0, 0)
_TRIANGULATING_RECORDS = [((100.0, 0.0), 180.0), ((0.0, 100.0), 270.0),
                          ((-100.0, 0.0), 0.0)]


def test_opportunistic_clear_respects_detour_threshold():
    """额外路程 = |cur-ĝ|+|ĝ-next|-|cur-next| = 900+100-800 = 200 m。

    阈值 150 m 时不得插入；阈值 400 m 时恰好插入一次并标记该频道已清除。
    """
    from src.strategy_p3 import StrategyResult, bearing_least_squares

    estimate = bearing_least_squares(_TRIANGULATING_RECORDS)
    assert estimate == pytest.approx((0.0, 0.0), abs=1e-9)

    for threshold, expected in ((150.0, 0), (400.0, 1)):
        transport = _FakeTransport(position=(0.0, 900.0))
        strategy = StrategyP4(insert_threshold=threshold)
        bearings = {3: list(_TRIANGULATING_RECORDS)}
        cleared, state = set(), {}
        strategy._opportunistic_clear(transport, [3], bearings, cleared,
                                      StrategyResult(), (0.0, 900.0),
                                      (0.0, 100.0), state)
        assert len(transport.clear_calls) == expected, (threshold, transport.clear_calls)
        assert (3 in cleared) is (expected == 1)


def test_opportunistic_clear_marks_cleared_and_never_repeats():
    """回归：插入成功必须立即标记 cleared，且同一频道至多插入一次。

    曾漏写 cleared.add(channel)，导致已清除频道在下一条扫描边被反复尝试，
    端到端清除率从 100% 掉到约 69%。
    """
    from src.strategy_p3 import StrategyResult

    transport = _FakeTransport(position=(0.0, 900.0))
    strategy = StrategyP4(insert_threshold=400.0)
    bearings = {3: list(_TRIANGULATING_RECORDS)}
    cleared, state = set(), {}
    result = StrategyResult()

    strategy._opportunistic_clear(transport, [3], bearings, cleared, result,
                                  (0.0, 900.0), (0.0, 100.0), state)
    assert 3 in cleared
    assert len(transport.clear_calls) == 1

    # 再次调用（模拟后续扫描边）：该频道已清除，不得再产生任何动作
    strategy._opportunistic_clear(transport, [3], bearings, cleared, result,
                                  (0.0, 900.0), (0.0, 100.0), state)
    assert len(transport.clear_calls) == 1


def test_opportunistic_clear_failure_does_not_loop():
    """插入失败（清除不中且测不到示向度）时：失败频道不得在后续扫描边重复插入。

    注意：单次 `_fast_clear` 内部会做"双向射线 + 近场网格"等多次尝试，
    因此这里比较的是"再次调用钩子是否新增动作"，而不是绝对次数。
    """
    from src.strategy_p3 import StrategyResult

    transport = _FakeTransport(position=(0.0, 900.0), clear_success=False,
                               measure_kind="no_signal")
    strategy = StrategyP4(insert_threshold=400.0)
    bearings = {3: list(_TRIANGULATING_RECORDS)}
    cleared, state = set(), {}
    result = StrategyResult()

    strategy._opportunistic_clear(transport, [3], bearings, cleared, result,
                                  (0.0, 900.0), (0.0, 100.0), state)
    attempted_once = len(transport.clear_calls)
    assert attempted_once >= 1, "失败也应至少尝试一次"
    assert 3 not in cleared
    assert result.failed_clears >= 1

    strategy._opportunistic_clear(transport, [3], bearings, cleared, result,
                                  (0.0, 900.0), (0.0, 100.0), state)
    assert len(transport.clear_calls) == attempted_once, "失败频道不得在后续扫描边重复插入"


def test_observations_complete_stops_at_three():
    """问题4 三条观测即停（减少检测动作），定位质量由严格兜底保证。"""
    strategy = StrategyP4()
    assert strategy.observations_target == 3
    assert strategy._observations_complete([]) is False
    assert strategy._observations_complete([((0.0, 0.0), 10.0)]) is False
    assert strategy._observations_complete([((0.0, 0.0), 10.0),
                                            ((1.0, 0.0), 20.0)]) is False
    assert strategy._observations_complete(list(_TRIANGULATING_RECORDS)) is True


# ------------------------------ 端到端清除 ------------------------------ #
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_strategy_p4_clears_all_directional_jammers(seed):
    jammers = random_case(seed=seed, directional_ratio=1.0)
    sim = MockSimulator(jammers, seed=seed)
    res = StrategyP4().run(sim)
    assert res.cleared == sim.total_jammers, f"未清除: {res.unresolved}"
    assert res.clearance_ratio == 1.0
    assert res.unresolved == []


def test_strategy_p4_handles_mixed_and_edge_cases():
    jammers = [
        Jammer(channel=1, x=1700.0, y=0.0, r_eff=1000.0, direction=0.0),   # 边界+背向圆心
        Jammer(channel=2, x=-1600.0, y=900.0, r_eff=1100.0),               # 全向
        Jammer(channel=3, x=10.0, y=0.0, r_eff=1500.0, direction=270.0),   # 近原点
        Jammer(channel=4, x=0.0, y=1750.0, r_eff=1300.0, direction=90.0),  # 朝外
    ]
    sim = MockSimulator(jammers, seed=42)
    res = StrategyP4().run(sim)
    assert res.cleared == 4, f"未清除: {res.unresolved}"


def test_strategy_p4_early_stops_at_16():
    """总数上限 16：清满 16 个即提前结束（不再扫描剩余频道）。"""
    jammers = random_case(n=16, seed=5, directional_ratio=0.5)
    sim = MockSimulator(jammers, seed=5)
    res = StrategyP4().run(sim)
    assert res.cleared == 16
    assert res.total == 16