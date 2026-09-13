"""Mock 模拟器物理规则测试：检测三态、定向覆盖、清除规则、计时一致性。"""

import math

import pytest

from src.geometry import angle_diff, bearing_between, distance
from src.mock_simulator import Jammer, MockSimulator, random_case


def make_sim():
    jammers = [
        Jammer(channel=1, x=300.0, y=400.0, r_eff=1500.0),              # 全向
        Jammer(channel=2, x=-500.0, y=0.0, r_eff=1500.0, direction=0.0),  # 定向朝东
        Jammer(channel=3, x=0.0, y=1700.0, r_eff=1200.0),               # 超距
    ]
    return MockSimulator(jammers, seed=3)


def test_enter_and_initial_state():
    sim = make_sim()
    sim.enter()
    assert sim.clock.position == (0.0, 0.0)
    assert sim.clock.channel == 1
    assert sim.virtual_time == 0.0


def test_measure_direction_with_pm1deg_error_and_fixed_bias():
    sim = make_sim()
    sim.enter()
    r1 = sim.measure((0.0, 0.0), 1)
    true_bearing = bearing_between((0.0, 0.0), (300.0, 400.0))
    assert r1["measure_result"] == "direction"
    assert abs(angle_diff(r1["svd_deg"], true_bearing)) <= 1.0 + 1e-9

    r2 = sim.measure((0.0, 0.0), 1)          # 同一地点重复检测：误差固定
    assert r2["svd_deg"] == pytest.approx(r1["svd_deg"])

    r3 = sim.measure((0.0, 100.0), 1)        # 不同地点：误差可不同，但仍 <=1 度
    assert abs(angle_diff(r3["svd_deg"], bearing_between((0.0, 100.0), (300.0, 400.0)))) <= 1.0


def test_measure_near_within_5m():
    sim = make_sim()
    sim.enter()
    r = sim.measure((300.0 + 3.0, 400.0), 1)
    assert r["measure_result"] == "near"
    assert r["svd_deg"] is None


def test_measure_no_signal_beyond_effective_radius():
    sim = make_sim()
    sim.enter()
    r = sim.measure((0.0, 0.0), 3)           # 距离 1700 m > r_eff 1200 m
    assert r["measure_result"] == "no_signal"


def test_directional_jammer_coverage_half_plane():
    sim = make_sim()
    sim.enter()
    behind = sim.measure((-900.0, 0.0), 2)   # 位于西侧：不在朝东覆盖角内
    assert behind["measure_result"] == "no_signal"
    front = sim.measure((0.0, 0.0), 2)       # 位于东侧且在接收半径内
    assert front["measure_result"] == "direction"
    assert abs(angle_diff(front["svd_deg"], bearing_between((0.0, 0.0), (-500.0, 0.0)))) <= 1.0


def test_clear_rules_and_channel_unchanged():
    sim = make_sim()
    sim.enter()
    sim.measure((0.0, 0.0), 5)               # 切到频道 5
    assert sim.clock.channel == 5

    miss = sim.clear((0.0, 0.0), 1)          # (0,0) 距 (300,400) 500m > 20m
    assert miss["clear_result"] == "no_target_in_range"
    assert sim.clock.channel == 5, "/clear 不应切换测向机频道"

    hit = sim.clear((310.0, 400.0), 1)       # 10m 内 -> 成功
    assert hit["clear_result"] == "success"
    assert sim.clock.channel == 5

    again = sim.clear((310.0, 400.0), 1)     # 同一源只能清除一次
    assert again["clear_result"] == "no_target_in_range"


def test_timing_matches_rules():
    sim = make_sim()
    sim.enter()
    c1 = sim.measure((300.0, 400.0), 1)["cost"]   # 500m/5 + 0切换 + 5s
    assert c1.move_s == pytest.approx(100.0)
    assert c1.switch_s == pytest.approx(0.0)
    assert c1.total_s == pytest.approx(105.0)

    c2 = sim.measure((300.0, 400.0), 2)["cost"]   # 原地换频道
    assert c2.switch_s == pytest.approx(1.0)
    assert c2.total_s == pytest.approx(6.0)

    c3 = sim.clear((300.0, 0.0), 3)["cost"]       # 400m/5 + 3s(未发现)
    assert c3.total_s == pytest.approx(83.0)
    assert sim.virtual_time == pytest.approx(194.0)


def test_random_case_counts_and_channels():
    jammers = random_case(seed=1)
    assert 10 <= len(jammers) <= 16
    channels = [j.channel for j in jammers]
    assert len(set(channels)) == len(channels)
    assert all(1 <= c <= 20 for c in channels)
    assert all(distance((0.0, 0.0), j.position) <= 1800.0 + 1e-9 for j in jammers)
    assert all(1000.0 <= j.r_eff <= 1500.0 for j in jammers)


def test_directional_ratio_produces_directional_jammers():
    jammers = random_case(n=16, seed=9, directional_ratio=1.0)
    assert all(j.direction is not None for j in jammers)
    assert all(0.0 <= j.direction < 360.0 for j in jammers)


def test_protocol_response_fields_match_spec():
    """附件2 第 5.2 / 6.2 节：响应字段与真实模拟器逐字段对齐。

    真实模拟器的每个业务响应都带 ``real_timestamp_ms``；``/enter`` 还额外返回
    三个时长字段，其中 ``remaining_real_duration_s`` 是"本局实际可用现实时间"，
    策略不得固定假定为 1200 s（见 scripts/run_simulator.py 的预算保护）。
    """
    sim = MockSimulator(random_case(n=10, seed=3), seed=3)
    enter = sim.enter()
    assert enter["virtual_time_s"] == 0.0
    assert enter["max_real_duration_s"] == 1200.0
    assert enter["max_virtual_duration_s"] > 0.0
    assert 0.0 <= enter["remaining_real_duration_s"] <= 1200.0
    measure = sim.measure((300.0, 400.0), 1)
    cleared = sim.clear((300.0, 0.0), 3)
    exit_body = sim.exit()
    for body in (enter, measure, cleared, exit_body):
        assert isinstance(body["real_timestamp_ms"], int)
    # accepted=false 时同样要给出三个字段，且 virtual_time_s 固定为 0
    rejected = sim.measure((0.0, 0.0), 99)
    assert rejected["accepted"] is False
    assert rejected["virtual_time_s"] == 0.0
    assert isinstance(rejected["real_timestamp_ms"], int)

