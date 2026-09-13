"""虚拟时钟测试：复现附件1 表2 的计时示例，验证计时规则实现无误。"""

import pytest

from src.kinematics import VirtualClock, simulate_worked_example


def test_worked_example_totals():
    costs, final_time = simulate_worked_example()
    totals = [c.total_s for c in costs]
    assert totals == pytest.approx([105.0, 6.0, 83.0, 5.0]), "附件1表2 的四步耗时"
    assert [c.move_s for c in costs] == pytest.approx([100.0, 0.0, 80.0, 0.0])
    assert [c.switch_s for c in costs] == pytest.approx([0.0, 1.0, 0.0, 0.0])
    assert [c.action_s for c in costs] == pytest.approx([5.0, 5.0, 3.0, 5.0])
    assert final_time == pytest.approx(199.0)


def test_clear_does_not_change_channel():
    clk = VirtualClock()
    clk.measure((300.0, 400.0), 2)
    assert clk.channel == 2
    clk.clear((300.0, 0.0), success=False, channel=3)
    assert clk.channel == 2, "/clear 的 channel 是目标干扰源频道，不切换测向机"


def test_switch_cost_only_when_channel_changes():
    clk = VirtualClock()
    c1 = clk.measure((0.0, 0.0), 1)          # 初始频道 1，同频道
    assert c1.switch_s == pytest.approx(0.0)
    c2 = clk.measure((0.0, 0.0), 20)         # 任意两频道切换均为 1s
    assert c2.switch_s == pytest.approx(1.0)
    c3 = clk.measure((0.0, 0.0), 2)
    assert c3.switch_s == pytest.approx(1.0)


def test_move_time_proportional_to_straight_line_distance():
    clk = VirtualClock()
    c = clk.measure((300.0, 400.0), 1)       # 500 m -> 100 s
    assert c.move_s == pytest.approx(100.0)
    c2 = clk.measure((300.0, 0.0), 1)        # 400 m -> 80 s
    assert c2.move_s == pytest.approx(80.0)
    assert clk.virtual_time == pytest.approx(100.0 + 5.0 + 80.0 + 5.0)


def test_success_clear_cost_is_5s_and_fail_is_3s():
    clk = VirtualClock()
    clk.measure((100.0, 0.0), 1)
    t0 = clk.virtual_time
    clk.clear((100.0, 0.0), success=True)
    assert clk.virtual_time - t0 == pytest.approx(5.0)
    t1 = clk.virtual_time
    clk.clear((100.0, 0.0), success=False)
    assert clk.virtual_time - t1 == pytest.approx(3.0)


def test_enter_exit_do_not_advance_clock():
    clk = VirtualClock()
    clk.enter()
    assert clk.virtual_time == 0.0
    clk.measure((500.0, 0.0), 1)
    t = clk.virtual_time
    clk.exit()
    assert clk.virtual_time == pytest.approx(t)
