"""端到端协议测试：RobotClient <-> MockServer（HTTP+JSON、幂等、未知字段、计时审计）。"""

import json

import pytest
import requests

from src.mock_server import MockServer
from src.mock_simulator import MockSimulator, random_case
from src.robot_client import RobotClient, ProtocolError

ROBOT_ID = "TEAM-ID-PLACEHOLDER"   # 占位符：实际队号由命令行/环境变量提供


def _load_run_simulator():
    """导入 scripts/run_simulator.py（脚本目录不在包路径内，故显式加载）。"""
    import importlib.util
    import os

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, "scripts", "run_simulator.py")
    spec = importlib.util.spec_from_file_location("run_simulator_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def env():
    sim = MockSimulator(random_case(n=3, seed=5), seed=5)
    srv = MockServer(sim, robot_id=ROBOT_ID).start()
    client = RobotClient(robot_id=ROBOT_ID, base_url=srv.base_url)
    yield sim, srv, client
    srv.stop()


def test_enter_measure_clear_exit_end_to_end(env):
    sim, srv, client = env
    enter = client.enter()
    assert enter["accepted"] is True

    first = sim.jammers[0]
    # 到该干扰源正上方测量（距离 0 -> near），再退到远处测量
    far_point = (first.x + 400.0, first.y)
    r = client.measure(far_point, first.channel)
    assert r["accepted"] is True
    assert r["measure_result"] in {"direction", "near", "no_signal"}

    # 直接在其 20m 内清除
    hit = client.clear((first.x + 10.0, first.y), first.channel)
    assert hit["clear_result"] == "success"
    assert sim.cleared_jammers == 1

    bye = client.exit()
    assert bye["exit_reason"] == "user_exit"
    # 本地虚拟时钟镜像必须与模拟器一致（协议与计时模型双向验证）
    assert client.audit_warnings == []
    assert client.virtual_time == pytest.approx(sim.virtual_time)


def test_logging_transport_exposes_client_clock(env, tmp_path):
    """LoggingTransport 必须把内部 RobotClient 的时钟透传给策略。

    机会式清除用 `transport.clock.position` 计算绕行距离，若包装层不透传，
    真实运行时会退化成"始终从 (0,0) 估算"，插入判据失真。
    """
    sim, srv, client = env
    module = _load_run_simulator()
    log_path = str(tmp_path / "run.jsonl")
    transport = module.LoggingTransport(client, log_path)
    try:
        transport.enter()
        transport.measure((300.0, 400.0), 1)
        # 时钟透传：位置与虚拟时间都能从包装层读到
        assert transport.clock is client.clock
        assert transport.clock.position == pytest.approx((300.0, 400.0))
        assert transport.clock.virtual_time == pytest.approx(105.0)
        # 策略侧的移动距离统计也应能读通
        from src.strategy_p3 import StrategyP3

        assert StrategyP3._move_distance(transport) == pytest.approx(500.0)
        assert StrategyP3._current_position(transport) == pytest.approx((300.0, 400.0))
    finally:
        transport.close()


def test_unknown_field_is_rejected_with_accepted_false(env):
    sim, srv, client = env
    client.enter()
    payload = client._base_payload("test-unknown")
    payload["position"] = {"x": 0, "y": 0}
    payload["channel"] = 1
    payload["oops"] = 1
    resp = requests.post(srv.base_url + "/measure",
                         data=json.dumps(payload).encode(),
                         headers={"Content-Type": "application/json"}, timeout=5)
    assert resp.status_code == 200
    assert resp.json()["accepted"] is False


def test_wrong_robot_id_rejected(env):
    sim, srv, client = env
    payload = {"arena_id": "default", "robot_id": "bad", "request_id": "x1"}
    resp = requests.post(srv.base_url + "/enter", data=json.dumps(payload).encode(),
                         headers={"Content-Type": "application/json"}, timeout=5)
    assert resp.json()["accepted"] is False


def test_unknown_path_returns_404(env):
    sim, srv, client = env
    resp = requests.post(srv.base_url + "/oops", data=b"{}",
                         headers={"Content-Type": "application/json"}, timeout=5)
    assert resp.status_code == 404


def test_bad_channel_returns_400(env):
    sim, srv, client = env
    client.enter()
    payload = client._base_payload("bad-channel")
    payload["position"] = {"x": 0, "y": 0}
    payload["channel"] = 1.5                    # 非整数
    resp = requests.post(srv.base_url + "/measure", data=json.dumps(payload).encode(),
                         headers={"Content-Type": "application/json"}, timeout=5)
    assert resp.status_code == 400


def test_out_of_range_channel_returns_400(env):
    sim, srv, client = env
    client.enter()
    payload = client._base_payload("bad")
    payload["position"] = {"x": 0, "y": 0}
    payload["channel"] = 99                     # 超出 1..20 -> HTTP 400（与真实模拟器一致）
    resp = requests.post(srv.base_url + "/measure", data=json.dumps(payload).encode(),
                         headers={"Content-Type": "application/json"}, timeout=5)
    assert resp.status_code == 400
    assert resp.json()["accepted"] is False


def test_client_raises_protocol_error_on_http_400(env):
    sim, srv, client = env
    client.enter()
    with pytest.raises(ProtocolError):
        client.measure((0.0, 0.0), 99)
