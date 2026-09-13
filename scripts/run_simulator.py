"""接入真实模拟器（jammers-simulator.exe）运行问题3 策略。

使用流程（详见 README / 本文件末尾说明）
--------------------------------------
1. 双击运行模拟器 -> 联网登录（参赛队号/队员姓名/手机号 + 密码）；
2. 在模拟器中选择"问题3 演练测试"（演练不限次数）或"问题3 正式测试"（仅 3 次），
   建议保持默认端口 2026，点击开始，等待 5 秒倒计时结束、界面提示接口就绪；
3. 运行本脚本：
       python scripts/run_simulator.py --robot-id <参赛队号> [--port 2026]
   脚本会等待接口开放、调用 /enter，随后自动完成扫描-定位-清除并 /exit；
4. 演练结束后在模拟器界面核对"干扰源总数/清除个数"（正式测试不显示真值，
   但会记录测试案例编码与加密行为日志，需导出放入支撑材料）。

安全保护
--------
* 现实时间预算（默认按 /enter 返回的 remaining_real_duration_s，留 60 s 余量）
  超时即停止动作并 /exit，避免 20 分钟程序运行超时；
* 每条请求/响应写入 logs/ 下的 JSONL 行为日志（自建，可作支撑材料）；
* 接口未开放时连接会失败：脚本自动轮询重试，不对连接失败做任何状态变更。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from typing import Any, Dict, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.config import DEFAULT_BASE_URL                      # noqa: E402
from src.robot_client import RobotClient, ProtocolError      # noqa: E402
from src.simulator_truth import (                            # noqa: E402
    DEFAULT_SIM_DATA_DIR,
    wait_for_result_after,
)
from src.strategy_p3 import StrategyP3                       # noqa: E402
from src.strategy_p4 import StrategyP4                       # noqa: E402


class TimeBudgetExceeded(RuntimeError):
    """现实时间预算耗尽：立即停止动作并优雅退出。"""


class LoggingTransport:
    """包装 RobotClient：写 JSONL 行为日志 + 现实时间预算 + 统计。"""

    def __init__(self, client: RobotClient, log_path: str,
                 budget_s: Optional[float] = None) -> None:
        self.client = client
        self.log_path = log_path
        self._fh = open(log_path, "w", encoding="utf-8")
        self.budget_s = budget_s
        self._t0 = time.perf_counter()
        self.counts: Dict[str, int] = {"measure": 0, "clear": 0, "clear_success": 0,
                                       "clear_fail": 0, "direction": 0, "near": 0,
                                       "no_signal": 0}
        self.cleared_channels: set = set()
        self.unresolved_measures: int = 0
        self._exit_resp: Optional[Dict[str, Any]] = None   # /exit 幂等：只真正发一次

    # ------------------------------------------------------------------ #
    def _remaining(self) -> Optional[float]:
        if self.budget_s is None:
            return None
        return self.budget_s - (time.perf_counter() - self._t0)

    def _check_budget(self) -> None:
        remaining = self._remaining()
        if remaining is not None and remaining <= 0:
            raise TimeBudgetExceeded("现实时间预算耗尽")

    def _log(self, path: str, payload: Dict[str, Any],
             response: Dict[str, Any]) -> None:
        self._fh.write(json.dumps({
            "ts": time.time(),
            "path": path,
            "request": payload,
            "response": response,
            "local_virtual_time_s": round(self.client.virtual_time, 6),
        }, ensure_ascii=False) + "\n")
        self._fh.flush()

    # ------------------------------------------------------------------ #
    def enter(self) -> Dict[str, Any]:
        resp = self.client.enter()
        self.adopt_enter_response(resp)
        return resp

    def adopt_enter_response(self, resp: Dict[str, Any]) -> None:
        """/enter 已由外部（等待接口就绪流程）完成时，同步日志与时间预算。"""
        self._log("/enter", {"request_id": "enter"}, resp)
        remaining = resp.get("remaining_real_duration_s")
        if isinstance(remaining, (int, float)):
            # 预留 60 s 收尾（最后一条动作登记 + 退出）
            self.budget_s = min(self.budget_s or float("inf"),
                                max(float(remaining) - 60.0, 5.0))
            self._t0 = time.perf_counter()

    def measure(self, position: Tuple[float, float], channel: int) -> Dict[str, Any]:
        self._check_budget()
        resp = self.client.measure(position, channel)
        self.counts["measure"] += 1
        kind = resp.get("measure_result")
        if kind in self.counts:
            self.counts[kind] += 1
        elif resp.get("accepted") is False:
            self.unresolved_measures += 1
        self._log("/measure", {"position": position, "channel": channel}, resp)
        return resp

    def clear(self, position: Tuple[float, float], channel: int) -> Dict[str, Any]:
        self._check_budget()
        resp = self.client.clear(position, channel)
        self.counts["clear"] += 1
        if resp.get("clear_result") == "success":
            self.counts["clear_success"] += 1
            self.cleared_channels.add(channel)
        else:
            self.counts["clear_fail"] += 1
        self._log("/clear", {"position": position, "channel": channel}, resp)
        return resp

    def exit(self) -> Dict[str, Any]:
        """发送 /exit（**幂等**：重复调用不再发请求）。

        真实模拟器收到 /exit 后立即结束测试并关闭接口；而策略内部存在多个收尾
        路径都会调用 exit（``_finalize_channel_clearing`` 与主流程各一次）。
        若重复发送，第二次会因接口已关闭而抛 ConnectionError，
        使一次**完全成功**的运行被误标记为失败（Mock 不加限制，故演练看不出来）。
        """
        if self._exit_resp is not None:
            return self._exit_resp
        resp = self.client.exit()
        self._log("/exit", {"request_id": "exit"}, resp)
        self._exit_resp = resp
        return resp

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:      # pragma: no cover - 关闭失败不致命
            pass

    @property
    def cleared_count(self) -> int:
        return len(self.cleared_channels)

    @property
    def wall_time_s(self) -> float:
        return time.perf_counter() - self._t0

    @property
    def clock(self):
        """把内部 RobotClient 的虚拟时钟暴露给策略。

        策略的 ``_current_position`` / ``_move_distance`` 通过
        ``transport.clock`` 读取真实位置与行程，包装层必须透明转发，
        否则机会式清除的绕行距离会按 (0,0) 估算而失真。
        """
        return self.client.clock


def wait_for_interface(client: RobotClient, timeout_s: float = 600.0,
                       poll_s: float = 3.0) -> Dict[str, Any]:
    """等待模拟器接口开放并完成 /enter。

    5 秒倒计时期间、测试未开始或已结束时接口未开放，连接会直接失败，
    因此这里对 ConnectionError 轮询重试；一旦 enter 成功立即返回响应。
    """
    deadline = time.perf_counter() + timeout_s
    attempt = 0
    while time.perf_counter() < deadline:
        attempt += 1
        try:
            resp = client.enter()
        except (ConnectionError, ProtocolError) as exc:
            print(f"[等待接口] 第 {attempt} 次连接未就绪（{type(exc).__name__}），"
                  f"{poll_s:.0f}s 后重试...")
            time.sleep(poll_s)
            continue
        if resp.get("accepted") is True:
            print(f"[等待接口] /enter 成功，剩余现实时间 "
                  f"{resp.get('remaining_real_duration_s')} s")
            return resp
        print(f"[等待接口] /enter 未被接受：{resp}，{poll_s:.0f}s 后重试...")
        time.sleep(poll_s)
    raise TimeoutError("等待模拟器接口就绪超时（请确认已开始测试且倒计时已结束）")


def build_mock_client(robot_id: str, seed: int, n: Optional[int]) -> Tuple[Any, Any, str]:
    """离线自检模式：启动本地 MockServer，用于验证脚本链路（不消耗测试机会）。"""
    from src.mock_server import MockServer
    from src.mock_simulator import MockSimulator, random_case

    sim = MockSimulator(random_case(n=n, seed=seed), seed=seed)
    server = MockServer(sim, robot_id=robot_id).start()
    client = RobotClient(robot_id=robot_id, base_url=server.base_url)
    return server, client, server.base_url


def main() -> None:
    parser = argparse.ArgumentParser(description="CUMCM2026 B题 问题3/4 运行器")
    parser.add_argument("--problem", type=int, default=3, choices=[3, 4],
                        help="问题号：3=全向，4=全向+定向混合")
    parser.add_argument("--robot-id", required=True, help="参赛队号（须与模拟器登录一致）")
    parser.add_argument("--host", default="127.0.0.1", help="模拟器地址（默认本机）")
    parser.add_argument("--port", type=int, default=2026, help="模拟器端口（默认 2026）")
    parser.add_argument("--base-url", default=None,
                        help="完整接口地址（提供后忽略 --host/--port）")
    parser.add_argument("--max-seconds", type=float, default=None,
                        help="自设现实时间预算（秒）；默认使用 /enter 返回的剩余时间-60s")
    parser.add_argument("--ready-timeout", type=float, default=600.0,
                        help="等待接口就绪的最长时间（秒）")
    parser.add_argument("--expected-total", type=int, default=0,
                        help="干扰源总数（留空则自动读取演练结果文件）")
    parser.add_argument("--case-code", default=None,
                        help="测试案例编码（正式测试从界面抄录）")
    parser.add_argument("--sim-data-dir", default=DEFAULT_SIM_DATA_DIR,
                        help="模拟器数据目录（用于自动读取演练真值）")
    parser.add_argument("--no-auto-truth", action="store_true",
                        help="不读取演练结果文件")
    parser.add_argument("--mock", action="store_true",
                        help="离线自检：本地起 MockServer，不连接真实模拟器")
    parser.add_argument("--mock-seed", type=int, default=0)
    parser.add_argument("--mock-n", type=int, default=None)
    parser.add_argument("--log-dir", default=os.path.join(ROOT, "logs"))
    parser.add_argument("--label", default=None,
                        help="本次运行的自定义标签，如「正式测试1」。"
                             "正式测试建议一律填写，便于事后自动汇总成论文表格")
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(args.log_dir, f"p{args.problem}_run_{stamp}.jsonl")

    server = None
    if args.mock:
        server, client, base_url = build_mock_client(args.robot_id, args.mock_seed, args.mock_n)
        print(f"[离线自检] MockServer: {base_url}")
    else:
        base_url = args.base_url or f"http://{args.host}:{args.port}"
        client = RobotClient(robot_id=args.robot_id, base_url=base_url)
        print(f"[连接真实模拟器] {base_url}（robot_id={args.robot_id}）")

    transport = LoggingTransport(client, log_path, budget_s=args.max_seconds)
    run_started_epoch = time.time()
    status = "completed"
    result = None
    try:
        enter_resp = wait_for_interface(client, timeout_s=args.ready_timeout)
        # /enter 已由等待流程完成：同步日志与时间预算，策略内不再重复 enter
        transport.adopt_enter_response(enter_resp)
        strategy_cls = StrategyP3 if args.problem == 3 else StrategyP4
        result = strategy_cls().run(transport, do_enter=False)
    except TimeBudgetExceeded:
        status = "time_budget_exceeded"
        print("[警告] 现实时间预算耗尽，提前结束并退出。")
        try:
            transport.exit()
        except Exception as exc:      # pragma: no cover
            print(f"[警告] /exit 失败：{exc}")
    except Exception as exc:          # noqa: BLE001 - 兜底保证退出
        msg = str(exc)
        if isinstance(exc, ConnectionError) and "exit" in msg:
            # 真实模拟器收到 /exit 后**立即结束测试并关闭接口**，客户端有时读不到
            # 响应而报"远程主机强迫关闭连接"。该请求其实已被处理（模拟器官方
            # result.json 已落盘），因此不算失败，避免把成功的运行误标为 error。
            status = "completed(exit_unreachable)"
            print(f"[提示] /exit 响应未读到（模拟器已结束测试并关闭接口）：{exc}")
        else:
            status = f"error:{type(exc).__name__}"
            print(f"[错误] {exc}")
            try:
                transport.exit()
            except Exception:         # pragma: no cover
                pass
    finally:
        transport.close()
        if server is not None:
            server.stop()

    cleared = transport.cleared_count if result is None else max(result.cleared, transport.cleared_count)
    avg_time = (client.virtual_time / cleared) if cleared else float("inf")

    # ------- 官方元数据：演练给真值；正式测试只给案例编码 -------
    truth = None
    if not args.mock and not args.no_auto_truth:
        truth = wait_for_result_after(args.sim_data_dir, since_mtime=run_started_epoch,
                                      problem_no=args.problem, timeout_s=20.0)
        if truth is None:
            print("[官方] 未找到本次结果文件（可用 --case-code 手动指定案例编码）")
    is_formal = bool(truth and truth.get("formal_index"))
    if truth is not None:
        if is_formal:
            print(f"[官方] 正式测试 #{truth.get('formal_index')}，案例编码 "
                  f"{truth.get('case_code')}（正式测试不公布干扰源总数）")
        else:
            print(f"[真值] 案例编码 {truth.get('case_code')}，干扰源总数 "
                  f"{truth.get('jammer_count')}（全向 "
                  f"{truth.get('omnidirectional_jammer_count')}，定向 "
                  f"{truth.get('directional_jammer_count')}）")
    # 注意：正式测试的结果文件**不含** jammer_count，必须用 .get() 容错，
    # 否则会 KeyError 导致整个报告写不出来（曾经丢过一次正式测试记录）。
    jam = truth.get("jammer_count") if truth else None
    expected_total = args.expected_total or (int(jam) if jam else None)
    case_code = args.case_code or (truth.get("case_code") if truth else None)
    # 合规：剔除官方文件里的队伍标识字段，避免队号进入支撑材料
    truth_clean = None if truth is None else {
        k: v for k, v in truth.items()
        if not k.startswith("_") and "team" not in k.lower()}
    report = {
        "status": status,
        "problem_no": args.problem,
        "run_mode": "mock" if args.mock else "live",
        "run_kind": "formal" if is_formal else ("mock" if args.mock else "practice"),
        "label": args.label,
        "robot_id": args.robot_id,
        "base_url": base_url,
        "cleared_jammers": cleared,
        "expected_total": expected_total,
        "case_code": case_code,
        "truth": truth_clean,
        "clearance_ratio": (cleared / expected_total) if expected_total else None,
        "total_virtual_time_s": client.virtual_time,
        "mean_clear_time_s": avg_time,
        "program_run_time_s": round(transport.wall_time_s, 3),
        "counts": transport.counts,
        "unresolved": [] if result is None else result.unresolved,
        "log_file": log_path,
        "audit_warnings": client.audit_warnings,
    }
    print("\n=== 本次运行结果 ===")
    for key, value in report.items():
        print(f"{key}: {value}")

    out = os.path.join(ROOT, "results", f"p{args.problem}_run_{stamp}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print(f"\n结果: {out}\n行为日志: {log_path}")


if __name__ == "__main__":
    main()
