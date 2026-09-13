"""机器狗 HTTP 客户端：严格按附件2 协议与模拟器通信。

协议要点（附件2 第 5 节）
------------------------
* POST + Content-Type: application/json；路径精确为 /enter /measure /clear /exit；
* 请求体 UTF-8 JSON 对象，字段：arena_id / robot_id / request_id (+position, channel)；
* 每条新动作使用新的 request_id；仅"网络故障重试同一动作"时复用原 request_id；
* 必须同时检查 HTTP 状态码与 accepted 字段；
* accepted=false 时 virtual_time_s=0 不是当前虚拟时刻（不得据此推进本地时钟）；
* 逐次串行发送，不得并发。

本客户端还内置**本地虚拟时钟复算**：用 kinematics.VirtualClock 镜像模拟器计时，
每条响应与响应的 virtual_time_s 交叉校验，偏差超阈值即记录异常（防御计费 bug）。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import requests

from .config import ARENA_ID, DEFAULT_BASE_URL
from .kinematics import VirtualClock

Point = Tuple[float, float]

__all__ = ["RobotClient", "ProtocolError", "RequestRecord"]


class ProtocolError(RuntimeError):
    """HTTP 状态或业务状态异常（不可重试的协议级错误）。"""


@dataclass
class RequestRecord:
    path: str
    payload: Dict[str, Any]
    status: Optional[int]
    body: Optional[Dict[str, Any]]
    attempts: int
    elapsed_s: float
    ok: bool


@dataclass
class RobotClient:
    robot_id: str
    base_url: str = DEFAULT_BASE_URL
    timeout: float = 5.0
    max_retries: int = 3
    retry_backoff_s: float = 0.2
    audit_tolerance_s: float = 1e-3
    session: requests.Session = field(default_factory=requests.Session)
    clock: VirtualClock = field(default_factory=VirtualClock)
    history: List[RequestRecord] = field(default_factory=list)
    audit_warnings: List[str] = field(default_factory=list)
    _seq: int = 0

    # ------------------------------------------------------------ 内部
    def _next_id(self, tag: str) -> str:
        self._seq += 1
        return f"{tag}-{self._seq}"

    def _base_payload(self, request_id: str) -> Dict[str, Any]:
        return {"arena_id": ARENA_ID, "robot_id": self.robot_id, "request_id": request_id}

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """发送请求：网络错误时复用原 request_id 重试；协议错误直接抛出。"""
        last_exc: Optional[Exception] = None
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        for attempt in range(1, self.max_retries + 1):
            t0 = time.perf_counter()
            try:
                resp = self.session.post(
                    self.base_url + path, data=data,
                    headers={"Content-Type": "application/json"}, timeout=self.timeout,
                )
            except requests.RequestException as exc:      # 连接失败/超时：可重试
                last_exc = exc
                self.history.append(RequestRecord(path, payload, None, None,
                                                  attempt, time.perf_counter() - t0, False))
                if attempt < self.max_retries:
                    time.sleep(self.retry_backoff_s * attempt)
                    continue
                raise ConnectionError(f"{path} 连接失败: {exc}") from exc

            elapsed = time.perf_counter() - t0
            body: Optional[Dict[str, Any]]
            try:
                body = resp.json()
            except ValueError:
                body = None
            self.history.append(RequestRecord(path, payload, resp.status_code, body,
                                              attempt, elapsed, resp.status_code == 200))

            if resp.status_code != 200:                   # 400/404/409/... 不可重试
                raise ProtocolError(f"{path} HTTP {resp.status_code}: {body}")
            if body is None:
                raise ProtocolError(f"{path} 响应不是 JSON: {resp.text[:200]}")
            if not isinstance(body.get("accepted"), bool):
                raise ProtocolError(f"{path} 响应缺少 accepted 字段: {body}")
            return body
        raise ConnectionError(f"{path} 重试耗尽: {last_exc}")

    # ------------------------------------------------------------ 4 条指令
    def enter(self) -> Dict[str, Any]:
        body = self._post("/enter", self._base_payload(self._next_id("enter")))
        if body["accepted"]:
            self.clock.enter()
            self._audit("/enter", body)
        return body

    def measure(self, position: Point, channel: int) -> Dict[str, Any]:
        payload = self._base_payload(self._next_id("measure"))
        payload["position"] = {"x": float(position[0]), "y": float(position[1])}
        payload["channel"] = int(channel)
        body = self._post("/measure", payload)
        if body["accepted"]:
            self.clock.measure(position, int(channel))
            self._audit("/measure", body)
        return body

    def clear(self, position: Point, channel: int) -> Dict[str, Any]:
        payload = self._base_payload(self._next_id("clear"))
        payload["position"] = {"x": float(position[0]), "y": float(position[1])}
        payload["channel"] = int(channel)
        body = self._post("/clear", payload)
        if body["accepted"]:
            success = body.get("clear_result") == "success"
            self.clock.clear(position, success=success, channel=int(channel))
            self._audit("/clear", body)
        return body

    def exit(self) -> Dict[str, Any]:
        body = self._post("/exit", self._base_payload(self._next_id("exit")))
        if body["accepted"]:
            self.clock.exit()
            self._audit("/exit", body)
        return body

    # ------------------------------------------------------------ 校验
    def _audit(self, path: str, body: Dict[str, Any]) -> None:
        """本地复算的虚拟时间与模拟器返回值交叉校验。"""
        reported = body.get("virtual_time_s")
        if not isinstance(reported, (int, float)):
            return
        delta = abs(float(reported) - self.clock.virtual_time)
        if delta > self.audit_tolerance_s:
            self.audit_warnings.append(
                f"{path}: 本地 {self.clock.virtual_time:.4f}s vs 模拟器 {reported:.4f}s"
                f"（差 {delta:.4f}s）"
            )

    @property
    def virtual_time(self) -> float:
        return self.clock.virtual_time
