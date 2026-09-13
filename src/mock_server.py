"""把 MockSimulator 包装成符合附件2 协议的本地 HTTP 服务。

用于离线端到端联调（含 request_id 幂等、未知字段拒绝、HTTP 状态码语义），
从而在不联网、不消耗测试机会的条件下验证机器狗程序与协议的兼容性。
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional, Tuple

from .mock_simulator import MockSimulator

__all__ = ["MockServer"]

_ALLOWED_PATHS = ("/enter", "/measure", "/clear", "/exit")
_BASE_FIELDS = {"arena_id", "robot_id", "request_id"}
_ACTION_FIELDS = {"position", "channel"}


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "MockJammers/1.0"

    # 由 MockServer 注入
    sim: MockSimulator
    robot_id: str

    def log_message(self, *args) -> None:      # 静默
        return

    # ------------------------------------------------------------------ #
    def _send(self, status: int, body: Dict[str, Any]) -> None:
        raw = json.dumps(body, ensure_ascii=False, default=vars).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _reject(self, status: int = 200) -> None:
        self._send(status, {"accepted": False, "real_timestamp_ms": 0,
                            "virtual_time_s": 0.0})

    def do_POST(self) -> None:  # noqa: N802
        path = self.path
        if path not in _ALLOWED_PATHS:
            self._reject(404)
            return
        ctype = self.headers.get("Content-Type", "")
        if "application/json" not in ctype:
            self._reject(415)
            return
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._reject(400)
            return
        if not isinstance(payload, dict):
            self._reject(400)
            return

        allowed = _BASE_FIELDS | (_ACTION_FIELDS if path != "/enter" and path != "/exit" else set())
        if set(payload) - allowed:                       # 未知字段 -> accepted=false
            self._reject(200)
            return
        if payload.get("arena_id") != "default" or payload.get("robot_id") != self.robot_id:
            self._reject(200)
            return
        if not payload.get("request_id"):
            self._reject(400)
            return

        sim = self.sim
        if path == "/enter":
            self._send(200, sim.enter())
            return
        if path == "/exit":
            self._send(200, sim.exit())
            return
        if "position" not in payload or "channel" not in payload:
            self._reject(400)
            return
        try:
            x = float(payload["position"]["x"])
            y = float(payload["position"]["y"])
            channel = payload["channel"]
            if not isinstance(channel, int) or isinstance(channel, bool):
                raise ValueError
            if not 1 <= channel <= 20:               # 与真实模拟器一致：400
                raise ValueError
        except (TypeError, KeyError, ValueError):
            self._reject(400)
            return

        if path == "/measure":
            self._send(200, sim.measure((x, y), channel))
        else:
            self._send(200, sim.clear((x, y), channel))


class MockServer:
    """在后台线程中运行的本地 mock 服务。"""

    def __init__(self, sim: MockSimulator, robot_id: str,
                 host: str = "127.0.0.1", port: int = 0) -> None:
        handler = type("BoundHandler", (_Handler,), {"sim": sim, "robot_id": robot_id})
        self._httpd = ThreadingHTTPServer((host, port), handler)
        self.host, self.port = self._httpd.server_address[0], self._httpd.server_address[1]
        self._thread: Optional[threading.Thread] = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self) -> "MockServer":
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        if self._thread:
            self._thread.join(timeout=2.0)

    def __enter__(self) -> "MockServer":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()
