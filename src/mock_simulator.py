"""离线 Mock 模拟器：复刻附件1/附件2 的物理规则与计时规则。

用途：在不消耗任何演练/正式测试机会的前提下开发与调试策略
（演练虽不限次数，但需联网登录；正式测试仅 3 次）。

复刻要点
--------
* 4 条指令语义与虚拟计时（复用 kinematics.VirtualClock）；
* 检测三态：no_signal / near(<=5m 且在覆盖角内) / direction(含 ±1° 误差)；
* 误差"同一地点固定"：按 (频道, 位置) 缓存，重复检测结果不变；
* 有效接收半径 r_eff ∈ [1000, 1500]，超出即 no_signal；
* 定向干扰源：覆盖角为定向方向 ±90°（含边界），范围外 no_signal；
* 清除：20m 内存在该频道未清除干扰源即成功（与朝向无关），同一源只能清一次；
* /clear 不切换测向机频道；/exit 不推进时钟。
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .config import (
    CHANNELS,
    CLEAR_RADIUS,
    JAMMER_NUM_RANGE,
    MAX_REAL_DURATION,
    NEAR_RADIUS,
    REGION_RADIUS,
    R_MAX,
    R_MIN,
)
from .geometry import angle_diff, bearing_between, distance
from .kinematics import VirtualClock

Point = Tuple[float, float]

__all__ = ["Jammer", "MockSimulator", "random_case"]


@dataclass
class Jammer:
    channel: int
    x: float
    y: float
    r_eff: float                       # 有效接收半径 (m)
    direction: Optional[float] = None  # None=全向；否则为定向方向 (度)
    cleared: bool = False

    @property
    def position(self) -> Point:
        return (self.x, self.y)


def random_case(
    n: Optional[int] = None,
    seed: int = 0,
    directional_ratio: float = 0.0,
    region_radius: float = REGION_RADIUS,
    directional_count: Optional[int] = None,
) -> List[Jammer]:
    """随机生成一个案例：n 个干扰源（默认 10~16 个），频道互不相同。

    ``directional_count`` 给定时，改为"恰好 k 个定向源"（其余全向），
    用于复现论文中固定混合比例（如 14 源中 6 个定向）的对照实验；
    该参数单独用一个随机流，不影响既有的 ``seed`` 复现结果。
    """
    rng = random.Random(seed)
    if n is None:
        n = rng.randint(*JAMMER_NUM_RANGE)
    channels = rng.sample(list(CHANNELS), n)
    jammers: List[Jammer] = []
    for ch in channels:
        r = region_radius * math.sqrt(rng.random())
        theta = rng.random() * 2.0 * math.pi
        is_dir = rng.random() < directional_ratio
        jammers.append(Jammer(
            channel=ch,
            x=r * math.cos(theta),
            y=r * math.sin(theta),
            r_eff=rng.uniform(R_MIN, R_MAX),
            direction=rng.uniform(0.0, 360.0) if is_dir else None,
        ))

    if directional_count is not None:
        k = max(0, min(int(directional_count), n))
        pick = random.Random(seed * 7919 + 13)
        chosen = set(pick.sample(range(n), k))
        for index, jam in enumerate(jammers):
            jam.direction = pick.uniform(0.0, 360.0) if index in chosen else None
    return jammers


class MockSimulator:
    """与真实模拟器等价的离线实现（内存版）。"""

    def __init__(
        self,
        jammers: Sequence[Jammer],
        speed: float = 5.0,
        eps_deg: float = 1.0,
        seed: int = 0,
        max_virtual_duration: float = 360_000.0,
    ) -> None:
        self.jammers: List[Jammer] = list(jammers)
        self.eps_deg = float(eps_deg)
        self.max_virtual_duration = float(max_virtual_duration)
        self.clock = VirtualClock(start=(0.0, 0.0), channel=1)
        self.clock.speed = float(speed)
        self.entered = False
        self.exited = False
        self._bias_cache: Dict[Tuple[int, float, float], float] = {}
        self._rng = random.Random(seed)
        self.log: List[dict] = []
        # 与真实模拟器对齐：本局实际可用的现实时间（/enter 响应字段）。
        # 真实环境里该值 = min(25 min 窗口剩余量, 1200 s)，可能不足 1200 s。
        self.remaining_real_duration_s = float(MAX_REAL_DURATION)

    # ------------------------------------------------------------ 内部工具
    @staticmethod
    def _now_ms() -> int:
        """真实时间戳（毫秒）。真实模拟器所有响应都带此字段（附件2 第 5.2 节）。"""
        return int(time.time() * 1000)

    # ------------------------------------------------------------ 接口
    def enter(self) -> dict:
        if self.entered and not self.exited:
            return self._reject("already_entered")
        self.entered, self.exited = True, False
        # 逐字段对齐附件2 第 6.2 节：/enter 返回三个时长字段，其中
        # remaining_real_duration_s 是"本局实际还能使用的完整现实时间"，
        # 策略不得固定假定为 1200 s（见 scripts/run_simulator.py 的预算保护）。
        payload = {
            "accepted": True,
            "real_timestamp_ms": self._now_ms(),
            "virtual_time_s": 0.0,
            "max_virtual_duration_s": self.max_virtual_duration,
            "max_real_duration_s": float(MAX_REAL_DURATION),
            "remaining_real_duration_s": self.remaining_real_duration_s,
        }
        self.log.append({"path": "/enter", **payload})
        return payload

    def measure(self, position: Point, channel: int) -> dict:
        if channel not in CHANNELS:
            return self._reject("bad_channel")
        result, svd = "no_signal", None
        jam = self._jammer_on(channel)
        if jam is not None and not jam.cleared:
            d = distance(position, jam.position)
            if d <= jam.r_eff and self._in_coverage(jam, position):
                if d <= NEAR_RADIUS:
                    result = "near"
                else:
                    bias = self._bias(jam.channel, position)
                    svd = (bearing_between(position, jam.position) + bias) % 360.0
                    result = "direction"
        cost = self.clock.measure(position, channel)
        payload = {
            "accepted": True,
            "real_timestamp_ms": self._now_ms(),
            "virtual_time_s": round(self.clock.virtual_time, 6),
            "measure_result": result,
            "svd_deg": None if svd is None else round(svd, 2),
            "cost": cost,
        }
        self.log.append({"path": "/measure", "position": position,
                         "channel": channel, **payload})
        return payload

    def clear(self, position: Point, channel: int) -> dict:
        if channel not in CHANNELS:
            return self._reject("bad_channel")
        target = None
        for jam in self.jammers:
            if jam.channel == channel and not jam.cleared:
                if distance(position, jam.position) <= CLEAR_RADIUS:
                    target = jam
                    break
        if target is not None:
            target.cleared = True
            result = "success"
        else:
            result = "no_target_in_range"
        cost = self.clock.clear(position, success=(result == "success"))
        payload = {
            "accepted": True,
            "real_timestamp_ms": self._now_ms(),
            "virtual_time_s": round(self.clock.virtual_time, 6),
            "clear_result": result,
            "cost": cost,
        }
        self.log.append({"path": "/clear", "position": position,
                         "channel": channel, **payload})
        return payload

    def exit(self) -> dict:
        self.exited = True
        self.clock.exit()
        payload = {"accepted": True, "real_timestamp_ms": self._now_ms(),
                   "virtual_time_s": round(self.clock.virtual_time, 6),
                   "exit_reason": "user_exit"}
        self.log.append({"path": "/exit", "position": self.clock.position, **payload})
        return payload

    # ------------------------------------------------------------ 内部
    def _reject(self, reason: str) -> dict:
        """业务拒绝：与附件2 第 5.2 节一致，保留 accepted/real_timestamp_ms/virtual_time_s。

        ``virtual_time_s`` 固定为 0 —— 附件2 明确要求"不要把这个 0 当作当前虚拟
        时刻"（accepted=false 不推进时钟）。
        """
        return {"accepted": False, "real_timestamp_ms": self._now_ms(),
                "virtual_time_s": 0.0, "reason": reason}

    def _jammer_on(self, channel: int) -> Optional[Jammer]:
        for jam in self.jammers:
            if jam.channel == channel:
                return jam
        return None

    @staticmethod
    def _in_coverage(jam: Jammer, position: Point) -> bool:
        """定向源：仅当检测点位于"定向方向 ±90°"的半平面内才可接收。"""
        if jam.direction is None:
            return True
        back_bearing = bearing_between(jam.position, position)
        return abs(angle_diff(back_bearing, jam.direction)) <= 90.0 + 1e-9

    def _bias(self, channel: int, position: Point) -> float:
        """同一地点、同一频道误差固定（题给），按位置缓存。"""
        key = (channel, round(position[0], 6), round(position[1], 6))
        if key not in self._bias_cache:
            self._bias_cache[key] = self._rng.uniform(-self.eps_deg, self.eps_deg)
        return self._bias_cache[key]

    # ------------------------------------------------------------ 统计
    @property
    def total_jammers(self) -> int:
        return len(self.jammers)

    @property
    def cleared_jammers(self) -> int:
        return sum(1 for j in self.jammers if j.cleared)

    @property
    def clearance_ratio(self) -> float:
        return self.cleared_jammers / self.total_jammers if self.jammers else 1.0

    @property
    def virtual_time(self) -> float:
        return self.clock.virtual_time

    @property
    def mean_clear_time(self) -> float:
        n = self.cleared_jammers
        return self.virtual_time / n if n else float("inf")
