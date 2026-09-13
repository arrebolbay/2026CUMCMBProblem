"""虚拟时钟与动作耗时模型。

与附件1《使用说明》第 1、2 节及附件2《通信接口说明》第 4 节逐条对齐：

    /enter  不推进时钟
    /measure  accepted=true 时：移动耗时 + 切换频道耗时 + 检测耗时(5s)
    /clear    accepted=true 时：移动耗时 + 精确定位耗时(3s) [+ 清除耗时(2s) 若成功]
    /exit    不推进时钟

移动耗时 = 前一次合法动作位置到本次位置的距离 / 5 m/s
切换频道耗时：仅当本次合法 /measure 的 channel 与当前频道不同，1s；
             /clear 的 channel 是"目标干扰源频道"，不切换、不耗时。

用途：
  1. 本地复算虚拟时间，与模拟器响应中的 virtual_time_s 交叉校验；
  2. 离线 mock 仿真中评估策略的时间代价（问题3/4 的目标函数）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from .config import (
    CLEAR_FAIL_TIME,
    CLEAR_SUCCESS_TIME,
    MEASURE_TIME,
    MOVE_SPEED,
    SWITCH_TIME,
)
from .geometry import distance

Point = Tuple[float, float]

__all__ = ["ActionCost", "VirtualClock", "simulate_worked_example"]


@dataclass
class ActionCost:
    """单条指令的耗时明细（单位：秒）。"""

    distance_m: float = 0.0
    move_s: float = 0.0
    switch_s: float = 0.0
    action_s: float = 0.0
    clear_result: Optional[str] = None      # None / "success" / "no_target_in_range"

    @property
    def total_s(self) -> float:
        return self.move_s + self.switch_s + self.action_s


class VirtualClock:
    """机器狗状态机（位置 + 测向机当前频道 + 虚拟时刻）。"""

    def __init__(
        self,
        start: Point = (0.0, 0.0),
        channel: int = 1,
        speed: float = MOVE_SPEED,
        measure_time: float = MEASURE_TIME,
        switch_time: float = SWITCH_TIME,
        clear_fail_time: float = CLEAR_FAIL_TIME,
        clear_success_time: float = CLEAR_SUCCESS_TIME,
    ) -> None:
        self.position: Point = (float(start[0]), float(start[1]))
        self.channel: int = int(channel)
        self.virtual_time: float = 0.0
        self.speed = float(speed)
        self.measure_time = float(measure_time)
        self.switch_time = float(switch_time)
        self.clear_fail_time = float(clear_fail_time)
        self.clear_success_time = float(clear_success_time)
        self.history: List[Tuple[str, ActionCost]] = []

    # ------------------------------------------------------------------ #
    def _move(self, target: Point) -> Tuple[float, float]:
        d = distance(self.position, target)
        return d, d / self.speed

    def measure(self, target: Point, channel: int) -> ActionCost:
        """合法 /measure：推进虚拟时钟。"""
        dist, move_s = self._move(target)
        switch_s = 0.0 if int(channel) == self.channel else self.switch_time
        cost = ActionCost(dist, move_s, switch_s, self.measure_time)
        self.position = (float(target[0]), float(target[1]))
        self.channel = int(channel)
        self.virtual_time += cost.total_s
        self.history.append(("measure", cost))
        return cost

    def clear(self, target: Point, success: bool, channel: Optional[int] = None) -> ActionCost:
        """合法 /clear：推进虚拟时钟；不改变测向机频道。"""
        dist, move_s = self._move(target)
        action_s = self.clear_success_time if success else self.clear_fail_time
        result = "success" if success else "no_target_in_range"
        cost = ActionCost(dist, move_s, 0.0, action_s, result)
        self.position = (float(target[0]), float(target[1]))
        self.virtual_time += cost.total_s
        self.history.append(("clear", cost))
        return cost

    def enter(self) -> ActionCost:
        cost = ActionCost()
        self.history.append(("enter", cost))
        return cost

    def exit(self) -> ActionCost:
        cost = ActionCost()
        self.history.append(("exit", cost))
        return cost


def simulate_worked_example() -> Tuple[List[ActionCost], float]:
    """复现附件1 表2 的示例，用于验证计时模型的正确性。

    指令序列：(300,400)测频道1 -> (300,400)测频道2 -> (300,0)清除频道3(未发现)
              -> (300,0)测频道2 -> /exit
    参考结果：各步总耗时 105, 6, 83, 5；退出时虚拟时刻 199。
    """
    clk = VirtualClock()
    clk.enter()
    costs = [
        clk.measure((300.0, 400.0), 1),   # 500m/5 = 100s + 5s = 105s
        clk.measure((300.0, 400.0), 2),   # 换频道 1s + 5s = 6s
        clk.clear((300.0, 0.0), success=False),  # 400m/5 = 80s + 3s = 83s
        clk.measure((300.0, 0.0), 2),     # 同频道，0s + 5s = 5s
    ]
    clk.exit()
    return costs, clk.virtual_time
