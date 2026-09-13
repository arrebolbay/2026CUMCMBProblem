"""2026 CUMCM B 题：无线电干扰源的快速自动定位与清除 —— 算法包。

模块划分
--------
config      : 全局常量（题目给定参数）
geometry    : 计算几何核心（楔形交、定位区域、直径、最小覆盖圆）
kinematics  : 虚拟时钟/耗时模型（与模拟器计时规则逐条对齐）
coverage    : 检测点网设计与覆盖性验证（问题 3 的布点下界）
"""

__all__ = ["config", "geometry", "kinematics", "coverage"]
