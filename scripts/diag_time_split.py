"""诊断：问题三/四的时间与行程分解，定位可压缩瓶颈（诊断用脚本）。

用法：
    & 'D:\\Anaconda3\\envs\\CUMCM\\python.exe' scripts\\diag_time_split.py
"""
from __future__ import annotations

import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import MOVE_SPEED  # noqa: E402
from src.coverage import seven_point_design  # noqa: E402
from src.mock_simulator import MockSimulator, random_case  # noqa: E402
from src.routing import tour_length, two_opt_tour  # noqa: E402
from src.strategy_p3 import StrategyP3  # noqa: E402
from src.strategy_p4 import StrategyP4  # noqa: E402


def backbone_length() -> float:
    """问题三覆盖骨干（中心 + 14 点环）从原点出发的开放巡游长度 (m)。"""
    ring = seven_point_design(1000.0, 14)
    return tour_length(two_opt_tour([(0.0, 0.0)] + list(ring)))


def split(tag: str, factory, n: int, ratio: float, seeds: int = 10) -> None:
    times, moves, meas, clr = [], [], [], []
    for s in range(seeds):
        sim = MockSimulator(random_case(n=n, seed=s, directional_ratio=ratio),
                            seed=s)
        r = factory().run(sim)
        times.append(r.virtual_time)
        moves.append(r.move_distance)
        meas.append(r.measures)
        clr.append(r.clears)
    t = statistics.fmean(times)
    mv = statistics.fmean(moves)
    ms = statistics.fmean(meas)
    cl = statistics.fmean(clr)
    move_s = mv / MOVE_SPEED
    meas_s = ms * 6.0          # 5 s 检测 + 1 s 切换
    clear_s = cl * 5.0
    other = t - move_s - meas_s - clear_s
    print(f"{tag}: 总 {t:7.1f}s = 移动 {move_s:7.1f}s ({100*move_s/t:4.1f}%) "
          f"+ 测量 {meas_s:6.1f}s ({100*meas_s/t:4.1f}%) "
          f"+ 清除 {clear_s:5.1f}s + 其他 {other:6.1f}s")
    print(f"      行程 {mv/1000:6.2f} km   测量 {ms:5.1f} 次   清除 {cl:4.1f} 次")


def main() -> None:
    bl = backbone_length()
    print(f"问题三覆盖骨干（中心+14 点环）开放巡游 = {bl/1000:.2f} km "
          f"= {bl/MOVE_SPEED:.0f}s（保证性覆盖的固定成本）")
    print(f"访问 16 个源的 TSP 下界 ≈ 0.7124*sqrt(16*pi*1800^2) = "
          f"{0.7124*(16*3.14159265*1800**2)**0.5/1000:.2f} km "
          f"= {0.7124*(16*3.14159265*1800**2)**0.5/MOVE_SPEED:.0f}s")
    print()
    split("P3 现状", StrategyP3, 16, 0.0)
    split("P4 现状", StrategyP4, 16, 1.0)


if __name__ == "__main__":
    main()
