"""对照实验：稀疏安全网 + 异常触发补网（"工程概率安全" vs "数学绝对安全"）。

命题（待检验）：用 16~18 个点的稀疏网快速跑一遍，只在"发现异常"时才启用
完整安全网补盲。若稀疏网的漏测概率足够低，则绝大多数案例更快，同时靠补网
保住 100% 清除。

本脚本分两层给结论：
  A. 几何层：直接算出各规模网点的**真实漏测概率**（对位置×朝向做蒙特卡洛），
     不依赖任何策略实现 —— 这是"概率安全"能否成立的前提。
  B. 策略层：实现"稀疏网跑一遍 → 触发条件满足则补网"，测时间与清除率，
     并与完整 22 点网基线对比。

触发条件（对照两种）：
  * ``trigger_below``：`已清除源数 < 该阈值` 才补网（题目保证 N ≥ 10，
    故阈值取 10 是"能确知漏了"的最弱充分信号）；
  * ``force_full``：无条件补网（等价于完整网，作为对照）。

用法：
    python scripts/exp_sparse_fallback.py
"""
from __future__ import annotations

import math
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import CHANNELS, REGION_RADIUS, R_MIN  # noqa: E402
from src.coverage import directional_survey_design  # noqa: E402
from src.mock_simulator import MockSimulator, random_case  # noqa: E402
from src.strategy_p3 import StrategyResult  # noqa: E402
from src.strategy_p4 import StrategyP4  # noqa: E402

Point = Tuple[float, float]


# ------------------------------------------------------------------ A 几何层
def miss_probability(net, n_g: int = 300_000, r_eff_low: float = R_MIN,
                     r_eff_high: float = 1500.0, seed: int = 0) -> Dict[str, float]:
    """对"随机位置 × 随机朝向 × 随机有效半径"估计漏测概率。

    源在 G、朝向 u（定向源）、有效接收半径 r_eff 时可被看到 ⇔ 存在网点 p
    使 ``|p-G| <= r_eff`` 且 ``|∠u − ∠(p−G)| <= 90°``。三者都随机时的不
    可见频率即"漏测概率"；另给出"一律按 r_eff = 1000 m 判"的保守口径。
    """
    rng = np.random.default_rng(seed)
    r = REGION_RADIUS * np.sqrt(rng.random(n_g))
    t = rng.random(n_g) * 2.0 * math.pi
    G = np.column_stack([r * np.cos(t), r * np.sin(t)])
    u = rng.random(n_g) * 2.0 * math.pi
    reff = rng.uniform(r_eff_low, r_eff_high, n_g)

    P = np.array(net, dtype=float)
    dx = P[None, :, 0] - G[:, None, 0]
    dy = P[None, :, 1] - G[:, None, 1]
    d = np.hypot(dx, dy)
    ang = np.arctan2(dy, dx)
    diff = np.abs((ang - u[:, None] + math.pi) % (2 * math.pi) - math.pi)
    facing = diff <= math.pi / 2.0 + 1e-12
    seen = (d <= reff[:, None]) & facing
    seen_min = (d <= R_MIN) & facing
    omni_min = d <= R_MIN
    return {
        "miss_dir_avg": float(1.0 - seen.any(axis=1).mean()),
        "miss_dir_worst_r": float(1.0 - seen_min.any(axis=1).mean()),
        "miss_omni_worst_r": float(1.0 - omni_min.any(axis=1).mean()),
    }


# ------------------------------------------------------------------ B 策略层
class SparseFallback(StrategyP4):
    """稀疏安全网 + 异常触发补网。"""

    def __init__(self, sparse_k: int = 18, trigger_below: int = 10,
                 force_full: bool = False, **kwargs) -> None:
        super().__init__(**kwargs)
        self.sparse_k = int(sparse_k)
        self.trigger_below = int(trigger_below)
        self.force_full = bool(force_full)
        self.phase_used = 0
        self.early_stopped = False

    def _phase(self, transport, pts, channels, bearings, cleared, result,
               state) -> None:
        for index, p in enumerate(pts):
            self._sweep(transport, p, channels, bearings, cleared, result)
            found = sum(1 for ch in channels
                        if len(bearings[ch]) >= self.early_stop_bearings)
            if found >= self.max_total:
                self.early_stopped = True
                return
            if index + 1 < len(pts):
                self._opportunistic_clear(transport, channels, bearings, cleared,
                                          result, p, pts[index + 1], state)
                if len(cleared) >= self.max_total:
                    self.early_stopped = True
                    return

    def run(self, transport, channel_order=None, do_enter=True):
        result = StrategyResult()
        if do_enter:
            transport.enter()
        channels = list(channel_order) if channel_order else list(CHANNELS)
        bearings: Dict[int, list] = {ch: [] for ch in channels}
        cleared: Set[int] = set()
        state: Dict[str, object] = {}
        self.early_stop_bearings = getattr(self, "early_stop_bearings", 2)
        self.phase_used = 0
        self.early_stopped = False

        tour = self._tour_order()
        k = max(1, min(int(self.sparse_k), len(tour)))
        self._phase(transport, tour[:k], channels, bearings, cleared, result, state)
        self.phase_used = 1
        rest = list(tour[k:])
        if not self.early_stopped and rest:
            if self.force_full or len(cleared) < self.trigger_below:
                self._phase(transport, rest, channels, bearings, cleared, result,
                            state)
                self.phase_used = 2
        self._finalize_channel_clearing(transport, channels, bearings, cleared,
                                        result)
        transport.exit()
        result.cleared = len(cleared)
        result.total = self._evaluate_total(transport)
        result.virtual_time = self._virtual_time(transport)
        result.move_distance = self._move_distance(transport)
        return result


def bench(factory, n: int, ratio: float, seeds: int) -> Dict[str, float]:
    times, moves, missed, bad, trig = [], [], 0, 0, 0
    for s in range(seeds):
        jammers = random_case(n=n, seed=s, directional_ratio=ratio)
        sim = MockSimulator(jammers, seed=s)
        st = factory()
        res = st.run(sim)
        times.append(res.mean_clear_time)
        moves.append(res.move_distance / 1000.0)
        missed += max(0, len(jammers) - res.cleared)
        if res.cleared < len(jammers):
            bad += 1
        if getattr(st, "phase_used", 0) == 2:
            trig += 1
    return {"time": statistics.fmean(times), "move": statistics.fmean(moves),
            "missed": missed, "bad": bad, "trig": trig, "seeds": seeds}


def main() -> None:
    tour = list(StrategyP4()._tour_order())
    L = sum(math.dist(tour[i], tour[i + 1]) for i in range(len(tour) - 1))
    print("22 点网：点数 %d，开放巡游 %.2f km" % (len(tour), L / 1000.0))

    print("\n===== A. 几何层：各规模网点的漏测概率（30 万次蒙特卡洛）=====")
    print("%-5s %-18s %-20s %-18s"
          % ("点数", "定向漏测(平均r)", "定向漏测(r=1000保守)", "全向漏测(r=1000)"))
    for k in (14, 16, 17, 18, 19, 20, 21, 22):
        pr = miss_probability(tour[:k])
        print("%-5d %-18.3e %-20.3e %-18.3e"
              % (k, pr["miss_dir_avg"], pr["miss_dir_worst_r"],
                 pr["miss_omni_worst_r"]))

    print("\n===== B. 策略层：稀疏网 + 触发补网（全定向 N=16，12 例）=====")
    rows = [
        ("完整 22 点网（基线）", lambda: StrategyP4()),
        ("稀疏 18 点（不补网）", lambda: SparseFallback(18, trigger_below=0)),
        ("稀疏 18 + 触发(清<10)", lambda: SparseFallback(18, 10)),
        ("稀疏 16 点（不补网）", lambda: SparseFallback(16, trigger_below=0)),
        ("稀疏 16 + 触发(清<10)", lambda: SparseFallback(16, 10)),
        ("稀疏 16 + 无条件补网", lambda: SparseFallback(16, 10, force_full=True)),
    ]
    print("%-24s %8s %8s %7s %8s %8s"
          % ("方案", "s/源", "行程km", "漏清源", "漏测案例", "补网次数"))
    for label, factory in rows:
        r = bench(factory, n=16, ratio=1.0, seeds=12)
        print("%-24s %8.1f %8.2f %7d %8d %8d"
              % (label, r["time"], r["move"], r["missed"], r["bad"], r["trig"]))


if __name__ == "__main__":
    main()

