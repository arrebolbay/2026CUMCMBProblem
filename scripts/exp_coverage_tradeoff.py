"""对照实验：**牺牲覆盖率换速度**的几条路线（不改动任何主策略文件）。

包含三组：
  A. 删减覆盖点（问题3 的 8 点网、问题4 的 22 点网）——直接少扫点。
  B. 乐观半径布点——按"r_eff ≈ 1500 u"而不是保守的 R_min = 1000 m 布点。
  C. 面积梯度游走（strategy_gradient.StrategyP3Gradient）——不预置布点。

统一口径：s/源（mean_clear_time）、行程 km、**未清源总数**、**有漏测的案例数**。
用法：
    python scripts/exp_coverage_tradeoff.py --problem 3 --seeds 30
    python scripts/exp_coverage_tradeoff.py --problem 4 --seeds 30
"""
from __future__ import annotations

import argparse
import math
import statistics
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.coverage import (directional_survey_design,  # noqa: E402
                          directional_certificate, seven_point_design,
                          two_ring_design)
from src.mock_simulator import MockSimulator, random_case  # noqa: E402
from src.strategy_p3 import StrategyP3  # noqa: E402
from src.strategy_p4 import StrategyP4  # noqa: E402
from src.strategy_fast import StrategyP3Fast, StrategyP4Fast  # noqa: E402
from src.strategy_gradient import (StrategyP3Gradient,  # noqa: E402
                                   StrategyP4Gradient)

Point = Tuple[float, float]


def p3_base_design() -> List[Point]:
    """问题3 的默认 8 点覆盖网（中心 + 半径 1000 m 七均布）。"""
    return seven_point_design(1000.0, 7)


def drop_points(design: Sequence[Point], keep_ring: int) -> List[Point]:
    """保留中心点 + 前 ``keep_ring`` 个环上点（其余删掉）。"""
    center = [p for p in design if math.hypot(*p) <= 1e-9]
    ring = [p for p in design if math.hypot(*p) > 1e-9]
    return center + ring[:keep_ring]


def run_case(strategy_factory, jammer_seed: int, n: int, ratio: float):
    jammers = random_case(n=n, seed=jammer_seed, directional_ratio=ratio)
    sim = MockSimulator(jammers, seed=jammer_seed)
    result = strategy_factory().run(sim)
    return result, len(jammers)


def bench(label: str, factory, n: int, ratio: float, seeds: int) -> dict:
    times, moves, unresolved, bad_cases, missed = [], [], 0, 0, 0
    for s in range(seeds):
        result, total = run_case(factory, s, n, ratio)
        times.append(result.mean_clear_time)
        moves.append(result.move_distance / 1000.0)
        unresolved += len(result.unresolved)
        missed += max(0, total - result.cleared)
        if result.unresolved or result.cleared < total:
            bad_cases += 1
    return {
        "label": label, "time": statistics.fmean(times),
        "move": statistics.fmean(moves), "unresolved": unresolved,
        "missed": missed, "bad_cases": bad_cases, "seeds": seeds, "n": n,
    }


def report(rows: List[dict]) -> None:
    print(f"{'方案':36s} {'s/源':>7s} {'行程km':>7s} {'漏清源':>6s} "
          f"{'漏测案例':>8s} {'清除率':>7s}")
    for r in rows:
        denom = max(1, r["seeds"] * r["n"])
        print(f"{r['label']:36s} {r['time']:7.1f} {r['move']:7.2f} "
              f"{r['missed']:6d} {r['bad_cases']:8d} "
              f"{100.0*(1-r['missed']/denom):6.2f}%")


def subsample_ring(ring: Sequence[Point], k: int) -> List[Point]:
    """从环上点里按角度均匀取 k 个（用于构造"减点"对照网）。"""
    if k >= len(ring):
        return list(ring)
    idx = sorted({int(round(i * len(ring) / k)) % len(ring) for i in range(k)})
    return [ring[i] for i in idx]


def p4_shrink(inner_k: int, outer_k: int) -> List[Point]:
    """问题4 减点网：中心 + 内环取 inner_k 个 + 外环取 outer_k 个。"""
    base = directional_survey_design()
    inner = [p for p in base if 1.0 <= math.hypot(*p) < 1400.0]
    outer = [p for p in base if math.hypot(*p) >= 1400.0]
    return [(0.0, 0.0)] + subsample_ring(inner, inner_k) + \
        subsample_ring(outer, outer_k)


def main() -> None:
    ap = argparse.ArgumentParser(description="牺牲覆盖率换速度的对照实验")
    ap.add_argument("--problem", type=int, choices=(3, 4), default=3)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--ratio", type=float, default=0.0,
                    help="定向源比例（问题4 常用 1.0 全定向）")
    args = ap.parse_args()

    rows: List[dict] = []
    if args.problem == 3:
        rows.append(bench("★ P3 主策略(默认骨干,全覆盖)",
                          lambda: StrategyP3(),
                          args.n, args.ratio, args.seeds))
        base = p3_base_design()
        rows.append(bench("P3 8点网(删点对照基线)",
                          lambda: StrategyP3(survey_points=base),
                          args.n, args.ratio, args.seeds))
        for k in (6, 5, 4, 3):
            pts = drop_points(base, k)
            rows.append(bench(
                f"P3 {k+1}点(删环点,无证书)",
                lambda pts=pts: StrategyP3(survey_points=pts),
                args.n, args.ratio, args.seeds))
        rows.append(bench("P3 面积梯度(不预置布点)",
                          lambda: StrategyP3Gradient(),
                          args.n, args.ratio, args.seeds))
        for b in (1, 4):
            rows.append(bench(
                f"P3 访问优先(探索预算{b})",
                lambda b=b: StrategyP3Fast(explore_budget=b),
                args.n, args.ratio, args.seeds))
    else:
        rows.append(bench("P4 22点(基准,全覆盖)",
                          lambda: StrategyP4(),
                          args.n, args.ratio, args.seeds))
        for ik, ok in ((8, 10), (6, 8), (4, 6)):
            pts = p4_shrink(ik, ok)
            rows.append(bench(
                f"P4 {len(pts)}点(内{ik}+外{ok},无证书)",
                lambda pts=pts: StrategyP4(survey_points=pts),
                args.n, args.ratio, args.seeds))
        rows.append(bench("P4 面积梯度(不预置布点)",
                          lambda: StrategyP4Gradient(),
                          args.n, args.ratio, args.seeds))
        for b in (1, 4):
            rows.append(bench(
                f"P4 访问优先(探索预算{b})",
                lambda b=b: StrategyP4Fast(explore_budget=b),
                args.n, args.ratio, args.seeds))
    report(rows)


if __name__ == "__main__":
    main()

