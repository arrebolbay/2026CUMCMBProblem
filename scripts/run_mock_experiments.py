"""在 Mock 模拟器上批量演练问题3/问题4 策略，产出演练统计数据。

对应题目要求的两项统计量：
    * 被清除干扰源个数 / 干扰源总数（清除比例，必须为 1）
    * 平均定位清除时间 = 定位清除总时间 / 被清除干扰源个数

除均值外还输出中位数、P95、最大值与移动距离，便于对照"移动耗时占 80% 以上"
的耗时账本，判断优化是否真的作用在行程而不是动作次数上。

用法：
    python scripts/run_mock_experiments.py --problem 3 --cases 300
    python scripts/run_mock_experiments.py --problem 4 --cases 300 \
        --n 14 --directional-count 6
    python scripts/run_mock_experiments.py --problem 4 --cases 300 \
        --n 16 --directional-count 16 --label alldir
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.mock_simulator import MockSimulator, random_case   # noqa: E402
from src.strategy_p3 import StrategyP3                      # noqa: E402
from src.strategy_p4 import StrategyP4                      # noqa: E402


def percentile(values, q: float) -> float:
    """线性插值分位数（q ∈ [0, 1]），与 numpy 默认定义一致。"""
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    if len(ordered) == 1:
        return float(ordered[0])
    pos = q * (len(ordered) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", type=int, default=3, choices=[3, 4])
    ap.add_argument("--cases", type=int, default=20)
    ap.add_argument("--directional-ratio", type=float, default=0.0)
    ap.add_argument("--n", type=int, default=None,
                    help="固定干扰源总数（默认 10~16 随机）")
    ap.add_argument("--directional-count", type=int, default=None,
                    help="恰好 k 个定向源（其余全向），优先于 --directional-ratio")
    ap.add_argument("--insert-threshold", type=float, default=400.0,
                    help="问题4 机会式清除的额外路程阈值 (m)")
    ap.add_argument("--label", type=str, default="",
                    help="输出文件名后缀，用于区分不同实验配置")
    args = ap.parse_args()

    rows = []
    for seed in range(args.cases):
        jammers = random_case(
            n=args.n, seed=seed,
            directional_ratio=args.directional_ratio,
            directional_count=args.directional_count,
        )
        strategy = (StrategyP3() if args.problem == 3
                    else StrategyP4(insert_threshold=args.insert_threshold))
        sim = MockSimulator(jammers, seed=seed)
        res = strategy.run(sim)
        n_dir = sum(1 for j in jammers if j.direction is not None)
        rows.append({
            "seed": seed, "n_jammers": sim.total_jammers, "n_directional": n_dir,
            "cleared": res.cleared, "clearance_ratio": res.clearance_ratio,
            "virtual_time_s": res.virtual_time,
            "mean_clear_time_s": res.mean_clear_time,
            "move_distance_m": res.move_distance,
            "mean_move_distance_m": res.mean_move_distance,
            "measures": res.measures, "clears": res.clears,
            "failed_clears": res.failed_clears, "unresolved": res.unresolved,
        })
        print(f"seed={seed:3d} N={sim.total_jammers:2d}(定向{n_dir:2d}) 清除={res.cleared:2d} "
              f"比例={res.clearance_ratio:.3f} 总时间={res.virtual_time:8.1f}s "
              f"平均={res.mean_clear_time:7.2f}s "
              f"移动={res.move_distance / 1000.0:6.2f}km "
              f"检测={res.measures:3d} 清除={res.clears:3d} 失败={res.failed_clears:3d}")

    ratios = [r["clearance_ratio"] for r in rows]
    times = [r["mean_clear_time_s"] for r in rows]
    summary = {
        "problem": args.problem, "cases": args.cases,
        "n": args.n, "directional_ratio": args.directional_ratio,
        "directional_count": args.directional_count,
        "insert_threshold": args.insert_threshold if args.problem == 4 else None,
        "clearance_ratio_min": min(ratios),
        "clearance_ratio_mean": statistics.fmean(ratios),
        "all_cleared": all(abs(r - 1.0) < 1e-12 for r in ratios),
        "mean_clear_time_mean_s": statistics.fmean(times),
        "mean_clear_time_median_s": statistics.median(times),
        "mean_clear_time_p95_s": percentile(times, 0.95),
        "mean_clear_time_min_s": min(times),
        "mean_clear_time_max_s": max(times),
        "mean_clear_time_std_s": statistics.pstdev(times) if len(times) > 1 else 0.0,
        "avg_virtual_time_s": statistics.fmean([r["virtual_time_s"] for r in rows]),
        "avg_move_distance_km": statistics.fmean(
            [r["move_distance_m"] for r in rows]) / 1000.0,
        "avg_mean_move_distance_km": statistics.fmean(
            [r["mean_move_distance_m"] for r in rows]) / 1000.0,
        "avg_measures": statistics.fmean([r["measures"] for r in rows]),
        "avg_clears": statistics.fmean([r["clears"] for r in rows]),
        "avg_failed_clears": statistics.fmean([r["failed_clears"] for r in rows]),
        "cases_with_unresolved": sum(1 for r in rows if r["unresolved"]),
    }
    print("\n=== 汇总 ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    suffix = f"_{args.label}" if args.label else ""
    out = os.path.join(ROOT, "results",
                       f"mock_p{args.problem}{suffix}_experiments.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "rows": rows}, fh, ensure_ascii=False, indent=2)
    print(f"\n已写入 {out}")


if __name__ == "__main__":
    main()
