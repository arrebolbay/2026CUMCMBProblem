# -*- coding: utf-8 -*-
"""小算子 A/B 实验：1-bearing 频道的"射线—路径"顺路插入（问题三，Mock 模拟器）。

口径与 scripts/run_mock_experiments.py 完全一致（同一 Mock、同一随机案例：
random_case(n=16, seed=s)），因此可与既有基准直接对比：
    被清除比例（必须 1.0） / 平均定位清除时间 = 总虚拟时间 / 清除个数
另给出**配对差值**（同 seed 下与基线相比），避免案例难度差异掩盖结论。

用法：python new_framework/run_ab.py --seeds 25 --n 16
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from typing import Dict, List, Sequence, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.mock_simulator import MockSimulator, random_case           # noqa: E402
from src.strategy_p3 import StrategyP3                              # noqa: E402
from new_framework.strategy_p3_rayinsert import StrategyP3RayInsert  # noqa: E402


def make_strategy(tag: str):
    if tag == "base":
        return StrategyP3()
    if tag == "v1_200":
        return StrategyP3RayInsert(ray_variant=1, ray_insert_limit=200.0)
    if tag == "v2_100":
        return StrategyP3RayInsert(ray_variant=2, ray_insert_limit=100.0,
                                   ray_min_lateral=150.0, ray_max_lateral=400.0)
    if tag == "v2_200":
        return StrategyP3RayInsert(ray_variant=2, ray_insert_limit=200.0,
                                   ray_min_lateral=150.0, ray_max_lateral=400.0)
    if tag == "v2_400":
        return StrategyP3RayInsert(ray_variant=2, ray_insert_limit=400.0,
                                   ray_min_lateral=150.0, ray_max_lateral=400.0)
    raise ValueError(tag)


def run_one(tag: str, n: int, seed: int) -> Dict[str, float]:
    jammers = random_case(n=n, seed=seed)
    strategy = make_strategy(tag)
    sim = MockSimulator(jammers, seed=seed)
    res = strategy.run(sim)
    return {
        "ratio": res.clearance_ratio,
        "s_per_source": res.mean_clear_time,
        "move_km": res.move_distance / 1000.0,
        "measures": float(res.measures),
        "clears": float(res.clears),
        "ray_fired": float(getattr(strategy, "ray_insert_fired", 0)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=25)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--tags", type=str,
                    default="base,v1_200,v2_100,v2_200,v2_400")
    args = ap.parse_args()
    tags = [t for t in args.tags.split(",") if t]

    table: Dict[str, List[Dict[str, float]]] = {t: [] for t in tags}
    t0 = time.time()
    for seed in range(args.seeds):
        for tag in tags:
            table[tag].append(run_one(tag, args.n, seed))
        line = "  ".join("%s=%7.1f" % (t, table[t][-1]["s_per_source"]) for t in tags)
        print("seed=%2d N=%d  %s" % (seed, args.n, line))
    print("\n用时 %.0f s" % (time.time() - t0))

    base = table[tags[0]]
    print("\n%-10s %8s %8s %8s %8s %8s %8s %9s" %
          ("config", "mean", "median", "min", "move(km)", "meas", "ray插入", "Δs/源(配对)"))
    for tag in tags:
        rows = table[tag]
        ss = [r["s_per_source"] for r in rows]
        diffs = [rows[i]["s_per_source"] - base[i]["s_per_source"]
                 for i in range(len(rows))]
        print("%-10s %8.1f %8.1f %8.1f %8.2f %8.1f %8.2f %9.1f  (改善案例 %d/%d, 最差 %+.1f)" %
              (tag, statistics.fmean(ss), statistics.median(ss), min(ss),
               statistics.fmean([r["move_km"] for r in rows]),
               statistics.fmean([r["measures"] for r in rows]),
               statistics.fmean([r["ray_fired"] for r in rows]),
               statistics.fmean(diffs),
               sum(1 for d in diffs if d < -1e-9), len(diffs), max(diffs)))
    print("\n清除比例(最小): " + ", ".join(
        "%s=%.4f" % (t, min(r["ratio"] for r in table[t])) for t in tags))


if __name__ == "__main__":
    main()
