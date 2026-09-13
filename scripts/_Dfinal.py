"""实验D-最终基准：22 点网 vs 基准 25 点网（修复探测步长后，300+ 例）。"""
import statistics
import sys

sys.path.insert(0, r"G:\C\CUMCM B题")
sys.path.insert(0, r"G:\C\CUMCM B题\scripts")

from _Dtest import bench, build  # noqa: E402

BASE = build(950.0, 12, 1875.0, 12, 0.5)
N22 = build(1000.0, 9, 1875.0, 12, 0.0)
N23 = build(1000.0, 10, 1875.0, 12, 0.5)
SEEDS = 50

for label, pts in (("基准 25 点(950x12+1875x12)", BASE),
                   ("22 点(1000x9+1875x12)", N22),
                   ("23 点(1000x10+1875x12)", N23)):
    print(f"\n### {label}")
    tot, ust = [], 0
    for ratio, tag in ((1.0, "全定向"), (0.5, "半定向"), (0.0, "全向")):
        for n in (10, 16):
            t, m, u, a = bench(pts, ratio, n, seeds=SEEDS)
            tot.append(t)
            ust += u
            print(f"   {tag} N={n:<2d}: {t:6.1f} s/源  行程{m:5.2f}km  "
                  f"动作{a:4.0f}  未清{u}")
    print(f"   → 6 档平均 {statistics.fmean(tot):.2f} s/源, 未清合计 {ust}")
