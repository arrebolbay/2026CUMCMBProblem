"""实验D-最终：多候选网（证书 + 端到端）综合选优。"""
import math
import statistics
import sys

sys.path.insert(0, r"G:\C\CUMCM B题")
sys.path.insert(0, r"G:\C\CUMCM B题\scripts")

from src.coverage import directional_certificate  # noqa: E402
from src.strategy_p4 import StrategyP4  # noqa: E402
from _Dtest import bench, build  # noqa: E402

CANDS = [
    ("基准 950x12+1875x12", 950.0, 12, 1875.0, 12, 0.5),
    ("950x12+1860x14", 950.0, 12, 1860.0, 14, 0.5),
    ("950x12+1845x16", 950.0, 12, 1845.0, 16, 0.5),
    ("950x12+1832x18", 950.0, 12, 1832.0, 18, 0.5),
    ("950x11+1865x13", 950.0, 11, 1865.0, 13, 0.5),
    ("950x10+1860x14", 950.0, 10, 1860.0, 14, 0.5),
    ("1000x10+1875x12", 1000.0, 10, 1875.0, 12, 0.5),
]
SEEDS = 40
print(f"证书 + 巡游 + 端到端（{SEEDS} 例/档，3 档 × N=10/16）× ")
hdr = f"{'候选':22s} {'巡游km':>7s} {'角隙°':>7s} {'fail':>5s}"
for r, tag in ((1.0, "全定向"), (0.5, "半定向"), (0.0, "全向")):
    for n in (10, 16):
        hdr += f" {tag[:2]}N{n:<2d}"
print(hdr)
summary = []
for label, r1, n1, r2, n2, ph in CANDS:
    pts = build(r1, n1, r2, n2, ph)
    cert = directional_certificate(pts, n_radial=80, n_angular=720,
                                   n_boundary=7200)
    tl = StrategyP4._strong_tour((0.0, 0.0), pts[1:], rounds=10, warm=4)
    L = sum(math.dist(tl[i], tl[i + 1]) for i in range(len(tl) - 1))
    line = f"{label:22s} {L/1000:7.3f} {cert['max_gap_deg']:7.2f}" \
           f" {cert['fail']:5d}"
    vals = []
    for ratio, _ in ((1.0, 0), (0.5, 0), (0.0, 0)):
        for n in (10, 16):
            t, m, u, a = bench(pts, ratio, n, seeds=SEEDS)
            line += f" {t:5.0f}"
            vals.append(t)
    summary.append((statistics.fmean(vals), label, cert["fail"]))
    print(line)
print("\n按 6 档平均排序（越小越好）：")
for avg, label, fail in sorted(summary):
    print(f"  {avg:7.2f} s/源  {label}  (cert_fail={fail})")
