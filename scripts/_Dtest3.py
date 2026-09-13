"""实验D-极值：r1=1000（中心可见性上限）下压点数。"""
import math
import statistics
import sys

sys.path.insert(0, r"G:\C\CUMCM B题")
sys.path.insert(0, r"G:\C\CUMCM B题\scripts")

from src.coverage import directional_certificate  # noqa: E402
from src.strategy_p4 import StrategyP4  # noqa: E402
from _Dtest import bench, build  # noqa: E402

CANDS = [
    ("1000x8+1875x12 (21)", 1000.0, 8, 1875.0, 12, 0.0),
    ("1000x9+1875x12 (22)", 1000.0, 9, 1875.0, 12, 0.0),
    ("1000x10+1875x12(23)", 1000.0, 10, 1875.0, 12, 0.5),
    ("1000x11+1875x12(24)", 1000.0, 11, 1875.0, 12, 0.5),
    ("1000x10+1860x13(24)", 1000.0, 10, 1860.0, 13, 0.5),
    ("1000x10+1850x14(25)", 1000.0, 10, 1850.0, 14, 0.5),
    ("1000x9+1860x14 (24)", 1000.0, 9, 1860.0, 14, 0.5),
    ("基准 950x12+1875x12", 950.0, 12, 1875.0, 12, 0.5),
]
SEEDS = 25
print(f"候选（{SEEDS} 例/档）")
hdr = f"{'候选':22s} {'巡游km':>7s} {'角隙°':>7s} {'fail':>5s}"
for tag in ("全定", "半定", "全向"):
    for n in (10, 16):
        hdr += f" {tag}N{n:<2d}"
print(hdr)
summary = []
for label, r1, n1, r2, n2, ph in CANDS:
    pts = build(r1, n1, r2, n2, ph)
    cert = directional_certificate(pts, n_radial=80, n_angular=720,
                                   n_boundary=7200)
    tl = StrategyP4._strong_tour((0.0, 0.0), pts[1:], rounds=10, warm=4)
    L = sum(math.dist(tl[i], tl[i + 1]) for i in range(len(tl) - 1))
    line = f"{label:22s} {L/1000:7.3f} {cert['max_gap_deg']:7.2f} {cert['fail']:5d}"
    vals = []
    for ratio in (1.0, 0.5, 0.0):
        for n in (10, 16):
            t, m, u, a = bench(pts, ratio, n, seeds=SEEDS)
            line += f" {t:5.0f}"
            vals.append(t)
    summary.append((statistics.fmean(vals), label, cert["fail"]))
    print(line)
print("\n按 6 档平均排序：")
for avg, label, fail in sorted(summary):
    print(f"  {avg:7.2f} s/源  {label}  (cert_fail={fail})")
