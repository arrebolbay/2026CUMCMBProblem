"""实验D-细化：细粒度搜索巡游更短的定向安全网（中精度筛 + 精筛复核）。"""
import itertools
import math
import sys

sys.path.insert(0, r"G:\C\CUMCM B题")

from src.coverage import directional_certificate  # noqa: E402
from src.strategy_p4 import StrategyP4  # noqa: E402

START = (0.0, 0.0)


def build(r1, n1, r2, n2, phase):
    pts = [START]
    for i in range(n1):
        a = 2 * math.pi * i / n1
        pts.append((r1 * math.cos(a), r1 * math.sin(a)))
    for i in range(n2):
        a = 2 * math.pi * (i + phase) / n2
        pts.append((r2 * math.cos(a), r2 * math.sin(a)))
    return pts


def tour_len(pts):
    t = StrategyP4._strong_tour(START, [p for p in pts if p != START],
                                rounds=10, warm=4)
    return sum(math.dist(t[i], t[i + 1]) for i in range(len(t) - 1))


def mid(pts):
    return directional_certificate(pts, n_radial=20, n_angular=180,
                                   n_boundary=1440)


def fine(pts):
    return directional_certificate(pts, n_radial=80, n_angular=720,
                                   n_boundary=7200)


cands = []
for n2, r2 in ((12, 1875.0), (12, 1890.0), (13, 1876.0), (14, 1860.0)):
    for n1 in (8, 9, 10, 11, 12):
        for r1 in (850.0, 900.0, 950.0, 1000.0):
            for phase in (0.0, 0.25, 0.5, 0.75):
                cands.append((r1, n1, r2, n2, phase))
print(f"候选 {len(cands)} 个（中精度筛）…")
results = []
for r1, n1, r2, n2, phase in cands:
    pts = build(r1, n1, r2, n2, phase)
    c = mid(pts)
    if c["fail"] or c["max_gap_deg"] > 178.0:
        continue
    results.append((tour_len(pts), n1 + n2 + 1, r1, n1, r2, n2, phase,
                    c["max_gap_deg"], pts))
results.sort()
print(f"中精度通过 {len(results)} 个；按巡游升序前 8（已用 rounds=10 强巡游）：")
print("  巡游km  点数  r1   n1   r2    n2  ph   角隙°")
for L, np_, r1, n1, r2, n2, ph, gap, _ in results[:8]:
    print(f"  {L/1000:6.3f}  {np_:4d} {r1:5.0f} {n1:3d} {r2:7.1f} {n2:3d}"
          f"  {ph:4.2f}  {gap:6.2f}")
print("\n精筛复核：")
best = None
for L, np_, r1, n1, r2, n2, ph, gap, pts in results[:8]:
    f = fine(pts)
    tag = "OK  " if f["fail"] == 0 else "FAIL"
    print(f"  [{tag}] {L/1000:6.3f}km {np_:2d}点 (r1={r1:.0f}×{n1} +"
          f" r2={r2:.1f}×{n2} ph={ph}) fail={f['fail']}"
          f" 角隙={f['max_gap_deg']:.2f}°")
    if f["fail"] == 0 and (best is None or L < best[0]):
        best = (L, pts, n1, r1, n2, r2, ph, f["max_gap_deg"])
cur = build(950.0, 12, 1875.0, 12, 0.5)
cf = fine(cur)
print(f"\n基准: 巡游{tour_len(cur)/1000:.3f}km fail={cf['fail']}"
      f" 角隙={cf['max_gap_deg']:.2f}°")
if best:
    print(f"最优候选: 巡游{best[0]/1000:.3f}km 角隙{best[7]:.2f}°"
          f" (r1={best[2]:.0f}×{best[1]}, r2={best[5]:.1f}×{best[4]}, ph={best[6]})"
          f"  净省 {(tour_len(cur)-best[0])/1000:.3f} km"
          f" = {(tour_len(cur)-best[0])/5.0/16:.2f} s/源")
