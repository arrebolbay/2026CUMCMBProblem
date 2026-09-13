"""实验D：搜索"更少点的定向安全双环网"（证书 + 巡游长度双重筛选）。"""
import itertools
import math
import sys

sys.path.insert(0, r"G:\C\CUMCM B题")

from src.coverage import directional_certificate  # noqa: E402
from src.strategy_p4 import StrategyP4  # noqa: E402

START = (0.0, 0.0)


def build(r1, n1, r2, n2, phase, with_center=True):
    pts = [START] if with_center else []
    for i in range(n1):
        a = 2 * math.pi * i / n1
        pts.append((r1 * math.cos(a), r1 * math.sin(a)))
    for i in range(n2):
        a = 2 * math.pi * (i + phase) / n2
        pts.append((r2 * math.cos(a), r2 * math.sin(a)))
    return pts


def tour_len(pts):
    t = StrategyP4._strong_tour(START, [p for p in pts if p != START],
                                rounds=6, warm=3)
    return sum(math.dist(t[i], t[i + 1]) for i in range(len(t) - 1))


def coarse(pts):
    return directional_certificate(pts, n_radial=24, n_angular=144,
                                   n_boundary=720)


def fine(pts):
    return directional_certificate(pts, n_radial=80, n_angular=720,
                                   n_boundary=7200)


def mid(pts):
    return directional_certificate(pts, n_radial=20, n_angular=180,
                                   n_boundary=1440)


results = []
cands = []
for n2 in (12, 14, 16, 18, 20, 24, 30):
    r2min = 1800.0 / math.cos(math.pi / n2)
    for dr in (3.0, 15.0):
        for n1 in (10, 12, 14, 16):
            for r1 in (800.0, 950.0, 1050.0):
                for phase in (0.0, 0.5):
                    cands.append((r1, n1, r2min + dr, n2, phase))
print(f"候选 {len(cands)} 个，中精度筛查（20×180 + 1440 边界）…")
for r1, n1, r2, n2, phase in cands:
    pts = build(r1, n1, r2, n2, phase)
    c = mid(pts)
    if c["fail"] or c["max_gap_deg"] > 178.0:
        continue
    results.append((tour_len(pts), n1 + n2 + 1, r1, n1, r2, n2, phase,
                    c["max_gap_deg"], pts))

results.sort()
print(f"中精度通过 {len(results)} 个；按巡游升序前 10：")
print("  巡游km  点数  r1   n1   r2    n2  ph   角隙°")
for L, np_, r1, n1, r2, n2, ph, gap, _ in results[:10]:
    print(f"  {L/1000:6.3f}  {np_:4d} {r1:5.0f} {n1:3d} {r2:7.1f} {n2:3d}"
          f"  {ph:4.2f}  {gap:6.2f}")

print("\n精筛（80×720 + 7200 边界）：")
ok_any = False
for L, np_, r1, n1, r2, n2, ph, gap, pts in results[:6]:
    f = fine(pts)
    flag = "OK  " if f["fail"] == 0 else "FAIL"
    if f["fail"] == 0:
        ok_any = True
    print(f"  [{flag}] 巡游{L/1000:6.3f}km {np_:2d}点"
          f" (r1={r1:.0f}×{n1} + r2={r2:.1f}×{n2} ph={ph})"
          f" fail={f['fail']} 角隙={f['max_gap_deg']:.2f}°")
cur = build(950.0, 12, 1875.0, 12, 0.5)
cf = fine(cur)
print(f"\n基准 25 点网: 巡游{tour_len(cur)/1000:.3f}km fail={cf['fail']}"
      f" 角隙={cf['max_gap_deg']:.2f}°   ← 有无候选全面超越: {ok_any}")
