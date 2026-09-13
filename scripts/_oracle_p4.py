"""Oracle：问题4"清源路径反哺覆盖"的理论可回收里程。

思路（用户提出）：25 点双环只是"定向安全覆盖"的**充分条件**；真正必需的只是
"实际 measure 点集构成合法定向覆盖证书"。而"走到源清除"的路径本身也是
measure 点（清除点做 discharge 即成为合法覆盖点）。于是：
    基线 = 25 点定向安全网巡游 + 收尾清源 TSP（两段）
    Oracle = TSP 过「16 个源位置(清除必走) + 贪心补洞点(凑够定向安全证书)」
可回收里程 = 基线 − Oracle。若平均 > 0.7 km/源 ⇒ 该方向值得重兵投入。

注意：源位置用**真值**（离线 Oracle，只用于估计上限，不用于正式策略）。
"""
import math
import statistics
import sys

sys.path.insert(0, r"G:\C\CUMCM B题")

from src.coverage import directional_certificate, directional_survey_design
from src.geometry import distance
from src.mock_simulator import random_case


def open_tour_len(start, points, rounds=8, warm_starts=3):
    """开放巡游长度（最近邻 + 2-opt + Or-opt，多次起点取最短）。"""
    pts = list(points)
    if not pts:
        return 0.0
    best = None
    for k in range(warm_starts):
        head = pts[k]
        rest = pts[:k] + pts[k + 1:]
        tour = [start, head]
        pool = list(rest)
        while pool:
            nxt = min(pool, key=lambda q: distance(tour[-1], q))
            pool.remove(nxt)
            tour.append(nxt)
        n = len(tour)
        for _ in range(rounds):
            imp = False
            for i in range(n - 2):
                for j in range(i + 2, n - 1):
                    a, b, c, d = tour[i], tour[i + 1], tour[j], tour[j + 1]
                    if (distance(a, b) + distance(c, d)
                            > distance(a, c) + distance(b, d) + 1e-9):
                        tour[i + 1:j + 1] = tour[i + 1:j + 1][::-1]
                        imp = True
            for seg in (1, 2, 3):
                s = 1
                while s + seg < len(tour):
                    blk = tour[s:s + seg]
                    prev = tour[s - 1]
                    nxt = tour[s + seg] if s + seg < len(tour) else None
                    out = distance(prev, blk[0])
                    if nxt is not None:
                        out += distance(blk[-1], nxt) - distance(prev, nxt)
                    rest2 = tour[:s] + tour[s + seg:]
                    b2 = None
                    for t in range(len(rest2) - 1):
                        u, v = rest2[t], rest2[t + 1]
                        add = (distance(u, blk[0]) + distance(blk[-1], v)
                               - distance(u, v))
                        if b2 is None or add < b2[0]:
                            b2 = (add, t + 1)
                    tail = distance(rest2[-1], blk[0])
                    if b2 is None or tail < b2[0]:
                        b2 = (tail, len(rest2))
                    if b2[0] < out - 1e-9:
                        tour = rest2[:b2[1]] + blk + rest2[b2[1]:]
                        imp = True
                    else:
                        s += 1
            if not imp:
                break
        L = sum(distance(tour[i], tour[i + 1]) for i in range(len(tour) - 1))
        if best is None or L < best:
            best = L
    return best


def cover_candidates():
    """候选补洞点：外环(1875, 每 10°) + 内环(950, 每 30°) + 中心。"""
    out = [(0.0, 0.0)]
    out += [(1875.0 * math.cos(2 * math.pi * i / 36),
             1875.0 * math.sin(2 * math.pi * i / 36)) for i in range(36)]
    out += [(950.0 * math.cos(2 * math.pi * i / 12),
             950.0 * math.sin(2 * math.pi * i / 12)) for i in range(12)]
    return out


def oracle_cover(src, cands):
    """源位置 + 贪心补洞点，直到定向安全证书通过。返回 (点数, max_gap_deg)。"""
    pts = list(src)
    used = []
    for _ in range(12):
        cert = directional_certificate(pts)
        if cert["ok"]:
            return len(used), cert["max_gap_deg"]
        best, best_gap = None, 1e9
        for c in cands:
            if any(distance(c, q) < 1.0 for q in pts):
                continue
            gap = directional_certificate(pts + [c])["max_gap_deg"]
            if gap < best_gap:
                best_gap, best = gap, c
        if best is None:
            break
        pts.append(best)
        used.append(best)
    return len(used), directional_certificate(pts)["max_gap_deg"]


design = [(0.0, 0.0)] + list(directional_survey_design())
cands = cover_candidates()
rows = []
for seed in range(15):
    jam = random_case(n=16, seed=seed, directional_ratio=1.0)
    src = [tuple(j.position) for j in jam]
    # 基线：25 点巡游（开放）+ 收尾从最后点到所有源的巡游
    base_cov = open_tour_len((0.0, 0.0), design)
    base_tail = open_tour_len(design[-1], src)
    base = base_cov + base_tail
    # Oracle：源 + 补洞点的合并巡游
    n_extra, gap = oracle_cover(src, cands)
    orc = open_tour_len((0.0, 0.0), src + [c for c in cands][:0])  # 占位
    orc = open_tour_len((0.0, 0.0), src)     # 源本身巡游
    # 补洞点在 oracle 里也要走：用 oracle_cover 返回的 used
    pts = list(src)
    for _ in range(n_extra):
        best, best_gap = None, 1e9
        for c in cands:
            if any(distance(c, q) < 1.0 for q in pts):
                continue
            g = directional_certificate(pts + [c])["max_gap_deg"]
            if g < best_gap:
                best_gap, best = g, c
        if best is not None:
            pts.append(best)
    orc = open_tour_len((0.0, 0.0), pts)
    rows.append((base / 1000, orc / 1000, n_extra, gap))

print(f"{'seed':>4} {'基线km':>8} {'Oraclekm':>9} {'补洞点':>6} {'角隙°':>7}")
for i, (b, o, ne, g) in enumerate(rows):
    print(f"{i:>4} {b:8.2f} {o:9.2f} {ne:>6} {g:7.2f}")
mb = statistics.fmean(r[0] for r in rows)
mo = statistics.fmean(r[1] for r in rows)
print(f"\n平均：基线 {mb:.2f} km | Oracle {mo:.2f} km | "
      f"可回收 {mb-mo:.2f} km = {(mb-mo)*1000/5/16:.1f} s/源（N=16）")
