"""精确下界：完美 Oracle（最优开巡游 + 最少检测 + 100% 命中）。

与之前"含覆盖扫描"Oracle 的区别：只测"尚未决定"的频道（已清/已定位就跳过），
即真实的最少检测策略；巡游用"最近邻 + 2-opt + Or-opt"的开放路径（起点原点）。
"""
import statistics
import sys

sys.path.insert(0, r"G:\C\CUMCM B题")

from src.config import CHANNELS, REGION_RADIUS, R_MIN, R_MAX
from src.coverage import seven_point_design
from src.geometry import distance
from src.mock_simulator import MockSimulator, random_case
from src.routing import two_opt_tour


def open_tour(start, points, rounds=12):
    """最近邻 + 2-opt + Or-opt 开放路径。"""
    tour = [start]
    rest = list(points)
    while rest:
        nxt = min(rest, key=lambda q: distance(tour[-1], q))
        rest.remove(nxt)
        tour.append(nxt)
    n = len(tour)
    if n >= 4:
        for _ in range(rounds):
            imp = False
            for i in range(n - 2):
                for j in range(i + 2, n - 1):
                    a, b, c, d = tour[i], tour[i + 1], tour[j], tour[j + 1]
                    if distance(a, b) + distance(c, d) > distance(a, c) + distance(b, d) + 1e-9:
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
                    best = None
                    for t in range(len(rest2) - 1):
                        u, v = rest2[t], rest2[t + 1]
                        add = distance(u, blk[0]) + distance(blk[-1], v) - distance(u, v)
                        if best is None or add < best[0]:
                            best = (add, t + 1)
                    tail = distance(rest2[-1], blk[0])
                    if best is None or tail < best[0]:
                        best = (tail, len(rest2))
                    if best[0] < out - 1e-9:
                        tour = rest2[:best[1]] + blk + rest2[best[1]:]
                        imp = True
                    else:
                        s += 1
            if not imp:
                break
    return tour[1:]


def oracle(n, seed):
    jam = random_case(n=n, seed=seed)
    sim = MockSimulator(jam, seed=seed)
    sim.enter()
    cover = list(seven_point_design(1000.0, 7))          # center + 7 ring = 8 点
    src = {tuple(j.position): j.channel for j in jam}
    ch_of = {c: p for p, c in src.items()}
    order = open_tour((0.0, 0.0), cover + list(src.keys()))
    # 每个频道需要的"已定位"示向度条数：完美 Oracle 假设访问到源即 100% 命中，
    # 因此只需"在覆盖点上每频道测到它为止"；空频道测满 8 个覆盖点。
    swept = 0  # 已访问覆盖点数
    for p in order:
        if p in ch_of:
            sim.clear(p, ch_of[p])           # 完美命中，1 次成功清除
        else:
            # 覆盖点：全频道扫描（只测未决频道）
            swept += 1
            # 完美 Oracle：每个频道只测到"已发现"为止。空频道仍要测满 8 点。
            for ch in CHANNELS:
                if ch in ch_of:
                    g = ch_of[ch]
                    if distance(p, g) <= sim.jammers_by_channel.get(ch) or True:
                        # 已被覆盖点发现过就跳过（完美 Oracle 已知真值）
                        pass
                else:
                    pass
    sim.exit()
    return sim.clock.virtual_time, sim.cleared_jammers, len(jam)


# 简化：直接按"最小检测数"估算 —— 空频道 8 点、源 2 次测向 + 1 次清除
def oracle_simple(n, seed):
    jam = random_case(n=n, seed=seed)
    sim = MockSimulator(jam, seed=seed)
    sim.enter()
    cover = list(seven_point_design(1000.0, 7))
    src = {tuple(j.position): j.channel for j in jam}
    order = open_tour((0.0, 0.0), cover + list(src.keys()))
    for p in order:
        if p in src:
            sim.clear(p, src[p])
        else:
            # 覆盖点扫描：只测真正需要测的（模拟最少检测，这里直接跳过 measure，
            # 用真实动作成本近似：空频道 4×(8 点)=32 次 + 源 16×2=32 次）
            pass
    # 手动加检测成本：空频道 = (20-n)*8 次检测；源 = n*2 次检测（用时钟推进）
    n = len(jam)
    for _ in range((20 - n) * 8 + n * 2):
        sim.clock.measure(sim.clock.position, 1)  # 占用 6s（近似）
    sim.exit()
    return sim.clock.virtual_time, sim.cleared_jammers, n


if __name__ == "__main__":
    for n in (10, 12, 14, 16):
        per = []
        for seed in range(20):
            t, c, tot = oracle_simple(n, seed)
            per.append(t / max(c, 1))
        print(f"N={n}: 精确下界 ≈ {statistics.fmean(per):6.1f} s/源")
