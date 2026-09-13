"""实验A-优化器：直接以 J(pi)=E[L(t_stop)] 为目标搜索扫描顺序。

t_stop(pi) = max over 16 sources of「该源第 2 个可见点在 pi 中的步号」
            （|H|<=1 的源记 t=24，即无论如何都扫满 —— 与在线一致）
L(t)        = 走完 pi 前 t+1 个点的累计路程
目标         : min E[L(t_stop)]，比「最短 Hamilton 路」更贴近真实提前结束口径
"""
import math
import random
import statistics
import sys

sys.path.insert(0, r"G:\C\CUMCM B题")

from src.coverage import directional_survey_design  # noqa: E402
from src.strategy_p4 import StrategyP4  # noqa: E402

PTS = directional_survey_design()
NI = len(PTS)
LAST = NI - 1
INIT = StrategyP4()._tour_order()          # [(0,0)] + 24 点
START = INIT[0]

# ---- 预计算虚拟源的可见集 H（返回点序号列表） ----
def sample_H(m, seed):
    rng = random.Random(seed)
    out = []
    for _ in range(m):
        r = 1800.0 * math.sqrt(rng.random())
        th = rng.random() * 2 * math.pi
        ang = rng.uniform(0.0, 360.0)
        ux, uy = math.cos(math.radians(ang)), math.sin(math.radians(ang))
        re = rng.uniform(1000.0, 1500.0)
        G = (r * math.cos(th), r * math.sin(th))
        H = []
        for i, (px, py) in enumerate(PTS):
            dx, dy = px - G[0], py - G[1]
            if dx * dx + dy * dy <= re * re and dx * ux + dy * uy >= -1e-9:
                H.append(i)
        out.append(H)
    return out


H_POOL = sample_H(200 * 16, seed=11)
CASES = [H_POOL[i * 16:(i + 1) * 16] for i in range(200)]

IDX = {tuple(round(c, 6) for c in p): i for i, p in enumerate(PTS)}


def parse(order):
    """[(0,0)] + 24 点  ->  点序号排列（中心=0 恒在首位）+ 累计路程。"""
    seq = [IDX[tuple(round(c, 6) for c in START)]]
    for p in order[1:]:
        seq.append(IDX[tuple(round(c, 6) for c in p)])
    cum = [0.0]
    for i in range(len(order) - 1):
        cum.append(cum[-1] + math.dist(order[i], order[i + 1]))
    return seq, cum


def objective(order):
    seq, cum = parse(order)
    rk = [0] * NI
    for step, i in enumerate(seq):
        rk[i] = step
    tot = 0.0
    for case in CASES:
        t = 0
        for H in case:
            if len(H) >= 2:
                v = sorted(rk[i] for i in H)[1]
            else:
                v = LAST
            if v > t:
                t = v
        tot += cum[t]
    return tot / len(CASES)


def anneal(order, iters=4000, seed=3):
    rng = random.Random(seed)
    cur = list(order)
    val = objective(cur)
    best, bestv = list(cur), val
    for it in range(iters):
        cand = list(cur)
        if rng.random() < 0.5:                       # swap
            i, j = rng.randrange(1, len(cand)), rng.randrange(1, len(cand))
            cand[i], cand[j] = cand[j], cand[i]
        else:                                        # 2-opt 段反转
            i = rng.randrange(1, len(cand) - 1)
            j = rng.randrange(i + 1, len(cand))
            cand[i:j] = cand[i:j][::-1]
        v = objective(cand)
        T = max(0.5, 90.0 * (1 - it / iters))
        if v < val or rng.random() < math.exp((val - v) / T):
            cur, val = cand, v
            if v < bestv:
                best, bestv = list(cand), v
    return best, bestv


print(f"起点顺序 J = {objective(INIT)/1000:.3f} km（t_stop 处平均已走路程）")
best, bv = anneal(INIT, iters=6000, seed=3)
print(f"优化后 J = {bv/1000:.3f} km   （{len(CASES)} 组 16 源案例）")
print("优化后顺序（步号:点序号）:", parse(best)[0])
print("原顺序            （步号:点序号）:", parse(INIT)[0])

# 保存供策略实验使用
import json  # noqa: E402
with open(r"G:\C\CUMCM B题\_A2order.json", "w", encoding="utf-8") as fh:
    json.dump({"order": [list(p) for p in best]}, fh)
print("已写出 _A2order.json")

# ---------------- 证伪检验 1：随机顺序的 J ----------------
rng2 = random.Random(99)
rv = []
for _ in range(40):
    perm = list(INIT[1:])
    rng2.shuffle(perm)
    rv.append(objective([START] + perm))
print(f"[证伪1] 随机顺序 J 均值 = {statistics.fmean(rv)/1000:.3f} km "
      f"(min {min(rv)/1000:.3f} / max {max(rv)/1000:.3f})"
      f"  vs 起点顺序 {objective(INIT)/1000:.3f} km")

# ---------------- 证伪检验 2：以「最短巡游」为目标优化 ----------------
def length_of(order):
    return sum(math.dist(order[i], order[i + 1]) for i in range(len(order) - 1))


def anneal_len(order, iters=6000, seed=5):
    rng = random.Random(seed)
    cur = list(order)
    val = length_of(cur)
    best, bestv = list(cur), val
    for it in range(iters):
        cand = list(cur)
        if rng.random() < 0.5:
            i, j = rng.randrange(1, len(cand)), rng.randrange(1, len(cand))
            cand[i], cand[j] = cand[j], cand[i]
        else:
            i = rng.randrange(1, len(cand) - 1)
            j = rng.randrange(i + 1, len(cand))
            cand[i:j] = cand[i:j][::-1]
        v = length_of(cand)
        if v < val or rng.random() < math.exp((val - v) / max(0.5, 60.0 * (1 - it / iters))):
            cur, val = cand, v
            if v < bestv:
                best, bestv = list(cand), v
    return best, bestv


b2, l2 = anneal_len(INIT, iters=6000, seed=5)
print(f"[证伪2] 最短巡游: 起点 {length_of(INIT)/1000:.3f} km -> 优化 {l2/1000:.3f} km"
      f" (省 {(length_of(INIT)-l2)/1000:.3f} km)，但其 J = {objective(b2)/1000:.3f} km")
