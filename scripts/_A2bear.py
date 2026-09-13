"""实验A-诊断：25 点网下"每个源可见点数 |H|"与"第二个 bearing 的位置"。

物理（与 mock_simulator 一致）：
    源 s=(G,u,r_eff) 在检测点 P 可见  <=>  |P-G| <= r_eff 且 (P-G)·u >= 0
    |H(s)| <= 1  =>  该源永远拿不到第 2 条示向度 => 提前结束不可能触发（必扫满）
"""
import math
import random
import statistics
import sys

sys.path.insert(0, r"G:\C\CUMCM B题")

from src.coverage import directional_survey_design  # noqa: E402
from src.strategy_p4 import StrategyP4  # noqa: E402

PTS = directional_survey_design()
N = len(PTS)


def visible(G, ux, uy, reff):
    out = []
    for i, (px, py) in enumerate(PTS):
        dx, dy = px - G[0], py - G[1]
        if dx * dx + dy * dy <= reff * reff and dx * ux + dy * uy >= -1e-9:
            out.append(i)
    return out


def sample_sources(m, seed=0, mode="directional"):
    rng = random.Random(seed)
    out = []
    for _ in range(m):
        r = 1800.0 * math.sqrt(rng.random())
        th = rng.random() * 2 * math.pi
        ang = rng.uniform(0.0, 360.0) if mode == "all" else rng.uniform(0.0, 360.0) * 0
        if mode == "mixed":          # 一半全向一半定向
            ang = rng.uniform(0.0, 360.0) if rng.random() < 0.5 else None
        elif mode == "all":
            ang = rng.uniform(0.0, 360.0)
        ux, uy = (1.0, 0.0) if ang is None else (math.cos(math.radians(ang)),
                                                 math.sin(math.radians(ang)))
        out.append(((r * math.cos(th), r * math.sin(th)), ux, uy,
                    rng.uniform(1000.0, 1500.0), ang is None))
    return out


M = 20000
srcs = sample_sources(M, seed=1, mode="all")
hist = {}
one = 0
for G, ux, uy, re, omn in srcs:
    k = len(visible(G, ux, uy, re))
    hist[k] = hist.get(k, 0) + 1
    if k <= 1:
        one += 1
print(f"【诊断1】{M} 个虚拟定向源在 25 点网上的 |可见点数| 分布：")
print("   " + "  ".join(f"|H|={k}:{100.0*v/M:.1f}%" for k, v in sorted(hist.items())))
print(f"   |H|<=1（永远拿不到第2条示向度 => 提前结束不可触发）= {100.0*one/M:.2f}%")

# 当前扫描顺序
tour = StrategyP4()._tour_order()
pos = {tuple(round(c, 6) for c in p): i for i, p in enumerate(tour)}
order_idx = []
for p in PTS:
    order_idx.append(pos[tuple(round(c, 6) for c in p)])
print(f"【诊断2】扫描顺序 rk[i]=该覆盖点在第几步被访问（0 起）：")
print("   " + " ".join(f"{i}:{order_idx[i]}" for i in range(N)))

# t_stop = 最后一个源拿到第 2 条 bearing 的步号
r2 = []
for G, ux, uy, re, omn in srcs:
    H = visible(G, ux, uy, re)
    if len(H) < 2:
        r2.append(N)                       # 扫满 25 点也拿不到
    else:
        r2.append(sorted(order_idx[i] for i in H)[1])
r2.sort()
print("【诊断3】\"第2条 bearing 出现步号\"分位："
      f" 中位={r2[M//2]}  P90={r2[int(0.9*M)]}  P99={r2[int(0.99*M)]}  max={r2[-1]}")
print(f"   平均={statistics.fmean(r2):.2f}  提前结束步号(max over sources)= {r2[-1]}")
print(f"   步号>=23（即尾两点才有）的比例 = {100.0*sum(1 for v in r2 if v >= 23)/M:.1f}%")
