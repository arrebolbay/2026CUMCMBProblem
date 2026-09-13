"""实验A-诊断2：early=1(≥1 bearing) vs early=2(≥2 bearings) 的触发步号与代价。

- t_first = 16 个源全被"看到"（≥1 bearing）的步号 —— early=1 的停止点
- t_2nd   = 16 个源全拿到第 2 条 bearing 的步号 —— early=2 的停止点（当前）
- early=1 的额外代价：停止时仍只有 1 条示向度的源，必须由收尾兜底（追击/网格）
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
TOUR = StrategyP4()._tour_order()
POS = {tuple(round(c, 6) for c in p): i for i, p in enumerate(TOUR)}
RK = [POS[tuple(round(c, 6) for c in p)] for p in PTS]      # rk[i]=第几步访问点 i

# 累计路径长度（步号 -> 已走距离）
CUM = [0.0]
for i in range(len(TOUR) - 1):
    CUM.append(CUM[-1] + math.dist(TOUR[i], TOUR[i + 1]))


def visible(G, ux, uy, reff):
    out = []
    for i, (px, py) in enumerate(PTS):
        dx, dy = px - G[0], py - G[1]
        if dx * dx + dy * dy <= reff * reff and dx * ux + dy * uy >= -1e-9:
            out.append(i)
    return out


def sample(ns, rng):
    """生成 ns 个源的可见点集。"""
    out = []
    for _ in range(ns):
        r = 1800.0 * math.sqrt(rng.random())
        th = rng.random() * 2 * math.pi
        ang = rng.uniform(0.0, 360.0)
        G = (r * math.cos(th), r * math.sin(th))
        out.append(visible(G, math.cos(math.radians(ang)),
                           math.sin(math.radians(ang)),
                           rng.uniform(1000.0, 1500.0)))
    return out


rng = random.Random(7)
M = 20000
tf, t2, n_one, n_zero = [], [], [], []
for _ in range(M):
    srcs = sample(16, rng)
    f = max((min(RK[i] for i in H) if H else NI - 1) for H in srcs)
    s = max((sorted(RK[i] for i in H)[1] if len(H) >= 2 else NI - 1) for H in srcs)
    tf.append(f)
    t2.append(s)
    n_one.append(sum(1 for H in srcs if len(H) == 1))
    n_zero.append(sum(1 for H in srcs if not H))


def q(v, p):
    return sorted(v)[int(p * (len(v) - 1))]


print(f"【诊断4】16 源案例（{M} 次），步号分位 [中位 / P75 / P90]：")
print(f"   t_first (early=1 停止)  : {q(tf,0.5):2d} / {q(tf,0.75):2d} / {q(tf,0.90):2d}")
print(f"   t_2nd   (early=2 停止)  : {q(t2,0.5):2d} / {q(t2,0.75):2d} / {q(t2,0.90):2d}")
print(f"   差值 t_2nd-t_first      : {statistics.fmean(t2)-statistics.fmean(tf):.2f} 步（均值）")
print(f"   扫描距离 均值: early=2 停止处 {statistics.fmean([CUM[t] for t in t2])/1000:.2f} km"
      f" | 扫满 25 点 {CUM[-1]/1000:.2f} km"
      f" | early=1 停止处 {statistics.fmean([CUM[t] for t in tf])/1000:.2f} km")
print(f"   提前结束触发率: early=1 {(1-sum(1 for v in tf if v==NI-1)/M)*100:.1f}%"
      f"   early=2 {(1-sum(1 for v in t2 if v==NI-1)/M)*100:.1f}%")
print(f"   t_first 处「仅 1 条示向度」的源数：均值 {statistics.fmean(n_one):.2f}"
      f"  中位 {q(n_one,0.5)}  P90 {q(n_one,0.9)}")
print(f"   0 可见点（漏测风险）的源数：均值 {statistics.fmean(n_zero):.4f}"
      f"  max {max(n_zero)}")
