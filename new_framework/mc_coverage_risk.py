# -*- coding: utf-8 -*-
"""概率安全评估：不同点数"定向安全网"的漏测率蒙特卡洛（问题四工程折中叙事用）。

口径
----
* 干扰源位置：圆域内**按面积均匀** r = 1800·√U、方位均匀；
* 定向方向 u：均匀（全向源不存在此风险，故只评估定向）；
* 有效接收半径 r_eff：**两种口径**
    - 最坏口径 r_eff = R_MIN = 1000 m（题面只给区间、未给分布时的保守取值）；
    - 均匀口径 r_eff ~ U[1000, 1500]（Mock/题面参数的自然假设）。
* "被看到"判据：存在网点 p 使 |p-G| ≤ r_eff 且 (p-G)·u ≥ 0（定向源半平面）。

输出：单源漏测率、16 源案例"至少漏 1 个"的概率、以及各网点集的最远可见距。
用法：python new_framework/mc_coverage_risk.py --samples 200000
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.coverage import two_ring_design          # noqa: E402
from src.config import REGION_RADIUS, R_MAX, R_MIN  # noqa: E402

NETS = {
    "22点(1000×9+1875×12) 证书网": lambda: two_ring_design(1000.0, 9, 1875.0, 12, 0.0),
    "24点(950×8+1850×16)": lambda: two_ring_design(950.0, 8, 1850.0, 16, 0.0),
    "19点(1000×8+1875×10)": lambda: two_ring_design(1000.0, 8, 1875.0, 10, 0.0),
    "18点(1000×7+1875×10)": lambda: two_ring_design(1000.0, 7, 1875.0, 10, 0.0),
    "16点(1000×7+1875×8)": lambda: two_ring_design(1000.0, 7, 1875.0, 8, 0.0),
}


def evaluate(samples: int, seed: int = 20260913):
    rng = np.random.default_rng(seed)
    r = REGION_RADIUS * np.sqrt(rng.random(samples))
    th = rng.random(samples) * 2.0 * math.pi
    G = np.column_stack([r * np.cos(th), r * np.sin(th)])
    u = rng.random(samples) * 2.0 * math.pi
    ux, uy = np.cos(u), np.sin(u)
    r_eff = R_MIN + (R_MAX - R_MIN) * rng.random(samples)

    rows = []
    for name, factory in NETS.items():
        P = np.asarray(factory(), dtype=float)
        vec = G[:, None, :] - P[None, :, :]
        d = np.linalg.norm(vec, axis=2)
        dot = vec[:, :, 0] * ux[:, None] + vec[:, :, 1] * uy[:, None]
        front = dot >= -1e-9
        seen_worst = ((d <= R_MIN + 1e-9) & front).any(axis=1)
        seen_real = ((d <= r_eff[:, None] + 1e-12) & front).any(axis=1)
        p_w = 1.0 - float(seen_worst.mean())
        p_r = 1.0 - float(seen_real.mean())
        rows.append({
            "net": name, "points": len(P),
            "miss_worst": p_w, "miss_uniform": p_r,
            "case_risk_worst": 1.0 - (1.0 - p_w) ** 16,
            "case_risk_uniform": 1.0 - (1.0 - p_r) ** 16,
            "max_visible_distance": float(d.min(axis=1).max()),
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=200000)
    args = ap.parse_args()
    rows = evaluate(args.samples)
    print("%-26s %6s %12s %12s %12s %12s" %
          ("网型", "点数", "单源漏测(最坏)", "单源漏测(均匀)", "单例风险(最坏)", "单例风险(均匀)"))
    for r in rows:
        print("%-26s %6d %12.2e %12.2e %11.3f%% %11.3f%%" %
              (r["net"], r["points"], r["miss_worst"], r["miss_uniform"],
               r["case_risk_worst"] * 100, r["case_risk_uniform"] * 100))
    print("\n注：单例风险 = 1-(1-p)^16（16 个定向源中至少一个从未被任何网点看到）。")
    print("三次正式测试中至少一次出现漏清的近似概率 ≈ 1-(1-单例风险)^3：")
    for r in rows:
        print("  %-26s 最坏口径 %.2f%% / 均匀口径 %.2f%%" %
              (r["net"], (1 - (1 - r["case_risk_worst"]) ** 3) * 100,
               (1 - (1 - r["case_risk_uniform"]) ** 3) * 100))


if __name__ == "__main__":
    main()
