"""实验：提高首次清除命中率（问题3 的真正瓶颈）。

**实测瓶颈**（12 例/档）：首次清除误差中位 14.3~15.6 m、**命中率仅 62.5%/70.3%**，
平均每源要 2.4~2.5 次 `/clear`。每次失败 = 3 s + 往返行程，且触发近场收口（更多行程）。
**根因**：`_source_target` 用"最小覆盖圆圆心"，其误差可达 `pursue_mec_limit=80 m`；
而清除半径只有 20 m。

**变体**：
  M 收紧 `pursue_mec_limit`（只在估计足够准时才前往，否则先补示向度）
  N 到达估计点后**先测一次向**（拿到近距离高精度示向度）再清除
  O 用**最小二乘交会**替代最小覆盖圆圆心作为清除点
  P M+N
"""
from __future__ import annotations

import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.geometry import (                                        # noqa: E402
    distance, localization_region, min_enclosing_circle,
)
from src.mock_simulator import MockSimulator, random_case         # noqa: E402
from src.strategy_p3 import (                                     # noqa: E402
    StrategyP3, bearing_least_squares,
)


class V(StrategyP3):
    """可开关变体：mec 阈值 / 到点先测向 / 最小二乘估计点。"""

    def __init__(self, mec=None, premeasure=False, use_lsq=False, insert=None,
                 sliver=False, bbox=None):
        super().__init__()
        if mec is not None:
            self.pursue_mec_limit = mec
        if insert is not None:
            self.insert_limit = insert
        if bbox is not None:
            self.region_bbox = bbox
        self.insert_sliver_sources = bool(sliver)
        self.premeasure = premeasure
        self.use_lsq = use_lsq

    def _source_target(self, records, here=None):
        if self.use_lsq and len(records) >= 2:
            est = bearing_least_squares(records)
            if est is not None:
                poly = localization_region([p for p, _ in records],
                                           [b for _, b in records])
                if poly:
                    _, radius = min_enclosing_circle(poly)
                    if radius <= self.pursue_mec_limit:
                        return (float(est[0]), float(est[1]))
        return super()._source_target(records, here)

    def _fast_clear(self, transport, channel, estimate, records, result):
        if self.premeasure:
            # 先在估计点做一次测向：近距离示向度的横向误差极小（d·tan1°），
            # 与已有示向度交会可把定位区域压到几米级，再清除。
            resp = transport.measure(estimate, channel)
            result.measures += 1
            kind = resp.get("measure_result")
            if kind == "near":
                if self._try_clear(transport, channel, estimate, result):
                    return True
            elif kind == "direction":
                records = list(records) + [(tuple(estimate), float(resp["svd_deg"]))]
                refined = bearing_least_squares(records)
                if refined is not None and distance(refined, estimate) <= 600.0:
                    if self._try_clear(transport, channel, refined, result):
                        return True
        return super()._fast_clear(transport, channel, estimate, records, result)


def bench(n, seeds=12, **kw):
    times, moves, meas, clr, ur = [], [], [], [], 0
    hit = []
    for seed in range(seeds):
        jam = random_case(n=n, seed=seed)
        sim = MockSimulator(jam, seed=seed)
        res = V(**kw).run(sim)
        times.append(res.mean_clear_time)
        moves.append(res.move_distance / 1000.0)
        meas.append(res.measures)
        clr.append(res.clears + res.failed_clears)
        ur += len(res.unresolved)
        src = {j.channel: tuple(j.position) for j in jam}
        seen = {}
        for e in sim.log:
            if e["path"] == "/clear" and e["channel"] not in seen \
                    and e["channel"] in src:
                seen[e["channel"]] = distance(tuple(e["position"]),
                                              src[e["channel"]])
        hit += [1.0 if v <= 20.0 else 0.0 for v in seen.values()]
    return (statistics.fmean(times), statistics.fmean(moves),
            statistics.fmean(meas), statistics.fmean(clr),
            100 * statistics.fmean(hit), ur)


if __name__ == "__main__":
    print("目标 176~300；下界 272.6/235.6/205.4/184.0")
    print("格式：时间s / 行程km / 检测 / clear次数 / 首击命中%")
    for label, kw in [
        ("A 基线(拒插细长)", {}),
        ("mec=120", dict(mec=120.0)),
        ("mec=160", dict(mec=160.0)),
        ("mec=200", dict(mec=200.0)),
        ("insert=4000", dict(insert=4000.0)),
        ("mec160+insert4000", dict(mec=160.0, insert=4000.0)),
        ("mec200+insert6000", dict(mec=200.0, insert=6000.0)),
        ("细长也插(对照)", dict(sliver=True, bbox=1800.0)),
    ]:
        cells, bad = [], 0
        for n in (10, 16):
            t, mv, ms, c, h, u = bench(n, **kw)
            cells.append(f"{t:6.1f}/{mv:5.2f}/{ms:5.0f}/{c:5.1f}/{h:4.1f}%")
            bad += u
        print(f"{label:16s} " + " | ".join(cells) + f"  未清={bad}")
