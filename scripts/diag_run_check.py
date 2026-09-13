"""事后自检：从行为日志判定一次测试是否"真的清完了"（不依赖任何真值）。

正式测试不公布干扰源总数，因此"是否漏清"必须靠**可自检的证据**判定，本脚本给出
三条判据：

  1. **覆盖骨干走完**：8 个全频道覆盖扫描点全部到达（最接近距离 ≤ 30 m）。
     由覆盖证书（最坏覆盖 998.25 m < R_min = 1000 m），任一全向源都必落入某个
     扫描点的有效接收范围内 ⇒ **不存在"从未被发现"的源**；
  2. **有信号频道全部清除**：凡曾返回 direction / near 的频道都在已清除集合中；
  3. **无未解决频道**：收尾阶段没有"有示向度但清不掉"的频道。

三条同时成立即可断言：**本局清除比例 = 1.000**，且报告中的"已清除源数"就是真值。

用法：
    python scripts/diag_run_check.py logs/p3_run_20260913_143144.jsonl
    python scripts/diag_run_check.py            # 缺省分析最新一份 p3 日志
"""
from __future__ import annotations

import glob
import io
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.coverage import seven_point_design          # noqa: E402
from src.strategy_p3 import StrategyP3               # noqa: E402


def analyse(path: str) -> dict:
    entries = []
    for line in io.open(path, encoding="utf-8", errors="replace"):
        line = line.strip()
        if line:
            try:
                entries.append(json.loads(line))
            except ValueError:
                pass

    seen, cleared, positions, per_channel = {}, set(), [], {}
    counts = {}
    virtual = None
    for d in entries:
        p = d.get("path")
        req, resp = d.get("request") or {}, d.get("response") or {}
        counts[p] = counts.get(p, 0) + 1
        if resp.get("accepted") and isinstance(resp.get("virtual_time_s"), (int, float)):
            virtual = float(resp["virtual_time_s"])
        if p in ("/measure", "/clear"):
            pos = req.get("position")
            if isinstance(pos, (list, tuple)) and len(pos) == 2:
                positions.append((float(pos[0]), float(pos[1])))
        if p == "/measure":
            ch = req.get("channel")
            if resp.get("measure_result") in ("direction", "near"):
                seen[ch] = seen.get(ch, 0) + 1
                per_channel[ch] = per_channel.get(ch, 0) + 1
        elif p == "/clear" and resp.get("clear_result") == "success":
            cleared.add(req.get("channel"))

    # 覆盖骨干自检
    strat = StrategyP3()
    ring = seven_point_design(strat.cover_ring_radius, strat.cover_ring_count)
    full_scan = [ring[i] for i in range(0, len(ring), strat.cover_probe_stride)]
    reached = 0
    for target in full_scan:
        dist = min((math.dist(target, q) for q in positions), default=float("inf"))
        if dist <= 30.0:
            reached += 1

    with_signal = {ch for ch, n in seen.items() if n}
    leftover = sorted(with_signal - cleared)
    n_meas, n_clr = counts.get("/measure", 0), counts.get("/clear", 0)
    n_fail = n_clr - len(cleared)
    move_s = (virtual or 0.0) - n_meas * 6.0 - (len(cleared) * 5.0 + n_fail * 3.0)

    return {
        "log": os.path.basename(path),
        "virtual_time_s": virtual,
        "move_km": move_s * 5.0 / 1000.0,
        "measures": n_meas,
        "clears": n_clr,
        "clear_fail": n_fail,
        "channels_with_signal": sorted(with_signal),
        "cleared_channels": sorted(cleared),
        "leftover": leftover,
        "coverage_backbone": "%d/%d" % (reached, len(full_scan)),
        "no_miss": (reached == len(full_scan)) and not leftover,
    }


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if path is None:
        found = sorted(glob.glob(os.path.join(ROOT, "logs", "p*_run_*.jsonl")))
        if not found:
            print("未找到行为日志")
            return
        path = found[-1]
    r = analyse(path)
    print("行为日志:", r["log"])
    print("  虚拟时间 %.1f s | 行程 %.2f km | 测量 %d | 清除 %d（失败 %d）"
          % (r["virtual_time_s"] or 0.0, r["move_km"], r["measures"],
             r["clears"], r["clear_fail"]))
    print("  有信号的频道 %d 个，已清除 %d 个，遗漏 %d 个"
          % (len(r["channels_with_signal"]), len(r["cleared_channels"]),
             len(r["leftover"])))
    print("  覆盖骨干到达 %s" % r["coverage_backbone"])
    print("  判定：", "未发现漏清（清除比例 = 1.000）✓" if r["no_miss"]
          else "⚠ 需要人工复查")


if __name__ == "__main__":
    main()
