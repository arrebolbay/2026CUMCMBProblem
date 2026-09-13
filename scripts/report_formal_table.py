"""把模拟器测试结果汇总成论文表格（题目表 1 格式），供正式测试后填写。

用法
----
    :: 列出所有"疑似正式测试"的记录（不含真值的那些，即正式测试）
    python scripts/report_formal_table.py

    :: 同时列出演练记录（演练会带 truth 字段，可据此核对）
    python scripts/report_formal_table.py --practice

    :: 只统计某一道题
    python scripts/report_formal_table.py --problem 3

数据来源：``scripts/run_simulator.py`` 每次测试后写入的
``results/p<problem>_run_<时间戳>.json``，其中
    case_code            测试案例编码（模拟器"日志列表"中也长期显示，可核对）
    cleared_jammers      本程序实际清除的干扰源个数（由 /clear 成功计数得到；
                         正式测试不公布真值，此数只依赖本程序自身记录）
    mean_clear_time_s    平均定位清除时间 = 虚拟时间 / 已清除源数（题目定义）
    program_run_time_s   程序运行时间 = /enter 到测试结束的现实时长（秒）

注意：演练记录含 ``truth`` 字段（干扰源总数等真值），正式测试没有该字段，
      因此本脚本默认只列 ``truth`` 缺失的记录作为正式测试结果。
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import math
import os
import sys
from typing import Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

PLACEHOLDER = "<参赛队号>"


def recover_from_log(log_path: str) -> Optional[dict]:
    """当结果 JSON 缺失（如程序在写报告前崩溃）时，从行为日志重建四项数据。

    同时做"漏清自检"：检查 8 个全频道覆盖扫描点是否全部到达 —— 全部到达即
    覆盖证书成立，任何全向源都必被看到，故"已清除数"就是真值。
    """
    entries = []
    try:
        with io.open(log_path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except ValueError:
                    continue
    except OSError:
        return None
    if not entries:
        return None

    seen, cleared, positions, ts = {}, set(), [], []
    for d in entries:
        path = d.get("path")
        req = d.get("request") or {}
        resp = d.get("response") or {}
        if isinstance(d.get("ts"), (int, float)):
            ts.append(d["ts"])
        if path in ("/measure", "/clear"):
            pos = req.get("position")
            if isinstance(pos, (list, tuple)) and len(pos) == 2:
                positions.append((float(pos[0]), float(pos[1])))
        if path == "/measure":
            ch = req.get("channel")
            if resp.get("measure_result") in ("direction", "near"):
                seen[ch] = seen.get(ch, 0) + 1
        elif path == "/clear" and resp.get("clear_result") == "success":
            cleared.add(req.get("channel"))

    virtual = None
    for d in entries:
        r = d.get("response") or {}
        if r.get("accepted") and isinstance(r.get("virtual_time_s"), (int, float)):
            virtual = float(r["virtual_time_s"])
    if not cleared or virtual is None:
        return None

    # 覆盖骨干自检
    sys.path.insert(0, ROOT)
    from src.coverage import seven_point_design
    from src.strategy_p3 import StrategyP3
    strat = StrategyP3()
    ring = seven_point_design(strat.cover_ring_radius, strat.cover_ring_count)
    full_scan = [ring[i] for i in range(0, len(ring), strat.cover_probe_stride)]
    reached = 0
    for target in full_scan:
        dist = min((math.dist(target, q) for q in positions), default=float("inf"))
        if dist <= 30.0:
            reached += 1

    return {
        "recovered": True,
        "log_file": log_path,
        "cleared_jammers": len(cleared),
        "mean_clear_time_s": virtual / len(cleared),
        "total_virtual_time_s": virtual,
        "program_run_time_s": (max(ts) - min(ts)) if len(ts) >= 2 else None,
        "coverage_backbone": "%d/%d" % (reached, len(full_scan)),
        "no_miss_proof": reached == len(full_scan),
        "unresolved": [],
        "case_code": None,
        "run_mode": "live",
        "run_kind": None,
        "label": None,
    }


def load_runs(problem: int, include_practice: bool):
    """读取 results/p<problem>_run_*.json，返回按时间排序的记录列表。

    ``run_mode`` 为 ``live`` 的是一次真实模拟器运行；``mock`` 为离线自检。
    正式测试请用 ``--label 正式测试N`` 标注，本脚本会优先汇总带标签的记录。
    """
    rows = []
    pattern = os.path.join(ROOT, "results", f"p{problem}_run_*.json")
    for path in sorted(glob.glob(pattern)):
        try:
            with io.open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            continue
        mode = data.get("run_mode")
        if mode is None:                     # 兼容加标记之前的历史记录
            mode = "mock" if data.get("truth") is None and \
                "mock" in str(data.get("base_url", "")) else "live"
        if mode != "live" and not include_practice:
            continue
        rows.append((os.path.basename(path), data, mode))

    # 结果 JSON 缺失的运行（程序曾在写报告前崩溃）：从行为日志恢复
    have = {name.replace(".json", "") for name, _, _ in rows}
    for log in sorted(glob.glob(os.path.join(ROOT, "logs", f"p{problem}_run_*.jsonl"))):
        stamp = os.path.basename(log).replace(".jsonl", "")
        if stamp in have:
            continue
        rec = recover_from_log(log)
        if rec is None:
            continue
        official = _match_official(problem, log)
        if official is None:
            continue          # 只恢复能与"官方测试记录"对上的运行，避免把 Mock 混进来
        rec["case_code"] = official.get("case_code")
        rec["run_kind"] = "formal" if official.get("formal_index") else "practice"
        rec["formal_index"] = official.get("formal_index")
        rows.append((os.path.basename(log) + "（由日志恢复）", rec, "live"))
    return rows


def _match_official(problem: int, log_path: str) -> Optional[dict]:
    """把一次运行与模拟器官方的测试记录对上（按时间邻近）。

    时间窗收紧到 [-30, +180] 秒：正式测试的 result.json 在测试结束后立刻落盘，
    而练习/Mock 运行不会与之邻近，从而避免误配。
    """
    sys.path.insert(0, ROOT)
    try:
        from src.simulator_truth import DEFAULT_SIM_DATA_DIR, list_results
    except Exception:
        return None
    try:
        mtime = os.path.getmtime(log_path)
        best, best_dt = None, None
        for item in list_results(DEFAULT_SIM_DATA_DIR, problem):
            dt = item.get("_mtime", 0) - mtime
            if -30.0 <= dt <= 180.0 and (best_dt is None or abs(dt) < abs(best_dt)):
                best, best_dt = item, dt
        return best
    except Exception:
        return None


def show(problem: int, include_practice: bool, only_labeled: bool) -> int:
    rows = load_runs(problem, include_practice)
    print(f"\n================ 问题 {problem} ================")
    if not rows:
        print("  暂无「真实模拟器」运行记录。正式测试请这样运行：")
        print(f"    python scripts/run_simulator.py --robot-id <参赛队号> "
              f"--problem {problem} --port 2026 \\")
        print("        --case-code <界面上的案例编码> --label 正式测试1")
        return 0

    print("%-4s %-22s %-8s %-13s %-11s %-10s %s"
          % ("序号", "测试案例编码", "清除个数", "平均清除时间", "程序运行时间",
             "标签", "记录文件"))
    formal = []
    for i, (name, d, mode) in enumerate(rows, 1):
        code = d.get("case_code") or "—"
        cleared = d.get("cleared_jammers")
        mean_t = d.get("mean_clear_time_s")
        wall = d.get("program_run_time_s")
        raw_label = d.get("label")
        kind = d.get("run_kind")
        if not raw_label and kind == "formal":
            raw_label = "正式#%s" % (d.get("formal_index") or "?")
        shown = raw_label or ("（演练）" if mode != "live" else "—")
        print("%-4d %-22s %-8s %-13s %-11s %-10s %s"
              % (i, str(code), str(cleared),
                 "%.2f" % mean_t if isinstance(mean_t, (int, float)) else "—",
                 "%.1f s" % wall if isinstance(wall, (int, float)) else "—",
                 str(shown)[:9], name))
        if mode == "live":
            formal.append((shown, code, cleared, mean_t, wall,
                           d.get("total_virtual_time_s"), kind,
                           d.get("formal_index")))

    if only_labeled:
        formal = [r for r in formal if r[6] == "formal"]
    # 按"正式测试序号"排序（缺失的排在后），保证 测试1/2/3 与官方次序一致
    formal.sort(key=lambda r: (r[7] is None, r[7] if r[7] else 0))
    if formal:
        print("\n---- 可直接填入论文表格（题目表 1 格式；取最近 3 次）----")
        print("%-18s %-14s %-16s %-12s"
              % ("测试案例编码", "清除干扰源个数", "平均定位清除时间", "程序运行时间"))
        for i, row in enumerate(formal[-3:], 1):
            label, code, cleared, mean_t, wall, vtime, kind, fidx = row
            print("%-18s %-14s %-16s %-12s"
                  % (f"测试 {i} {code}", cleared,
                     "%.2f" % mean_t if isinstance(mean_t, (int, float)) else "—",
                     "%.1f s" % wall if isinstance(wall, (int, float)) else "—"))
        print("\n  【四项口径说明】")
        print("   测试案例编码      = 模拟器【日志列表】中长期显示的案例编码；")
        print("                       运行时用 --case-code 抄录，可与该列表核对。")
        print("   清除干扰源个数    = 本程序 /clear 成功计数（正式测试不公布真值，"
              "此数只依赖自身记录）。")
        print("   平均定位清除时间  = 虚拟时间 / 已清除源数（题目定义）。")
        print("   程序运行时间      = /enter 到测试结束的**现实**时长（题目定义）：")
        print("                       题目明确“每次检测的 5 秒只增加虚拟时间、"
              "不要求在现实中等待”，")
        print("                       故本程序现实运行时间很短（通常几秒~1 分钟），"
              "远优于 20 分钟上限。")
        print("   备查：虚拟时间（题目内部时钟）见各记录 JSON 的 total_virtual_time_s。")
        print("\n  【为什么正式测试也能说“100% 清除”】")
        print("   正式测试不公布真值。但覆盖骨干的 8 个全频道扫描点全部到达时，")
        print("   依覆盖证书（最坏覆盖 998.25 m < R_min = 1000 m）任一全向源都必被看到；")
        print("   此时“有信号的频道全部清除、且无未解决频道”即证明已清空本局所有源。")
        print("   逐次自检：python scripts/diag_run_check.py <该次的 logs/p*_run_*.jsonl>")
    else:
        print("\n  提示：暂无带标签的真实运行记录。若已完成正式测试，"
              "请确认运行时加了 --label。")
    print("\n别忘了：从模拟器【日志列表】导出该次正式测试的加密日志（文件名勿改），")
    print("        放入支撑材料 zip 的 正式测试加密日志/ 目录。")
    return len(formal)


def main() -> None:
    ap = argparse.ArgumentParser(description="汇总正式测试结果，生成论文表格")
    ap.add_argument("--problem", type=int, choices=(3, 4),
                    help="只统计指定问题；缺省时两道题都统计")
    ap.add_argument("--practice", action="store_true",
                    help="同时列出离线自检（mock）记录")
    ap.add_argument("--labeled-only", action="store_true",
                    help="只汇总带 --label 的记录（推荐正式测试使用）")
    args = ap.parse_args()
    problems = [args.problem] if args.problem else [3, 4]
    total = 0
    for p in problems:
        total += show(p, args.practice, args.labeled_only)
    print(f"\n共找到 {total} 条可用记录。")


if __name__ == "__main__":
    main()
