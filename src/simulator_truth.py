"""读取模拟器本地数据目录中的演练真值（只读，不修改模拟器任何文件）。

背景
----
* 机器狗 HTTP 接口**不会**返回干扰源个数，也不返回有效接收半径（附件2 明确规定）；
* **演练测试**结束后，模拟器界面显示真值，同时在数据目录写入明文结果文件：

      JammersSimulatorData/behavior-logs/practice-p3-<run_no>-<case_code>.result.json

  内容形如::

      {"version":2,"problem_no":3,"practice_run_no":4621099077388779008,
       "case_code":"NYJT-TNFF-7CNM-DG69","window_started_at_utc":"...",
       "ended_at_utc":"...","jammer_count":11,
       "omnidirectional_jammer_count":11,"directional_jammer_count":0}

* 同目录 *.jlog / *.psum 为**加密**行为日志（JMBPLOG1 / JMBPSUM1 封装）；
  正式测试只生成加密日志且界面不显示真值，故正式测试无法读取真值。

本模块仅用于演练统计与论文表1（案例编码 / 清除比例），策略本身不依赖真值。
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

DEFAULT_SIM_DATA_DIR = (
    r"F:\BaiduNetdiskDownload\CUMCM2026B\Jammers-simulator-win64"
    r"\Jammers-simulator\JammersSimulatorData"
)

__all__ = [
    "DEFAULT_SIM_DATA_DIR",
    "read_result_file",
    "list_results",
    "find_latest_result",
    "wait_for_result_after",
]


def read_result_file(path: str) -> Dict[str, Any]:
    """读取单个 *.result.json（明文），返回字典。"""
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path} 不是 JSON 对象")
    return data


def list_results(data_dir: str = DEFAULT_SIM_DATA_DIR,
                 problem_no: Optional[int] = None) -> List[Dict[str, Any]]:
    """列出结果文件（按修改时间升序），每项附带 _path / _mtime。"""
    logs_dir = os.path.join(data_dir, "behavior-logs")
    if not os.path.isdir(logs_dir):
        return []
    out: List[Dict[str, Any]] = []
    for name in os.listdir(logs_dir):
        if not name.endswith(".result.json"):
            continue
        if problem_no is not None and f"-p{problem_no}-" not in name:
            continue
        path = os.path.join(logs_dir, name)
        try:
            data = read_result_file(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        data["_path"] = path
        data["_mtime"] = os.path.getmtime(path)
        out.append(data)
    out.sort(key=lambda d: d["_mtime"])
    return out


def find_latest_result(data_dir: str = DEFAULT_SIM_DATA_DIR,
                       problem_no: Optional[int] = None,
                       since_mtime: Optional[float] = None) -> Optional[Dict[str, Any]]:
    """取最近一次结果；since_mtime 给定时只接受该时刻之后写入的文件。"""
    results = list_results(data_dir, problem_no)
    if since_mtime is not None:
        results = [r for r in results if r["_mtime"] >= since_mtime - 1.0]
    return results[-1] if results else None


def wait_for_result_after(data_dir: str = DEFAULT_SIM_DATA_DIR,
                          since_mtime: Optional[float] = None,
                          problem_no: Optional[int] = None,
                          timeout_s: float = 20.0,
                          poll_s: float = 1.0) -> Optional[Dict[str, Any]]:
    """等待本次测试的结果文件出现（测试结束后可能延迟数秒写入）。"""
    deadline = time.time() + timeout_s
    while True:
        found = find_latest_result(data_dir, problem_no, since_mtime)
        if found is not None:
            return found
        if time.time() >= deadline:
            return None
        time.sleep(poll_s)