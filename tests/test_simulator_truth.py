"""演练真值读取模块测试（在临时目录内构造结果文件，不触碰真实模拟器数据）。"""

import json
import os
import time

from src.simulator_truth import (
    find_latest_result,
    list_results,
    read_result_file,
    wait_for_result_after,
)

SAMPLE = {
    "version": 2,
    "problem_no": 3,
    "practice_run_no": 4621099077388779008,
    "case_code": "NYJT-TNFF-7CNM-DG69",
    "window_started_at_utc": "2026-09-10T16:30:36.318Z",
    "ended_at_utc": "2026-09-10T16:30:42.859Z",
    "jammer_count": 11,
    "omnidirectional_jammer_count": 11,
    "directional_jammer_count": 0,
}


def _make(tmp_path, name="practice-p3-4621099077388779008-NYJT-TNFF-7CNM-DG69.result.json",
          payload=None):
    logs = tmp_path / "behavior-logs"
    logs.mkdir(parents=True, exist_ok=True)
    path = logs / name
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload or SAMPLE, fh)
    return str(tmp_path), str(path)


def test_read_result_file(tmp_path):
    _, path = _make(tmp_path)
    data = read_result_file(path)
    assert data["jammer_count"] == 11
    assert data["case_code"] == "NYJT-TNFF-7CNM-DG69"


def test_list_and_find_latest(tmp_path):
    data_dir, path = _make(tmp_path)
    results = list_results(data_dir, problem_no=3)
    assert len(results) == 1
    assert results[0]["_path"] == path
    assert find_latest_result(data_dir, problem_no=3)["jammer_count"] == 11
    # 频道/问题号过滤：要求不存在的结果文件时返回 None
    assert find_latest_result(data_dir, problem_no=4) is None
    # since_mtime 在未来 -> 视为本次测试没有新结果
    assert find_latest_result(data_dir, since_mtime=time.time() + 3600) is None


def test_missing_directory_returns_empty(tmp_path):
    assert list_results(str(tmp_path / "nope")) == []


def test_wait_for_result_after(tmp_path):
    data_dir, _ = _make(tmp_path)
    found = wait_for_result_after(data_dir, since_mtime=time.time() - 60,
                                  problem_no=3, timeout_s=2.0, poll_s=0.2)
    assert found is not None and found["jammer_count"] == 11
    none_found = wait_for_result_after(data_dir, since_mtime=time.time() + 3600,
                                       problem_no=3, timeout_s=0.5, poll_s=0.2)
    assert none_found is None


def test_encrypted_logs_are_ignored(tmp_path):
    data_dir, _ = _make(tmp_path)
    (tmp_path / "behavior-logs" / "practice-p3-1-AAAA.jlog").write_bytes(b"JMBPLOG1\x00\xff")
    results = list_results(data_dir, problem_no=3)
    assert len(results) == 1, "加密 .jlog/.psum 不应被当作结果文件"