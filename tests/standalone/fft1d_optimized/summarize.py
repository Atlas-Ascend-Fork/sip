# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Audit complete records and report paired ACL Event medians without raw device data."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import statistics


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    a = p.parse_args()
    identity = json.loads((a.root / "identity.json").read_text())
    records = json.loads((a.root / "records.json").read_text())
    progress = json.loads((a.root / "progress.json").read_text())
    expected = {
        (tuple(c), label, round_id): samples
        for c, samples, rounds, versions in identity["cases"]
        for round_id in range(rounds)
        for label in versions
    }
    actual = {}
    values = {}
    for r in records:
        key = (tuple(r["case"]), r["variant"], r["round"])
        if (
            key in actual
            or key not in expected
            or r["samples"] != expected[key]
            or r["status"] != "PASS"
        ):
            raise RuntimeError("Unexpected, duplicate or failed record")
        n, b, dim, op, direction, pattern, alias = r["case"]
        path = a.root / r["timings"]
        log = path.with_name("run.log")
        if sha(log) != r["log_sha256"]:
            raise RuntimeError("Log hash mismatch")
        result = re.findall(r"^RESULT (.*)$", log.read_text(), re.M)
        if len(result) != 1:
            raise RuntimeError("Missing unique API result")
        fields = dict(re.findall(r"(\w+)=(\S+)", result[0]))
        contract = dict(
            dim=dim,
            op=op,
            N=n,
            M=n,
            batch=b,
            direction=direction,
            device=identity["device"],
            normalization="none",
            samples=r["samples"],
            warmup=10,
            profiling=0,
            input_unchanged=1,
            alias=int(alias),
        )
        rows = b * n ** (dim - 1)
        real_bytes = rows * n * 4
        half_bytes = rows * (n // 2 + 1) * 8
        contract.update(
            input_bytes=real_bytes * 2
            if op == "c2c"
            else half_bytes
            if op == "c2r"
            else real_bytes,
            output_bytes=real_bytes * 2
            if op == "c2c"
            else real_bytes
            if op == "c2r"
            else half_bytes,
        )
        if any(fields.get(k) != str(v) for k, v in contract.items()):
            raise RuntimeError("API contract mismatch")
        with path.open() as f:
            timing = list(csv.DictReader(f))
        if [int(t["sample"]) for t in timing] != list(range(r["samples"])):
            raise RuntimeError("Timing sequence mismatch")
        times = [float(t["acl_us"]) for t in timing]
        if abs(statistics.median(times) - r["median_us"]) > 1e-10:
            raise RuntimeError("Stored median mismatch")
        actual[key] = r
        values[key] = times
    if (
        set(actual) != set(expected)
        or progress["state"] != "COMPLETE"
        or progress["completed"] != len(expected)
    ):
        raise RuntimeError("Incomplete coverage")
    for c, _, rounds, versions in identity["cases"]:
        paired = []
        for label in versions:
            for k in range(rounds):
                paired.append(actual[(tuple(c), label, k)])
        if len({r["input_sha256"] for r in paired}) != 1:
            raise RuntimeError("Variants used different inputs")
    lines = [
        "# Ascend950 一维 FFT 合入验证",
        "",
        f"设备验证 {len(records)}/{len(expected)} PASS；最大相对 L2={max(r['relative_l2'] for r in records):.9e}。",
        f"启动时存在其他进程：{sum(r['shared'] for r in records)}/{len(records)} 组。",
        "FP64 golden，阈值1e-3；1D为batch=N，2D为batch=1方阵。均为非原地正式 ACL Event 计时。",
        "每版正反顺序各300次，合并600个原始样本取中位数；不含Plan、分配和H2D/D2H，不启用profiler。",
        "",
        "| 维度 | N | 算子 | 原版 μs | 优化版 μs | 耗时降低 | 两轮均更快 |",
        "|---:|---:|---|---:|---:|---:|---|",
    ]
    cells = []
    for case, samples, rounds, variants in identity["cases"]:
        if samples != 300:
            continue
        c = tuple(case)
        baseline = statistics.median(
            values[(c, "baseline", 0)] + values[(c, "baseline", 1)]
        )
        candidate = statistics.median(
            values[(c, "candidate", 0)] + values[(c, "candidate", 1)]
        )
        both = all(
            actual[(c, "candidate", i)]["median_us"]
            < actual[(c, "baseline", i)]["median_us"]
            for i in range(2)
        )
        gain = (1 - candidate / baseline) * 100
        tag = "是" if both else "否"
        lines.append(
            f"|{case[2]}D|{case[0]}|{case[3]} {case[4]}|{baseline:.3f}|{candidate:.3f}|{gain:+.2f}%|{tag}|"
        )
        cells.append(
            dict(
                dim=case[2],
                n=case[0],
                op=case[3],
                direction=case[4],
                baseline_us=baseline,
                candidate_us=candidate,
                reduction_percent=gain,
                both_faster=both,
            )
        )
    lines += [
        "",
        "全部目标二次幂尺寸均启用优化，包括计时退化的点；不按性能结果回退。",
        "本报告仅覆盖明确列出的范围；不代表任意batch、布局或非二次幂的性能结论。",
        "",
    ]
    (a.root / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    (a.root / "summary.json").write_text(
        json.dumps(
            dict(
                passed=len(records),
                samples=sum(r["samples"] for r in records),
                shared=sum(r["shared"] for r in records),
                max_relative_l2=max(r["relative_l2"] for r in records),
                cells=cells,
            ),
            indent=2,
        )
        + "\n"
    )
    print(
        f"AUDIT PASS: {len(records)} records, {sum(r['samples'] for r in records)} samples"
    )


if __name__ == "__main__":
    main()
