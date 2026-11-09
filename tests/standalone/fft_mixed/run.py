# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Validate the mixed FFT default through the existing standalone FFT benchmark."""

import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path
import re
import statistics
from subprocess import STDOUT, run
import traceback

import numpy as np

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "fft_validation", HERE.parent / "fft1d_optimized/run.py"
)
validation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validation)


def environment(install):
    env = dict(os.environ)
    for key in list(env):
        if key.startswith(("SIP_FFT", "ASDSIP_FFT", "FFT_TEST")):
            env.pop(key)
    env["LD_LIBRARY_PATH"] = str(install / "lib") + ":" + env.get("LD_LIBRARY_PATH", "")
    env.update(SIP_FFT_BACKEND="internal", ASDSIP_FFT_PLAN_TRACE="1")
    return env


def library_hashes(install):
    """Hash installed SIP shared libraries, including resolved symbolic links."""
    return {
        path.name: validation.sha(path)
        for path in (install / "lib").iterdir()
        if path.name.startswith("libasdsip") and path.suffix == ".so" and path.is_file()
    }


def check_route(log, case):
    traces = re.findall(r"^ASDSIP_MIXED_FFT (.+)$", log, re.M)
    if len(traces) != 1:
        raise RuntimeError("Missing or duplicate common mixed core")
    fields = dict(re.findall(r"(\w+)=(\S+)", traces[0]))
    expected = dict(
        N=str(case["n"]),
        batch=str(case["batch"]),
        inverse=str(int(case["direction"] == "inverse")),
        **{"pass": str({"c2c": 0, "r2c": 1, "c2r": 2}[case["op"]])},
    )
    if any(fields.get(key) != value for key, value in expected.items()):
        raise RuntimeError("Unexpected mixed transform configuration")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("candidate", "binary", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument(
        "--smoke", action="store_true", help="One round, three Event samples"
    )
    parser.add_argument(
        "--alias", action="store_true", help="Validate exact in-place execution"
    )
    args = parser.parse_args()
    if args.device < 0 or any(
        os.environ.get(key)
        for key in (
            "ASCEND_VISIBLE_DEVICES",
            "ASCEND_RT_VISIBLE_DEVICES",
            "NPU_VISIBLE_DEVICES",
            "LD_PRELOAD",
            "LD_AUDIT",
        )
    ):
        parser.error(
            "Use an explicit nonnegative device ID without loader/device remapping"
        )
    args.binary = args.binary.resolve(strict=True)
    args.output = args.output.resolve()
    installs = {"candidate": args.candidate.resolve(strict=True)}
    if args.baseline:
        installs = {"baseline": args.baseline.resolve(strict=True), **installs}
    envs = {name: environment(path) for name, path in installs.items()}
    args.output.mkdir(parents=True, exist_ok=False)
    pins = {}
    for name, path in installs.items():
        pins[name] = library_hashes(path)
        result = run(
            ["ldd", str(args.binary)],
            env=envs[name],
            text=True,
            capture_output=True,
            check=True,
        )
        (args.output / (name + "_loader.txt")).write_text(result.stdout + result.stderr)
        if "not found" in result.stdout or any(
            str(path / "lib" / lib) not in result.stdout
            for lib in ("libasdsip.so", "libasdsip_core.so")
        ):
            raise RuntimeError("Unexpected installed library resolution")
    cases = json.loads((HERE / "cases.json").read_text())
    samples, rounds = (3, 1) if args.smoke else (300, 2)
    validation.save(
        args.output / "identity.json",
        dict(
            libraries=pins,
            cases=cases,
            device=args.device,
            samples=samples,
            rounds=rounds,
            alias=args.alias,
        ),
    )
    records = []
    try:
        for case in cases:
            folder = args.output / case["id"]
            folder.mkdir()
            spec = (
                case["n"],
                case["batch"],
                1,
                case["op"],
                case["direction"],
                "random",
                args.alias,
            )
            golden, input_hash = validation.generate(spec, folder)
            for round_id in range(rounds):
                labels = list(installs)
                for name in labels if round_id % 2 == 0 else labels[::-1]:
                    job = folder / f"r{round_id}_{name}"
                    job.mkdir()
                    shared = validation.snapshot(job / "device.txt", args.device)
                    command = [
                        str(args.binary),
                        "1",
                        case["op"],
                        str(case["n"]),
                        str(case["batch"]),
                        str(folder / "input.bin"),
                        str(job / "output.bin"),
                        str(job / "timings.csv"),
                        str(samples),
                        "10",
                        case["direction"],
                        str(case["n"]),
                        str(args.device),
                        str(int(args.alias)),
                    ]
                    with (job / "run.log").open("w") as log:
                        completed = run(
                            command,
                            env=envs[name],
                            stdout=log,
                            stderr=STDOUT,
                            timeout=300,
                        )
                    log = (job / "run.log").read_text()
                    if completed.returncode or "input_unchanged=1" not in log:
                        raise RuntimeError(f"Execution failed: {job}")
                    if name == "candidate":
                        check_route(log, case)
                    output = np.fromfile(
                        job / "output.bin",
                        dtype=np.float32 if case["op"] == "c2r" else np.complex64,
                    )
                    if (
                        output.size != golden.size
                        or not np.isfinite(output).all()
                        or not np.isfinite(golden).all()
                    ):
                        raise RuntimeError("Invalid output/golden length or finiteness")
                    difference = np.linalg.norm(
                        (output.reshape(golden.shape) - golden).ravel()
                    )
                    norm = np.linalg.norm(golden.ravel())
                    error = float(difference / norm if norm else difference)
                    if not np.isfinite(error) or error > 1e-3:
                        raise RuntimeError(f"FP64 relative L2 failed: {error}")
                    with (job / "timings.csv").open() as stream:
                        rows = list(csv.DictReader(stream))
                    values = [float(row["acl_us"]) for row in rows]
                    if (
                        len(values) != samples
                        or not np.isfinite(values).all()
                        or min(values) <= 0
                    ):
                        raise RuntimeError("Invalid Event samples")
                    record = dict(
                        case=case,
                        variant=name,
                        round=round_id,
                        median_us=statistics.median(values),
                        relative_l2=error,
                        shared=shared,
                        input_sha256=input_hash,
                        output_sha256=validation.sha(job / "output.bin"),
                        timings_sha256=validation.sha(job / "timings.csv"),
                    )
                    validation.save(job / "result.json", record)
                    records.append(record)
                    print(case["id"], round_id, name, "PASS", flush=True)
            if validation.sha(folder / "input.bin") != input_hash:
                raise RuntimeError("Input file changed")
        for name, path in installs.items():
            if pins[name] != library_hashes(path):
                raise RuntimeError("Installed libraries changed")
        validation.save(args.output / "results.json", records)
        validation.save(
            args.output / "status.json",
            dict(
                state="COMPLETE",
                requests=len(records),
                max_relative_l2=max(r["relative_l2"] for r in records),
            ),
        )
    except BaseException:
        (args.output / "error.txt").write_text(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
