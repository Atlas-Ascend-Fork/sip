# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Compare two installed SIP libraries through the public tensor FFT API.

Run on an explicitly selected Ascend950 device after sourcing CANN. Raw device
snapshots stay in the output directory; do not add them to a public PR.
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import numpy as np


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def snapshot(path, device):
    parts = []
    for kind in ("health", "proc-mem"):
        p = subprocess.run(
            ["npu-smi", "info", "-t", kind, "-i", str(device)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        parts.append(p.stdout + p.stderr)
        path.write_text("\n=== PROC-MEM ===\n".join(parts))
        if p.returncode or re.findall(r"NPU ID\s*:\s*(\d+)", parts[-1]) != [
            str(device)
        ]:
            raise RuntimeError("Device snapshot failed")
    if re.findall(r"Health Status\s*:\s*(\S+)", parts[0]) != ["OK"]:
        raise RuntimeError("Device is unhealthy")
    pids = re.findall(r"Process id:\s*(\d+)", parts[1])
    if not pids and "No process in device." not in parts[1]:
        raise RuntimeError("Cannot determine device occupancy")
    return bool(pids)


def generate(case, root):
    n, b, dim, op, direction, pattern, alias = case
    shape = (b,) + (n,) * dim
    axes = tuple(range(-dim, 0))
    rng = np.random.default_rng(20261029 + n)
    raw = rng.standard_normal(shape, dtype=np.float32)
    if pattern == "zero":
        raw.fill(0)
    elif pattern == "constant":
        raw.fill(1)
    elif pattern == "impulse":
        raw.fill(0)
        raw.reshape(-1, n)[:, 1] = 1
    elif pattern == "nyquist":
        raw[...] = np.where(np.arange(n) % 2, -1, 1)
    if op == "c2c":
        x = (
            raw
            + 1j
            * (
                rng.standard_normal(shape, dtype=np.float32)
                if pattern == "random"
                else raw
            )
        ).astype(np.complex64)
        g = (
            np.fft.fftn(x.astype(np.complex128), axes=axes)
            if direction == "forward"
            else np.fft.ifftn(x.astype(np.complex128), axes=axes) * n**dim
        )
    elif op == "r2c":
        x = raw
        g = np.fft.rfftn(x.astype(np.float64), axes=axes)
        if direction == "inverse":
            g = g.conj()
    else:
        x = np.fft.rfftn(raw.astype(np.float64), axes=axes).astype(np.complex64)
        z = x.astype(np.complex128)
        g = (
            np.fft.irfftn(
                z if direction == "inverse" else z.conj(), s=(n,) * dim, axes=axes
            )
            * n**dim
        )
    x.tofile(root / "input.bin")
    return g, sha(root / "input.bin")


def optimized(case):
    n, b, dim, op, direction, _, _ = case
    return (
        ((dim == 1 and b == n) or (dim == 2 and b == 1))
        and n in [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
        and (op == "c2c" or direction == ("inverse" if op == "c2r" else "forward"))
    )


def check_route(case, label, log):
    n, _, dim, op, direction, _, _ = case
    expected = label == "candidate" and optimized(case)
    trace = "A5_C2C_Optimized" if op == "c2c" else "paired_boundary=1"
    if (trace in log) != expected:
        raise RuntimeError("Unexpected optimized route")
    real_stages = re.findall(r"A5_REAL_FFT_STAGE stage=(\S+) key=(\d+)", log)
    optimized_kernel = "FftRealRegBaseOptimizedKernel"
    optimized_stages = [key for name, key in real_stages if name == optimized_kernel]
    if op == "c2c" or not expected:
        if optimized_stages:
            raise RuntimeError("Unexpected optimized real kernel")
    else:
        configs = {
            32: 1,
            64: 1,
            128: 2,
            256: 6,
            512: 6,
            1024: 2,
            2048: 4,
            4096: 3,
            8192: 2,
        }
        variants = re.findall(r"A5_REAL_FFT_VARIANT ([^\r\n]+)", log)
        if len(variants) != 1:
            raise RuntimeError("Missing or duplicate real axis trace")
        fields = dict(re.findall(r"(\w+)=(\S+)", variants[0]))
        required = dict(
            kind=op,
            N=str(n),
            effective_batch=str(n),
            inverse=str(int(direction == "inverse")),
            launches="1",
            config=str(configs[n]),
            paired_boundary="1",
        )
        if any(fields.get(k) != v for k, v in required.items()) or fields.get(
            "slots"
        ) not in ("1", "2"):
            raise RuntimeError("Incorrect optimized real configuration")
        key = (
            1000000
            + 128 * n
            + 8 * configs[n]
            + 4 * (op == "c2r")
            + (fields["slots"] == "2")
        )
        if real_stages != [(optimized_kernel, str(key))]:
            raise RuntimeError("Incorrect optimized real kernel/key")
    if expected:
        complex_rows = [
            int(v)
            for v in re.findall(r"A5_C2C_Optimized N=\d+ effective_batch=(\d+)", log)
        ]
        expected_rows = [n] * dim if op == "c2c" else [n // 2 + 1] if dim == 2 else []
        if sorted(complex_rows) != sorted(expected_rows):
            raise RuntimeError(f"Incorrect complex axis route: {complex_rows}")


def suite():
    ops = [
        ("c2c", "forward"),
        ("c2c", "inverse"),
        ("r2c", "forward"),
        ("c2r", "inverse"),
    ]
    lengths = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    # Formal 1D rows and complete 2D execution, both execution orders.
    for dim in [1, 2]:
        for n in lengths:
            for op, direction in ops:
                yield (
                    (n, n if dim == 1 else 1, dim, op, direction, "random", False),
                    300,
                    2,
                    ["baseline", "candidate"],
                )
    # The restored real configurations and composed 2D boundary cases.
    for dim, n in [(1, 128), (1, 256), (2, 32), (2, 512)]:
        for op, direction in ops:
            for pattern in ["zero", "impulse", "constant", "nyquist"]:
                yield (
                    (n, n if dim == 1 else 1, dim, op, direction, pattern, False),
                    2,
                    1,
                    ["candidate"],
                )
    for dim in [1, 2]:
        for n in lengths:
            for op, direction in ops:
                yield (
                    (n, n if dim == 1 else 1, dim, op, direction, "random", True),
                    2,
                    1,
                    ["baseline", "candidate"],
                )
    # Non-target batches, mixed-radix and 3D retain original kernels.
    for n, b, dim in [
        (32, 1, 1),
        (512, 511, 1),
        (1024, 1025, 1),
        (96, 96, 1),
        (32, 2, 2),
        (32, 1, 3),
    ]:
        for op, direction in ops:
            yield (
                (n, b, dim, op, direction, "random", False),
                2,
                1,
                ["baseline", "candidate"],
            )
    for dim in [1, 2]:
        for n in [32, 1024]:
            for op, direction in [("r2c", "inverse"), ("c2r", "forward")]:
                yield (
                    (n, n if dim == 1 else 1, dim, op, direction, "random", False),
                    2,
                    1,
                    ["baseline", "candidate"],
                )


def main():
    p = argparse.ArgumentParser()
    for name in ["baseline", "candidate", "binary", "output"]:
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--device", type=int, required=True)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    if a.device < 0 or any(
        os.environ.get(k)
        for k in [
            "ASCEND_RT_VISIBLE_DEVICES",
            "ASCEND_VISIBLE_DEVICES",
            "NPU_VISIBLE_DEVICES",
            "LD_PRELOAD",
            "LD_AUDIT",
        ]
    ):
        raise RuntimeError("Use a verified physical device without remapping/injection")
    libraries = {}
    envs = {}
    for label in ["baseline", "candidate"]:
        directory = getattr(a, label).resolve() / "lib"
        libraries[label] = {}
        for library in directory.iterdir():
            if library.name.startswith("libasdsip") and library.name.endswith(".so"):
                libraries[label][str(library)] = sha(library)
        if len(libraries[label]) != 3:
            raise RuntimeError("Expected three installed SIP libraries")
        env = dict(os.environ)
        for key in list(env):
            if key.startswith(("ASDSIP_FFT", "SIP_FFT")) or key == "PROFILING_MODE":
                env.pop(key)
        env.update(
            LD_LIBRARY_PATH=str(directory) + ":" + env.get("LD_LIBRARY_PATH", "")
        )
        loader = subprocess.run(
            ["ldd", str(a.binary)], env=env, capture_output=True, text=True, check=True
        ).stdout
        (a.output / (label + "_loader.txt")).write_text(loader)
        loaded = dict(re.findall(r"^\s*(libasdsip\S*)\s+=>\s+(\S+)\s+\(", loader, re.M))
        if (
            "not found" in loader
            or not {"libasdsip.so", "libasdsip_core.so"} <= loaded.keys()
            or any(
                str(Path(path).resolve()) not in libraries[label]
                or sha(Path(path)) != libraries[label][str(Path(path).resolve())]
                for path in loaded.values()
            )
        ):
            raise RuntimeError("Incorrect dynamic library resolution")
        envs[label] = env
    selected = list(suite())
    if a.smoke:
        selected = [
            (c, 2, 1, ["baseline", "candidate"])
            for c, samples, _, _ in selected
            if samples == 300
        ]
    save(
        a.output / "identity.json",
        dict(
            libraries=libraries,
            binary=sha(a.binary),
            runner=sha(Path(__file__)),
            device=a.device,
            cases=selected,
        ),
    )
    records = []
    expected = sum(rounds * len(versions) for _, _, rounds, versions in selected)
    try:
        for index, (case, samples, rounds, versions) in enumerate(selected):
            n, b, dim, op, direction, pattern, alias = case
            root = (
                a.output
                / f"{index:03}_{op}_{direction}_n{n}_b{b}_d{dim}_{pattern}_alias{int(alias)}"
            )
            root.mkdir()
            golden, input_hash = generate(case, root)
            comparison_hash = None
            for round_id in range(rounds):
                for label in versions if round_id % 2 == 0 else versions[::-1]:
                    attempt = root / f"{round_id}_{label}"
                    attempt.mkdir()
                    shared = snapshot(attempt / "device.txt", a.device)
                    command = [
                        str(a.binary),
                        str(dim),
                        op,
                        str(n),
                        str(b),
                        str(root / "input.bin"),
                        str(attempt / "output.bin"),
                        str(attempt / "timings.csv"),
                        str(samples),
                        "10",
                        direction,
                        str(n),
                        str(a.device),
                        str(int(alias)),
                    ]
                    run = subprocess.run(
                        command,
                        env=envs[label],
                        capture_output=True,
                        text=True,
                        timeout=300,
                    )
                    log = run.stdout + run.stderr
                    (attempt / "run.log").write_text(log)
                    if run.returncode or "input_unchanged=1" not in log:
                        raise RuntimeError(f"Execution failed: {attempt}")
                    check_route(case, label, log)
                    output = np.fromfile(
                        attempt / "output.bin",
                        dtype=np.float32 if op == "c2r" else np.complex64,
                    )
                    if output.size != golden.size or not np.isfinite(output).all():
                        raise RuntimeError("Output length/finiteness failed")
                    output = output.reshape(golden.shape)
                    delta = np.linalg.norm((output - golden).ravel())
                    norm = np.linalg.norm(golden.ravel())
                    error = float(delta / norm) if norm else float(delta)
                    if not np.isfinite(error) or error > 1e-3:
                        raise RuntimeError(f"FP64 relative L2 failed: {error}")
                    digest = sha(attempt / "output.bin")
                    if not optimized(case):
                        if comparison_hash is not None and digest != comparison_hash:
                            raise RuntimeError("Fallback output changed")
                        comparison_hash = digest
                    with (attempt / "timings.csv").open() as f:
                        rows = list(csv.DictReader(f))
                    values = [float(row["acl_us"]) for row in rows]
                    if (
                        len(values) != samples
                        or not np.isfinite(values).all()
                        or min(values) <= 0
                    ):
                        raise RuntimeError("Invalid timing samples")
                    record = dict(
                        case=list(case),
                        variant=label,
                        round=round_id,
                        samples=samples,
                        median_us=float(np.median(values)),
                        relative_l2=error,
                        status="PASS",
                        shared=shared,
                        input_sha256=input_hash,
                        output_sha256=digest,
                        timings=str((attempt / "timings.csv").relative_to(a.output)),
                        log_sha256=sha(attempt / "run.log"),
                    )
                    save(attempt / "result.json", record)
                    records.append(record)
                    save(
                        a.output / "progress.json",
                        dict(
                            completed=len(records),
                            expected=expected,
                            state="RUNNING",
                            latest=record,
                        ),
                    )
                    (attempt / "output.bin").unlink()
            if sha(root / "input.bin") != input_hash:
                raise RuntimeError("Input file changed")
            (root / "input.bin").unlink()
            del golden
        save(a.output / "records.json", records)
        save(
            a.output / "progress.json",
            dict(
                completed=len(records),
                expected=expected,
                state="COMPLETE",
                max_relative_l2=max(r["relative_l2"] for r in records),
            ),
        )
    except Exception as exc:
        save(a.output / "records.json", records)
        save(
            a.output / "progress.json",
            dict(
                completed=len(records),
                expected=expected,
                state="FAILED",
                error=str(exc),
            ),
        )
        raise


if __name__ == "__main__":
    main()
