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


def optimized(case, multirow=True):
    # Candidate dispatch enables multirow by default; --multirow only selects test cases.
    n, b, dim, op, direction, _, _ = case
    return (
        (
            (
                dim == 1
                and (
                    b == n
                    or (
                        multirow
                        and 1 <= b <= 8192
                        and n <= (4096 if op == "c2c" else 8192)
                    )
                )
            )
            or (dim == 2 and b == 1)
        )
        and n in [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
        and (op == "c2c" or direction == ("inverse" if op == "c2r" else "forward"))
    )


def common_mixed(case):
    n, b, dim, op, direction, _, _ = case
    return (
        32 <= n <= 8192
        and n % 32 == 0
        and n & (n - 1)
        and ((dim == 1 and 1 <= b <= 8192) or (dim == 2 and b == 1))
        and (
            op == "c2c"
            or (n > 1024 and direction == ("inverse" if op == "c2r" else "forward"))
        )
    )


def check_route(case, label, log, parent_optimized=False):
    n, b, dim, op, direction, _, _ = case
    if common_mixed(case):
        traces = re.findall(r"^ASDSIP_MIXED_FFT (.+)$", log, re.M)
        if label == "candidate" or parent_optimized:
            expected = (
                [(b, {"c2c": 0, "r2c": 1, "c2r": 2}[op])]
                if dim == 1
                else (
                    [(n, 0), (n, 0)]
                    if op == "c2c"
                    else [(n, 1 if op == "r2c" else 2), (n // 2 + 1, 0)]
                )
            )
            fields = [dict(re.findall(r"(\w+)=(\S+)", trace)) for trace in traces]
            actual = [(int(f.get("batch", -1)), int(f.get("pass", -1))) for f in fields]
            if sorted(actual) != sorted(expected) or any(
                f.get("N") != str(n)
                or f.get("inverse") != str(int(direction == "inverse"))
                for f in fields
            ):
                raise RuntimeError("Incorrect mixed FFT route")
        return
    expected = optimized(case, label == "candidate") and (
        label == "candidate" or parent_optimized
    )
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
            effective_batch=str(b if dim == 1 else n),
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
        if label == "candidate" and dim == 1 and fields["slots"] == "2":
            key += 20000000 if n >= 1024 else 10000000
        if real_stages != [(optimized_kernel, str(key))]:
            raise RuntimeError("Incorrect optimized real kernel/key")
    if expected:
        complex_rows = [
            int(v)
            for v in re.findall(r"A5_C2C_Optimized N=\d+ effective_batch=(\d+)", log)
        ]
        expected_rows = (
            ([b] if dim == 1 else [n] * dim)
            if op == "c2c"
            else [n // 2 + 1]
            if dim == 2
            else []
        )
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
    # Off-diagonal 1D batches now use candidate multirow; mixed uses its common core.
    # Non-target 2D batches and 3D still exercise fallback routes.
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


def multirow_suite():
    formal = [
        (64, 8192, "c2c", "forward"),
        (256, 8192, "c2c", "forward"),
        (512, 8192, "c2c", "inverse"),
        (4096, 8192, "c2c", "forward"),
        (256, 8192, "r2c", "forward"),
        (1024, 8192, "c2r", "inverse"),
        (4096, 8192, "r2c", "forward"),
        (8192, 8192, "c2r", "inverse"),
        (544, 8192, "c2c", "forward"),
        (2016, 8192, "c2c", "inverse"),
        (2944, 8192, "c2c", "inverse"),
        (2976, 8192, "c2c", "inverse"),
        (3808, 8192, "c2c", "inverse"),
        (3840, 8192, "c2c", "forward"),
        (5312, 8192, "r2c", "forward"),
        (5344, 8192, "c2r", "inverse"),
        (6688, 8192, "r2c", "forward"),
        (6720, 8192, "c2r", "inverse"),
        (6720, 8192, "c2c", "forward"),
        (6752, 8192, "c2c", "inverse"),
        (8160, 8192, "c2c", "inverse"),
    ]
    functional = [
        (64, 7, "c2c", "inverse"),
        (512, 65, "c2c", "inverse"),
        (1024, 129, "r2c", "forward"),
        (4096, 193, "c2r", "inverse"),
        (2944, 129, "c2c", "inverse"),
        (3808, 193, "c2c", "inverse"),
        (5344, 65, "r2c", "forward"),
        (6688, 193, "c2r", "inverse"),
        (3840, 193, "c2c", "inverse"),
        (6720, 65, "c2c", "inverse"),
        (6752, 193, "c2c", "inverse"),
        (8160, 7, "c2c", "forward"),
        (512, 8192, "r2c", "forward"),
    ]
    for samples, rounds, cases in ((300, 2, formal), (3, 1, functional)):
        for n, b, op, direction in cases:
            yield (
                (n, b, 1, op, direction, "random", False),
                samples,
                rounds,
                ["baseline", "candidate"],
            )


def mixed_2d_suite():
    formal = [
        (544, "c2c", "forward"),
        (2016, "c2c", "inverse"),
        (1536, "r2c", "forward"),
        (1536, "c2r", "inverse"),
    ]
    functional = [
        (3840, "c2c", "forward"),
        (6752, "c2c", "inverse"),
        (5344, "r2c", "forward"),
    ]
    for samples, rounds, cases in ((300, 2, formal), (3, 1, functional)):
        for n, op, direction in cases:
            yield (
                (n, 1, 2, op, direction, "random", False),
                samples,
                rounds,
                ["baseline", "candidate"],
            )


def main():
    p = argparse.ArgumentParser()
    for name in ["baseline", "candidate", "binary", "output"]:
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--device", type=int, required=True)
    p.add_argument("--smoke", action="store_true")
    selection = p.add_mutually_exclusive_group()
    selection.add_argument(
        "--mixed-2d",
        action="store_true",
        help="Run seven square-2D common mixed replacements against the previous mixed scheduler",
    )
    selection.add_argument(
        "--multirow",
        action="store_true",
        help="Run the bounded multirow integration suite against its optimized parent",
    )
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
    selected = list(
        mixed_2d_suite() if a.mixed_2d else multirow_suite() if a.multirow else suite()
    )
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
                    check_route(case, label, log, a.multirow or a.mixed_2d)
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
                    if not optimized(case) and not common_mixed(case):
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
