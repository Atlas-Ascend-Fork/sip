# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Execute complete actual SiP comparison CLIs without NPU dependencies."""

import os
import re
import shutil
import string
import tempfile
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = [
    "tests/ut/unittest/base/conj/conj_data/compare_data.py",
    "tests/ut/unittest/blas/ccopy/ccopy_data/compare_data.py",
    "tests/ut/unittest/blas/cdotc/cdotc_data/compare_data.py",
    "tests/ut/unittest/blas/cdotu/cdotu_data/compare_data.py",
    "tests/ut/unittest/blas/scnrm2/scnrm2_data/compare_data.py",
    "tests/ut/unittest/blas/scopy/scopy_data/compare_data.py",
    "tests/ut/unittest/blas/sdot/sdot_data/compare_data.py",
    "tests/ut/unittest/blas/snrm2/snrm2_data/compare_data.py",
    "tests/ut/unittest/blas/ssyr/ssyr_data/compare_data.py",
    "tests/ut/unittest/blas/ssyr2/ssyr2_data/compare_data.py",
]
SCENARIOS = (
    "empty",
    "golden_only",
    "output_only",
    "golden_more",
    "output_more",
    "matching",
    "mismatching",
    "matching_nan",
)


def run_cli(command, environment):
    """Execute the Linux CLI directly, keeping both output streams and exit status."""
    assert Path(command[0]).is_absolute()
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as stdout:
        with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as stderr:
            pid = os.posix_spawn(
                command[0],
                command,
                environment,
                file_actions=[
                    (os.POSIX_SPAWN_DUP2, stdout.fileno(), 1),
                    (os.POSIX_SPAWN_DUP2, stderr.fileno(), 2),
                ],
            )
            _, status = os.waitpid(pid, 0)
            returncode = (
                os.WEXITSTATUS(status) if os.WIFEXITED(status) else -os.WTERMSIG(status)
            )
            stdout.seek(0)
            stderr.seek(0)
            return returncode, stdout.read(), stderr.read()


@pytest.mark.parametrize("relative", SCRIPTS)
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_actual_cli_requires_paired_inputs_and_preserves_numeric_rules(
    relative, scenario, tmp_path, monkeypatch
):
    source = ROOT / relative
    dtype_names = re.findall(r"np_dtype\s*=\s*np\.([\w]+)", source.read_text())
    assert len(dtype_names) == 1 and dtype_names[0] in ("float32", "complex64")
    dtype = getattr(np, dtype_names[0])
    script = tmp_path / "compare_data.py"
    shutil.copyfile(source, script)
    values = [np.nan, 2.0] if scenario == "matching_nan" else [1.0, 2.0]
    # Explicit NumPy file fixtures, not actual NPU outputs or measurements.
    if scenario not in ("empty", "output_only"):
        np.array(values, dtype=dtype).tofile(tmp_path / "0_golden.bin")
    if scenario not in ("empty", "golden_only"):
        output = [3.0, 4.0] if scenario == "mismatching" else values
        np.array(output, dtype=dtype).tofile(tmp_path / "0_output.bin")
    if scenario == "golden_more":
        np.array([5.0, 6.0], dtype=dtype).tofile(tmp_path / "1_golden.bin")
    if scenario == "output_more":
        np.array([5.0, 6.0], dtype=dtype).tofile(tmp_path / "1_output.bin")
    with monkeypatch.context() as context:
        context.chdir(tmp_path)
        returncode, stdout, stderr = run_cli(
            [sys.executable, str(script)],
            dict(os.environ, TORCH_DEVICE_BACKEND_AUTOLOAD="0"),
        )
    expected = 0 if scenario in ("matching", "matching_nan") else 1
    assert returncode == expected, stdout + stderr
    assert not stderr, stderr
    lines = stdout.splitlines()
    # Human-readable status punctuation is not a machine protocol. Keep the
    # status word, full diagnostic and real process status as separate checks.
    fields = lines[0].split(maxsplit=1)
    status = fields[0].rstrip(string.punctuation)
    message = f" {fields[1]}" if len(fields) > 1 else ""
    assert status == ("FAILED" if expected else "PASSED"), stdout
    if scenario in (
        "empty",
        "golden_only",
        "output_only",
        "golden_more",
        "output_more",
    ):
        diagnostic = (
            "empty file list"
            if scenario in ("empty", "golden_only", "output_only")
            else "file list length mismatch"
        )
        golden_count = len(list(tmp_path.glob("*golden*.bin")))
        output_count = len(list(tmp_path.glob("*output*.bin")))
        assert len(lines) == 1, stdout
        assert message == (
            f" {diagnostic} (golden={golden_count}, output={output_count})"
        ), stdout
    elif scenario == "mismatching":
        assert len(lines) == 3 and not message, stdout
        for index, (actual, golden) in enumerate(zip([3.0, 4.0], values)):
            assert lines[index + 1] == (
                f"index: {index}, output: {dtype(actual)}, golden: {dtype(golden)}"
            ), stdout
    else:
        assert len(lines) == 1 and not message, stdout
        golden = np.fromfile(tmp_path / "0_golden.bin", dtype=dtype)
        output = np.fromfile(tmp_path / "0_output.bin", dtype=dtype)
        np.testing.assert_array_equal(output, golden)
