# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Run complete existing comparison CLIs with aligned Host binary fixtures."""

import os
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


def fixture_arrays(scenario, dtype):
    if scenario == "empty_pair":
        return np.array([], dtype=dtype), np.array([], dtype=dtype), 1
    if scenario == "short_output":
        return np.array([3, 3, 3], dtype=dtype), np.array([3], dtype=dtype), 1
    if scenario == "short_golden":
        return np.array([3], dtype=dtype), np.array([3, 3, 3], dtype=dtype), 1
    if scenario == "difference":
        return np.array([1, 2, 3], dtype=dtype), np.array([1, 2, 9], dtype=dtype), 1
    if scenario == "equal_scalar":
        return np.array([2], dtype=dtype), np.array([2], dtype=dtype), 0
    if scenario == "equal_nan":
        return (
            np.array([np.nan, np.nan], dtype=dtype),
            np.array([np.nan, np.nan], dtype=dtype),
            0,
        )
    return np.array([1, 2, 3], dtype=dtype), np.array([1, 2, 3], dtype=dtype), 0


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
@pytest.mark.parametrize(
    "scenario",
    [
        "empty_pair",
        "short_output",
        "short_golden",
        "difference",
        "equal_scalar",
        "equal_nan",
        "equal_vector",
    ],
)
def test_complete_comparison_cli_rejects_element_broadcast(
    tmp_path, relative, scenario
):
    source = ROOT / relative
    dtype = (
        np.complex64 if "np_dtype = np.complex64" in source.read_text() else np.float32
    )
    script = tmp_path / "compare_data.py"
    shutil.copyfile(source, script)
    golden, output, expected = fixture_arrays(scenario, dtype)
    golden.tofile(tmp_path / "fixture_golden.bin")
    output.tofile(tmp_path / "fixture_output.bin")
    assert Path(sys.executable).is_absolute()
    returncode, stdout, stderr = run_cli(
        [sys.executable, "-I", str(script)], dict(os.environ)
    )
    assert returncode == expected, stdout + stderr
    assert not stderr, stderr
    lines = stdout.splitlines()
    # Human-readable status punctuation is not a machine protocol. Keep the
    # status word, full diagnostic and real process status as separate checks.
    fields = lines[0].split(maxsplit=1)
    status = fields[0].rstrip(string.punctuation)
    diagnostic = f" {fields[1]}" if len(fields) > 1 else ""
    assert status == ("FAILED" if expected else "PASSED"), stdout
    if scenario in ("empty_pair", "short_output", "short_golden"):
        assert len(lines) == 1, stdout
        assert diagnostic == (
            f" invalid element count (fixture_output.bin: {output.size}, "
            f"fixture_golden.bin: {golden.size})"
        ), stdout
    elif scenario == "difference":
        assert len(lines) == 2 and not diagnostic, stdout
        expected_line = f"index: 2, output: {output[2]}, golden: {golden[2]}"
        assert lines[1] == expected_line, stdout
    else:
        assert len(lines) == 1 and not diagnostic, stdout
        np.testing.assert_array_equal(output, golden)
