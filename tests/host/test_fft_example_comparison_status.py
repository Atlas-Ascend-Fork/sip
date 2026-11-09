# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Run complete real FFT CSV/binary validation CLI; all arrays are Host fixtures."""

import importlib.util
import os
from pathlib import Path
import shutil
import tempfile
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = [
    Path("example/Ascend950/FFT/fft-2d-r2c-4096x4096/gen_data.py"),
    Path("example/Ascend950/FFT/fft-2d-c2r-4096x4096/gen_data.py"),
]
FFT_N = 2048


def load_script(path):
    spec = importlib.util.spec_from_file_location("fft_host_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
    "scenario", ["numerical_fail", "short_file", "correct_output", "generate_only"]
)
def test_existing_fft_shell_validation_can_use_process_status(
    tmp_path, relative, scenario
):
    script = tmp_path / "gen_data.py"
    shutil.copyfile(ROOT / relative, script)
    module = load_script(script)
    directory = tmp_path / "data"
    directory.mkdir()
    output = directory / f"output_{FFT_N}_1.bin"
    if scenario != "generate_only":
        if "r2c" in str(relative):
            _, golden = module.generate_golden_numpy(FFT_N, 1)
            values = np.ascontiguousarray(golden).view(np.float32).ravel()
        else:
            _, golden, _ = module.generate_golden_numpy(FFT_N, 1)
            values = golden.ravel()
        if scenario == "numerical_fail":
            values = np.zeros_like(values)
        elif scenario == "short_file":
            values = values[:2]
        values.tofile(output)
    environment = dict(os.environ, TORCH_DEVICE_BACKEND_AUTOLOAD="0")
    command = [sys.executable, "-I", str(script), str(FFT_N), "1"]
    returncode, stdout, stderr = run_cli(command, environment)
    assert returncode == (1 if scenario in ("numerical_fail", "short_file") else 0), (
        stdout + stderr
    )
    assert not stderr, stderr
    if scenario == "numerical_fail":
        assert "Result: FAIL" in stdout
    elif scenario == "short_file":
        assert "Error: expected" in stdout
    elif scenario == "correct_output":
        assert "Result: PASS" in stdout
    else:
        assert "NPU output not found" in stdout
    assert (directory / f"input_{FFT_N}_1.bin").exists()
    assert (directory / f"golden_{FFT_N}_1.bin").exists()
