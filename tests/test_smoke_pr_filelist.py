#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Exercise PR smoke selection without compiling examples or accessing an NPU."""

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_test.sh"
ALL_CATEGORIES = "base blas fft domain filter interpolation"


class SmokeFileListTests(unittest.TestCase):
    def setUp(self):
        self.bash = shutil.which("bash")
        if self.bash is None and os.name == "nt":
            candidate = (
                Path(os.environ.get("ProgramFiles", "C:/Program Files"))
                / "Git/bin/bash.exe"
            )
            if candidate.is_file():
                self.bash = str(candidate)
        if self.bash is None:
            self.skipTest("bash is required for shell regression tests")
        text = SCRIPT.read_text(encoding="utf-8")
        start = text.index("function get_valid_pr_dir()")
        end = text.index("function fn_example_prepare()", start)
        self.selector = text[start:end]

    def select(self, content):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            if content is not None:
                (root / "pr_filelist.txt").write_bytes(content)
            harness = root / "select.sh"
            harness.write_text(
                "#!/bin/bash\nset -euo pipefail\n"
                'cd -- "$(dirname -- "$0")"\nCODE_ROOT=$PWD\n'
                "log_info() { :; }\nlog_warn() { :; }\nVALID_TEST_DIRS=()\n"
                + self.selector
                + '\nget_valid_pr_dir\nprintf "SELECTED=%s\\n" "${VALID_TEST_DIRS[*]}"\n',
                encoding="utf-8",
                newline="\n",
            )
            result = subprocess.run(
                [self.bash, harness.as_posix()],
                check=True,
                capture_output=True,
                text=True,
                timeout=15,
            )
            return result.stdout.rsplit("SELECTED=", 1)[1].strip()

    def test_missing_file_uses_all_categories(self):
        self.assertEqual(self.select(None), ALL_CATEGORIES)

    def test_empty_and_whitespace_lists_use_all_categories(self):
        for content in (b"", b"\n\r\n", b" \t\r\n"):
            with self.subTest(content=content):
                self.assertEqual(self.select(content), ALL_CATEGORIES)

    def test_committed_operator_change_does_not_need_worktree_diff(self):
        self.assertEqual(self.select(b"ops/blas/dot.cpp\n"), "blas")

    def test_name_status_and_crlf_entries_are_supported(self):
        self.assertEqual(self.select(b"M\tops/fft/c2c.cpp\r\n"), "fft")

    def test_documentation_only_skips_examples(self):
        self.assertEqual(self.select(b"docs/guide.md\nREADME.md\n"), "")

    def test_unknown_and_ambiguous_paths_use_all_categories(self):
        for content in (b"CMakeLists.txt\n", b"ops/blas/file name.cpp\n"):
            with self.subTest(content=content):
                self.assertEqual(self.select(content), ALL_CATEGORIES)

    def test_categories_are_deduplicated(self):
        self.assertEqual(
            self.select(b"ops/fft/one.cpp\nops/blas/two.cpp\nops/fft/three.cpp\n"),
            "blas fft",
        )


if __name__ == "__main__":
    unittest.main()
