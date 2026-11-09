# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


import importlib.util
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[4]
SPEC = importlib.util.spec_from_file_location(
    "sip_native_build_util", ROOT / "scripts/build_util.py"
)
BUILD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILD)


class BuildMetadataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="sip-metadata-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.reused = self.root / "cached package"
        self.fresh = self.root / "fresh package"
        self.reused.mkdir()
        self.fresh.mkdir()

    @staticmethod
    def metadata(name="current", detail="native build options"):
        return {
            name: (
                f"operation/{name}.o",
                name,
                f"{name}_kernel",
                json.dumps({"options": detail}),
                32,
                "AiCore",
                "1129660997",
            )
        }

    def write(self, metadata, output):
        BUILD.write_meta(metadata, str(output), "ascend910b")
        return (output / "meta.ini").read_bytes()

    @staticmethod
    def golden_metadata(name):
        fixtures = Path(__file__).with_name("build_metadata_fixtures")
        return (fixtures / name).read_bytes()

    def assert_rewrite_matches_fresh(self, previous, current):
        self.assertGreater(len(self.write(previous, self.reused)), 0)
        expected = self.write(current, self.fresh)
        actual = self.write(current, self.reused)
        self.assertEqual(actual, expected)
        self.assertEqual(
            actual.splitlines(keepends=True)[-1],
            self.golden_metadata("empty.ini").splitlines(keepends=True)[-1],
        )

    def test_removing_previous_kernels_removes_all_old_metadata(self):
        previous = {}
        for index in range(3):
            previous.update(
                self.metadata(f"previous_tactic_{index}", "old option " * 20)
            )
        self.assert_rewrite_matches_fresh(previous, self.metadata())

    def test_shorter_compile_info_does_not_leave_prior_options(self):
        previous = self.metadata(detail="obsolete compilation option " * 40)
        self.assert_rewrite_matches_fresh(previous, self.metadata(detail="current"))

    def test_empty_rebuild_removes_prior_kernel_records(self):
        self.assert_rewrite_matches_fresh(self.metadata(), {})

    def test_fresh_metadata_has_current_count_and_record(self):
        actual = self.write(self.metadata(), self.fresh)
        self.assertEqual(actual, self.golden_metadata("current.ini"))

    def test_fresh_empty_metadata_is_supported(self):
        actual = self.write({}, self.fresh)
        self.assertEqual(actual, self.golden_metadata("empty.ini"))

    def test_identical_rebuild_preserves_complete_bytes(self):
        self.assert_rewrite_matches_fresh(self.metadata(), self.metadata())

    def test_larger_rebuild_matches_fresh_generation(self):
        current = self.metadata(detail="new option " * 40)
        self.assert_rewrite_matches_fresh(self.metadata(detail="old"), current)

    def test_existing_metadata_permissions_are_preserved(self):
        metadata = self.metadata()
        original = self.write(metadata, self.fresh)
        output = self.reused / "meta.ini"
        previous_mask = os.umask(0o027)
        try:
            descriptor = os.open(
                output,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP,
            )
        finally:
            os.umask(previous_mask)
        with os.fdopen(descriptor, "wb") as metadata_file:
            self.assertEqual(metadata_file.write(original), len(original))
        self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o640)
        rewritten = self.write(metadata, self.reused)
        self.assertEqual(rewritten, original)
        self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o640)


if __name__ == "__main__":
    unittest.main()
