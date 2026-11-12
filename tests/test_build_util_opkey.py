#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Unit tests for scripts/build_util.py tbe tactic op_key parsing (issue #389).

op_key 由 scripts/update_tbe_tactic_json.py 以 tactic_name + "." + target_version 拼接
生成，tactic 名本身可含点号（如 tactic.v2 -> key 为 tactic.v2.<version>）；修复前用
split(".") 两元素解包，多点号 key 抛 ValueError 中断构建。修复后 rsplit(".", 1)
按最后一个点号拆分，单点号与无点号 key 行为不变。
"""

import configparser
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import build_util  # noqa: E402


TARGET_VERSION = "Ascend910B"

# tactic json 最小合法内容（get_info_from_file 要求的字段）
TACTIC_JSON = {
    "binFileName": "kernel_bin",
    "compileInfo": {"block_dim": "1"},
    "opParaSize": 8,
    "coreType": "VectorCore",
    "magic": "RT_DEV_BINARY_MAGIC_ELF",
    "kernelName": "kernel_bin",
}


def make_tactic_file(kernel_dir, relative_op_path):
    """按 relative_op_path（.json 结尾）在 kernel_dir/<target_version>/ 下生成配套 json + o 占位文件"""
    json_path = Path(kernel_dir) / TARGET_VERSION / relative_op_path
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(TACTIC_JSON), encoding="utf-8")
    obj_path = json_path.with_suffix(".o")
    obj_path.write_bytes(b"\x00obj")


def make_input_paras(cache_dir, kernel_dir, op_items):
    tbe_ini = configparser.RawConfigParser()
    tbe_ini.optionxform = lambda option: option
    tbe_ini.add_section("ConvOp")
    for key, value in op_items.items():
        tbe_ini.set("ConvOp", key, value)
    return {
        "env_code_root": str(cache_dir),
        "target_version_list": [TARGET_VERSION],
        "env_cache_dir": str(cache_dir),
        "tbe_kernel_path": str(kernel_dir),
        "tbe_ini": tbe_ini,
    }


class BuildUtilOpKeyTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.cache_dir = Path(self._tmp.name)
        self.kernel_dir = self.cache_dir / "kernel"
        self.kernel_dir.mkdir()
        self.addCleanup(self._tmp.cleanup)

    def read_meta(self):
        meta_path = self.cache_dir / "obj" / TARGET_VERSION / "meta.ini"
        self.assertTrue(meta_path.exists(), "meta.ini not generated")
        return meta_path.read_text(encoding="utf-8")

    def test_multi_dot_key_parsed(self):
        """多点号 key（tactic 名含点号）正确拆分并按版本过滤，构建不再中断"""
        make_tactic_file(self.kernel_dir, Path("conv") / "tactic_v2.json")
        paras = make_input_paras(
            self.cache_dir,
            self.kernel_dir,
            {"tactic.v2." + TARGET_VERSION: "conv/tactic_v2.json"},
        )
        build_util.copy_tbe_code_all_version(paras)
        meta = self.read_meta()
        self.assertIn("tactic.v2.Object=", meta)
        self.assertIn("tactic.v2.OpName=ConvOp", meta)

    def test_multi_dot_key_version_filtered(self):
        """多点号 key 版本不匹配时跳过，不产出 meta 记录"""
        make_tactic_file(self.kernel_dir, Path("conv") / "tactic_v9.json")
        paras = make_input_paras(
            self.cache_dir, self.kernel_dir, {"tactic.v9.Other": "conv/tactic_v9.json"}
        )
        build_util.copy_tbe_code_all_version(paras)
        self.assertNotIn("tactic.v9.Object=", self.read_meta())

    def test_single_dot_and_plain_key_unchanged(self):
        """单点号与无点号 key 行为与修复前一致：单点号按版本过滤、无点号直接生效"""
        make_tactic_file(self.kernel_dir, Path("conv") / "plain.json")
        make_tactic_file(self.kernel_dir, Path("conv") / "matched.json")
        make_tactic_file(self.kernel_dir, Path("conv") / "mismatched.json")
        paras = make_input_paras(
            self.cache_dir,
            self.kernel_dir,
            {
                "plain": "conv/plain.json",
                "matched." + TARGET_VERSION: "conv/matched.json",
                "mismatched.Other": "conv/mismatched.json",
            },
        )
        build_util.copy_tbe_code_all_version(paras)
        meta = self.read_meta()
        self.assertIn("plain.Object=", meta)
        self.assertIn("matched.Object=", meta)
        self.assertNotIn("mismatched.Object=", meta)


if __name__ == "__main__":
    unittest.main()
