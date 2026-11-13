#!/usr/bin/env python3
# Copyright (c) 2026 gcw_2GoAgJFe. All rights reserved.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Run directly after building torch_sip: python test/base/kaiser_window.py."""

import itertools
import sys
import unittest
from unittest import mock

try:
    import torch
    import torch_npu  # noqa: F401
    import torch_sip
except ImportError as exc:
    print(f"BLOCKED: {exc}", file=sys.stderr)
    sys.exit(2)


class TestKaiserWindow(unittest.TestCase):
    def test_values(self):
        tolerances = {
            torch.float16: 2**-10,
            torch.bfloat16: 2**-7,
            torch.float32: 2**-13,
        }
        for dtype, length, periodic, beta in itertools.product(
            tolerances,
            (0, 1, 2, 3, 17, 4097, 65536),
            (False, True),
            (0.0, 4.5, -12.0, 88.0),
        ):
            with self.subTest(dtype=dtype, length=length, periodic=periodic, beta=beta):
                actual = torch_sip.asd_kaiser_window(
                    length, periodic, beta, dtype=dtype
                )
                self.assertEqual(actual.shape, (length,))
                self.assertEqual(actual.dtype, dtype)
                self.assertEqual(actual.device.type, "npu")
                self.assertTrue(actual.is_contiguous())
                expected = torch.kaiser_window(
                    length, periodic, beta, dtype=torch.float64
                ).to(dtype)
                torch.testing.assert_close(
                    actual.cpu(),
                    expected,
                    rtol=tolerances[dtype],
                    atol=tolerances[dtype],
                )

    def test_nonfinite_beta(self):
        for length, periodic, beta in itertools.product(
            (0, 1, 2, 17), (False, True), (float("nan"), float("inf"), -float("inf"))
        ):
            with self.subTest(length=length, periodic=periodic, beta=beta):
                actual = torch_sip.asd_kaiser_window(length, periodic, beta).cpu()
                if length == 0:
                    self.assertEqual(actual.numel(), 0)
                elif length == 1:
                    self.assertEqual(actual.item(), 1.0)
                else:
                    self.assertTrue(torch.isnan(actual).all().item())

    def test_defaults_and_direct_dispatch(self):
        saved_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            actual = torch_sip.asd_kaiser_window(17, beta=4.5, layout=None)
            self.assertEqual(actual.dtype, torch.float32)
            self.assertFalse(actual.requires_grad)
            direct = torch.ops.torch_sip.asd_kaiser_window(17, beta=4.5)
            self.assertTrue(torch.equal(actual, direct))
            defaults = torch_sip.asd_kaiser_window(17)
            explicit = torch_sip.asd_kaiser_window(
                17, True, 12.0, dtype=torch.float32, layout=torch.strided, device="npu"
            )
            self.assertTrue(torch.equal(defaults, explicit))
        finally:
            torch.set_default_dtype(saved_dtype)

    def test_gradient_leaf(self):
        for length in (0, 1, 17):
            actual = torch_sip.asd_kaiser_window(length, requires_grad=True)
            self.assertTrue(actual.requires_grad)
            self.assertTrue(actual.is_leaf)
            (actual * 2).sum().backward()
            torch.testing.assert_close(actual.grad, torch.full_like(actual, 2))

    def test_current_stream_and_device(self):
        current_device = torch.npu.current_device()
        stream = torch.npu.Stream(device=current_device)
        with torch.npu.stream(stream):
            actual = torch_sip.asd_kaiser_window(
                4097, device=torch.device("npu", current_device)
            )
            repeated = torch_sip.asd_kaiser_window(4097)
        stream.synchronize()
        self.assertEqual(actual.device.index, current_device)
        self.assertTrue(torch.equal(actual, repeated))
        if torch.npu.device_count() > 1:
            other_device = (current_device + 1) % torch.npu.device_count()
            actual = torch_sip.asd_kaiser_window(17, device=f"npu:{other_device}")
            self.assertEqual(actual.device.index, other_device)
            self.assertEqual(torch.npu.current_device(), current_device)
            expected = torch.kaiser_window(17, dtype=torch.float64).float()
            torch.testing.assert_close(actual.cpu(), expected, rtol=2**-13, atol=2**-13)

    def test_dispatch_modes_and_compiler_entry(self):
        from torch.overrides import TorchFunctionMode
        from torch.utils._python_dispatch import TorchDispatchMode

        class FunctionRecorder(TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append(str(func))
                return func(*args, **(kwargs or {}))

        class DispatchRecorder(TorchDispatchMode):
            def __init__(self):
                super().__init__()
                self.calls = []

            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                self.calls.append(str(func))
                return func(*args, **(kwargs or {}))

        expected = torch.ops.torch_sip.asd_kaiser_window.default(17)
        for mode_type in (FunctionRecorder, DispatchRecorder):
            with mode_type() as mode:
                actual = torch_sip.asd_kaiser_window(17)
            self.assertTrue(
                any("torch_sip.asd_kaiser_window" in name for name in mode.calls)
            )
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        # Compilers must see the registered OpOverload, rather than a pybind call.
        with (
            mock.patch.object(torch_sip, "_kaiser_is_compiling", return_value=True),
            mock.patch.object(
                torch_sip,
                "_asd_kaiser_window",
                side_effect=AssertionError("unexpected eager entry"),
            ),
        ):
            actual = torch_sip.asd_kaiser_window(17)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_torch_dispatch_mode_uses_registered_python_entry(self):
        from torch.utils._python_dispatch import TorchDispatchMode

        class PassthroughMode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                return func(*args, **(kwargs or {}))

        expected = torch.empty(1, device="npu")
        with (
            PassthroughMode(),
            mock.patch.object(
                torch.ops.torch_sip.asd_kaiser_window,
                "default",
                return_value=expected,
            ) as registered_call,
        ):
            actual = torch_sip.asd_kaiser_window(17, beta=4.5)
        self.assertIs(actual, expected)
        registered_call.assert_called_once()

    def test_registered_argument_compatibility(self):
        registered = torch.ops.torch_sip.asd_kaiser_window.default
        cases = [
            ((-1,), {}),
            ((17,), {"dtype": torch.float64}),
            ((17,), {"device": "cpu"}),
            ((17,), {"layout": torch.sparse_coo}),
            ((17.5,), {}),
            ((17,), {"periodic": 1}),
            ((17,), {"layout": 0}),
            ((17,), {"device": object()}),
            ((17,), {"device": 0}),
            ((17,), {"device": -1}),
            ((17,), {"dtype": 0}),
            ((17,), {"requires_grad": 1}),
            ((True,), {}),
            ((2**63,), {}),
        ]
        for args, kwargs in cases:
            with self.subTest(args=args, kwargs=kwargs):
                try:
                    expected = registered(*args, **kwargs)
                except Exception as expected_error:
                    with self.assertRaises(type(expected_error)):
                        torch_sip.asd_kaiser_window(*args, **kwargs)
                else:
                    actual = torch_sip.asd_kaiser_window(*args, **kwargs)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    if not torch.npu.is_available():
        print("BLOCKED: no NPU available", file=sys.stderr)
        sys.exit(2)
    unittest.main()
