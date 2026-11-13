/*
 * Copyright (c) 2026 gcw_2GoAgJFe. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/PythonTorchFunctionTLS.h>
#include <c10/core/impl/TorchDispatchModeTLS.h>
#include <torch/extension.h>
#include <torch/csrc/Device.h>
#include <torch/csrc/utils/python_arg_parser.h>

#include "kaiser_window_python.h"

namespace sip_pta {
namespace {
using KaiserSignature = at::Tensor(int64_t, bool, double, std::optional<at::ScalarType>, std::optional<at::Layout>,
                                   std::optional<at::Device>, bool);

pybind11::object RegisteredPythonCall(const pybind11::args& args, const pybind11::kwargs& kwargs)
{
    auto
        op = pybind11::module_::import("torch").attr("ops").attr("torch_sip").attr("asd_kaiser_window").attr("default");
    return op(*args, **kwargs);
}

pybind11::object InvokeKaiserWindow(const pybind11::args& args, const pybind11::kwargs& kwargs)
{
    if (at::impl::torch_function_mode_enabled() || c10::impl::dispatch_mode_enabled()) {
        return RegisteredPythonCall(args, kwargs);
    }
    // Unlike PythonArgParser's accelerator-index shorthand, the registered
    // Device schema accepts only None, a device object, or a device string.
    PyObject* suppliedDevice = kwargs ? PyDict_GetItemString(kwargs.ptr(), "device") : nullptr;
    if (suppliedDevice != nullptr && suppliedDevice != Py_None && !PyUnicode_Check(suppliedDevice) &&
        !THPDevice_Check(suppliedDevice)) {
        return RegisteredPythonCall(args, kwargs);
    }
    static torch::PythonArgParser parser(
        {"asd_kaiser_window(int64_t window_length, bool periodic=True, double beta=12.0, *, "
         "ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool requires_grad=False)"});
    torch::ParsedArgs<7> parsed;
    std::optional<torch::PythonArgs> parsedArguments;
    try {
        parsedArguments.emplace(parser.parse(args.ptr(), kwargs.ptr(), parsed));
    } catch (const torch::TypeError&) {
        // Preserve the registered operator's handling of noncanonical inputs.
        return RegisteredPythonCall(args, kwargs);
    }
    auto& arguments = *parsedArguments;
    if (arguments.has_torch_function()) {
        return RegisteredPythonCall(args, kwargs);
    }
    const auto length = arguments.toInt64(0);
    const auto periodic = arguments.toBool(1);
    const auto beta = arguments.toDouble(2);
    const auto dtype = arguments.scalartypeOptional(3);
    const auto layout = arguments.layoutOptional(4);
    const auto device = arguments.deviceOptional(5);
    const auto requiresGrad = arguments.toBool(6);
    static const auto op = c10::Dispatcher::singleton()
                               .findSchemaOrThrow("torch_sip::asd_kaiser_window", "")
                               .typed<KaiserSignature>();
    at::Tensor output;
    {
        pybind11::gil_scoped_release release;
        // Typed dispatch retains thread-local modes and profiler hooks without boxing
        // these scalar factory arguments into an IValue stack on every call.
        output = op.call(length, periodic, beta, dtype, layout, device, requiresGrad);
    }
    return pybind11::cast(output);
}
} // namespace

void RegisterKaiserWindowPython(pybind11::module_& module)
{
    module.def("_asd_kaiser_window_typed", &InvokeKaiserWindow);
}
} // namespace sip_pta
