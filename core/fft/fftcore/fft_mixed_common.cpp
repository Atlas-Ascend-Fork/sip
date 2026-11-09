/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "fftcore/fft_mixed_common_core.h"

#include <mki/utils/rt/rt.h>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <iostream>
#include <cstdlib>
#include "ops.h"
using namespace AsdSip;
using namespace Mki;
using namespace AsdSip::OpParam;
namespace {
Tensor Signal(int64_t count, void* data = nullptr)
{
    Tensor t;
    t.desc = {TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {count}, {}, 0};
    t.dataSize = count * sizeof(float);
    t.data = data;
    return t;
}
} // namespace
struct FftMixedCommonCore::Impl {
    FftMixedCommon param;
    LaunchParam launch;
    RunInfo run;
    std::unique_ptr<Kernel> kernel;
    void* tw = nullptr;
    void* tiling = nullptr;
    std::vector<uint8_t> hostTiling;
    ~Impl()
    {
        if (tw)
            MkiRtMemFreeDevice(tw);
        if (tiling)
            MkiRtMemFreeDevice(tiling);
    }
};
FftMixedCommonCore::FftMixedCommonCore(FftMixedCommon param) : impl_(std::make_unique<Impl>()) { impl_->param = param; }
FftMixedCommonCore::~FftMixedCommonCore() = default;
bool FftMixedCommonCore::init()
{
    auto& p = *impl_;
    if (!ValidMixedCommon(p.param))
        return false;
    std::vector<float> tw;
    tw.resize(2 * p.param.n);
    const double sign = p.param.inverse ? 1.0 : -1.0;
    for (int64_t k = 0; k < p.param.n; ++k) {
        const double angle = sign * 6.28318530717958647692 * double(k) / double(p.param.n);
        tw[2 * k] = float(std::cos(angle));
        tw[2 * k + 1] = float(std::sin(angle));
    }
    if (MkiRtMemMallocDevice(&p.tw, tw.size() * sizeof(float), MKIRT_MEM_DEFAULT) != MKIRT_SUCCESS ||
        MkiRtMemCopy(p.tw, tw.size() * sizeof(float), tw.data(), tw.size() * sizeof(float),
                     MKIRT_MEMCOPY_HOST_TO_DEVICE) != MKIRT_SUCCESS)
        return false;
    p.launch.SetParam(p.param);
    p.launch.AddInTensor(Signal(MixedCommonFloats(p.param, false)));
    p.launch.AddInTensor(Signal(int64_t(tw.size()), p.tw));
    p.launch.AddOutTensor(Signal(MixedCommonFloats(p.param, true)));
    const std::string name = "FftMixedCommonOperation";
    auto* op = Ops::Instance().GetOperationByName(name);
    if (!op || !op->InferShape(p.launch).Ok())
        return false;
    p.kernel.reset(op->GetBestKernel(p.launch));
    if (!p.kernel || !p.kernel->CanSupport(p.launch))
        return false;
    p.kernel->SetLaunchWithTiling(false);
    const auto bytes = p.kernel->GetTilingSize(p.launch);
    p.hostTiling.resize(bytes, 0);
    p.kernel->SetTilingHostAddr(p.hostTiling.data(), bytes);
    if (!p.kernel->Init(p.launch).Ok() || MkiRtMemMallocDevice(&p.tiling, bytes, MKIRT_MEM_DEFAULT) != MKIRT_SUCCESS ||
        MkiRtMemCopy(p.tiling, bytes, p.hostTiling.data(), bytes, MKIRT_MEMCOPY_HOST_TO_DEVICE) != MKIRT_SUCCESS)
        return false;
    p.run.SetTilingDeviceAddr(static_cast<uint8_t*>(p.tiling));
    const char* trace = std::getenv("ASDSIP_FFT_PLAN_TRACE");
    if (trace && std::string(trace) == "1")
        std::cerr << "ASDSIP_MIXED_FFT N=" << p.param.n << " batch=" << p.param.rows << " pass=" << int(p.param.pass)
                  << " inverse=" << p.param.inverse << " two_frequency=" << (p.param.n >= 1824) << '\n';
    return true;
}
void FftMixedCommonCore::Run(Tensor& x, Tensor& y, void* stream, workspace::Workspace& w)
{
    Run(x.data, y.data, stream, w);
}
size_t FftMixedCommonCore::EstimateWorkspaceSize()
{
    return getAlignedSize(MixedCommonFloats(impl_->param, true) * sizeof(float));
}
void FftMixedCommonCore::Run(void* x, void* y, void* stream, workspace::Workspace& workspace)
{
    auto& p = *impl_;
    const size_t inBytes = MixedCommonFloats(p.param, false) * sizeof(float);
    const size_t outBytes = MixedCommonFloats(p.param, true) * sizeof(float);
    const auto in = reinterpret_cast<uintptr_t>(x), out = reinterpret_cast<uintptr_t>(y);
    if (!in || !out || in % 32 || out % 32 || (in != out && (in < out ? out - in < inBytes : in - out < outBytes)))
        throw std::invalid_argument("Mixed FFT requires aligned disjoint buffers or exact aliasing");
    const bool alias = x == y;
    void* target = alias ? workspace.allocate(EstimateWorkspaceSize()) : y;
    try {
        p.launch.GetInTensor(0).data = x;
        p.launch.GetOutTensor(0).data = target;
        p.run.SetStream(stream);
        if (!p.kernel->Run(p.launch, p.run).Ok())
            throw std::runtime_error("Mixed FFT launch failed");
        if (alias &&
            MkiRtMemCopyAsync(y, outBytes, target, outBytes, MKIRT_MEMCOPY_DEVICE_TO_DEVICE, stream) != MKIRT_SUCCESS)
            throw std::runtime_error("Mixed FFT alias copy-back failed");
    } catch (...) {
        if (alias)
            workspace.recycleLast();
        throw;
    }
    if (alias)
        workspace.recycleLast();
}
