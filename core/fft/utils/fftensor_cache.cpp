/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <unordered_map>

#include "log/log.h"

#include "utils/ops_base.h"
#include "utils/fftensor_cache.h"

constexpr int LEFT_MOVE_SIX = 6;
constexpr int RIGHT_MOVE_TWO = 2;
template <typename T>
inline void HashCombine(std::size_t& seed, const T& v)
{
    std::hash<T> hasher;
    seed ^= hasher(v) + 0x9e3779b9 + (seed << LEFT_MOVE_SIX) + (seed >> RIGHT_MOVE_TWO);
}

namespace {

bool g_doesTensorholdData(std::shared_ptr<AsdSip::FFTensor> fftTensorPtr)
{
    // 空指针防护（issue #380）：builder 契约禁止返回 nullptr（getCoeff 已校验），
    // 此处兜底返回 false，避免对可空 shared_ptr 直接解引用
    return fftTensorPtr != nullptr && fftTensorPtr->hostData != nullptr;
}

} // namespace

namespace AsdSip {

FFTensor::~FFTensor()
{
    if (data != nullptr) {
        FreeTensorInDevice(*this);
    }
    if (hostData != nullptr) {
        if (desc.dtype == Mki::TensorDType::TENSOR_DTYPE_INT32) {
            delete[] static_cast<int32_t*>(hostData);
        } else if (desc.dtype == Mki::TensorDType::TENSOR_DTYPE_FLOAT) {
            delete[] static_cast<float*>(hostData);
        } else if (desc.dtype == Mki::TensorDType::TENSOR_DTYPE_UINT32) {
            delete[] static_cast<std::uint32_t*>(hostData);
        }
        hostData = nullptr;
    }
}

bool CoeffKey::operator==(const CoeffKey& other) const
{
    return this->coreType == other.coreType && this->userDefine == other.userDefine && this->shape == other.shape &&
           this->isForward == other.isForward;
}

size_t CoeffKeyHash::operator()(const CoeffKey& one) const
{
    size_t hashSeed = 0;
    ::HashCombine(hashSeed, one.coreType);
    ::HashCombine(hashSeed, one.userDefine);
    ::HashCombine(hashSeed, one.isForward);
    for (auto dim : one.shape) {
        ::HashCombine(hashSeed, dim);
    }

    return hashSeed;
}

} // namespace AsdSip

namespace FFTensorCache {

static std::unordered_map<AsdSip::CoeffKey, std::weak_ptr<AsdSip::FFTensor>, AsdSip::CoeffKeyHash> coeffs;

std::shared_ptr<AsdSip::FFTensor> getCoeff(AsdSip::CoeffKey& key, std::function<AsdSip::FFTensor*()> func)
{
    std::shared_ptr<AsdSip::FFTensor> shared{nullptr};
    if (coeffs.find(key) == coeffs.end() || !(shared = coeffs.at(key).lock())) {
        // builder 契约：禁止返回 nullptr（issue #380）。构建失败一律以 throw 表达；
        // 此处对返回值判空兜底，避免空指针进入缓存后被 g_doesTensorholdData 解引用
        AsdSip::FFTensor* built = func();
        if (built == nullptr) {
            ASDSIP_LOG(ERROR) << "FFTensorCache coefficient builder returned nullptr.";
            throw std::runtime_error("fftensor cache coefficient builder returned nullptr.");
        }
        shared.reset(built);
        coeffs[key] = std::weak_ptr<AsdSip::FFTensor>(shared);
        if (g_doesTensorholdData(shared) && !MallocTensorInDevice(*shared).Ok()) {
            throw std::runtime_error("malloc coefficient matrix failed.");
        }
    }

    return shared;
}

} // namespace FFTensorCache
