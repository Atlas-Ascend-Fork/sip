/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once

#include <cmath>
#include "utils/dim_constants.h"
#include "utils/the_view.h"

namespace wten {

template <typename T>
void arange(int64_t start, int64_t end, int64_t step, TheTensor<T>& out)
{
    // 零步长防护（issue #293）：step==0 时 -step==0，落入 else 分支即有符号整数
    // 除零 UB（典型 SIGFPE），且除零先于下方 size 校验发生，校验无法挽救
    if (step == 0) {
        ASDSIP_LOG(ERROR) << "arange step must not be zero.";
        return;
    }
    int64_t n;
    if (step > 0) {
        n = (end - 1 - start) / step + 1;
    } else {
        n = (-end - 1 + start) / -step + 1;
    }

    if (n != static_cast<int64_t>(out.size())) {
        ASDSIP_LOG(ERROR) << "Output does not match size.";
        return;
    }

    T* out_data = out.data();
    int64_t item = start;
    for (int64_t i = 0; i < n; i++, item += step) {
        out_data[i] = item;
    }
}

template <typename T>
bool commonShapeCheck(TheTensor<T>& left, TheTensor<T>& right, TheTensor<T>& out)
{
    if (left.ndim() != SHAPE_TWO || right.ndim() != SHAPE_TWO) {
        ASDSIP_LOG(ERROR) << "Inputs must be 2-dimension.";
        return false;
    }

    if (out.ndim() != SHAPE_TWO) {
        ASDSIP_LOG(ERROR) << "Output must be 2-dimension.";
        return false;
    }

    if (left.shape()[DIM_ONE] != SHAPE_ONE) {
        ASDSIP_LOG(ERROR) << "Left's second dimension must be 1.";
        return false;
    }

    if (right.shape()[DIM_ZERO] != SHAPE_ONE) {
        ASDSIP_LOG(ERROR) << "right's first dimension must be 1.";
        return false;
    }

    if (out.shape()[DIM_ZERO] != left.shape()[DIM_ZERO] || out.shape()[DIM_ONE] != right.shape()[DIM_ONE] ||
        out.shape()[DIM_ONE] == 0) {
        ASDSIP_LOG(ERROR) << "Output's shape does not match inputs' shape, or out.shape()[DIM_ONE] == 0.";
        return false;
    }

    return true;
}

template <typename T>
void outerAdd(TheTensor<T>& left, TheTensor<T>& right, TheTensor<T>& out)
{
    bool checkStatus = commonShapeCheck(left, right, out);
    if (!checkStatus) {
        return;
    }

    size_t step = out.shape()[DIM_ONE];
    for (size_t n = 0; n < out.size(); n++) {
        size_t i = n / step;
        size_t j = n % step;
        out[n] = left[i] + right[j];
    }
}

template <typename T>
void outerMul(TheTensor<T>& left, TheTensor<T>& right, TheTensor<T>& out)
{
    bool checkStatus = commonShapeCheck(left, right, out);
    if (!checkStatus) {
        return;
    }

    size_t step = out.shape()[1];
    for (size_t n = 0; n < out.size(); n++) {
        size_t i = n / step;
        size_t j = n % step;
        out[n] = left[i] * right[j];
    }
}

template <typename T>
void mulScalar(TheTensor<T>& in, T scalar, TheTensor<T>& out)
{
    if (in.shape() != out.shape()) {
        ASDSIP_LOG(ERROR) << "Output does not match Input.";
        return;
    }

    for (size_t i = 0; i < out.size(); i++) {
        out[i] = in[i] * scalar;
    }
}

template <typename T>
void modScalar(TheTensor<T>& src, T scalar, TheTensor<T>& dst)
{
    if (src.shape() != dst.shape()) {
        ASDSIP_LOG(ERROR) << "Output does not match Input.";
        return;
    }
    // 取模零防护（issue #293）：scalar==0 时整数取模为 UB（典型 SIGFPE）
    if (scalar == T(0)) {
        ASDSIP_LOG(ERROR) << "modScalar scalar must not be zero.";
        return;
    }

    for (size_t index = 0; index < dst.size(); index++) {
        dst[index] = src[index] % scalar;
    }
}

template <typename T>
void neg(TheTensor<T>& in, TheTensor<T>& out)
{
    if (in.shape() != out.shape()) {
        ASDSIP_LOG(ERROR) << "Output does not match Input.";
        return;
    }

    for (size_t i = 0; i < out.size(); i++) {
        out[i] = 0 - in[i];
    }
}

template <typename T>
void sin(TheTensor<T>& in, TheTensor<T>& out)
{
    if (in.shape() != out.shape()) {
        ASDSIP_LOG(ERROR) << "Output does not match Input.";
        return;
    }

    for (size_t i = 0; i < out.size(); i++) {
        out[i] = std::sin(in[i]);
    }
}

template <typename T>
void cos(TheTensor<T>& in, TheTensor<T>& out)
{
    if (in.shape() != out.shape()) {
        ASDSIP_LOG(ERROR) << "Output does not match Input.";
        return;
    }

    for (size_t i = 0; i < out.size(); i++) {
        out[i] = std::cos(in[i]);
    }
}

template <typename T>
void transpose(TheTensor<T>& in, const std::vector<int64_t>& permute_dims, TheTensor<T>& out)
{
    if (in.ndim() != permute_dims.size()) {
        ASDSIP_LOG(ERROR) << "permute_dims does not match input's shape.";
        return;
    }

    if (in.ndim() != out.ndim()) {
        ASDSIP_LOG(ERROR) << "Output's ndim does not match input's ndim.";
        return;
    }

    // 元素范围 + 唯一性校验：越界值会引发 vector 越界读写，
    // 重复值会让 reversed_dims 留下未写入槽位、结果静默错误（issue #289）
    std::vector<bool> seen(permute_dims.size(), false);
    for (int64_t i = 0; i < static_cast<int64_t>(permute_dims.size()); i++) {
        if (permute_dims[i] < 0 || permute_dims[i] >= static_cast<int64_t>(permute_dims.size()) ||
            seen[permute_dims[i]]) {
            ASDSIP_LOG(ERROR) << "permute_dims must be a permutation of [0, ndim).";
            return;
        }
        seen[permute_dims[i]] = true;
    }

    for (int64_t i = 0; i < static_cast<int64_t>(permute_dims.size()); i++) {
        if (in.shape()[permute_dims[i]] != out.shape()[i]) {
            ASDSIP_LOG(ERROR) << "Output's permute dim does not match input's permute dim.";
            return;
        }
    }

    std::vector<int64_t> reversed_dims(permute_dims.size());
    for (int64_t i = 0; i < static_cast<int64_t>(permute_dims.size()); i++) {
        reversed_dims[permute_dims[i]] = i;
    }

    const std::vector<int64_t>& in_shape = in.shape();
    const std::vector<int64_t>& out_strides = out.strides();

    for (int64_t index = 0; index < static_cast<int64_t>(out.size()); index++) {
        int64_t remain = index;
        int64_t permute_index = 0;
        for (int64_t j = static_cast<int64_t>(in.ndim()) - 1; j >= 0; j--) {
            if (in_shape[j] == 0) {
                continue;
            }
            permute_index += (remain % in_shape[j]) * out_strides[reversed_dims[j]];
            remain /= in_shape[j];

            if (remain == 0) {
                break;
            }
        }
        out[permute_index] = in[index];
    }
}

template <typename T>
void copy(TheTensor<T>& src, TheView<T>& dst)
{
    if (src.shape() != dst.shape()) {
        ASDSIP_LOG(ERROR) << "Output does not match Input.";
        return;
    }

    for (int64_t index = 0; index < dst.size(); index++) {
        dst[index] = src[index];
    }
}

template <typename T>
void copy(TheView<T>& src, TheView<T>& dst)
{
    if (src.shape() != dst.shape()) {
        ASDSIP_LOG(ERROR) << "Output does not match Input.";
        return;
    }

    for (int64_t index = 0; index < dst.size(); index++) {
        dst[index] = src[index];
    }
}

template <typename T, typename E>
void cast(TheTensor<T>& src, TheTensor<E>& dst)
{
    if (src.shape() != dst.shape()) {
        ASDSIP_LOG(ERROR) << "Output does not match Input.";
        return;
    }

    for (size_t index = 0; index < dst.size(); index++) {
        dst[index] = static_cast<E>(src[index]);
    }
}

template <typename T>
TheTensor<T> arange(int64_t start, int64_t end, int64_t step)
{
    // 工厂重载在调用内层 arange 前先自行计算 n，step==0 时同样落入除零，
    // 需与内层重载同等防护（issue #293 / PR #218 检视补充）
    if (step == 0) {
        ASDSIP_LOG(ERROR) << "arange step must not be zero.";
        throw std::runtime_error("arange step must not be zero.");
    }

    int64_t n;
    if (step > 0) {
        n = (end - 1 - start) / step + 1;
    } else {
        n = (-end - 1 + start) / -step + 1;
    }

    if (n <= 0) {
        ASDSIP_LOG(ERROR) << "Infered size must be greater than 0.";
        throw std::runtime_error("Infered size must be greater than 0.");
    }

    TheTensor<T> out{{n}};

    arange(start, end, step, out);

    return out;
}

template <typename T>
TheTensor<T> outerAdd(TheTensor<T>& left, TheTensor<T>& right)
{
    if (left.ndim() != SHAPE_TWO || right.ndim() != SHAPE_TWO) {
        ASDSIP_LOG(ERROR) << "Inputs must be 2-dimension.";
        throw std::runtime_error("Inputs must be 2-dimension.");
    }

    TheTensor<T> out{{left.shape()[0], right.shape()[1]}};

    outerAdd(left, right, out);

    return out;
}

template <typename T>
TheTensor<T> outerMul(TheTensor<T>& left, TheTensor<T>& right)
{
    if (left.ndim() != SHAPE_TWO || right.ndim() != SHAPE_TWO) {
        ASDSIP_LOG(ERROR) << "Inputs must be 2-dimension.";
        throw std::runtime_error("Inputs must be 2-dimension.");
    }

    TheTensor<T> out{{left.shape()[0], right.shape()[1]}};

    outerMul(left, right, out);

    return out;
}

template <typename T>
TheTensor<T> mulScalar(TheTensor<T>& in, T scalar)
{
    TheTensor<T> out{in.shape()};

    mulScalar(in, scalar, out);

    return out;
}

template <typename T>
TheTensor<T> modScalar(TheTensor<T>& in, T scalar)
{
    // 工厂重载先校验再分配，避免 scalar==0 时内层 return 后返回未填充张量
    // （PR #218 检视 P3；抛异常风格与 arange/transpose 工厂重载一致）
    if (scalar == T(0)) {
        ASDSIP_LOG(ERROR) << "modScalar scalar must not be zero.";
        throw std::runtime_error("modScalar scalar must not be zero.");
    }

    TheTensor<T> out{in.shape()};

    modScalar(in, scalar, out);

    return out;
}

template <typename T>
TheTensor<T> neg(TheTensor<T>& in)
{
    TheTensor<T> out{in.shape()};

    neg(in, out);

    return out;
}

template <typename T>
TheTensor<T> sin(TheTensor<T>& in)
{
    TheTensor<T> out{in.shape()};

    sin(in, out);

    return out;
}

template <typename T>
TheTensor<T> cos(TheTensor<T>& in)
{
    TheTensor<T> out{in.shape()};

    cos(in, out);

    return out;
}

template <typename T>
TheTensor<T> transpose(TheTensor<T>& in, const std::vector<int64_t>& permute_dims)
{
    // 先校验范围/唯一性再取下标，防止越界读（issue #289）；
    // 非法输入按本文件值返回版助手既有约定抛异常（PR #220 检视口径统一）
    if (in.ndim() != permute_dims.size()) {
        ASDSIP_LOG(ERROR) << "permute_dims does not match input's shape.";
        throw std::runtime_error("permute_dims does not match input's shape.");
    }
    std::vector<bool> seen(permute_dims.size(), false);
    for (int64_t i = 0; i < static_cast<int64_t>(permute_dims.size()); i++) {
        if (permute_dims[i] < 0 || permute_dims[i] >= static_cast<int64_t>(permute_dims.size()) ||
            seen[permute_dims[i]]) {
            ASDSIP_LOG(ERROR) << "permute_dims must be a permutation of [0, ndim).";
            throw std::runtime_error("permute_dims must be a permutation of [0, ndim).");
        }
        seen[permute_dims[i]] = true;
    }

    std::vector<int64_t> out_shape(in.shape());
    for (size_t i = 0; i < out_shape.size(); i++) {
        out_shape[i] = in.shape()[permute_dims[i]];
    }
    TheTensor<T> out{out_shape};

    transpose(in, permute_dims, out);

    return out;
}

template <typename T, typename E>
TheTensor<E> cast(TheTensor<T>& src)
{
    TheTensor<E> dst{src.shape()};

    cast(src, dst);

    return dst;
}

} // namespace wten
