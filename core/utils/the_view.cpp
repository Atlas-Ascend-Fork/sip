/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "utils/the_view.h"
#include "log/log.h"

namespace wten {

template <typename T>
TheView<T> TheView<T>::select(const TheView<T>& view, int64_t dim, int64_t index)
{
    if (dim < 0 || dim >= view.ndim()) {
        ASDSIP_LOG(ERROR) << "Invalid dim.";
        throw std::runtime_error("Invalid dim.");
    }

    if (view.shape_[dim] == 0) {
        ASDSIP_LOG(ERROR) << "Invalid view shape.";
        throw std::runtime_error("Invalid view shape.");
    }

    if (index >= view.shape_[dim] || index < 0) {
        ASDSIP_LOG(ERROR) << "Invalid index.";
        throw std::runtime_error("Invalid index.");
    }

    TheView selected{view};
    selected.size_ /= selected.shape_[dim];
    selected.offset_ += selected.strides_[dim] * index;
    selected.shape_.erase(selected.shape_.begin() + dim);
    selected.strides_.erase(selected.strides_.begin() + dim);

    return selected;
}

template <typename T>
TheView<T> TheView<T>::slice(const TheView<T>& view, int64_t dim, int64_t start, int64_t end, int64_t step)
{
    // 与 select() 同型防护：dim 范围 / start 非负，防止越界读与负偏移（issue #290）
    if (dim < 0 || dim >= view.ndim()) {
        ASDSIP_LOG(ERROR) << "Invalid dim.";
        throw std::runtime_error("Invalid dim.");
    }

    if (start < 0) {
        ASDSIP_LOG(ERROR) << "start must be no less than 0.";
        throw std::runtime_error("start must be no less than 0.");
    }

    if (start >= end) {
        ASDSIP_LOG(ERROR) << "start must be less than end.";
        throw std::runtime_error("start must be less than end.");
    }

    if (step <= 0) {
        ASDSIP_LOG(ERROR) << "step must be greater than 0.";
        throw std::runtime_error("step must be greater than 0.");
    }

    if (end > view.shape_[dim]) {
        ASDSIP_LOG(ERROR) << "end must be no greater than view shape[ " << dim << " ]";
        throw std::runtime_error("end must be no greater than view shape[dim].");
    }

    TheView sliced{view};

    int64_t n = (end - start + step - 1) / step;

    sliced.size_ /= sliced.shape_[dim];
    sliced.size_ *= n;
    sliced.offset_ += start * sliced.strides_[dim];
    sliced.shape_[dim] = n;
    sliced.strides_[dim] *= step;

    return sliced;
}

template class TheView<double>;
template class TheView<float>;
template class TheView<int64_t>;

} // namespace wten
