/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASDSIP_FFT_API_H
#define ASDSIP_FFT_API_H

#include "utils/aspb_status.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct aclTensor aclTensor;

namespace AsdSip {
typedef void* asdFftHandle;

enum asdFftStatus { SUCCESS = 0, FAILED = 1 };

enum asdFftType {
    ASCEND_FFT_C2C = 0x10,
    ASCEND_FFT_C2R = 0x11,
    ASCEND_FFT_R2C = 0x12,
    ASCEND_STFT_C2C = 0x20,
    ASCEND_STFT_C2R = 0x21,
    ASCEND_STFT_R2C = 0x22,
    ASCEND_FFT_C2C_SEP = 0x30
};

enum asdFftDirection {
    ASCEND_FFT_FORWARD = 0x10,
    ASCEND_FFT_INVERSE = 0x11,
};

enum asdFft1dDimType {
    ASCEND_FFT_HORIZONTAL = 0x10,
    ASCEND_FFT_VERTICAL = 0x11,
};

// Creates only an opaque handle, and only allocates space basic data structures for it.
// This function does not initialize handle.
AsdSip::AspbStatus asdFftCreate(asdFftHandle& handle);

// Binds a stream to handle.
AsdSip::AspbStatus asdFftSetStream(asdFftHandle handle, void* stream);

// Initializes handle with 1D setting.
// Any given handle can only be initialized once.
AsdSip::AspbStatus asdFftMakePlan1D(asdFftHandle handle, int64_t fftSize, asdFftType fftType, asdFftDirection direction,
                                    int64_t batchSize,
                                    asdFft1dDimType dimType = asdFft1dDimType::ASCEND_FFT_HORIZONTAL);

AspbStatus asdFftIstftMakePlan(asdFftHandle handle, const aclTensor* input, const int64_t nFft,
                               const int64_t hopLengthOpt, const int64_t winLengthOpt, const bool center = true,
                               const bool normalized = false, const bool onesidedOpt = false, int64_t lengthOpt = 0,
                               const bool returnComplex = true);

// Executes a c2c fft transform.
AsdSip::AspbStatus asdFftExecC2C(asdFftHandle handle, const aclTensor* input, const aclTensor* output);

// Executes a c2c fft transform with real part and imag part separated.
AsdSip::AspbStatus asdFftExecC2CSeparated(asdFftHandle handle, const aclTensor* inputReal, const aclTensor* inputImag,
                                          const aclTensor* outputReal, const aclTensor* outputImag);

// Executes a c2r fft transform.
AsdSip::AspbStatus asdFftExecC2R(asdFftHandle handle, const aclTensor* input, const aclTensor* output);

// Executes a r2c fft transform.
AsdSip::AspbStatus asdFftExecR2C(asdFftHandle handle, const aclTensor* input, const aclTensor* output);

AspbStatus asdFftExecIstft(asdFftHandle handle, const aclTensor* input, const aclTensor* windowOpt,
                           const aclTensor* output);

// Destroy a handle, free all spaces allocated for it.
AsdSip::AspbStatus asdFftDestroy(asdFftHandle handle);

AsdSip::AspbStatus asdFftMakePlan2D(asdFftHandle handle, int64_t fftSizeX, int64_t fftSizeY, asdFftType fftType,
                                    asdFftDirection direction, int32_t batchSize);

AspbStatus asdFftMakePlan3D(asdFftHandle handle, int64_t fftSizeX, int64_t fftSizeY, int64_t fftSizeZ,
                            asdFftType fftType, asdFftDirection direction, int32_t batchSize);

AsdSip::AspbStatus asdFftGetWorkspaceSize(asdFftHandle handle, size_t& workspaceSize);

AsdSip::AspbStatus asdFftSetWorkspace(asdFftHandle handle, void* workspace);

AsdSip::AspbStatus asdFftSynchronize(asdFftHandle handle);

AsdSip::AspbStatus asdFftGetType(asdFftHandle handle, asdFftType& fftType);

// Shifts the zero-frequency component of batched complex slices to the center of each slice
// (aligned with the reference CUDA fftShiftBatch2DKernel):
//   output[s][(r + (row + 1) / 2) % row][(c + (col + 1) / 2) % col] = input[s][r][c]
// input holds `batch` slices of `row` x `col` complex64 elements, slice s starting at element
// offset `s * dist`; output is a contiguous `batch` x `row` x `col` complex64 tensor.
// A non-positive `dist` means the input slices are contiguous (dist = row * col).
// NOTE: the current version supports 2D slices (row x col) only; support for other
// dimensions will be added in a later release.
AsdSip::AspbStatus asdFftShift(int64_t row, int64_t col, int64_t batch, int64_t dist, const aclTensor* input,
                               aclTensor* output, void* stream, void* workspace = nullptr);
} // namespace AsdSip

#ifdef __cplusplus
}
#endif

#endif
