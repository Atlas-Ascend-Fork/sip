#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
set -euo pipefail
: "${ASCEND_HOME_PATH:?source CANN environment first}"
SOURCE=$(realpath "${1:?source tree}")
INSTALL=$(realpath "${2:?SIP install prefix}")
DEST=${3:?binary directory}
BUILD=$(realpath "${4:?SIP build directory}")
mkdir -p "${DEST:?binary directory}"
g++ -std=c++17 -O2 -D_GLIBCXX_USE_CXX11_ABI=0 "${SOURCE:?source tree}/tests/standalone/fft_transpose/bench_transpose.cpp" \
    -I"${SOURCE:?source tree}/3rdparty/mki/include" -I"${SOURCE:?source tree}/core/fft/fftcore/fftoperation/include" \
    -I"${SOURCE:?source tree}/core/fft/utils/include" -I"${SOURCE:?source tree}/core/utils/include" -I"${ASCEND_HOME_PATH:?CANN root}/include" \
    -L"${INSTALL:?install prefix}/lib" -L"${ASCEND_HOME_PATH:?CANN root}/lib64" \
    -Wl,-rpath-link,"${INSTALL:?install prefix}/lib" -Wl,-rpath-link,"${ASCEND_HOME_PATH:?CANN root}/lib64" \
    -Wl,--start-group "${BUILD:?build directory}/core/fft/fftcore/fftoperation/libfft_operation.a" \
    "${BUILD:?build directory}/core/fft/utils/libfft_utils.a" "${BUILD:?build directory}/core/base/libbase.a" "${BUILD:?build directory}/core/utils/libutils.a" \
    -Wl,--end-group -lasdsip_core -lmki -lascendcl -lnnopbase -lopapi -lc_sec -ldl -o "${DEST:?binary directory}/transpose_bench"
