#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set -eo pipefail
SOURCE=$(realpath "${1:?source tree}")
INSTALL=$(realpath "${2:?SIP install prefix}")
DEST=${3:?binary directory}
: "${ASCEND_HOME_PATH:?source the CANN environment first}"
mkdir -p "${DEST:?binary directory}"

g++ -std=c++17 -O2 "${SOURCE:?source tree}/tests/standalone/fft_mixed/test_plan.cpp" \
    -I"${SOURCE:?source tree}/ops/include" -I"${SOURCE:?source tree}/core/fft" -o "${DEST:?binary directory}/test_plan"
"${DEST:?binary directory}/test_plan"
# Reuse the repository's public tensor-API benchmark, including alias checks.
g++ -std=c++17 -O2 -D_GLIBCXX_USE_CXX11_ABI=0 "${SOURCE:?source tree}/tests/standalone/fft1d_optimized/bench.cpp" \
    -I"${INSTALL:?install prefix}/include" -I"${SOURCE:?source tree}/core/utils/include" -I"${ASCEND_HOME_PATH:?CANN root}/include" \
    -L"${INSTALL:?install prefix}/lib" -L"${ASCEND_HOME_PATH:?CANN root}/lib64" \
    -Wl,-rpath-link,"${INSTALL:?install prefix}/lib" -Wl,-rpath-link,"${ASCEND_HOME_PATH:?CANN root}/lib64" \
    -lasdsip -lascendcl -lnnopbase -o "${DEST:?binary directory}/fft_bench"
