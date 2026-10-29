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
HERE=$(realpath "$(dirname "$0")")
mkdir -p "$DEST"
g++ -std=c++17 -O2 "$HERE/test_plan.cpp" -I"$SOURCE" \
    -I"$SOURCE/ops/fft/c2c/fft_regbase/fft_regbase/large" -o "$DEST/test_plan"
"$DEST/test_plan"
python3 "$HERE/test_layout.py"
g++ -std=c++17 -O2 -D_GLIBCXX_USE_CXX11_ABI=0 "$HERE/bench.cpp" \
    -I"$INSTALL/include" -I"$SOURCE/core/utils/include" -I"$ASCEND_HOME_PATH/include" \
    -L"$INSTALL/lib" -L"$ASCEND_HOME_PATH/lib64" \
    -Wl,-rpath-link,"$INSTALL/lib" -Wl,-rpath-link,"$ASCEND_HOME_PATH/lib64" \
    -lasdsip -lascendcl -lnnopbase -o "$DEST/fft_bench"
