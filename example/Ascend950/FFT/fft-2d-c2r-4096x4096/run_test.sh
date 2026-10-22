#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
set -e
DEVICE_ID="${1:-0}"
SKIP_BUILD="${2:-0}"
FFT_N=4096
BATCH=1
STAGE=2
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DATA_DIR="$SCRIPT_DIR/data"
EXE_NAME=fft_2d_c2r_4096x4096

echo "=== FFT C2R 2D [4096,4096] Precision Test (device=${DEVICE_ID}) ==="
mkdir -p "$DATA_DIR"
# 清理上次运行残留的输出（文件名不含 stage，切换 stage 时同样需清理），
# 保证首次 gen_data.py 仅生成数据；精度比对在设备程序执行后的第二次
# gen_data.py 中进行（失败经退出码传播）
rm -f "$DATA_DIR/output_${FFT_N}_${BATCH}.bin"
python3 "$SCRIPT_DIR/gen_data.py" ${FFT_N} ${BATCH} --stage ${STAGE}
if [ "$SKIP_BUILD" != "1" ]; then
    bash "$SCRIPT_DIR/build.sh"
fi
cd "$SCRIPT_DIR/build"
./${EXE_NAME} ${FFT_N} ${BATCH} ${DEVICE_ID} "$DATA_DIR" --stage ${STAGE}
cd "$SCRIPT_DIR"
python3 "$SCRIPT_DIR/gen_data.py" ${FFT_N} ${BATCH} --stage ${STAGE}
