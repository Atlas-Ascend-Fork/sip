#!/bin/bash

# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# 统计变量
TOTAL=0
PASSED=0
FAILED=0
LOG_DIR="./test_logs"

mkdir -p "$LOG_DIR"

echo "============================================================"
echo "顺序执行所有测试脚本并实时回显日志"
echo "============================================================"

# 查找所有包含 main 入口的 Python 脚本
while IFS= read -r -d '' FILE; do
    if grep -q "if __name__ == .__main__.:" "$FILE"; then
        TOTAL=$((TOTAL + 1))

        TEMP_LOG="$LOG_DIR/$(basename "$FILE").log"

        echo "------------------------------------------------------------"
        echo -e "🚀 [RUNNING] \e[36m$FILE\e[0m"

        # 执行并将所有输出实时存入日志
        python3 "$FILE" > "$TEMP_LOG" 2>&1
        STATUS=$?

        # --- 直接打印日志内容 ---
        cat "$TEMP_LOG"

        if [ $STATUS -eq 0 ]; then
            echo -e "\e[32m结果: [PASS]\e[0m"
            PASSED=$((PASSED + 1))
        else
            echo -e "\e[31m结果: [FAIL (Status: $STATUS)]\e[0m"
            FAILED=$((FAILED + 1))
            FAILED_LIST+="\n  - $FILE"
        fi

        # 清理
        rm -f "$TEMP_LOG"
        echo "------------------------------------------------------------"
        echo ""
    fi
done < <(find . -name '*.py' -not -path '*/.*' -not -path '*__pycache__*' -print0)

# 最终汇总
echo "============================================================"
echo "测试结束汇总:"
echo "  总运行数: $TOTAL"
echo -e "  成功通过: \e[32m$PASSED\e[0m"
echo -e "  失败项数: \e[31m$FAILED\e[0m"

if [ $TOTAL -eq 0 ]; then
    echo -e "\e[31m未发现可执行的测试脚本\e[0m"
    rm -rf "$LOG_DIR"
    exit 1
elif [ $FAILED -gt 0 ]; then
    echo -e "\e[31m具体失败脚本: $FAILED_LIST\e[0m"
    rm -rf "$LOG_DIR"
    exit 1
else
    echo -e "\e[32m🎉 所有测试全部通过！\e[0m"
    rm -rf "$LOG_DIR"
    exit 0
fi
