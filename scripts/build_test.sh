#!/bin/bash
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#

SCRIPT_DIR=$(cd $(dirname $0) && pwd)
CODE_ROOT=$(cd $SCRIPT_DIR/.. && pwd)
OUTPUT_DIR=$CODE_ROOT/output
EXAMPLE_DIR=$CODE_ROOT/example
EX_TMP_DIR=""
TOTAL_EXAMPLES=0
FAIL_COUNT=0
FAILED_FILES=()
VALID_TEST_DIRS=()
BUILD_OPTION_LIST="default example_test"

log_info() { echo "[INFO ] $*"; }
log_ok()   { echo "[ OK  ] $*"; }
log_warn() { echo "[WARN ] $*"; }
log_fail() { echo "[FAIL ] $*"; }
log_line()    { echo "================================================================================"; }
log_subline() { echo "--------------------------------------------------------------------------------"; }

function get_valid_pr_dir()
{
    log_info "Getting PR changed file list..."

    # CI（arm_pre_smoke）会把 PR 变更清单下载到仓库根目录 pr_filelist.txt。选例只以该
    # 清单为准：旧实现对比 origin/<当前分支> 与工作区，开发者已提交并推送 PR 分支、
    # 工作区干净时两者指向同一提交，diff 恒为空，即使 PR 相对目标分支改动了算子也会
    # 被误判为无改动而静默漏测（issue #334）。
    # 清单不存在/不可读/为空时回退全量测试，避免静默跳过；仅 Markdown 变更保留跳过行为。
    local list_file="$CODE_ROOT/pr_filelist.txt"
    local file_list=""

    if [ ! -e "$list_file" ]; then
        log_warn "PR file list not found: $list_file, fall back to full smoke test"
        VALID_TEST_DIRS=("base" "blas" "fft" "domain" "filter" "interpolation")
        return
    fi

    if [ ! -r "$list_file" ]; then
        log_warn "PR file list not readable: $list_file, fall back to full smoke test"
        VALID_TEST_DIRS=("base" "blas" "fft" "domain" "filter" "interpolation")
        return
    fi

    # 剥离 CI 产出清单可能携带的行尾 CR 后逐行解析，空行跳过
    file_list=$(tr -d '\r' < "$list_file")
    if [ -z "$(printf '%s' "$file_list" | tr -d '[:space:]')" ]; then
        log_warn "PR file list is empty: $list_file, fall back to full smoke test"
        VALID_TEST_DIRS=("base" "blas" "fft" "domain" "filter" "interpolation")
        return
    fi

    log_info "Using PR file list: $list_file"
    echo "$file_list"

    VALID_TEST_DIRS=()

    while IFS= read -r f; do
        [ -z "$f" ] && continue

        case "${f,,}" in
            *.md)
                continue
                ;;
            *interpbycoeff*|*interp_with_coeff*)
                VALID_TEST_DIRS+=("interpolation")
                ;;
            *interpolation*)
                VALID_TEST_DIRS+=("domain")
                ;;
            *blas*)
                VALID_TEST_DIRS+=("blas")
                ;;
            *fft*)
                VALID_TEST_DIRS+=("fft")
                ;;
            *base*)
                VALID_TEST_DIRS+=("base")
                ;;
            *filter*)
                VALID_TEST_DIRS+=("filter")
                ;;
            *)
                log_info "Unknown file type: $f"
                VALID_TEST_DIRS=("base" "blas" "fft" "domain" "filter" "interpolation")
                return
                ;;
        esac
    done <<< "$file_list"

    if [ ${#VALID_TEST_DIRS[@]} -eq 0 ]; then
        log_info "Only documentation changed, skipping tests"
        VALID_TEST_DIRS=()
    else
        VALID_TEST_DIRS=($(printf "%s\n" "${VALID_TEST_DIRS[@]}" | sort -u))
        log_info "Test directories: ${VALID_TEST_DIRS[*]}"
    fi
}

function fn_example_prepare()
{
    [ -d "$EXAMPLE_DIR" ] || { log_fail "Example dir not found: $EXAMPLE_DIR"; return 1; }

    source "$OUTPUT_DIR/set_env.sh"

    [ -n "$ASDSIP_HOME_PATH" ] || { log_fail "ASDSIP_HOME_PATH is not set"; exit 1; }

    if [ ! -d "$ASDSIP_HOME_PATH" ]; then
        new_path="${ASDSIP_HOME_PATH%latest/}"
        [ -d "$new_path" ] || { log_fail "ASDSIP_HOME_PATH not exist: $ASDSIP_HOME_PATH"; exit 1; }
        export ASDSIP_HOME_PATH="$new_path"
    fi

    export LD_LIBRARY_PATH="$ASDSIP_HOME_PATH/lib:$LD_LIBRARY_PATH"

    EX_TMP_DIR="$EXAMPLE_DIR/tmp_example_bin_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$EX_TMP_DIR" || return 1

    log_info "Temp dir: $EX_TMP_DIR"
    trap 'fn_example_cleanup' EXIT
}

function fn_example_test_one()
{
    cpp=$1
    file=$(basename ${cpp%.cpp})
    exe=$file

    TOTAL_EXAMPLES=$((TOTAL_EXAMPLES + 1))

    log_line
    echo "[TEST $TOTAL_EXAMPLES] $cpp"

    g++ $cpp \
        -I$ASCEND_HOME_PATH/include/aclnn \
        -I$ASCEND_HOME_PATH/include \
        -L$ASCEND_HOME_PATH/lib64 -lascendcl -lopapi -lnnopbase \
        -I$ASDSIP_HOME_PATH/include \
        -L$ASDSIP_HOME_PATH/lib -lmki \
        -lasdsip -lasdsip_core -lasdsip_host \
        -o $EX_TMP_DIR/$exe

    if [ $? -ne 0 ]; then
        log_fail "Compile failed"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        FAILED_FILES+=($cpp)
        return
    fi

    log_ok "Compile success"
    echo "[RUN ] $EX_TMP_DIR/$exe"

    $EX_TMP_DIR/$exe
    ret=$?

    if [ $ret -ne 0 ]; then
        log_fail "Run failed (exit code: $ret)"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        FAILED_FILES+=($cpp)
    else
        log_ok "Pass"
    fi
}


function fn_example_cleanup()
{
    [ -d "$EX_TMP_DIR" ] || return
    log_info "Cleaning example tmp dir: $EX_TMP_DIR"
    rm -rf "$EX_TMP_DIR"
}


function fn_example_test()
{
    fn_example_prepare || return

    log_line
    echo "                            EXAMPLE TEST"
    log_line
    echo "Code root       : $CODE_ROOT"
    echo "Search directory: ${EXAMPLE_DIR#$CODE_ROOT/}"
    echo "Temp directory  : ${EX_TMP_DIR#$CODE_ROOT/}"
    log_subline

    # Find example files
    example_files=()
    for cat in "${VALID_TEST_DIRS[@]}"; do
        # Find matching directory (case insensitive)
        found_dir=""
        for dir in "$EXAMPLE_DIR/A2"/*; do
            [ -d "$dir" ] || continue
            dirname="${dir##*/}"
            [ "${dirname,,}" = "${cat,,}" ] && {
                found_dir="$dir"
                break
            }
        done

        [ -z "$found_dir" ] && continue

        # Find all example_*.cpp files in directory
        while IFS= read -r -d '' file; do
            example_files+=("$file")
        done < <(find "$found_dir" -type f -name "example_*.cpp" -print0 2>/dev/null)
    done

    [ ${#example_files[@]} -eq 0 ] && {
        log_fail "No example_*.cpp files found in: ${VALID_TEST_DIRS[*]}"
        return
    }

    # Print summary
    printf "Found %d example file(s) in:\n" ${#example_files[@]}
    printf "  - %s\n" "${VALID_TEST_DIRS[@]}"
    log_subline
    echo "Files found:"
    for i in "${!example_files[@]}"; do
        rel_path=${example_files[$i]#$EXAMPLE_DIR/}
        printf "  %3d. %s\n" $((i+1)) "$rel_path"
    done

    # Run tests
    for cpp in "${example_files[@]}"; do
        fn_example_test_one "$cpp"
    done

    # Print test summary
    log_line
    echo "                            TEST SUMMARY"
    log_line
    passed=$((TOTAL_EXAMPLES - FAIL_COUNT))
    echo "Total tests     : $TOTAL_EXAMPLES"
    echo "Passed          : $passed"
    echo "Failed          : $FAIL_COUNT"

    [ $FAIL_COUNT -ne 0 ] && {
        log_subline
        log_fail "Failed cases: ${#FAILED_FILES[@]} file(s)"
        for i in "${!FAILED_FILES[@]}"; do
            rel_path=${FAILED_FILES[$i]#$EXAMPLE_DIR/}
            printf "  %3d. %s\n" $((i+1)) "$rel_path"
        done
    }
    log_line

    # 任一示例编译/运行失败时以非零状态退出，供 smoke_pr 等 CI 目标感知失败（issue #224）
    if [ $FAIL_COUNT -ne 0 ]; then
        return 1
    fi
    return 0
}

function fn_main()
{
    if [[ " ${BUILD_OPTION_LIST} " == *" $1 "* ]]; then
        arg1=$1
        shift
    else
        arg1="default"
    fi

    case "${arg1}" in
    "default")
        VALID_TEST_DIRS=("base" "blas" "fft" "domain" "filter" "interpolation")
        fn_example_test
        ;;
    "example_test")
        get_valid_pr_dir
        fn_example_test
        ;;
    *)
        echo "Unknown build option: ${arg1}"
        exit 1
        ;;
esac
}

fn_main "$@"
