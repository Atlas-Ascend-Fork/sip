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
path="${BASH_SOURCE[0]}"
# 守卫匹配自身文件名（issue #269）：原 'set_env.sh' 与本文件 set_sample_env.sh
# 无连续子串关系，条件恒假，source 后全部环境变量不导出且报误导性错误文案
if [[ -f "$path" ]] && [[ "$path" =~ 'set_sample_env.sh' ]];then
    asdsip_dir_path=$(cd "$(dirname "$path")"; pwd )
    export ASDSIP_HOME_PATH="${asdsip_dir_path}"
    export LD_LIBRARY_PATH=$ASDSIP_HOME_PATH/lib:$LD_LIBRARY_PATH
    export PATH=$ASDSIP_HOME_PATH/bin:$PATH
    export ASDOPS_LOG_TO_STDOUT=0
    export ASDOPS_LOG_LEVEL=INFO
    export ASDOPS_LOG_TO_FILE=0
    export ASDOPS_LOG_TO_FILE_FLUSH=0
    export ASDOPS_LOG_TO_BOOST_TYPE=asdsip #算子库对应加速库日志类型，默认asdsip
    export ASDOPS_LOG_PATH=~
else
    echo "There is no 'set_env.sh' to import"
fi
