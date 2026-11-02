#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

path="${BASH_SOURCE[0]}"
if [[ -f "$path" ]] && [[ "$path" =~ 'set_env.sh' ]];then
    asdsip_dir_path=$(cd "$(dirname "$path")"; pwd )
    export ASDSIP_HOME_PATH="${asdsip_dir_path}/latest/"
    export LD_LIBRARY_PATH="${ASDSIP_HOME_PATH}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
else
    echo "There is no 'set_env.sh' to import"
fi
