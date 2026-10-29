# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

add_subdirectory(conj)
add_subdirectory(swaplast2axes)
add_subdirectory(mul)
add_subdirectory(solve)

set(operations_objects ${ops_objects} PARENT_SCOPE)

set_source_files_properties(${BINARY_SRC_LIST} PROPERTIES GENERATED TRUE)

add_custom_command(
    OUTPUT ${BINARY_SRC_LIST}
    DEPENDS ${BINARY_TARGET_LIST}
    COMMAND python3 ${MKI_PACKAGE_DIR}/scripts/build_util.py --binary_dir ${CMAKE_BINARY_DIR} --op_type "ascendc"
)

add_custom_target(BASE_BINARY_SRC_TARGET ALL
    DEPENDS ${BINARY_SRC_LIST}
)

add_library(base_obj OBJECT ${BINARY_SRC_LIST})
add_dependencies(base_obj BASE_BINARY_SRC_TARGET)
