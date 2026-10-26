# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

add_subdirectory(common_tiling)
add_subdirectory(asum)
add_subdirectory(colwise_mul)
add_subdirectory(complex_mat_dot)
add_subdirectory(ssyr2)
add_subdirectory(caxpy)
add_subdirectory(cgemm)
add_subdirectory(interpolation)
add_subdirectory(interpbycoeff)
add_subdirectory(dot)
add_subdirectory(swap)
add_subdirectory(cal)
add_subdirectory(strmm)
add_subdirectory(strmv)
add_subdirectory(copy)
add_subdirectory(ctrmv)
add_subdirectory(iamax)
add_subdirectory(cgerc)
add_subdirectory(nrm2)
add_subdirectory(ssyr)
add_subdirectory(rot)
add_subdirectory(cgemv)
add_subdirectory(cgemv_batched)
add_subdirectory(hcgemm_batched)
add_subdirectory(cgemm_batched)
add_subdirectory(cmatinv_batched)
add_subdirectory(cgetri_batched)
add_subdirectory(sgetri_batched)
add_subdirectory(smatinv_batched)

message("blas ops_objects: ${ops_objects}")
set(operations_objects ${operations_objects} ${ops_objects} ops_common_tiling PARENT_SCOPE)

set_source_files_properties(${BINARY_SRC_LIST} PROPERTIES GENERATED TRUE)
message("BINARY_SRC_LIST: ${BINARY_SRC_LIST}")
message("BINARY_TARGET_LIST: ${BINARY_TARGET_LIST}")

add_custom_command(
    OUTPUT ${BINARY_SRC_LIST}
    DEPENDS ${BINARY_TARGET_LIST}
    COMMAND python3 ${MKI_PACKAGE_DIR}/scripts/build_util.py --binary_dir ${CMAKE_BINARY_DIR} --op_type "ascendc"
)

add_custom_target(BLAS_BINARY_SRC_TARGET ALL
    DEPENDS ${BINARY_SRC_LIST}
)

add_library(blas_obj OBJECT ${BINARY_SRC_LIST})
add_dependencies(blas_obj BLAS_BINARY_SRC_TARGET)
