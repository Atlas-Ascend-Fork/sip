# Copyright (c) 2026 gcw_2GoAgJFe. All rights reserved.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
add_operation(KaiserWindowOperation ${CMAKE_CURRENT_LIST_DIR}/kaiser_window_operation.cpp)
if(BUILD_KaiserWindowOperation)
    target_sources(KaiserWindowOperation PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/kaiser_window/kaiser_window_kernel.cpp
        ${CMAKE_CURRENT_LIST_DIR}/kaiser_window/tiling/kaiser_window_tiling.cpp
    )
endif()
add_kernel(kaiser_window ascend910b vector
    kaiser_window/op_kernel/kaiser_window.cpp KaiserWindowKernel
    DEPENDS ${CMAKE_CURRENT_LIST_DIR}/kaiser_window/op_kernel/kaiser_window.h
        ${CMAKE_CURRENT_LIST_DIR}/kaiser_window/op_kernel/kernel_common.h
        ${CMAKE_CURRENT_LIST_DIR}/kaiser_window/tiling/tiling_data.h)
add_kernel(kaiser_window_arch35 ascend950 vector
    kaiser_window/op_kernel/arch35/kaiser_window_arch35.cpp KaiserWindowArch35Kernel
    DEPENDS ${CMAKE_CURRENT_LIST_DIR}/kaiser_window/op_kernel/arch35/kaiser_window_arch35.h
        ${CMAKE_CURRENT_LIST_DIR}/kaiser_window/op_kernel/kernel_common.h
        ${CMAKE_CURRENT_LIST_DIR}/kaiser_window/tiling/tiling_data.h)
