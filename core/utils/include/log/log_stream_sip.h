/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef LOG_STREAM_H
#define LOG_STREAM_H
#include <sstream>
#include "log/log_entity.h"
namespace AsdSip {
class LogStreamSip {
public:
    LogStreamSip(const char* filePath, int line, const char* funcName, LogLevel level,
                 ErrorType errCode = AsdSip::ErrorType::ACL_SUCCESS);
    ~LogStreamSip();
    template <typename T>
    LogStreamSip& operator<<(const T& value)
    {
        stream_ << value;
        return *this;
    }
    void FormatSip(const char* format, ...);

private:
    LogEntity logEntity_;
    bool useStream_ = true;
    std::ostringstream& stream_; // errCode_ 未初始化且全仓零引用，真实错误码存于 logEntity_.errCode，删除（issue #298）
};
} // namespace AsdSip
#endif
