/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef LOG_CORE_H
#define LOG_CORE_H
#include <memory>
#include <mutex>
#include <vector>
#include "log/log_entity.h"
#include "log/log_sink_sip.h"

namespace AsdSip {
class LogCoreSip {
public:
    LogCoreSip();
    virtual ~LogCoreSip() = default;
    static LogCoreSip& InstanceSip();
    LogLevel GetLogLevel() const;
    void SetLogLevel(LogLevel level);
    void LogSip(const char* log, uint64_t logLen);
    void AddSink(std::shared_ptr<LogSinkSip> sink);
    const std::vector<std::shared_ptr<LogSinkSip>>& GetAllSinks() const;

private:
    std::vector<std::shared_ptr<LogSinkSip>> sinks_;
    std::mutex sinkMutex_; // 保护 sinks_ 的遍历快照与增删（issue #297）
    LogLevel level_ = LogLevel::INFO;
};
} // namespace AsdSip
#endif
