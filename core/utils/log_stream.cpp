/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <securec.h>
#include <thread>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cstring>
#include <ctime>
#include <cstdarg>
#include <unistd.h>
#include <syscall.h>
#include "aclnn/acl_meta.h"
#include "log/log_core_sip.h"
#include "log/log_stream_sip.h"

namespace AsdSip {
thread_local std::ostringstream g_stream;

thread_local long g_threadId = -1;
long GetThreadId()
{
    if (g_threadId == -1) {
        g_threadId = syscall(SYS_gettid);
    }
    return g_threadId;
}

LogStreamSip::LogStreamSip(const char* filePath, int line, const char* funcName, LogLevel level, ErrorType errCode)
    : stream_(g_stream)
{
    if (filePath != nullptr && funcName != nullptr) {
        const char* str = strrchr(filePath, '/');
        logEntity_.fileName = str ? str + 1 : filePath;
        logEntity_.time = std::chrono::system_clock::now();
        logEntity_.level = level;
        logEntity_.errCode = errCode;
        logEntity_.threadId = GetThreadId();
        logEntity_.funcName = funcName;
        logEntity_.line = line;
    }

    constexpr int microSecond = 1000000;
    int us = std::chrono::duration_cast<std::chrono::microseconds>(logEntity_.time.time_since_epoch()).count() %
             microSecond;
    std::time_t tmpTime = std::chrono::system_clock::to_time_t(logEntity_.time);
    struct tm timeinfo;
    localtime_r(&tmpTime, &timeinfo);
    char timeStr[64] = {0}; // 64 日志buffer长度
    (void)strftime(timeStr, sizeof(timeStr), "%Y-%m-%d %H:%M:%S", &timeinfo);
    std::string fileName = (logEntity_.fileName != nullptr) ? logEntity_.fileName : "EmptyFile";
    stream_.str("");
    stream_ << "[" << timeStr << "." << us << "] [asdsip] [" << LogLevelToString(logEntity_.level) << "] ["
            << logEntity_.threadId << "] [" << fileName << ":" << logEntity_.line << "] ";
}

void LogStreamSip::FormatSip(const char* format, ...)
{
    useStream_ = false;
    std::string logHead = stream_.str();
    const size_t logContentSize = 1024; // 1024 日志内容最大支持1024
    // VLA 改为 vector 缓冲：logHead.size() 运行期确定，VLA 属 GCC 扩展、非标准 C++（issue #298）
    std::vector<char> logBuffer(logHead.size() + logContentSize + 1, 0);

    int ret = memcpy_s(logBuffer.data(), logBuffer.size(), logHead.c_str(), logHead.size());
    if (ret != 0) {
        std::cout << "asdsip_log memcpy_s fail, error:" << ret << std::endl;
        return;
    }

    va_list args;
    va_start(args, format);
    int logLen = vsnprintf_s(logBuffer.data() + logHead.size(), logContentSize + 1, logContentSize, format, args);
    va_end(args);
    if (logLen < 0) {
        std::cout << "asdsip_log vsnprintf_s fail" << std::endl;
        return;
    }

    logBuffer[logHead.size() + logLen] = '\n';
    LogCoreSip::InstanceSip().LogSip(logBuffer.data(), logHead.size() + logLen + 1);
}

LogStreamSip::~LogStreamSip()
{
    // 如果有错误码，在日志末尾添加
    if (logEntity_.level == LogLevel::ERROR && logEntity_.errCode != ErrorType::ACL_SUCCESS) {
        stream_ << " [Error Code: " << LogErrCodeToInt(logEntity_.errCode) << "]";
    }

    if (useStream_) {
        stream_ << "\n";
        std::string log = stream_.str();
        LogCoreSip::InstanceSip().LogSip(log.c_str(), log.size());
    }
}
} // namespace AsdSip
