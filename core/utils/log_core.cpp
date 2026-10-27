/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdlib>
#include <string>
#include <cstring>
#include <unordered_map>
#include <iostream>
#include <algorithm>
#include <mutex>
#include <vector>
#include "utils/env.h"
#include "log/log_sink_stdout_sip.h"
#include "log/log_sink_file_sip.h"
#include "log/log.h"
#include "log/log_core_sip.h"

namespace AsdSip {
static bool GetLogToStdoutFromEnv()
{
    const char* envLogToStdout = std::getenv("ASCEND_SLOG_PRINT_TO_STDOUT");
    return envLogToStdout != nullptr && strlen(envLogToStdout) <= MAX_ENV_STRING_LEN &&
           strcmp(envLogToStdout, "1") == 0;
}

static bool GetLogToFileFromEnv()
{
    const char* envLogToFile = std::getenv("ASCEND_PROCESS_LOG_PATH");
    return envLogToFile != nullptr && strlen(envLogToFile) <= MAX_ENV_STRING_LEN;
}

static LogLevel GetLogLevelFromEnv()
{
    /* 设置应用类日志的日志级别及各模块日志级别，仅支持调试日志。
    取值为：
        0：对应DEBUG级别。
        1：对应INFO级别。
        2：对应WARNING级别。
        3：对应ERROR级别，默认为ERROR级别。
        4：对应NULL级别，不输出日志。
        其他值为非法值。 */
    // LogLevel {DEBUG, INFO, WARN, ERROR } FATAL对应NULL级别，不输出日志。
    const char* env = std::getenv("ASCEND_GLOBAL_LOG_LEVEL");
    if (env == nullptr || strlen(env) > MAX_ENV_STRING_LEN) {
        return LogLevel::ERROR;
    }
    std::string envLogLevel(env);
    std::transform(envLogLevel.begin(), envLogLevel.end(), envLogLevel.begin(), ::toupper);
    static std::unordered_map<std::string, LogLevel> levelMap{{"0", LogLevel::DEBUG},
                                                              {"1", LogLevel::INFO},
                                                              {"2", LogLevel::WARN},
                                                              {"3", LogLevel::ERROR},
                                                              {"4", LogLevel::FATAL}};
    auto levelIt = levelMap.find(envLogLevel);
    // LogLevel::FATAL对应NULL级别，不输出日志。
    return levelIt != levelMap.end() ? levelIt->second : LogLevel::ERROR;
}

static std::vector<std::string> splitString(const std::string& s, char delimiter)
{
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (std::getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

static LogLevel GetModuleLogLevelFromEnv(bool& isSet)
{
    /* 设置应用类日志的日志级别及各模块日志级别，仅支持调试日志。
    格式为：export ASCEND_MODULE_LOG_LEVEL=module_name=module_level
    module_name：为模块名，支持设置为GE、ASCENDCL、DRV、RUNTIME等，详细请参见CANN软件安装目录/include/base/log_types.h。
    module_level：为对应模块的日志级别，支持设置为如下值。
        0：对应DEBUG级别。
        1：对应INFO级别。
        2：对应WARNING级别。
        3：对应ERROR级别，默认为ERROR级别。
        4：对应NULL级别，不输出日志。
        其他值为非法值。 */
    const char* env = std::getenv("ASCEND_MODULE_LOG_LEVEL");
    if (env == nullptr || strlen(env) > MAX_ENV_STRING_LEN) {
        isSet = false;
        return LogLevel::ERROR;
    }
    std::string envs(env);
    std::vector<std::string> envLogLevels = splitString(envs, ':');
    std::string envLogLevel = "-1";

    for (const auto& i : envLogLevels) {
        std::vector<std::string> key_value = splitString(i, '=');
        if (key_value.size() >= 1) {
            if (key_value[0] == "OP") {
                if (key_value.size() == 2) { // 2: correct key-value pair
                    isSet = true;
                    envLogLevel = key_value[1];
                }
                break;
            }
        }
    }

    if (!isSet) {
        return LogLevel::ERROR;
    }

    static std::unordered_map<std::string, LogLevel> levelMap{{"0", LogLevel::DEBUG},
                                                              {"1", LogLevel::INFO},
                                                              {"2", LogLevel::WARN},
                                                              {"3", LogLevel::ERROR},
                                                              {"4", LogLevel::FATAL}};
    auto levelIt = levelMap.find(envLogLevel);
    // LogLevel::FATAL对应NULL级别，不输出日志。
    return levelIt != levelMap.end() ? levelIt->second : LogLevel::ERROR;
}

LogCoreSip::LogCoreSip()
{
    bool isModuleSet = false;
    auto moduleLevel_ = GetModuleLogLevelFromEnv(isModuleSet);
    level_ = isModuleSet ? moduleLevel_ : GetLogLevelFromEnv();
    if (strcmp(LogLevelToString(level_).c_str(), "FATAL") == 0) {
        return;
    }

    if (GetLogToStdoutFromEnv()) { // print but not to file
        AddSink(std::make_shared<LogSinkStdoutSip>());
    } else if (GetLogToFileFromEnv()) { // not print but to file（else-if 分支中 stdout 条件恒真，为死条件，issue #298）
        AddSink(std::make_shared<AsdSip::LogSinkFileSip>());
    }
}

LogCoreSip& LogCoreSip::InstanceSip()
{
    static LogCoreSip logCore;
    return logCore;
}

LogLevel LogCoreSip::GetLogLevel() const { return level_; }

void LogCoreSip::SetLogLevel(LogLevel level) { level_ = level; }

void LogCoreSip::LogSip(const char* log, uint64_t logLen)
{
    // 锁内拷贝 shared_ptr 快照后遍历，避免持锁进入 sink 回调引入锁序问题；
    // 同时保证与 AddSink 的并发增删不产生迭代器失效（issue #297）
    std::vector<std::shared_ptr<LogSinkSip>> snapshot;
    {
        std::lock_guard<std::mutex> lock(sinkMutex_);
        snapshot = sinks_;
    }
    for (auto& sink : snapshot) {
        sink->LogSip(log, logLen);
    }
}

void LogCoreSip::AddSink(std::shared_ptr<LogSinkSip> sink)
{
    if (sink == nullptr) {
        ASDSIP_LOG(ERROR) << "sink is NULL";
        return;
    }
    std::lock_guard<std::mutex> lock(sinkMutex_);
    sinks_.push_back(sink);
}

const std::vector<std::shared_ptr<LogSinkSip>>& LogCoreSip::GetAllSinks() const { return sinks_; }
} // namespace AsdSip
