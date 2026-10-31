/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdio>
#include <cstdlib>

#include "gtest/gtest.h"
#include "acl/acl.h"
#include "mki/utils/rt/rt.h"
// #include "acl_meta.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

#if GTEST_OS_ESP8266 || GTEST_OS_ESP32

#if GTEST_OS_ESP8266
extern "C" {
#endif

void setup() { testing::InitGoogleTest(); }

void loop() { RUN_ALL_TESTS(); }

#if GTEST_OS_ESP8266
}
#endif

#elif GTEST_OS_QURT

GTEST_API_ int main()
{
    printf("Running main() from %s\n", __FILE__);
    testing::InitGoogleTest();
    return RUN_ALL_TESTS();
}
#else

int Init()
{
    // 固定写法，acl初始化
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    // ret = aclrtSetDevice(deviceId);
    // CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    // ret = aclrtCreateStream(stream);
    // CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

// UT 用例间设备保持 (ASDSIP_UT_KEEP_DEVICE=1, 见 test_util/util.cpp): 用例结束不复位设备,
// 全部用例结束后在此统一复位一次, 归还设备资源 (规避 atexit, G.FUU.04)。
class UtKeepDeviceEnvironment : public testing::Environment {
public:
    void TearDown() override
    {
        const char* keep = std::getenv("ASDSIP_UT_KEEP_DEVICE");
        if (keep == nullptr || keep[0] != '1') {
            return;
        }
        const char* envDeviceId = std::getenv("ASDSIP_DEVICE_ID");
        int devId = (envDeviceId != nullptr && envDeviceId[0] != '\0') ? std::atoi(envDeviceId) : 0;
        printf("[Custom] reset kept device %d\n", devId);
        Mki::MkiRtDeviceResetCurrent(devId);
        aclrtResetDevice(devId);
    }
};

GTEST_API_ int main(int argc, char** argv)
{
    printf("[Custom] Running main() from %s\n", __FILE__);

    int deviceId = 0;
    // aclrtStream stream;
    auto ret = Init();
    // CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    printf("[Custom] init acl device finished\n");

    testing::InitGoogleTest(&argc, argv);
    testing::AddGlobalTestEnvironment(new UtKeepDeviceEnvironment);
    auto runRes = RUN_ALL_TESTS();
    ret = aclFinalize();
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("aclFinalize failed. ERROR: %d\n", ret); return ret);
    return runRes;
}
#endif
