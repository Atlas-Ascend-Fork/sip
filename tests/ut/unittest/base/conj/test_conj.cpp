/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <gtest/gtest.h>
#include <filesystem>
#include <cmath>
#include "test_common.h"
#include "mki/utils/status/status.h"
#include "mki/utils/log/log.h"
#include "test_util/float_util.h"
#include "test_util/util.cpp"
#include "ops.h"
#include "base_api.h"
#include "base_inner_api.h"
#include "blas_api.h"
#include "utils/mem_base_inner.h"
using namespace AsdSip;
using namespace Mki;

namespace {
constexpr float ATOL = 0.001;
constexpr float RTOL = 0.001;
constexpr float EXTENT_OF_ERROR = 0.001;

std::string GetConjOutputDirectory()
{
    const char* current_dir_env = std::getenv("CURRENT_DIR");
    if (current_dir_env) {
        // 直接返回一个确定的字符串，不涉及任何路径操作
        return std::string(current_dir_env);
    } else {
        return std::string("get env failed.");
    }
}
} // namespace

TEST(TestOpConj, TestConjCase0)
{
    // 获取加速库所在路径 + 算子文件拼接
    std::string originPath = GetConjOutputDirectory() + "/tests/ut/unittest/base/conj/conj_data";
    std::string destPath = GetConjOutputDirectory() + "/build/tests/ut/unittest/base/conj";
    std::string copy_cmd = "cp -rf " + ShellQuote(originPath) + " " + ShellQuote(destPath);
    std::string chmod_cmd = "chmod -R 755 " + ShellQuote(destPath + "/conj_data/");
    int copy_rc = system(copy_cmd.c_str());
    ASSERT_EQ(copy_rc, 0) << "data prep command failed: " << copy_cmd;

    int chmod_rc = system(chmod_cmd.c_str());
    ASSERT_EQ(chmod_rc, 0) << "data prep command failed: " << chmod_cmd;

    Status status;
    int64_t m = 4;
    int64_t n = 6;
    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {m, n}});
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {m, n}});

    TensorContext context;

    int deviceId = 0;
    MkiRtStream stream = OpTestInit(deviceId);
    OpTestMallocInTensors(inTensorDescs, context);

    Tensor input = context.inTensors[0];
    Tensor output = context.inTensors[1];

    for (int i = 0; i < context.inTensors.size(); i++) {
        std::string filename = GetElementDtype(context.inTensors[i].desc.dtype) + "_input_" + std::to_string(i) +
                               ".bin";
        if (SaveTensorToBin(context.inTensors[i], destPath + "/conj_data/" + filename)) {
            std::cout << "Tensor saved successfully" << std::endl;
        } else {
            std::cerr << "Failed to save tensor" << std::endl;
        }
    }
    std::string gen_data_cmd = "cd " + ShellQuote(destPath + "/conj_data/") + " && python3 gen_data.py";
    int gen_data_rc = system(gen_data_cmd.c_str());
    ASSERT_EQ(gen_data_rc, 0) << "data prep command failed: " << gen_data_cmd;

    AspbStatus blasStatus = Conj(input, output, stream);
    if (blasStatus != AsdSip::ErrorType::ACL_SUCCESS) {
        status = Status::FailStatus(-1, "run ops failed!");
    }
    ASSERT_EQ(status.Ok(), true);

    MkiRtStreamSynchronize(stream);
    CopyDeviceTensorToHostTensor(output);

    std::string filename = GetElementDtype(output.desc.dtype) + "_output_" + ".bin";
    if (SaveOutTensorToBin(output, destPath + "/conj_data/" + filename)) {
        std::cout << "Tensor saved successfully" << std::endl;
    } else {
        std::cerr << "Failed to save tensor" << std::endl;
    }

    OpTestEnd(deviceId, context, stream);

    std::string cmp_data_cmd = "cd " + ShellQuote(destPath + "/conj_data/") + " && python3 compare_data.py";

    int res = system(cmp_data_cmd.c_str());
    std::cout << "res = " << res << std::endl;
    ASSERT_EQ(res, 0);
}
