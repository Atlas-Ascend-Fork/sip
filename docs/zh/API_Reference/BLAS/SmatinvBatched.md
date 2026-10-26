# SmatinvBatched

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：不支持
<!-- end id1 -->

<!-- npu="A3" id2 -->
- <term>Atlas A3系列产品</term>：支持
<!-- end id2 -->

<!-- npu="910b" id3 -->
- <term>Atlas A2系列产品</term>：支持
<!-- end id3 -->

<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2推理产品</term>：不支持
<!-- end id4 -->

<!-- npu="310p" id5 -->
- <term>Atlas推理系列产品</term>：不支持
<!-- end id5 -->

<!-- npu="910" id6 -->
- <term>Atlas训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

- 接口功能：\
asdBlasMakeSmatinvBatchedPlan：初始化该句柄对应的算子配置\
asdBlasSmatinvBatched：对批量单精度实数矩阵进行求逆（fp32）。
- 计算公式：\
计算批量实数矩阵的逆矩阵，每个实数矩阵需满足条件$A^{-1}A=I$，其中A为非奇异方阵，且A是一个n*n的输入方阵，I为单位矩阵。

  示例：\
输入“A”为：\
[2, 1, 1, 1\
1, 2, 1, 1\
1, 1, 2, 1\
1, 1, 1, 2]\

  [3, 1, 1, 1\
1, 3, 1, 1\
1, 1, 3, 1\
1, 1, 1, 3]\
输入“n”为： 4\
输入“batchSize”为：2\
调用“asdBlasSmatinvBatched”算子后，\
输出“Ainv”为：\
[0.6, -0.2, -0.2, -0.2\
-0.2, 0.6, -0.2, -0.2\
-0.2, -0.2, 0.6, -0.2\
-0.2, -0.2, -0.2, 0.6]

   [0.375, -0.125, -0.125, -0.125\
-0.125, 0.375, -0.125, -0.125\
-0.125, -0.125, 0.375, -0.125\
-0.125, -0.125, -0.125, 0.375]

## 函数原型

```Cpp
AspbStatus asdBlasMakeSmatinvBatchedPlan(
  asdBlasHandle            handle,
  const int64_t            n,
  const int64_t            batchSize)
```

```Cpp
AspbStatus asdBlasSmatinvBatched(
  asdBlasHandle            handle,
  const int64_t            n,
  aclTensor *              A,
  const int64_t            lda,
  aclTensor *              Ainv,
  const int64_t            lda_inv,
  aclTensor *              info,
  int64_t                  batchSize)
```

## asdBlasMakeSmatinvBatchedPlan

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 880px"><colgroup>
    <col style="width: 250px">
    <col style="width: 120px">
    <col style="width: 510px">
  </colgroup>
  <thead>
      <tr>
        <th>参数名</th>
        <th>输入/输出</th>
        <th>描述</th>
      </tr></thead>
  <tbody>
    <tr>
      <td>handle（asdBlasHandle）</td>
      <td>输入</td>
      <td>算子的句柄</td>
    </tr>
    <tr>
      <td>n（int64_t）</td>
      <td>输入</td>
      <td>单批次矩阵A的行数。</td>
    </tr>
    <tr>
      <td>batchSize（int64_t）</td>
      <td>输入</td>
      <td>实数矩阵求逆中的矩阵数量。</td>
    </tr>
    </tbody>
    </table>

- **返回值**：

  返回状态码，具体参见[SiP返回码](../../context/SiP返回码.md)。

## asdBlasSmatinvBatched

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 880px"><colgroup>
    <col style="width: 250px">
    <col style="width: 120px">
    <col style="width: 510px">
  </colgroup>
  <thead>
      <tr>
        <th>参数名</th>
        <th>输入/输出</th>
        <th>描述</th>
      </tr></thead>
  <tbody>
    <tr>
      <td>handle（asdBlasHandle）</td>
      <td>输入</td>
      <td>算子的句柄</td>
    </tr>
    <tr>
      <td>n（int64_t）</td>
      <td>输入</td>
      <td>单批次矩阵A的行数。</td>
    </tr>
    <tr>
      <td>A（aclTensor *）</td>
      <td>输入</td>
      <td><ul><li>输入的矩阵，对应公式中的'A'。</li><li>行主序。</li><li>数据类型支持FLOAT。</li><li>数据格式支持ND。</li><li>shape为[batchSize, n, n]。</li></ul></td>
    </tr>
    <tr>
      <td>lda（ int64_t）</td>
      <td>输入</td>
      <td>A左右相邻元素间的内存地址偏移量（当前约束为n）。</td>
    </tr>
    <tr>
      <td>Ainv（aclTensor *）</td>
      <td>输出</td>
      <td><ul><li>输出的逆矩阵。</li><li>数据类型支持FLOAT。</li><li>数据格式支持ND。</li><li>shape为[batchSize, n, n]。</li></ul></td>
    </tr>
    <tr>
      <td>lda_inv（int64_t）</td>
      <td>输入</td>
      <td>输出的逆矩阵的左右相邻元素间的内存地址偏移量（当前约束为n）。</td>
    </tr>
    <tr>
      <td>info（aclTensor *）</td>
      <td>输入</td>
      <td><ul><li>每个batch矩阵的求逆结果信息。</li><li>数据类型支持int32_t。</li><li>数据格式支持ND。</li><li>shape为[batchSize]。</li></ul></td>
    </tr>
    <tr>
      <td>batchSize（int64_t）</td>
      <td>输入</td>
      <td>实数矩阵求逆中的矩阵数量。</td>
    </tr>
    </tbody>
    </table>

- **返回值**：

  返回状态码，具体参见[SiP返回码](../../context/SiP返回码.md)。

## 约束说明

- lda、lda_inv、info参数在当前版本实际未启用。
- 输入参数n小于等于256。
- 输入参数batchSize小于等于3000。
- 执行接口的n、batchSize必须与asdBlasMakeSmatinvBatchedPlan创建plan时的参数一致（n大于等于32时还要求batchSize一致），不一致时返回ACL_ERROR_INVALID_PARAM，须重新创建plan后再执行。
- 内部实现自动按矩阵规模分流：n小于32走Gauss-Jordan消元kernel，n大于等于32走LU分解求逆kernel（其中n对16求余为0或8、且batchSize大于等于2或n小于等于128时，走每核独立矩阵的高性能kernel），对用户透明。

## 调用示例

示例代码如下，该样例旨在提供快速上手、开发和调试算子的最小化实现，其核心目标是使用最精简的代码展示算子的核心功能，而非提供生产级的安全保障。不推荐用户直接将示例代码作为业务代码，若用户将示例代码应用在自身的真实业务场景中且发生了安全问题，则需用户自行承担。

完整可执行工程（含编译脚本与残差校验）参见[example_smatinv_batched](../../../../example/A2/BLAS/smatinv_batched/README.md)。

```Cpp
#include <iostream>
#include <vector>
#include "asdsip.h"
#include "acl/acl.h"
#include "acl_meta.h"

using namespace AsdSip;

#define ASD_STATUS_CHECK(err)                                                \
    do {                                                                     \
        AsdSip::AspbStatus err_ = (err);                                     \
        if (err_ != AsdSip::ErrorType::ACL_SUCCESS) {                        \
            std::cout << "Execute failed." << std::endl;                     \
            exit(-1);                                                        \
        } else {                                                             \
            std::cout << "Execute successfully." << std::endl;               \
        }                                                                    \
    } while (0)

void printTensor(const float *tensorData, int64_t batchSize, int64_t rows, int64_t cols)
{
    for(int64_t b = 0; b < batchSize; b++) {
        for (int64_t i = 0; i < rows; i++) {
            for (int64_t j = 0; j < cols; j++) {
                std::cout << tensorData[b * rows * cols + i * cols + j] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}

int Init(int64_t deviceId, aclrtStream *stream)
{
    // 固定写法，acl初始化
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // 调用aclrtMemcpy将host侧数据复制到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(shape.data(),
        shape.size(),
        dataType,
        strides.data(),
        0,
        aclFormat::ACL_FORMAT_ND,
        shape.data(),
        shape.size(),
        *deviceAddr);
    return 0;
}

int main(int argc, char **argv)
{
    int deviceId = 0;

    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    int64_t batchSize = 3;
    int64_t n = 4;

    int64_t tensorASize = batchSize * n * n;
    std::vector<float> tensorInAData;
    std::vector<float> tensorInAinvData;
    std::vector<int32_t> tensorInInfoData;
    tensorInAData.resize(tensorASize);
    tensorInAinvData.resize(tensorASize);
    tensorInInfoData.resize(batchSize);

    for (int32_t batchIdx = 0; batchIdx < batchSize; batchIdx++) {
        for (int32_t i = 0; i < n; i++) {
            for (int32_t j = 0; j < n; j++) {
                if (i == j) {
                    tensorInAData[n * n * batchIdx + n * i + j] = 2.0f + batchIdx;
                } else {
                    tensorInAData[n * n * batchIdx + n * i + j] = 1.0f;
                }
            }
        }
    }

    for (int32_t batchIdx = 0; batchIdx < batchSize; batchIdx++) {
        for (int32_t i = 0; i < n; i++) {
            for (int32_t j = 0; j < n; j++) {
                tensorInAinvData[n * n * batchIdx + n * i + j] = -1.0f;
            }
        }
    }

    for (int32_t batchIdx = 0; batchIdx < batchSize; batchIdx++) {
        tensorInInfoData[batchIdx] = 0;
    }

    std::cout << "------- input TensorInA -------" << std::endl;
    printTensor(tensorInAData.data(), batchSize, n, n);
    std::cout << "------- input TensorInAinv -------" << std::endl;
    printTensor(tensorInAinvData.data(), batchSize, n, n);

    std::vector<int64_t> aShape = {batchSize, n, n};
    std::vector<int64_t> ainvShape = {batchSize, n, n};
    std::vector<int64_t> infoShape = {batchSize};
    aclTensor *inputA = nullptr;
    aclTensor *inputAinv = nullptr;
    aclTensor *inputInfo = nullptr;
    void *inputADeviceAddr = nullptr;
    void *inputAinvDeviceAddr = nullptr;
    void *inputInfoDeviceAddr = nullptr;
    ret = CreateAclTensor(tensorInAData, aShape, &inputADeviceAddr, aclDataType::ACL_FLOAT, &inputA);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);
    ret = CreateAclTensor(tensorInAinvData, ainvShape, &inputAinvDeviceAddr, aclDataType::ACL_FLOAT, &inputAinv);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);
    ret = CreateAclTensor(tensorInInfoData, infoShape, &inputInfoDeviceAddr, aclDataType::ACL_INT32, &inputInfo);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);

    asdBlasHandle handle;
    asdBlasCreate(handle);

    size_t lwork = 0;
    void *buffer = nullptr;
    asdBlasMakeSmatinvBatchedPlan(handle, n, batchSize);
    asdBlasGetWorkspaceSize(handle, lwork);
    std::cout << "lwork = " << lwork << std::endl;
    if (lwork > 0) {
        ret = aclrtMalloc(&buffer, static_cast<int64_t>(lwork), ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    asdBlasSetWorkspace(handle, buffer);
    asdBlasSetStream(handle, stream);
    asdBlasSynchronize(handle);

    ASD_STATUS_CHECK(asdBlasSmatinvBatched(handle, n, inputA, n, inputAinv, n, inputInfo, batchSize));

    asdBlasSynchronize(handle);
    asdBlasDestroy(handle);

    ret = aclrtMemcpy(tensorInAinvData.data(),
        tensorASize * sizeof(float),
        inputAinvDeviceAddr,
        tensorASize * sizeof(float),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("copy Ainv from device to host failed. ERROR: %d\n", ret); return ret);

    std::cout << "------- output TensorInAinv -------" << std::endl;
    printTensor(tensorInAinvData.data(), batchSize, n, n);

    aclDestroyTensor(inputA);
    aclDestroyTensor(inputAinv);
    aclDestroyTensor(inputInfo);
    aclrtFree(inputADeviceAddr);
    aclrtFree(inputAinvDeviceAddr);
    aclrtFree(inputInfoDeviceAddr);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
```
