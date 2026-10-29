# solve

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
solveGetWorkspaceSize：计算solve算子所需的workspace大小。\
solve：对批量单精度实数方阵求解线性方程组 A·X = B。
- 计算公式：

  $$
  A_b X_b = B_b \quad\Rightarrow\quad X_b = A_b^{-1} B_b
  $$

  其中：b为批次号，A_b为n阶非奇异方阵，B_b为n×nrhs的右端项矩阵，X_b为求解结果。

  示例（batch=2, n=3, nrhs=2）：\
输入“A”为：\
[[[4, 1, 0],\
[1, 3, 1],\
[0, 1, 2]],\
[[4, 1, 0],\
[1, 3, 1],\
[0, 1, 2]]]\
输入“B”为：\
[[[7, 12],\
[15, 20],\
[13, 16]],\
[[9, 6],\
[8, 11],\
[7, 10]]]\
调用solve算子后，输出“X”为：\
[[[1, 2],\
[3, 4],\
[5, 6]],\
[[2, 1],\
[1, 2],\
[3, 4]]]

## 函数原型

若需使用“solve”算子，必须先调用“solveGetWorkspaceSize”接口获取所需workspace大小，再调用“solve”接口执行计算。

```Cpp
AsdSip::AspbStatus solveGetWorkspaceSize(
  size_t &size)
```

```Cpp
AsdSip::AspbStatus solve(
  const aclTensor *    a,
  const aclTensor *    b,
  aclTensor *          x,
  void *               stream,
  void *               workspace = nullptr)
```

- **返回值**：

  返回状态码，具体参见[SiP返回码](../../context/SiP返回码.md)。

## solveGetWorkspaceSize

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
      <td>size（size_t &）</td>
      <td>输出</td>
      <td>solve算子所需的workspace大小（单位：字节）。当前版本固定返回0，调用solve时workspace传nullptr即可。</td>
    </tr>
  </tbody>
  </table>

- **返回值**：

  返回状态码，具体参见[SiP返回码](../../context/SiP返回码.md)。

## solve

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
      <td>a（const aclTensor *）</td>
      <td>输入</td>
      <td><ul><li>表示系数矩阵，对应公式中的'A'。</li><li>数据类型仅支持FLOAT32，数据格式支持ND。</li><li>shape为[batch, n, n]，n∈[2, 256]。</li><li>要求为非奇异方阵。</li></ul></td>
    </tr>
    <tr>
      <td>b（const aclTensor *）</td>
      <td>输入</td>
      <td><ul><li>表示右端项矩阵，对应公式中的'B'。</li><li>数据类型仅支持FLOAT32，需与a一致。</li><li>shape为[batch, n, nrhs]，nrhs≥1。</li><li>batch维必须与a一致（不支持广播，需调用方先展开）。</li></ul></td>
    </tr>
    <tr>
      <td>x（aclTensor *）</td>
      <td>输出</td>
      <td><ul><li>表示求解结果，对应公式中的'X'。</li><li>数据类型与shape均与b一致。</li></ul></td>
    </tr>
    <tr>
      <td>workspace（void *）</td>
      <td>输入</td>
      <td>solve算子所需要的workspace。当前版本传nullptr即可（size为0）。</td>
    </tr>
    <tr>
      <td>stream（void *）</td>
      <td>输入</td>
      <td>NPU执行流。</td>
    </tr>
  </tbody>
  </table>

- **返回值**：

  返回状态码，具体参见[SiP返回码](../../context/SiP返回码.md)。

## 约束说明

  - 输入、输出张量均为3维ND格式，需为行主序连续内存（不支持非连续stride）。
  - 数据类型仅支持FLOAT32。
  - 矩阵阶数n支持[2, 256]；batch≥1；右端项列数nrhs≥1。
  - 不支持batch维广播：b的batch维必须与a一致。
  - a必须非奇异。算子不做奇异检测，奇异阵的结果未定义（与LAPACK行为一致）；近奇异矩阵精度随条件数下降，业务侧应保证输入非奇异。
  - workspace需求为0，传nullptr即可；n>64的大矩阵路径所需临时空间由算子内部管理。

## 调用示例

示例代码如下，该样例旨在提供快速上手、开发和调试算子的最小化实现，其核心目标是使用最精简的代码展示算子的核心功能，而非提供生产级的安全保障。不推荐用户直接将示例代码作为业务代码，若用户将示例代码应用在自身的真实业务场景中且发生了安全问题，则需用户自行承担。

```Cpp
#include <iostream>
#include <vector>
#include "asdsip.h"
#include "acl/acl.h"
#include "acl_meta.h"

using namespace AsdSip;

#define ASD_STATUS_CHECK(err)                            \
    do {                                                 \
        AsdSip::AspbStatus err_ = (err);                 \
        if (err_ != AsdSip::ErrorType::ACL_SUCCESS) {    \
            std::cout << "Execute failed." << std::endl; \
            exit(-1);                                    \
        }                                                \
    } while (0)

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

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream *stream)
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

void printTensor(const float *tensorData, int64_t nums)
{
    for (int64_t i = 0; i < nums; i++) {
        std::cout << tensorData[i] << " ";
    }
    std::cout << std::endl;
}

int main(int argc, char **argv)
{
    int deviceId = 0;

    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    int64_t batch = 2;
    int64_t n = 3;
    int64_t nrhs = 2;

    // A = [[4,1,0],[1,3,1],[0,1,2]]（两个批次相同）
    std::vector<float> tensorInAData = {4.0f, 1.0f, 0.0f, 1.0f, 3.0f, 1.0f, 0.0f, 1.0f, 2.0f,
                                        4.0f, 1.0f, 0.0f, 1.0f, 3.0f, 1.0f, 0.0f, 1.0f, 2.0f};
    // B 由已知解 X 反算：X0 = [[1,2],[3,4],[5,6]]，X1 = [[2,1],[1,2],[3,4]]
    std::vector<float> tensorInBData = {7.0f, 12.0f, 15.0f, 20.0f, 13.0f, 16.0f,
                                        9.0f, 6.0f, 8.0f, 11.0f, 7.0f, 10.0f};
    std::vector<float> tensorOutXData(batch * n * nrhs, 0.0f);

    std::vector<int64_t> aShape = {batch, n, n};
    std::vector<int64_t> bShape = {batch, n, nrhs};
    std::vector<int64_t> xShape = {batch, n, nrhs};

    aclTensor *inputA = nullptr;
    aclTensor *inputB = nullptr;
    aclTensor *outputX = nullptr;
    void *inputADeviceAddr = nullptr;
    void *inputBDeviceAddr = nullptr;
    void *outputXDeviceAddr = nullptr;
    ret = CreateAclTensor(tensorInAData, aShape, &inputADeviceAddr, aclDataType::ACL_FLOAT, &inputA);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);
    ret = CreateAclTensor(tensorInBData, bShape, &inputBDeviceAddr, aclDataType::ACL_FLOAT, &inputB);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);
    ret = CreateAclTensor(tensorOutXData, xShape, &outputXDeviceAddr, aclDataType::ACL_FLOAT, &outputX);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);

    size_t workspaceSize = 0;
    ASD_STATUS_CHECK(solveGetWorkspaceSize(workspaceSize));
    ASD_STATUS_CHECK(solve(inputA, inputB, outputX, stream, nullptr));

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    ret = aclrtMemcpy(tensorOutXData.data(),
        GetShapeSize(xShape) * sizeof(float),
        outputXDeviceAddr,
        GetShapeSize(xShape) * sizeof(float),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("copy x from device to host failed. ERROR: %d\n", ret); return ret);
    std::cout << "------- output X -------" << std::endl;
    printTensor(tensorOutXData.data(), GetShapeSize(xShape));
    std::cout << "Execute successfully." << std::endl;

    aclDestroyTensor(inputA);
    aclDestroyTensor(inputB);
    aclDestroyTensor(outputX);
    aclrtFree(inputADeviceAddr);
    aclrtFree(inputBDeviceAddr);
    aclrtFree(outputXDeviceAddr);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
