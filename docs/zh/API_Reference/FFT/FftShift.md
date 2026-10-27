# FftShift

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->

<!-- npu="A3" id2 -->
- <term>Atlas A3系列产品</term>：不支持
<!-- end id2 -->

<!-- npu="910b" id3 -->
- <term>Atlas A2系列产品</term>：不支持
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
  asdFftShift：将批量二维复数切片的零频分量移动到各切片中心（fftshift，对齐 CUDA 参考实现 fftShiftBatch2DKernel 语义），常用于二维 FFT 前后的频谱重排，典型场景为 MRI 重建等医学图像处理流水线。
- 计算公式：

  $$
  output[s][(r + (row + 1) / 2) \bmod row][(c + (col + 1) / 2) \bmod col] = input[s][r][c]
  $$

  其中：
  - $input[s]$ 为第 $s$ 个输入二维复数切片（$row \times col$ 复数矩阵），切片在输入中的起始元素偏移为 $s \cdot dist$；
  - $output[s]$ 为第 $s$ 个输出二维复数切片，输出为连续的 $batch \times row \times col$ 布局；
  - $dist \le 0$ 表示输入切片连续（$dist = row \cdot col$）；
  - 该算子为纯数据排列，输出与参考实现二进制一致，无浮点误差。

## 函数原型

```Cpp
AspbStatus asdFftShift(
  int64_t               row,
  int64_t               col,
  int64_t               batch,
  int64_t               dist,
  const aclTensor *     input,
  aclTensor *           output,
  void *                stream,
  void *                workspace = nullptr)
```

## asdFftShift

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
      <td>row（int64_t）</td>
      <td>输入</td>
      <td>单个切片的行数，需为正数。</td>
    </tr>
    <tr>
      <td>col（int64_t）</td>
      <td>输入</td>
      <td>单个切片的列数，需为正数。</td>
    </tr>
    <tr>
      <td>batch（int64_t）</td>
      <td>输入</td>
      <td>切片数量，需为正数。</td>
    </tr>
    <tr>
      <td>dist（int64_t）</td>
      <td>输入</td>
      <td>输入切片间元素步距。dist &le; 0 表示输入切片连续（按 row * col 步距分布）；dist &gt; 0 时需满足 dist &ge; row * col（切片间存在 padding）。</td>
    </tr>
    <tr>
      <td>input（aclTensor *）</td>
      <td>输入</td>
      <td>输入张量，数据类型 complex64。包含 batch 个 row &times; col 的复数切片，第 s 个切片的起始元素偏移为 s * dist，元素总数需不少于 (batch - 1) * dist + row * col。</td>
    </tr>
    <tr>
      <td>output（aclTensor *）</td>
      <td>输出</td>
      <td>输出张量，数据类型 complex64，形状为 [batch, row, col] 的连续布局。</td>
    </tr>
    <tr>
      <td>stream（void *）</td>
      <td>输入</td>
      <td>指定执行任务的流。</td>
    </tr>
    <tr>
      <td>workspace（void *）</td>
      <td>输入</td>
      <td>预留参数，当前无需传入workspace，使用默认值 nullptr 即可。</td>
    </tr>
  </tbody>
  </table>

- **返回值**：

  返回状态码，具体参见[SiP返回码](../../context/SiP返回码.md)。

## 约束说明

- 当前版本仅支持 2D 切片（row × col）的 fftshift，其他维度支持待后续版本补齐。
- 输入、输出 tensor 数据类型仅支持 complex64。
- row、col、batch 需为正数；dist &gt; 0 时需满足 dist &ge; row * col。
- 不支持输入 tensor 与输出 tensor 为同一个 tensor。
- 典型 MRI shape（batch=12000, row=500, col=50，单次搬运 4.8GB）在 Ascend 950PR 上实测约 3.33 ms（有效带宽约 1441 GB/s）。

## 调用示例

示例代码如下，该样例旨在提供快速上手、开发和调试算子的最小化实现，其核心目标是使用最精简的代码展示算子的核心功能，而非提供生产级的安全保障。不推荐用户直接将示例代码作为业务代码，若用户将示例代码应用在自身的真实业务场景中且发生了安全问题，则需用户自行承担。

```Cpp
#include <cstdlib>
#include <iostream>
#include <vector>
#include "asdsip.h"
#include "acl/acl.h"
#include "aclnn/acl_meta.h"
using namespace AsdSip;

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

#define ASD_STATUS_CHECK(err)                            \
    do {                                                 \
        AsdSip::AspbStatus err_ = (err);                 \
        if (err_ != AsdSip::ErrorType::ACL_SUCCESS) {    \
            std::cout << "Execute failed." << std::endl; \
            return -1;                                   \
        }                                                \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    // 固定写法，AscendCL初始化
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor)
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
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main()
{
    int32_t deviceId = 0;
    // 优先从环境变量ASDSIP_DEVICE_ID 获取实际 device id, 未设置时默认 0
    const char* envDeviceId = std::getenv("ASDSIP_DEVICE_ID");
    if (envDeviceId != nullptr) {
        deviceId = std::atoi(envDeviceId);
    }
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 创造tensor的Host侧数据: batch 个 row x col 复数切片
    int64_t batch = 2, row = 4, col = 6;
    int64_t dist = 0; // 0 表示输入切片连续 (dist = row * col)
    const int64_t tensorSize = batch * row * col;
    std::vector<int64_t> inShape = {batch, row, col};
    std::vector<std::complex<float>> inputHostData(tensorSize, std::complex<float>(0, 0));
    for (int64_t i = 0; i < tensorSize; i++) {
        inputHostData[i] = std::complex<float>(static_cast<float>(i), static_cast<float>(i) + 1.0f);
    }
    std::vector<std::complex<float>> outHostData(tensorSize, std::complex<float>(0, 0));
    void* inputDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* input = nullptr;
    aclTensor* out = nullptr;
    ret = CreateAclTensor(inputHostData, inShape, &inputDeviceAddr, aclDataType::ACL_COMPLEX64, &input);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);
    ret = CreateAclTensor(outHostData, inShape, &outDeviceAddr, aclDataType::ACL_COMPLEX64, &out);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);

    // 执行 FftShift: output[s][(r + (row+1)/2) % row][(c + (col+1)/2) % col] = input[s][r][c]
    ASD_STATUS_CHECK(asdFftShift(row, col, batch, dist, input, out, stream));
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    std::vector<std::complex<float>> outData(tensorSize, std::complex<float>(0, 0));
    ret = aclrtMemcpy(outData.data(), tensorSize * sizeof(outData[0]), outDeviceAddr, tensorSize * sizeof(outData[0]),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Host侧golden: 纯排列算子, 逐元素二进制比对
    const int64_t shiftR = (row + 1) / 2;
    const int64_t shiftC = (col + 1) / 2;
    int64_t diff = 0;
    for (int64_t s = 0; s < batch; s++) {
        for (int64_t r = 0; r < row; r++) {
            for (int64_t c = 0; c < col; c++) {
                auto expect = inputHostData[s * row * col + r * col + c];
                auto got = outData[s * row * col + (r + shiftR) % row * col + (c + shiftC) % col];
                if (expect != got) {
                    diff++;
                }
            }
        }
    }
    std::cout << "FftShift result: " << (diff == 0 ? "PASS" : "FAIL") << " (" << tensorSize - diff << "/"
              << tensorSize << " elements exact)" << std::endl;

    aclDestroyTensor(input);
    aclDestroyTensor(out);
    aclrtFree(inputDeviceAddr);
    aclrtFree(outDeviceAddr);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
