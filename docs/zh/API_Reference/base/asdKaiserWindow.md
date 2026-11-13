# asdKaiserWindow

## 产品支持情况

| 产品 | 支持情况 |
| --- | --- |
| Ascend 950PR&950DT系列产品 | 支持 |
| Atlas A3系列产品 | 支持 |
| Atlas A2系列产品 | 支持 |
| Atlas 200I/500 A2推理产品 | 不支持 |
| Atlas推理系列产品 | 不支持 |
| Atlas训练系列产品 | 不支持 |

## 功能说明

`asdKaiserWindow` 在 NPU 上生成一维凯泽窗，`asdKaiserWindowGetWorkspaceSize` 查询工作空间大小。设窗长度为 $N > 1$，$M=N$（周期窗）或 $M=N-1$（对称窗），输出为：

$$
w[n] = \frac{I_0\left(\beta \sqrt{1 - \left(\frac{2n}{M} - 1\right)^2}\right)}{I_0(\beta)}, \quad 0 \le n < N
$$

其中 $I_0$ 为零阶第一类修正贝塞尔函数。`beta` 取相反数时结果相同。`N=0` 返回空输出，`N=1` 返回单元素 1；当 `N>1` 且 `beta` 为 NaN 或无穷大时，输出为 NaN。

PyTorch 接口及示例参见 [torch_sip.asd_kaiser_window](asd_kaiser_window_torch_sip.md)。

## 函数原型

```cpp
namespace AsdSip {
extern "C" {
AspbStatus asdKaiserWindowGetWorkspaceSize(size_t* workspaceSize);
AspbStatus asdKaiserWindow(aclTensor* outTensor, int64_t windowLength,
                           bool periodic, double beta, void* stream, void* workspace);
}
} // namespace AsdSip
```

这两个接口声明在 `base_api.h` 的 `AsdSip` 命名空间内。

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| workspaceSize | 输出 | 非空指针，接收工作空间大小，当前返回 0 字节。 |
| outTensor | 输出 | 调用方预先创建的 NPU `aclTensor`；一维、ND 格式，形状为 `[windowLength]`，数据类型为 `ACL_FLOAT16`、`ACL_BF16` 或 `ACL_FLOAT`。长度大于 1 时须连续，可使用存储范围内的连续偏移视图。 |
| windowLength | 输入 | 窗长度，取值范围 `[0, INT32_MAX]`，必须与输出形状一致。 |
| periodic | 输入 | `true` 生成周期窗；`false` 生成对称窗。 |
| beta | 输入 | 控制窗形的参数。设备端计算使用 float32，输出按 `outTensor` 的数据类型转换。 |
| stream | 输入 | NPU 执行流；长度非零时须为有效流。 |
| workspace | 输入 | 当前不需要工作空间，可传入 `nullptr`。 |

返回状态码；错误码说明参见 [SiP返回码](../../context/SiP返回码.md)。

## 约束说明

- 输出数据及流由调用方管理；非空输出需保证设备内存和张量描述符在异步执行期间有效。
- 仅支持上述数据类型及一维 ND 输出。`windowLength=0` 时不下发 kernel。
- 通常使用有限的 `beta`；实现会将其绝对值转为 float32，转换后非有限时，长度大于 1 的输出为 NaN。

## 调用示例

以下片段假定已初始化 ACL、设置设备并创建 `stream` 和满足参数要求的输出张量 `output`。需根据实际应用处理状态码并在适当时机同步执行流。

```cpp
#include "base_api.h"

size_t workspaceSize = 0;
auto status = AsdSip::asdKaiserWindowGetWorkspaceSize(&workspaceSize);
if (status == AsdSip::ErrorType::ACL_SUCCESS && workspaceSize == 0) {
    status = AsdSip::asdKaiserWindow(output, 4096, true, 12.0, stream, nullptr);
}
```
