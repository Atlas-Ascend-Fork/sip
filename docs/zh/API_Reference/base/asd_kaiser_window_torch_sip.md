# torch_sip.asd_kaiser_window

## 功能说明

在 NPU 上生成一维凯泽窗，支持周期窗与对称窗。底层 C 接口和公式参见 [asdKaiserWindow](asdKaiserWindow.md)。

## 接口原型

```python
torch_sip.asd_kaiser_window(
    window_length, periodic=True, beta=12.0, *,
    dtype=None, layout=torch.strided, device=None, requires_grad=False
) -> torch.Tensor
```

## 参数说明

| 参数 | 说明 |
| --- | --- |
| window_length | 窗长度，整数，范围 `[0, INT32_MAX]`。 |
| periodic | 是否生成周期窗，默认 `True`；`False` 生成对称窗。 |
| beta | 窗形参数，默认 `12.0`。 |
| dtype | 输出类型，可选 `torch.float16`、`torch.bfloat16` 或 `torch.float32`；`None` 选择 float32，不随 PyTorch 全局默认类型变化。 |
| layout | 仅支持 `torch.strided`；`None` 也选择 strided。 |
| device | NPU 设备，如 `"npu"`、`"npu:0"`；`None` 使用当前 NPU。 |
| requires_grad | 默认 `False`；为 `True` 时返回可累积梯度的叶子张量。 |

返回形状为 `[window_length]` 的连续 NPU 张量。长度为 0 时返回空张量，长度为 1 时返回 `[1]`；长度大于 1 且 `beta` 为 NaN 或无穷大时，输出为 NaN。

## 约束说明

- 支持 Atlas A2、Atlas A3 和 Ascend 950PR&950DT；需安装与当前 SiP 构建版本配套的 CANN、torch、torch_npu 和 torch_sip。
- 不支持 CPU 设备、非 strided 布局或上表以外的数据类型。设备端计算使用 float32；`beta` 绝对值转换为 float32 后非有限时，长度大于 1 的输出为 NaN。
- 此接口创建叶子张量；`requires_grad=True` 表示后续运算可累积输出张量的梯度，不表示对 `window_length` 或 `beta` 求导。

## 使用示例

已构建并安装匹配的 SiP 与 torch_sip 后，执行：

```python
import torch
import torch_npu
import torch_sip

window = torch_sip.asd_kaiser_window(
    4096, periodic=True, beta=12.0, dtype=torch.float16, device="npu"
)
print(window.shape, window.dtype, window.device)

leaf = torch_sip.asd_kaiser_window(17, requires_grad=True)
leaf.sum().backward()
print(leaf.grad)
```

在 `sip_pta` 目录构建接口并运行仓内单测：

```bash
bash build.sh --install
python test/base/kaiser_window.py
```
