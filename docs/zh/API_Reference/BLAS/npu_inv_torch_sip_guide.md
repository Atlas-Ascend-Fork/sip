# torch_sip.npu_inv 批量实数矩阵求逆 —— 接入指导

面向 PyTorch 用户的 `torch_sip.npu_inv` / `torch_sip.npu_inv_backward` 使用说明。底层 BLAS 纯 C++ 接口（asdBlasSmatinvBatched）参见 [SmatinvBatched.md](SmatinvBatched.md)。

- **环境**: Atlas A2 系列（Ascend910B3）/ CANN 9.3.0 / torch 2.x + torch_npu / AscendSiPBoost 9.1.0
- **源码位置**: [sip_pta/](../../../../sip_pta/)（PyTorch 扩展）、[ops/blas/sgetri_batched/](../../../../ops/blas/sgetri_batched/) 与 [ops/blas/smatinv_batched/](../../../../ops/blas/smatinv_batched/)（device kernel）

---

## 1. 安装与初始化

```bash
# 前置: CANN 环境已 source, 当前 python 环境已安装 torch + torch_npu
cd sip_pta
bash build.sh --install        # 构建 torch_sip whl 并覆盖安装到当前环境
```

验证安装：

```python
import torch, torch_npu, torch_sip
a = torch.eye(8, dtype=torch.float32, device="npu:0")
y = torch_sip.npu_inv(a)
print(torch.allclose(y, a))    # True
```

## 2. 接口签名

| 接口 | 语义 |
|---|---|
| `torch_sip.npu_inv(a, *, out=None)` | 批量方阵求逆，语义对齐 `torch.linalg.inv`，支持自动微分 |
| `torch_sip.npu_inv_backward(y, grad_y)` | 手动解析反向：`dA = -(Yᵀ dY) Yᵀ`（复数为共轭转置 `-(Y^H dY) Y^H`） |

### 输入约束

| 项 | 约束 |
|---|---|
| shape | `(*, N, N)`，任意批维；最后一两维必须为方阵 |
| dtype | `float32` / `float16` / `bfloat16` / `complex64`（`complex32` 仅 CPU 构造后上传场景） |
| N | 快速域 N≤256；N>256 自动走 torch LU 分块保底路径 |
| batch | 快速域 B≤3000；更大自动分块批处理 |
| 内存布局 | 连续张量最优（非连续自动物化，有拷贝开销） |

### 路由行为（对用户透明）

```
npu_inv(a):
  complex64            → cgetri 融合 kernel（原生复数路径）
  实数 N<32            → Gauss-Jordan 消元 kernel（smatinv_batched）
  实数 32≤N≤256        → LU 求逆链（sgetri_batched）
      N%16∈{0,8} 且 B≥2（或 N≤128） → 每核独立矩阵高性能 kernel（sgetri_p2_batched）
      其余              → 同上，host 侧 pad 到 8 的倍数后仍走高性能 kernel
  实数 N>256           → torch 小算子 LU 分块保底
```

低精度（fp16/bf16）内部以 fp32 计算后回转；pad 补齐由 host 侧自动完成，用户无感。

## 3. 前向用法

### 3.1 自动微分（推荐，与 torch 用法一致）

```python
a = torch.randn(8, 128, 128, dtype=torch.float32, device="npu:0")
a = a + 5.0 * torch.eye(128, device="npu:0")     # 保证非奇异
a.requires_grad_(True)

y = torch_sip.npu_inv(a)
y.backward(torch.randn_like(y))                  # dA 自动写入 a.grad
```

`requires_grad=True` 时自动建图（实数走 C++ autograd，复数走 composite），`y.backward(dy)` 即得梯度，无需手写反向。

### 3.2 手动两阶段（推理训练混合场景，性能最优）

正向不建图 + 反向纯解析式，省去 autograd 引擎开销：

```python
with torch.no_grad():                            # 正向阶段: 不建图
    y = torch_sip.npu_inv(a)
    # y 由调用方保存到反传阶段
...
dy = <上游产生的 dL/dy>                           # 反传阶段
ga = torch_sip.npu_inv_backward(y, dy)           # dA = -(Y^T dY) Y^T
```

`npu_inv_backward` 的 fp32 连续 `[B,N,N]` 输入走 aten 直发快路径（`baddbmm + bmm`），与手写裸表达式同一派发链路，无自定义算子 dispatch 开销。

### 3.3 out 参数

```python
out = torch.empty_like(a)
torch_sip.npu_inv(a, out=out)                    # 语义对齐 torch.linalg.inv(out=)
```

## 4. 从 torch.linalg.inv 迁移

```python
# 迁移前
y = torch.linalg.inv(a)
# 迁移后 (仅替换函数名, 语义/梯度/out= 均对齐)
y = torch_sip.npu_inv(a)
```

差异说明：
- `torch.linalg.inv` 对奇异阵在 LU 保底路径抛 `SingularMatrixError` 同语义错误；快速域对奇异阵结果未定义（与 LAPACK getri 行为一致，业务侧应保证输入非奇异）。
- 非方阵/维数<2/不支持 dtype 抛 `ValueError`/`TypeError`，与 torch 一致。

## 5. 精度与性能参考

（完整数据见性能/精度报告，以下为典型值。）

**精度**（fp32，残差 `max‖A·A⁻¹−I‖`）：

| 场景 | 典型残差 |
|---|---|
| 对角占优随机阵 N≤256 | ~1e-6 级 |
| 一般随机阵 | ~1e-5 级 |

**性能**（Atlas 800T A2，同口径对照 torch 小算子拼接 GJ 版本）：

| 场景 | 结论 |
|---|---|
| 前向 B≥2、N∈[8,256]（命中每核独立矩阵 kernel） | 较 GJ 提升 4x~11x（N 越小收益越大） |
| 前向其余快速域 shape | 与 GJ 相当或更优 |
| 反向（fp32 全 shape） | 与 GJ 裸表达式同带或更优（0.072~0.083ms 常见 shape） |

## 6. 常见问题

**Q1: 复数 complex32 报错？** 本 torch_npu 版本不支持 c32 设备侧 copy/cast，请在 CPU 构造后 `.to(torch.complex64)` 再上传 NPU。

**Q2: batch 切片传入结果全批错位？** 首维切片（`a[s:e]`）数据基址带 storage_offset，扩展层已自动物化；若使用底层 BLAS 裸接口（asdBlasSmatinvBatched），请保证传入 tensor `storage_offset()==0`。

**Q3: 图模式（torch.compile）可用吗？** 已验证 npugraph_ex/aclgraph 下可融合，输入需为 NPU 常驻 tensor。

**Q4: N>256 或 B>3000 会失败吗？** 不会，自动走 torch LU 分块保底/分块批处理，性能低于快速域但保证正确性。
