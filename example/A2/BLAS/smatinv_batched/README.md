# 信号处理加速库SmatinvBatchedOperation C++ Demo

## 介绍

该目录下为信号处理加速库SmatinvBatchedOperation C++调用示例（批量实数矩阵求逆）。

## 功能说明

- 算子功能：批量计算实数矩阵的逆。
- 计算公式：

    $$
    Ainv[k] = A[k]^{-1} \quad \text{for } k = 1, 2, \dots, batchSize
    $$

    其中：
    - $A[k]$ 为实数方阵（行主序，lda = n），fp32 或 fp16
    - 支持规模：n < 256，batchSize <= 3000
- 内部实现（对用户透明，按 n 与 batchSize 自动选择）：
  - n < 32：Smatinv Gauss-Jordan 消元 kernel（fp32/fp16 原生）
  - 32 <= n <= 256：SgetriBatched LU 分解求逆 kernel（fp32；fp16 升 fp32 计算后回转）
- 接口：

    | 接口 | 说明 |
    |---|---|
    | `asdBlasMakeSmatinvBatchedPlan(handle, n, batchSize)` | 创建 plan |
    | `asdBlasSmatinvBatched(handle, n, A, lda, Ainv, lda_inv, info, batchSize)` | fp32 执行 |
    | `asdBlasHSmatinvBatched(...)` | fp16 执行 |

## 使用说明

### 环境配置

 - 配置CANN环境变量

    ```sh
    source [CANN安装路径]/set_env.sh
    ```

    默认：source /usr/local/Ascend/ascend-toolkit/set_env.sh

### SiP编译

 - 用户应进入SiP根目录，执行如下指令进行信号处理加速库的编译，并设置加速库环境变量。

    ```sh
    cd ${SiP_root_path}
    bash build.sh
    source output/set_env.sh
    ```

 - 更多编译命令说明请参考[编译与构建](../../../../docs/compilation_build.md)

### 运行demo

- 进入 `example` 目录并执行构建脚本。

    ```sh
    cd  ${示例所在目录}
    bash build.sh
    ```

## 额外说明

- 示例中 n=64/batch=2 为演示规模，可根据实际场景修改（n < 256）。
- 对角加扰（对角占优）保证示例矩阵可逆；实际场景请保证输入矩阵非奇异（info 指示）。
- PyTorch 场景接入请使用 `torch_sip.npu_inv`（自动路由到本算子，含反向），参考 sip_pta 侧 `test/blas/smatinv_batched.py` 与 docs。

## 产品支持情况

适用于Atlas A2/A3 训练系列产品、Atlas 800I A2 推理产品、Atlas A3 推理系列产品。
