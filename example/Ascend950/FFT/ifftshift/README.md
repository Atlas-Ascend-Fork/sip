# 信号处理加速库IfftShiftOperation C++ Demo

## 介绍

该目录下为信号处理加速库IfftShiftOperation C++调用示例。

## 功能说明

- 算子功能：asdFftShift 的逆变换，将批量二维复数切片的零频分量从各切片中心移回角落（ifftshift，对齐 CUDA 参考实现 ifftShiftScaleBatch2DKernel 语义），并支持逐元素 scale 乘子。
- 计算公式：
    $$
    output[s][(r + row / 2) \bmod row][(c + col / 2) \bmod col] = scale \times input[s][r][c]
    $$

    其中：
    - $input[s]$ 为第 $s$ 个输入二维复数切片（$row \times col$ 复数矩阵），切片在输入中的起始元素偏移为 $s \cdot dist$；
    - $output[s]$ 为第 $s$ 个输出二维复数切片，输出为连续的 $batch \times row \times col$ 布局；
    - $dist \le 0$ 表示输入切片连续（$dist = row \cdot col$）；
    - $scale$ 为逐元素输出乘子（kernel 内按 float32 逐实部/虚部相乘），可将逆 FFT 的 1/(row*col) 归一化折叠进本 pass；$scale = 1$ 时为纯数据排列，无浮点误差。

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

    特别说明：
    - 上述编译方式仅支持编译通过git下载的加速库，以zip压缩包方式下载的加速库不支持该编译方式；
    - 由于编译过程需要联网下载依赖库，因此编译环境需要联网；
    - 该编译过程包括获取ascend-boost-comm（昇腾分布式通信加速库）组件并编译该组件，和编译信号加速库两个步骤。更多命令介绍可查看SiP仓库的[build.sh](../../../../build.sh)文件。

 - 更多编译命令说明请参考[编译与构建](../../../../docs/compilation_build.md)

### 运行demo

- 进入 `example` 目录并执行构建脚本。

    ```sh
    cd  ${示例所在目录}
    bash build.sh
    ```

## 额外说明

示例中生成的数据不代表实际场景，可根据具体使用场景进行数据修改。

## 产品支持情况

适用于Ascend 950PR/950DT 系列产品。

## 场景说明

提供示例代码分别对应不同场景，编译运行时需要根据具体场景对应更改build脚本：

- **example_acl_ifft_shift.cpp**

    【注】：默认编译脚本可编译运行该示例。示例使用 batch=2、row=4、col=6 的连续输入切片，并取 scale = 1/(row*col) 演示将逆 FFT 归一化折叠进移位 pass（scale = 1.0 时为纯逆排列）；运行后在Host侧与逐元素计算的期望结果做二进制比对并打印 PASS/FAIL。

    **输入**

    | TensorName | DataType  | DataFormat |           Shape           | Value |
    | :--------: | :-------: | :--------: | :-----------------------: | :---: |
    |   input    | complex64 |     nd     |  [batch, row, col]（或按 dist 步距分布的扁平布局） |   -   |

    **输出**

    | TensorName | DataType  | DataFormat |           Shape           | Value |
    | :--------: | :-------: | :--------: | :-----------------------: | :---: |
    |     out    | complex64 |     nd     |       [batch, row, col]   |   -   |
