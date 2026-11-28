# 信号处理加速库Operation C++ Demo

## 介绍

该目录下为信号处理加速库Operation C++调用示例。

## 使用说明

### 环境配置

 - 配置CANN与SiP环境变量（示例构建依赖SiP库的`ASDSIP_HOME_PATH`，仅source CANN无法通过`example/build.sh`的前置检查）

    ```sh
    # CANN环境变量（默认路径安装；非root用户将/usr/local替换为${HOME}）
    source /usr/local/Ascend/cann/set_env.sh
    # SiP库环境变量（源码编译直接构建场景）
    source output/set_env.sh
    # 若使用run包安装，对应为（${install_path}为安装目录）
    # source ${install_path}/asdsip/set_env.sh
    ```

### SiP编译

 - 用户应进入SiP根目录，执行如下指令进行信号处理加速库的编译，并设置加速库环境变量。

    ```sh
    SiP_root_path="$(git rev-parse --show-toplevel)" || exit 1
    cd "${SiP_root_path:?Set the repository root path first}" || exit 1
    bash build.sh
    source output/set_env.sh
    ```

    特别说明：
    - 上述编译方式仅支持编译通过git下载的加速库，以zip压缩包方式下载的加速库不支持该编译方式；
    - 由于编译过程需要联网下载依赖库，因此编译环境需要联网；
    - 该编译过程包括获取ascend-boost-comm（昇腾分布式通信加速库）组件并编译该组件，和编译信号加速库两个步骤。更多命令介绍可查看SiP仓库`build.sh`文件。

 - 更多编译命令说明请参考[编译与构建](../docs/compilation_build.md)

### 运行demo

- 进入 `example` 目录并执行构建脚本。

    ```sh
    cd  ${示例所在目录}
    bash build.sh
    ```

## 额外说明

示例中生成的数据不代表实际场景，可根据具体使用场景进行数据修改。
