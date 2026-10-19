# 编译与构建

## SiP编译

### 下载加速库源码

```shell
git clone https://gitcode.com/cann/sip.git
```

您可自行选择需要的分支。

### 编译

用户需进入加速库的根目录并进行编译

```shell
cd ${SiP_root_path}
bash build.sh
```

【注】：SiP库编译依赖ascend-boost-comm，因此该编译过程涉及①拉取并编译ascend-boost-comm（昇腾分布式通信加速库）仓；②编译SiP加速库两个过程。更多命令介绍可查看sip仓主目录下的[`README.md`](../README.md)和[`build.sh`](../build.sh)文件。

> 完整依赖清单请参考根目录 [`requirements.txt`](../requirements.txt) 与 [`README.md`](../README.md) 的「环境构建」章节。

### SiP编译相关说明

SiP的基本编译命令是`bash build.sh`，在默认构建模式下，该命令会生成版本信息，并创建安装包。

  ```sh
  bash build.sh
  ```

  可通过使用--help参数获取build.sh脚本功能以及对应的指令。

  ```sh
  --help                         Displays help message.
  --dev                          仅编译算子库, 若type为空，默认为dev.
  --clean                        清除缓存和依赖的三方库.
  --ut                           编译执行单元测试用例.
```

### 编译目标芯片架构配置

CANN 版本低于 9.1.0 时，即使配置启用了 `ascend950`，也会自动跳过全部 A5 设备内核。
主机库和其他已启用架构的内核仍会编译；此类安装包不支持在 A5 上执行算子。
CANN 9.1.0 及以上版本按原配置编译 A5 内核。这是 SiP 的构建兼容策略，
并不表示 CANN 9.0 的所有 API 均不支持 A5。

配置阶段会打印检测到的 CANN 版本和实际编译的目标架构。
版本来自当前 `ASCEND_HOME_PATH` 下的 `include/version/cann_version.h`，
若无法读取则使用 `compiler/version.info`。
请求 A5 但无法确定版本，或过滤后没有兼容的设备目标时，配置会报错。
构建前请加载对应的 CANN 环境；构建过程不会改写源配置或第三方安装。

SiP库支持多种Ascend芯片架构，通过`build_config.json`的`targets`字段配置需要编译的目标芯片。项目自带的[`configs/build_config.json`](../configs/build_config.json)内容如下（该文件由主仓脚本读取，内核编译目标实际读取的配置见下文「自定义编译配置文件」一节）：

```json
{
    "targets": {
        "ascend310b": false,
        "ascend310p": false,
        "ascend910b": true,
        "ascend950": true
    }
}
```

各字段含义说明：

| 芯片架构 | 对应产品系列 |
| --- | --- |
| `ascend310b` | Atlas 200I/500 A2 推理卡 |
| `ascend310p` | Atlas 300I 推理卡 |
| `ascend910b` | Atlas A2 训练/推理服务器 |
| `ascend950` | Ascend 950PR/950DT |

将需要编译的目标设置为`true`，不需要编译的目标设置为`false`。

> **注意**：默认配置同时启用了`ascend910b`和`ascend950`两个目标。若您的运行环境仅支持其中一种芯片（例如仅支持Ascend910），请将不支持的芯片架构设置为`false`，否则编译不支持的芯片目标时会因硬件特性不匹配而失败（典型报错如`simd_vf function 'RegCompute' must be a free function or static member function`）。

#### 自定义编译配置文件

内核编译目标的请求（即`mki`组件为哪些芯片架构生成内核编译开关）由构建期拉取的`mki`组件（位于`3rdparty/mki/`，来自`ascend-boost-comm`）读取的配置决定。未设置`BUILD_CONFIG_DIR`时，`mki`使用其内置的`configs/build_config.json`（全架构启用，对SiP而言会请求`ascend910b`和`ascend950`）；需要关闭不支持的架构时，通过环境变量`BUILD_CONFIG_DIR`指定一个包含`build_config.json`的目录即可，无需修改`3rdparty/mki/`下的文件。目录中的`build_config.json`格式与上文[`configs/build_config.json`](../configs/build_config.json)一致，需完整列出内核涉及的架构键（至少包含`ascend910b`与`ascend950`），将不编译的目标设置为`false`。例如在仓库根目录复用主仓配置文件（构建过程会切换工作目录，故用`realpath configs`在设置时将配置目录解析为绝对路径）：

```sh
cd "${SiP_root_path:?Set the repository root path first}" || exit 1
BUILD_CONFIG_DIR="$(realpath configs)" || exit 1
BUILD_CONFIG_FILE="$(realpath configs/build_config.json)" || exit 1
export BUILD_CONFIG_DIR BUILD_CONFIG_FILE
bash build.sh
```

上例同时导出两个变量，使主仓脚本与`mki`读取同一份配置：`BUILD_CONFIG_DIR`是目录变量，`mki`据此读取`<BUILD_CONFIG_DIR>/build_config.json`；`BUILD_CONFIG_FILE`是文件变量，主仓[`scripts/build_util.py`](../scripts/build_util.py)据此选择架构（如启用`--use_ops_fft`联合编译时的ops-fft SoC选择与`scripts/update_tbe_tactic_json.py`）。仅设置`BUILD_CONFIG_DIR`即可控制内核编译目标的默认配置，但不会改变主仓脚本读取的配置；如需同时调整，建议两处保持一致。

> **说明**：`mki`组件实际读取的配置由其`scripts/build_util.py`中的`get_build_target_list()`函数决定——未设置`BUILD_CONFIG_DIR`时读取`3rdparty/mki/configs/build_config.json`，设置后读取`<BUILD_CONFIG_DIR>/build_config.json`。`mki`配置请求的目标仍会经过本仓的CANN版本兼容过滤（见上文说明）：CANN版本低于9.1.0时自动跳过全部`ascend950`内核，实际构建的目标以编译日志中的`requested device targets`与`effective device targets`为准。

### SiP关键文件介绍

1. `scripts`目录：
   - `install.sh`：安装脚本
   - `uninstall.sh`：卸载脚本
   - `release.sh`：全自动构建与打包脚本
   - `set_env.sh`：SiP的环境变量设置文件
   - `build_util.py`：编译辅助脚本，负责读取编译目标配置（`build_config.json`）及算子二进制打包
2. `configs`目录：

   - `build_config.json`：编译目标芯片架构配置文件
   - `op_list.yaml`：算子列表配置
3. `output`目录：

   - `version.info`
4. `output/lib`目录：

   - `libasdsip.so`：SiP加速库的动态链接库文件
   - `libmki.so`：MKI库的动态链接库文件

## 配置文件

### 编译文件`build.sh`

文件名：`build.sh`
加速库编译文件，文件中可设置日志存放目录、日志文件、编译器版本等，一般无需更改。

#### 环境变量设置文件`set_env.sh`

​**文件名**​：`scripts/set_env.sh`
加速库安装完成后，系统提供`set_env.sh`（进程级环境变量设置脚本），以自动完成环境变量设置，用户进程结束后自动失效。
若想查看相关变量含义说明，可访问[环境变量参考-CANN社区版-昇腾社区](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/maintenref/envvar/envref_07_0001.html)。
