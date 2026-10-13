# ops-fft 联合构建

## 1. 概述

联合构建为 SiP 的 FFT 算子提供第二种执行后端：直调 ops-fft 预编译库 `libcann_ops_fft.so` 导出的 device kernel，常量与 tiling 复用 SiP plan 期产物。

| 后端 | 说明 |
| --- | --- |
| internal | SiP 自带 kernel；纯 internal 构建的默认后端，联合构建下须显式 `SIP_FFT_BACKEND=internal` |
| ops-fft | 直调存件 kernel，须以 `--use_ops_fft` 构建，联合构建下为默认后端（严格档） |

双后端的数值一致性分两档（成因见第 2 节）：

| 路径 | 一致性 |
| --- | --- |
| Ascend910B 全部覆盖路径；Ascend950 混合基路径 | 逐位一致 |
| Ascend950 radix-2 路径（直调存件旧版 kernel） | 数值等价：rel ~2.3e-07，以毒化 + rel err + CPU 锚点验收 |

## 2. 存件版本背景

存件移植自 2026-04 的 SiP 快照（97769d8），此后不再跟随 SiP 演进；SiP 于 2026-06（94ea107）完成 FFT 性能重写。存件 kernel 与 SiP 现行实现的对应关系：

| 存件 kernel | SiP 现行实现 | 一致性 |
| --- | --- | --- |
| Ascend910B 全部 kernel | 同名实现 | 同源未改，逐位一致 |
| `fft_c2c_arch35_mix_multi_core`（混合基） | arch35 混合基 | 同源未改，逐位一致 |
| `fft_c2c_arch35_multi_core`（radix-2） | butterfly 重写版 | 存件为旧版（先 DFT 后乘 twiddle），舍入顺序与 butterfly 不同，仅数值等价；性能低约 14 倍（n=2^21 batch=64：53.3 ms vs 732.8 ms） |
| —（无对应符号） | RegBase/Packed/Gauss（950 纯 2 幂 n < 2^21）；Stockham/RegBase（950 实变换新调度） | 94ea107 新增，未移植进存件 |

命名辨析：`fft_c2c_arch35_`**`mix`**`_multi_core`（含 `mix`）为混合基 kernel；`fft_c2c_arch35_multi_core`（不含 `mix`）为 radix-2 旧版 kernel。

由此得到两条路由事实（详见 4.2）：

1. 950 纯 2 幂 n < 2^21 由 `a5::Select` 调度至 RegBase/Packed/Gauss，存件无对应 kernel，不接入本后端；
2. 950 R2C/C2R 的命中形状全部被 `a5::SelectReal` 优先调度至 Stockham/RegBase，arch35 R2C/C2R core 不会被实例化——直调适配已就位但当前不可达，internal 模式下同样不会走到。

存件 kernel `rfft1_d_fast_dft` 显式不接入：其 tiling（4 指针 + 140B 值传）与 SiP 侧 `DftR2CTilingData`（36B 指针传）不同源，无法安全直调。

## 3. 构建

### 3.1 默认构建

`bash build.sh` 产出纯 internal 安装包，不依赖 `3rdparty/ops-fft/` 存件。

### 3.2 联合构建

```sh
bash build.sh --use_ops_fft
```

存件为强依赖，构建前按 `lib/<soc>/manifest.info` 逐项校验：

- 硬门禁：`soc` 与编译目标一致；存件架构与本机一致（用 binutils 工具直读 `libcann_ops_fft.so` 的 ELF 头比对，当前 Ascend910B 存件为 aarch64、Ascend950 存件为 x86_64）。
- `cann_version`、`abi` 为信息性字段，不参与校验：接口为 `extern "C"` 纯 C，CANN ABI 向后兼容。

校验失败，或编译目标不含 Ascend910B/Ascend950 时，构建立即终止。校验通过后：`libasdsip.so` 以 `-Wl,--no-as-needed` 显式链接 `libcann_ops_fft.so`；存件与头文件收编进安装包（登记 `filelist.csv`，卸载不残留），运行期由 `set_env.sh` 设置的 `LD_LIBRARY_PATH` 定位。

### 3.3 SoC 推导

- 仅启用 Ascend910B/Ascend950 之一：跟随该目标。
- 两者同时启用：按构建机架构选取（aarch64 → Ascend910B 存件，x86_64 → Ascend950 存件）。

## 4. 运行

### 4.1 后端开关

进程级环境变量 `SIP_FFT_BACKEND`：

| 取值 | 语义 |
| --- | --- |
| 未设置 | 联合构建默认严格档（等价 `ops-fft`，杜绝静默 internal）；纯 internal 构建为 internal |
| `ops-fft` | 强制档：路径未适配或存件缺所需 kernel 时返回 `ACL_ERROR_API_NOT_SUPPORT`，不回退。适用于 A/B 验证等须显式生效或显式失败的场景。 |
| `ops-fft-auto` | 自动档：适配且 stub 就绪的 step 走 ops-fft，其余自动回退 internal（各 core 的 Run 分支逐 step 判定）。适用于全量 UT 与存量业务。 |
| `internal` | 显式 internal：联合构建下的逃生口，A/B 对比的 internal 侧使用。 |

- 开关仅影响 kernel 选择，接口与计算结果不变。
- 直调命中日志 `run via ops-fft kernel ... success` 为 INFO 级，须开启 `ASCEND_GLOBAL_LOG_LEVEL=1` 与 `ASCEND_SLOG_PRINT_TO_STDOUT=1`。
- 自动档的回退提示为 ERROR 级（WARN 在默认日志级别下被过滤），默认可见。

### 4.2 路由表

多步 plan 须全部 step 适配才整体接入（如 2D 纵向分解的 `fft_stride`+`fft_b` 组合）；含 Transpose 的多步分解与 core-any 兜底路径不接入。

**Ascend910B**——全部可达，全部逐位一致：

| 命中条件 | 直调 kernel |
| --- | --- |
| C2C：纯 2 幂，256 < n < 32768（batch 任意） | `fft_b` |
| C2C：纯 2 幂，n ≥ 32768 | `fft_n` |
| C2C：纵向（stride > 1），纯 2 幂，2^8 ≤ n ≤ 2^18 | `fft_stride` |
| C2C：n ≤ 256（DFT 路径） | `dft` |
| C2C：混合基（质因子 ⊆ {2,3,5,7,11,13,17,19,23,29,31,37,41,43,47}），n > 256 | `fft_mix_1`~`6` |
| R2C：n ≤ 1024 | `dft_r2c` |
| R2C：混合基，n > 1024 | `fft_r2c_1`~`3`（含 `_even`/`_odd`） |
| C2R：n ≤ 1024 | `dft_c2r` |
| C2R：混合基，n > 1024 | `fft_c2r_1`~`3`（含 `_even`/`_odd`） |
| 2D C2C 融合：{32,64,128}×{32,64,128} | `dd` |

**Ascend950**：

| 命中条件 | 直调 kernel | 一致性 |
| --- | --- | --- |
| C2C：非纯 2 幂，质因子 ⊆ {2,3,5,7,11,13,17,19}；纯 2 幂 n ≥ 2^21 且 batch ≤ 32 亦经 all-2s plan 命中 | `fft_c2c_arch35_mix_multi_core` | 逐位一致 |
| C2C：纯 2 幂，n ≥ 2^21 且 batch > 32 | `fft_c2c_arch35_multi_core`（旧版） | 数值等价 |
| R2C / C2R：非纯 2 幂，n > 1024，质因子 ⊆ {2,3,5,7} | `fft_r2c_multi_core` / `fft_c2r_multi_core` | 当前不可达 |

补充说明：

- **radix-2 可达域由双层路由决定**：`a5::Select` 仅在纯 2 幂 n ≥ 2^21 时回落 arch35；arch35 内部仅 batch > 32 时选 radix-2 kernel。n < 2^21 的纯 2 幂调度至 RegBase/Packed/Gauss（未接入，强制档请求时报错）。
- **R2C/C2R 当前不可达**：命中形状全部被 `a5::SelectReal` 调度至 Stockham/RegBase（未接入本后端），arch35 R2C/C2R core 不会被实例化；适配就位以对齐存件导出，待调度演进后即生效。

### 4.3 验证

独立 A/B 样例 `example/A2/FFT/ops_fft_ab/`（自动编译 + 执行 + 双后端逐位比对 + 日志命中确认）：

```sh
source <CANN 安装路径>/set_env.sh
export ASDSIP_HOME_PATH=<SiP 安装路径>
cd example/A2/FFT/ops_fft_ab
bash build.sh            # Ascend910B：n=8192 命中 fft_b
bash build.sh 15000 4    # Ascend950：混合基 n=15000 命中 arch35 mix
```

通过标准：输出 `[AB] 双后端输出逐位一致`，且日志出现 `run via ops-fft kernel ... success`。未命中路由或使用 internal-only 安装包时，样例在 ops-fft 阶段直接报错（严格语义）。

联合构建下 UT 不设置开关即为严格档，未适配形状会报错；全量执行请使用自动档（未适配 step 回退 internal，ERROR 日志可辨）；纯 internal 构建不受影响。联合构建安装包上运行未适配形状的既有业务（如 2D/3D 分解、istft）须显式 `SIP_FFT_BACKEND=internal` 或 `ops-fft-auto`。

## 5. 存件维护

存件由 ops-fft 仓产出，本仓只收编、不从源码构建。布局：

```text
3rdparty/ops-fft/
├── include/cann_ops_fft.h            # ops-fft 对外头文件
└── lib/<soc>/                        # Ascend910B / Ascend950
    ├── libcann_ops_fft.so            # soname 链：.so -> .so.1 -> .so.x.y.z 三件套
    └── manifest.info
```

`manifest.info` 字段（`build.sh` 仅校验 `soc`，其余为溯源信息）：

```text
soc=<Ascend910B|Ascend950>            # 须与所在目录名一致
abi=<ON|OFF>                          # 信息性
cann_version=<x.y.z>                  # 信息性：制作环境 CANN 版本
cann_path=<路径>                      # 信息性：制作环境 CANN 路径
source_head=<commit>                  # 信息性：ops-fft 源码 commit
so_md5sum=<md5sum>                    # md5sum libcann_ops_fft.so
```

升级注意：若存件后续同步 94ea107 优化版 kernel，radix-2 直调可升级为逐位一致，但 tiling/常量契约会变（butterfly 不含 per-stage DFT 矩阵），`RunViaOpsFftRadix2` 需按新契约重写。
