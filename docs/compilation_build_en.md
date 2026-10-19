# Compilation and Build

## SiP Compilation

### Download the Acceleration Library Source Code

```shell
git clone https://gitcode.com/cann/sip.git
```

You can select any branch as needed.

### Compile

Users need to enter the root directory of the acceleration library and compile

```shell
cd ${SiP_root_path}
bash build.sh
```

Note: SiP library compilation depends on ascend-boost-comm. Therefore, the compilation process involves two steps: ① pulling and compiling the ascend-boost-comm (Ascend Distributed Communication Acceleration Library) repository; ② compiling the SiP acceleration library. For more command introductions, refer to the [`README.md`](../README.md) and [`build.sh`](../build.sh) files in the main directory of the SiP repository.

> For the complete dependency list, refer to the [`requirements.txt`](../requirements.txt) and the "Environment Setup" section in [`README.md`](../README.md) in the root directory.

### SiP Compilation Related Notes

The basic compilation command for SiP is `bash build.sh`. In the default build mode, this command generates version information and creates an installation package.

  ```sh
  bash build.sh
  ```

  Use the --help parameter to obtain the build.sh script functions and corresponding instructions.

  ```sh
  --help                         Displays help message.
  --dev                          Compile only the operator library. If type is empty, the default is dev.
  --clean                        Clear caches and dependent third-party libraries.
  --ut                           Compile and execute unit test cases.
```

### Compilation Target Chip Architecture Configuration

CANN versions older than 9.1.0 automatically omit all `ascend950` (A5) device
kernels, even when requested in the target configuration. The host library and
enabled kernels for other architectures are still built. Such a package cannot
execute A5 operators. With CANN 9.1.0 or newer, the configured A5 targets are built.
This is a SiP build compatibility policy, not a statement about all CANN 9.0 APIs.

Configuration prints the detected CANN version and effective device targets.
Version detection uses the active `ASCEND_HOME_PATH` installation's
`include/version/cann_version.h`, falling back to `compiler/version.info`.
An unknown version with A5 requested, or no compatible device targets remaining,
stops configuration with an error. Source the appropriate CANN environment
before building. No source configuration or third-party installation is rewritten.

The SiP library supports multiple Ascend chip architectures. Target chips to compile are configured through the `targets` field of `build_config.json`. The project's built-in [`configs/build_config.json`](../configs/build_config.json) has the following content (this file is read by the repository's own scripts; for the configuration actually read for kernel compilation targets, see the "Custom Compilation Configuration File" section below):

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

The meaning of each field is as follows:

| Chip Architecture | Corresponding Product Series |
| --- | --- |
| `ascend310b` | Atlas 200I/500 A2 Inference Card |
| `ascend310p` | Atlas 300I Inference Card |
| `ascend910b` | Atlas A2 Training/Inference Server |
| `ascend950` | Ascend 950PR/950DT |

Set the targets that need compilation to `true`, and set the targets that do not need compilation to `false`.

> **Note**: The default configuration enables both `ascend910b` and `ascend950` targets simultaneously. If your runtime environment only supports one of these chips (for example, only supports Ascend910), set the unsupported chip architecture to `false`. Otherwise, compiling an unsupported chip target will fail due to hardware feature mismatch (typical error such as `simd_vf function 'RegCompute' must be a free function or static member function`).

#### Custom Compilation Configuration File

The device kernel targets requested (i.e., the architectures for which the `mki` component generates kernel build switches) are determined by the configuration read by the `mki` component (located at `3rdparty/mki/`, pulled from `ascend-boost-comm` during the build). When `BUILD_CONFIG_DIR` is not set, the `mki` component uses its built-in `configs/build_config.json` (all architectures enabled; for SiP this requests `ascend910b` and `ascend950`). To disable unsupported architectures, specify a directory containing a `build_config.json` through the environment variable `BUILD_CONFIG_DIR`; no file under `3rdparty/mki/` needs to be modified. The `build_config.json` in that directory follows the same format as [`configs/build_config.json`](../configs/build_config.json) above and must list all architecture keys involved in kernel building (at least `ascend910b` and `ascend950`); set the targets you do not want to build to `false`. For example, reuse the repository configuration from the repository root (the build process changes working directories, so `realpath configs` resolves the configuration directory to an absolute path at export time):

```sh
cd "${SiP_root_path:?Set the repository root path first}" || exit 1
BUILD_CONFIG_DIR="$(realpath configs)" || exit 1
BUILD_CONFIG_FILE="$(realpath configs/build_config.json)" || exit 1
export BUILD_CONFIG_DIR BUILD_CONFIG_FILE
bash build.sh
```

The example above exports both variables so that the repository's own scripts and the `mki` component read the same file: `BUILD_CONFIG_DIR` is a directory variable, from which `mki` reads `<BUILD_CONFIG_DIR>/build_config.json`; `BUILD_CONFIG_FILE` is a file variable used by the repository's own [`scripts/build_util.py`](../scripts/build_util.py) for target selection (e.g., SoC selection for the ops-fft joint build with `--use_ops_fft` enabled, and `scripts/update_tbe_tactic_json.py`). Setting `BUILD_CONFIG_DIR` alone already controls the default kernel compilation targets, but it does not change the configuration read by the repository's own scripts; if you adjust both sides, keep them consistent.

> **Note**: The configuration actually read by the `mki` component is determined by the `get_build_target_list()` function in its `scripts/build_util.py`: when `BUILD_CONFIG_DIR` is not set, `3rdparty/mki/configs/build_config.json` is read; when it is set, `<BUILD_CONFIG_DIR>/build_config.json` is read. Targets requested through the `mki` configuration are still subject to the CANN version compatibility filtering of this repository (see the notes above): with CANN versions older than 9.1.0, all `ascend950` kernels are skipped automatically. The actually built targets are shown in the build log as `requested device targets` and `effective device targets`.

### SiP Key File Introduction

1. `scripts` directory:
   - `install.sh`: Installation script
   - `uninstall.sh`: Uninstallation script
   - `release.sh`: Fully automatic build and packaging script
   - `set_env.sh`: SiP environment variable setup file
   - `build_util.py`: Compilation helper script. Responsible for reading the compilation target configuration (`build_config.json`) and operator binary packaging.
2. `configs` directory:

   - `build_config.json`: Compilation target chip architecture configuration file
   - `op_list.yaml`: Operator list configuration
3. `output` directory:

   - `version.info`
4. `output/lib` directory:

   - `libasdsip.so`: Dynamic link library file of the SiP acceleration library
   - `libmki.so`: Dynamic link library file of the MKI library

## Configuration Files

### Compilation File `build.sh`

File name: `build.sh`
This is the acceleration library compilation file. You can set the log storage directory, log file, compiler version, etc. in this file. Generally, no modification is required.

#### Environment Variable Setup File `set_env.sh`

​**File name**​: `scripts/set_env.sh`
After the acceleration library is installed, the system provides `set_env.sh` (a process-level environment variable setup script) to automatically complete environment variable setup. The settings expire automatically after the user process ends.
If you want to view the descriptions of related variables, visit [Environment Variable Reference - CANN Community Edition - Ascend Community](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/maintenref/envvar/envref_07_0001.html).
