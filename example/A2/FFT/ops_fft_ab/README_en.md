# ops-fft Joint Build Backend A/B Verification Sample

## Introduction

This is a standalone verification sample (it does not depend on or modify other samples): the same input is processed by the C2C 1D FFT with the internal and ops-fft backends respectively. The outputs are compared bit by bit and the time consumed is reported. The INFO logs are also used to confirm whether the ops-fft kernel is actually hit. No application code changes are required; the backend is switched only by the environment variable `SIP_FFT_BACKEND`.

## Environment Setup

```sh
source [*CANN installation directory*]/set_env.sh
export ASDSIP_HOME_PATH=[*SiP installation directory*]
```

> Note: The SiP installation package must be built with `bash build.sh --use_ops_fft` (the default build is internal-only, without the ops-fft backend).

## Usage

```sh
bash build.sh            # Default n=8192 batch=4 (Ascend910B, fft_b path)
bash build.sh 15000 4    # Ascend950 mixed-radix shape (arch35 mix path)
```

## Result Interpretation

- `[AB] The outputs of the two backends are bit-for-bit identical`: The computation results of the two backends are consistent. Reaching the comparison stage indicates that ops-fft has taken effect. In the logs, `run via ops-fft kernel ... success` is evidence that the direct call is hit (on Ascend910B, `FFTCoreB run via ops-fft kernel (blocks=...)`; on Ascend950, `FftC2CCoreArch35 run via ops-fft kernel (arch35 mix, ...)`).
- The run stops with an error: ops-fft is mandatorily requested, but the current shape does not hit a covered path (see the routing conditions), or the installation package is an internal-only build (the ops-fft library is not linked). In this case, `asdFftExec` returns `ACL_ERROR_API_NOT_SUPPORT`. Check the routing table or confirm that the run is on an installation package built with the joint build.

For routing conditions and more descriptions, refer to [ops-fft joint build](../../../../docs/ops_fft_joint_build.md).
