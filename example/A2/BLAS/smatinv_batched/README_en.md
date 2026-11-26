# Signal Processing Acceleration Library SmatinvBatchedOperation C++ Demo

## Introduction

This directory contains C++ call examples of the SmatinvBatchedOperation operator (batched real matrix inversion) in the signal processing acceleration library.

## Description

- Operator function: Computes inverses of real matrices in batches.
- Calculation formula:

    $$
    Ainv[k] = A[k]^{-1} \quad \text{for } k = 1, 2, \dots, batchSize
    $$

    where:
    - $A[k]$ is a real square matrix (row-major order, lda = n), fp32 or fp16
    - Supported scale: n < 256 and batchSize <= 3000

- Internal implementation (transparent to users; automatically selected based on n and batchSize):

  - n < 32: Smatinv Gauss-Jordan elimination kernel (native fp32/fp16)
  - 32 <= n <= 256: SgetriBatched LU decomposition inversion kernel (fp32; fp16 is converted to fp32 for computation and then converted back)

- Interfaces:

    | Interface | Description |
    |---|---|
    | `asdBlasMakeSmatinvBatchedPlan(handle, n, batchSize)` | Creates a plan |
    | `asdBlasSmatinvBatched(handle, n, A, lda, Ainv, lda_inv, info, batchSize)` | fp32 execution |
    | `asdBlasHSmatinvBatched(...)` | fp16 execution |

## Usage

### Environment Setup

 - Set CANN environment variables

    ```sh
    source [*CANN installation directory*]/set_env.sh
    ```

    Default: source /usr/local/Ascend/ascend-toolkit/set_env.sh

### Building SiP

 - Enter the SiP root directory and run the following commands to compile the signal processing acceleration library and set the acceleration library environment variables.

    ```sh
    SiP_root_path="$(git rev-parse --show-toplevel)" || exit 1
    cd "${SiP_root_path:?Set the repository root path first}" || exit 1
    bash build.sh
    source output/set_env.sh
    ```

 - For more compilation command descriptions, refer to [Compilation and Build](../../../../docs/compilation_build_en.md)

### Running the Demo

- Go to the `example` directory and run the build script.

    ```sh
    cd  ${*example_directory*}
    bash build.sh
    ```

## Additional Notes

- In the example, n=64/batch=2 is a demo scale. You can modify it according to the actual scenario (n < 256).
- Diagonal perturbation (diagonal dominance) ensures that the example matrices are invertible. In actual scenarios, ensure that the input matrices are non-singular (indicated by info).
- For PyTorch integration, use `torch_sip.npu_inv` (automatically routed to this operator, including the backward pass). For details, refer to `test/blas/smatinv_batched.py` and the docs on the sip_pta side.

## Product Support

This example applies to Atlas A2/A3 training products, Atlas 800I A2 inference product, and Atlas A3 inference products.
