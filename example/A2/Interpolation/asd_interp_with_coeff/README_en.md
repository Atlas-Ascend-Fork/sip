# Signal Processing Acceleration Library AsdInterpWithCoeffOperation C++ Demo

## Introduction

This directory contains C++ call examples of the AsdInterpWithCoeffOperation operator in the signal processing acceleration library.

## Description

- Operator function: Supports vector interpolation. It is mainly used for channel estimation of data symbols or interpolation of equalization coefficients.
- Calculation formula:
    $$
    y[b,i] = \sum_{j=0}^{L-1} \text{coefficient}[b,i,j] \cdot x[b, \text{base}[b,i] + j]
    $$

    where:
    - $x$: input complex signal (batched)
    - $\text{coefficient}$: interpolation coefficient matrix (batched)
    - $y$: output complex signal (batched)
    - $b = 0, 1, \dots, B-1$ is the batch index
    - $i = 0, 1, \dots, M-1$ is the output position index
    - $j = 0, 1, \dots, L-1$ is the coefficient index

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

    Special notes:
    - The preceding compilation method supports only the acceleration library downloaded via git. It does not support compiling the acceleration library downloaded as a zip archive.
    - The compilation process downloads dependency libraries over the network. Therefore, the compilation environment requires Internet access.
    - The compilation process includes two steps: obtaining and compiling the ascend-boost-comm (Ascend Distributed Communication Acceleration Library) component, and compiling the signal acceleration library. For more command descriptions, see the [build.sh](../../../../build.sh) file in the SiP repository.

 - For more compilation command descriptions, refer to [Compilation and Build](../../../../docs/compilation_build_en.md)

### Running the Demo

- Go to the `example` directory and run the build script.

    ```sh
    cd  ${*example_directory*}
    bash build.sh
    ```

## Additional Notes

The data generated in the example does not represent actual scenarios. You can modify the data according to the specific use case.

## Product Support

This example applies to Atlas A2/A3 training products, Atlas 800I A2 inference product, and Atlas A3 inference products.
