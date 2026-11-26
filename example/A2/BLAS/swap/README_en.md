# Signal Processing Acceleration Library SwapOperation C++ Demo

## Introduction

This directory contains C++ call examples of the SwapOperation operator in the signal processing acceleration library.

## Description

- Operator function: Swaps two real (complex) vectors
- Calculation formula:
    $$
    \begin{aligned}
    \mathbf{x} &\leftrightarrow \mathbf{y} \\
    x_i &\leftrightarrow y_i \quad \text{for } i = 1, 2, \dots, n
    \end{aligned}
    $$

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

## Scenario Description

The example code provided corresponds to different scenarios. When compiling and running, change the build script according to the specific scenario:

- **example_sswap.cpp**

    Note: The default build script compiles and runs this example.

    **Input**

    | TensorName |  DataType  | DataFormat |      Shape      |
    | :--------: | :--------: | :--------: | :-------------: |
    |     x      |  float32   |     nd     |       [n]       |
    |     y      |  float32   |     nd     |       [n]       |

    ---

- **example_cswap.cpp**

    Note: After replacing example_sswap.cpp with example_cswap.cpp in the build script, the modified build script can compile and run the example.

    **Input**

    | TensorName |   DataType   | DataFormat |      Shape      |
    | :--------: | :----------: | :--------: | :-------------: |
    |     x      |  complex64   |     nd     |       [n]       |
    |     y      |  complex64   |     nd     |       [n]       |
