# Signal Processing Acceleration Library CalOperation C++ Demo

## Introduction

This directory contains C++ call examples of the CalOperation operator in the signal processing acceleration library.

## Description

- Operator function:

    - Sscal: Scales a real vector, that is, multiplies each element in the vector by a real number alpha.
    - Cscal: Scales a complex vector, that is, multiplies each element in the vector by a complex number alpha.
    - Csscal: Scales a complex vector, that is, multiplies each element in the vector by a real number alpha.

- Calculation formula:

    - Formula of Sscal
    $$x_i = \alpha \cdot x_i \quad \text{for } i = 0, 1, \dots, n-1$$
    - Formula of Csscal
    $$\begin{aligned}
    \text{Re}(x_{\text{out},i}) &= \alpha \cdot \text{Re}(x_{\text{in},i}) \\
    \text{Im}(x_{\text{out},i}) &= \alpha \cdot \text{Im}(x_{\text{in},i})
    \end{aligned} \quad \text{for } i = 0, 1, \dots, n-1$$
    - Formula of Cscal:
    $$\begin{aligned}
    \text{Re}(x_{\text{out},i}) &= \text{Re}(\alpha)\text{Re}(x_{\text{in},i}) - \text{Im}(\alpha)\text{Im}(x_{\text{in},i}) \\
    \text{Im}(x_{\text{out},i}) &= \text{Re}(\alpha)\text{Im}(x_{\text{in},i}) + \text{Im}(\alpha)\text{Re}(x_{\text{in},i})
    \end{aligned} \quad \text{for } i = 0, 1, \dots, n-1$$

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

- **example_sscal.cpp**

    Note: The default build script compiles and runs this example.

    **Input**

    | TensorName |  DataType  | DataFormat |      Shape      |
    | :--------: | :--------: | :--------: | :-------------: |
    |     x      |  float32   |     nd     |       [n]       |
    |   alpha    |  float32   |     nd     |       [1]       |

    ---

- **example_csscal.cpp**

    Note: After replacing example_sscal.cpp with example_csscal.cpp in the build script, the modified build script can compile and run the example.

    **Input**

    | TensorName |   DataType   | DataFormat |      Shape      |
    | :--------: | :----------: | :--------: | :-------------: |
    |     x      |  complex64   |     nd     |       [n]       |
    |   alpha    |   float32    |     nd     |       [1]       |

    ---

- **example_cscal.cpp**

    Note: After replacing example_sscal.cpp with example_cscal.cpp in the build script, the modified build script can compile and run the example.

    **Input**

    | TensorName |   DataType   | DataFormat |      Shape      |
    | :--------: | :----------: | :--------: | :-------------: |
    |     x      |  complex64   |     nd     |       [n]       |
    |   alpha    |  complex64   |     nd     |       [1]       |
