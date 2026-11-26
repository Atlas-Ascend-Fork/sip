# Signal Processing Acceleration Library AsdConvolveOperation C++ Demo

## Introduction

This directory contains C++ call examples of the AsdConvolveOperation operator in the signal processing acceleration library.

## Description

- Operator function: Performs one-dimensional filtering on the given signal.
- Calculation formula:
    $$
    w[k] = \sum_{j} u[j] \cdot v[k - j + 1]
    $$
    where w[k] is the element at position k of the output, u[j] is the one-dimensional signal at input position j, and v[k-j+1] is the filtering convolution kernel at position k-j+1. The one-dimensional signal is a complex vector, and the filtering convolution kernel is a real vector.

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

- **example_asd_convolve_complex32.cpp**

    Note: The default build script compiles and runs this example.

    **Input**

    | TensorName |  DataType  | DataFormat |      Shape      |
    | :--------: | :--------: | :--------: | :-------------: |
    |   signal   | complex32  |     nd     | [BatchCount, n] |

    **Output**

    | TensorName |  DataType  | DataFormat |      Shape      |
    | :--------: | :--------: | :--------: | :-------------: |
    |   output   | complex32  |     nd     | [BatchCount, n] |

    ---

- **example_asd_convolve_complex64.cpp**

    Note: After replacing example_asd_convolve_complex32.cpp with example_asd_convolve_complex64.cpp in the build script, the modified build script can compile and run the example.

    **Input**

    | TensorName |  DataType  | DataFormat |      Shape      |
    | :--------: | :--------: | :--------: | :-------------: |
    |   signal   | complex64  |     nd     | [BatchCount, n] |

    **Output**

    | TensorName |  DataType  | DataFormat |      Shape      |
    | :--------: | :--------: | :--------: | :-------------: |
    |   output   | complex64  |     nd     | [BatchCount, n] |
