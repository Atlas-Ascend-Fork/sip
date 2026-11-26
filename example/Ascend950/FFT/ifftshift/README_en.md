# Signal Processing Acceleration Library IfftShiftOperation C++ Demo

## Introduction

This directory contains C++ call examples of the IfftShiftOperation operator in the signal processing acceleration library.

## Description

- Operator function: The inverse transform of asdFftShift. It moves the zero-frequency component of batched 2D complex slices from the center of each slice back to the corner (ifftshift, aligned with the semantics of the CUDA reference implementation ifftShiftScaleBatch2DKernel) and supports an element-wise scale multiplier.
- Calculation formula:
    $$
    output[s][(r + row / 2) \bmod row][(c + col / 2) \bmod col] = scale \times input[s][r][c]
    $$

    where:
    - $input[s]$ is the $s$-th input 2D complex slice (a $row \times col$ complex matrix), and the start element offset of the slice in the input is $s \cdot dist$;
    - $output[s]$ is the $s$-th output 2D complex slice, and the output is a contiguous $batch \times row \times col$ layout;
    - $dist \le 0$ indicates that the input slices are contiguous ($dist = row \cdot col$);
    - $scale$ is the element-wise output multiplier (multiplied element-wise by real part/imaginary part in float32 within the kernel), which can fold the 1/(row*col) normalization of the inverse FFT into this pass. When $scale = 1$, it is pure data permutation, with no floating-point error.

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

This example applies to Ascend 950PR&950DT products.

## Scenario Description

The example code provided corresponds to different scenarios. When compiling and running, change the build script according to the specific scenario:

- **example_acl_ifft_shift.cpp**

    Note: The default build script compiles and runs this example. The example uses contiguous input slices with batch=2, row=4, and col=6, and takes scale = 1/(row*col) to demonstrate folding the inverse FFT normalization into the shift pass (when scale = 1.0, it is a pure inverse permutation). After running, the results are compared binary-wise with the expected results computed element-wise on the host side, and PASS/FAIL is printed.

    **Input**

    | TensorName | DataType  | DataFormat |           Shape           | Value |
    | :--------: | :-------: | :--------: | :-----------------------: | :---: |
    |   input    | complex64 |     nd     |  [batch, row, col] (or a flat layout distributed with the dist step) |   -   |

    **Output**

    | TensorName | DataType  | DataFormat |           Shape           | Value |
    | :--------: | :-------: | :--------: | :-----------------------: | :---: |
    |     out    | complex64 |     nd     |       [batch, row, col]   |   -   |
