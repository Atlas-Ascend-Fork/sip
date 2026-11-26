# Signal Processing Acceleration Library FFT3DOperation C++ Demo

## Introduction

This directory contains C++ call examples of the FFT3DOperation operator in the signal processing acceleration library.

## Description

- Operator function: Performs a batched 3D fast Fourier transform.
- Calculation formula:
    Given a 3D discrete signal:

     ![Formula](../../../../docs/en/API_Reference/figures/FFT_3D_1.png)

    Its 3D discrete Fourier transform is defined as:\
      ![Formula](../../../../docs/en/API_Reference/figures/FFT_3D_2.png)

    where:\
     ![Formula](../../../../docs/en/API_Reference/figures/FFT_3D_3.png)

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

Note: The following input/output tables list only tensors. The FFT type is a host-side `asdFftType` enumeration parameter set by `asdFftMakePlan3D`; it is not a tensor input.

- **example_acl_fft_c2c_3d.cpp**

    Note: The default build script compiles and runs this example.

    **Input**

    | TensorName |   DataType  | DataFormat |          Shape          |       Value     |
    | :--------: | :---------: | :--------: | :---------------------: | :-------------: |
    |   input    |  complex64  |     nd     |  [batch, Nfft1, Nfft2, Nfft3]  |       -         |

    **Output**

    | TensorName |  DataType  | DataFormat |           Shape           |       Value     |
    | :--------: | :--------: | :--------: | :-----------------------: | :-------------: |
    |     out    |  complex64 |     nd     |   [batch, Nfft1, Nfft2, Nfft3]   |         -       |

    ---

- **example_acl_fft_c2r_3d.cpp**

    Note: After replacing example_acl_fft_c2c_3d.cpp with example_acl_fft_c2r_3d.cpp in the build script, the modified build script can compile and run the example.

    **Input**

    | TensorName |   DataType  | DataFormat |              Shape              |       Value     |
    | :--------: | :---------: | :--------: | :-----------------------------: | :-------------: |
    |   input    |  complex64  |     nd     |  [batch, Nfft1, Nfft2 , Nfft3 / 2 + 1]  |       -         |

    **Output**

    | TensorName |  DataType  | DataFormat |           Shape           |       Value     |
    | :--------: | :--------: | :--------: | :-----------------------: | :-------------: |
    |     out    |   float32   |     nd     |   [batch, Nfft1, Nfft2, Nfft3]   |         -       |

     ---

- **example_acl_fft_r2c_3d.cpp**

    Note: After replacing example_acl_fft_c2c_3d.cpp with example_acl_fft_r2c_3d.cpp in the build script, the modified build script can compile and run the example.

    **Input**

    | TensorName |   DataType   | DataFormat |           Shape           |       Value     |
    | :--------: | :----------: | :--------: | :-----------------------: | :-------------: |
    |    input   |    float32    |     nd     |   [batch, Nfft1, Nfft2, Nfft3]   |         -       |

    **Output**

    | TensorName |   DataType  | DataFormat |              Shape              |       Value     |
    | :--------: | :---------: | :--------: | :-----------------------------: | :-------------: |
    |    out     |  complex64  |     nd     |  [batch, Nfft1, Nfft2, Nfft3 / 2 + 1]  |       -         |

- **example_acl_fft_c2c_3d_sep.cpp**

    Note: After replacing example_acl_fft_c2c_3d.cpp with example_acl_fft_c2c_3d_sep.cpp in the build script, the modified build script can compile and run the example.

    **Input**

    | TensorName |   DataType   | DataFormat |           Shape           |       Value     |
    | :--------: | :----------: | :--------: | :-----------------------: | :-------------: |
    |    input   |    float32    |     nd     |   [batch, Nfft1, Nfft2, Nfft3]   |         -       |

    **Output**

    | TensorName |   DataType  | DataFormat |              Shape              |       Value     |
    | :--------: | :---------: | :--------: | :-----------------------------: | :-------------: |
    |    out     |  complex64  |     nd     |  [batch, Nfft1, Nfft2, Nfft3]  |       -         |
