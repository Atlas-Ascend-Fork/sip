# torch_sip - PyTorch Extension for AscendSiP

This is a PyTorch extension that provides Python bindings for AscendSiP operations using Torch Library (TORCH_LIBRARY).

## Directory Structure

```txt
sip_pta/
├── torch_sip/
│   └── __init__.py          # Python interface
├── csrc/
│   └── base
│       └── asd_mul.cpp      # C++ implementation and TORCH_LIBRARY registration
├── test/
│   └── base
│        └── asd_mul.py      # Python interface test script
├── setup.py                 # Build script (invoked by build.sh)
└── build.sh                 # One-click build script (environment self-check + automatic dependency pulling + build + optional installation)
```

## Build Method

### Prerequisites

- C++ compiler (g++, gcc >= 10.3.0)
- Python >= 3.8
- PyTorch with NPU support (torch_npu must be installed)
- Ascend CANN toolkit >= 8.5.0 (its set_env.sh must be sourced)
- git (used to automatically pull the build dependency ascend-boost-comm)

### One-Click Build (Recommended)

```bash
conda activate your_env            # torch + torch_npu must be installed
source <*CANN installation directory*>/set_env.sh   # Set the CANN environment
cd sip_pta
bash build.sh --install            # Build and install the whl into the current environment
```

`build.sh` automatically completes the following steps:

1. Environment self-check (python3 / torch / torch_npu / CANN; if `ASCEND_HOME_PATH` is not set, common paths are automatically attempted to be sourced)
2. Check the source of the MKI header files (`BOOST_COMM_PATH`, default `../3rdparty/ascend-boost-comm/src/include`). If it does not exist, ascend-boost-comm is automatically cloned with `git clone` into the repository `3rdparty/` directory — **no need to manually clone the sip repository dependencies**
3. Run `python3 setup.py build bdist_wheel` to generate `dist/torch_sip-<version>-<python_tag>-<platform_tag>.whl`
4. `--install` option: install over the existing installation into the current Python environment

Other options:

```bash
bash build.sh                # Build the whl only, without installing
bash build.sh --clean        # Clean build artifacts
```

### Manual Build (Equivalent Method)

<details>
<summary>For fine-grained control, build manually by following these steps</summary>

Configure the environment (required only for building the whl; for runtime, see the "Script Running Method" below)

   1. conda environment activation: conda activate your_env; torch_npu must be installed
   2. CANN configuration: for example, source /opt/xxx/ascend-toolkit/set_env.sh
   3. Environment variable: export ASDSIP_HOME_PATH=asdsip installation directory. If not configured, the default is /usr/local/Ascend/asdsip/latest (you can also directly source the set_env.sh in the asdsip installation directory, which is equivalent)
   4. Environment variable: export BOOST_COMM_PATH=ascend-boost-comm header file directory (MKI header files, not included in the asdsip package). If not configured, the default is ../3rdparty/ascend-boost-comm/src/include; if that directory does not exist (the sip repository build.sh has not been run), first run git clone https://gitcode.com/cann/ascend-boost-comm.git into ../3rdparty/
   5. Environment variable: export USE_NINJA=1. This is an optional configuration that controls whether to enable ninja build

```bash
cd sip_pta
python3 setup.py build bdist_wheel
pip install --force-reinstall --no-deps ./dist/torch_sip-<version>-<python_tag>-<platform_tag>.whl
```

</details>

## Usage

> **Note**: The `asd_mul` of the current asdsip version supports only `complex64` complex input, not real number types such as float32. Real number input reports `ERR99999 UNKNOWN application exception` on the device side.

```python
import torch
import torch_sip

# Create complex tensors on the NPU
x = torch.complex(torch.randn(10, 10).npu(), torch.randn(10, 10).npu())
y = torch.complex(torch.randn(10, 10).npu(), torch.randn(10, 10).npu())

# Call the asd_mul operation
result = torch_sip.asd_mul(x, y)

print(result)  # Complex element-wise multiplication result
```

### Script Running Method (Minimal Environment Combination, Verified by Actual Testing)

**Prerequisites**: The scripts in this section depend on the asdsip dynamic library and the MKI header file directory. Confirm that the following two steps are completed first; otherwise, errors about missing libraries/header files occur during running (issue #201):

1. **Install the asdsip package**: Run `bash build.sh` in the sip repository root directory and install the artifact `output/Ascend-cann-SIP_*.run` (or use the existing `/usr/local/Ascend/asdsip` installation directory);
2. **Generate the 3rdparty directory**: After `bash build.sh` has been run once in the sip repository root directory, `3rdparty/ascend-boost-comm` (the source of the MKI header files) is created automatically. If it is still missing, manually run
   `git clone https://gitcode.com/cann/ascend-boost-comm.git ../3rdparty/ascend-boost-comm` (the default value of `BOOST_COMM_PATH` relative to sip_pta).

Run the following commands from the root directory of the SiP source repository to configure the running environment and run the test script:

```bash
# 1. source the set_env.sh in the asdsip installation directory (automatically exports ASDSIP_HOME_PATH and adds lib to LD_LIBRARY_PATH)
source <*asdsip installation directory*>/set_env.sh        # For example, /usr/local/Ascend/asdsip/set_env.sh
# 2. Set the CANN environment
source <*CANN installation directory*>/set_env.sh          # For example, /usr/local/Ascend/ascend-toolkit/set_env.sh
# 3. Activate the environment
conda activate your_env
# Run the test script
python sip_pta/test/base/asd_mul.py
```

Note: The content of the asdsip `set_env.sh` is exactly `export ASDSIP_HOME_PATH=<*directory*>/latest/` and
`export LD_LIBRARY_PATH=$ASDSIP_HOME_PATH/lib:$LD_LIBRARY_PATH`. Manually running these two export commands is equivalent to sourcing the script; either option works.
The `BOOST_COMM_PATH` required at build time is not needed at runtime at all; `ASCEND_HOME_PATH`/`ASCEND_TOOLKIT_HOME` and others are automatically exported by the CANN `set_env.sh` and do not need to be set manually.

## Implementation Details

### Torch Library

This implementation uses **Torch Library** (TORCH_LIBRARY):

**Advantages:**

- Better integration with the PyTorch operator system
- Automatic dispatch to the NPU backend
- Can be used with torch.compile() and other PyTorch optimizations
- Simpler than pybind11 - no manual Python binding code is required

**Key Files:**

**base-type operators:**

1. **csrc/base/asd_mul.cpp** - Complete implementation

   ```cpp
   namespace sip_pta {
   at::Tensor asdMul(const at::Tensor& x, const at::Tensor& y)
   {
   c10_npu::NPUGuard guard(x.device());

       at::Tensor z = at::empty_like(x);

       auto sip_stream = c10_npu::getCurrentNPUStream().stream(false);
       int64_t n = x.numel();
       uint8_t* workspace_ptr = nullptr;
       EXEC_FUNC(AsdSip::asdMul, n, x, y, z, sip_stream, workspace_ptr);
       return z;
   }
   TORCH_LIBRARY_FRAGMENT(torch_sip, m) { m.def("asd_mul(Tensor x, Tensor y) -> Tensor"); }

   TORCH_LIBRARY_IMPL(torch_sip, PrivateUse1, m) { m.impl("asd_mul", &asdMul); }
   } // namespace sip_pta
   ```

2. **torch_sip/__init__.py** - Python interface

   ```python
   def asd_mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.ops.torch_sip.asd_mul(x, y)
   ```

**fft-type operators:** - Complete implementation

1. **csrc/fft/asd_fft_c2c.cpp** - Complete implementation

   ```cpp
   namespace sip_pta {
   at::Tensor asdFftC2C(const at::Tensor& in, bool isForward)
   {
       c10_npu::NPUGuard guard(in.device());

       at::Tensor input = in.is_contiguous() ? in : in.contiguous();
       at::Tensor output = at::empty_like(input);
       auto selfShape = input.sizes();
       TORCH_CHECK(selfShape.size() >= sip_pta::DIM_2 && selfShape.size() <= sip_pta::DIM_4,
                   "Input tensor must have 2, 3 or 4 dimensions, but got ", selfShape.size());
       op_api::FFTParam param; // Prepare FFTParam

       if (isForward) {
           param.direction = AsdSip::ASCEND_FFT_FORWARD;
       } else {
           param.direction = AsdSip::ASCEND_FFT_INVERSE;
       }
       param.batchSize = selfShape[0];
       param.fftType = AsdSip::ASCEND_FFT_C2C;
       param.dimType = AsdSip::asdFft1dDimType::ASCEND_FFT_HORIZONTAL;
       if (selfShape.size() == sip_pta::DIM_2) {
           // 1D
           param.fftXSize = selfShape[sip_pta::DIM_1];
       } else if (selfShape.size() == sip_pta::DIM_3) {
           // 2D
           param.fftXSize = selfShape[sip_pta::DIM_1];
           param.fftYSize = selfShape[sip_pta::DIM_2];
       } else if (selfShape.size() == sip_pta::DIM_4) {
           // 3D
           param.fftXSize = selfShape[sip_pta::DIM_1];
           param.fftYSize = selfShape[sip_pta::DIM_2];
           param.fftZSize = selfShape[sip_pta::DIM_3];
       }
       EXEC_FFT_FUNC(AsdSip::asdFftExecC2C, param, input, output);
       // Use the macro EXEC_FFT_FUNC. The parameters are, in order, the operator entry function, FFTParam, and the input parameters of the operator entry function (except handle, in strict order)
       return output;
   }

   // 2. Operator registration (Fragment)
   TORCH_LIBRARY_FRAGMENT(torch_sip, m)
   {
       m.def("asd_fft_c2c(Tensor input, bool is_forward) -> Tensor", &asdFftC2C);
   }
   TORCH_LIBRARY_IMPL(torch_sip, PrivateUse1, m) { m.impl("asd_fft_c2c", &asdFftC2C); }
   } // namespace sip_pta
   ```

2. **torch_sip/__init__.py** - Python interface

   ```python
   def asd_fft_c2c(input_tensor: torch.Tensor, is_forward: bool = True) -> torch.Tensor:
        return torch.ops.torch_sip.asd_fft_c2c(input_tensor, is_forward)
   ```

### C2C Separated Real and Imaginary Part Interface Constraints

`asd_fft_c2c_sep` is the separated C2C FFT interface provided by PTA. Unlike `asd_fft_c2c`, which receives a single complex
Tensor, it uses real-part and imaginary-part Tensors separately for input and output:

- `in_real` and `in_imag` must be `torch.float32` and must have exactly the same shape. This is consistent with
  the `FLOAT32` and `ND` contract of the direct C interface `asdFftExecC2CSeparated`.
- `is_forward` is a required Boolean parameter. Passing `True` performs the forward FFT, and passing `False` performs
  the inverse FFT. The Python wrapper function defaults to `True`, but the underlying `torch.ops` call must explicitly pass this parameter.
- The return value is `(out_real, out_imag)`. Both Tensors are `torch.float32` and have the same shape as
  the corresponding inputs.
- PTA copies non-contiguous inputs into contiguous Tensors before calling the underlying interface. Therefore, the caller can pass
  non-contiguous layouts, but the returned output is a newly allocated contiguous Tensor.

```python
real = torch.randn(16, 128, dtype=torch.float32, device="npu")
imag = torch.randn(16, 128, dtype=torch.float32, device="npu")
out_real, out_imag = torch.ops.torch_sip.asd_fft_c2c_sep(real, imag, True)
```

**blas-type operators:** - Complete implementation

1. **csrc/blas/asd_blas_ctrmv.cpp** - Complete implementation

   ```cpp
   namespace sip_pta {
   at::Tensor asdBlasCtrmv(const at::Tensor& A, at::Tensor& x, int64_t uplo, int64_t trans, int64_t diag)
   {
   // Basic data type check
   // According to the operator definition, currently only ComplexFloat single-precision complex is supported
   TORCH_CHECK(A.scalar_type() == at::kComplexFloat && x.scalar_type() == at::kComplexFloat,
   "asd_blas_ctrmv: Tensors must be ComplexFloat (Complex64).");

   // Dimension check
   // The matrix must be a square matrix, and the vector length must match the matrix dimension
   TORCH_CHECK(A.dim() == sip_pta::DIM_2 && A.size(0) == A.size(1),
   "asd_blas_ctrmv: Tensor A must be a 2D square matrix.");
   TORCH_CHECK(x.dim() == sip_pta::DIM_1 && x.size(0) == A.size(0),
   "asd_blas_ctrmv: Tensor x length must match matrix A dimension.");

   c10_npu::NPUGuard guard(A.device());
   // Extract tensor dimensions and attribute parameters
   int64_t n = A.size(0);
   int64_t lda = n;
   int64_t incx = 1;

   // Convert enumeration parameters to underlying types
   AsdSip::asdBlasFillMode_t uplo_val = static_cast<AsdSip::asdBlasFillMode_t>(uplo);
   AsdSip::asdBlasOperation_t trans_val = static_cast<AsdSip::asdBlasOperation_t>(trans);
   AsdSip::asdBlasDiagType_t diag_val = static_cast<AsdSip::asdBlasDiagType_t>(diag);

   // Build the Plan parameter list passed to getBlasHandle; add all parameters of AsdSip::asdBlasMakeCtrmvPlan except handle in order
   std::vector<int64_t> planParam = {static_cast<int64_t>(uplo_val), n};

   // Define the Plan building function
   auto makePlan = [uplo_val, n](AsdSip::asdBlasHandle handle) {
   AsdSip::asdBlasMakeCtrmvPlan(handle, uplo_val, n);
   };

   // Execute the underlying operator,
   EXEC_BLAS_FUNC(AsdSip::asdBlasCtrmv, makePlan, planParam,
   uplo_val, trans_val, diag_val, n, A, lda, x, incx);
   // EXEC_BLAS_FUNC. The parameters are, in order, the operator entry function, the makePlan function, planParam, and the input parameters of the operator entry function (except handle, in strict order)
   // CTRMV is an in-place modification operator; directly return the modified x
   return x;
   }

   // Operator registration module
   TORCH_LIBRARY_FRAGMENT(torch_sip, m)
   {
   // x is updated in place; Tensor(a!) is used to mark the mutable reference
   m.def("asd_blas_ctrmv(Tensor A, Tensor(a!) x, int uplo, int trans, int diag) -> Tensor(a!)");
   }

   TORCH_LIBRARY_IMPL(torch_sip, PrivateUse1, m)
   {
   m.impl("asd_blas_ctrmv", &asdBlasCtrmv);
   }
   }
   ```

2. **torch_sip/__init__.py** - Python interface

   ```python
   def asd_blas_ctrmv(mat_a, vec_x, uplo="L", trans="N", diag="N"):
       """
       Args:
         mat_a: (N, N) complex triangular matrix
         vec_x: (N,) complex vector
         uplo: "L" (lower triangular), "U" (upper triangular)
         trans: "N" (no transpose), "T" (transpose), or "C" (conjugate transpose)
         diag: "N" (non-unit diagonal) or "U" (unit diagonal, with diagonal elements regarded as 1)
       """
       uplo_map = {"L": 0, "U": 1}
       trans_map = {"N": 0, "T": 1, "C": 2}
       diag_map = {"N": 0, "U": 1}

       u_val = uplo_map.get(uplo.upper(), 0)
       t_val = trans_map.get(trans.upper(), 0)
       d_val = diag_map.get(diag.upper(), 0)

       return torch.ops.torch_sip.asd_blas_ctrmv(mat_a, vec_x, u_val, t_val, d_val)
   ```

## Adding a New Operator

To add a new operator (for example, `asd_add`):

1. **Add it to csrc/asd_mul.cpp (or create a new file):**

   ```cpp
   at::Tensor asd_add(const at::Tensor &x, const at::Tensor &y) {
       // Implementation code
   }

   TORCH_LIBRARY(TorchSip, m) {
       m.def("asd_add(Tensor x, Tensor y) -> Tensor");
   }

   TORCH_LIBRARY_IMPL(TorchSip, PrivateUse1, m) {
       m.impl("asd_add", &asd_add);
   }
   ```

2. **Add it to torch_sip/__init__.py:**

   ```python
   def asd_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
       return torch.ops.TorchSip.asd_add(x, y)
   ```

## Troubleshooting

### Build Errors

- **"torch/extension.h not found"**: Ensure that PyTorch with NPU support (torch_npu) is installed
- **"asdsip library not found"**: Set `ASDSIP_HOME_PATH` to the asdsip installation directory, or source its set_env.sh
- **"g++ not found"**: Install the g++ compiler

### Runtime Errors

- **"Device mismatch"**: Ensure that tensors are on NPU devices
- **"Shape mismatch"**: Check whether the input tensor shapes are compatible
- **"torch.ops.TorchSip not found"**: Ensure that the extension is built and loaded
- **"ImportError: libasdsip.so: cannot open shared object file"**: `LD_LIBRARY_PATH` does not include the asdsip lib directory.
  Source the set_env.sh in the asdsip installation directory, or manually run
  `export LD_LIBRARY_PATH=<*asdsip installation directory*>/lib:$LD_LIBRARY_PATH`
- **"call AsdSip::asdMul failed ... ERR99999"**: Check the input dtype. The current version of `asd_mul` supports only `complex64`,
  not real number types such as float32. You can set `ASCEND_LAUNCH_BLOCKING=1` for synchronous execution to obtain a more accurate error stack
