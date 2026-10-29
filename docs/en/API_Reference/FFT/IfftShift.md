# IfftShift

## Applicable Product

|Product             |  Supported  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     ✓  |
|  <term>Atlas A3 training products/Atlas A3 inference products</term>   |     ×    |
|  <term>Atlas A2 training products/Atlas A2 inference products</term>     |     ×    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>    |     ×    |
|  <term>Atlas training products</term>    |     ×    |

## Function Description

- API function:\
  `asdIfftShift`: The inverse of `asdFftShift`. Shifts the zero-frequency component of batched 2D complex slices from the slice center back to the corner (ifftshift, aligned with the reference CUDA `ifftShiftScaleBatch2DKernel`). It also applies an element-wise `scale` multiplier so that the inverse-FFT normalization `1/(row*col)` can be folded into this pass. It is typically used for spectrum re-arrangement before/after 2D FFT, e.g. in MRI reconstruction pipelines.
- Formula:

  $$
  output[s][(r + row / 2) \bmod row][(c + col / 2) \bmod col] = scale \times input[s][r][c]
  $$

  Where:
  - $input[s]$ is the $s$-th input 2D complex slice (an $row \times col$ complex matrix); slice $s$ starts at element offset $s \cdot dist$ in the input.
  - $output[s]$ is the $s$-th output 2D complex slice; the output is a contiguous $batch \times row \times col$ layout.
  - $dist \le 0$ means the input slices are contiguous ($dist = row \cdot col$).
  - $scale$ is an element-wise output multiplier (applied to the real/imaginary parts separately in float32 inside the kernel). For even dimensions the permutation coincides with `asdFftShift`; for odd dimensions the two differ by one (floor vs. ceil) and are exact inverses of each other (round-trip identity).
  - With $scale = 1$ the operator is a pure data permutation: the output is binary-identical to the reference implementation, with no floating-point error.

## Function Prototype

```Cpp
AspbStatus asdIfftShift(
  int64_t               row,
  int64_t               col,
  int64_t               batch,
  int64_t               dist,
  double                scale,
  const aclTensor *     input,
  aclTensor *           output,
  void *                stream,
  void *                workspace = nullptr)
```

## asdIfftShift

- **Parameter description:**

  <table style="undefined;table-layout: fixed; width: 880px"><colgroup>
    <col style="width: 250px">
    <col style="width: 120px">
    <col style="width: 510px">
  </colgroup>
  <thead>
      <tr>
        <th>Parameter</th>
        <th>Input/Output</th>
        <th>Description</th>
      </tr></thead>
  <tbody>
    <tr>
      <td>row (int64_t)</td>
      <td>Input</td>
      <td>Number of rows of each slice. Must be positive.</td>
    </tr>
    <tr>
      <td>col (int64_t)</td>
      <td>Input</td>
      <td>Number of columns of each slice. Must be positive.</td>
    </tr>
    <tr>
      <td>batch (int64_t)</td>
      <td>Input</td>
      <td>Number of slices. Must be positive.</td>
    </tr>
    <tr>
      <td>dist (int64_t)</td>
      <td>Input</td>
      <td>Element stride between adjacent input slices. <code>dist &le; 0</code> means the input slices are contiguous (stride <code>row * col</code>); when <code>dist &gt; 0</code>, it must satisfy <code>dist &ge; row * col</code> (padding between slices).</td>
    </tr>
    <tr>
      <td>scale (double)</td>
      <td>Input</td>
      <td>Element-wise output multiplier, converted to <code>float32</code> in the kernel and applied to the real/imaginary part of every element. A typical value is <code>1.0 / (row * col)</code> (folds the inverse-FFT normalization into the pass, saving a separate scaling round-trip); <code>scale = 1</code> leaves the values bit-exact unchanged (pure permutation).</td>
    </tr>
    <tr>
      <td>input (aclTensor *)</td>
      <td>Input</td>
      <td>Input tensor, data type <code>complex64</code>. Contains <code>batch</code> complex slices of <code>row &times; col</code>; slice <code>s</code> starts at element offset <code>s * dist</code>. The total element count must be at least <code>(batch - 1) * dist + row * col</code>.</td>
    </tr>
    <tr>
      <td>output (aclTensor *)</td>
      <td>Output</td>
      <td>Output tensor, data type <code>complex64</code>, contiguous shape <code>[batch, row, col]</code>.</td>
    </tr>
    <tr>
      <td>stream (void *)</td>
      <td>Input</td>
      <td>Stream for executing the operator.</td>
    </tr>
    <tr>
      <td>workspace (void *)</td>
      <td>Input</td>
      <td>Reserved parameter. No workspace is required currently; the default value <code>nullptr</code> can be used.</td>
    </tr>
  </tbody>
  </table>

- **Return value:**

  A status code is returned. For details about the return values, see [SiP Return Codes](../../context/sip_return_codes.md).

## Constraints

- The current version supports 2D slices (`row` x `col`) only; support for other dimensions will be added in a later release.
- Only <code>complex64</code> is supported for the input and output tensors.
- <code>row</code>, <code>col</code> and <code>batch</code> must be positive; when <code>dist &gt; 0</code>, it must satisfy <code>dist &ge; row * col</code>.
- The input tensor and output tensor must not be the same tensor.
- <code>scale</code> is applied in <code>float32</code> inside the kernel; the output is binary-identical to a reference that performs the same <code>float32</code> multiplies.
- Measured on the typical MRI shape (<code>batch=12000, row=500, col=50</code>, 4.8 GB moved per call), the operator takes about 3.43 ms on Ascend 950PR (effective bandwidth about 1399 GB/s); with the <code>1/25000</code> normalization folded in (<code>scale = 1/25000</code>), about 4.73 ms (effective bandwidth about 1016 GB/s), still faster than running the shift and a separate scaling pass.

## Calling Example

The following sample code provides a minimal implementation for quick start, development, and debugging of the operator. Its core goal is to demonstrate the core functionality of the operator with the simplest code, rather than providing production-level security. Users are not recommended to directly use the sample code as service code. If users apply the sample code to their actual service scenarios and security issues occur, they shall bear the consequences themselves.

```Cpp
#include <cstdlib>
#include <iostream>
#include <vector>
#include "asdsip.h"
#include "acl/acl.h"
#include "aclnn/acl_meta.h"
using namespace AsdSip;

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

#define ASD_STATUS_CHECK(err)                            \
    do {                                                 \
        AsdSip::AspbStatus err_ = (err);                 \
        if (err_ != AsdSip::ErrorType::ACL_SUCCESS) {    \
            std::cout << "Execute failed." << std::endl; \
            return -1;                                   \
        }                                                \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    // Standard AscendCL initialization
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // Allocate device memory via aclrtMalloc
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // Copy host data to device via aclrtMemcpy
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Strides of a contiguous tensor
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // Create the aclTensor via aclCreateTensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main()
{
    int32_t deviceId = 0;
    // Read the actual device id from ASDSIP_DEVICE_ID first; default to 0 when unset
    const char* envDeviceId = std::getenv("ASDSIP_DEVICE_ID");
    if (envDeviceId != nullptr) {
        deviceId = std::atoi(envDeviceId);
    }
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // Host data: batch slices of row x col complex elements
    int64_t batch = 2, row = 4, col = 6;
    int64_t dist = 0; // 0 means the input slices are contiguous (dist = row * col)
    // scale = 1/(row*col): demonstrates folding the inverse-FFT normalization
    // into the shift pass; scale = 1.0 gives the pure inverse permutation
    const double scale = 1.0 / static_cast<double>(row * col);
    const float scale32 = static_cast<float>(scale);
    const int64_t tensorSize = batch * row * col;
    std::vector<int64_t> inShape = {batch, row, col};
    std::vector<std::complex<float>> inputHostData(tensorSize, std::complex<float>(0, 0));
    for (int64_t i = 0; i < tensorSize; i++) {
        inputHostData[i] = std::complex<float>(static_cast<float>(i), static_cast<float>(i) + 1.0f);
    }
    std::vector<std::complex<float>> outHostData(tensorSize, std::complex<float>(0, 0));
    void* inputDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* input = nullptr;
    aclTensor* out = nullptr;
    ret = CreateAclTensor(inputHostData, inShape, &inputDeviceAddr, aclDataType::ACL_COMPLEX64, &input);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);
    ret = CreateAclTensor(outHostData, inShape, &outDeviceAddr, aclDataType::ACL_COMPLEX64, &out);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);

    // Run IfftShift: output[s][(r + row/2) % row][(c + col/2) % col] = scale * input[s][r][c]
    ASD_STATUS_CHECK(asdIfftShift(row, col, batch, dist, scale, input, out, stream));
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    std::vector<std::complex<float>> outData(tensorSize, std::complex<float>(0, 0));
    ret = aclrtMemcpy(outData.data(), tensorSize * sizeof(outData[0]), outDeviceAddr, tensorSize * sizeof(outData[0]),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Host golden: inverse permutation + float32 real/imag scaling, element-wise binary comparison
    const int64_t shiftR = row / 2;
    const int64_t shiftC = col / 2;
    int64_t diff = 0;
    for (int64_t s = 0; s < batch; s++) {
        for (int64_t r = 0; r < row; r++) {
            for (int64_t c = 0; c < col; c++) {
                auto& srcv = inputHostData[s * row * col + r * col + c];
                auto expect = std::complex<float>(srcv.real() * scale32, srcv.imag() * scale32);
                auto got = outData[s * row * col + (r + shiftR) % row * col + (c + shiftC) % col];
                if (expect != got) {
                    diff++;
                }
            }
        }
    }
    std::cout << "IfftShift result: " << (diff == 0 ? "PASS" : "FAIL") << " (";
    std::cout << tensorSize - diff << "/" << tensorSize << " elements exact)" << std::endl;
    // Print the first 8 values of the output tensor
    for (int64_t i = 0; i < 8; i++) {
        std::cout << outData[i] << "\t";
    }
    std::cout << "\nend result" << std::endl;
    std::cout << "Execute successfully." << std::endl;

    aclDestroyTensor(input);
    aclDestroyTensor(out);
    aclrtFree(inputDeviceAddr);
    aclrtFree(outDeviceAddr);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
