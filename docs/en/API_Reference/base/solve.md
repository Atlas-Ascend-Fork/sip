# solve

## Applicable Product

|Product             |  Supported  |
|:-------------------------|:----------:|
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>    |     ×    |
|  <term>Atlas training products</term>    |     ×    |
|  <term>Atlas A3 training products/Atlas A3 inference products</term>   |     √    |
|  <term>Atlas A2 training products/Atlas A2 inference products</term>     |     √    |
|  <term>Ascend 950PR/Ascend 950DT</term>   |     ×    |

## Function Description

- API function:\
`solveGetWorkspaceSize`: computes the workspace size required by the `solve` operator.\
`solve`: solves the linear system A·X = B for a batch of single-precision real square matrices.
- Formula:

  $$
  A_b X_b = B_b \quad\Rightarrow\quad X_b = A_b^{-1} B_b
  $$

  Where: *b* is the batch index, *A_b* is a nonsingular n-by-n square matrix, *B_b* is an n-by-nrhs right-hand-side matrix, and *X_b* is the solution.

Example (batch=2, n=3, nrhs=2):\
The input `A` is:\
[[[4, 1, 0],\
[1, 3, 1],\
[0, 1, 2]],\
[[4, 1, 0],\
[1, 3, 1],\
[0, 1, 2]]]\
The input `B` is:\
[[[7, 12],\
[15, 20],\
[13, 16]],\
[[9, 6],\
[8, 11],\
[7, 10]]]\
After the `solve` operator is called, the output `X` is:\
[[[1, 2],\
[3, 4],\
[5, 6]],\
[[2, 1],\
[1, 2],\
[3, 4]]]

## Function Prototype

To use the `solve` operator, the `solveGetWorkspaceSize` API must be called first to obtain the required workspace size, and then the `solve` API is called to perform the computation.

```Cpp
AsdSip::AspbStatus solveGetWorkspaceSize(
  size_t &size)
```

```Cpp
AsdSip::AspbStatus solve(
  const aclTensor *    a,
  const aclTensor *    b,
  aclTensor *          x,
  void *               stream,
  void *               workspace = nullptr)
```

- **Return value:**

  For details about the return values, see [SiP Return Codes](../../context/sip_return_codes.md).

## solveGetWorkspaceSize

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
      <td>size (size_t &amp;)</td>
      <td>Output</td>
      <td>Workspace size required by the `solve` operator, in bytes. In the current version it is always 0, so `workspace` can be set to `nullptr` when calling `solve`.</td>
    </tr>
  </tbody>
  </table>

- **Return value:**

  For details about the return values, see [SiP Return Codes](../../context/sip_return_codes.md).

## `solve`

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
      <td>a (const aclTensor *)</td>
      <td>Input</td>
      <td><ul><li>Coefficient matrices, corresponding to 'A' in the formula.</li><li>Only FLOAT32 is supported. The data format must be ND.</li><li>The shape is [batch, n, n], where n∈[2, 256].</li><li>The matrices must be nonsingular.</li></ul></td>
    </tr>
    <tr>
      <td>b (const aclTensor *)</td>
      <td>Input</td>
      <td><ul><li>Right-hand-side matrices, corresponding to 'B' in the formula.</li><li>Only FLOAT32 is supported and must be the same as that of `a`.</li><li>The shape is [batch, n, nrhs], where nrhs≥1.</li><li>The batch dimension must be the same as that of `a` (broadcasting is not supported; expand it in the caller).</li></ul></td>
    </tr>
    <tr>
      <td>x (aclTensor *)</td>
      <td>Output</td>
      <td><ul><li>Solution matrices, corresponding to 'X' in the formula.</li><li>The data type and shape are the same as those of `b`.</li></ul></td>
    </tr>
    <tr>
      <td>workspace (void *)</td>
      <td>Input</td>
      <td>Workspace required by the `solve` operator. In the current version it can be set to `nullptr` (size is 0).</td>
    </tr>
    <tr>
      <td>stream (void *)</td>
      <td>Input</td>
      <td>NPU execution stream.</td>
    </tr>
  </tbody>
  </table>

- **Return value:**

  For details about the return values, see [SiP Return Codes](../../context/sip_return_codes.md).

## Constraints

  - The input and output tensors must be 3-D in ND format and stored contiguously in row-major order (non-contiguous strides are not supported).
  - Only FLOAT32 is supported.
  - The matrix order n must be in [2, 256]; batch≥1; the number of right-hand-side columns nrhs≥1.
  - Batch broadcasting is not supported: the batch dimension of `b` must be the same as that of `a`.
  - `a` must be nonsingular. The operator does not detect singular matrices: results for singular matrices are undefined (consistent with LAPACK), and the accuracy for nearly singular matrices degrades as the condition number grows. The caller must guarantee that the input is nonsingular.
  - The workspace size is 0, so `nullptr` can be passed; for the large-matrix path (n>64), the required temporary space is managed internally by the operator.

## Calling Example

The example code is as follows. This sample is intended to provide a minimal implementation for quick start, development, and debugging of the operator. Its core goal is to demonstrate the core functionality of the operator using the simplest code, rather than providing production-grade security assurance. Users are advised not to directly use the example code for business purposes. If users apply the example code in their own real business scenarios and security issues occur, the users shall bear the consequences themselves.

```Cpp
#include <iostream>
#include <vector>
#include "asdsip.h"
#include "acl/acl.h"
#include "acl_meta.h"

using namespace AsdSip;

#define ASD_STATUS_CHECK(err)                            \
    do {                                                 \
        AsdSip::AspbStatus err_ = (err);                 \
        if (err_ != AsdSip::ErrorType::ACL_SUCCESS) {    \
            std::cout << "Execute failed." << std::endl; \
            exit(-1);                                    \
        }                                                \
    } while (0)

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

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream *stream)
{
    // Fixed boilerplate: acl initialization
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // Allocate device memory
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // Copy host data to device memory
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Compute strides of the contiguous tensor
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // Create the aclTensor
    *tensor = aclCreateTensor(shape.data(),
        shape.size(),
        dataType,
        strides.data(),
        0,
        aclFormat::ACL_FORMAT_ND,
        shape.data(),
        shape.size(),
        *deviceAddr);
    return 0;
}

void printTensor(const float *tensorData, int64_t nums)
{
    for (int64_t i = 0; i < nums; i++) {
        std::cout << tensorData[i] << " ";
    }
    std::cout << std::endl;
}

int main(int argc, char **argv)
{
    int deviceId = 0;

    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    int64_t batch = 2;
    int64_t n = 3;
    int64_t nrhs = 2;

    // A = [[4,1,0],[1,3,1],[0,1,2]] (the same for both batches)
    std::vector<float> tensorInAData = {4.0f, 1.0f, 0.0f, 1.0f, 3.0f, 1.0f, 0.0f, 1.0f, 2.0f,
                                        4.0f, 1.0f, 0.0f, 1.0f, 3.0f, 1.0f, 0.0f, 1.0f, 2.0f};
    // B is derived from known solutions: X0 = [[1,2],[3,4],[5,6]], X1 = [[2,1],[1,2],[3,4]]
    std::vector<float> tensorInBData = {7.0f, 12.0f, 15.0f, 20.0f, 13.0f, 16.0f,
                                        9.0f, 6.0f, 8.0f, 11.0f, 7.0f, 10.0f};
    std::vector<float> tensorOutXData(batch * n * nrhs, 0.0f);

    std::vector<int64_t> aShape = {batch, n, n};
    std::vector<int64_t> bShape = {batch, n, nrhs};
    std::vector<int64_t> xShape = {batch, n, nrhs};

    aclTensor *inputA = nullptr;
    aclTensor *inputB = nullptr;
    aclTensor *outputX = nullptr;
    void *inputADeviceAddr = nullptr;
    void *inputBDeviceAddr = nullptr;
    void *outputXDeviceAddr = nullptr;
    ret = CreateAclTensor(tensorInAData, aShape, &inputADeviceAddr, aclDataType::ACL_FLOAT, &inputA);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);
    ret = CreateAclTensor(tensorInBData, bShape, &inputBDeviceAddr, aclDataType::ACL_FLOAT, &inputB);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);
    ret = CreateAclTensor(tensorOutXData, xShape, &outputXDeviceAddr, aclDataType::ACL_FLOAT, &outputX);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);

    size_t workspaceSize = 0;
    ASD_STATUS_CHECK(solveGetWorkspaceSize(workspaceSize));
    ASD_STATUS_CHECK(solve(inputA, inputB, outputX, stream, nullptr));

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    ret = aclrtMemcpy(tensorOutXData.data(),
        GetShapeSize(xShape) * sizeof(float),
        outputXDeviceAddr,
        GetShapeSize(xShape) * sizeof(float),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("copy x from device to host failed. ERROR: %d\n", ret); return ret);
    std::cout << "------- output X -------" << std::endl;
    printTensor(tensorOutXData.data(), GetShapeSize(xShape));
    std::cout << "Execute successfully." << std::endl;

    aclDestroyTensor(inputA);
    aclDestroyTensor(inputB);
    aclDestroyTensor(outputX);
    aclrtFree(inputADeviceAddr);
    aclrtFree(inputBDeviceAddr);
    aclrtFree(outputXDeviceAddr);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
