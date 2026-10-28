# Iamax

<!-- md-trans-meta sourceCommit=25a0c73916203ce4555696808917dc3c4c840c42 translatedAt=2026-10-10T12:18:53.341Z pushedAt=2026-10-14T04:00:26.083Z -->

## Applicable Product

<!-- npu="950" id1 -->
- <term>Ascend 950PR/Ascend 950DT products</term>: Not supported
<!-- end id1 -->

<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Supported
<!-- end id2 -->

<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Supported
<!-- end id3 -->

<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 inference products</term>: Not supported
<!-- end id4 -->

<!-- npu="310p" id5 -->
- <term>Atlas inference products</term>: Not supported
<!-- end id5 -->

<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Not supported
<!-- end id6 -->

## Function Description

- API function:\
`asdBlasMakeIamaxPlan`: Initializes the `Iamax` operator configuration corresponding to the handle.\
`asdBlasIsamax`: Finds the element with the maximum absolute value in a real vector and returns its index. If multiple elements are equal, the smallest index is returned.\
`asdBlasIcamax`: Finds the element with the maximum sum of absolute values of the real and imaginary parts in a complex vector and returns its index. If multiple elements have equal sums of absolute values, the index of the first such element found is returned.
- Formula:
  - Formula for `asdBlasIsamax`

  $$
  i_{max}=argmax_{i=1}^n(|(x_{i})| )
  $$

    Example:\
    The input `x` is:\
    [3, 4, 4,3]\
    After the `asdBlasIsamax` operator is called, the output `result` is:\
    2

  - Formula for `asdBlasIcamax`

  $$
  i_{max}=argmax_{i=1}^n(|Im(x_{i}) |+|Re(x_{i} )|)
  $$
    where `Re(∙)` denotes the real part of a complex number, and `Im(∙)` denotes the imaginary part of a complex number.\
    Example:\
    The input `x` is:\
    [3+4i,4-3i,4+2i,3-1i]\
    After the `asdBlasIcamax` operator is called, the output `result` is:\
    1

## Function Prototype

```Cpp
AspbStatus asdBlasMakeIamaxPlan(
  asdBlasHandle handle)
```

```Cpp
AspbStatus asdBlasIcamax(
  asdBlasHandle     handle,
  const int64_t     n,
  aclTensor *       x,
  const int64_t     incx,
  aclTensor *       result)
```

```Cpp
AspbStatus asdBlasIsamax(
  asdBlasHandle     handle,
  const int64_t     n,
  aclTensor *       x,
  const int64_t     incx,
  aclTensor *       result)
```

## asdBlasMakeIamaxPlan

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
      <td>handle (asdBlasHandle)</td>
      <td>Input</td>
      <td>Handle of the operator.</td>
    </tr>
    </tbody>
    </table>

- **Return value:**

  For details about the return values, see [SiP Return Codes](../../context/sip_return_codes.md).

## asdBlasIsamax & asdBlasIcamax

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
      <td>handle (asdBlasHandle)</td>
      <td>Input</td>
      <td>Handle of the operator.</td>
    </tr>
    <tr>
      <td>n (int64_t)</td>
      <td>Input</td>
      <td>Total number of elements.</td>
    </tr>
    <tr>
      <td>x (aclTensor *)</td>
      <td>Input</td>
      <td><ul><li>Corresponds to "x" in the formula.</li><li>Supported data type for <code>asdBlasIsamax</code>: <code>FLOAT32</code>.</li><li>Supported data type for <code>asdBlasIcamax</code>: <code>COMPLEX64</code>.</li><li>Data format: <code>ND</code>.</li><li>Shape: [n].</li></ul></td>
    </tr>
    <tr>
      <td>incx (int64_t)</td>
      <td>Input</td>
      <td>Memory address offset between adjacent elements of x (currently constrained to 1).</td>
    </tr>
    <tr>
      <td>result (aclTensor *)</td>
      <td>Output</td>
      <td><ul><li>Output result, corresponding to "result" in the formula.</li><li>Supported data type: <code>INT32</code>, containing only one element.</li><li>Data format: <code>ND</code>.</li><li>Shape: [1].</li></ul></td>
    </tr>
    </tbody>
    </table>
- **Return value:**

  For details about the return values, see [SiP Return Codes](../../context/sip_return_codes.md).

## Constraints

  - The number of input elements n is currently supported in the range [1, 6.71e+06].
  - The operator's input shape is [n], and the output shape is [1].
  - During actual computation, the operator does not support high-dimensional ND operations (dimensions ≥ 3).

## Calling Example

The example code is as follows. This sample is intended to provide a minimal implementation for quick start, development, and debugging of the operator. Its core goal is to demonstrate the core functionality of the operator using the simplest code, rather than providing production-grade security assurance. Users are advised not to directly use the example code for business purposes. If users apply the example code in their own real business scenarios and security issues occur, the users shall bear the consequences themselves.

- **asdBlasIsamax**

```Cpp
#include <iostream>
#include <vector>
#include "asdsip.h"
#include "acl/acl.h"
#include "acl_meta.h"

using namespace AsdSip;

#define ASD_STATUS_CHECK(err)                                                \
    do {                                                                     \
        AsdSip::AspbStatus err_ = (err);                                     \
        if (err_ != AsdSip::ErrorType::ACL_SUCCESS) {                                      \
            std::cout << "Execute failed." << std::endl; \
            exit(-1);                                                        \
        }                                                                    \
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
    // Boilerplate: Initialize ACL.
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
    // Call aclrtMalloc to allocate device-side memory.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // Call aclrtMemcpy to copy host-side data to device-side memory.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Compute the strides of the contiguous tensor.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // Call the aclCreateTensor API to create aclTensor.
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

int main(int argc, char **argv)
{
    int deviceId = 0;

    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    int64_t n = 6;
    int64_t incx = 1;

    int64_t xSize = n;
    int64_t ySize = 1;
    std::vector<float> tensorInXData;
    tensorInXData.resize(xSize);
    for (int64_t i = 0; i < xSize; i++) {
        if (i < xSize / 2) {
            tensorInXData[i] = 1.0 * i;
        } else {
            tensorInXData[i] = xSize - 1.0 * i;
        }
    }

    std::vector<int32_t> tensorOutYData;
    tensorOutYData.resize(ySize);

    std::cout << "------- input X -------" << std::endl;
    for (int64_t i = 0; i < xSize; i++) {
        std::cout << tensorInXData[i] << " ";
    }
    std::cout << std::endl;

    std::vector<int64_t> xShape = {xSize};
    std::vector<int64_t> yShape = {ySize};
    aclTensor *inputX = nullptr;
    aclTensor *outputY = nullptr;
    void *inputXDeviceAddr = nullptr;
    void *outputYDeviceAddr = nullptr;
    ret = CreateAclTensor(tensorInXData, xShape, &inputXDeviceAddr, aclDataType::ACL_FLOAT, &inputX);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);
    ret = CreateAclTensor(tensorOutYData, yShape, &outputYDeviceAddr, aclDataType::ACL_INT32, &outputY);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);

    asdBlasHandle handle;
    asdBlasCreate(handle);

    size_t lwork = 0;
    void *buffer = nullptr;
    asdBlasMakeIamaxPlan(handle);
    asdBlasGetWorkspaceSize(handle, lwork);
    std::cout << "lwork = " << lwork << std::endl;
    if (lwork > 0) {
        ret = aclrtMalloc(&buffer, static_cast<int64_t>(lwork), ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    asdBlasSetWorkspace(handle, buffer);
    asdBlasSetStream(handle, stream);

    ASD_STATUS_CHECK(asdBlasIsamax(handle, n, inputX, incx, outputY));

    asdBlasSynchronize(handle);
    asdBlasDestroy(handle);

    ret = aclrtMemcpy(tensorOutYData.data(),
        ySize * sizeof(int32_t),
        outputYDeviceAddr,
        ySize * sizeof(int32_t),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

    std::cout << "------- result -------" << std::endl;
    std::cout << tensorOutYData[0] << std::endl;
    std::cout << "Execute successfully." << std::endl;

    aclDestroyTensor(inputX);
    aclDestroyTensor(outputY);
    aclrtFree(inputXDeviceAddr);
    aclrtFree(outputYDeviceAddr);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```

- **asdBlasIcamax**

```Cpp
#include <iostream>
#include <vector>
#include "asdsip.h"
#include "acl/acl.h"
#include "acl_meta.h"

using namespace AsdSip;

#define ASD_STATUS_CHECK(err)                                                \
    do {                                                                     \
        AsdSip::AspbStatus err_ = (err);                                     \
        if (err_ != AsdSip::ErrorType::ACL_SUCCESS) {                                      \
            std::cout << "Execute failed." << std::endl; \
            exit(-1);                                                        \
        }                                                                    \
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
    // Boilerplate: Initialize ACL.
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
    // Call aclrtMalloc to allocate device-side memory.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // Call aclrtMemcpy to copy host-side data to device-side memory.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Compute the strides of the contiguous tensor.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // Call the aclCreateTensor API to create aclTensor.
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

int main(int argc, char **argv)
{
    int deviceId = 0;

    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    int64_t n = 6;
    int64_t incx = 1;

    int64_t xSize = 6;
    int64_t ySize = 1;
    std::vector<std::complex<float>> tensorInXData;
    tensorInXData.resize(xSize);
    for (int64_t i = 0; i < xSize; i++) {
        if (i < xSize / 2) {
            tensorInXData[i] = {(float)(1.0 * i), (float)(1.0 * i)};
        } else {
            tensorInXData[i] = {(float)(xSize - 1.0 * i), (float)(xSize - 1.0 * i)};
        }
    }

    std::vector<int32_t> tensorOutYData;
    tensorOutYData.resize(ySize);

    std::cout << "n: " << n << std::endl;
    std::cout << "------- input x -------" << std::endl;
    for (int64_t i = 0; i < xSize; i++) {
        std::cout << tensorInXData[i] << " ";
    }
    std::cout << std::endl;

    std::vector<int64_t> xShape = {xSize};
    std::vector<int64_t> yShape = {ySize};
    aclTensor *inputX = nullptr;
    aclTensor *outputY = nullptr;
    void *inputXDeviceAddr = nullptr;
    void *outputYDeviceAddr = nullptr;
    ret = CreateAclTensor(tensorInXData, xShape, &inputXDeviceAddr, aclDataType::ACL_COMPLEX64, &inputX);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);
    ret = CreateAclTensor(tensorOutYData, yShape, &outputYDeviceAddr, aclDataType::ACL_INT32, &outputY);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);

    asdBlasHandle handle;
    asdBlasCreate(handle);

    size_t lwork = 0;
    void *buffer = nullptr;
    asdBlasMakeIamaxPlan(handle);
    asdBlasGetWorkspaceSize(handle, lwork);
    std::cout << "lwork = " << lwork << std::endl;
    if (lwork > 0) {
        ret = aclrtMalloc(&buffer, static_cast<int64_t>(lwork), ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    asdBlasSetWorkspace(handle, buffer);
    asdBlasSetStream(handle, stream);

    ASD_STATUS_CHECK(asdBlasIcamax(handle, n, inputX, incx, outputY));

    asdBlasSynchronize(handle);
    asdBlasDestroy(handle);

    ret = aclrtMemcpy(tensorOutYData.data(),
        ySize * sizeof(int32_t),
        outputYDeviceAddr,
        ySize * sizeof(int32_t),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

    std::cout << "------- result -------" << std::endl;
    std::cout << tensorOutYData[0] << std::endl;
    std::cout << "Execute successfully." << std::endl;

    aclDestroyTensor(inputX);
    aclDestroyTensor(outputY);
    aclrtFree(inputXDeviceAddr);
    aclrtFree(outputYDeviceAddr);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
