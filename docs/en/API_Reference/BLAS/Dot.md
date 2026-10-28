# Dot

<!-- md-trans-meta sourceCommit=25a0c73916203ce4555696808917dc3c4c840c42 translatedAt=2026-10-10T12:18:23.837Z pushedAt=2026-10-14T03:23:53.834Z -->

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
`asdBlasMakeDotPlan`: initializes the `Dot` operator configuration corresponding to the handle.\
`asdBlasSdot`: computes the dot product of two real vectors.\
`asdBlasCdotu`: computes the dot product of two complex vectors.\
`asdBlasCdotc`: computes the dot product of one complex vector after taking its conjugate and another complex vector.
- Formula:
  - Formula for `asdBlasSdot`

  $$
  result=\sum _{i=1}^n(x[i] * y[i])
  $$

        Example:\
         The input `x` is:\
        [1.0, 2.0]\
        The input `y` is:\
        [1.0, 2.0]\
        After the `asdBlasSdot` operator is called, the output `result` is:\
        5.0

  - Formula for asdBlasCdotu

  $$
  result=\sum _{i=1}^n((x[i]) * y[i])
  $$
        where `x[i]` and `y[i]` are complex numbers, and `conj` denotes the conjugate operation.\
        Example:\
        The input `x` is:\
        [ 0.1554+0.8840j, -0.3564-0.2552j]\
        The input `y` is:\
        [-0.1404+1.3380j, -0.4876+0.1842j]\
        After the `asdBlasCdotu` operator is called, the output `result` is:\
        -0.9839 + 0.1425j

- Formula of asdBlasCdotc

  $$
  result=\sum _{i=1}^n(conj(x[i]) * y[i])
  $$
        where `x[i]` and `y[i]` are complex numbers, and `conj` denotes the conjugate operation.\
        Example:\
        The input `x` is:\
        [ 0.1554+0.8840j, -0.3564-0.2552j]\
        The input `y` is:\
        [-0.1404+1.3380j, -0.4876+0.1842j]\
        After the `asdBlasCdotc` operator is called, the output `result` is:\
        1.2877-0.1420j

## Function Prototype

```cpp
AspbStatus asdBlasMakeDotPlan(
  asdBlasHandle handle)
```

```cpp
AspbStatus asdBlasSdot(
  asdBlasHandle      handle,
  const int64_t      n,
  aclTensor *        x,
  const int64_t      incx,
  aclTensor *        y,
  const int64_t      incy,
  aclTensor *        result)
```

```cpp
AspbStatus asdBlasCdotu(
  asdBlasHandle      handle,
  const int64_t      n,
  aclTensor *        x,
  const int64_t      incx,
  aclTensor *        y,
  const int64_t      incy,
  aclTensor *        result)
```

```cpp
AspbStatus asdBlasCdotc(
  asdBlasHandle         handle,
  const int64_t         n,
  aclTensor *           x,
  const int64_t         incx,
  aclTensor *           y,
  const int64_t         incy,
  aclTensor *           result)
```

## asdBlasMakeDotPlan

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

## asdBlasSdot & asdBlasCdotu & asdBlasCdotc

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
      <td>Number of elements in vector x or vector y.</td>
    </tr>
    <tr>
      <td>x (aclTensor *)</td>
      <td>Input</td>
      <td><ul><li>Corresponds to "x" in the formula.</li><li>Supported data types for <code>asdBlasSdot</code>: <code>FLOAT32</code>.</li><li>Supported data types for <code>asdBlasCdotu</code> & <code>asdBlasCdotc</code>: <code>COMPLEX64</code>.</li><li>Data format: <code>ND</code>.</li><li>Shape: [n].</li></ul></td>
    </tr>
    <tr>
      <td>incx (int64_t)</td>
      <td>Input</td>
      <td>Memory address offset between adjacent elements of vector x (currently constrained to 1).</td>
    </tr>
    <tr>
      <td>y (aclTensor *)</td>
      <td>Input</td>
      <td><ul><li>Corresponds to "y" in the formula.</li><li>Supported data types for <code>asdBlasSdot</code>: <code>FLOAT32</code>.</li><li>Supported data types for <code>asdBlasCdotu</code> & <code>asdBlasCdotc</code>: <code>COMPLEX64</code>.</li><li>Data format: <code>ND</code>.</li><li>Shape: [n].</li></ul></td>
    </tr>
    <tr>
      <td>incy (int64_t)</td>
      <td>Input</td>
      <td>Memory address offset between adjacent elements of vector y (currently constrained to 1).</td>
    </tr>
    <tr>
      <td>result (aclTensor *)</td>
      <td>Output</td>
      <td><ul><li>Output result, corresponding to "result" in the formula.</li><li>Supported data type: <code>FLOAT32</code>, containing only one element.</li><li>Data format: <code>ND</code>.</li><li>Shape: [1].</li></ul></td>
    </tr>
    </tbody>
    </table>

- **Return value:**

  For details about the return values, see [SiP Return Codes](../../context/sip_return_codes.md).

## Constraints

  - The number of input elements `n` is currently supported in the range [1, 6.71e+06].
  - The operator's input shape is [n], and the output shape is [1].
  - During actual computation, the operator does not support high-dimensional ND operations (dimensions ≥ 3).

## Calling Example

The example code is as follows. This sample is intended to provide a minimal implementation for quick start, development, and debugging of the operator. Its core goal is to demonstrate the core functionality of the operator using the simplest code, rather than providing production-grade security assurance. Users are advised not to directly use the example code for business purposes. If users apply the example code in their own real business scenarios and security issues occur, the users shall bear the consequences themselves.

- **asdBlasSdot**

```cpp
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
    // Set the device ID used by the operator.
    int deviceId = 0;
    // Boilerplate: Create an execution stream.
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // Create the host-side data of the tensor.
    int64_t n = 5;
    int64_t incx = 1;
    int64_t incy = 1;

    int64_t xSize = 5;
    std::vector<float> tensorInXData;
    tensorInXData.resize(xSize);
    for (int64_t i = 0; i < xSize; i++) {
        tensorInXData[i] = 1.0 + i;
    }

    int64_t ySize = 5;
    std::vector<float> tensorInYData;
    tensorInYData.resize(xSize);
    for (int64_t i = 0; i < ySize; i++) {
        tensorInYData[i] = 10.0 + i;
    }

    int64_t resultSize = 1;
    std::vector<float> resultData;
    resultData.resize(resultSize);

    std::cout << "------- input x -------" << std::endl;
    for (int64_t i = 0; i < xSize; i++) {
        std::cout << tensorInXData[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "------- input y -------" << std::endl;
    for (int64_t i = 0; i < ySize; i++) {
        std::cout << tensorInYData[i] << " ";
    }
    std::cout << std::endl;

    // Create the input/output tensors.
    std::vector<int64_t> xShape = {xSize};
    std::vector<int64_t> yShape = {ySize};
    std::vector<int64_t> resultShape = {resultSize};
    aclTensor *inputX = nullptr;
    aclTensor *inputY = nullptr;
    aclTensor *result = nullptr;
    void *inputXDeviceAddr = nullptr;
    void *inputYDeviceAddr = nullptr;
    void *resultDeviceAddr = nullptr;
    ret = CreateAclTensor(tensorInXData, xShape, &inputXDeviceAddr, aclDataType::ACL_FLOAT, &inputX);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);
    ret = CreateAclTensor(tensorInYData, yShape, &inputYDeviceAddr, aclDataType::ACL_FLOAT, &inputY);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);
    ret = CreateAclTensor(resultData, resultShape, &resultDeviceAddr, aclDataType::ACL_FLOAT, &result);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);

    // Create the operator execution handle.
    asdBlasHandle handle;
    asdBlasCreate(handle);

    // Create the workspace required for operator execution.
    size_t lwork = 0;
    void *buffer = nullptr;
    asdBlasMakeDotPlan(handle);
    asdBlasGetWorkspaceSize(handle, lwork);
    std::cout << "lwork = " << lwork << std::endl;
    if (lwork > 0) {
        ret = aclrtMalloc(&buffer, static_cast<int64_t>(lwork), ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    asdBlasSetWorkspace(handle, buffer);

    // Configure the operator execution information.
    asdBlasSetStream(handle, stream);

    // Call the API to execute the operator (fixed calling logic).
    ASD_STATUS_CHECK(asdBlasSdot(handle, n, inputX, incx, inputY, incy, result));
    asdBlasSynchronize(handle);

    // Destroy the operator handle after scheduling the operator.
    asdBlasDestroy(handle);

    // Copy the device-side data of the output tensor to host-side memory.
    ret = aclrtMemcpy(resultData.data(),
        resultSize * sizeof(float),
        resultDeviceAddr,
        resultSize * sizeof(float),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

    std::cout << "------- result -------" << std::endl;
    for (int64_t i = 0; i < 1; i++) {
        std::cout << resultData[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "Execute successfully." << std::endl;

    // Release resources.
    aclDestroyTensor(inputX);
    aclDestroyTensor(inputY);
    aclDestroyTensor(result);
    aclrtFree(inputXDeviceAddr);
    aclrtFree(inputYDeviceAddr);
    aclrtFree(resultDeviceAddr);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```

- **asdBlasCdotu**

```cpp

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <complex>
#include "asdsip.h"
#include "acl/acl.h"
#include "acl_meta.h"

using namespace AsdSip;

#define ASD_STATUS_CHECK(err)                                  \
    do {                                                       \
        AsdSip::AspbStatus err_ = (err);                       \
        if (err_ != AsdSip::ErrorType::ACL_SUCCESS) {          \
            std::cout << "Execute failed." << std::endl;       \
            exit(-1);                                          \
        } else {                                               \
            std::cout << "Execute successfully." << std::endl; \
        }                                                      \
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

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

void printTensor(const std::complex<float>* tensorData, int64_t tensorSize)
{
    for (int64_t i = 0; i < tensorSize; i++) {
        std::cout << tensorData[i] << " ";
    }
    std::cout << std::endl;
}

int Init(int32_t deviceId, aclrtStream* stream)
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
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor)
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
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

void printTensor(std::vector<std::complex<float>> tensorData, int64_t tensorSize)
{
    for (int64_t i = 0; i < tensorSize; i++) {
        std::cout << tensorData[i] << " ";
    }
    std::cout << std::endl;
}

int main(int argc, char** argv)
{
    int deviceId = 0;

    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    int64_t n = 8;
    int64_t xSize = 8;
    int64_t ySize = 8;

    std::vector<std::complex<float>> tensorInXData;
    tensorInXData.resize(xSize);
    for (int64_t i = 0; i < xSize; i++) {
        tensorInXData[i] = {2.0, (float)(1.0 + i)};
    }

    std::vector<std::complex<float>> tensorInYData;
    tensorInYData.resize(ySize);
    for (int64_t i = 0; i < ySize; i++) {
        tensorInYData[i] = {3.0, 4.0};
    }

    int64_t resultSize = 1;
    std::vector<std::complex<float>> resultData;
    resultData.resize(resultSize);

    std::cout << "------- input TensorInX -------" << std::endl;
    printTensor(tensorInXData.data(), xSize);

    std::cout << "------- input TensorInY -------" << std::endl;
    printTensor(tensorInYData.data(), ySize);

    std::vector<int64_t> xShape = {xSize};
    std::vector<int64_t> yShape = {ySize};
    std::vector<int64_t> resultShape = {resultSize};
    aclTensor* inputX = nullptr;
    aclTensor* inputY = nullptr;
    aclTensor* result = nullptr;
    void* inputXDeviceAddr = nullptr;
    void* inputYDeviceAddr = nullptr;
    void* resultDeviceAddr = nullptr;
    ret = CreateAclTensor(tensorInXData, xShape, &inputXDeviceAddr, aclDataType::ACL_COMPLEX64, &inputX);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);
    ret = CreateAclTensor(tensorInYData, yShape, &inputYDeviceAddr, aclDataType::ACL_COMPLEX64, &inputY);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);
    ret = CreateAclTensor(resultData, resultShape, &resultDeviceAddr, aclDataType::ACL_COMPLEX64, &result);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);

    asdBlasHandle handle;
    asdBlasCreate(handle);

    size_t lwork = 0;
    void* buffer = nullptr;
    asdBlasMakeDotPlan(handle);
    asdBlasGetWorkspaceSize(handle, lwork);
    std::cout << "lwork = " << lwork << std::endl;
    if (lwork > 0) {
        ret = aclrtMalloc(&buffer, static_cast<int64_t>(lwork), ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    asdBlasSetWorkspace(handle, buffer);
    asdBlasSetStream(handle, stream);

    ASD_STATUS_CHECK(asdBlasCdotu(handle, n, inputX, 1, inputY, 1, result));

    asdBlasSynchronize(handle);
    asdBlasDestroy(handle);

    ret = aclrtMemcpy(resultData.data(), resultSize * sizeof(std::complex<float>), resultDeviceAddr,
                      resultSize * sizeof(std::complex<float>), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

    std::cout << "------- result -------" << std::endl;
    printTensor(resultData.data(), resultSize);

    aclDestroyTensor(inputX);
    aclDestroyTensor(inputY);
    aclDestroyTensor(result);
    aclrtFree(inputXDeviceAddr);
    aclrtFree(inputYDeviceAddr);
    aclrtFree(resultDeviceAddr);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}

```

- **asdBlasCdotc**

```cpp

#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <random>
#include "asdsip.h"
#include "acl/acl.h"
#include "acl_meta.h"

using namespace AsdSip;

#define ASD_STATUS_CHECK(err)                                  \
    do {                                                       \
        AsdSip::AspbStatus err_ = (err);                       \
        if (err_ != AsdSip::ErrorType::ACL_SUCCESS) {          \
            std::cout << "Execute failed." << std::endl;       \
            exit(-1);                                          \
        } else {                                               \
            std::cout << "Execute successfully." << std::endl; \
        }                                                      \
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

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

void printTensor(const std::complex<float>* tensorData, int64_t tensorSize)
{
    for (int64_t i = 0; i < tensorSize; i++) {
        std::cout << tensorData[i] << " ";
    }
    std::cout << std::endl;
}

int Init(int32_t deviceId, aclrtStream* stream)
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
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor)
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
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

void printTensor(std::vector<std::complex<float>> tensorData, int64_t tensorSize)
{
    for (int64_t i = 0; i < tensorSize; i++) {
        std::cout << tensorData[i] << " ";
    }
    std::cout << std::endl;
}

int main(int argc, char** argv)
{
    int deviceId = 0;

    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    int64_t n = 8;
    int64_t xSize = 8;
    int64_t ySize = 8;

    std::vector<std::complex<float>> tensorInXData;
    tensorInXData.resize(xSize);
    for (int64_t i = 0; i < xSize; i++) {
        tensorInXData[i] = {2.0, (float)(1.0 + i)};
    }
    std::vector<std::complex<float>> tensorInYData;
    tensorInYData.resize(ySize);
    for (int64_t i = 0; i < ySize; i++) {
        tensorInYData[i] = {3.0, 4.0};
    }

    int64_t resultSize = 1;
    std::vector<std::complex<float>> resultData;
    resultData.resize(resultSize);

    std::cout << "------- input TensorInX -------" << std::endl;
    printTensor(tensorInXData.data(), xSize);

    std::cout << "------- input TensorInY -------" << std::endl;
    printTensor(tensorInYData.data(), ySize);

    std::vector<int64_t> xShape = {xSize};
    std::vector<int64_t> yShape = {ySize};
    std::vector<int64_t> resultShape = {resultSize};
    aclTensor* inputX = nullptr;
    aclTensor* inputY = nullptr;
    aclTensor* result = nullptr;
    void* inputXDeviceAddr = nullptr;
    void* inputYDeviceAddr = nullptr;
    void* resultDeviceAddr = nullptr;
    ret = CreateAclTensor(tensorInXData, xShape, &inputXDeviceAddr, aclDataType::ACL_COMPLEX64, &inputX);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);
    ret = CreateAclTensor(tensorInYData, yShape, &inputYDeviceAddr, aclDataType::ACL_COMPLEX64, &inputY);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);
    ret = CreateAclTensor(resultData, resultShape, &resultDeviceAddr, aclDataType::ACL_COMPLEX64, &result);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);

    asdBlasHandle handle;
    asdBlasCreate(handle);

    size_t lwork = 0;
    void* buffer = nullptr;
    asdBlasMakeDotPlan(handle);
    asdBlasGetWorkspaceSize(handle, lwork);
    std::cout << "lwork = " << lwork << std::endl;
    if (lwork > 0) {
        ret = aclrtMalloc(&buffer, static_cast<int64_t>(lwork), ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    asdBlasSetWorkspace(handle, buffer);
    asdBlasSetStream(handle, stream);

    ASD_STATUS_CHECK(asdBlasCdotc(handle, n, inputX, 1, inputY, 1, result));

    asdBlasSynchronize(handle);
    asdBlasDestroy(handle);

    ret = aclrtMemcpy(resultData.data(), resultSize * sizeof(std::complex<float>), resultDeviceAddr,
                      resultSize * sizeof(std::complex<float>), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

    std::cout << "------- result -------" << std::endl;
    printTensor(resultData.data(), resultSize);

    aclDestroyTensor(inputX);
    aclDestroyTensor(inputY);
    aclDestroyTensor(result);
    aclrtFree(inputXDeviceAddr);
    aclrtFree(inputYDeviceAddr);
    aclrtFree(resultDeviceAddr);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}

```
