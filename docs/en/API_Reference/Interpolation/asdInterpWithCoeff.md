# asdInterpWithCoeff

<!-- md-trans-meta sourceCommit=edd7b7636f7c5636841ec1b2148cde7afc4cdefb translatedAt=2026-09-07T14:09:26.778Z pushedAt=2026-09-11T12:35:55.606Z -->

## Applicable Product

|Product             |  Supported  |
|:-------------------------|:----------:|
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>    |     ×    |
|  <term>Atlas training products</term>    |     ×    |
|  <term>Atlas A3 training products/Atlas A3 inference products</term>   |     √    |
|  <term>Atlas A2 training products/Atlas A2 inference products</term>     |     √    |
|  <term>Ascend 950PR/Ascend 950DT</term>   |     ×  |

## Function Description

- API function:\
`asdInterpWithCoeffGetWorkspaceSize`: Computes the workspace size required by the `asdInterpWithCoeff` operator.\
`asdInterpWithCoeff`: Supports vector interpolation operations, mainly used for channel estimation of data symbols or equalization coefficient interpolation.
- Formula:

  $$
  result_{b,i,j}=\sum_{k=0}^{nRs-1} (A)_{b,i,k}\times(B)_{b,k,j}
  $$

  where `A` is `coefficient` (shape [batch, 14-nRs, nRs]), `B` is `x` (shape [batch, nRs, totalSubcarrier]), and `result` is `y` (shape [batch, 14-nRs, totalSubcarrier]); the sum runs over the reference-signal dimension `nRs`, that is, each output element is the weighted sum of a `coefficient` row vector and an `x` column vector.

  Example:

  The input `A` (`coefficient`, [1, 12, 2]) is:\
[ [ 1+1i, 1+1i ],\
  [ 1+1i, 1+1i ],\
  ...\
  [ 1+1i, 1+1i ] ] (12 rows in total, all 1+1i)\
The input `B` (`x`, [1, 2, 12]) is:\
[ [ 1+1i, 1+1i, ..., 1+1i ],\
  [ 2+2i, 2+2i, ..., 2+2i ] ]\
After the `asdInterpWithCoeff` operator is called, the output `result` ([1, 12, 12]) is:\
[ [ 0+6i, 0+6i, ..., 0+6i ],\
  [ 0+6i, 0+6i, ..., 0+6i ],\
  ...\
  [ 0+6i, 0+6i, ..., 0+6i ] ] (every element is 0+6i, e.g. result[0,0,0]=(1+1i)×(1+1i)+(1+1i)×(2+2i)=2i+4i=6i)

## Function Prototype

```Cpp
AspbStatus asdInterpWithCoeffGetWorkspaceSize(
  size_t &             workspaceSize)
```

```Cpp
AspbStatus asdInterpWithCoeff(
  const aclTensor *    x,
  const aclTensor *    coefficient,
  aclTensor *          y,
  void *               stream,
  void *               workSpace = nullptr)
```

## asdInterpWithCoeffGetWorkspaceSize

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
      <td>workspaceSize (size_t &)</td>
      <td>Output</td>
      <td>Workspace required by the operator.</td>
    </tr>
  </tbody>
    </table>
- **Return value:**

  For details about the return values, see [SiP Return Codes](../../context/sip_return_codes.md).

## asdInterpWithCoeff

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
      <td>x (aclTensor *)</td>
      <td>Input</td>
      <td><ul><li>Corresponds to "B" in the formula.</li><li>Supported data types: <code>COMPLEX32</code> and <code>COMPLEX64</code>.</li><li>Data format: <code>ND</code>.</li>
      <li>shape: [batch, nRs, totalSubcarrier].<ul><li>batch: number of beams, value range 1 to 1024 (for 6G, the maximum value is 16 (number of streams of the terminal) * 64 (number of beams received by the base station) = 1024).</li><li>nRs: number of reference signals, value 2 or 4.</li><li>totalSubcarrier = nRB * 12.</li><li>nRB: number of resource blocks, value range 1 to 2730 (each RB contains 12 subcarriers; for 5G, the value range is 1 to 273; for 6G, the value is 4 to 10 times that of 5G).</li>
      </ul></li></ul></td>
    </tr>
    <tr>
    <td>coefficient (aclTensor *)</td>
      <td>Input</td>
      <td><ul><li>Corresponds to "A" in the formula.</li><li>Supported data types: <code>COMPLEX32</code> and <code>COMPLEX64</code>.</li><li>Data format: <code>ND</code>.</li>
      <li>shape: [batch, 14-nRs, nRs].<ul><li>batch: number of beams, value range 1 to 1024 (for 6G, the maximum value is 16 (number of streams of the terminal) * 64 (number of beams received by the base station) = 1024).</li><li>nRs: number of reference signals, value 2 or 4.</li></ul></li></ul></td>
    </tr>
    <tr>
      <td>y (aclTensor *)</td>
      <td>Output</td>
      <td><ul><li>Corresponds to "result" in the formula.</li><li>Supported data types: <code>COMPLEX32</code> and <code>COMPLEX64</code>.</li><li>Data format: <code>ND</code>.</li>
      <li>shape: [batch, 14-nRs, totalSubcarrier].<ul><li>batch: beam count, value range is 1 to 1024 (for 6G, the maximum value is 16 (number of streams of the terminal) * 64 (number of beams received by the base station) = 1024).</li><li>nRs: number of reference signals, value is 2 or 4.</li><li>totalSubcarrier = nRB*12.</li><li>nRB: number of resource blocks, value range is 1 to 2730 (each RB contains 12 subcarriers; for 5G, the value range is 1 to 273, and for 6G, the value is 4 to 10 times that of 5G).</li></ul></li></ul></td>
    </tr>
    <tr>
      <td>stream (void *)</td>
      <td>Input</td>
      <td>NPU execution stream.</td>
    </tr>
    <tr>
      <td>workSpace (void *)</td>
      <td>Input</td>
      <td>Workspace required by the <code>asdInterpWithCoeff</code> operator.</td>
    </tr>
  </tbody>
  </table>

- **Return value:**

  For details about the return values, see [SiP Return Codes](../../context/sip_return_codes.md).

## Constraints

None

## Calling Example

The example code is as follows. This sample is intended to provide a minimal implementation for quick start, development, and debugging of the operator. Its core goal is to demonstrate the core functionality of the operator using the simplest code, rather than providing production-grade security assurance. Users are advised not to directly use the example code for business purposes. If users apply the example code in their own real business scenarios and security issues occur, the users shall bear the consequences themselves.

```Cpp
#include <iostream>
#include <complex>
#include <vector>
#include "interp_api.h"
#include "acl/acl.h"
#include "acl_meta.h"

using namespace AsdSip;

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
    aclInit(nullptr);
    aclrtSetDevice(deviceId);
    aclrtCreateStream(stream);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T) * 2; // 2 : complex
    // Call aclrtMalloc to apply for device-side memory.
    aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    // Call aclrtMemcpy to copy host-side data to device-side memory.
    aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);

    // Compute the strides of the contiguous tensor.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // Call the aclCreateTensor API to create an aclTensor.
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
    //Create an execution stream (fixed pattern).
    aclrtStream stream;
    Init(deviceId, &stream);

    // Create host-side data for the tensor.
    int64_t batch = 1;
    int64_t nRs = 2;
    int64_t totalSubcarrier = 32;
    int64_t nSignal = 14;

    int64_t xSize = batch * nRs * totalSubcarrier * 2;
    std::vector<float> tensorInXData;
    tensorInXData.resize(xSize);
    for (int64_t i = 0; i < xSize; i++) {
        tensorInXData[i] = 1.0 + i;
    }

    int64_t coeffSize = batch * (nSignal - nRs) * nRs * 2;
    std::vector<float> coeffData;
    coeffData.resize(xSize);
    for (int64_t i = 0; i < coeffSize; i++) {
        coeffData[i] = 1;
    }

    int64_t resultSize = batch * (nSignal - nRs) * totalSubcarrier * 2;
    std::vector<float> resultData;
    resultData.resize(resultSize);
    for (int64_t i = 0; i < resultSize; i++) {
        resultData[i] = 2;
    }

    // int64_t xSize = batch * nRs * totalSubcarrier;
    // std::vector<std::complex<float>> tensorInXData(xSize, std::complex<float>(0, 0));
    // for (int i = 0; i < xSize; i++) {
    //     tensorInXData[i] = std::complex<float>(i * 2, i * 2 + 1);
    // }
    // int64_t coeffSize = batch * (nSignal - nRs) * nRs;
    // std::vector<std::complex<float>> coeffData(xSize, std::complex<float>(0, 0));
    // for (int i = 0; i < coeffSize; i++) {
    //     coeffData[i] = std::complex<float>(1, 1);
    // }
    // int64_t resultSize = batch * (nSignal - nRs) * totalSubcarrier;
    // std::vector<std::complex<float>> resultData(xSize, std::complex<float>(0, 0));
    // for (int i = 0; i < resultSize; i++) {
    //     resultData[i] = std::complex<float>(2, 2);
    // }

    std::cout << "------- input x -------" << std::endl;
    for (int64_t i = 0; i < xSize; i++) {
        std::cout << tensorInXData[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "------- input coeff -------" << std::endl;
    for (int64_t i = 0; i < coeffSize; i++) {
        std::cout << coeffData[i] << " ";
    }
    std::cout << std::endl;

    // Create the input/output tensors.
    aclTensor *inputX = nullptr;
    aclTensor *inputCoeff = nullptr;
    aclTensor *result = nullptr;
    void *inputXDeviceAddr = nullptr;
    void *inputYDeviceAddr = nullptr;
    void *resultDeviceAddr = nullptr;
    CreateAclTensor(tensorInXData, {batch, nRs, totalSubcarrier}, &inputXDeviceAddr, aclDataType::ACL_COMPLEX64, &inputX);
    CreateAclTensor(coeffData, {batch, nSignal-nRs, nRs}, &inputYDeviceAddr, aclDataType::ACL_COMPLEX64, &inputCoeff);
    CreateAclTensor(resultData, {batch, nSignal-nRs, totalSubcarrier}, &resultDeviceAddr, aclDataType::ACL_COMPLEX64, &result);

    size_t lwork = 0;
    void *buffer = nullptr;
    AsdSip::asdInterpWithCoeffGetWorkspaceSize(lwork);
    if (lwork > 0) {
        aclrtMalloc(&buffer, static_cast<int64_t>(lwork), ACL_MEM_MALLOC_HUGE_FIRST);
    }
    asdInterpWithCoeff(inputX, inputCoeff, result, stream, buffer);
    aclrtSynchronizeStream(stream);
    // Copy the device-side data of the output tensor to host-side memory.
    aclrtMemcpy(resultData.data(),
        resultSize * sizeof(float),
        resultDeviceAddr,
        resultSize * sizeof(float),
        ACL_MEMCPY_DEVICE_TO_HOST);

    std::cout << "------- result -------" << std::endl;
    for (int64_t i = 0; i < nSignal - nRs; i++) {
        for (int64_t j = 0; j < totalSubcarrier * 2; j++) {
            std::cout << resultData[i * totalSubcarrier * 2 + j] << " ";
        }
        std::cout << std::endl;
    }

    // Release resources.
    aclDestroyTensor(inputX);
    aclDestroyTensor(inputCoeff);
    aclDestroyTensor(result);
    aclrtFree(inputXDeviceAddr);
    aclrtFree(inputYDeviceAddr);
    aclrtFree(resultDeviceAddr);
    if (lwork > 0) {
        aclrtFree(buffer);
    }

    // Reset the deviceId used by the operator after scheduling the operator.
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
