# SmatinvBatched

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT series products</term>: Not supported
<!-- end id1 -->

<!-- npu="A3" id2 -->
- <term>Atlas A3 series products</term>: Supported
<!-- end id2 -->

<!-- npu="910b" id3 -->
- <term>Atlas A2 series products</term>: Supported
<!-- end id3 -->

<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 inference products</term>: Not supported
<!-- end id4 -->

<!-- npu="310p" id5 -->
- <term>Atlas inference series products</term>: Not supported
<!-- end id5 -->

<!-- npu="910" id6 -->
- <term>Atlas training series products</term>: Not supported
<!-- end id6 -->

## Function Description

- Interface function:\
asdBlasMakeSmatinvBatchedPlan: Initializes the operator configuration corresponding to the handle\
asdBlasSmatinvBatched: Computes the inverse of a batch of single-precision real matrices (fp32).
- Calculation formula:\
Computes the inverse matrix of each real matrix in a batch. Each real matrix must satisfy $A^{-1}A=I$, where A is a non-singular square matrix of size n*n and I is the identity matrix.

## Function Prototype

```Cpp
AspbStatus asdBlasMakeSmatinvBatchedPlan(
  asdBlasHandle            handle,
  const int64_t            n,
  const int64_t            batchSize)
```

```Cpp
AspbStatus asdBlasSmatinvBatched(
  asdBlasHandle            handle,
  const int64_t            n,
  aclTensor *              A,
  const int64_t            lda,
  aclTensor *              Ainv,
  const int64_t            lda_inv,
  aclTensor *              info,
  int64_t                  batchSize)
```

## asdBlasMakeSmatinvBatchedPlan

- **Parameter Description:**

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
      <td>Operator handle</td>
    </tr>
    <tr>
      <td>n (int64_t)</td>
      <td>Input</td>
      <td>Number of rows of matrix A in a single batch.</td>
    </tr>
    <tr>
      <td>batchSize (int64_t)</td>
      <td>Input</td>
      <td>Number of matrices in the real matrix inversion.</td>
    </tr>
    </tbody>
    </table>

- **Return Value**:

  Returns a status code. For details, see [SiP Return Codes](../../context/sip_return_codes.md).

## asdBlasSmatinvBatched

- **Parameter Description:**

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
      <td>Operator handle</td>
    </tr>
    <tr>
      <td>n (int64_t)</td>
      <td>Input</td>
      <td>Number of rows of matrix A in a single batch.</td>
    </tr>
    <tr>
      <td>A (aclTensor *)</td>
      <td>Input</td>
      <td><ul><li>Input matrix, corresponding to 'A' in the formula.</li><li>Row-major order.</li><li>Data type: FLOAT.</li><li>Data format: ND.</li><li>Shape: [batchSize, n, n].</li></ul></td>
    </tr>
    <tr>
      <td>lda (int64_t)</td>
      <td>Input</td>
      <td>Memory address offset between adjacent elements of A (currently constrained to n).</td>
    </tr>
    <tr>
      <td>Ainv (aclTensor *)</td>
      <td>Output</td>
      <td><ul><li>Output inverse matrix.</li><li>Data type: FLOAT.</li><li>Data format: ND.</li><li>Shape: [batchSize, n, n].</li></ul></td>
    </tr>
    <tr>
      <td>lda_inv (int64_t)</td>
      <td>Input</td>
      <td>Memory address offset between adjacent elements of the output inverse matrix (currently constrained to n).</td>
    </tr>
    <tr>
      <td>info (aclTensor *)</td>
      <td>Input</td>
      <td><ul><li>Inversion result information of each batch matrix.</li><li>Data type: int32_t.</li><li>Data format: ND.</li><li>Shape: [batchSize].</li></ul></td>
    </tr>
    <tr>
      <td>batchSize (int64_t)</td>
      <td>Input</td>
      <td>Number of matrices in the real matrix inversion.</td>
    </tr>
    </tbody>
    </table>

- **Return Value**:

  Returns a status code. For details, see [SiP Return Codes](../../context/sip_return_codes.md).

## Constraints

- The lda, lda_inv and info parameters are not actually used in the current version.
- Input parameter n must be less than or equal to 256.
- Input parameter batchSize must be less than or equal to 3000.
- The n and batchSize passed to the execution API must be identical to those used when creating the plan with asdBlasMakeSmatinvBatchedPlan (batchSize must also match when n >= 32); otherwise ACL_ERROR_INVALID_PARAM is returned and the plan must be re-created before execution.
- Internally, the implementation automatically dispatches by matrix size: Gauss-Jordan elimination kernel for n < 32, LU-decomposition-based inversion kernel for n >= 32 (with a high-performance per-core-matrix kernel for n%16 ∈ {0, 8} combined with batchSize >= 2 or n <= 128). This is transparent to users.

## Calling Example

For a complete executable project (including the build script and residual verification), see [example_smatinv_batched](../../../../example/A2/BLAS/smatinv_batched/README.md).
