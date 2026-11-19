# BLAS Common Interfaces

<!-- md-trans-meta sourceCommit=cc624cb911fb7f15fd153fbb95bbad1e3d963968 translatedAt=2026-09-07T13:42:57.606Z pushedAt=2026-09-11T12:35:55.495Z -->

## Operator Usage Instructions

To use BLAS operators, first create a handle, then call the plan interface of the corresponding operator to initialize and bind the operator configuration associated with the handle. Next, call the BLAS common interface `asdBlasGetWorkspaceSize` to obtain the workspace size required for computation and the executor that contains the operator computation flow. Then call `asdBlasSetWorkspace` to set the required workspace for the corresponding plan. Finally, call the BLAS operator interface to execute the computation. After the submitted work completes, call `asdBlasDestroy` to destroy the handle and its plan. The workspace and stream passed to the BLAS APIs remain caller-owned: synchronize the handle before releasing the workspace with the runtime, and destroy the stream separately.

## Common Interface Description

- **asdBlasCreate**

    **Function description**: Creates a globally unique handle.\
  **Function prototype**: AspbStatus asdBlasCreate(asdBlasHandle &handle)\
  **Parameter description**:
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
      <td>Input/Output</td>
      <td>Handle of the <code>asdBlasCreate</code> interface.</td>
    </tr>
  </tbody>
    </table>

- **asdBlasSetStream**

    **Function description**: Binds a caller-owned stream created using the runtime to a specific plan instance. The caller must keep the stream valid until submitted work completes and destroy it separately.\
    **Function prototype**: AspbStatus asdBlasSetStream(asdBlasHandle handle, void *stream)\
    Parameter description:
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
      <td>Handle of the <code>asdBlasSetStream</code> interface.</td>
    </tr>
    <tr>
      <td>stream (void *)</td>
      <td>Input</td>
      <td>Pointer to the stream object.</td>
    </tr>
  </tbody>
    </table>

- **asdBlasDestroy**

    **Function description**: Destroys the handle and its associated plan, and releases resources owned by the plan. The handle is invalid after this call. Caller-provided workspace and stream are not freed.\
    **Function prototype**: AspbStatus asdBlasDestroy(asdBlasHandle handle)\
    **Parameter description:**
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
      <td>Handle of the <code>asdBlasDestroy</code> API.</td>
    </tr>
  </tbody>
    </table>

- **asdBlasSetWorkspace**

    **Function description**: Binds caller-provided workspace to the corresponding plan. The caller retains ownership and must keep the workspace valid until submitted work completes, then release it with the runtime.\
    **Function prototype**: AspbStatus asdBlasSetWorkspace(asdBlasHandle handle, void *workSpace)\
    **Parameter description:**
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
      <td>Handle of the <code>asdBlasSetWorkspace</code> API.</td>
    </tr>
    <tr>
      <td>workSpace (void *)</td>
      <td>Input</td>
      <td>Pointer to caller-owned storage for the required workspace. The library does not free this memory.</td>
    </tr>
  </tbody>
    </table>

- **asdBlasSynchronize**

    **Function description**: Synchronously waits for operator execution to complete.\
    **Function prototype**: AspbStatus asdBlasSynchronize(asdBlasHandle handle)\
    **Parameter description:**
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
      <td>Handle of the <code>asdBlasSynchronize</code> API.</td>
    </tr>
  </tbody>
    </table>

- **asdBlasGetWorkspaceSize**

    **Function description**: Computes the required workspace size and the executor that contains the operator computation flow.\
    **Function prototype**: AspbStatus asdBlasGetWorkspaceSize(asdBlasHandle handle, size_t &workspaceSize);\
    Parameter description:
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
      <td>Handle of the <code>asdBlasGetWorkspaceSize</code> API.</td>
    </tr>
    <tr>
      <td>workspaceSize (size_t &)</td>
      <td>Input/Output</td>
      <td>Size of the required workspace.</td>
    </tr>
  </tbody>
    </table>
