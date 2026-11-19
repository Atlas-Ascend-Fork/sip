# BLAS公共接口

## 算子使用说明

若需使用BLAS算子，需先创建句柄，然后调用对应算子的plan接口，初始化该句柄对应的算子配置并进行绑定，接着调用BLAS公共接口“asdBlasGetWorkspaceSize”获取计算所需workspace大小以及包含了算子计算流程的执行器，然后调用“asdBlasSetWorkspace”给对应的plan设置需要的workspace，最后调用BLAS算子接口执行计算。提交的计算完成后，调用“asdBlasDestroy”销毁句柄及其plan。“asdBlasSetWorkspace”传入的workspace和“asdBlasSetStream”传入的stream由调用方持有；释放workspace前先同步句柄，workspace由调用方通过runtime释放，stream也由调用方单独销毁。

## 公共接口说明

- **asdBlasCreate**

    功能描述：创建全局唯一的handle。\
函数原型：AspbStatus asdBlasCreate(asdBlasHandle &handle)\
    **参数说明：**
    <table style="undefined;table-layout: fixed; width: 880px"><colgroup>
    <col style="width: 250px">
    <col style="width: 120px">
    <col style="width: 510px">
  </colgroup>
  <thead>
      <tr>
        <th>参数名</th>
        <th>输入/输出</th>
        <th>描述</th>
      </tr></thead>
  <tbody>
    <tr>
      <td>handle（asdBlasHandle）</td>
      <td>输入/输出</td>
      <td>asdBlasCreate接口的句柄。</td>
    </tr>
  </tbody>
    </table>

- **asdBlasSetStream**

    功能描述：将调用方持有的runtime stream与具体的plan实例进行绑定。调用方应确保stream在已提交的计算完成前有效，并单独销毁stream。\
    函数原型：AspbStatus asdBlasSetStream(asdBlasHandle handle, void *stream)\
    **参数说明：**
    <table style="undefined;table-layout: fixed; width: 880px"><colgroup>
    <col style="width: 250px">
    <col style="width: 120px">
    <col style="width: 510px">
  </colgroup>
  <thead>
      <tr>
        <th>参数名</th>
        <th>输入/输出</th>
        <th>描述</th>
      </tr></thead>
  <tbody>
    <tr>
      <td>handle（asdBlasHandle）</td>
      <td>输入</td>
      <td>asdBlasSetStream接口的句柄。</td>
    </tr>
    <tr>
      <td>stream（void *）</td>
      <td>输入</td>
      <td>指向流对象的指针。</td>
    </tr>
  </tbody>
    </table>

- **asdBlasDestroy**

    功能描述：销毁句柄及其绑定的plan，并释放plan自身持有的资源。调用后句柄失效。调用方传入的workspace和stream不会由此接口释放。\
    函数原型：AspbStatus asdBlasDestroy(asdBlasHandle handle)\
    **参数说明：**
    <table style="undefined;table-layout: fixed; width: 880px"><colgroup>
    <col style="width: 250px">
    <col style="width: 120px">
    <col style="width: 510px">
  </colgroup>
  <thead>
      <tr>
        <th>参数名</th>
        <th>输入/输出</th>
        <th>描述</th>
      </tr></thead>
  <tbody>
    <tr>
      <td>handle（asdBlasHandle）</td>
      <td>输入</td>
      <td>asdBlasDestroy接口的句柄。</td>
    </tr>
  </tbody>
    </table>

- **asdBlasSetWorkspace**

    功能描述：将调用方提供的workspace绑定到对应的plan。workspace由调用方持有；已提交的计算完成前须保持其有效，之后由调用方通过runtime释放。\
    函数原型：AspbStatus asdBlasSetWorkspace(asdBlasHandle handle, void *workSpace)\
    **参数说明：**
    <table style="undefined;table-layout: fixed; width: 880px"><colgroup>
    <col style="width: 250px">
    <col style="width: 120px">
    <col style="width: 510px">
  </colgroup>
  <thead>
      <tr>
        <th>参数名</th>
        <th>输入/输出</th>
        <th>描述</th>
      </tr></thead>
  <tbody>
    <tr>
      <td>handle（asdBlasHandle）</td>
      <td>输入</td>
      <td>asdBlasSetWorkspace接口的句柄。</td>
    </tr>
    <tr>
      <td>workSpace（void *）</td>
      <td>输入</td>
      <td>指向调用方持有的工作空间内存。该接口不会释放此内存。</td>
    </tr>
  </tbody>
    </table>

- **asdBlasSynchronize**

    功能描述：同步等待算子执行。\
    函数原型：AspbStatus asdBlasSynchronize(asdBlasHandle handle)\
    **参数说明：**
    <table style="undefined;table-layout: fixed; width: 880px"><colgroup>
    <col style="width: 250px">
    <col style="width: 120px">
    <col style="width: 510px">
  </colgroup>
  <thead>
      <tr>
        <th>参数名</th>
        <th>输入/输出</th>
        <th>描述</th>
      </tr></thead>
  <tbody>
    <tr>
      <td>handle（asdBlasHandle）</td>
      <td>输入</td>
      <td>asdBlasSynchronize接口的句柄。</td>
    </tr>
  </tbody>
    </table>

- **asdBlasGetWorkspaceSize**

    功能描述：计算所需workspace大小以及包含了算子计算流程的执行器。\
    函数原型：AspbStatus asdBlasGetWorkspaceSize(asdBlasHandle handle, size_t &workspaceSize);\
    **参数说明：**
    <table style="undefined;table-layout: fixed; width: 880px"><colgroup>
    <col style="width: 250px">
    <col style="width: 120px">
    <col style="width: 510px">
  </colgroup>
  <thead>
      <tr>
        <th>参数名</th>
        <th>输入/输出</th>
        <th>描述</th>
      </tr></thead>
  <tbody>
    <tr>
      <td>handle（asdBlasHandle）</td>
      <td>输入</td>
      <td>asdBlasGetWorkspaceSize接口的句柄。</td>
    </tr>
    <tr>
      <td>workspaceSize（size_t &）</td>
      <td>输入/输出</td>
      <td>所需工作空间大小。</td>
    </tr>
  </tbody>
    </table>
