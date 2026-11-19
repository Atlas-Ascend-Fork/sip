# FFT公共接口

- **asdFftCreate**

    功能描述：注册FFT句柄。\
    函数原型：AspbStatus asdFftCreate(asdFftHandle &handle)\
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
      <td>handle（asdFftHandle）</td>
      <td>输出</td>
      <td>用于接收新创建的FFT句柄。</td>
    </tr>
  </tbody>
    </table>

- **asdFftSetStream**

    功能描述：绑定NPU执行流。\
    函数原型：AspbStatus asdFftSetStream(asdFftHandle handle, void *stream)\
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
      <td>handle（asdFftHandle）</td>
      <td>输入</td>
      <td>asdFftSetStream接口的句柄。</td>
    </tr>
    <tr>
      <td>stream（void *）</td>
      <td>输入</td>
      <td>指向流对象的指针。</td>
    </tr>
  </tbody>
    </table>

- **asdFftDestroy**

    功能描述：销毁句柄并释放句柄占用的空间。\
    函数原型：AspbStatus asdFftDestroy(asdFftHandle handle)\
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
      <td>handle（asdFftHandle）</td>
      <td>输入</td>
      <td>asdFftDestroy接口的句柄。</td>
    </tr>
  </tbody>
    </table>

- **asdFftGetWorkspaceSize**

    功能描述：计算当前plan下的FFT执行流需要的workspace的大小。\
    函数原型：AspbStatus asdFftGetWorkspaceSize(asdFftHandle handle, size_t &workSize)\
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
      <td>handle（asdFftHandle）</td>
      <td>输入</td>
      <td>asdFftGetWorkspaceSize接口的句柄。</td>
    </tr>
    <tr>
      <td>workSize（size_t &）</td>
      <td>输入</td>
      <td>所需的工作空间大小。</td>
    </tr>
  </tbody>
    </table>

- **asdFftSetWorkspace**

    功能描述：配置当前handle绑定的FFT计算过程所需的workspace。\
    函数原型：AspbStatus asdFftSetWorkspace(asdFftHandle handle, void *workspace)\
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
      <td>handle（asdFftHandle）</td>
      <td>输入</td>
      <td>asdFftSetWorkspace接口的句柄。</td>
    </tr>
    <tr>
      <td>workspace（void *）</td>
      <td>输入</td>
      <td>指向工作空间的指针。</td>
    </tr>
  </tbody>
    </table>

- **asdFftSynchronize**

    功能描述：同步NPU状态。\
    函数原型：AspbStatus asdFftSynchronize(asdFftHandle handle)\
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
      <td>handle（asdFftHandle）</td>
      <td>输入</td>
      <td>asdFftSynchronize接口的句柄。</td>
    </tr>
  </tbody>
    </table>

- **asdFftGetType**

    功能描述：返回当前handle绑定FFT计算的类型，包括ASCEND_FFT_C2C、ASCEND_FFT_C2R、ASCEND_FFT_R2C。\
    函数原型：AspbStatus asdFftGetType(asdFftHandle handle, asdFftType &fftType)\
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
      <td>handle（asdFftHandle）</td>
      <td>输入</td>
      <td>asdFftGetType接口的句柄。</td>
    </tr>
    <tr>
      <td>fftType（asdFftType）</td>
      <td>输入/输出</td>
      <td>用于接收FFT类型的值。</td>
    </tr>
  </tbody>
    </table>
