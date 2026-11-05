# 环境变量参考

加速库安装完成后，提供进程级环境变量设置脚本“set_env.sh”，以自动完成环境变量设置，用户进程结束后自动失效。

## 信号加速库环境变量说明

- **基础环境变量：**
  <table style="undefined;table-layout: fixed; width: 650px"><colgroup>
    <col style="width: 250px">
    <col style="width: 400px">
  </colgroup>
  <thead>
      <tr>
        <th>环境变量名</th>
        <th>说明</th>
      </tr></thead>
  <tbody>
    <tr>
      <td>ASDSIP_HOME_PATH</td>
      <td>软件包安装后文件存储路径。</td>
    </tr>
     <tr>
      <td>LD_LIBRARY_PATH</td>
      <td>Linux系统中加载动态库时的搜索路径列表。</td>
    </tr>
  </tbody>
    </table>

- **信号加速库相关环境变量：**
  <table style="undefined;table-layout: fixed; width: 650px"><colgroup>
    <col style="width: 250px">
    <col style="width: 400px">
  </colgroup>
  <thead>
      <tr>
        <th>环境变量名</th>
        <th>说明</th>
      </tr></thead>
  <tbody>
    <tr>
      <td>ASCEND_PROCESS_LOG_PATH</td>
      <td>设置日志保存路径。</td>
    </tr>
     <tr>
      <td>ASCEND_SLOG_PRINT_TO_STDOUT</td>
      <td>是否开启日志打印。开启后，日志将不会保存在log文件中，而是将产生的日志直接打印显示。</td>
    </tr>
     <tr>
      <td>ASCEND_GLOBAL_LOG_LEVEL</td>
      <td>设置应用类日志的日志级别及各模块日志级别，仅支持调试日志。</td>
    </tr>
     <tr>
       <td>ASCEND_MODULE_LOG_LEVEL</td>
       <td>设置应用类日志的各模块日志级别，仅支持调试日志。</td>
     </tr>

  </tbody>

  </table>

- **2D C2C 实验路径开关（仅 Ascend950 生效）：**

  以下开关控制 2D C2C FFT（`asdFftMakePlan2D` + `asdFftExecC2C`）在 Ascend950 上的
  实验性计算路径，**默认全部关闭**，关闭时走默认混合基流水。实验路径仅供性能实验
  与问题定位使用，开启后的性能与数值行为可能与默认路径存在差异。

  <table style="undefined;table-layout: fixed; width: 650px"><colgroup>
    <col style="width: 250px">
    <col style="width: 400px">
  </colgroup>
  <thead>
      <tr>
        <th>环境变量名</th>
        <th>说明</th>
      </tr></thead>
  <tbody>
    <tr>
      <td>ASDSIP_DENSE_GEMM_PASS</td>
      <td>GEMM 短维 pass 开关。取值 ≤0（默认）关闭；&gt;0 开启。开启后，短维长度 ≤64（满足 2L≤128 单 N-tile 条件）的 plan 使用稠密 DFT 矩阵 GEMM 替代短维变换 pass。</td>
    </tr>
     <tr>
      <td>ASDSIP_FUSED_NPASS</td>
      <td>N 维融合 pass 开关。取值 ≤0（默认）关闭；&gt;0 开启。仅对长维为 500、短维在 [1,52] 范围内的 plan 生效（融合内核实例按 500×50 硬编码）。</td>
    </tr>
     <tr>
      <td>ASDSIP_HYBRID_PASS</td>
      <td>混合核开关（AIC GEMM→UB 瓦片 + AIV 融合 N 变换单核流水，替代 GEMM pass + 转置 + N pass）。取值 ≤0（默认）关闭；&gt;0 开启。仅对 500×50 形状生效；与 ASDSIP_DENSE_GEMM_PASS 同时开启时混合核优先。</td>
    </tr>
     <tr>
      <td>ASDSIP_HYBRID_DEBUG</td>
      <td>混合核内核旁路诊断。0（默认）关闭；1=AIC 旁路；2=AIV 旁路；3=双侧旁路。仅在 ASDSIP_HYBRID_PASS 开启时有意义。</td>
    </tr>
  </tbody>

  </table>

- **环境变量长度限制：**

 环境变量值的长度上限为 **12800 字节**。超长的环境变量将被忽略（不影响其它变量），并在日志中输出 WARN 级别提示。若发现设置的环境变量未生效，请检查是否超出长度限制。
