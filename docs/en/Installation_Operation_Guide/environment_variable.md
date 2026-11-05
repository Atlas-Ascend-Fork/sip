# Environment Variables

<!-- md-trans-meta sourceCommit=123cf1bc29afe4aaef9c2dc77591172984259a9a translatedAt=2026-09-07T14:17:40.584Z pushedAt=2026-09-11T12:35:55.645Z -->

After the acceleration library is installed, a process-level environment variable setting script `set_env.sh` is provided to automatically complete environment variable configuration. The setting automatically becomes invalid after the user process exits.

## SiP Environment Variables

- **Basic environment variables:**
  <table style="undefined;table-layout: fixed; width: 650px"><colgroup>
    <col style="width: 250px">
    <col style="width: 400px">
  </colgroup>
  <thead>
      <tr>
        <th>Environment Variable Name</th>
        <th>Description</th>
      </tr></thead>
  <tbody>
    <tr>
      <td>ASDSIP_HOME_PATH</td>
      <td>Storage path of files after the software package is installed.</td>
    </tr>
     <tr>
      <td>LD_LIBRARY_PATH</td>
      <td>List of search paths for loading dynamic libraries on Linux.</td>
    </tr>
  </tbody>
    </table>

- **SiP-related environment variables:**
  <table style="undefined;table-layout: fixed; width: 650px"><colgroup>
    <col style="width: 250px">
    <col style="width: 400px">
  </colgroup>
  <thead>
      <tr>
        <th>Environment Variable Name</th>
        <th>Description</th>
      </tr></thead>
  <tbody>
    <tr>
      <td>ASCEND_PROCESS_LOG_PATH</td>
      <td>Sets the log storage path.</td>
    </tr>
     <tr>
      <td>ASCEND_SLOG_PRINT_TO_STDOUT</td>
      <td>Specifies whether to enable log printing. When enabled, logs are not saved to a log file but are directly printed and displayed.</td>
    </tr>
     <tr>
      <td>ASCEND_GLOBAL_LOG_LEVEL</td>
      <td>Sets the log level of application logs and the log level of each module. Only debug logs are supported.</td>
    </tr>
     <tr>
       <td>ASCEND_MODULE_LOG_LEVEL</td>
       <td>Sets the log level of each module for application logs. Only debug logs are supported.</td>
     </tr>

  </tbody>

  </table>

- **2D C2C experimental path switches (Ascend950 only):**

  The following switches control the experimental computing paths of 2D C2C FFT
  (`asdFftMakePlan2D` + `asdFftExecC2C`) on Ascend950. They are **disabled by default**;
  when disabled, the default mixed-radix pipeline is used. The experimental paths are
  for performance experiments and issue diagnosis only; the performance and numerical
  behavior after enabling them may differ from the default path.

  <table style="undefined;table-layout: fixed; width: 650px"><colgroup>
    <col style="width: 250px">
    <col style="width: 400px">
  </colgroup>
  <thead>
      <tr>
        <th>Environment Variable Name</th>
        <th>Description</th>
      </tr></thead>
  <tbody>
    <tr>
      <td>ASDSIP_DENSE_GEMM_PASS</td>
      <td>GEMM short-dimension pass switch. Values ≤0 (default) disable it; values &gt;0 enable it. When enabled, plans whose short dimension is ≤64 (satisfying the 2L≤128 single N-tile condition) use a dense DFT matrix GEMM instead of the short-dimension transform pass.</td>
    </tr>
     <tr>
      <td>ASDSIP_FUSED_NPASS</td>
      <td>Fused N-dimension pass switch. Values ≤0 (default) disable it; values &gt;0 enable it. It takes effect only for plans whose long dimension is 500 and short dimension is within [1,52] (the fused kernel instance is hardcoded to 500×50).</td>
    </tr>
     <tr>
      <td>ASDSIP_HYBRID_PASS</td>
      <td>Hybrid kernel switch (single-kernel pipeline of AIC GEMM→UB tile + AIV fused N transform, replacing the GEMM pass + transpose + N pass). Values ≤0 (default) disable it; values &gt;0 enable it. It takes effect only for the 500×50 shape; when both this switch and ASDSIP_DENSE_GEMM_PASS are enabled, the hybrid kernel takes precedence.</td>
    </tr>
     <tr>
      <td>ASDSIP_HYBRID_DEBUG</td>
      <td>Hybrid kernel bypass diagnosis. 0 (default) disables it; 1=AIC bypass; 2=AIV bypass; 3=both-side bypass. It is meaningful only when ASDSIP_HYBRID_PASS is enabled.</td>
    </tr>
  </tbody>

  </table>

- **Environment variable length limit:**

  The maximum length of an environment variable value is **12800 bytes**. Environment variables exceeding this limit are ignored (without affecting other variables), and a WARN-level message is output to the log. If an environment variable you set does not take effect, check whether it exceeds the length limit.
