# Environment Variables

<!-- md-trans-meta sourceCommit=123cf1bc29afe4aaef9c2dc77591172984259a9a translatedAt=2026-10-10T12:22:07.064Z pushedAt=2026-10-14T09:43:19.559Z -->

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

- **Environment variable length limit:**

 The maximum length of an environment variable value is **12800 bytes**. Environment variables exceeding this limit are ignored (without affecting other variables), and a WARN-level message is output to the log. If an environment variable you set does not take effect, check whether it exceeds the length limit.
