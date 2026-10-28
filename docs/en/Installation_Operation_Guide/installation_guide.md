# Installation and Deployment

<!-- md-trans-meta sourceCommit=25a0c73916203ce4555696808917dc3c4c840c42 translatedAt=2026-10-10T12:22:23.587Z pushedAt=2026-10-14T09:43:52.659Z -->

See **[Install CANN](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/softwareinst/instg/instg_0089.html?OS=openEuler&InstallType=netyum)**, and install the signal processing acceleration library according to the following steps:

- Install Toolkit development kit package.
- Install the ops operator package.
- Install the NNAL neural network acceleration library.

Run the acceleration library:
Ensure that the environment variables have been correctly configured by referring to the following example commands. The default path "${HOME}/Ascend" after installation as a non-root user is used as an example. Replace it with the actual path of `set_env.sh` as needed.

```Cpp
source ${HOME}/Ascend/nnal/asdsip/set_env.sh
```

The preceding environment variable settings take effect only in the current window. You can write the preceding commands to an environment variable configuration file (such as the .bashrc file) as needed. For the environment variable list, see [Environment Variable Reference](../Installation_Operation_Guide/environment_variable.md).
