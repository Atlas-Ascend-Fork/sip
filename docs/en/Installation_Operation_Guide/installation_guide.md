# Installation and Deployment

<!-- md-trans-meta sourceCommit=cc624cb911fb7f15fd153fbb95bbad1e3d963968 translatedAt=2026-09-07T14:17:57.652Z pushedAt=2026-09-11T12:35:55.647Z -->

See **[Install CANN](https://www.hiascend.com/document/detail/en/canncommercial/900/softwareinst/instg/instg_0089.html?OS=openEuler&InstallType=netyum)**, and install the signal processing acceleration library according to the following steps:

- Install Toolkit development kit package.
- Install the ops operator package.
- Install the NNAL neural network acceleration library.

Run the acceleration library:
Ensure that the environment variables have been correctly configured by referring to the following example commands. The default path "${HOME}/Ascend" after installation as a non-root user is used as an example. Replace it with the actual path of `set_env.sh` as needed.

```Cpp
source ${HOME}/Ascend/nnal/asdsip/set_env.sh
```

The above environment variable configuration takes effect only in the current window. You can write the above commands into an environment variable configuration file (such as the `.bashrc` file) as needed. For the list of environment variables, see *Environment Variables*.
