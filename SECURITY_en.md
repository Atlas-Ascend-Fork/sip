# Security Statement

## Run User Recommendations

For security reasons, it is recommended not to use administrator accounts such as root to execute any command. Follow the principle of least privilege.

## File Permission Control

- It is recommended that users set the running system umask to 0027 or higher on the host (including the host machine) and in containers, ensuring that new folders have a default maximum permission of 750 and new files have a default maximum permission of 640.
- It is recommended that users take security measures such as permission control for sensitive content, including personal privacy data, commercial assets, source files, and various files saved during operator development. For example, for permission control of the installation directory of this project and permission control of input public data files, refer to [A-Recommended Maximum Permissions for Files (Folders) in Different Scenarios](#permission-control) when setting permissions.
- When operators run, operator compilation files may be cached in the `kernel_meta_*` folder under the running directory to speed up subsequent operator calls. Users can perform permission control on the generated files as needed.
- Permission control is required during user installation and usage. It is recommended to set permissions with reference to the file permission reference in [A-Recommended Maximum Permissions for Files (Folders) in Different Scenarios](#permission-control).

## Build Security Statement

When compiling and installing this project from source code, you need to compile it yourself. Some intermediate files are generated during compilation. It is recommended that you perform proper permission control on the intermediate files after compilation to ensure file security.

## Run Security Statement

- It is recommended that users write corresponding operator invocation scripts based on the resource status of the running environment. If the operator invocation scripts do not match the resource status, for example, the space used for generating input data or benchmark calculation results exceeds the memory capacity limit, or the data saved locally by the scripts exceeds the disk space, errors may occur and the process may exit unexpectedly.
- When an operator runs abnormally, it exits the process and prints error information. It is recommended to locate the specific error cause according to the error message, including setting operators to execute synchronously and viewing log files.
- When operators are invoked through [PyTorch](https://gitcode.com/Ascend/pytorch), running errors may occur due to version mismatch. For details, refer to the [PyTorch Security Statement](https://gitcode.com/Ascend/pytorch/blob/master/SECURITYNOTE.md).

## Vulnerability Mechanism Description

[Vulnerability Management](https://gitcode.com/cann/community/blob/master/security/security.md)

## Appendix

### <a name="permission-control"></a>A-Recommended Maximum Permissions for Files (Folders) in Different Scenarios

| Type           | Linux Permission Reference Maximum |
| -------------- | ---------------  |
| User home directory                        |   750 (rwxr-x---)            |
| Program files (including script files, library files, and so on)       |   550 (r-xr-x---)             |
| Program file directories                      |   550 (r-xr-x---)            |
| Configuration files                          |  640 (rw-r-----)             |
| Configuration file directories                      |   750 (rwxr-x---)            |
| Log files (recording completed or archived)        |  440 (r--r-----)             |
| Log files (being recorded)                |    640 (rw-r-----)           |
| Log file directories                      |   750 (rwxr-x---)            |
| Debug files                         |  640 (rw-r-----)         |
| Debug file directories                     |   750 (rwxr-x---)  |
| Temporary file directories                      |   750 (rwxr-x---)   |
| Maintenance and upgrade file directories                  |   770 (rwxrwx---)    |
| Service data files                      |   640 (rw-r-----)    |
| Service data file directories                  |   750 (rwxr-x---)      |
| Key component, private key, certificate, and ciphertext file directories    |  700 (rwx------)      |
| Key components, private keys, certificates, and encrypted ciphertext        | 600 (rw-------)      |
| Encryption/decryption interfaces and scripts            |   500 (r-x------)        |
