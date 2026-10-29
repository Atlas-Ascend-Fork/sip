# Ascend950 一维 FFT 优化验证

默认优化范围：水平连续布局，FP32/complex64，二次幂长度 N=32～8192，batch=N。
C2C 支持正逆向；R2C 正向、C2R 逆向启用，全部目标二次幂尺寸使用优化路径，不因性能退化而回退。
二维 batch=1 的同尺寸方阵也启用：保留原转置流程，实数轴 batch=N，半谱复数轴 batch=N/2+1。
其他长度、batch、平台和方向、非方阵及三维 Plan 保持原选路。逆变换不归一化。

实现复用已有寄存器 FFT 与 radix-4 模板，增加实数镜像频点成对转换、小尺寸单核融合、
短周期系数缓存、后续阶段系数连续读取和首级连续写回。只编译所选配置，没有运行时调参开关。

## 构建与运行

分别构建上游基线和本分支，安装到不同目录。在已配置 CANN 的 Ascend950 环境中执行：

```bash
bash tests/standalone/fft1d_optimized/build.sh /path/to/sip /path/to/candidate /path/to/bin
python3 tests/standalone/fft1d_optimized/run.py \
  --baseline /path/to/baseline --candidate /path/to/candidate \
  --binary /path/to/bin/fft_bench --device 2 --output /path/to/new-results
```

需要 Python NumPy。`--device` 使用核验过的 ACL 设备编号；测试不重置设备。
可先添加 `--smoke` 检查全部目标长度的短执行。每次运行使用新结果目录，保留失败记录。

主机测试检查选路边界和不同核数下的 UB 容量；布局测试检查首级交错顺序和系数访问边界。
设备验证覆盖一维和二维三个公开接口、C2C 双向、随机/零/脉冲/常量/Nyquist、原地输出和非目标回退。
原地用例每次执行前恢复输入，恢复不计入 Event；非原地用例检查输入未被改写。

精度以实际 FP32/complex64 输入生成 FP64 NumPy golden，相对 L2 ≤ 1e-3，且尺寸及有限性正确。
零 golden 要求误差范数 ≤ 1e-3。回退用例另比较两版输出逐字节一致。
正式计时每版预热10次、正反测试顺序各300个 ACL Event 样本，合并600个样本取中位数。
计时不含 Plan、分配、H2D/D2H，不启用 profiler。记录每组设备占用、库哈希和实际 kernel 选路。
原始结果目录包含设备和进程信息，仅在测试环境保存；PR 只包含汇总报告。

已完成的设备验证与性能结果见 [REPORT.md](REPORT.md)。
