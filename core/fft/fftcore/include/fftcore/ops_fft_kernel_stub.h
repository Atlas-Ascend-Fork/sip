/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_FFT_KERNEL_STUB_H
#define OPS_FFT_KERNEL_STUB_H

#include <cstdint>
#include <cstdlib>
#include <cstring>

#include <mki/utils/rt/rt.h>

#include "utils/aspb_status.h"

// dft 值传 tiling: 逐字段同源于 ops-fft Fft1DDFTTilingData / AsdSip::DftTilingData;
// dft_r2c/dft_c2r 的 tiling 为指针传递, 布局由 device buffer 保证
struct OpsFftDftTilingData {
    int32_t batchSize;
    int32_t m;
    int32_t n;
    int32_t k;
    int32_t transA;
    int32_t transB;
    int32_t m0;
    int32_t n0;
    int32_t k0;
    int32_t loopTime;
    int32_t batchNumPreLoop;
    int32_t batchNumPreCore;
    int32_t batchTailNum;
    int32_t batchTailNumPreCore;
    int32_t batchTailCoreNum;
};

static_assert(sizeof(OpsFftDftTilingData) == 60, "Fft1DDFTTilingData layout mismatch");

// ops-fft 预编译制品 libcann_ops_fft.so 导出的 kernel host stub (bisheng <<<>>> 生成物):
//   - fft_b: 910B(arch32) C2C radix-2, tiling 对应 sip FftBTilingData/Fft1DBTilingData
//   - fft_n: 910B(arch32) C2C radix-2 大点数, tiling 对应 sip FftNTilingData/Fft1DNTilingData
//   - fft_stride: 910B(arch32) C2C 纵向 stride, tiling 对应 sip FftStrideTilingData/Fft1DStrideTilingData
//   - fft_mix_1..6: 910B(arch32) C2C 混合基, tiling 对应 sip FftAllMixTilingData
//   - dft/dft_r2c/dft_c2r: 910B(arch32) DFT 矩阵乘族 (tiling 值传递)
//   - fft_r2c_1..3(_even/_odd)/fft_c2r_1..3(_even/_odd): 910B(arch32) R2C/C2R 混合基
//   - dd: 910B(arch32) 2D 融合, tiling 对应 sip DdTilingData/Fft2DTilingData
//   - fft_c2c_arch35_mix_multi_core: 950(arch35) C2C 混合基与纯 2 幂共用, tiling 对应 sip FftAllMixTilingData
//   - fft_c2c_arch35_multi_core: 950(arch35) C2C radix-2, 即 sip 97769d8 旧版 kernel 的移植
//     (数值等价判据, tiling/常量契约详见 fft_c2c_arch35.cpp)
//   - fft_r2c_multi_core: 950(arch35) R2C 混合基, 多 tw_post_process 常量, tiling 对应 sip FftAllMixTilingData
//   - fft_c2r_multi_core: 950(arch35) C2R 混合基, tiling 对应 sip FftAllMixTilingData
//   - rfft1_d_fast_dft: 950(arch35) R2C fast dft, tiling 与 sip 不同源 (4 指针 + 140B 值传),
//     暂不接入, 请求时由统一拦截报错
// stub 声明为 weak (存件 so 只含单一 SoC 的 arch 产物): 任一存件下均可加载, 缺失侧为 nullptr
extern "C" {
__attribute__((weak)) void fft_b(uint32_t blocks, void* unused, void* stream, void* ffts_addr, void* gm_input,
                                 void* gm_w, void* gm_t, void* gm_index, void* gm_output, void* gm_workspace,
                                 void* gm_tiling);
__attribute__((weak)) void fft_n(uint32_t blocks, void* unused, void* stream, void* ffts_addr, void* gm_input,
                                 void* gm_w, void* gm_t, void* gm_index, void* gm_output, void* gm_workspace,
                                 void* gm_tiling);
__attribute__((weak)) void fft_stride(uint32_t blocks, void* unused, void* stream, void* ffts_addr, void* gm_input,
                                      void* gm_s_matrix, void* gm_output, void* gm_workspace, void* gm_tiling);
#define OPS_FFT_MIX_STUB(idx)                                                                              \
    __attribute__((weak)) void fft_mix_##idx(                                                              \
        uint32_t blocks, void* unused, void* stream, void* ffts_addr, void* gm_input, void* gm_dft_matrix, \
        void* gm_tw_matrix, void* gm_radix_list, void* gm_output, void* gm_workspace, void* gm_tiling)
OPS_FFT_MIX_STUB(1);
OPS_FFT_MIX_STUB(2);
OPS_FFT_MIX_STUB(3);
OPS_FFT_MIX_STUB(4);
OPS_FFT_MIX_STUB(5);
OPS_FFT_MIX_STUB(6);
#undef OPS_FFT_MIX_STUB

// R2C/C2R 混合基: 常量为输入输出索引 + A/B 表 + dft/tw/radix, 按 aivSplitWay(1-3) 与奇偶(parity) 选变体
#define OPS_FFT_R2C_C2R_STUB(name, suffix)                                                                  \
    __attribute__((weak)) void name##suffix(uint32_t blocks, void* unused, void* stream, void* ffts_addr,   \
                                            void* gm_input, void* gm_input_index, void* gm_a, void* gm_b,   \
                                            void* gm_output_index, void* gm_dft_matrix, void* gm_tw_matrix, \
                                            void* gm_radix_list, void* gm_output, void* gm_workspace, void* gm_tiling)
OPS_FFT_R2C_C2R_STUB(fft_r2c_, 1);
OPS_FFT_R2C_C2R_STUB(fft_c2r_, 1);
OPS_FFT_R2C_C2R_STUB(fft_r2c_, 2);
OPS_FFT_R2C_C2R_STUB(fft_c2r_, 2);
OPS_FFT_R2C_C2R_STUB(fft_r2c_, 3);
OPS_FFT_R2C_C2R_STUB(fft_c2r_, 3);
OPS_FFT_R2C_C2R_STUB(fft_r2c_, 1_even);
OPS_FFT_R2C_C2R_STUB(fft_c2r_, 1_even);
OPS_FFT_R2C_C2R_STUB(fft_r2c_, 2_even);
OPS_FFT_R2C_C2R_STUB(fft_c2r_, 2_even);
OPS_FFT_R2C_C2R_STUB(fft_r2c_, 3_even);
OPS_FFT_R2C_C2R_STUB(fft_c2r_, 3_even);
OPS_FFT_R2C_C2R_STUB(fft_r2c_, 1_odd);
OPS_FFT_R2C_C2R_STUB(fft_c2r_, 1_odd);
OPS_FFT_R2C_C2R_STUB(fft_r2c_, 2_odd);
OPS_FFT_R2C_C2R_STUB(fft_c2r_, 2_odd);
OPS_FFT_R2C_C2R_STUB(fft_r2c_, 3_odd);
OPS_FFT_R2C_C2R_STUB(fft_c2r_, 3_odd);
#undef OPS_FFT_R2C_C2R_STUB

// dft 的 tiling 为裸栈传递 (stub 从 [sp+0] 读 60 字节), 与 AAPCS64 值传参约定不符,
// 须经 dft_core.cpp 的 OpsFftCallDft 汇编垫片; dft_r2c/dft_c2r 为指针传递, 可直调
__attribute__((weak)) void dft(uint32_t blocks, void* unused, void* stream, void* gm_input, void* gm_dft_matrix,
                               void* gm_output, void* gm_workspace, OpsFftDftTilingData tiling);
__attribute__((weak)) void dft_r2c(uint32_t blocks, void* unused, void* stream, void* gm_input, void* gm_dft_matrix,
                                   void* gm_output, void* gm_workspace, void* gm_tiling);
__attribute__((weak)) void dft_c2r(uint32_t blocks, void* unused, void* stream, void* gm_input, void* gm_dft_matrix,
                                   void* gm_output, void* gm_workspace, void* gm_tiling);
__attribute__((weak)) void dd(uint32_t blocks, void* unused, void* stream, void* ffts_addr, void* gm_input,
                              void* gm_p_matrix, void* gm_q_matrix, void* gm_output, void* gm_workspace,
                              void* gm_tiling);
__attribute__((weak)) void fft_c2c_arch35_mix_multi_core(uint32_t blocks, void* unused, void* stream, void* gm_input,
                                                         void* gm_dft_matrix, void* gm_tw_matrix, void* gm_radix_list,
                                                         void* gm_output, void* gm_workspace, void* gm_tiling);
// 950(arch35) 三内核经反汇编核对: 薄启动器逐一转发 device 参数指针, tiling 均指针传递, 无需垫片
__attribute__((weak)) void fft_c2c_arch35_multi_core(uint32_t blocks, void* unused, void* stream, void* gm_input,
                                                     void* gm_dft_matrix, void* gm_tw_matrix, void* gm_radix_list,
                                                     void* gm_output, void* gm_workspace, void* gm_tiling);
__attribute__((weak)) void fft_r2c_multi_core(uint32_t blocks, void* unused, void* stream, void* gm_input,
                                              void* gm_dft_matrix, void* gm_tw_matrix, void* gm_tw_post_process,
                                              void* gm_radix_list, void* gm_output, void* gm_workspace,
                                              void* gm_tiling);
__attribute__((weak)) void fft_c2r_multi_core(uint32_t blocks, void* unused, void* stream, void* gm_input,
                                              void* gm_dft_matrix, void* gm_tw_matrix, void* gm_radix_list,
                                              void* gm_output, void* gm_workspace, void* gm_tiling);
}

// CANN runtime: 硬件同步地址 (kernel 跨核同步用), 原型声明避免引入 acl 头依赖
extern "C" int aclrtGetHardwareSyncAddr(void** addr);

namespace AsdSip {

// 后端开关: SIP_FFT_BACKEND=ops-fft 为强制请求(未适配/缺件即报错, 不回退);
// SIP_FFT_BACKEND=ops-fft-auto 为自动模式(适配且 stub 就绪的 step 走 ops-fft,
// 其余 step 打 ERROR 日志后自动回退 internal, 见 OpsFftBackendUnifyIntercept);
// 未设置时: 联合构建(存件已链接)默认严格档, 杜绝静默 internal; 纯 internal 构建仍为
// internal; 显式 internal 与其他未识别取值均为 internal (sip 自身 kernel)
inline bool UseOpsFftKernelBackend()
{
    const char* backend = std::getenv("SIP_FFT_BACKEND");
    if (backend == nullptr || backend[0] == '\0') {
        // 联合构建检测: 910B 存件含 fft_b, 950 存件含 arch35 mix, 任一非空即已链接存件
        return fft_b != nullptr || fft_c2c_arch35_mix_multi_core != nullptr;
    }
    return std::strcmp(backend, "ops-fft") == 0 || std::strcmp(backend, "ops-fft-auto") == 0;
}

// 自动回退档位: 仅 ops-fft-auto 为真 (严格 ops-fft 下拦截器仍全有或全无)
inline bool UseOpsFftAutoFallback()
{
    const char* backend = std::getenv("SIP_FFT_BACKEND");
    return backend != nullptr && std::strcmp(backend, "ops-fft-auto") == 0;
}

// stub 可用性判空 (weak 符号在当前 so 中缺失时为 nullptr)
inline bool OpsFftBStubAvailable() { return fft_b != nullptr; }

inline bool OpsFftNStubAvailable() { return fft_n != nullptr; }

inline bool OpsFftStrideStubAvailable() { return fft_stride != nullptr; }

inline bool OpsFftMixStubAvailable()
{
    return fft_mix_1 != nullptr && fft_mix_2 != nullptr && fft_mix_3 != nullptr && fft_mix_4 != nullptr &&
           fft_mix_5 != nullptr && fft_mix_6 != nullptr;
}

inline bool OpsFftDftStubAvailable() { return dft != nullptr; }

inline bool OpsFftDftR2CStubAvailable() { return dft_r2c != nullptr; }

inline bool OpsFftDftC2RStubAvailable() { return dft_c2r != nullptr; }

// R2C/C2R 混合基按 aivSplitWay(1-3) 与奇偶选变体, 存件须同时含全部变体
inline bool OpsFftR2CStubAvailable()
{
    return fft_r2c_1 != nullptr && fft_r2c_2 != nullptr && fft_r2c_3 != nullptr && fft_r2c_1_even != nullptr &&
           fft_r2c_2_even != nullptr && fft_r2c_3_even != nullptr && fft_r2c_1_odd != nullptr &&
           fft_r2c_2_odd != nullptr && fft_r2c_3_odd != nullptr;
}

inline bool OpsFftC2RStubAvailable()
{
    return fft_c2r_1 != nullptr && fft_c2r_2 != nullptr && fft_c2r_3 != nullptr && fft_c2r_1_even != nullptr &&
           fft_c2r_2_even != nullptr && fft_c2r_3_even != nullptr && fft_c2r_1_odd != nullptr &&
           fft_c2r_2_odd != nullptr && fft_c2r_3_odd != nullptr;
}

inline bool OpsFftDdStubAvailable() { return dd != nullptr; }

inline bool OpsFftArch35MixStubAvailable() { return fft_c2c_arch35_mix_multi_core != nullptr; }

inline bool OpsFftArch35Radix2StubAvailable() { return fft_c2c_arch35_multi_core != nullptr; }

inline bool OpsFftArch35R2CStubAvailable() { return fft_r2c_multi_core != nullptr; }

inline bool OpsFftArch35C2RStubAvailable() { return fft_c2r_multi_core != nullptr; }

// ops-fft 常量 H2D 拷贝 (plan 级缓存判空语义的配套原语):
// *dev 非空即代表拷贝成功过; 重试先释放旧 buffer, 任一失败路径都释放并置空,
// 保证外层判空缓存在重试下不泄漏、不残留脏数据
inline bool OpsFftUploadConst(void** dev, const void* host, size_t size)
{
    if (dev == nullptr || host == nullptr || size == 0) {
        return false;
    }
    if (*dev != nullptr) {
        Mki::MkiRtMemFreeDevice(*dev);
        *dev = nullptr;
    }
    if (Mki::MkiRtMemMallocDevice(dev, size, MKIRT_MEM_DEFAULT) != MKIRT_SUCCESS) {
        *dev = nullptr;
        return false;
    }
    if (Mki::MkiRtMemCopy(*dev, size, host, size, MKIRT_MEMCOPY_HOST_TO_DEVICE) != MKIRT_SUCCESS) {
        Mki::MkiRtMemFreeDevice(*dev);
        *dev = nullptr;
        return false;
    }
    return true;
}

// ops-fft 910B launch 的核数语义 (与 ops-fft host 侧一致):
//   - 全核: GetCoreNumAic(), 最小 1 (fft_n/fft_stride/dft 族/dd)
//   - 按 batch 收缩: min(batch, min(coreNum, 20)), 最小 1 (fft_b/fft_mix/fft_r2c/fft_c2r)
// ops-fft 950(arch35) launch 的核数语义: __vector__/__aicore__ kernel 全 AIV 核数
// (GetCoreNumAiv, 最小 1)——aclfftFft1DC2C/aclfftIrfft1DFft 反汇编实测, 与 sip 各
// arch35 tiling 的 SetBlockDim(AIV 全核) 一致
inline uint32_t OpsFftAllCoreNum(uint32_t cubeCoreNum) { return cubeCoreNum == 0 ? 1 : cubeCoreNum; }

inline uint32_t OpsFftBatchCoreNum(uint32_t batch, uint32_t cubeCoreNum)
{
    uint32_t maxCore = cubeCoreNum > 20U ? 20U : cubeCoreNum;
    uint32_t need = batch > maxCore ? maxCore : batch;
    return need == 0 ? 1 : need;
}

class FFTPlan;

// ops-fft 后端统一拦截 (定义于 fft_api.cpp): 请求 ops-fft 时逐 step 校验
// 适配性与 stub 可用性, 任一不满足即报错中止(ACL_ERROR_API_NOT_SUPPORT), 不回退 internal
AspbStatus OpsFftBackendUnifyIntercept(const FFTPlan& plan);

} // namespace AsdSip

#endif // OPS_FFT_KERNEL_STUB_H
