# -*- coding: utf-8 -*-
"""
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
"""

__version__ = "0.1.0"

from typing import Tuple, Union
import torch

# 副作用导入：加载 C++ 扩展以注册 torch.ops.torch_sip 算子族（issue #257 连带：
# 修改本文件触发 ruff F401，noqa 保留必要的副作用导入语义）
import sip_custom_ops_lib  # noqa: F401


def asd_mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.ops.torch_sip.asd_mul(x, y)


def conj(x: torch.Tensor) -> torch.Tensor:
    return torch.ops.torch_sip.conj(x)


def swap_last2_axes(x: torch.Tensor) -> torch.Tensor:
    return torch.ops.torch_sip.swap_last2_axes(x)


def asd_blas_asum(x: torch.Tensor) -> torch.Tensor:
    return torch.ops.torch_sip.asd_blas_asum(x)


def asd_blas_cal(x: torch.Tensor, alpha: Union[float, complex, int]) -> torch.Tensor:
    return torch.ops.torch_sip.asd_blas_cal(x, alpha)


def asd_blas_caxpy(
    x: torch.Tensor, y: torch.Tensor, alpha: Union[float, complex, int]
) -> torch.Tensor:
    if not isinstance(alpha, complex):
        alpha = complex(alpha)
    return torch.ops.torch_sip.asd_blas_caxpy(x, y, alpha)


def _get_trans_enum(trans: str) -> int:
    """内部辅助：将字符串转为底层 C++ 预期的整数枚举"""
    mapping = {"N": 0, "T": 1, "C": 2}
    t = trans.upper()
    if t not in mapping:
        raise ValueError(f"Invalid trans mode: {trans}. Expected 'N', 'T', or 'C'.")
    return mapping[t]


def asd_blas_cgemm(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    mat_c: torch.Tensor,
    alpha: Union[float, complex] = 1.0,
    beta: Union[float, complex] = 0.0,
    trans_a: str = "N",
    trans_b: str = "N",
) -> torch.Tensor:
    """
    复数矩阵乘加算子 (CGEMM): C = alpha * op(A) * op(B) + beta * C
    支持 A, B, C 的物理列优先排布。
    """
    t_a = _get_trans_enum(trans_a)
    t_b = _get_trans_enum(trans_b)

    # 强制转换为 complex，匹配 PTA 层的 Scalar 转换
    alpha_c = complex(alpha)
    beta_c = complex(beta)

    # 调用 torch.ops 中的 C++ 实现
    return torch.ops.torch_sip.asd_blas_cgemm(
        mat_a, mat_b, mat_c, alpha_c, beta_c, t_a, t_b
    )


def asd_blas_cgemm_batched(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    mat_c: torch.Tensor,
    alpha: Union[float, complex] = 1.0,
    beta: Union[float, complex] = 0.0,
    trans_a: str = "N",
    trans_b: str = "N",
) -> torch.Tensor:
    """
    复数矩阵乘加算子 (Batched CGEMM): C[i] = alpha * op(A[i]) * op(B[i]) + beta * C[i]
    输入 Tensor 形状应为 [Batch, M, N]。
    当前只支持alpha=1, beta=0, trans_a='N', trans_b='N'; 0 < m, k, n <= 32
    """
    t_a = _get_trans_enum(trans_a)
    t_b = _get_trans_enum(trans_b)

    alpha_c = complex(alpha)
    beta_c = complex(beta)

    return torch.ops.torch_sip.asd_blas_cgemm_batched(
        mat_a, mat_b, mat_c, alpha_c, beta_c, t_a, t_b
    )


def asd_blas_cgemv(mat_a, vec_x, vec_y, alpha=1.0, beta=0.0, trans="N"):
    t = _get_trans_enum(trans)

    alpha = complex(alpha)
    beta = complex(beta)
    return torch.ops.torch_sip.asd_blas_cgemv(mat_a, vec_x, vec_y, alpha, beta, t)


def asd_blas_cgerc(mat_a, vec_x, vec_y, alpha=1.0):
    """
    mat_a: (M, N) - 结果矩阵，将被原地修改
    vec_x: (M,)
    vec_y: (N,)
    alpha: complex scalar
    """
    alpha = complex(alpha)
    return torch.ops.torch_sip.asd_blas_cgerc(mat_a, vec_x, vec_y, alpha)


def asd_blas_cmatinv_batched(mat_a):
    """
    计算批处理复数矩阵求逆

    Args:
        mat_a: (Batch, N, N) 复数张量 (支持 torch.complex64, torch.complex32)

    Returns:
        Tensor: 形状与 mat_a 相同的逆矩阵
    """
    return torch.ops.torch_sip.asd_blas_cmatinv_batched(mat_a)


def asd_blas_ctrmv(mat_a, vec_x, uplo="L", trans="N", diag="N"):
    """
    单精度复数三角矩阵-向量乘法 (原地修改 vec_x)

    Args:
        mat_a: (N, N) 复数三角矩阵
        vec_x: (N,) 复数向量
        uplo: "L" (下三角), "U" (上三角)
        trans: "N" (无转置), "T" (转置) 或 "C" (共轭转置)
        diag: "N" (非单位对角) 或 "U" (单位对角，对角线元素视为1)
    """
    uplo_map = {"L": 0, "U": 1}
    trans_map = {"N": 0, "T": 1, "C": 2}
    diag_map = {"N": 0, "U": 1}

    # 非法选项显式报错（issue #257，对齐 strmm/strmv 的 _char_to_enum 口径）：
    # 原 dict.get(key, 0) 把拼错的 uplo/trans/diag 静默映射为默认值按错误语义执行
    u_val = _char_to_enum(uplo, uplo_map)
    t_val = _char_to_enum(trans, trans_map)
    d_val = _char_to_enum(diag, diag_map)

    return torch.ops.torch_sip.asd_blas_ctrmv(mat_a, vec_x, u_val, t_val, d_val)


def asd_blas_dot(vec_x, vec_y, conjugate=False):
    """
    向量点积计算 (支持实数与复数)

    Args:
        vec_x: 输入向量 x
        vec_y: 输入向量 y
        conjugate: 是否计算共轭点积 (仅限复数)
    """
    return torch.ops.torch_sip.asd_blas_dot(vec_x, vec_y, conjugate)


def asd_blas_iamax(vec_x):
    """
    寻找向量中绝对值（或复数实虚部绝对值之和）最大的元素索引

    Args:
        vec_x: (N,) 目标向量，支持 Float32 或 Complex64

    Returns:
        int: 最大元素对应的索引标量
    """
    # 获取底层的 INT32 索引张量
    res_tensor = torch.ops.torch_sip.asd_blas_iamax(vec_x)

    # 转为标量数值
    idx = res_tensor.item()
    return idx


def asd_blas_cgemv_batched(mat_a, vec_x, vec_y, alpha=1.0, beta=0.0, trans="N"):
    t = _get_trans_enum(trans)

    alpha = complex(alpha)
    beta = complex(beta)
    return torch.ops.torch_sip.asd_blas_cgemv_batched(
        mat_a, vec_x, vec_y, alpha, beta, t
    )


def asd_blas_copy(tensor_x, out=None):
    x_input = tensor_x.contiguous()

    if out is not None:
        if not out.is_contiguous():
            raise ValueError("asd_blas_copy: The 'out' tensor must be contiguous.")
        return torch.ops.torch_sip.asd_blas_copy_out(x_input, out)
    else:
        return torch.ops.torch_sip.asd_blas_copy(x_input)


def asd_blas_colwise_mul(mat, vec):
    return torch.ops.torch_sip.asd_blas_colwise_mul(mat, vec)


def asd_blas_complex_mat_dot(mat_x, mat_y):
    return torch.ops.torch_sip.asd_blas_complex_mat_dot(mat_x, mat_y)


def asd_blas_nrm2(vec_x):
    """
    计算向量的欧氏范数 (L2 Norm)
    内部自动根据数据类型分发到 Snrm2 (实数) 或 Scnrm2 (复数) 算子

    Args:
        vec_x: (N,) 目标向量，支持 Float32 或 Complex64

    Returns:
        float: 向量的欧氏范数标量值
    """
    res_tensor = torch.ops.torch_sip.asd_blas_nrm2(vec_x)
    return res_tensor.item()


def asd_blas_csrot(vec_x: torch.Tensor, vec_y: torch.Tensor, c: float, s: float):
    """
    公式:
        x' = c * x + s * y
        y' = -s * x + c * y

    Args:
        vec_x: (N,) 复数向量，将被原地覆盖更新
        vec_y: (N,) 复数向量，将被原地覆盖更新
        c: 旋转矩阵的余弦值 (实数)
        s: 旋转矩阵的正弦值 (实数)

    Returns:
        Tuple[Tensor, Tensor]
    """
    return torch.ops.torch_sip.asd_blas_csrot(vec_x, vec_y, float(c), float(s))


def asd_blas_ssyr(
    mat_a: torch.Tensor, vec_x: torch.Tensor, alpha: float = 1.0, uplo: str = "L"
):
    """
    公式:
        A = alpha * x * x^T + A
    Args:
        mat_a: (N, N) 对称矩阵，将被原地更新
        vec_x: (N,) 列向量
        alpha: 缩放因子
        uplo: "U" (仅更新上三角) 或 "L" (仅更新下三角)
    Returns:
        Tensor: 更新后的 mat_a 引用
    """
    # 映射枚举值
    uplo_map = {"L": 0, "U": 1}
    # 非法选项显式报错（issue #257，对齐 strmm/strmv 口径）
    u_val = _char_to_enum(uplo, uplo_map)

    return torch.ops.torch_sip.asd_blas_ssyr(mat_a, vec_x, alpha, u_val)


def asd_blas_ssyr2(
    mat_a: torch.Tensor,
    vec_x: torch.Tensor,
    vec_y: torch.Tensor,
    alpha: float = 1.0,
    uplo: str = "L",
):
    """
    公式:
        A = alpha * x * y^T + alpha * y * x^T + A
    Args:
        mat_a: (N, N) 目标对称矩阵，其内容将被原地覆盖更新
        vec_x: (N,) 输入向量 x
        vec_y: (N,) 输入向量 y
        alpha: 更新的缩放标量因子
        uplo: "U" (仅更新上三角) 或 "L" (仅更新下三角)
    Returns:
        Tensor: 更新完成后的 mat_a 引用
    """
    # 构建枚举值映射字典
    uplo_map = {"L": 0, "U": 1}
    # 非法选项显式报错（issue #257，对齐 strmm/strmv 口径）
    u_val = _char_to_enum(uplo, uplo_map)
    return torch.ops.torch_sip.asd_blas_ssyr2(mat_a, vec_x, vec_y, alpha, u_val)


def _char_to_enum(char_val: str, valid_dict: dict) -> int:
    char_upper = char_val.upper()
    if char_upper not in valid_dict:
        raise ValueError(
            f"Invalid option '{char_val}'. Expected one of {list(valid_dict.keys())}."
        )
    return valid_dict[char_upper]


def asd_blas_strmm(mat_a, mat_b, alpha=1.0, side="L", uplo="L", trans="N"):
    side_val = _char_to_enum(side, {"L": 0, "R": 1})
    uplo_val = _char_to_enum(uplo, {"L": 0, "U": 1})
    trans_val = _char_to_enum(trans, {"N": 0, "T": 1})

    alpha_val = float(alpha)

    return torch.ops.torch_sip.asd_blas_strmm(
        mat_a, mat_b, alpha_val, side_val, uplo_val, trans_val
    )


def asd_blas_strmv(mat_a, vec_x, uplo="L", trans="N", diag="N"):
    uplo_val = _char_to_enum(uplo, {"L": 0, "U": 1})
    trans_val = _char_to_enum(trans, {"N": 0, "T": 1})
    diag_val = _char_to_enum(diag, {"N": 0, "U": 1})

    return torch.ops.torch_sip.asd_blas_strmv(
        mat_a, vec_x, uplo_val, trans_val, diag_val
    )


def asd_blas_swap(vec_x, vec_y):
    return torch.ops.torch_sip.asd_blas_swap(vec_x, vec_y)


def asd_interp_with_coeff(x, coefficient):
    """
    向量插值算子 (InterpWithCoeff) 简化入口
    :param x: 输入信号张量，形状 [batch, nRs, totalSubcarrier]
    :param coefficient: 插值系数张量，形状 [batch, 14-nRs, nRs]
    :return: 输出张量，形状 [batch, 14-nRs, totalSubcarrier]
    """
    return torch.ops.torch_sip.asd_interp_with_coeff(x, coefficient)


def asd_convolve(signal, kernel):
    """
    一维滤波卷积算子 (Convolve) 简化入口
    :param signal: 输入的复数一维信号，形状 [BatchCount, n]
    :param kernel: 输入的实数滤波卷积核，形状 [k]
    :return: 滤波后的复数信号，形状 [BatchCount, n]
    """
    return torch.ops.torch_sip.asd_convolve(signal, kernel)


def rs_interpolation_by_sinc(
    input_tensor, sinc_tab, pos_floor, pos_to_tab_index, interp_num=16, quant_num=32
):
    """
    批量一维复数向量 Sinc 插值算子
    :param input_tensor: 原始信号，Complex64, 形状 [batch, signal_length]
    :param sinc_tab: 插值系数矩阵，Float32
    :param pos_floor: 坐标向下取整值，Int32, 形状 [batch, interp_length]
    :param pos_to_tab_index: 坐标小数映射的行号索引，Int16, 形状 [batch, interp_length]
    :param interp_num: 插值点数 (默认 16，需为偶数)
    :param quant_num: 量化点数 (默认 32，需为2的幂)
    :return: 插值结果，Complex64, 形状 [batch, interp_length]
    """
    return torch.ops.torch_sip.rs_interpolation_by_sinc(
        input_tensor, sinc_tab, pos_floor, pos_to_tab_index, interp_num, quant_num
    )


def asd_fft_c2c(input_tensor: torch.Tensor, is_forward: bool = True) -> torch.Tensor:
    return torch.ops.torch_sip.asd_fft_c2c(input_tensor, is_forward)


def asd_fft_c2c_sep(
    in_real: torch.Tensor, in_imag: torch.Tensor, is_forward: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.torch_sip.asd_fft_c2c_sep(in_real, in_imag, is_forward)


def asd_fft_r2c(input_tensor: torch.Tensor, is_forward: bool = True) -> torch.Tensor:
    """
    Perform Real-to-Complex FFT on NPU.
    Input: Real tensor (Float32)
    Output: Complex tensor (Complex64) with last dimension size n/2 + 1
    """
    return torch.ops.torch_sip.asd_fft_r2c(input_tensor, is_forward)


def asd_fft_c2r(input_tensor: torch.Tensor, is_forward: bool = True) -> torch.Tensor:
    return torch.ops.torch_sip.asd_fft_c2r(input_tensor, is_forward)
