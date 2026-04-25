"""Triton W4A16 GEMM (prefill, M > 1).

x: [M, K] bf16
w_packed : [N, K/2]
scales: [N, K/G] fp16
y: [M, N] bf16

Dequantizes int4 -> bf16 * group_scale on the fly with fp32 accumulate.
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "GROUP_M": 8}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "GROUP_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "GROUP_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "GROUP_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "GROUP_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 128, "GROUP_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "GROUP_M": 8}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "GROUP_M": 8}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "GROUP_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "GROUP_M": 8}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "GROUP_M": 8}, num_stages=5, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32, "GROUP_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32, "GROUP_M": 8}, num_stages=4, num_warps=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "GROUP_M": 8}, num_stages=4, num_warps=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 64, "GROUP_M": 8}, num_stages=5, num_warps=2),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _w4a16_gemm_kernel(
    x_ptr, w_ptr, s_ptr, y_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_sn, stride_sg,
    stride_ym, stride_yn,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    BLOCK_K: tl.constexpr = GROUP_SIZE

    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m  = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n  = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k  = tl.arange(0, BLOCK_K)
    offs_kp = tl.arange(0, BLOCK_K // 2)

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :]  * stride_xk
    w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_kp[None, :] * stride_wk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    num_k_iters = tl.cdiv(K, BLOCK_K)
    for k_it in range(num_k_iters):
        k_base = k_it * BLOCK_K

        x_mask = (offs_m[:, None] < M) & ((k_base + offs_k[None, :]) < K)
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)

        w_mask = (offs_n[:, None] < N) & ((k_base // 2 + offs_kp[None, :]) < K // 2)
        w_packed = tl.load(w_ptrs, mask=w_mask, other=0)

        lo = (w_packed & 0xF).to(tl.int8)
        hi = ((w_packed >> 4) & 0xF).to(tl.int8)
        lo = tl.where(lo >= 8, lo - 16, lo)
        hi = tl.where(hi >= 8, hi - 16, hi)
        w_unpacked = tl.interleave(lo, hi)

        s = tl.load(
            s_ptr + offs_n * stride_sn + k_it * stride_sg,
            mask=offs_n < N, other=0.0,
        )
        w_bf16 = (w_unpacked.to(tl.float32) * s[:, None].to(tl.float32)).to(tl.bfloat16)

        acc += tl.dot(x, tl.trans(w_bf16), allow_tf32=False, out_dtype=tl.float32)

        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += (BLOCK_K // 2) * stride_wk

    y = acc.to(tl.bfloat16)
    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    y_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(y_ptrs, y, mask=y_mask)


def triton_w4a16_gemm(
    x: torch.Tensor,
    w_packed: torch.Tensor,
    scales: torch.Tensor,
    group_size: int = 128,
) -> torch.Tensor:
    M, K = x.shape
    N = w_packed.shape[0]
    y = torch.empty((M, N), device=x.device, dtype=torch.bfloat16)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )

    _w4a16_gemm_kernel[grid](
        x, w_packed, scales, y,
        M, N, K,
        x.stride(0), x.stride(1),
        w_packed.stride(0), w_packed.stride(1),
        scales.stride(0), scales.stride(1),
        y.stride(0), y.stride(1),
        GROUP_SIZE=group_size,
    )
    return y
