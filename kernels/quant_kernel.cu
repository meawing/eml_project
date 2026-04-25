// Symmetric groupwise int4 quantizer.
//
// For each row of the [N, K] weight matrix we partition the K axis into
// groups of `group_size`. Within each group: scale = max(|w|) / 7,
// q = round(w / scale) clamped to [-8, 7]. Two int4 nibbles pack into one
// uint8 with the lower-k element in the low nibble. Scales are stored fp16.

#include <cuda_runtime.h>
#include <torch/extension.h>

#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>

namespace {

constexpr int INT4_MIN = -8;
constexpr int INT4_MAX = 7;
constexpr float INT4_MAX_F = 7.0f;

__device__ __forceinline__ int clamp_int4(int v) {
  if (v < INT4_MIN) return INT4_MIN;
  if (v > INT4_MAX) return INT4_MAX;
  return v;
}

__global__ void quantize_fp16_to_int4_kernel(
    const half* __restrict__ w,
    uint8_t* __restrict__ packed,
    half* __restrict__ scales,
    int n_rows,
    int k_cols,
    int group_size) {
  const int row = blockIdx.x;
  if (row >= n_rows) return;

  const int groups_per_row = k_cols / group_size;
  const int row_w_offset = row * k_cols;
  const int row_packed_offset = row * (k_cols / 2);
  const int row_scale_offset = row * groups_per_row;

  for (int g = 0; g < groups_per_row; ++g) {
    const int group_start = g * group_size;
    const int group_end = group_start + group_size;

    __shared__ float s_scale;
    if (threadIdx.x == 0) {
      float max_abs = 0.0f;
      for (int k = group_start; k < group_end; ++k) {
        const float v = __half2float(w[row_w_offset + k]);
        const float a = fabsf(v);
        if (a > max_abs) max_abs = a;
      }
      const float scale = (max_abs > 0.0f) ? (max_abs / INT4_MAX_F) : 1.0f;
      s_scale = scale;
      scales[row_scale_offset + g] = __float2half(scale);
    }
    __syncthreads();

    const int pair_start = group_start / 2;
    const int pair_end = group_end / 2;
    for (int p = pair_start + threadIdx.x; p < pair_end; p += blockDim.x) {
      const int k0 = p * 2;
      const int k1 = k0 + 1;
      const float inv_scale = 1.0f / s_scale;

      const float x0 = __half2float(w[row_w_offset + k0]);
      const float x1 = __half2float(w[row_w_offset + k1]);

      const int q0 = clamp_int4(__float2int_rn(x0 * inv_scale));
      const int q1 = clamp_int4(__float2int_rn(x1 * inv_scale));

      const uint8_t n0 = static_cast<uint8_t>(q0 & 0xF);
      const uint8_t n1 = static_cast<uint8_t>(q1 & 0xF);
      packed[row_packed_offset + p] = static_cast<uint8_t>(n0 | (n1 << 4));
    }
    __syncthreads();
  }
}

}  // namespace

std::vector<torch::Tensor> quantize_fp16_to_int4_cuda(
    torch::Tensor w,
    int64_t group_size) {
  auto w_contig = w.contiguous();
  const int64_t n_rows = w_contig.size(0);
  const int64_t k_cols = w_contig.size(1);
  const int64_t groups_per_row = k_cols / group_size;

  auto packed = torch::empty(
      {n_rows, k_cols / 2},
      torch::TensorOptions().device(w.device()).dtype(torch::kUInt8));
  auto scales = torch::empty(
      {n_rows, groups_per_row},
      torch::TensorOptions().device(w.device()).dtype(torch::kFloat16));

  at::cuda::CUDAGuard device_guard(w.device());
  const int threads = 256;
  const dim3 grid(n_rows);
  const dim3 block(threads);

  quantize_fp16_to_int4_kernel<<<grid, block>>>(
      reinterpret_cast<const half*>(w_contig.data_ptr<at::Half>()),
      packed.data_ptr<uint8_t>(),
      reinterpret_cast<half*>(scales.data_ptr<at::Half>()),
      static_cast<int>(n_rows),
      static_cast<int>(k_cols),
      static_cast<int>(group_size));

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {packed, scales};
}
