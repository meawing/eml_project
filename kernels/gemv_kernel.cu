// bf16 x int4 GEMV (M == 1 decode).
//
// One warp computes one output row. BN rows per block share a single
// activation vector x[K] staged into shared memory. W is streamed from
// HBM in 16-byte chunks (one uint4 = 32 packed int4 weights per thread)
// and dequantized on the fly with the per-group fp16 scale, fp32 accum,
// final reduction via warp shuffle.
//
// At M=1 CUDA-core dot-product replacement instead of tensor-core makes int4 decode faster
//

#include <cuda_runtime.h>
#include <torch/extension.h>

#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace {

constexpr int BN = 8;
constexpr int T_ROW = 32;
constexpr int T_BLOCK = BN * T_ROW;

constexpr int CHUNK_BYTES = 16;
constexpr int K_ALIGN = T_ROW * 2 * CHUNK_BYTES;  // 1024

__global__ void gemv_kernel(
    const __nv_bfloat16* __restrict__ x,
    const uint8_t* __restrict__ w_packed,
    const __half* __restrict__ scales,
    __nv_bfloat16* __restrict__ y,
    int N,
    int K,
    int group_size) {
  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int n = blockIdx.x * BN + warp_id;

  extern __shared__ __nv_bfloat16 smem_x[];

  const int total_vec = K / 8;
  for (int v = tid; v < total_vec; v += T_BLOCK) {
    *reinterpret_cast<uint4*>(&smem_x[v * 8]) =
        *reinterpret_cast<const uint4*>(&x[v * 8]);
  }
  __syncthreads();

  const int half_k = K / 2;
  const int groups_per_row = K / group_size;
  const int bytes_per_thread = K / (T_ROW * 2);
  const int w_row_base = n * half_k;
  const int scale_row_base = n * groups_per_row;
  const int my_byte_base = lane * bytes_per_thread;

  float acc = 0.0f;

  for (int off_bytes = 0; off_bytes < bytes_per_thread; off_bytes += CHUNK_BYTES) {
    int k_off = (my_byte_base + off_bytes) * 2;
    int g = k_off / group_size;
    float sc = __half2float(scales[scale_row_base + g]);

    uint4 raw = *reinterpret_cast<const uint4*>(
        &w_packed[w_row_base + my_byte_base + off_bytes]);
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&raw);

    float partial = 0.0f;
#pragma unroll
    for (int b = 0; b < CHUNK_BYTES; b++) {
      uint8_t byte = bytes[b];
      int lo_n = byte & 0xF;
      int hi_n = (byte >> 4) & 0xF;
      int lo = (lo_n >= 8) ? lo_n - 16 : lo_n;
      int hi = (hi_n >= 8) ? hi_n - 16 : hi_n;
      float x0 = __bfloat162float(smem_x[k_off + b * 2]);
      float x1 = __bfloat162float(smem_x[k_off + b * 2 + 1]);
      partial += x0 * static_cast<float>(lo) + x1 * static_cast<float>(hi);
    }
    acc += partial * sc;
  }

#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    acc += __shfl_down_sync(0xffffffff, acc, off);
  }

  if (lane == 0) {
    y[n] = __float2bfloat16(acc);
  }
}

}  // namespace

torch::Tensor gemv_kernel_cuda(
    torch::Tensor x,
    torch::Tensor w_packed,
    torch::Tensor scales,
    int64_t group_size) {
  auto xc = x.contiguous();
  auto wc = w_packed.contiguous();
  auto sc = scales.contiguous();

  const int64_t M = xc.size(0);
  const int64_t K = xc.size(1);
  const int64_t N = wc.size(0);
  TORCH_CHECK(K % K_ALIGN == 0,
              "K must be divisible by ", K_ALIGN);

  auto y = torch::empty(
      {M, N},
      torch::TensorOptions().device(x.device()).dtype(torch::kBFloat16));

  at::cuda::CUDAGuard guard(x.device());

  dim3 grid(static_cast<unsigned int>(N / BN));
  dim3 block(T_BLOCK);
  const int smem_bytes = static_cast<int>(K * sizeof(__nv_bfloat16));

  gemv_kernel<<<grid, block, smem_bytes>>>(
      reinterpret_cast<const __nv_bfloat16*>(xc.data_ptr<at::BFloat16>()),
      wc.data_ptr<uint8_t>(),
      reinterpret_cast<const __half*>(sc.data_ptr<at::Half>()),
      reinterpret_cast<__nv_bfloat16*>(y.data_ptr<at::BFloat16>()),
      static_cast<int>(N),
      static_cast<int>(K),
      static_cast<int>(group_size));

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}
