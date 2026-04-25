#include <torch/extension.h>

#include <vector>

std::vector<torch::Tensor> quantize_fp16_to_int4_cuda(
    torch::Tensor w,
    int64_t group_size);

torch::Tensor gemv_kernel_cuda(
    torch::Tensor x,
    torch::Tensor w_packed,
    torch::Tensor scales,
    int64_t group_size);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("quantize_fp16_to_int4", &quantize_fp16_to_int4_cuda);
  m.def("gemv_kernel", &gemv_kernel_cuda);
}
