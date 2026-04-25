import quant_kernels_cuda


def gemv_kernel(x, w_packed, scales, group_size=128):
    return quant_kernels_cuda.gemv_kernel(x, w_packed, scales, group_size)
