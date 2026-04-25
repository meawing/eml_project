import quant_kernels_cuda


def quantize_fp16_to_int4(w, group_size=128):
    return quant_kernels_cuda.quantize_fp16_to_int4(w, group_size)
