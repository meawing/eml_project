from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name="qkernels",
    version="0.1.0",
    ext_modules=[
        CUDAExtension(
            name="quant_kernels_cuda",
            sources=[
                "kernels/bindings.cpp",
                "kernels/quant_kernel.cu",
                "kernels/gemv_kernel.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    packages=["qkernels", "qkernels.matmul", "qkernels.quantization"],
)
