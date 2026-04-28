## Prerequisites

- NVIDIA GPU + driver 580+ (tested on RTX PRO 6000 Blackwell `sm_120`).
- Ubuntu 22.04/24.04, Python 3.12.
- CUDA 12.8 `nvcc` — build-time only; PyTorch ships its own runtime.

## Setup

```bash
apt-get update && apt-get install -y \
    python3.12 python3.12-dev build-essential ninja-build curl ca-certificates \
    cuda-nvcc-12-8 cuda-cudart-dev-12-8 cuda-crt-12-8

curl -fsSL https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"

uv sync
source .venv/bin/activate

# Build the CUDA extension. Adjust TORCH_CUDA_ARCH_LIST for your GPU.
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=/usr/local/cuda-12.8/bin:$PATH
export TORCH_CUDA_ARCH_LIST="12.0"
python setup.py build_ext --inplace

# Make `qkernels` importable from the repo without installing it.
python -c "import site, pathlib; p=pathlib.Path(site.getsitepackages()[0])/'qkernels.pth'; p.write_text(str(pathlib.Path.cwd()))"

python -m ipykernel install --user --name eml-project --display-name "Python 3.12 (eml-project)"
```