## Prerequisites

- NVIDIA GPU + driver (tested on RTX PRO 6000 Blackwell `sm_120`).
- Linux with `apt`, Python 3.12.
- CUDA Toolkit 12.8 with `nvcc` -- needed once to build the CUDA extension.
  Minimal install via apt: `cuda-nvcc-12-8 cuda-cudart-dev-12-8 cuda-crt-12-8`
  (`/usr/local/cuda-12.8/bin/nvcc`). Torch 2.11+cu128 ships its own runtime,
  so `nvcc` isn't needed at run time -- only for the build below.

## Setup

```bash
apt-get update && sudo apt-get install -y \
    python3.12 python3.12-dev build-essential ninja-build curl ca-certificates

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