# How to setup

Install CUDA version 12.9 under /usr/local/cuda
Make sure CUDA_HOME is set to /usr/local/cuda and /usr/local/cuda/bin is in your PATH.
Make sure the rocm-llvm package is installed (for amdgcn codegen)
Make sure uv is installed.
Create the .venv virtual environment via uv sync.
When tuning for AMD, you will also need the .venv-rocm virtual environment, which is currently not defined by any pyproject toml
but instead is just the same dependencies in there with amd package variants.
When tuning, you will also need the cutlass_codegen .venv in kernels/cutlass_codegen, which is defined by the pyproject.toml in that directory.
