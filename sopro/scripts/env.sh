#!/bin/sh
# Invoke from the chosen working directory. All mutable caches stay there.
workspace=${SOPRO_WORKDIR:-$PWD}
export SOPRO_WORKDIR="$workspace"
export HF_HUB_DISABLE_XET=1
export HF_HOME="$workspace/cache/huggingface"
export UV_CACHE_DIR="$workspace/cache/uv"
export UV_PYTHON_INSTALL_DIR="$workspace/cache/uv-python"
export TORCH_HOME="$workspace/cache/torch"
export XDG_CACHE_HOME="$workspace/cache/xdg"
export TMPDIR="$workspace/cache/tmp"
export PYTHONPYCACHEPREFIX="$workspace/cache/pycache"
export MPLCONFIGDIR="$workspace/cache/matplotlib"
export NUMBA_CACHE_DIR="$workspace/cache/numba"
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export UV_CONCURRENT_DOWNLOADS=8
export UV_HTTP_TIMEOUT=60
export UV_HTTP_RETRIES=2
export PYTHONUNBUFFERED=1
mkdir -p "$TMPDIR" "$workspace/results" "$workspace/logs" "$workspace/fixtures" "$workspace/exports"
exec "$@"
