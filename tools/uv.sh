#!/usr/bin/env bash
# WSL/Linux entry point; native libraries are not Python dependency metadata.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$ROOT/.native.env" ]]; then
    source "$ROOT/.native.env"
fi
UV="${UV_BIN:-uv}"
if ! command -v "$UV" >/dev/null 2>&1; then
    UV="$ROOT/.tools/uv/bin/uv"
fi
if [[ ! -x "$UV" ]] && ! command -v "$UV" >/dev/null 2>&1; then
    echo "Install uv first: https://docs.astral.sh/uv/getting-started/installation/" >&2
    exit 1
fi
export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT/.tools/uv-cache}"
# Keep WSL development separate from other Python environments.
export UV_PROJECT_ENVIRONMENT="$ROOT/.venv-wsl"
if [[ -n "${SUITESPARSE_DIR:-}" ]]; then
    export SUITESPARSE_INCLUDE_DIR="$SUITESPARSE_DIR/include/suitesparse"
    export SUITESPARSE_LIB_DIR="$SUITESPARSE_DIR/lib"
    export SUITESPARSE_LIBRARY_DIR="$SUITESPARSE_LIB_DIR"
    export PKG_CONFIG_PATH="$SUITESPARSE_LIB_DIR/pkgconfig:${PKG_CONFIG_PATH:-}"
    export LD_LIBRARY_PATH="$SUITESPARSE_LIB_DIR:${LD_LIBRARY_PATH:-}"
fi
if [[ -n "${MKL_LIBRARY_DIR:-}" ]]; then
    export LD_LIBRARY_PATH="$MKL_LIBRARY_DIR:${LD_LIBRARY_PATH:-}"
fi
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export LD_LIBRARY_PATH="$UV_PROJECT_ENVIRONMENT/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
if [[ -d /usr/lib/wsl/lib ]]; then
    export LD_LIBRARY_PATH="/usr/lib/wsl/lib:$LD_LIBRARY_PATH"
fi
if [[ -f "$CUDA_HOME/include/cublas_v2.h" ]]; then
    export CPATH="$CUDA_HOME/include:${CPATH:-}"
fi
export CHOLMOD_USE_GPU="${CHOLMOD_USE_GPU:-0}"
export MKL_INTERFACE_LAYER="${MKL_INTERFACE_LAYER:-LP64}"
if [[ "${1:-}" == sync || "${1:-}" == run ]]; then
    HEADER="${SUITESPARSE_INCLUDE_DIR:-/usr/include/suitesparse}/SuiteSparse_config.h"
    if [[ ! -f "$HEADER" ]]; then
        echo "SuiteSparse >=7.4 is required. Set SUITESPARSE_DIR to its installed prefix." >&2
        exit 1
    fi
    MAJOR="$(awk '$2 == "SUITESPARSE_MAIN_VERSION" {print $3}' "$HEADER")"
    MINOR="$(awk '$2 == "SUITESPARSE_SUB_VERSION" {print $3}' "$HEADER")"
    if [[ -z "$MAJOR" || -z "$MINOR" ]] || (( MAJOR < 7 || (MAJOR == 7 && MINOR < 4) )); then
        echo "Found SuiteSparse ${MAJOR:-?}.${MINOR:-?}; need >=7.4." >&2
        exit 1
    fi
fi
exec "$UV" --directory "$ROOT" "$@"
