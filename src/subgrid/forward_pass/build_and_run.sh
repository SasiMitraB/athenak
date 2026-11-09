#!/bin/bash
# Export the weights of the conserved-variable models (needs the project venv), configure, build
# and run the three forward-pass checks (gate, cooling, ksg).
#   ./build_and_run.sh              # Kokkos Serial backend
#   GPU=1 ./build_and_run.sh        # CUDA, as builds/build_gpu.sh (ARCH=ADA89 by default)
#   ./build_and_run.sh <cmake flags>  # e.g. -DKokkos_ENABLE_OPENMP=ON
# SKIP_EXPORT=1 reuses data/.  BUILD_DIR overrides the build directory.
set -e
ROOT=/home/sasi/Projects/SubgridCGMModel
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
if [ -z "$SKIP_EXPORT" ]; then
  (cd $ROOT && venv/bin/python "$HERE/export_weights.py")
fi
BUILD=${BUILD_DIR:-build}
FLAGS=("$@")
if [ -n "$GPU" ]; then
  ARCH=${ARCH:-ADA89}
  case $ARCH in ADA89) SM=sm_89;; HOPPER90) SM=sm_90;; AMPERE80) SM=sm_80;; AMPERE86) SM=sm_86;;
    TURING75) SM=sm_75;; VOLTA70) SM=sm_70;; *) echo "unsupported ARCH $ARCH"; exit 1;; esac
  for d in /usr/local/cuda-12.8 /usr/local/cuda-12.6 /usr/local/cuda-12.4 /usr/local/cuda-12; do
    if [ -d $d/bin ]; then export CUDA_ROOT=$d CUDA_HOME=$d PATH=$d/bin:$PATH; break; fi
  done
  export NVCC_WRAPPER_DEFAULT_ARCH=$SM
  chmod +x $ROOT/athenak/kokkos/bin/nvcc_wrapper
  BUILD=${BUILD_DIR:-build_gpu}
  FLAGS+=(-DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_$ARCH=ON
          -DCMAKE_CXX_COMPILER=$ROOT/athenak/kokkos/bin/nvcc_wrapper)
fi
cmake -S . -B $BUILD -DCMAKE_BUILD_TYPE=Release "${FLAGS[@]}"
cmake --build $BUILD -j
rc=0
for m in gate cooling ksg; do
  echo "=== $m ==="
  ./$BUILD/${m}_forward || rc=1
done
exit $rc
