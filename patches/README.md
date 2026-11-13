# Apple GPU (Metal) patches

These patches are taken unchanged from the scaluq team's F2 investigation
(forest1040/scaluq, branch `research/f2-metal`, `ref/f2_metal/patches/`;
qulacs/scaluq#463), discussed upstream in kokkos/kokkos#9750.

| File | Applies to | Purpose |
|---|---|---|
| `kokkos-5.2.0-acpp-metal.patch` | Kokkos 5.2.0 (`d9397f95`, the `kokkos/` submodule) | Lets the Kokkos SYCL backend build with AdaptiveCpp generic SSCP and run correctly on Metal |
| `adaptivecpp-metal-num-subgroups.patch` | AdaptiveCpp `develop` @ `6c923f68` | Adds the missing Metal builtin `__acpp_sscp_get_num_subgroups` |

`CMakeLists.txt` applies the Kokkos patch to `kokkos/` automatically when
`-DAthena_ENABLE_ACPP_METAL=ON` is set.

## Kokkos 5 compatibility (all builds)

| File | Purpose |
|---|---|
| `kokkos-5.2.0-dualview-public.patch` | One-line change by us: makes `DualView::d_view`/`h_view` public again. Kokkos deprecated direct access in 4.6 (kokkos#7716) and made the members private in 5.2.0 (kokkos#8980). AthenaK only reads these members, so this stopgap is safe until AthenaK moves to `view_device()`/`view_host()`. |

`CMakeLists.txt` always applies this patch. It also sets
`Kokkos_ENABLE_DEPRECATED_CODE_4=ON` to keep `View::HostMirror`, whose
removal Kokkos deferred to 5.3.

## Build (recipe from the scaluq report)

```sh
brew install llvm@21 lld@21
# Use the macOS 26 SDK. With SDK 27, math.h includes float.h partially, and
# libc++ 21 then never defines FLT_MAX etc.
export SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX26.sdk
L=/opt/homebrew/opt/llvm@21

# AdaptiveCpp develop @ 6c923f68 with adaptivecpp-metal-num-subgroups.patch applied;
# metal-cpp from https://developer.apple.com/metal/cpp/files/metal-cpp_26.zip
cmake -S AdaptiveCpp -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=$L/bin/clang -DCMAKE_CXX_COMPILER=$L/bin/clang++ \
  -DLLVM_DIR=$L/lib/cmake/llvm -DCLANG_EXECUTABLE_PATH=$L/bin/clang++ \
  -DACPP_LLD_PATH=/opt/homebrew/opt/lld@21/bin/ld64.lld \
  -DACPP_COMPILER_FEATURE_PROFILE=full -DWITH_METAL_BACKEND=ON \
  -DMETAL_INCLUDE_DIR=/path/to/metal-cpp -DWITH_OPENCL_BACKEND=OFF \
  -DCMAKE_INSTALL_PREFIX=$HOME/opt/adaptivecpp
cmake --build build -j && cmake --install build

# AthenaK
# Release only: in Debug, Kokkos_SYCL_Abort.hpp calls glibc's __assert_fail
cmake -S athenak -B build-metal -DCMAKE_OSX_SYSROOT=$SDKROOT -DCMAKE_C_COMPILER=$L/bin/clang \
  -DCMAKE_CXX_COMPILER=$HOME/opt/adaptivecpp/bin/acpp \
  -DAthena_ENABLE_ACPP_METAL=ON -DPROBLEM=...
cmake --build build-metal -j
```

## Running

- `KOKKOS_DEVICE_ID=1` (or `--kokkos-device-id=1`) is required. Device 0 is
  AdaptiveCpp's OpenMP host device.
- Apple GPUs have no fp64, so the build is forced to single precision.
- Optional: `KOKKOS_IMPL_ACPP_COARSE_EVENTS=1` turns on coarse-grained events
  (slower for small kernels in scaluq's measurements).

## Known limitations (from the scaluq report and kokkos#9750)

- Reductions and scans use a two-pass scheme under AdaptiveCpp, because the
  last-work-group pattern is wrong on Metal (AdaptiveCpp#2311).
- `Kokkos::printf` is a no-op on device.
- `Random_XorShift64_Pool::get_state()` can hang with very large launches
  (around 2^24 work-items) on Metal.
- A run-time choice of `team_scratch(level)` can fail with "Conflicting address
  space information in PHI node" (AdaptiveCpp#2313).
- Kernel launch latency is about 20–65 µs per small kernel.
