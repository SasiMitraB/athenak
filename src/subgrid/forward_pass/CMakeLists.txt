cmake_minimum_required(VERSION 3.16)
project(forward_pass_kokkos CXX)

# Kokkos from the AthenaK checkout.  Pick the backend on the cmake command line, e.g.
#   NVIDIA:  -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_ADA89=ON   (use your GPU's arch)
#   AMD:     -DKokkos_ENABLE_HIP=ON  -DKokkos_ARCH_AMD_GFX90A=ON
# With no backend given, Kokkos_ENABLE_SERIAL (the default) is used: correct but not on a GPU.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../../kokkos ${CMAKE_BINARY_DIR}/kokkos)

# gate/cooling/ksg_forward: checks against the JAX references in data/ (export_weights.py);
# predict: batch prediction for benchmarking/cpp_backend.py
foreach(target gate_forward cooling_forward ksg_forward predict)
  add_executable(${target} ${target}.cpp)
  target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_compile_definitions(${target} PRIVATE FP_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data/")
  target_link_libraries(${target} Kokkos::kokkos)
endforeach()
