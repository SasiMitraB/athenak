// Model-specific GPU pre- and post-processing around the stencil networks of stencil_net.hpp:
//   gate:  the 24 network inputs from the 6 cached fields (gate_model.features)
//   rvv:   f = sigmoid(z_0) and the shape tensor S = e^A / Tr e^A from z_0..z_5
//          (rvv_tensor_model.predict)
#pragma once

#include "stencil_net.hpp"

namespace sn {

constexpr int N_BASE = 6, N_FEAT = 24;

// base (6, nz, ny, nx) -> feat (24, nz, ny, nx)
inline void features(const sn::Vec &base, sn::Vec &feat, int nz, int ny, int nx) {
    using namespace sn;
    const size_t ncell = (size_t)nz * ny * nx;
    Kokkos::parallel_for(
        "features", Kokkos::RangePolicy<>(0, ncell), KOKKOS_LAMBDA(const size_t cell) {
            const int ix = cell % nx, iy = (cell / nx) % ny, iz = cell / ((size_t)nx * ny);
            // central difference of channel c along axis a (0 = x, 1 = y, 2 = z)
            auto d = [&](int c, int a) -> Real {
                int pz = iz, py = iy, px = ix, mz = iz, my = iy, mx = ix;
                if (a == 0) { px = (ix + 1) % nx; mx = (ix + nx - 1) % nx; }
                if (a == 1) { py = Kokkos::min(iy + 1, ny - 1); my = Kokkos::max(iy - 1, 0); }
                if (a == 2) { pz = (iz + 1) % nz; mz = (iz + nz - 1) % nz; }
                const size_t o = (size_t)c * ncell;
                return 0.5f * (base(o + ((size_t)pz * ny + py) * nx + px) -
                               base(o + ((size_t)mz * ny + my) * nx + mx));
            };
            for (int c = 0; c < N_BASE; ++c) feat((size_t)c * ncell + cell) = base((size_t)c * ncell + cell);
            const int grad_of[3] = {0, 1, 5};  // log_rho, log_T, s0
            for (int g = 0; g < 3; ++g)
                for (int a = 0; a < 3; ++a)
                    feat((size_t)(N_BASE + 3 * g + a) * ncell + cell) = d(grad_of[g], a);
            Real du[3][3];  // du[i][j] = d_j u_i
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j) du[i][j] = d(2 + i, j);
            const int pi[6] = {0, 1, 2, 0, 0, 1}, pj[6] = {0, 1, 2, 1, 2, 2};
            for (int p = 0; p < 6; ++p)
                feat((size_t)(15 + p) * ncell + cell) = 0.5f * (du[pi[p]][pj[p]] + du[pj[p]][pi[p]]);
            feat((size_t)21 * ncell + cell) = du[2][1] - du[1][2];
            feat((size_t)22 * ncell + cell) = du[0][2] - du[2][0];
            feat((size_t)23 * ncell + cell) = du[1][0] - du[0][1];
        });
}

// f (ncell) and S (6, ncell) from the raw outputs z (6, ncell).  In float: double is 1/64 rate
// on consumer GPUs, and a symmetric 3x3 Jacobi solve converges to float precision anyway.
inline void shape_from_z(const sn::Vec &z, sn::Vec &f, sn::Vec &S, size_t ncell) {
    using namespace sn;
    Kokkos::parallel_for(
        "shape", Kokkos::RangePolicy<>(0, ncell), KOKKOS_LAMBDA(const size_t i) {
            const Real r2 = 0.70710678f, r6 = 0.40824829f;
            const Real z1 = z(1 * ncell + i), z2 = z(2 * ncell + i), z3 = z(3 * ncell + i),
                       z4 = z(4 * ncell + i), z5 = z(5 * ncell + i);
            f(i) = sigmoid(z(i));
            // A = sum z_k E^(k)
            Real A[3][3] = {{r2 * z1 + r6 * z2, r2 * z3, r2 * z4},
                            {r2 * z3, -r2 * z1 + r6 * z2, r2 * z5},
                            {r2 * z4, r2 * z5, -2 * r6 * z2}};
            // symmetric eigen-decomposition by cyclic Jacobi: A = V diag(l) V^T
            Real V[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
            const int P[3] = {0, 0, 1}, Q[3] = {1, 2, 2};
            for (int sweep = 0; sweep < 6; ++sweep) {
                for (int r = 0; r < 3; ++r) {
                    const int p = P[r], q = Q[r];
                    if (Kokkos::fabs(A[p][q]) < 1e-30f) continue;
                    const Real th = (A[q][q] - A[p][p]) / (2 * A[p][q]);
                    const Real t = (th >= 0 ? Real(1) : Real(-1)) /
                                   (Kokkos::fabs(th) + Kokkos::sqrt(th * th + 1));
                    const Real c = 1 / Kokkos::sqrt(t * t + 1), s = t * c;
                    for (int k = 0; k < 3; ++k) {  // A <- A J
                        const Real akp = A[k][p], akq = A[k][q];
                        A[k][p] = c * akp - s * akq;
                        A[k][q] = s * akp + c * akq;
                    }
                    for (int k = 0; k < 3; ++k) {  // A <- J^T A
                        const Real apk = A[p][k], aqk = A[q][k];
                        A[p][k] = c * apk - s * aqk;
                        A[q][k] = s * apk + c * aqk;
                    }
                    for (int k = 0; k < 3; ++k) {  // V <- V J
                        const Real vkp = V[k][p], vkq = V[k][q];
                        V[k][p] = c * vkp - s * vkq;
                        V[k][q] = s * vkp + c * vkq;
                    }
                }
            }
            const Real l[3] = {A[0][0], A[1][1], A[2][2]};
            const Real lmax = Kokkos::fmax(l[0], Kokkos::fmax(l[1], l[2]));
            Real e[3], tr = 0;
            for (int k = 0; k < 3; ++k) { e[k] = Kokkos::exp(l[k] - lmax); tr += e[k]; }
            const int pi[6] = {0, 1, 2, 0, 0, 1}, pj[6] = {0, 1, 2, 1, 2, 2};
            for (int c = 0; c < 6; ++c) {
                Real s = 0;
                for (int k = 0; k < 3; ++k) s += V[pi[c]][k] * e[k] * V[pj[c]][k];
                S(c * ncell + i) = s / tr;
            }
        });
}

}  // namespace sn
