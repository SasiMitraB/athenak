// Model-specific GPU pre- and post-processing around the stencil networks of stencil_net.hpp,
// for the conserved-variable models (models/conserved/).  Every model reads only the coarse
// conserved state u0 = (rho, m_x, m_y, m_z, E) (IDN, IM1..IM3, IEN), the passive scalar left out:
//   features:  the 20 network inputs of conserved.features from u0
//   ksg_from_z: f = sigmoid(z_0) and k_sg = f k_max, k_max = (E - m^2 / 2 rho) / rho
//              (ksg_model, PredictKsg)
//   cooling_inputs: the cooling model's inputs, the features, the gate and, for a model
//              trained with --ksg-model, log10 f and log10 (1 - f) of the k_sg network
//   shape_from_z: the shape tensor S = e^A / Tr e^A of R from z_1..z_5
//              (rvv_tensor_model.shape_tensor), for the R shape model to come
#pragma once

#include "stencil_net.hpp"

namespace sn {

constexpr int N_U = 5;      // u0 channels: rho, m_x, m_y, m_z, E
constexpr int N_FEAT = 20;  // conserved.channels
constexpr int N_KSG_IN = 2;  // cooling_model.KSG_CHANNELS: log10 f, log10 (1 - f)
constexpr int N_COOL_IN = N_FEAT + 1;                 // features + gate
constexpr int N_COOL_IN_KSG = N_COOL_IN + N_KSG_IN;  // ... + the k_sg channels

// The 20 network inputs of one cell, in the order of conserved.channels:
//   0 log10 rho, 1-3 m, 4 log10 E,
//   5-7 grad log10 rho, 8-10 grad log10 E (d/dx d/dy d/dz),
//   11-16 Sm_ij = (d_j m_i + d_i m_j) / 2 (xx yy zz xy xz yz), 17-19 curl m,
// gradients as central differences in units of a cell.
// u(c, a, s): channel c of u0 (rho, m_x, m_y, m_z, E) at the cell (s = 0) or at its neighbour
//             s = -1, +1 along axis a (0 = x, 1 = y, 2 = z); the caller applies the boundaries.
// put(c, v):  stores input c.
// Used by features() below and by AthenaK (subgrid_predict_ksg.cpp), so both see one formula.
template <class U, class Put>
KOKKOS_INLINE_FUNCTION void cell_features(const U &u, const Put &put) {
    // the state channel c of conserved.features: logs of rho and E
    auto q = [&](int c, int a, int s) -> Real {
        const Real v = u(c, a, s);
        return (c == 0 || c == 4) ? Kokkos::log10(v) : v;
    };
    auto d = [&](int c, int a) -> Real { return 0.5f * (q(c, a, 1) - q(c, a, -1)); };
    for (int c = 0; c < N_U; ++c) put(c, q(c, 0, 0));
    for (int a = 0; a < 3; ++a) {
        put(5 + a, d(0, a));
        put(8 + a, d(4, a));
    }
    Real dm[3][3];  // dm[i][j] = d_j m_i
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) dm[i][j] = d(1 + i, j);
    const int pi[6] = {0, 1, 2, 0, 0, 1}, pj[6] = {0, 1, 2, 1, 2, 2};
    for (int p = 0; p < 6; ++p) put(11 + p, 0.5f * (dm[pi[p]][pj[p]] + dm[pj[p]][pi[p]]));
    put(17, dm[2][1] - dm[1][2]);
    put(18, dm[0][2] - dm[2][0]);
    put(19, dm[1][0] - dm[0][1]);
}

// u0 (5, nz, ny, nx) -> feat channels 0..19 of (n_chan, nz, ny, nx), n_chan >= 20 (cell_features)
// with the boundaries of stencil.pad: periodic in z and x, edge-copied in y (so a one-sided
// half difference at the y ends).  Channels >= 20 (the cooling model's gate) are left to the
// caller.
inline void features(const Vec &u, Vec &feat, int nz, int ny, int nx) {
    const size_t ncell = (size_t)nz * ny * nx;
    Kokkos::parallel_for(
        "features", Kokkos::RangePolicy<>(0, ncell), KOKKOS_LAMBDA(const size_t cell) {
            const int ix = cell % nx, iy = (cell / nx) % ny, iz = cell / ((size_t)nx * ny);
            auto u_at = [&](int c, int a, int s) -> Real {
                int x = ix, y = iy, z = iz;
                if (a == 0) x = (ix + s + nx) % nx;
                if (a == 1) y = Kokkos::clamp(iy + s, 0, ny - 1);
                if (a == 2) z = (iz + s + nz) % nz;
                return u((size_t)c * ncell + ((size_t)z * ny + y) * nx + x);
            };
            cell_features(u_at, [&](int c, Real v) { feat((size_t)c * ncell + cell) = v; });
        });
}

// k_max = (E - m^2 / 2 rho) / rho, the unresolved energy per unit mass (conserved.k_max)
KOKKOS_INLINE_FUNCTION Real k_max(const Vec &u, size_t ncell, size_t i) {
    const Real rho = u(i), mx = u(ncell + i), my = u(2 * ncell + i), mz = u(3 * ncell + i),
               E = u(4 * ncell + i);
    return (E - 0.5f * (mx * mx + my * my + mz * mz) / rho) / rho;
}

// f = sigmoid(z_0) = k_sg / k_max and k_sg (ncell each) from the raw output z (>= ncell) and u0.
inline void ksg_from_z(const Vec &z, const Vec &u, Vec &f, Vec &ksg, size_t ncell) {
    Kokkos::parallel_for(
        "ksg", Kokkos::RangePolicy<>(0, ncell), KOKKOS_LAMBDA(const size_t i) {
            f(i) = sigmoid(z(i));
            ksg(i) = f(i) * k_max(u, ncell, i);
        });
}

// softplus(x) = ln(1 + e^x), without overflow for any x
KOKKOS_INLINE_FUNCTION Real softplus(Real x) {
    return Kokkos::fmax(x, Real(0)) + Kokkos::log1p(Kokkos::exp(-Kokkos::fabs(x)));
}

// The k_sg inputs of the cooling model (cooling_model.ksg_input) from the k_sg network's raw
// output z0 (ncell): log10 f into feat channel dst and log10 (1 - f) into dst + 1, f =
// sigmoid(z0), as -softplus(-z0) / ln 10 and -softplus(z0) / ln 10 (finite for any z0).
inline void ksg_channels(const Vec &z0, Vec &feat, int dst, size_t ncell) {
    Kokkos::parallel_for(
        "ksg_channels", Kokkos::RangePolicy<>(0, ncell), KOKKOS_LAMBDA(const size_t i) {
            const Real inv_ln10 = 0.43429448f, z = z0(i);
            feat((size_t)dst * ncell + i) = -softplus(-z) * inv_ln10;
            feat((size_t)(dst + 1) * ncell + i) = -softplus(z) * inv_ln10;
        });
}

// S (6, ncell) from the raw outputs z_1..z_5 at z(1 * ncell ..), for the R shape model.  In
// float: double is 1/64 rate on consumer GPUs, and a symmetric 3x3 Jacobi solve converges to
// float precision anyway.
inline void shape_from_z(const Vec &z, Vec &S, size_t ncell) {
    Kokkos::parallel_for(
        "shape", Kokkos::RangePolicy<>(0, ncell), KOKKOS_LAMBDA(const size_t i) {
            const Real r2 = 0.70710678f, r6 = 0.40824829f;
            const Real z1 = z(1 * ncell + i), z2 = z(2 * ncell + i), z3 = z(3 * ncell + i),
                       z4 = z(4 * ncell + i), z5 = z(5 * ncell + i);
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

// Copy channel src of x into channel dst of feat (the cooling model's gate: x = u0 + gate,
// channel 5, into feat channel 20).
inline void copy_channel(const Vec &x, int src, Vec &feat, int dst, size_t ncell) {
    Kokkos::parallel_for(
        "copy_channel", Kokkos::RangePolicy<>(0, ncell), KOKKOS_LAMBDA(const size_t i) {
            feat((size_t)dst * ncell + i) = x((size_t)src * ncell + i);
        });
}

// The cooling model's network inputs feat (N_COOL_IN or N_COOL_IN_KSG, ncell) from x = u0 plus
// the gate (6, ncell): the 20 features, the gate in channel 20 and, if ksg is given (a model
// trained with --ksg-model), the k_sg network ksg run on the same 20 features (its own
// workspace ksg_work, raw output z0 (ncell)) and its log10 f, log10 (1 - f) in channels 21, 22.
// The k_sg network reads only the first N_FEAT channels of feat (stencil_net pad), so it shares
// the buffer.  In AthenaK the gate is gate_forward's prediction (p > 0.5).
inline void cooling_inputs(const Vec &x, Vec &feat, int nz, int ny, int nx,
                           const Model *ksg = nullptr, Work *ksg_work = nullptr,
                           Vec *z0 = nullptr) {
    const size_t ncell = (size_t)nz * ny * nx;
    features(x, feat, nz, ny, nx);
    copy_channel(x, N_U, feat, N_FEAT, ncell);
    if (ksg) {
        forward(*ksg, feat, *ksg_work, *z0);
        ksg_channels(*z0, feat, N_COOL_IN, ncell);
    }
}

// Check that m is a cooling model and return whether it takes the k_sg channels.
inline bool cooling_takes_ksg(const Model &m) {
    if (m.n_in != N_COOL_IN && m.n_in != N_COOL_IN_KSG)
        throw std::runtime_error("cooling model has " + std::to_string(m.n_in) + " inputs, not " +
                                 std::to_string(N_COOL_IN) + " or " + std::to_string(N_COOL_IN_KSG));
    if (m.n_out != 1) throw std::runtime_error("cooling model must have n_out = 1");
    return m.n_in == N_COOL_IN_KSG;
}

// Check that m is a k_sg model (20 inputs, 1 output).
inline void check_ksg(const Model &m) {
    if (m.n_in != N_FEAT) throw std::runtime_error("k_sg model must have 20 input channels");
    if (m.n_out != 1) throw std::runtime_error("k_sg model must have n_out = 1");
}

}  // namespace sn
