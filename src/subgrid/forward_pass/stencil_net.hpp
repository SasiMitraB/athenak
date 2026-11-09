// Shared Kokkos core of the stencil models (models/stencil.py): the forward pass of
//
//   raw inputs (n_in, nz, ny, nx) -> standardise -> pad (wrap z,x; edge y)
//   -> VALID 3D conv (stencil^3 * n_in -> h0) + gelu -> dense + gelu ... -> dense -> n_out raw outputs
//
// Runs on whatever Kokkos backend it was built with (CUDA/HIP/SYCL/OpenMP/Serial).
//
// forward() = pad() + layers().  pad() is the training-time padding of one whole grid (the
// standalone tests); AthenaK fills the padded input Work::xp itself from each MeshBlock and
// its ghost zones, for a whole MeshBlockPack at once, and calls layers().
//
// GPU path (layers_gemm): every layer is a matrix product over all cells,
//   C (N x M) = act(W^T (N x K) A (K x M) + b),   M = cells, K = fan-in, N = fan-out,
// with the activations feature-major in global memory (row n holds the M cells; a layer's
// output stays in L2 for the next one).  A wide layer is one kernel of 128 x 128 (or 64 x 64)
// output tiles per team of 256 threads, each thread an 8 x 8 (4 x 4) register block; the K
// loop stages BK rows of A and W in double-buffered shared memory, prefetched through
// registers so the global loads overlap the arithmetic.  Bias and gelu are applied in the
// epilogue.  The stencil conv is the same kernel with A gathered on the fly (implicit im2col)
// from the standardised, padded inputs.  A narrow layer (n_out <= 8, the output heads) is one
// thread per cell.
//
// Host path (layers_simple): one team per cell, the team's threads split the hidden units.
//
// A model file (written by export_weights.py) holds one input standardisation and one or more
// networks that see the same padded input (rvv_tensor has separate networks for f and S).
//
// weights.bin (little-endian):
//   int32  magic 0x43474D32, stencil, n_in, n_nets
//   float32 x_mean[n_in], x_std[n_in]
//   per net: int32 n_layers, int32 n_out[n_layers]
//            per layer: float32 w, b   (layer 0: w is (kz, ky, kx, n_in, n_out) row-major, DHWIO;
//                                       others: (in, out) row-major)
// reference.bin: model specific, see export_weights.py.
#pragma once

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace sn {

using Real = float;
using Vec = Kokkos::View<Real *>;
using ConstVec = Kokkos::View<const Real *>;
constexpr int MAX_LAYERS = 8;
constexpr int MAX_NETS = 2;
constexpr int MAX_OUT = 8;     // total outputs of a model
constexpr int NARROW = 8;      // layers this narrow use the one-thread-per-cell kernel
constexpr int TEAM = 256;      // threads per GEMM team
constexpr int M_ALIGN = 128;   // cell padding of the activation buffers (the largest tile)

template <class T> inline void read_vec(std::ifstream &f, std::vector<T> &v, size_t n) {
    v.resize(n);
    f.read(reinterpret_cast<char *>(v.data()), n * sizeof(T));
    if (!f) throw std::runtime_error("unexpected end of file");
}

// jax.nn.gelu default (approximate = True)
KOKKOS_INLINE_FUNCTION Real gelu(Real x) {
    const Real c = 0.7978845608028654f;  // sqrt(2/pi)
    return 0.5f * x * (1.0f + tanhf(c * (x + 0.044715f * x * x * x)));
}

KOKKOS_INLINE_FUNCTION Real sigmoid(Real x) { return 1.0f / (1.0f + expf(-x)); }

// One network.  Layer 0: w is (k,k,k,C,n_out) row-major; others (n_in, n_out).  w and b are
// per-layer device copies (host path); w_off and b_off locate the same arrays in Model::flat
// (GPU path), 16-byte aligned.
struct DeviceNet {
    int n_layers = 0, n_out = 0, max_width = 0;
    int fan_in[MAX_LAYERS] = {}, fan_out[MAX_LAYERS] = {};
    size_t w_off[MAX_LAYERS] = {}, b_off[MAX_LAYERS] = {};
    ConstVec w[MAX_LAYERS], b[MAX_LAYERS];
};

// All networks of a model plus the input standardisation.
struct Model {
    int stencil = 0, n_in = 0, n_nets = 0, n_out = 0, max_width = 0;  // n_out: sum over the nets
    DeviceNet net[MAX_NETS];
    ConstVec x_mean, x_std;
    ConstVec flat;                   // all weights and biases, packed
    Kokkos::View<const int *> taps;  // per conv fan-in row: c | dx << 10 | dy << 14 | dz << 18
};

inline ConstVec to_device(const std::string &label, const std::vector<Real> &h) {
    Vec d(label, h.size());
    Kokkos::deep_copy(d, Vec::host_mirror_type(const_cast<Real *>(h.data()), h.size()));
    return d;
}

inline Model load_model(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::vector<int32_t> head, sizes;
    read_vec(f, head, 4);
    if (head[0] != 0x43474D32) throw std::runtime_error("bad magic in " + path);
    Model m;
    m.stencil = head[1];
    m.n_in = head[2];
    m.n_nets = head[3];
    if (m.n_nets > MAX_NETS) throw std::runtime_error("too many networks");
    std::vector<Real> mean, std_, w, b, flat;
    read_vec(f, mean, m.n_in);
    read_vec(f, std_, m.n_in);
    m.x_mean = to_device("x_mean", mean);
    m.x_std = to_device("x_std", std_);
    for (int n = 0; n < m.n_nets; ++n) {
        DeviceNet &net = m.net[n];
        std::vector<int32_t> nl;
        read_vec(f, nl, 1);
        net.n_layers = nl[0];
        if (net.n_layers > MAX_LAYERS) throw std::runtime_error("too many layers");
        read_vec(f, sizes, net.n_layers);
        for (int l = 0; l < net.n_layers; ++l) {
            net.fan_in[l] = (l == 0) ? m.n_in * m.stencil * m.stencil * m.stencil : sizes[l - 1];
            net.fan_out[l] = sizes[l];
            net.max_width = std::max(net.max_width, net.fan_out[l]);
            read_vec(f, w, (size_t)net.fan_in[l] * net.fan_out[l]);
            read_vec(f, b, net.fan_out[l]);
            net.w[l] = to_device("w" + std::to_string(l), w);
            net.b[l] = to_device("b" + std::to_string(l), b);
            flat.resize((flat.size() + 3) / 4 * 4);  // 16-byte aligned rows for float4 loads
            net.w_off[l] = flat.size();
            flat.insert(flat.end(), w.begin(), w.end());
            flat.resize((flat.size() + 3) / 4 * 4);
            net.b_off[l] = flat.size();
            flat.insert(flat.end(), b.begin(), b.end());
        }
        net.n_out = net.fan_out[net.n_layers - 1];
        m.n_out += net.n_out;
        m.max_width = std::max(m.max_width, net.max_width);
    }
    if (m.n_out > MAX_OUT) throw std::runtime_error("too many outputs");
    if (m.n_in >= 1024 || m.stencil >= 16) throw std::runtime_error("n_in or stencil too large");
    const int k = m.stencil;
    std::vector<int> taps(m.n_in * k * k * k);
    for (size_t r = 0; r < taps.size(); ++r) {
        const int tap = r / m.n_in, c = r % m.n_in;
        taps[r] = c | (tap % k) << 10 | (tap / k % k) << 14 | (tap / (k * k)) << 18;
    }
    Kokkos::View<int *> d("taps", taps.size());
    Kokkos::deep_copy(d, Kokkos::View<int *, Kokkos::HostSpace>(taps.data(), taps.size()));
    m.taps = d;
    m.flat = to_device("flat", flat);
    return m;
}

inline void describe(const Model &m) {
    std::printf("Kokkos execution space: %s\n", Kokkos::DefaultExecutionSpace::name());
    std::printf("model: stencil %d, n_in %d, %d net(s)\n", m.stencil, m.n_in, m.n_nets);
    for (int n = 0; n < m.n_nets; ++n) {
        std::printf("  net %d:", n);
        for (int l = 0; l < m.net[n].n_layers; ++l)
            std::printf(" %d->%d", m.net[n].fan_in[l], m.net[n].fan_out[l]);
        std::printf("\n");
    }
}

// ----------------------------------------------------------------------------- work arrays

// True when the default execution space cannot see host memory (CUDA, HIP, SYCL without UVM).
constexpr bool on_device =
    !Kokkos::SpaceAccessibility<Kokkos::HostSpace,
                                Kokkos::DefaultExecutionSpace::memory_space>::accessible;

// Work arrays of the forward pass over nb grids ("blocks") of nz x ny x nx output cells each,
// every block with its own halo of h = stencil / 2 cells: a block is padded to pz x py x px,
// and the padded cells of all blocks, p = ((b * pz + iz) * py + iy) * px + ix, number plane.
//   xp:     the standardised, padded inputs; channel c of padded cell p at xp_index(p, c, ...),
//           i.e. (b, z, y, x, c) on the host path and (c, b, z, y, x) on the GPU path
//   corner: padded cell of the stencil corner of each output cell
//           cell = ((b * nz + iz) * ny + iy) * nx + ix  ->  p of (b, iz, iy, ix)
//   act:    (GPU) two activation buffers, max_width x mpad, feature-major
// Outputs are (n_out, nb, nz, ny, nx).  The standalone tests use nb = 1 and fill xp with pad();
// AthenaK fills xp itself from u0 and its ghost zones (subgrid_predict_ksg.cpp).
struct Work {
    int nb = 0, nz = 0, ny = 0, nx = 0, h = 0, pz = 0, py = 0, px = 0, n_in = 0, mpad = 0;
    size_t ncell = 0, plane = 0;
    Vec xp, act[2];
    Kokkos::View<int *> corner;
};

// Offset of channel c of padded cell p in Work::xp.
KOKKOS_INLINE_FUNCTION size_t xp_index(size_t p, int c, size_t plane, int n_in) {
    return on_device ? (size_t)c * plane + p : p * n_in + c;
}

inline Work make_work(const Model &m, int nz, int ny, int nx, int nb = 1) {
    Work w;
    w.nb = nb, w.nz = nz, w.ny = ny, w.nx = nx, w.n_in = m.n_in;
    w.h = m.stencil / 2;
    w.pz = nz + 2 * w.h, w.py = ny + 2 * w.h, w.px = nx + 2 * w.h;
    w.ncell = (size_t)nb * nz * ny * nx;
    w.plane = (size_t)nb * w.pz * w.py * w.px;
    if (w.plane * m.n_in >= (size_t)1 << 31)
        throw std::runtime_error("forward pass: padded input too large for int offsets");
    w.xp = Vec("xp", w.plane * m.n_in);
    if (on_device) {
        w.mpad = (w.ncell + M_ALIGN - 1) / M_ALIGN * M_ALIGN;
        for (auto &a : w.act) a = Vec("act", (size_t)m.max_width * w.mpad);
    }
    w.corner = Kokkos::View<int *>("corner", w.ncell);
    auto corner = w.corner;
    const int pz = w.pz, py = w.py, px = w.px;
    Kokkos::parallel_for(
        "corner", Kokkos::RangePolicy<>(0, w.ncell), KOKKOS_LAMBDA(const size_t cell) {
            const int ix = cell % nx, iy = (cell / nx) % ny, iz = (cell / ((size_t)nx * ny)) % nz;
            const int b = cell / ((size_t)nx * ny * nz);
            corner(cell) = (((size_t)b * pz + iz) * py + iy) * px + ix;
        });
    return w;
}

// xp from raw x (n_in, nz, ny, nx) of one grid (nb = 1), standardised and padded with the
// boundaries of the training frames (stencil.pad): periodic in z and x, edge copy in y.
inline void pad(const Model &m, const Vec &x, Work &work) {
    if (work.nb != 1) throw std::runtime_error("pad(): one grid only");
    const int nz = work.nz, ny = work.ny, nx = work.nx, h = work.h, py = work.py, px = work.px;
    const int n_in = m.n_in;
    const size_t plane = work.plane;
    const ConstVec x_mean = m.x_mean, x_std = m.x_std;
    Vec xp = work.xp;
    Kokkos::parallel_for(
        "pad", Kokkos::RangePolicy<>(0, plane * n_in), KOKKOS_LAMBDA(const size_t i) {
            const size_t p = on_device ? i % plane : i / n_in;
            const int c = on_device ? i / plane : i % n_in;
            const int ix = p % px, iy = (p / px) % py, iz = p / ((size_t)px * py);
            const int sz = ((iz - h) % nz + nz) % nz;         // periodic
            const int sy = Kokkos::clamp(iy - h, 0, ny - 1);  // edge copy
            const int sx = ((ix - h) % nx + nx) % nx;         // periodic
            xp(xp_index(p, c, plane, n_in)) =
                (x(((size_t)c * nz + sz) * ny * nx + (size_t)sy * nx + sx) - x_mean(c)) / x_std(c);
        });
}

// ----------------------------------------------------------------------------- host path

// Simple kernel, one team per cell: the host (Serial/OpenMP) path and a cross-check.
// work.xp filled (layout (b, z, y, x, c), so the stencil sum reads contiguous channels).
// out: (n_out, ncell) on the device, the outputs of net 0 first, then those of net 1.
inline void layers_simple(const Model &m, Work &work, Vec &out) {
    const int k = m.stencil, n_in = m.n_in, py = work.py, px = work.px;
    const size_t ncell = work.ncell;
    const Vec xp = work.xp;
    const Kokkos::View<const int *> corner = work.corner;

    using Team = Kokkos::TeamPolicy<>::member_type;
    using Scratch = Kokkos::View<Real *, Kokkos::DefaultExecutionSpace::scratch_memory_space,
                                 Kokkos::MemoryUnmanaged>;
    const int max_width = m.max_width;
    const size_t scratch_bytes = 2 * Scratch::shmem_size(max_width);
    Kokkos::TeamPolicy<> policy(ncell, Kokkos::AUTO);
    policy.set_scratch_size(0, Kokkos::PerTeam(scratch_bytes));

    Kokkos::parallel_for(
        "mlp", policy, KOKKOS_LAMBDA(const Team &team) {
            const int cell = team.league_rank();
            const size_t c0 = corner(cell);
            Scratch a(team.team_scratch(0), max_width);
            Scratch a2(team.team_scratch(0), max_width);

            int out0 = 0;  // first output slot of the current net
            for (int n = 0; n < m.n_nets; ++n) {
                const DeviceNet &net = m.net[n];
                // conv: w0 index ((dz*k + dy)*k + dx)*n_in + c, then output channel
                const int H0 = net.fan_out[0];
                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, H0), [&](const int o) {
                    Real acc = net.b[0](o);
                    for (int dz = 0; dz < k; ++dz)
                        for (int dy = 0; dy < k; ++dy)
                            for (int dx = 0; dx < k; ++dx) {
                                const size_t tap = (size_t)((dz * k + dy) * k + dx) * n_in;
                                const size_t xin =
                                    (c0 + ((size_t)dz * py + dy) * px + dx) * n_in;
                                for (int c = 0; c < n_in; ++c)
                                    acc += xp(xin + c) * net.w[0]((tap + c) * H0 + o);
                            }
                    a(o) = (net.n_layers == 1) ? acc : gelu(acc);
                });
                team.team_barrier();

                // pointwise dense layers (gelu on all but the last), ping-pong a <-> a2
                for (int l = 1; l < net.n_layers; ++l) {
                    const int nin = net.fan_in[l], nout = net.fan_out[l];
                    const bool last = (l == net.n_layers - 1);
                    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nout), [&](const int o) {
                        Real acc = net.b[l](o);
                        for (int i = 0; i < nin; ++i) acc += a(i) * net.w[l]((size_t)i * nout + o);
                        a2(o) = last ? acc : gelu(acc);
                    });
                    team.team_barrier();
                    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nout),
                                         [&](const int o) { a(o) = a2(o); });
                    team.team_barrier();
                }
                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, net.n_out), [&](const int o) {
                    out((size_t)(out0 + o) * ncell + cell) = a(o);
                });
                team.team_barrier();  // a is reused by the next net
                out0 += net.n_out;
            }
        });
}

// ----------------------------------------------------------------------------- GPU path

struct alignas(16) F4 { Real x, y, z, w; };

// One layer C = act(W^T A + b).  A is either dense (A[k * lda + m]) or, for the conv, gathered:
// A[k][m] = xp[c(k) * plane + corner[m] + offset(k)] with (c, dz, dy, dx) of row k from taps.
struct Layer {
    const Real *A = nullptr;
    int lda = 0;
    bool conv = false;
    const Real *xp = nullptr;
    const int *taps = nullptr, *corner = nullptr;
    int plane = 0, pyx = 0, px = 0, ncell = 0;
    const Real *W = nullptr, *B = nullptr;
    int K = 0, N = 0;
    Real *C = nullptr;
    int ldc = 0, m_valid = 0;  // C[n * ldc + m] for m < m_valid
    bool gelu = false;
};

KOKKOS_INLINE_FUNCTION F4 load4(const Real *p) { return *reinterpret_cast<const F4 *>(p); }
KOKKOS_INLINE_FUNCTION void store4(Real *p, const F4 &v) { *reinterpret_cast<F4 *>(p) = v; }

// BM x BM output tile, TR x TR outputs per thread in TR/4 float4 groups spread over the tile
// (bank-conflict-free shared loads), BK = 1024 / BM rows of A and W per stage.
template <int BM, int TR> inline void gemm_layer(const Layer &L, int M) {
    constexpr int BN = BM, BK = 1024 / BM, NT = BN / TR, G = TR / 4, GS_M = BM / G, GS_N = BN / G;
    static_assert((BM / TR) * (BN / TR) == TEAM, "TEAM threads per tile");
    static_assert(BK * BM == 4 * TEAM, "one float4 of A and of W per thread per stage");
    using Policy = Kokkos::TeamPolicy<Kokkos::LaunchBounds<TEAM, 2>>;
    using Team = Policy::member_type;
    const int nbn = (L.N + BN - 1) / BN, nbm = (M + BM - 1) / BM;
    Policy policy(nbm * nbn, TEAM);
    const size_t bytes = sizeof(Real) * 2 * BK * (BM + BN) + 64;
    policy.set_scratch_size(0, Kokkos::PerTeam(bytes));

    Kokkos::parallel_for(
        "gemm_layer", policy, KOKKOS_LAMBDA(const Team &team) {
            Real *As = static_cast<Real *>(team.team_scratch(0).get_shmem_aligned(bytes - 64, 16));
            Real *Ws = As + 2 * BK * BM;
            const int tid = team.team_rank(), tx = tid % NT, ty = tid / NT;
            const int bm = team.league_rank() / nbn, bn = team.league_rank() % nbn;
            const int m0 = bm * BM, n0 = bn * BN;
            const int lr = tid / (BM / 4), lc = (tid % (BM / 4)) * 4;  // this thread's float4
            const bool vec_w = (L.N % 4 == 0);
            int corner[4] = {0, 0, 0, 0};  // conv: this thread's 4 cells are the same for every k
            if (L.conv)
#pragma unroll
                for (int q = 0; q < 4; ++q) corner[q] = L.corner[Kokkos::min(m0 + lc + q, L.ncell - 1)];

            F4 ra, rw;
            auto gload = [&](int k0) {
                const int k = k0 + lr;
                if (k >= L.K) {
                    ra = F4{0, 0, 0, 0};
                    rw = F4{0, 0, 0, 0};
                    return;
                }
                if (L.conv) {
                    const int tp = L.taps[k];
                    const int off = (tp & 1023) * L.plane + (tp >> 18) * L.pyx +
                                    ((tp >> 14) & 15) * L.px + ((tp >> 10) & 15);
                    const Real *xk = L.xp + off;
                    ra = F4{xk[corner[0]], xk[corner[1]], xk[corner[2]], xk[corner[3]]};
                } else {
                    ra = load4(L.A + (size_t)k * L.lda + m0 + lc);
                }
                const int n = n0 + lc;
                if (vec_w && n + 3 < L.N) {
                    rw = load4(L.W + (size_t)k * L.N + n);
                } else {
                    Real v[4];
#pragma unroll
                    for (int q = 0; q < 4; ++q)
                        v[q] = (n + q < L.N) ? L.W[(size_t)k * L.N + n + q] : Real(0);
                    rw = F4{v[0], v[1], v[2], v[3]};
                }
            };
            auto sstore = [&](int buf) {
                store4(As + buf * BK * BM + lr * BM + lc, ra);
                store4(Ws + buf * BK * BN + lr * BN + lc, rw);
            };

            Real acc[TR][TR];
#pragma unroll
            for (int i = 0; i < TR; ++i)
#pragma unroll
                for (int j = 0; j < TR; ++j) acc[i][j] = 0;

            const int nt = (L.K + BK - 1) / BK;
            gload(0);
            sstore(0);
            team.team_barrier();
            for (int kt = 0; kt < nt; ++kt) {
                const int buf = kt & 1;
                if (kt + 1 < nt) gload((kt + 1) * BK);  // in flight during the arithmetic
                const Real *a_s = As + buf * BK * BM, *w_s = Ws + buf * BK * BN;
#pragma unroll
                for (int kk = 0; kk < BK; ++kk) {
                    Real a[TR], w[TR];
#pragma unroll
                    for (int g = 0; g < G; ++g) {
                        const F4 va = load4(a_s + kk * BM + g * GS_M + ty * 4);
                        const F4 vw = load4(w_s + kk * BN + g * GS_N + tx * 4);
                        a[4 * g] = va.x, a[4 * g + 1] = va.y, a[4 * g + 2] = va.z, a[4 * g + 3] = va.w;
                        w[4 * g] = vw.x, w[4 * g + 1] = vw.y, w[4 * g + 2] = vw.z, w[4 * g + 3] = vw.w;
                    }
#pragma unroll
                    for (int i = 0; i < TR; ++i)
#pragma unroll
                        for (int j = 0; j < TR; ++j) acc[i][j] += a[i] * w[j];
                }
                if (kt + 1 < nt) sstore(buf ^ 1);  // the other buffer was last read before the
                team.team_barrier();                // previous barrier
            }

            // epilogue: bias, gelu, store (4 consecutive cells per float4); fully unrolled so
            // that acc stays in registers
#pragma unroll
            for (int gj = 0; gj < G; ++gj)
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int n = n0 + gj * GS_N + tx * 4 + j;
                    if (n >= L.N) continue;
                    const Real b = L.B[n];
#pragma unroll
                    for (int gi = 0; gi < G; ++gi) {
                        const int mb = m0 + gi * GS_M + ty * 4;
                        Real v[4];
#pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            v[i] = acc[4 * gi + i][4 * gj + j] + b;
                            if (L.gelu) v[i] = gelu(v[i]);
                        }
                        Real *c = L.C + (size_t)n * L.ldc + mb;
                        if (mb + 3 < L.m_valid && L.ldc % 4 == 0)
                            store4(c, F4{v[0], v[1], v[2], v[3]});
                        else
#pragma unroll
                            for (int i = 0; i < 4; ++i)
                                if (mb + i < L.m_valid) c[i] = v[i];
                    }
                }
        });
}

// Narrow layer (N <= NARROW): one thread per cell, a team of 128 cells, with W (K x N) staged in
// shared memory and read as warp-wide broadcasts; four interleaved k streams per thread keep
// several loads of A in flight.
inline void narrow_layer(const Layer &L, int M) {
    constexpr int NT = 128;
    using Team = Kokkos::TeamPolicy<>::member_type;
    Kokkos::TeamPolicy<> policy((M + NT - 1) / NT, NT);
    const size_t bytes = sizeof(Real) * (size_t)L.K * NARROW + 64;
    if ((int)bytes > Kokkos::TeamPolicy<>::scratch_size_max(0))
        throw std::runtime_error("narrow layer fan-in too large for shared memory");
    policy.set_scratch_size(0, Kokkos::PerTeam(bytes));
    Kokkos::parallel_for(
        "narrow_layer", policy, KOKKOS_LAMBDA(const Team &team) {
            Real *Wsh = static_cast<Real *>(team.team_scratch(0).get_shmem_aligned(bytes - 64, 16));
            for (int e = team.team_rank(); e < L.K * NARROW; e += NT) {  // Wsh[k][j], N padded to 8
                const int k = e / NARROW, j = e % NARROW;
                Wsh[e] = (j < L.N) ? L.W[(size_t)k * L.N + j] : Real(0);
            }
            team.team_barrier();
            const int m = team.league_rank() * NT + team.team_rank();
            if (m >= M) return;
            Real acc[4][NARROW] = {};
            int k = 0;
            for (; k + 4 <= L.K; k += 4) {
#pragma unroll
                for (int s = 0; s < 4; ++s) {
                    const Real a = L.A[(size_t)(k + s) * L.lda + m];
                    const F4 w0 = *reinterpret_cast<const F4 *>(Wsh + (k + s) * NARROW);
                    const F4 w1 = *reinterpret_cast<const F4 *>(Wsh + (k + s) * NARROW + 4);
                    acc[s][0] += a * w0.x, acc[s][1] += a * w0.y, acc[s][2] += a * w0.z,
                        acc[s][3] += a * w0.w;
                    acc[s][4] += a * w1.x, acc[s][5] += a * w1.y, acc[s][6] += a * w1.z,
                        acc[s][7] += a * w1.w;
                }
            }
            for (; k < L.K; ++k) {
                const Real a = L.A[(size_t)k * L.lda + m];
#pragma unroll
                for (int j = 0; j < NARROW; ++j) acc[0][j] += a * Wsh[k * NARROW + j];
            }
            if (m < L.m_valid)
#pragma unroll
                for (int j = 0; j < NARROW; ++j)
                    if (j < L.N) {
                        const Real v = L.B[j] + (acc[0][j] + acc[1][j]) + (acc[2][j] + acc[3][j]);
                        L.C[(size_t)j * L.ldc + m] = L.gelu ? gelu(v) : v;
                    }
        });
}

inline void run_layer(const Layer &L, int M) {
    if (!L.conv && L.N <= NARROW)
        narrow_layer(L, M);
    else if (L.N > 64)
        gemm_layer<128, 8>(L, M);
    else
        gemm_layer<64, 4>(L, M);
}

// work.xp filled (layout (c, b, z, y, x)); out as layers_simple.
inline void layers_gemm(const Model &m, Work &work, Vec &out) {
    const int ncell = work.ncell, py = work.py, px = work.px;
    int out0 = 0;
    for (int n = 0; n < m.n_nets; ++n) {
        const DeviceNet &net = m.net[n];
        for (int l = 0; l < net.n_layers; ++l) {
            Layer L;
            const bool last = (l == net.n_layers - 1);
            L.conv = (l == 0);
            if (L.conv) {
                L.xp = work.xp.data(), L.taps = m.taps.data(), L.corner = work.corner.data();
                L.plane = work.plane, L.pyx = py * px, L.px = px;
            } else {
                L.A = work.act[(l - 1) % 2].data(), L.lda = work.mpad;
            }
            L.ncell = ncell;
            L.W = m.flat.data() + net.w_off[l], L.B = m.flat.data() + net.b_off[l];
            L.K = net.fan_in[l], L.N = net.fan_out[l];
            if (last) {
                L.C = out.data() + (size_t)out0 * ncell, L.ldc = ncell, L.m_valid = ncell;
            } else {
                L.C = work.act[l % 2].data(), L.ldc = work.mpad, L.m_valid = work.mpad;
            }
            L.gelu = !last;
            run_layer(L, work.mpad);
        }
        out0 += net.n_out;
    }
}

// The network on a filled work.xp: the GEMM kernels on a GPU, the simple kernel on the host.
// out: (n_out, ncell), at least n_out * work.ncell long.
inline void layers(const Model &m, Work &work, Vec &out) {
    if (out.size() < (size_t)m.n_out * work.ncell)
        throw std::runtime_error("forward pass: output buffer too small");
    if constexpr (on_device)
        layers_gemm(m, work, out);
    else
        layers_simple(m, work, out);
}

// The forward pass of one grid with the training boundaries.
// x: raw (n_in, nz, ny, nx); out: (n_out, nz, ny, nx); work: make_work(m, nz, ny, nx).
inline void forward(const Model &m, const Vec &x, Work &work, Vec &out) {
    pad(m, x, work);
    layers(m, work, out);
}

// ----------------------------------------------------------------------------- test helpers

// Largest |a - b| over two host arrays, and the largest relative to max(1, |b|).
inline void compare(const std::vector<Real> &a, const std::vector<Real> &b, const char *name,
                    double tol, bool &ok) {
    double max_abs = 0, max_rel = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = std::fabs((double)a[i] - b[i]);
        max_abs = std::max(max_abs, d);
        max_rel = std::max(max_rel, d / std::max(1.0, (double)std::fabs(b[i])));
    }
    std::printf("%-8s max |kokkos - jax| = %.3e, max rel = %.3e\n", name, max_abs, max_rel);
    if (!(max_abs < tol)) ok = false;
}

inline std::vector<Real> to_host(const Vec &v) {
    std::vector<Real> h(v.size());
    Kokkos::deep_copy(Vec::host_mirror_type(h.data(), h.size()), v);
    return h;
}

inline Vec from_host(const std::string &label, const std::vector<Real> &h) {
    Vec d(label, h.size());
    Kokkos::deep_copy(d, Vec::host_mirror_type(const_cast<Real *>(h.data()), h.size()));
    return d;
}

// Common main(): initialises Kokkos, runs `run`, finalises (Views die inside run).
template <class F> inline int main_wrapper(int argc, char *argv[], F run) {
    Kokkos::initialize(argc, argv);
    int rc;
    try {
        rc = run();
    } catch (const std::exception &e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        rc = 2;
    }
    Kokkos::finalize();
    return rc;
}

}  // namespace sn
