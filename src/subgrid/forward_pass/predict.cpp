// Batch prediction with the Kokkos forward passes, for the benchmarks (benchmarking/cpp_backend.py):
//
//   predict <gate|cooling|rvv> <weights.bin> <in.bin> <out.bin>
//
// in.bin:  int32 n_frames, n_chan, nz, ny, nx; float32 x[n_frames][n_chan][nz][ny][nx], the raw
//          (unnormalised, unpadded) inputs:
//            gate     6 cached fields (log_rho, log_T, ux, uy, uz, s0); features built here
//            cooling  the 6 fields plus the gate channel
//            rvv      log_rho, log_kmax, ux, uy, uz, s0
// out.bin: int32 n_frames, n_out, nz, ny, nx; float32 y[n_frames][n_out][nz][ny][nx]:
//            gate     the raw network outputs (logit of the gate, then the f_mix logit if any)
//            cooling  z
//            rvv      z_0..z_5, f, S (xx yy zz xy xz yz): 13 channels
// Prints the mean time per frame of the forward pass on the device (inputs already there) and
// including the host <-> device copies.
#include "model_ops.hpp"

#include <chrono>

static int run(int argc, char *argv[]) {
    using namespace sn;
    if (argc != 5) throw std::runtime_error("usage: predict <gate|cooling|rvv> weights.bin in.bin out.bin");
    const std::string kind = argv[1];
    if (kind != "gate" && kind != "cooling" && kind != "rvv")
        throw std::runtime_error("unknown model kind " + kind);
    Model m = load_model(argv[2]);

    std::ifstream f(argv[3], std::ios::binary);
    if (!f) throw std::runtime_error(std::string("cannot open ") + argv[3]);
    std::vector<int32_t> dims;
    read_vec(f, dims, 5);
    const int n_frames = dims[0], n_chan = dims[1], nz = dims[2], ny = dims[3], nx = dims[4];
    const size_t ncell = (size_t)nz * ny * nx;
    const int want_chan = (kind == "gate") ? N_BASE : m.n_in;
    if (n_chan != want_chan)
        throw std::runtime_error(kind + " expects " + std::to_string(want_chan) + " input channels, got " +
                                 std::to_string(n_chan));
    if (kind == "gate" && m.n_in != N_FEAT) throw std::runtime_error("gate model must have 24 inputs");
    if (kind == "rvv" && m.n_out != 6) throw std::runtime_error("rvv model must have 6 outputs");
    std::vector<Real> x_h;
    read_vec(f, x_h, (size_t)n_frames * n_chan * ncell);

    const int n_out = (kind == "rvv") ? 13 : m.n_out;
    std::vector<Real> y_h((size_t)n_frames * n_out * ncell);

    Work work = make_work(m, nz, ny, nx);
    Vec x("x", n_chan * ncell), feat("feat", kind == "gate" ? N_FEAT * ncell : 0);
    Vec out("out", m.n_out * ncell), fv("f", ncell), S("S", 6 * ncell);
    auto compute = [&]() {
        if (kind == "gate") {
            features(x, feat, nz, ny, nx);
            forward(m, feat, work, out);
        } else {
            forward(m, x, work, out);
            if (kind == "rvv") shape_from_z(out, fv, S, ncell);
        }
    };

    double t_compute = 0, t_total = 0;
    using clock = std::chrono::steady_clock;
    for (int i = -1; i < n_frames; ++i) {  // i = -1: warm-up on frame 0, not timed or written
        const int fr = std::max(i, 0);
        const auto t0 = clock::now();
        Kokkos::deep_copy(x, Vec::host_mirror_type(x_h.data() + (size_t)fr * n_chan * ncell,
                                                   n_chan * ncell));
        Kokkos::fence();
        const auto t1 = clock::now();
        compute();
        Kokkos::fence();
        const auto t2 = clock::now();
        Real *y = y_h.data() + (size_t)fr * n_out * ncell;
        Kokkos::deep_copy(Vec::host_mirror_type(y, m.n_out * ncell), out);
        if (kind == "rvv") {
            Kokkos::deep_copy(Vec::host_mirror_type(y + 6 * ncell, ncell), fv);
            Kokkos::deep_copy(Vec::host_mirror_type(y + 7 * ncell, 6 * ncell), S);
        }
        const auto t3 = clock::now();
        if (i >= 0) {
            t_compute += std::chrono::duration<double>(t2 - t1).count();
            t_total += std::chrono::duration<double>(t3 - t0).count();
        }
    }

    std::ofstream o(argv[4], std::ios::binary);
    const int32_t head[5] = {n_frames, n_out, nz, ny, nx};
    o.write(reinterpret_cast<const char *>(head), sizeof(head));
    o.write(reinterpret_cast<const char *>(y_h.data()), y_h.size() * sizeof(Real));
    if (!o) throw std::runtime_error(std::string("cannot write ") + argv[4]);
    std::printf("%s on %s: %d frames of %dx%dx%d, %.3f ms per frame on the device, "
                "%.3f ms with host <-> device copies\n",
                kind.c_str(), Kokkos::DefaultExecutionSpace::name(), n_frames, nz, ny, nx,
                1e3 * t_compute / n_frames, 1e3 * t_total / n_frames);
    return 0;
}

int main(int argc, char *argv[]) {
    Kokkos::initialize(argc, argv);  // removes the --kokkos-* arguments
    int rc;
    try {
        rc = run(argc, argv);
    } catch (const std::exception &e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        rc = 2;
    }
    Kokkos::finalize();
    return rc;
}
