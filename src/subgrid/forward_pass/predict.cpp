// Batch prediction with the Kokkos forward passes of the conserved-variable models, for the
// benchmarks (benchmarking/cpp_backend.py):
//
//   predict <gate|cooling|ksg> <weights.bin> <in.bin> <out.bin> [<ksg_weights.bin>]
//
// ksg_weights.bin: for a cooling model trained with --ksg-model (23 inputs), the k_sg model
//          whose log10 f, log10 (1 - f) it takes; run here on the same u0 (model_ops.hpp
//          cooling_inputs).
// in.bin:  int32 n_frames, n_chan, nz, ny, nx; float32 x[n_frames][n_chan][nz][ny][nx], the raw
//          (unnormalised, unpadded) inputs, the network inputs are built here:
//            gate, ksg  u0 = rho, m_x, m_y, m_z, E (5 channels)
//            cooling    u0 plus the gate channel (6)
// out.bin: int32 n_frames, n_out, nz, ny, nx; float32 y[n_frames][n_out][nz][ny][nx]:
//            gate     the raw network outputs (logit of the gate, then the f_mix logit if any)
//            cooling  z
//            ksg      z_0, f, k_sg: 3 channels
// Prints the mean time per frame of the forward pass on the device (inputs already there) and
// including the host <-> device copies.
#include "model_ops.hpp"

#include <chrono>

static int run(int argc, char *argv[]) {
    using namespace sn;
    if (argc != 5 && argc != 6)
        throw std::runtime_error("usage: predict <gate|cooling|ksg> weights.bin in.bin out.bin [ksg_weights.bin]");
    const std::string kind = argv[1];
    if (kind != "gate" && kind != "cooling" && kind != "ksg")
        throw std::runtime_error("unknown model kind " + kind);
    Model m = load_model(argv[2]);
    const bool cooling = (kind == "cooling"), ksg = (kind == "ksg");
    const bool with_ksg = cooling && cooling_takes_ksg(m);
    if (!cooling && m.n_in != N_FEAT)
        throw std::runtime_error(kind + " model has " + std::to_string(m.n_in) +
                                 " inputs, not the conserved-variable features");
    if (ksg) check_ksg(m);
    if (with_ksg != (argc == 6))
        throw std::runtime_error(with_ksg ? "this cooling model takes k_sg inputs: give ksg_weights.bin"
                                          : "ksg_weights.bin given, but the model takes no k_sg inputs");
    Model mk;
    if (with_ksg) {
        mk = load_model(argv[5]);
        check_ksg(mk);
    }

    std::ifstream f(argv[3], std::ios::binary);
    if (!f) throw std::runtime_error(std::string("cannot open ") + argv[3]);
    std::vector<int32_t> dims;
    read_vec(f, dims, 5);
    const int n_frames = dims[0], n_chan = dims[1], nz = dims[2], ny = dims[3], nx = dims[4];
    const size_t ncell = (size_t)nz * ny * nx;
    const int want_chan = N_U + (cooling ? 1 : 0);
    if (n_chan != want_chan)
        throw std::runtime_error(kind + " expects " + std::to_string(want_chan) + " input channels, got " +
                                 std::to_string(n_chan));
    std::vector<Real> x_h;
    read_vec(f, x_h, (size_t)n_frames * n_chan * ncell);

    const int n_out = ksg ? 3 : m.n_out;
    std::vector<Real> y_h((size_t)n_frames * n_out * ncell);

    Work work = make_work(m, nz, ny, nx);
    Vec x("x", n_chan * ncell), feat("feat", m.n_in * ncell), out("out", m.n_out * ncell);
    Vec fv("f", ksg ? ncell : 0), kv("k_sg", ksg ? ncell : 0), z0("z0", with_ksg ? ncell : 0);
    Work ksg_work = with_ksg ? make_work(mk, nz, ny, nx) : Work();
    auto compute = [&]() {
        if (cooling)
            cooling_inputs(x, feat, nz, ny, nx, with_ksg ? &mk : nullptr, &ksg_work, &z0);
        else
            features(x, feat, nz, ny, nx);
        forward(m, feat, work, out);
        if (ksg) ksg_from_z(out, x, fv, kv, ncell);
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
        if (ksg) {
            Kokkos::deep_copy(Vec::host_mirror_type(y + ncell, ncell), fv);
            Kokkos::deep_copy(Vec::host_mirror_type(y + 2 * ncell, ncell), kv);
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
