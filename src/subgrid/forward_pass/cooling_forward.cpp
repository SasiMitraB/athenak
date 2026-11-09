// Forward pass of the conserved-variable cooling model (models/conserved/cooling_model.py)
// with Kokkos, on any backend.
//
// Input (6, nz, ny, nx): u0 = rho, m_x, m_y, m_z, E and the gate channel (0/1: the gate the
// model was trained with, gate_forward's prediction for the predicted-gate model).  The network
// inputs are the 20 of conserved.features (model_ops.hpp) plus the gate, and, for a model
// trained with --ksg-model (23 inputs), log10 f and log10 (1 - f) of the k_sg network
// (ksg_weights.bin, run here on the same features: the PredictKsg -> PredictC chain).
// Output: z (nz, ny, nx), the log of the cooling factor of eq-cooling-model referred to
// Lfull(T_U), T_U from E - m^2 / 2 rho; the caller builds
//   C = g (rho / mu m_u)^2 [Lfull(T_U) e^z - Lcut(T_CG)]
// where the gate is 1.
//
// Test: loads data/cooling/{weights,ksg_weights,reference}.bin written by export_weights.py
// cooling and compares to the JAX output on a cached frame.  Build/run: ./build_and_run.sh
#include "model_ops.hpp"

static int run() {
    using namespace sn;
    const std::string dir = std::string(FP_DATA_DIR) + "cooling/";
    const char *rep = std::getenv("SN_REPEAT");  // timed repeats
    const int N_REPEAT = rep ? std::atoi(rep) : 20;
    Model m = load_model(dir + "weights.bin");
    describe(m);
    const bool with_ksg = cooling_takes_ksg(m);
    Model mk;
    if (with_ksg) {
        mk = load_model(dir + "ksg_weights.bin");
        std::printf("k_sg inputs from the k_sg network:\n");
        describe(mk);
        check_ksg(mk);
    }

    std::ifstream f(dir + "reference.bin", std::ios::binary);
    if (!f) throw std::runtime_error("cannot open reference.bin");
    std::vector<int32_t> dims;
    read_vec(f, dims, 3);
    const int nz = dims[0], ny = dims[1], nx = dims[2];
    const size_t ncell = (size_t)nz * ny * nx;
    std::vector<Real> x_h, z_ref, ksg_ref;
    read_vec(f, x_h, (N_U + 1) * ncell);
    read_vec(f, z_ref, ncell);
    if (with_ksg) read_vec(f, ksg_ref, N_KSG_IN * ncell);

    Vec x = from_host("x", x_h), feat("feat", m.n_in * ncell);
    Work work = make_work(m, nz, ny, nx);
    Work ksg_work = with_ksg ? make_work(mk, nz, ny, nx) : Work();
    Vec z("z", ncell), z0("z0", with_ksg ? ncell : 0);
    auto pass = [&]() {
        cooling_inputs(x, feat, nz, ny, nx, with_ksg ? &mk : nullptr, &ksg_work, &z0);
        forward(m, feat, work, z);
    };
    pass();
    Kokkos::fence();
    Kokkos::Timer timer;
    for (int r = 0; r < N_REPEAT; ++r) pass();
    Kokkos::fence();
    std::printf("grid %dx%dx%d: %.3f ms per forward pass%s (mean of %d)\n", nz, ny, nx,
                1e3 * timer.seconds() / N_REPEAT, with_ksg ? " incl. the k_sg network" : "",
                N_REPEAT);

    bool ok = true;
    if (with_ksg) {
        std::vector<Real> fh = to_host(feat);
        compare(std::vector<Real>(fh.begin() + N_COOL_IN * ncell, fh.end()), ksg_ref,
                "ksg in", 1e-3, ok);
    }
    compare(to_host(z), z_ref, "z", 1e-3, ok);
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

int main(int argc, char *argv[]) { return sn::main_wrapper(argc, argv, run); }
