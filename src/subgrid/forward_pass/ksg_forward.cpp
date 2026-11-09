// Forward pass of the conserved-variable subgrid kinetic energy model
// (models/conserved/ksg_model.py, PredictKsg) with Kokkos, on any backend.
//
// Input u0 (5, nz, ny, nx): rho, m_x, m_y, m_z, E.  The 20 network inputs are built here as in
// conserved.features (model_ops.hpp).  The network returns z_0, then
//   f = sigmoid(z_0) = k_sg / k_max,   k_max = (E - m^2 / 2 rho) / rho,   k_sg = f k_max.
//
// Test: loads data/ksg/{weights,reference}.bin written by export_weights.py ksg and compares
// to the JAX output on a cached frame.  Build/run: ./build_and_run.sh
#include "model_ops.hpp"

static int run() {
    using namespace sn;
    const std::string dir = std::string(FP_DATA_DIR) + "ksg/";
    const char *rep = std::getenv("SN_REPEAT");  // timed repeats
    const int N_REPEAT = rep ? std::atoi(rep) : 20;
    Model m = load_model(dir + "weights.bin");
    describe(m);
    if (m.n_in != N_FEAT) throw std::runtime_error("k_sg model must have 20 input channels");
    if (m.n_out != 1) throw std::runtime_error("k_sg model must have n_out = 1");

    std::ifstream f(dir + "reference.bin", std::ios::binary);
    if (!f) throw std::runtime_error("cannot open reference.bin");
    std::vector<int32_t> dims;
    read_vec(f, dims, 3);
    const int nz = dims[0], ny = dims[1], nx = dims[2];
    const size_t ncell = (size_t)nz * ny * nx;
    std::vector<Real> u_h, z_ref, f_ref, k_ref;
    read_vec(f, u_h, N_U * ncell);
    read_vec(f, z_ref, ncell);
    read_vec(f, f_ref, ncell);
    read_vec(f, k_ref, ncell);

    Vec u = from_host("u0", u_h), feat("feat", N_FEAT * ncell);
    Work work = make_work(m, nz, ny, nx);
    Vec z("z", ncell), fv("f", ncell), ksg("ksg", ncell);
    auto pass = [&]() {
        features(u, feat, nz, ny, nx);
        forward(m, feat, work, z);
        ksg_from_z(z, u, fv, ksg, ncell);
    };
    pass();
    Kokkos::fence();
    Kokkos::Timer timer;
    for (int r = 0; r < N_REPEAT; ++r) pass();
    Kokkos::fence();
    std::printf("grid %dx%dx%d: %.3f ms per forward pass (mean of %d)\n", nz, ny, nx,
                1e3 * timer.seconds() / N_REPEAT, N_REPEAT);

    // f spans ~8 decades: compare z (= logit f) absolutely and f, k_sg relatively
    bool ok = true;
    compare(to_host(z), z_ref, "z0", 1e-3, ok);
    auto rel = [&](const std::vector<Real> &a, const std::vector<Real> &b, const char *name) {
        double r = 0;
        for (size_t i = 0; i < a.size(); ++i)
            r = std::max(r, std::fabs((double)a[i] / b[i] - 1));
        std::printf("%-8s max |kokkos / jax - 1| = %.3e\n", name, r);
        if (!(r < 1e-3)) ok = false;
    };
    rel(to_host(fv), f_ref, "f");
    rel(to_host(ksg), k_ref, "k_sg");
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

int main(int argc, char *argv[]) { return sn::main_wrapper(argc, argv, run); }
