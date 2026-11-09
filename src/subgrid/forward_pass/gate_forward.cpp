// Forward pass of the gate model (models/gate_model.py) with Kokkos, on any backend.
//
// Inputs (6, nz, ny, nx): log_rho, log_T, ux, uy, uz, s0.  The 24 network inputs are built
// here as in gate_model.features: those six, grad log_rho, grad log_T, grad s0 (central
// differences in coarse cells, d/dx d/dy d/dz), the strain rate S_ij of u (xx yy zz xy xz yz)
// and the vorticity curl u, with periodic z, x and edge-copied y.
// Outputs: p = sigmoid(logit), the probability of mixing gas (gate = p > 0.5), and, if the
// model has the auxiliary head, f_mix = sigmoid(second output).
//
// Test: loads data/gate/{weights,reference}.bin written by export_weights.py gate and compares
// to the JAX output.  Paths are hard-coded.  Build/run: ./build_and_run.sh
#include "model_ops.hpp"

static const char *DIR = "/home/sasi/Projects/SubgridCGMModel/models/forward_pass_c/data/gate/";
static int run() {
    using namespace sn;
    const char *rep = std::getenv("SN_REPEAT");  // timed repeats
    const int N_REPEAT = rep ? std::atoi(rep) : 20;
    Model m = load_model(std::string(DIR) + "weights.bin");
    describe(m);
    if (m.n_in != N_FEAT) throw std::runtime_error("gate model must have 24 input channels");

    std::ifstream f(std::string(DIR) + "reference.bin", std::ios::binary);
    if (!f) throw std::runtime_error("cannot open reference.bin");
    std::vector<int32_t> dims;
    read_vec(f, dims, 3);
    const int nz = dims[0], ny = dims[1], nx = dims[2];
    const size_t ncell = (size_t)nz * ny * nx;
    std::vector<Real> x_h, logit_ref, f_ref;
    read_vec(f, x_h, N_BASE * ncell);
    read_vec(f, logit_ref, ncell);
    if (m.n_out > 1) read_vec(f, f_ref, ncell);

    Vec base = from_host("base", x_h), feat("feat", N_FEAT * ncell);
    Work work = make_work(m, nz, ny, nx);
    Vec out("out", m.n_out * ncell);
    auto pass = [&]() {
        features(base, feat, nz, ny, nx);
        forward(m, feat, work, out);
    };
    pass();
    Kokkos::fence();
    Kokkos::Timer timer;
    for (int r = 0; r < N_REPEAT; ++r) pass();
    Kokkos::fence();
    std::printf("grid %dx%dx%d: %.3f ms per forward pass (mean of %d)\n", nz, ny, nx,
                1e3 * timer.seconds() / N_REPEAT, N_REPEAT);

    // p = sigmoid(logit), f_mix = sigmoid(out 1)
    std::vector<Real> o = to_host(out), p(ncell), fm;
    int flips = 0;
    for (size_t i = 0; i < ncell; ++i) {
        p[i] = 1.0f / (1.0f + std::exp(-o[i]));
        flips += ((o[i] > 0) != (logit_ref[i] > 0));
    }
    bool ok = true;
    compare(std::vector<Real>(o.begin(), o.begin() + ncell), logit_ref, "logit", 1e-3, ok);
    if (m.n_out > 1) {
        for (size_t i = 0; i < ncell; ++i) fm.push_back(1.0f / (1.0f + std::exp(-o[ncell + i])));
        std::vector<Real> fr(ncell);
        for (size_t i = 0; i < ncell; ++i) fr[i] = 1.0f / (1.0f + std::exp(-f_ref[i]));
        compare(fm, fr, "f_mix", 1e-4, ok);
    }
    std::printf("gate decisions differing from JAX: %d of %zu\n", flips, ncell);
    ok = ok && flips == 0;
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

int main(int argc, char *argv[]) { return sn::main_wrapper(argc, argv, run); }
