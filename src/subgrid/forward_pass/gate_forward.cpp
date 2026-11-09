// Forward pass of the conserved-variable gate model (models/conserved/gate_model.py) with
// Kokkos, on any backend.
//
// Input u0 (5, nz, ny, nx): rho, m_x, m_y, m_z, E.  The 20 network inputs are built here as in
// conserved.features (model_ops.hpp).  Outputs: p = sigmoid(logit), the probability of mixing
// gas (gate = p > 0.5), and, if the model has the auxiliary head, f_mix = sigmoid(second output).
//
// Test: loads data/gate/{weights,reference}.bin written by export_weights.py gate and compares
// to the JAX output on a cached frame.  Build/run: ./build_and_run.sh
#include "model_ops.hpp"

static int run() {
    using namespace sn;
    const std::string dir = std::string(FP_DATA_DIR) + "gate/";
    const char *rep = std::getenv("SN_REPEAT");  // timed repeats
    const int N_REPEAT = rep ? std::atoi(rep) : 20;
    Model m = load_model(dir + "weights.bin");
    describe(m);
    if (m.n_in != N_FEAT) throw std::runtime_error("gate model must have 20 input channels");

    std::ifstream f(dir + "reference.bin", std::ios::binary);
    if (!f) throw std::runtime_error("cannot open reference.bin");
    std::vector<int32_t> dims;
    read_vec(f, dims, 3);
    const int nz = dims[0], ny = dims[1], nx = dims[2];
    const size_t ncell = (size_t)nz * ny * nx;
    std::vector<Real> u_h, logit_ref, f_ref;
    read_vec(f, u_h, N_U * ncell);
    read_vec(f, logit_ref, ncell);
    if (m.n_out > 1) read_vec(f, f_ref, ncell);

    Vec u = from_host("u0", u_h), feat("feat", N_FEAT * ncell);
    Work work = make_work(m, nz, ny, nx);
    Vec out("out", m.n_out * ncell);
    auto pass = [&]() {
        features(u, feat, nz, ny, nx);
        forward(m, feat, work, out);
    };
    pass();
    Kokkos::fence();
    Kokkos::Timer timer;
    for (int r = 0; r < N_REPEAT; ++r) pass();
    Kokkos::fence();
    std::printf("grid %dx%dx%d: %.3f ms per forward pass (mean of %d)\n", nz, ny, nx,
                1e3 * timer.seconds() / N_REPEAT, N_REPEAT);

    // decisions: count flips, and the flips of cells whose JAX logit is not within the
    // tolerance of 0 (those would be real disagreements, not round-off at the boundary)
    std::vector<Real> o = to_host(out);
    int flips = 0, hard_flips = 0;
    for (size_t i = 0; i < ncell; ++i)
        if ((o[i] > 0) != (logit_ref[i] > 0)) {
            ++flips;
            hard_flips += std::fabs(logit_ref[i]) > 1e-3f;
        }
    bool ok = true;
    compare(std::vector<Real>(o.begin(), o.begin() + ncell), logit_ref, "logit", 1e-3, ok);
    if (m.n_out > 1) {
        std::vector<Real> fm(ncell), fr(ncell);
        for (size_t i = 0; i < ncell; ++i) {
            fm[i] = 1.0f / (1.0f + std::exp(-o[ncell + i]));
            fr[i] = 1.0f / (1.0f + std::exp(-f_ref[i]));
        }
        compare(fm, fr, "f_mix", 1e-4, ok);
    }
    std::printf("gate decisions differing from JAX: %d of %zu (%d with |logit| > 1e-3)\n", flips,
                ncell, hard_flips);
    ok = ok && hard_flips == 0;
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

int main(int argc, char *argv[]) { return sn::main_wrapper(argc, argv, run); }
