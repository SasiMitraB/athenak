// Forward pass of the cooling model (models/cooling_model.py) with Kokkos, on any backend.
//
// Inputs (7, nz, ny, nx): log_rho, log_T, ux, uy, uz, s0 and the gate channel (0/1: the true
// gate or gate_forward's prediction).  Output: z (nz, ny, nx), the log of the cooling
// correction factor of eq-cooling-model; the caller applies it where the gate is 1.
//
// Test: loads data/cooling/{weights,reference}.bin written by export_weights.py cooling and
// compares to the JAX output.  Paths are hard-coded.  Build/run: ./build_and_run.sh
#include "stencil_net.hpp"

static const char *DIR = "/home/sasi/Projects/SubgridCGMModel/models/forward_pass_c/data/cooling/";

static int run() {
    using namespace sn;
    const char *rep = std::getenv("SN_REPEAT");  // timed repeats
    const int N_REPEAT = rep ? std::atoi(rep) : 20;
    Model m = load_model(std::string(DIR) + "weights.bin");
    describe(m);
    if (m.n_out != 1) throw std::runtime_error("cooling model must have n_out = 1");

    std::ifstream f(std::string(DIR) + "reference.bin", std::ios::binary);
    if (!f) throw std::runtime_error("cannot open reference.bin");
    std::vector<int32_t> dims;
    read_vec(f, dims, 3);
    const int nz = dims[0], ny = dims[1], nx = dims[2];
    const size_t ncell = (size_t)nz * ny * nx;
    std::vector<Real> x_h, z_ref;
    read_vec(f, x_h, m.n_in * ncell);
    read_vec(f, z_ref, ncell);

    Vec x = from_host("x", x_h);
    Work work = make_work(m, nz, ny, nx);
    Vec z("z", ncell);
    forward(m, x, work, z);
    Kokkos::fence();
    Kokkos::Timer timer;
    for (int r = 0; r < N_REPEAT; ++r) forward(m, x, work, z);
    Kokkos::fence();
    std::printf("grid %dx%dx%d: %.3f ms per forward pass (mean of %d)\n", nz, ny, nx,
                1e3 * timer.seconds() / N_REPEAT, N_REPEAT);

    bool ok = true;
    compare(to_host(z), z_ref, "z", 1e-3, ok);
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

int main(int argc, char *argv[]) { return sn::main_wrapper(argc, argv, run); }
