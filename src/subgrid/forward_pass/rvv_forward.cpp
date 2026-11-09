// Forward pass of the subgrid momentum-flux model (models/rvv_tensor_model.py) with Kokkos,
// on any backend.
//
// Inputs (6, nz, ny, nx): log_rho, log_kmax, ux, uy, uz, s0.  The network returns z_0..z_5
// (one network, or two: z_0 and z_1..z_5, export_weights.py rvv writes either); then
//   f = sigmoid(z_0) = k_sg / k_max,   A = sum_k z_k E^(k),   S = e^A / Tr e^A
// with the orthonormal symmetric traceless basis E^(k) of rvv_tensor_model.BASIS.  S is
// returned as 6 components (xx yy zz xy xz yz).  R_ij = 2 rho f k_max S_ij is left to the caller.
// The Galilean projection of the velocity weights is folded into the exported weights.
//
// Test: loads data/rvv/{weights,reference}.bin written by export_weights.py rvv and compares
// to the JAX output.  Paths are hard-coded.  Build/run: ./build_and_run.sh
#include "model_ops.hpp"

static const char *DIR = "/home/sasi/Projects/SubgridCGMModel/models/forward_pass_c/data/rvv/";

static int run() {
    using namespace sn;
    const char *rep = std::getenv("SN_REPEAT");  // timed repeats
    const int N_REPEAT = rep ? std::atoi(rep) : 20;
    Model m = load_model(std::string(DIR) + "weights.bin");
    describe(m);
    if (m.n_out != 6) throw std::runtime_error("rvv model must have 6 outputs in total");

    std::ifstream f(std::string(DIR) + "reference.bin", std::ios::binary);
    if (!f) throw std::runtime_error("cannot open reference.bin");
    std::vector<int32_t> dims;
    read_vec(f, dims, 3);
    const int nz = dims[0], ny = dims[1], nx = dims[2];
    const size_t ncell = (size_t)nz * ny * nx;
    std::vector<Real> x_h, f_ref, S_ref;
    read_vec(f, x_h, m.n_in * ncell);
    read_vec(f, f_ref, ncell);
    read_vec(f, S_ref, 6 * ncell);

    Vec x = from_host("x", x_h);
    Work work = make_work(m, nz, ny, nx);
    Vec z("z", 6 * ncell), fv("f", ncell), S("S", 6 * ncell);
    auto pass = [&]() {
        forward(m, x, work, z);
        shape_from_z(z, fv, S, ncell);
    };
    pass();
    Kokkos::fence();
    Kokkos::Timer timer;
    for (int r = 0; r < N_REPEAT; ++r) pass();
    Kokkos::fence();
    std::printf("grid %dx%dx%d: %.3f ms per forward pass (mean of %d)\n", nz, ny, nx,
                1e3 * timer.seconds() / N_REPEAT, N_REPEAT);

    bool ok = true;
    compare(to_host(fv), f_ref, "f", 1e-4, ok);
    compare(to_host(S), S_ref, "S", 1e-4, ok);
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

int main(int argc, char *argv[]) { return sn::main_wrapper(argc, argv, run); }
