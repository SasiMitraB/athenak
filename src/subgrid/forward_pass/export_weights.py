"""Export the gate, cooling and rvv-tensor models to flat binaries for the Kokkos forward
passes (gate_forward.cpp, cooling_forward.cpp, rvv_forward.cpp), plus a reference input/output
pair computed with the JAX model to check the C++ against.

Run from the project root with the venv (the pickles hold jax-era arrays, and the reference
uses the real model code):
  venv/bin/python models/forward_pass_c/export_weights.py                  # all three
  venv/bin/python models/forward_pass_c/export_weights.py rvv [model.pkl]  # one, other pickle

Writes models/forward_pass_c/data/<model>/{weights,reference}.bin.  Format of weights.bin:
see stencil_net.hpp.  reference.bin (little-endian):
  int32 nz, ny, nx, then float32 arrays
  gate:    x_raw[6][nz][ny][nx], logit[nz][ny][nx], (aux logit of f_mix if the model has it)
  cooling: x_raw[7][nz][ny][nx] (last = gate), z[nz][ny][nx]
  rvv:     x_raw[6][nz][ny][nx], f[nz][ny][nx], S[6][nz][ny][nx]  (xx yy zz xy xz yz)
"""
import pickle
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path("/home/sasi/Projects/SubgridCGMModel")
sys.path.insert(0, str(ROOT))
from models import gate_model, rvv_tensor_model, stencil  # noqa: E402

OUT = ROOT / "models/forward_pass_c/data"
NZ, NY, NX = 16, 64, 16
MAGIC = 0x43474D32
DEFAULT_PKL = {"gate": ROOT / "outputs/gate_20261023_152610/gate_model.pkl",
               "cooling": ROOT / "outputs/cooling_20261023_162055/cooling_model.pkl",
               "rvv": ROOT / "outputs/rvv_20261022_191644/rvv_tensor_model.pkl"}


def write_weights(path, m, nets):
    """nets: list of param lists [{"w", "b"}, ...] sharing the input standardisation of m."""
    x_mean = np.asarray(m["x_mean"], np.float32).reshape(-1)
    x_std = np.asarray(m["x_std"], np.float32).reshape(-1)
    with open(path, "wb") as f:
        np.array([MAGIC, m["stencil"], len(x_mean), len(nets)], np.int32).tofile(f)
        x_mean.tofile(f)
        x_std.tofile(f)
        for params in nets:
            sizes = [np.asarray(l["w"]).shape[-1] for l in params]
            np.array([len(params), *sizes], np.int32).tofile(f)
            for l in params:  # layer 0 w is (kz, ky, kx, n_in, out) = DHWIO, row-major
                np.ascontiguousarray(l["w"], np.float32).tofile(f)
                np.ascontiguousarray(l["b"], np.float32).tofile(f)
    return x_mean, x_std


def model_nets(kind, m):
    """The networks of pickled model m in the order the C++ expects; for rvv the Galilean
    projection of the velocity weights is folded into the stencil weights."""
    params = m["params"]
    if kind != "rvv":
        return [params]
    fold = lambda net: [{**net[0], "w": np.asarray(rvv_tensor_model.galilean(net[0]["w"]))}, *net[1:]]
    return [fold(params["f"]), fold(params["S"])] if isinstance(params, dict) else [fold(params)]


def write_reference(path, *arrays):
    with open(path, "wb") as f:
        np.array([NZ, NY, NX], np.int32).tofile(f)
        for a in arrays:
            np.ascontiguousarray(a, np.float32).tofile(f)


def random_inputs(x_mean, x_std, seed=0, smooth=False):
    """Raw inputs (n, nz, ny, nx) around the training statistics.  smooth: low-wavenumber
    waves plus a little noise, so derived gradient features stay in a realistic range."""
    rng = np.random.default_rng(seed)
    n = len(x_mean)
    noise = rng.standard_normal((n, NZ, NY, NX))
    if smooth:
        z, y, x = np.meshgrid(*(np.arange(k) / k for k in (NZ, NY, NX)), indexing="ij")
        waves = np.stack([np.sin(2 * np.pi * (rng.integers(1, 3) * z + rng.integers(0, 3) * x
                                              + rng.integers(1, 4) * y) + rng.random() * 6)
                          for _ in range(n)])
        noise = waves + 0.05 * noise
    return (x_mean[:, None, None, None] + x_std[:, None, None, None] * noise).astype(np.float32)


def normalise(x, x_mean, x_std, h):
    xn = (x - x_mean[:, None, None, None]) / x_std[:, None, None, None]
    return stencil.pad(jnp.asarray(xn)[None], h)


def export_gate(pkl):
    m = pickle.load(open(pkl, "rb"))
    mean, std = write_weights(OUT / "gate/weights.bin", m, [m["params"]])
    base = random_inputs(mean[:6], std[:6], smooth=True)
    feat = gate_model.features(jnp.asarray(base)[None])[0]
    xn = stencil.pad(((feat - m["x_mean"][:, 0, 0, 0][:, None, None, None])
                      / m["x_std"][:, 0, 0, 0][:, None, None, None])[None], m["stencil"] // 2)
    out = np.asarray(stencil.apply(m["params"], xn)[0])
    write_reference(OUT / "gate/reference.bin", base, *(out[..., k] for k in range(out.shape[-1])))
    print(f"gate: {pkl}, outputs {out.shape[-1]}, gate = 1 in {(out[..., 0] > 0).mean():.1%}")


def export_cooling(pkl):
    m = pickle.load(open(pkl, "rb"))
    mean, std = write_weights(OUT / "cooling/weights.bin", m, [m["params"]])
    x = random_inputs(mean, std)
    x[-1] = (np.random.default_rng(1).random((NZ, NY, NX)) < 0.3)  # gate channel
    z = np.asarray(stencil.apply(m["params"], normalise(x, mean, std, m["stencil"] // 2))[0, ..., 0])
    write_reference(OUT / "cooling/reference.bin", x, z)
    print(f"cooling: {pkl}, z in [{z.min():.3f}, {z.max():.3f}]")


def export_rvv(pkl):
    m = pickle.load(open(pkl, "rb"))
    params = m["params"]
    mean, std = write_weights(OUT / "rvv/weights.bin", m, model_nets("rvv", m))
    x = random_inputs(mean, std)
    f, S = rvv_tensor_model.predict(params, normalise(x, mean, std, m["stencil"] // 2))
    f, S = np.asarray(f[0]), np.asarray(S[0])
    S6 = np.stack([S[..., i, j] for i, j in gate_model.PAIRS])
    write_reference(OUT / "rvv/reference.bin", x, f, S6)
    print(f"rvv: {pkl}, f in [{f.min():.3e}, {f.max():.3e}]")


EXPORT = {"gate": export_gate, "cooling": export_cooling, "rvv": export_rvv}

if __name__ == "__main__":
    jax.config.update("jax_default_matmul_precision", "highest")  # no TF32 on the GPU: a true float32 reference
    names = [a for a in sys.argv[1:2] if a in EXPORT] or list(EXPORT)
    for name in names:
        (OUT / name).mkdir(parents=True, exist_ok=True)
        EXPORT[name](Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_PKL[name])
