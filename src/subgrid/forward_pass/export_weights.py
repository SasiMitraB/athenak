"""Export the conserved-variable gate, cooling and k_sg models (models/conserved/) to flat
binaries for the Kokkos forward passes (gate_forward.cpp, cooling_forward.cpp, ksg_forward.cpp),
plus a reference input/output pair computed with the JAX model to check the C++ against.

The reference input is a cached validation frame (REF_FRAME) of the model's own cache, the
coarse conserved state u0 = (rho, m_x, m_y, m_z, E), so the features (logs, gradients) are in
their real range.

Run from the project root with the venv (the pickles hold jax-era arrays, and the reference
uses the real model code):
  venv/bin/python athenak/src/subgrid/forward_pass/export_weights.py                  # all three
  venv/bin/python athenak/src/subgrid/forward_pass/export_weights.py ksg [model.pkl]  # one, other pickle

Writes data/<model>/{weights,reference}.bin next to this file.  Format of weights.bin: see
stencil_net.hpp.  reference.bin (little-endian):
  int32 nz, ny, nx, then float32 arrays
  gate:    u0[5][nz][ny][nx], logit[nz][ny][nx], (aux logit of f_mix if the model has it)
  cooling: x[6][nz][ny][nx] (u0, then the gate the model was trained with: the gate model's
           prediction, or the true g of the targets), z[nz][ny][nx], and for a model trained
           with --ksg-model its k_sg inputs log10 f, log10 (1 - f) [2][nz][ny][nx]; the k_sg
           model it names is written to data/cooling/ksg_weights.bin
  ksg:     u0[5][nz][ny][nx], z0[nz][ny][nx], f[nz][ny][nx], k_sg[nz][ny][nx]
"""
import importlib
import pickle
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path("/home/sasi/Projects/SubgridCGMModel")
sys.path.insert(0, str(ROOT))
from models import stencil  # noqa: E402
from models.conserved import conserved  # noqa: E402

OUT = Path(__file__).resolve().parent / "data"
MAGIC = 0x43474D32
REF_FRAME = 64  # a validation frame (val_every 5) after t_min = 2.5
DEFAULT_PKL = {"gate": ROOT / "outputs/gate_conserved_20261109_220911/gate_model.pkl",
               # predicted gate + k_sg inputs (--gate-model, --ksg-model)
               "cooling": ROOT / "outputs/cooling_conserved_20261109_234502/cooling_model.pkl",
               "ksg": ROOT / "outputs/ksg_conserved_20261109_222634/ksg_model.pkl"}


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
    """The networks of pickled model m in the order the C++ expects (one for each model)."""
    if m.get("inputs") != "conserved":
        raise ValueError(f"{kind}: not a conserved-variable model; the C++ builds only those inputs")
    return [m["params"]]


def write_reference(path, *arrays):
    nz, ny, nx = arrays[0].shape[-3:]
    with open(path, "wb") as f:
        np.array([nz, ny, nx], np.int32).tofile(f)
        for a in arrays:
            np.ascontiguousarray(a, np.float32).tofile(f)


def load(pkl):
    """The pickled model, its module and the reference frame (X, Y) of its cache."""
    m = pickle.load(open(pkl, "rb"))
    mod = importlib.import_module(m["module"])
    with np.load(stencil.frame_path(stencil.cache_dir(m["cache"]), REF_FRAME)) as d:
        return m, mod, d["X"], d["Y"]


def network(m, x):
    """Raw network outputs (nz, ny, nx, n_out) of model m on its inputs x (C, nz, ny, nx)."""
    xn = stencil.pad(((x - m["x_mean"]) / m["x_std"])[None], m["stencil"] // 2)
    return np.asarray(stencil.apply(m["params"], xn)[0])


def export_gate(pkl):
    m, mod, u, _ = load(pkl)
    write_weights(OUT / "gate/weights.bin", m, model_nets("gate", m))
    out = network(m, mod.features(jnp.asarray(u)[None])[0])
    write_reference(OUT / "gate/reference.bin", u, *(out[..., k] for k in range(out.shape[-1])))
    print(f"gate: {pkl}, frame {REF_FRAME}, outputs {out.shape[-1]}, "
          f"gate = 1 in {(out[..., 0] > 0).mean():.1%}")


def export_cooling(pkl):
    m, mod, u, y = load(pkl)
    write_weights(OUT / "cooling/weights.bin", m, model_nets("cooling", m))
    U, Y = jnp.asarray(u)[None], jnp.asarray(y)[None]
    g = mod.gate_input(U, Y, m["k_cut"], m["gate_model"])  # (1, 1, nz, ny, nx)
    ksg, extra = None, []
    if m.get("ksg_model"):
        mk = pickle.load(open(m["ksg_model"], "rb"))
        write_weights(OUT / "cooling/ksg_weights.bin", mk, model_nets("ksg", mk))
        ksg = mod.ksg_input(U, m["ksg_model"])  # (1, 2, nz, ny, nx)
        extra = [np.asarray(ksg[0])]
    z = network(m, mod.network_input(U, g, ksg)[0])[..., 0]
    write_reference(OUT / "cooling/reference.bin", np.concatenate([u, np.asarray(g[0])]), z, *extra)
    print(f"cooling: {pkl}, frame {REF_FRAME}, gate {m['gate_model'] or 'true g'} = 1 in "
          f"{float(g.mean()):.1%}, k_sg inputs {m.get('ksg_model')}, "
          f"z in [{z.min():.3f}, {z.max():.3f}]")


def export_ksg(pkl):
    m, mod, u, _ = load(pkl)
    write_weights(OUT / "ksg/weights.bin", m, model_nets("ksg", m))
    z = network(m, mod.features(jnp.asarray(u)[None])[0])[..., 0]
    f = np.asarray(jax.nn.sigmoid(z))
    k = f * np.asarray(conserved.k_max(jnp.asarray(u)))
    write_reference(OUT / "ksg/reference.bin", u, z, f, k)
    print(f"ksg: {pkl}, frame {REF_FRAME}, f in [{f.min():.3e}, {f.max():.3e}]")


EXPORT = {"gate": export_gate, "cooling": export_cooling, "ksg": export_ksg}

if __name__ == "__main__":
    jax.config.update("jax_default_matmul_precision", "highest")  # no TF32 on the GPU: a true float32 reference
    names = [a for a in sys.argv[1:2] if a in EXPORT] or list(EXPORT)
    for name in names:
        (OUT / name).mkdir(parents=True, exist_ok=True)
        EXPORT[name](Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_PKL[name])
