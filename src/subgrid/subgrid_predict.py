"""The machine-learning closures of the AthenaK <subgrid> module, in Python.

AthenaK (subgrid_python.cpp) embeds a Python interpreter, imports this module and calls,
once per stage, Predictor.predict on host copies of the coarse conserved state u0 of the
MeshBlockPack. It fills the subgrid cache in place:

  k_sg  z0 (model_ksg) -> f = sigmoid(z0) = k_sg/k_max, k_sg = f k_max   (fsg, ksg)
  R     z1..z5 (model_r) -> R = 2 rho k_sg S                            (rsg, aeval, aevec)
  Q     w (model_q) -> Q = v R + q                                      (qsg)
  C     g (gate network), z_C (from u0, g and the k_sg channels)        (csg)
        C = g (rho/mu m_u)^2 [Lambda_full(T_U) e^{z_C} - Lambda_cut(T_CG)]

The models are the trained JAX models of <project>/models/conserved, run with their own
code: <subgrid>/weights_ksg, weights_c and weights_gate are the pickles their training
writes (ksg_model.pkl, cooling_model.pkl, gate_model.pkl), loaded with
cooling_model.load_conserved_model, and the network inputs and outputs are those of
features / network_input, apply, ksg_input and gate_input. weights_gate defaults to the
gate model the cooling model was trained with. R and Q are still placeholders (0).

What this module adds is AthenaK's view of the grid: a MeshBlockPack of blocks with ng
ghost zones instead of the whole padded frame of training. Training padded the coarse
grid with the run's boundaries (stencil.pad: periodic, or an edge copy at a physical
boundary). Per MeshBlock and face:
 - block / periodic face: the ghost zones of u0 hold the neighbour's cells and are used
   as they are. A network output is exact out to ng - reach ghost layers (reach =
   stencil/2 + 1); beyond, reads are clamped to the block.
 - physical face: the ghost zones are replaced by an edge copy of the active cells (for
   the gradients), and the network inputs beyond the face by an edge copy of those of
   the last active cell (for the stencil), as in training.
So in the active cells the result does not depend on the MeshBlock decomposition. The
Subgrid constructor checks the ghost zones are enough for every input read to be exact.

The networks run in float32 with JAX (on the GPU if JAX has one), the rest in the
precision of AthenaK's Real. C uses AthenaK's own cooling curve (srcterms/ismcooling.hpp,
transcribed below) so that its second term cancels Hydro's cooling exactly.
"""
import os
import sys
from pathlib import Path

import numpy as np

# <project> (athenak is its submodule): models/ is imported from here unless <subgrid>/
# models_path says otherwise. Relative model paths (as the cooling pickle's gate_model
# and ksg_model) are relative to it too.
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# BoundaryFlag codes sent by subgrid_python.cpp, per MeshBlock face
BC_OPEN = 0      # block or periodic: the ghost zones hold neighbouring cells
BC_REFLECT = 1   # physical, reflect
BC_OTHER = 2     # physical, any other condition
# face order (BoundaryFace): inner_x1, outer_x1, inner_x2, outer_x2, inner_x3, outer_x3


# ----------------------------------------------------------------------------- grid

def clamped_cells(ind, bcs, box, h):
    """Per axis (x3, x2, x1, the array order k, j, i), (nmb, n + 2h) block cells for the
    cells o - h .. o + n + h - 1 of an output box ((o3, n3), (o2, n2), (o1, n1)) plus a halo
    of h: the cell itself, clamped to the active cells towards a physical face (the edge
    copy of training) and to the block towards a neighbour."""
    out = []
    for a, (o, n) in zip((2, 1, 0), box):
        lo = np.where(bcs[:, 2*a] == BC_OPEN, 0, ind.s[a])[:, None]
        hi = np.where(bcs[:, 2*a + 1] == BC_OPEN, ind.nb[a] - 1, ind.e[a])[:, None]
        out.append(np.clip(o + np.arange(-h, n + h)[None, :], lo, hi).astype(np.int32))
    return out


def gather(x, cells):
    """x (nmb, C, n3, n2, n1) at the cells (k, j, i) of clamped_cells, (nmb, C, P3, P2, P1)."""
    jnp = _jax().numpy
    k, j, i = cells
    m = jnp.arange(x.shape[0])[:, None, None, None]
    g = x[m, :, k[:, :, None, None], j[:, None, :, None], i[:, None, None, :]]
    return jnp.moveaxis(g, -1, 1)  # advanced indices first: (nmb, P3, P2, P1, C)


def _jax():
    import jax
    return jax


def ghost_source(ind, bcs, a):
    """(nmb, nb) cell of each block cell along axis a (0 = x1) whose f fills it: itself,
    except behind a physical face, where f is mirrored (reflect) or copied from the last
    active cell (any other condition)."""
    s, e, nb = ind.s[a], ind.e[a], ind.nb[a]
    i = np.arange(nb)[None, :]
    inner, outer = bcs[:, 2*a][:, None], bcs[:, 2*a + 1][:, None]
    src = np.broadcast_to(i, (bcs.shape[0], nb)).copy()
    src = np.where((i < s) & (inner == BC_REFLECT), np.minimum(2*s - 1 - i, e), src)
    src = np.where((i < s) & (inner == BC_OTHER), s, src)
    src = np.where((i > e) & (outer == BC_REFLECT), np.maximum(2*e + 1 - i, s), src)
    src = np.where((i > e) & (outer == BC_OTHER), e, src)
    return src


class Indices:
    """Cell indices of a MeshBlock (as AthenaK's mb_indcs), per axis 0, 1, 2 = x1, x2, x3:
    active cells s..e, n active cells, nb cells with ghost zones."""

    def __init__(self, ng, nx):
        self.ng = ng
        self.n = tuple(nx)
        self.nb = tuple(n + 2*ng for n in nx)
        self.s = (ng,)*3
        self.e = tuple(ng + n - 1 for n in nx)


# ----------------------------------------------------------------------------- cooling

# SPEX cooling curve of srcterms/ismcooling.hpp (Schure et al. 2009, 4.12 < log T < 8.16),
# float as there
_LHD = np.array([
    -22.5977, -21.9689, -21.5972, -21.4615, -21.4789, -21.5497, -21.6211, -21.6595,
    -21.6426, -21.5688, -21.4771, -21.3755, -21.2693, -21.1644, -21.0658, -20.9778,
    -20.8986, -20.8281, -20.7700, -20.7223, -20.6888, -20.6739, -20.6815, -20.7051,
    -20.7229, -20.7208, -20.7058, -20.6896, -20.6797, -20.6749, -20.6709, -20.6748,
    -20.7089, -20.8031, -20.9647, -21.1482, -21.2932, -21.3767, -21.4129, -21.4291,
    -21.4538, -21.5055, -21.5740, -21.6300, -21.6615, -21.6766, -21.6886, -21.7073,
    -21.7304, -21.7491, -21.7607, -21.7701, -21.7877, -21.8243, -21.8875, -21.9738,
    -22.0671, -22.1537, -22.2265, -22.2821, -22.3213, -22.3462, -22.3587, -22.3622,
    -22.3590, -22.3512, -22.3420, -22.3342, -22.3312, -22.3346, -22.3445, -22.3595,
    -22.3780, -22.4007, -22.4289, -22.4625, -22.4995, -22.5353, -22.5659, -22.5895,
    -22.6059, -22.6161, -22.6208, -22.6213, -22.6184, -22.6126, -22.6045, -22.5945,
    -22.5831, -22.5707, -22.5573, -22.5434, -22.5287, -22.5140, -22.4992, -22.4844,
    -22.4695, -22.4543, -22.4392, -22.4237, -22.4087, -22.3928], np.float32)


def cool_fn_full(temp):
    """Lambda_full(T) [erg cm^3 / s], ISMCoolFnFull of srcterms/ismcooling.hpp, T > 0."""
    dt = temp.dtype
    logt = np.log10(temp)
    lhd = _LHD.astype(dt)
    ipps = np.clip((dt.type(25.0)*logt).astype(np.int64) - 103, 0, 100)
    x0 = dt.type(4.12) + dt.type(0.04)*ipps.astype(dt)
    dx = logt - x0
    spex = 10.0**((lhd[ipps + 1]*dx - lhd[ipps]*(dx - dt.type(0.04)))*dt.type(25.0))
    low = (2.0e-19*np.exp(-1.184e5/(temp + 1.0e3))
           + 2.8e-28*np.sqrt(temp)*np.exp(-92.0/temp))  # Koyama & Inutsuka (2002)
    high = 10.0**(0.45*logt - 26.065)                    # CGOLS fit
    return np.where(logt <= 4.2, low, np.where(logt > 8.15, high, spex)).astype(dt)


def cool_fn(temp):
    """Lambda_cut(T): Lambda_full cut to 1.05e4 < T <= 0.95e6 K, ISMCoolFn."""
    logt = np.log10(temp)
    inside = (logt > np.log10(1.05e4)) & (logt <= np.log10(0.95e6))
    return np.where(inside, cool_fn_full(np.where(inside, temp, 1.0e5)), 0.0)


# ----------------------------------------------------------------------------- models

class Model:
    """A trained conserved-variable model from its pickle (as models/conserved/*_model.py
    write them: params, x_mean, x_std, stencil, module, inputs = "conserved"), run with its
    own module's code.

    The module must provide features(X), the network inputs (B, C, ...) of coarse states
    X (B, 5, ...) (models/conserved/conserved.features), or network_input(X, *extras) when
    the model takes channels beyond u0 (the cooling model), and apply(params, x), the
    network on standardised inputs padded by stencil // 2 (B, C, nz + 2h, ...) ->
    (B, nz, ny, nx[, n_out]). kind ("ksg", "gate", "cooling", ...) checks that the module is
    models/.../<kind>_model.py; None skips the check."""

    def __init__(self, path, kind=None):
        jax = _jax()
        self.path, self.kind = str(path), kind
        if kind is not None:
            from models.conserved import cooling_model
            self.m, self.mod = cooling_model.load_conserved_model(path, kind)
        else:
            import importlib
            import pickle
            with open(path, "rb") as f:
                self.m = pickle.load(f)
            if self.m.get("inputs") != "conserved":
                raise ValueError(f"{path} is not a conserved-variable model")
            self.mod = importlib.import_module(self.m["module"])
        self.stencil = int(self.m["stencil"])
        self.h = self.stencil//2
        self.n_in = int(np.shape(self.m["x_mean"])[0])
        self.params = jax.tree_util.tree_map(jax.numpy.asarray, self.m["params"])
        self.x_mean = jax.numpy.asarray(self.m["x_mean"])
        self.x_std = jax.numpy.asarray(self.m["x_std"])
        self._run = jax.jit(self._forward)

    def _forward(self, params, x_mean, x_std, u, extras, cells):
        # the model's own inputs on the whole blocks (u edge-copied at physical faces),
        # then the box with its halo at the clamped cells, standardised, and its network
        x = self.mod.network_input(u, *extras) if extras else self.mod.features(u)
        return self.mod.apply(params, (gather(x, cells) - x_mean)/x_std)

    def __call__(self, u, cells, extras=()):
        """Raw output on the box of cells (clamped_cells with halo h); see Pack.run."""
        return self._run(self.params, self.x_mean, self.x_std, u, tuple(extras),
                         tuple(_jax().numpy.asarray(c) for c in cells))

    def describe(self):
        hidden = self.m.get("hidden")
        return (f"subgrid model {self.path}: {self.m['module']}, stencil {self.stencil}, "
                f"{self.n_in} inputs" + (f", hidden {list(hidden)}" if hidden else ""))


def _resolve(path, root):
    p = Path(path)
    return p if p.is_absolute() else Path(root)/p


# ----------------------------------------------------------------------------- the pack

class Pack:
    """What the predict_* functions see of one stage: the MeshBlockPack's u0 and the
    subgrid cache they fill. Arrays are numpy, in AthenaK's Real (pack.real, float64
    unless AthenaK is built in single precision), indexed (block m, [component,] k, j, i)
    = (m, [c,] x3, x2, x1) over the whole MeshBlock, ghost zones included: n3, n2, n1 =
    pack.cells, the active cells ng .. ng + nx - 1 along each axis.

    In (read only):
      U      (nmb, 5, n3, n2, n1)  rho, m_x, m_y, m_z, E of u0
      kmax   (nmb, n3, n2, n1)     (E - m^2/2 rho)/rho, the unresolved energy per mass
    Out (the cache, AthenaK's memory: write in place, e.g. pack.out["rsg"][...] = R;
    every array is 0 when the predict_* functions start):
      fsg, ksg (nmb, n3, n2, n1)   f = k_sg/k_max, k_sg       all ghost layers
      rsg      (nmb, 6, ...)       R_ij, xx yy zz xy xz yz    active cells + 1 ghost layer
      aeval    (nmb, 3, ...)       eigenvalues of A           active cells + 1 ghost layer
      aevec    (nmb, 9, ...)       eigenvectors of A          active cells + 1 ghost layer
      qsg      (nmb, 3, ...)       Q_i, x y z                 active cells + 1 ghost layer
      csg      (nmb, n3, n2, n1)   C                          active cells
    Also: nmb, ng, nx = (nx1, nx2, nx3) active cells per block, meta (the units of C).

    To run a network, use pack.run(model, pack.box(g)): it applies the boundaries of
    training (see the module docstring) and returns the outputs on the active cells and g
    ghost layers; write them with pack.out[name][pack.sl(box)] (or [(slice(None),
    slice(None)) + pack.sl(box)[1:]] for a component array, as predict_r shows)."""

    def __init__(self, buffers, meta, cache):
        self.meta = meta
        self.nmb, nvar = meta["nmb"], meta["nvar"]
        self.real = np.float64 if meta["real_size"] == 8 else np.float32
        self.ind = Indices(meta["ng"], (meta["nx1"], meta["nx2"], meta["nx3"]))
        self.ng, self.nx = self.ind.ng, self.ind.n
        self.cells = self.ind.nb[::-1]  # n3, n2, n1
        self.bcs = np.frombuffer(buffers["bcs"], dtype=np.int32).reshape(self.nmb, 6)
        self._cache = cache

        def view(name, ncomp=None):
            a = np.frombuffer(buffers[name], dtype=self.real)
            shape = (-1,) + ((ncomp,) if ncomp else ()) + self.cells
            return a.reshape(shape)[:self.nmb]

        self.U = view("u0", nvar)[:, :5]
        self.out = {k: view(k) for k in ("fsg", "ksg", "csg")}
        self.out.update({k: view(k, n) for k, n in
                         (("rsg", 6), ("aeval", 3), ("aevec", 9), ("qsg", 3))})
        rho = self.U[:, 0]
        self.kmax = (self.U[:, 4] - 0.5*(self.U[:, 1:4]**2).sum(1)/rho)/rho
        self._u = None

    def box(self, ghost):
        """The output box of the active cells and ghost layers around them, per axis
        (x3, x2, x1) the (first cell, number of cells)."""
        return tuple((self.ng - ghost, n + 2*ghost) for n in self.nx[::-1])

    def sl(self, box):
        """Index of the cells of box in a (nmb, n3, n2, n1) array."""
        return (slice(None),) + tuple(slice(o, o + n) for o, n in box)

    @property
    def u(self):
        """u0 (nmb, 5, n3, n2, n1) in float32 on the JAX device, its ghost zones behind
        physical faces an edge copy of the active cells (the training boundary)."""
        if self._u is None:
            whole = tuple((0, n) for n in self.cells)
            self._u = gather(_jax().numpy.asarray(self.U.astype(np.float32)),
                             self.clamped(whole, 0))
        return self._u

    def run(self, model, box, extras=()):
        """The raw outputs of model (a Model) on the cells of box, a JAX array (nmb, n3,
        n2, n1[, n_out]) of the box's size, as its module's apply returns them. The network
        reads its inputs on the box plus stencil // 2 cells, with the boundaries of
        training. extras: channels the model takes after the 20 features, each (nmb, C,
        n3, n2, n1) on the whole blocks (passed to its module's network_input)."""
        return model(self.u, self.clamped(box, model.h), extras)

    def clamped(self, box, h):
        key = ("cells", self.bcs.tobytes(), self.ind.nb, box, h)
        if key not in self._cache:
            self._cache[key] = clamped_cells(self.ind, self.bcs, box, h)
        return self._cache[key]

    def ghost_source(self, a):
        key = ("ghost", self.bcs.tobytes(), self.ind.nb, a)
        if key not in self._cache:
            self._cache[key] = ghost_source(self.ind, self.bcs, a)
        return self._cache[key]


# ----------------------------------------------------------------------------- closures
#
# One function per closure term, called each stage in this order by Predictor.predict.
# Each gets the Pack, the model type of its term ("zero", "neutral" or "ml", from
# <subgrid>/model_<term>) and, for ml, its Model (<subgrid>/weights_<term>), and fills its
# part of pack.out (all zero on entry, so "zero" needs nothing).

def predict_ksg(pack, kind, model):
    """k_sg, the subgrid kinetic energy per unit mass (<subgrid>/model_ksg = zero or ml).

    The network (models/conserved/ksg_model.py) gives z0 per cell, then
        f = sigmoid(z0) = k_sg / k_max,   k_sg = f max(k_max, 0).
    Fills pack.out["fsg"] and pack.out["ksg"] in every cell, ghost zones included
    (SubgridConToPrim needs k_sg wherever it converts u0 to w0). Behind a physical face f
    is mirrored (reflect) or copied from the last active cell (any other boundary), and
    k_sg = f k_max with the ghost cell's own u0.

    Returns z0 (nmb, n3, n2, n1), which predict_c turns into the cooling model's k_sg
    channels, or None with model_ksg = zero (f = k_sg = 0)."""
    if kind != "ml":
        return None
    jax = _jax()
    z0 = pack.run(model, tuple((0, n) for n in pack.cells))
    f = np.asarray(jax.nn.sigmoid(z0)).astype(pack.real)
    k, j, i = (pack.ghost_source(a) for a in (2, 1, 0))
    m = np.arange(pack.nmb)[:, None, None, None]
    f = f[m, k[:, :, None, None], j[:, None, :, None], i[:, None, None, :]]
    pack.out["fsg"][...] = f
    pack.out["ksg"][...] = f*np.maximum(pack.kmax, 0.0)
    return z0


def predict_r(pack, kind, model):
    """R, the subgrid momentum flux (<subgrid>/model_r; ml: <subgrid>/weights_r).
    NOT IMPLEMENTED: R stays 0 with model_r = zero, the other types raise.

    Fill, in the active cells and 1 ghost layer (pack.box(1)):
        pack.out["rsg"]    (nmb, 6, n3, n2, n1)  R_ij, components xx, yy, zz, xy, xz, yz
                                                 (IRXX..IRYZ of subgrid.hpp)
        pack.out["aeval"], pack.out["aevec"]     A's eigenvalues (3) and eigenvectors
                                                 (9); only predict_q reads them, so their
                                                 layout is up to predict_q
    The planned model (subgrid.hpp): the network gives z1..z5, then
        A = sum_k z_k E^(k)   traceless symmetric (BASIS of models/rvv_tensor_model.py)
        S = e^A / Tr e^A      (rvv_tensor_model.shape_tensor)
        R = 2 rho k_sg S      (k_sg = pack.out["ksg"], filled by predict_ksg)
    and neutral is z = 0, i.e. S = I/3, R = (2/3) rho k_sg I.

    How to add the model:
      - put its training code in models/conserved/ and pickle it like the others (params,
        x_mean, x_std, stencil, module, inputs = "conserved"); its module needs features
        (or network_input) and apply, see Model. Set <subgrid>/model_r = ml and
        weights_r = <its .pkl> (relative paths are from the project root).
      - here: box = pack.box(1); z = pack.run(model, box) gives its raw outputs (nmb, n3,
        n2, n1, n_out) on box; build R from them and write it, e.g.
            sl = pack.sl(box)
            pack.out["rsg"][(slice(None), slice(None)) + sl[1:]] = np.moveaxis(R, -1, 1)
        with R (nmb, n3, n2, n1, 6) in pack.real.
      - the Subgrid constructor checks <mesh>/nghost >= 1 + stencil/2 + 1 for model_r =
        ml (Predictor.info reports the stencil), so R in the ghost layer is exact.
      - nothing else to change: once model_r is not zero, AthenaK takes div R from rsg
        by central differences (SubgridDivergence) and subtracts it from the momentum
        (SubgridSrcTerms). The timestep does not yet see R (SubgridNewTimeStep: the
        wave speed sqrt((gamma P + 3 R_dd)/rho) is a to do), so a large R may need a
        smaller <time>/cfl_number."""
    if kind == "zero":
        return
    raise NotImplementedError(f"<subgrid>/model_r = {kind}: predict_r in "
                              f"{Path(__file__).name} is a placeholder")


def predict_q(pack, kind, model):
    """Q, the subgrid energy flux (<subgrid>/model_q; ml: <subgrid>/weights_q).
    NOT IMPLEMENTED: Q stays 0 with model_q = zero, the other types raise.

    Fill, in the active cells and 1 ghost layer (pack.box(1)):
        pack.out["qsg"]    (nmb, 3, n3, n2, n1)  Q_x, Q_y, Q_z
    The planned model (subgrid.hpp): the network gives w_hat, then
        w = B c w_hat
        R^{1/2} = sqrt(2 rho k_sg) e^{A/2} / sqrt(Tr e^A)   from A's eigen-decomposition
                                                            (pack.out["aeval"], ["aevec"],
                                                            written by predict_r)
        q = sqrt(rho) psi R^{1/2} w,   Q_i = v_j R_ij + q_i  (v = m/rho from pack.U, R =
                                                            pack.out["rsg"])
    and neutral is w = 0, i.e. Q = v . R.

    How to add the model: as for predict_r, with model_q = ml and weights_q; predict_q runs
    after predict_r, so R and A are available. Once model_q is not zero, AthenaK takes
    div Q from qsg by central differences (SubgridDivergence) and subtracts it from the
    energy (SubgridSrcTerms)."""
    if kind == "zero":
        return
    raise NotImplementedError(f"<subgrid>/model_q = {kind}: predict_q in "
                              f"{Path(__file__).name} is a placeholder")


def predict_c(pack, kind, model, gate_model, z0, ln10):
    """C, the cooling correction (<subgrid>/model_c; ml: weights_c, weights_gate).

        C = g (rho / mu m_u)^2 [Lambda_full(T_U) e^{z_C} - Lambda_cut(T_CG)]
    with
        g     the gate: 1 where the gate network's probability is > 0.5 (gate_input of
              models/conserved/cooling_model.py)
        z_C   the cooling network, on the 20 features, g and, if it was trained with
              --ksg-model, the k_sg channels log10 f, log10 (1 - f) from z0 (ksg_input)
        T_U   = (gamma - 1) k_max (in K), from u0 alone
        T_CG  = T_U (1 - f), the temperature of the coarse cooling that Hydro applies
    and AthenaK's cooling curves (cool_fn_full, cool_fn), so the second term cancels
    Hydro's cooling exactly where g = 1. Fills pack.out["csg"] in the active cells, in
    code units: rho^2 Lambda / cooling_unit (pack.meta) is the code energy density per
    code time of n^2 Lambda. With model_c = zero or neutral, g = 0 and C = 0.

    The gate is evaluated on the active cells and stencil/2 (of the cooling model) ghost
    layers, the cells the cooling network's stencil reads."""
    if kind != "ml":
        return
    jax = _jax()
    jnp = jax.numpy
    hc, real, meta = model.h, pack.real, pack.meta
    gbox = pack.box(hc)
    p = jax.nn.sigmoid(pack.run(gate_model, gbox))
    gate = jnp.zeros((pack.nmb, 1) + pack.cells, jnp.float32)
    gate = gate.at[(slice(None), 0) + pack.sl(gbox)[1:]].set((p > 0.5).astype(jnp.float32))
    extras = [gate]
    if model.m.get("ksg_model"):
        extras.append(jnp.stack([-jax.nn.softplus(-z0), -jax.nn.softplus(z0)], axis=1)/ln10)
    act = pack.sl(pack.box(0))
    z = np.asarray(pack.run(model, pack.box(0), extras))

    g = np.asarray(gate[:, 0])[act] > 0
    rho, kmax, f = pack.U[:, 0][act], pack.kmax[act], pack.out["fsg"][act]
    temp_u = real(meta["temp_unit"]*meta["gm1"])*kmax
    on = g & (temp_u > 0.0)  # no correction where the energy is at a floor
    with np.errstate(all="ignore"):
        t_u = np.where(on, temp_u, 1.0e5).astype(real)
        t_cg = (t_u*(1.0 - f)).astype(real)
        c = rho*rho/real(meta["cooling_unit"])*(cool_fn_full(t_u)*np.exp(z.astype(real))
                                                 - cool_fn(t_cg))
    pack.out["csg"][act] = np.where(on, c, 0.0)


# ----------------------------------------------------------------------------- predictor

MODEL_TYPES = ("zero", "neutral", "ml")


class Predictor:
    """The closure models of one MeshBlockPack (one per MPI rank), made by create().

    config (from <subgrid>): model_ksg, model_r, model_q, model_c in MODEL_TYPES and, for
    ml models, weights_<term> (the model pickles; weights_gate optional), models_path."""

    def __init__(self, config):
        self.type = {}
        for term in ("ksg", "r", "q", "c"):
            t = config.get("model_" + term, "zero")
            if t not in MODEL_TYPES:
                raise ValueError(f"<subgrid>/model_{term} = '{t}' not implemented")
            self.type[term] = t
        if self.type["ksg"] == "neutral":
            raise ValueError("<subgrid>/model_ksg = neutral is not defined, use zero or ml")

        self.models = {}
        self._cache = {}
        self.ln10 = None
        if "ml" not in self.type.values():
            return
        root = config.get("models_path") or str(PROJECT_ROOT)
        if root not in sys.path:
            sys.path.insert(0, root)
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")  # share the GPU
        jax = _jax()                                                      # with Kokkos
        jax.config.update("jax_default_matmul_precision", "highest")  # true float32
        from models.cooling_model import LN10
        self.ln10 = LN10

        def need(key, model):
            if not config.get(key):
                raise ValueError(f"<subgrid>/{key} is required with model_{model} = ml")
            return _resolve(config[key], root)

        if self.type["ksg"] == "ml":
            self.models["ksg"] = Model(need("weights_ksg", "ksg"), "ksg")
        for term in ("r", "q"):  # their modules are not fixed yet: no kind check
            if self.type[term] == "ml":
                self.models[term] = Model(need("weights_" + term, term))
        if self.type["c"] == "ml":
            c = self.models["c"] = Model(need("weights_c", "c"), "cooling")
            gate = config.get("weights_gate") or c.m.get("gate_model")
            if not gate:
                raise ValueError(f"{c.path} was trained with the true gate: give "
                                 "<subgrid>/weights_gate, a gate model")
            self.models["gate"] = Model(_resolve(gate, root), "gate")
            if c.m.get("ksg_model"):
                if "ksg" not in self.models:
                    raise ValueError(f"{c.path} takes the k_sg channels of a k_sg model, "
                                     "which need <subgrid>/model_ksg = ml")
                trained = _resolve(c.m["ksg_model"], root).resolve()
                if trained != Path(self.models["ksg"].path).resolve():
                    print(f"subgrid: warning: {c.path} was trained with the k_sg channels "
                          f"of {trained}, but is given those of <subgrid>/weights_ksg = "
                          f"{self.models['ksg'].path}", flush=True)

    def info(self):
        """What the Subgrid constructor checks the mesh against: stencil of each network
        (0 if not ml), whether the cooling model takes the k_sg channels, and a line per
        network for the log."""
        st = {"stencil_" + t: (self.models[t].stencil if t in self.models else 0)
              for t in ("ksg", "r", "q", "c", "gate")}
        return {**st, "c_with_ksg": int("c" in self.models
                                        and bool(self.models["c"].m.get("ksg_model"))),
                "description": "\n".join(m.describe() for m in self.models.values())}

    def predict(self, buffers, meta):
        """Fills the subgrid cache from u0 (called by subgrid_python.cpp each stage).
        buffers: host memory of AthenaK's Real,
          u0 (nmb, nvar, n3, n2, n1) in;  fsg, ksg, csg (nmb, n3, n2, n1), rsg (nmb, 6, ...),
          aeval (nmb, 3, ...), aevec (nmb, 9, ...), qsg (nmb, 3, ...) out
        with n3, n2, n1 the block sizes with ghost zones, as allocated (the first nmb
        blocks of each are used), and bcs, int32 (nmb, 6) BC_* codes of the faces.
        Written in place: no reference to a buffer may outlive the call. meta: nmb, nvar,
        ng, nx1, nx2, nx3, real_size (4 or 8), gm1, temp_unit, cooling_unit."""
        pack = Pack(buffers, meta, self._cache)
        for a in pack.out.values():
            a[...] = 0.0
        m, t = self.models, self.type
        z0 = predict_ksg(pack, t["ksg"], m.get("ksg"))
        predict_r(pack, t["r"], m.get("r"))
        predict_q(pack, t["q"], m.get("q"))
        predict_c(pack, t["c"], m.get("c"), m.get("gate"), z0, self.ln10)


def create(config):
    """Entry point for subgrid_python.cpp: config is a dict of the <subgrid> strings."""
    return Predictor(dict(config))
