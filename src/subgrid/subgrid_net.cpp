//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file subgrid_net.cpp
//! \brief the network evaluations of the subgrid models: their inputs from u0 (boundaries
//! as in training, see subgrid_inputs.hpp) and the forward pass. The only file that
//! launches network kernels (see subgrid_inputs.hpp for why).

#include <cmath>
#include <cstddef>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "bvals/bvals.hpp"
#include "hydro/hydro.hpp"
#include "subgrid/subgrid_net.hpp"
#include "subgrid/subgrid_inputs.hpp"

namespace subgrid {

namespace {
// no input channels beyond the 20 features
struct NoExtraInputs {
  template <class Put>
  KOKKOS_INLINE_FUNCTION void operator()(int, int, int, int, const Put &) const {}
};

// the cooling model's channels after the features (FillCoolingInputs)
struct CoolingExtraInputs {
  sn::Vec gate;  // raw gate outputs, logit in channel 0, on the cells of box
  OutputBox box;
  DvceArray4D<Real> f;
  bool with_ksg;

  template <class Put>
  KOKKOS_INLINE_FUNCTION void operator()(int m, int k, int j, int i,
                                         const Put &put) const {
    put(sn::N_FEAT, gate(box.Cell(m, k, j, i)) > 0 ? sn::Real(1) : sn::Real(0));
    if (with_ksg) {
      // f = sigmoid(z0) in float: floored so that an underflow (z0 < ~-87) or a round-up
      // to 1 (z0 > ~17) stays finite, as the training's -softplus(-+z0) would be
      const Real fc = f(m, k, j, i), tiny = 1.0e-37;
      put(sn::N_COOL_IN, static_cast<sn::Real>(log10(fmax(fc, tiny))));
      put(sn::N_COOL_IN + 1, static_cast<sn::Real>(log10(fmax(1.0 - fc, tiny))));
    }
  }
};

// The kernel of FillNetInputs, one padded input cell (m, k, j, i) per call.
template <class Extra>
struct NetInputsKernel {
  int is, ie, js, je, ks, ke, n1, n2, n3;  // active cells and block size, with ghosts
  int o1, o2, o3, h, n_in, pz, py, px;
  size_t plane;
  sn::ConstVec x_mean, x_std;
  sn::Vec xp;
  DvceArray5D<Real> u0;
  decltype(DualArray2D<BoundaryFlag>().d_view) bcs;  // MeshBlock::mb_bcs on the device
  Extra extra;

  KOKKOS_INLINE_FUNCTION void operator()(const int m, const int k, const int j,
                                         const int i) const {
    // range of u0 read along each axis (x1, x2, x3): the active cells towards a physical
    // boundary, all cells (with ghost zones) towards a neighbour
    const int lo[3] = {IsPhysical(bcs(m, BoundaryFace::inner_x1)) ? is : 0,
                       IsPhysical(bcs(m, BoundaryFace::inner_x2)) ? js : 0,
                       IsPhysical(bcs(m, BoundaryFace::inner_x3)) ? ks : 0};
    const int hi[3] = {IsPhysical(bcs(m, BoundaryFace::outer_x1)) ? ie : n1-1,
                       IsPhysical(bcs(m, BoundaryFace::outer_x2)) ? je : n2-1,
                       IsPhysical(bcs(m, BoundaryFace::outer_x3)) ? ke : n3-1};
    const int src[3] = {Kokkos::clamp(o1 + i, lo[0], hi[0]),
                        Kokkos::clamp(o2 + j, lo[1], hi[1]),
                        Kokkos::clamp(o3 + k, lo[2], hi[2])};
    // u0 channel c at the source cell (s = 0) or its neighbour s = -1, +1 along axis a
    auto u = [&](int c, int a, int s) -> sn::Real {
      int idx[3] = {src[0], src[1], src[2]};
      idx[a] = Kokkos::clamp(idx[a] + s, lo[a], hi[a]);
      return static_cast<sn::Real>(u0(m, c, idx[2], idx[1], idx[0]));
    };
    const size_t p = ((static_cast<size_t>(m)*pz + (k + h))*py + (j + h))*px + (i + h);
    auto put = [&](int c, sn::Real v) {
      xp(sn::xp_index(p, c, plane, n_in)) = (v - x_mean(c))/x_std(c);
    };
    sn::cell_features(u, put);
    extra(m, src[2], src[1], src[0], put);
  }
};

//----------------------------------------------------------------------------------------
//! \fn void FillNetInputs
//! \brief Sizes net for the output cells box of every MeshBlock of the pack and fills its
//! padded input array (box plus stencil/2 cells on each side) with the standardised
//! inputs: channels 0..19 the features of u0 (sn::cell_features), then extra(m, k, j, i,
//! put) at the source block cell (k, j, i), which stores channel c >= 20 with put(c,
//! value). The caller then runs EvaluateNet(net).

template <class Extra>
void FillNetInputs(MeshBlockPack *pp, SubgridNet &net, const OutputBox &box,
                   const Extra &extra) {
  static_assert(IDN == 0 && IM1 == 1 && IM2 == 2 && IM3 == 3 && IEN == 4,
                "the subgrid inputs read u0 channels 0..4 as rho, m_x, m_y, m_z, E");
  auto &indcs = pp->pmesh->mb_indcs;
  const int nmb = pp->nmb_thispack;
  net.Resize(nmb, box.n3, box.n2, box.n1);
  const int h = net.model.stencil/2;

  NetInputsKernel<Extra> kern;
  kern.is = indcs.is; kern.ie = indcs.ie;
  kern.js = indcs.js; kern.je = indcs.je;
  kern.ks = indcs.ks; kern.ke = indcs.ke;
  kern.n1 = indcs.nx1 + 2*indcs.ng;
  kern.n2 = indcs.nx2 + 2*indcs.ng;
  kern.n3 = indcs.nx3 + 2*indcs.ng;
  kern.o1 = box.o1; kern.o2 = box.o2; kern.o3 = box.o3;
  kern.h = h;
  kern.n_in = net.model.n_in;
  kern.pz = net.work.pz; kern.py = net.work.py; kern.px = net.work.px;
  kern.plane = net.work.plane;
  kern.x_mean = net.model.x_mean;
  kern.x_std = net.model.x_std;
  kern.xp = net.work.xp;
  kern.u0 = pp->phydro->u0;
  kern.bcs = pp->pmb->mb_bcs.d_view;
  kern.extra = extra;
  par_for("subgrid_inputs", DevExeSpace(), 0, nmb-1, -h, box.n3-1+h, -h, box.n2-1+h,
          -h, box.n1-1+h, kern);
}

} // namespace

void FillFeatureInputs(MeshBlockPack *pp, SubgridNet &net, const OutputBox &box) {
  FillNetInputs(pp, net, box, NoExtraInputs());
}

void FillCoolingInputs(MeshBlockPack *pp, SubgridNet &net, const OutputBox &box,
                       const sn::Vec &gate, const OutputBox &gate_box,
                       const DvceArray4D<Real> &f, bool with_ksg) {
  FillNetInputs(pp, net, box, CoolingExtraInputs{gate, gate_box, f, with_ksg});
}

void EvaluateNet(SubgridNet &net) {
  sn::layers(net.model, net.work, net.out);
}

} // namespace subgrid
