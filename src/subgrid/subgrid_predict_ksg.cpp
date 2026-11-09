//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file subgrid_predict_ksg.cpp
//! \brief prediction of the subgrid kinetic energy k_sg

#include <cstddef>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "bvals/bvals.hpp"
#include "hydro/hydro.hpp"
#include "subgrid/subgrid.hpp"
#include "subgrid/subgrid_net.hpp"
#include "subgrid/subgrid_inputs.hpp"

namespace subgrid {

//----------------------------------------------------------------------------------------
//! \fn TaskStatus Subgrid::PredictKsg
//! \brief Predicts z0 from u0 (model_ksg), then f = sigmoid(z0) = k_sg/k_max and
//! k_sg = f k_max, with k_max = (E - m^2/2 rho)/rho from u0, in every cell of every
//! MeshBlock including all ng ghost layers. With model_ksg = zero, f = k_sg = 0.
//!
//! The network (subgrid/forward_pass, models/conserved/ksg_model.py) maps a stencil^3 box
//! of 20 inputs per cell (log10 rho, m, log10 E and central differences of them, see
//! sn::cell_features) to z0 of the centre cell, with the boundaries of training
//! (subgrid_inputs.hpp):
//!  - block / periodic face: the output is exact out to ng - reach ghost layers
//!    (reach = stencil/2 + 1); beyond, an approximation the Subgrid constructor keeps out
//!    of the layers that the reconstruction reads.
//!  - physical face: the active cells see exactly the training-time inputs. The ghost
//!    cells take f mirrored (reflect) or copied from the last active cell (others), and
//!    k_sg = f k_max with their own u0, so that 0 <= k_sg < k_max there too.
//! So in the active cells the result does not depend on the MeshBlock decomposition and
//! matches the JAX model on the whole grid.

TaskStatus Subgrid::PredictKsg(Driver *pdrive, int stage) {
  if (model_ksg.type != SubgridModelType::ml) {
    Kokkos::deep_copy(DevExeSpace(), fsg, 0.0);
    Kokkos::deep_copy(DevExeSpace(), ksg, 0.0);
    return TaskStatus::complete;
  }

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int n1 = indcs.nx1 + 2*indcs.ng;
  const int n2 = indcs.nx2 + 2*indcs.ng;
  const int n3 = indcs.nx3 + 2*indcs.ng;
  const int nmb = pmy_pack->nmb_thispack;

  SubgridNet &net = *model_ksg.net;
  auto &u0 = pmy_pack->phydro->u0;
  auto &bcs = pmy_pack->pmb->mb_bcs.d_view;

  // (1) standardised inputs of every cell of the padded blocks (stencil/2 cells beyond
  // the ghost zones), in the layout of the forward pass
  FillFeatureInputs(pmy_pack, net, OutputBox{0, 0, 0, n3, n2, n1});

  // (2) z0 in every cell of every block, ghost zones included
  EvaluateNet(net);
  const sn::Vec out = net.out;  // after FillFeatureInputs, which (re)sizes it

  // (3) f and k_sg, with f behind physical boundaries set by the boundary condition
  auto &fsg_ = fsg;
  auto &ksg_ = ksg;
  par_for("ksg_f", DevExeSpace(), 0, nmb-1, 0, n3-1, 0, n2-1, 0, n1-1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    const int ii = GhostSource(i, is, ie, bcs(m, BoundaryFace::inner_x1),
                               bcs(m, BoundaryFace::outer_x1));
    const int jj = GhostSource(j, js, je, bcs(m, BoundaryFace::inner_x2),
                               bcs(m, BoundaryFace::outer_x2));
    const int kk = GhostSource(k, ks, ke, bcs(m, BoundaryFace::inner_x3),
                               bcs(m, BoundaryFace::outer_x3));
    const size_t cell = ((static_cast<size_t>(m)*n3 + kk)*n2 + jj)*n1 + ii;
    const Real f = static_cast<Real>(sn::sigmoid(out(cell)));
    const Real rho = u0(m,IDN,k,j,i);
    const Real kmax = (u0(m,IEN,k,j,i) - 0.5*(SQR(u0(m,IM1,k,j,i)) + SQR(u0(m,IM2,k,j,i))
                       + SQR(u0(m,IM3,k,j,i)))/rho)/rho;
    fsg_(m,k,j,i) = f;
    ksg_(m,k,j,i) = f*fmax(kmax, 0.0);
  });
  return TaskStatus::complete;
}

} // namespace subgrid
