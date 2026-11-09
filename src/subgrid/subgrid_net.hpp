#ifndef SUBGRID_SUBGRID_NET_HPP_
#define SUBGRID_SUBGRID_NET_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file subgrid_net.hpp
//! \brief a stencil network of the Kokkos forward pass (subgrid/forward_pass), with its
//! work arrays. Included only by the files that evaluate a model, so that the rest of
//! AthenaK does not see the forward-pass headers. The network computes in float
//! (sn::Real), whatever the precision of AthenaK's Real.

#include "subgrid/forward_pass/model_ops.hpp"

namespace subgrid {

struct SubgridNet {
  sn::Model model;  // weights, input standardisation (read once)
  sn::Work work;    // padded inputs and activations for one MeshBlockPack (made on use)
  sn::Vec out;      // raw network outputs, (n_out, nmb, ncells3, ncells2, ncells1)

  // (re)allocates work and out for nmb blocks of n3 x n2 x n1 output cells
  void Resize(int nmb, int n3, int n2, int n1) {
    if (work.nb == nmb && work.nz == n3 && work.ny == n2 && work.nx == n1) return;
    work = sn::make_work(model, n3, n2, n1, nmb);
    out = sn::Vec("subgrid_net_out", static_cast<size_t>(model.n_out)*work.ncell);
  }
};

} // namespace subgrid
#endif // SUBGRID_SUBGRID_NET_HPP_
