//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file subgrid_divergence.cpp
//! \brief divergences of the subgrid fluxes R and Q

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "hydro/hydro.hpp"
#include "subgrid/subgrid.hpp"

namespace subgrid {
//----------------------------------------------------------------------------------------
//! \fn TaskStatus Subgrid::SubgridDivergence
//! \brief Computes, in the active cells, from the cached R and Q (PredictClosures, valid
//! in 1 ghost layer)
//!   (div R)_i = d_j R_ij,   div Q = d_j Q_j
//! by second-order central differences on the uniform grid,
//!   d_j F = (F(x + dx_j) - F(x - dx_j)) / (2 dx_j),
//! along each axis with more than one cell (in 2D/1D the missing axes contribute 0).
//! R is stored as (xx, yy, zz, xy, xz, yz) (SubgridRIndex), Q as (x, y, z).
//!
//! A term whose model is zero is 0 and is not differenced, so that with model_r and
//! model_q = zero the run is Hydro's bit for bit. The sources do not telescope: momentum
//! and energy are not conserved to round-off.

TaskStatus Subgrid::SubgridDivergence(Driver *pdrive, int stage) {
  const bool do_r = (model_r.type != SubgridModelType::zero);
  const bool do_q = (model_q.type != SubgridModelType::zero);
  if (!do_r) Kokkos::deep_copy(DevExeSpace(), divr, 0.0);
  if (!do_q) Kokkos::deep_copy(DevExeSpace(), divq, 0.0);
  if (!do_r && !do_q) return TaskStatus::complete;

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmb1 = pmy_pack->nmb_thispack - 1;
  const bool multi_d = pmy_pack->pmesh->multi_d;
  const bool three_d = pmy_pack->pmesh->three_d;
  auto &size = pmy_pack->pmb->mb_size;
  auto &rsg_ = rsg;
  auto &qsg_ = qsg;
  auto &divr_ = divr;
  auto &divq_ = divq;

  par_for("subgrid_div", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    // central difference of component c of the 5D array a along x1, x2, x3 (0 along an
    // axis with one cell)
    auto d1 = [&](const DvceArray5D<Real> &a, int c) {
      return (a(m,c,k,j,i+1) - a(m,c,k,j,i-1))/(2.0*size.d_view(m).dx1);
    };
    auto d2 = [&](const DvceArray5D<Real> &a, int c) {
      return multi_d ? (a(m,c,k,j+1,i) - a(m,c,k,j-1,i))/(2.0*size.d_view(m).dx2) : 0.0;
    };
    auto d3 = [&](const DvceArray5D<Real> &a, int c) {
      return three_d ? (a(m,c,k+1,j,i) - a(m,c,k-1,j,i))/(2.0*size.d_view(m).dx3) : 0.0;
    };
    if (do_r) {
      divr_(m,IVECX,k,j,i) = d1(rsg_,IRXX) + d2(rsg_,IRXY) + d3(rsg_,IRXZ);
      divr_(m,IVECY,k,j,i) = d1(rsg_,IRXY) + d2(rsg_,IRYY) + d3(rsg_,IRYZ);
      divr_(m,IVECZ,k,j,i) = d1(rsg_,IRXZ) + d2(rsg_,IRYZ) + d3(rsg_,IRZZ);
    }
    if (do_q) {
      divq_(m,k,j,i) = d1(qsg_,IVECX) + d2(qsg_,IVECY) + d3(qsg_,IVECZ);
    }
  });
  return TaskStatus::complete;
}

} // namespace subgrid
