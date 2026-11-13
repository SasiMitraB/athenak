//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file subgrid_srcterms.cpp
//! \brief source terms from the subgrid closures

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "hydro/hydro.hpp"
#include "subgrid/subgrid.hpp"

namespace subgrid {
//----------------------------------------------------------------------------------------
//! \fn TaskStatus Subgrid::SubgridSrcTerms
//! \brief Adds the subgrid source terms to u0 in active cells, from the cache filled at
//! the end of the previous stage (or by EvaluateClosures before stage 1):
//!   u0(m) -= beta dt div R,  u0(E) -= beta dt (div Q + C)
//! Runs after HydroSrcTerms, which has removed the coarse cooling
//! beta dt (rho/mu m_u)^2 Lambda_cut(T_CG) with T_CG from w0. C (PredictClosures) is the
//! predicted true cooling minus that coarse cooling, so the energy a cell loses is
//!   beta dt [(rho/mu m_u)^2 Lambda_cut(T_CG) + C] = beta dt g (rho/mu m_u)^2
//!                                                   Lambda_full(T_U) e^{z_C}
//! where the gate is 1 (exactly, unless an energy floor changed w0), and the coarse
//! cooling alone where it is 0. C may be either sign.
//! div R and div Q (SubgridDivergence) are the cell-centred divergences of the fluxes
//! R and Q of the momentum and energy equations. A term whose model is zero is left out,
//! so that with every model zero this is a no-op (Hydro bit for bit).

TaskStatus Subgrid::SubgridSrcTerms(Driver *pdrive, int stage) {
  const bool do_r = (model_r.type != SubgridModelType::zero);
  const bool do_q = (model_q.type != SubgridModelType::zero);
  const bool do_c = (model_c.type == SubgridModelType::ml);  // C = 0 otherwise
  if (!do_r && !do_q && !do_c) return TaskStatus::complete;

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmb1 = pmy_pack->nmb_thispack - 1;
  const Real beta_dt = (pdrive->beta[stage-1])*(pmy_pack->pmesh->dt);
  auto &u0 = pmy_pack->phydro->u0;
  auto &divr_ = divr;
  auto &divq_ = divq;
  auto &csg_ = csg;
  par_for("subgrid_srcterms", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    if (do_r) {
      u0(m,IM1,k,j,i) -= beta_dt*divr_(m,IVECX,k,j,i);
      u0(m,IM2,k,j,i) -= beta_dt*divr_(m,IVECY,k,j,i);
      u0(m,IM3,k,j,i) -= beta_dt*divr_(m,IVECZ,k,j,i);
    }
    if (do_q) u0(m,IEN,k,j,i) -= beta_dt*divq_(m,k,j,i);
    if (do_c) u0(m,IEN,k,j,i) -= beta_dt*csg_(m,k,j,i);
  });
  return TaskStatus::complete;
}

} // namespace subgrid
