//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file subgrid_c2p.cpp
//! \brief conversion of conserved to primitive variables with the subgrid kinetic
//! energy removed from the internal energy

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "eos/ideal_c2p_hyd.hpp"
#include "hydro/hydro.hpp"
#include "subgrid/subgrid.hpp"

namespace subgrid {
//----------------------------------------------------------------------------------------
//! \fn TaskStatus Subgrid::SubgridConToPrim
//! \brief Replaces Hydro's ConToPrim task: e_int = E - 1/2 m^2/rho - rho k_sg, with
//! Hydro's floors acting on this e_int, over the whole block (ghost zones included).
//!
//! Each cell is converted by Hydro's own SingleC2P_IdealHyd on the state
//! (rho, m, E - K), K = rho k_sg the subgrid kinetic energy density from the cache
//! (PredictClosures, from the same u0). Where a floor resets the energy, u0(IEN) is set
//! to the floored E - K plus K: the subgrid energy is kept and only the thermal part is
//! raised.
//! The cached k_sg is not updated after a floor, so there k_sg != f k_max of the new u0
//! until the next PredictClosures. With k_sg = 0 this is Hydro's ConToPrim bit for bit.

TaskStatus Subgrid::SubgridConToPrim(Driver *pdrive, int stage) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int ng = indcs.ng;
  const int n1 = indcs.nx1 + 2*ng;
  const int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
  const int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
  const int nmb = pmy_pack->nmb_thispack;
  const int nhyd  = pmy_pack->phydro->nhydro;
  const int nscal = pmy_pack->phydro->nscalars;
  auto &eos = pmy_pack->phydro->peos->eos_data;
  auto &cons = pmy_pack->phydro->u0;
  auto &prim = pmy_pack->phydro->w0;
  auto &ksg_ = ksg;

  const int nji  = n2*n1;
  const int nkji = n3*nji;
  const int nmkji = nmb*nkji;

  int nfloord_=0, nfloore_=0, nfloort_=0;
  Kokkos::parallel_reduce("subgrid_c2p",Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, int &sumd, int &sume, int &sumt) {
    const int m = idx/nkji;
    const int k = (idx - m*nkji)/nji;
    const int j = (idx - m*nkji - k*nji)/n1;
    const int i = idx - m*nkji - k*nji - j*n1;

    // subgrid kinetic energy density, from the density k_sg was predicted with
    const Real ek_sg = cons(m,IDN,k,j,i)*ksg_(m,k,j,i);

    HydCons1D u;
    u.d  = cons(m,IDN,k,j,i);
    u.mx = cons(m,IM1,k,j,i);
    u.my = cons(m,IM2,k,j,i);
    u.mz = cons(m,IM3,k,j,i);
    u.e  = cons(m,IEN,k,j,i) - ek_sg;

    HydPrim1D w;
    bool dfloor_used=false, efloor_used=false, tfloor_used=false;
    SingleC2P_IdealHyd(u, eos, w, dfloor_used, efloor_used, tfloor_used);

    // update counters, reset conserved if a floor was hit
    if (dfloor_used) {
      cons(m,IDN,k,j,i) = u.d;
      sumd++;
    }
    if (efloor_used) {
      cons(m,IEN,k,j,i) = u.e + ek_sg;
      sume++;
    }
    if (tfloor_used) {
      cons(m,IEN,k,j,i) = u.e + ek_sg;
      sumt++;
    }
    prim(m,IDN,k,j,i) = w.d;
    prim(m,IVX,k,j,i) = w.vx;
    prim(m,IVY,k,j,i) = w.vy;
    prim(m,IVZ,k,j,i) = w.vz;
    prim(m,IEN,k,j,i) = w.e;
    for (int n=nhyd; n<(nhyd+nscal); ++n) {
      if (cons(m,n,k,j,i) < 0.0) {
        cons(m,n,k,j,i) = 0.0;
      }
      prim(m,n,k,j,i) = cons(m,n,k,j,i)/u.d;
    }
  }, Kokkos::Sum<int>(nfloord_), Kokkos::Sum<int>(nfloore_), Kokkos::Sum<int>(nfloort_));

  pmy_pack->pmesh->ecounter.neos_dfloor += nfloord_;
  pmy_pack->pmesh->ecounter.neos_efloor += nfloore_;
  pmy_pack->pmesh->ecounter.neos_tfloor += nfloort_;
  return TaskStatus::complete;
}

} // namespace subgrid
