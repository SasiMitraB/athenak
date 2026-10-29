//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file subgrid_newdt.cpp
//! \brief function to compute the subgrid hydro timestep across all MeshBlock(s) in a
//! MeshBlockPack. When <subgrid> exists, Mesh::NewTimeStep uses Subgrid::dtnew in place
//! of Hydro::dtnew. Currently a copy of the non-relativistic Hydro::NewTimeStep.

#include <math.h>

#include <limits>
#include <iostream>
#include <algorithm> // min

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "subgrid/subgrid.hpp"

namespace subgrid {

//----------------------------------------------------------------------------------------
// \!fn void Subgrid::NewTimeStep()
// \brief calculate the minimum timestep within a MeshBlockPack for subgrid hydro

TaskStatus Subgrid::NewTimeStep(Driver *pdrive, int stage) {
  if (stage != (pdrive->nexp_stages)) {
    return TaskStatus::complete; // only execute last stage
  }

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, nx1 = indcs.nx1;
  int js = indcs.js, nx2 = indcs.nx2;
  int ks = indcs.ks, nx3 = indcs.nx3;

  Real dt1 = std::numeric_limits<float>::max();
  Real dt2 = std::numeric_limits<float>::max();
  Real dt3 = std::numeric_limits<float>::max();

  // capture class variables for kernel
  auto &w0_ = pmy_pack->phydro->w0;
  auto &eos = pmy_pack->phydro->peos->eos_data;
  auto &mbsize = pmy_pack->pmb->mb_size;
  const int nmkji = (pmy_pack->nmb_thispack)*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji  = nx2*nx1;

  // find smallest dx/(v +/- Cs) in each direction
  Kokkos::parallel_reduce("SubgridNudt",Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, Real &min_dt1, Real &min_dt2, Real &min_dt3) {
    // compute m,k,j,i indices of thread and call function
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks;
    j += js;

    Real p = eos.IdealGasPressure(w0_(m,IEN,k,j,i));
    Real cs = eos.IdealHydroSoundSpeed(w0_(m,IDN,k,j,i), p);
    Real max_dv1 = fabs(w0_(m,IVX,k,j,i)) + cs;
    Real max_dv2 = fabs(w0_(m,IVY,k,j,i)) + cs;
    Real max_dv3 = fabs(w0_(m,IVZ,k,j,i)) + cs;

    min_dt1 = fmin((mbsize.d_view(m).dx1/max_dv1), min_dt1);
    min_dt2 = fmin((mbsize.d_view(m).dx2/max_dv2), min_dt2);
    min_dt3 = fmin((mbsize.d_view(m).dx3/max_dv3), min_dt3);
  }, Kokkos::Min<Real>(dt1), Kokkos::Min<Real>(dt2),Kokkos::Min<Real>(dt3));

  // compute minimum of dt1/dt2/dt3 for 1D/2D/3D problems
  dtnew = dt1;
  if (pmy_pack->pmesh->multi_d) { dtnew = std::min(dtnew, dt2); }
  if (pmy_pack->pmesh->three_d) { dtnew = std::min(dtnew, dt3); }

  return TaskStatus::complete;
}
} // namespace subgrid
