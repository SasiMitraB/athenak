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
//! \brief Computes div R and div Q in active cells by finite differences of the cached
//! R and Q (which are valid in 1 ghost layer).
//! Placeholder: div R = div Q = 0.

TaskStatus Subgrid::SubgridDivergence(Driver *pdrive, int stage) {
  // TODO: finite differences of rsg, qsg
  Kokkos::deep_copy(DevExeSpace(), divr, 0.0);
  Kokkos::deep_copy(DevExeSpace(), divq, 0.0);
  return TaskStatus::complete;
}

} // namespace subgrid
