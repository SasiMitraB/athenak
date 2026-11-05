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
//! the end of the previous stage:
//!   u0(m) -= beta dt div R,  u0(E) -= beta dt (div Q + C)
//! (sign of C to be confirmed, see codebase_notes/subgrid_tasklist_plan.md).
//! Placeholder: adds nothing.

TaskStatus Subgrid::SubgridSrcTerms(Driver *pdrive, int stage) {
  // TODO: Real beta_dt = (pdrive->beta[stage-1])*(pmy_pack->pmesh->dt);
  return TaskStatus::complete;
}

} // namespace subgrid
