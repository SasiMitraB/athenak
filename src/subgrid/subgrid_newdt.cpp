//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file subgrid_newdt.cpp
//! \brief function to compute the subgrid timestep across all MeshBlock(s) in a
//! MeshBlockPack

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "hydro/hydro.hpp"
#include "subgrid/subgrid.hpp"

namespace subgrid {
//----------------------------------------------------------------------------------------
//! \fn TaskStatus Subgrid::SubgridNewTimeStep
//! \brief Replaces Hydro's NewTimeStep task (last stage only). Computes the minimum of
//! the CFL timestep with c_d = sqrt((gamma P + 3 R_dd)/rho), t_d and t_E over active
//! cells, and stores it in dtnew, which Mesh::NewTimeStep uses in place of Hydro::dtnew.
//! Placeholder: calls Hydro::NewTimeStep (which also updates the diffusion and source
//! term timesteps) and copies Hydro::dtnew, which is identical with R = 0.

TaskStatus Subgrid::SubgridNewTimeStep(Driver *pdrive, int stage) {
  if (stage != (pdrive->nexp_stages)) {
    return TaskStatus::complete; // only execute last stage
  }
  // TODO: subgrid wave speed c_d and the t_d, t_E limits
  TaskStatus tstat = pmy_pack->phydro->NewTimeStep(pdrive, stage);
  dtnew = pmy_pack->phydro->dtnew;
  return tstat;
}

} // namespace subgrid
