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
#include "hydro/hydro.hpp"
#include "subgrid/subgrid.hpp"

namespace subgrid {
//----------------------------------------------------------------------------------------
//! \fn TaskStatus Subgrid::SubgridConToPrim
//! \brief Replaces Hydro's ConToPrim task: e_int = E - 1/2 m^2/rho - rho k_sg, with
//! Hydro's floors acting on this e_int, over the whole block (ghost zones included).
//! Placeholder: k_sg = 0, so calls Hydro's ConsToPrim, which is then identical.

TaskStatus Subgrid::SubgridConToPrim(Driver *pdrive, int stage) {
  // TODO: subtract rho k_sg from e_int before applying the floors
  return pmy_pack->phydro->ConToPrim(pdrive, stage);
}

} // namespace subgrid
