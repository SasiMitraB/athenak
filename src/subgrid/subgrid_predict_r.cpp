//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file subgrid_predict_r.cpp
//! \brief prediction of the subgrid momentum flux R

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "hydro/hydro.hpp"
#include "subgrid/subgrid.hpp"

namespace subgrid {
//----------------------------------------------------------------------------------------
//! \fn TaskStatus Subgrid::PredictR
//! \brief Predicts z1..z5 from u0 (model_r), builds the traceless symmetric A, then
//! S = e^A / Tr e^A and R = 2 rho k_sg S. Stores R and the eigen-decomposition of A
//! (used by PredictQ) in 1 ghost layer. With model_r = neutral, z1..z5 = 0 (S = I/3).
//! Placeholder: R = 0 and the eigen-decomposition of A = 0.

TaskStatus Subgrid::PredictR(Driver *pdrive, int stage) {
  // TODO: evaluate model_r
  Kokkos::deep_copy(DevExeSpace(), rsg, 0.0);
  Kokkos::deep_copy(DevExeSpace(), aeval, 0.0);
  Kokkos::deep_copy(DevExeSpace(), aevec, 0.0);
  return TaskStatus::complete;
}

} // namespace subgrid
