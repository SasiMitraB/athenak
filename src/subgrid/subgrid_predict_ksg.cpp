//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file subgrid_predict_ksg.cpp
//! \brief prediction of the subgrid kinetic energy k_sg

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "hydro/hydro.hpp"
#include "subgrid/subgrid.hpp"

namespace subgrid {
//----------------------------------------------------------------------------------------
//! \fn TaskStatus Subgrid::PredictKsg
//! \brief Predicts z0 from u0 (model_ksg), then f = sigmoid(z0) and k_sg = f k_max, with
//! k_max from u0. Fills fsg, ksg in all ng ghost layers (one-sided gradients in the
//! outermost layer). With model_ksg = zero, f = 0 directly.
//! Placeholder: f = k_sg = 0 for every model type.

TaskStatus Subgrid::PredictKsg(Driver *pdrive, int stage) {
  // TODO: evaluate model_ksg
  Kokkos::deep_copy(DevExeSpace(), fsg, 0.0);
  Kokkos::deep_copy(DevExeSpace(), ksg, 0.0);
  return TaskStatus::complete;
}

} // namespace subgrid
