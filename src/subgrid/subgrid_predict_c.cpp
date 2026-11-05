//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file subgrid_predict_c.cpp
//! \brief prediction of the cooling correction C

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "hydro/hydro.hpp"
#include "subgrid/subgrid.hpp"

namespace subgrid {
//----------------------------------------------------------------------------------------
//! \fn TaskStatus Subgrid::PredictC
//! \brief Predicts the gate g and z_C from u0 (model_c), then
//! C = g (rho/mu m_p)^2 [Lambda_full(T_cg) e^{z_C} - Lambda_cut(T_cg)], with T_cg from f
//! and the cooling curves shared with HydroSrcTerms. Fills csg in active cells. With
//! model_c = zero or neutral, g = 0.
//! Placeholder: C = 0.

TaskStatus Subgrid::PredictC(Driver *pdrive, int stage) {
  // TODO: evaluate model_c
  Kokkos::deep_copy(DevExeSpace(), csg, 0.0);
  return TaskStatus::complete;
}

} // namespace subgrid
