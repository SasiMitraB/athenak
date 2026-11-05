//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file subgrid_predict_q.cpp
//! \brief prediction of the subgrid energy flux Q

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "hydro/hydro.hpp"
#include "subgrid/subgrid.hpp"

namespace subgrid {
//----------------------------------------------------------------------------------------
//! \fn TaskStatus Subgrid::PredictQ
//! \brief Predicts w_hat from u0 (model_q), then w = B c w_hat,
//! R^{1/2} = sqrt(2 rho k_sg) e^{A/2} / sqrt(Tr e^A) from the stored eigen-decomposition
//! of A, q = sqrt(rho) psi R^{1/2} w and Q_i = v_j R_ij + q_i. Fills qsg in 1 ghost
//! layer. With model_q = neutral, w = 0.
//! Placeholder: Q = 0.

TaskStatus Subgrid::PredictQ(Driver *pdrive, int stage) {
  // TODO: evaluate model_q
  Kokkos::deep_copy(DevExeSpace(), qsg, 0.0);
  return TaskStatus::complete;
}

} // namespace subgrid
