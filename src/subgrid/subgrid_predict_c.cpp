//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file subgrid_predict_c.cpp
//! \brief prediction of the cooling correction C

#include <cstddef>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "bvals/bvals.hpp"
#include "hydro/hydro.hpp"
#include "eos/eos.hpp"
#include "units/units.hpp"
#include "srcterms/ismcooling.hpp"
#include "subgrid/subgrid.hpp"
#include "subgrid/subgrid_net.hpp"
#include "subgrid/subgrid_inputs.hpp"

namespace subgrid {

//----------------------------------------------------------------------------------------
//! \fn TaskStatus Subgrid::PredictC
//! \brief Predicts the gate g (model_gate) and z_C (model_c) from u0 and the f of
//! PredictKsg, then
//!   C = g (rho/mu m_u)^2 [Lambda_full(T_U) e^{z_C} - Lambda_cut(T_CG)]
//! (models/conserved/cooling_model.py, eq-cooling-model) in the active cells, with
//!   T_U  = (gamma - 1) (E - m^2/2 rho)/rho = (gamma - 1) k_max   (u0 alone)
//!   T_CG = T_U (1 - f)                     (the e_int of SubgridConToPrim, from the same
//!                                           u0 and f: the temperature of the coarse
//!                                           cooling that HydroSrcTerms applies from w0)
//! and the cooling curves of HydroSrcTerms (srcterms/ismcooling.hpp) in its units, so the
//! second term cancels Hydro's cooling exactly where g = 1. With model_c = zero or
//! neutral, g = 0 and C = 0.
//!
//! The gate network runs on the active cells and stencil/2 (of model_c) ghost layers, the
//! cells whose gate the cooling network's stencil reads; the cooling network on the
//! active cells. Both take their inputs as PredictKsg does (subgrid_inputs.hpp), the gate
//! and f at the same clamped source cell as the features, i.e. edge-copied at physical
//! boundaries as in training. The Subgrid constructor checks the ghost zones are enough
//! for every input read to be exact, so the active cells match the JAX model whatever the
//! decomposition.

TaskStatus Subgrid::PredictC(Driver *pdrive, int stage) {
  if (model_c.type != SubgridModelType::ml) {
    Kokkos::deep_copy(DevExeSpace(), csg, 0.0);
    return TaskStatus::complete;
  }

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int ng = indcs.ng;
  const int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmb = pmy_pack->nmb_thispack;

  SubgridNet &cnet = *model_c.net;
  SubgridNet &gnet = *model_gate.net;
  const int hc = cnet.model.stencil/2;
  const bool with_ksg = (cnet.model.n_in == sn::N_COOL_IN_KSG);

  // (1) gate logits on the active cells and hc ghost layers
  const OutputBox gbox{ng - hc, ng - hc, ng - hc, nx3 + 2*hc, nx2 + 2*hc, nx1 + 2*hc};
  FillFeatureInputs(pmy_pack, gnet, gbox);
  EvaluateNet(gnet);
  const sn::Vec gate = gnet.out;

  // (2) z_C on the active cells, from the features, the gate and the k_sg channels
  const OutputBox cbox{ks, js, is, nx3, nx2, nx1};
  FillCoolingInputs(pmy_pack, cnet, cbox, gate, gbox, fsg, with_ksg);
  EvaluateNet(cnet);
  const sn::Vec z = cnet.out;

  // (3) C, in the units of SourceTerms::ISMCooling: rho^2 Lambda/cooling_unit is the code
  // energy density per code time of n^2 Lambda
  const Real gm1 = pmy_pack->phydro->peos->eos_data.gamma - 1.0;
  const Real temp_unit = pmy_pack->punit->temperature_cgs();
  const Real n_unit = pmy_pack->punit->density_cgs()/pmy_pack->punit->mu()
                      /pmy_pack->punit->atomic_mass_unit_cgs;
  const Real cooling_unit = pmy_pack->punit->pressure_cgs()/pmy_pack->punit->time_cgs()
                            /n_unit/n_unit;
  auto &u0 = pmy_pack->phydro->u0;
  auto &fsg_ = fsg;
  auto &csg_ = csg;
  par_for("csg", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real c = 0.0;
    if (gate(gbox.Cell(m, k, j, i)) > 0) {
      const Real rho = u0(m,IDN,k,j,i);
      const Real kmax = (u0(m,IEN,k,j,i) - 0.5*(SQR(u0(m,IM1,k,j,i))
                         + SQR(u0(m,IM2,k,j,i)) + SQR(u0(m,IM3,k,j,i)))/rho)/rho;
      const Real temp_u = temp_unit*gm1*kmax;
      if (temp_u > 0.0) {  // no correction where the energy is at a floor
        const Real temp_cg = temp_u*(1.0 - fsg_(m,k,j,i));
        const Real ez = exp(static_cast<Real>(z(cbox.Cell(m, k, j, i))));
        c = rho*rho/cooling_unit*(ISMCoolFnFull(temp_u)*ez - ISMCoolFn(temp_cg));
      }
    }
    csg_(m,k,j,i) = c;
  });
  return TaskStatus::complete;
}

} // namespace subgrid
