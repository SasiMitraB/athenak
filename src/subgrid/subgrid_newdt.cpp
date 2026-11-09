//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file subgrid_newdt.cpp
//! \brief function to compute the subgrid timestep across all MeshBlock(s) in a
//! MeshBlockPack

#include <cfloat>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "hydro/hydro.hpp"
#include "eos/eos.hpp"
#include "units/units.hpp"
#include "srcterms/srcterms.hpp"
#include "srcterms/ismcooling.hpp"
#include "subgrid/subgrid.hpp"

namespace subgrid {
//----------------------------------------------------------------------------------------
//! \fn TaskStatus Subgrid::SubgridNewTimeStep
//! \brief Replaces Hydro's NewTimeStep task (last stage only). Computes the minimum of
//! the CFL timestep with c_d = sqrt((gamma P + 3 R_dd)/rho), t_d and t_E over active
//! cells, and stores it in dtnew, which Mesh::NewTimeStep uses in place of Hydro::dtnew.
//! Implemented: Hydro::NewTimeStep (which also updates the diffusion and source term
//! timesteps; identical to the CFL step with R = 0), and with model_c = ml the cooling
//! time of the total cooling, coarse plus C (SubgridSrcTerms),
//!   t_cool = e_int / |rho [rho Lambda_cut(T_CG) - Gamma] + C|,
//! in the form of SourceTerms::NewTimeStep (e_int from w0, both scaled by the CFL number
//! in Mesh::NewTimeStep). Hydro's own cooling time of the coarse cooling alone still
//! applies, so where C < 0 the step is no longer than before.
//! To do: the subgrid wave speed c_d and the t_d, t_E limits.

TaskStatus Subgrid::SubgridNewTimeStep(Driver *pdrive, int stage) {
  if (stage != (pdrive->nexp_stages)) {
    return TaskStatus::complete; // only execute last stage
  }
  TaskStatus tstat = pmy_pack->phydro->NewTimeStep(pdrive, stage);
  dtnew = pmy_pack->phydro->dtnew;
  if (model_c.type != SubgridModelType::ml) return tstat;

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is, nx1 = indcs.nx1;
  const int js = indcs.js, nx2 = indcs.nx2;
  const int ks = indcs.ks, nx3 = indcs.nx3;
  const int nmkji = (pmy_pack->nmb_thispack)*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji  = nx2*nx1;
  auto *phyd = pmy_pack->phydro;
  const Real gm1 = phyd->peos->eos_data.gamma - 1.0;
  const Real temp_unit = pmy_pack->punit->temperature_cgs();
  const Real n_unit = pmy_pack->punit->density_cgs()/pmy_pack->punit->mu()
                      /pmy_pack->punit->atomic_mass_unit_cgs;
  const Real cooling_unit = pmy_pack->punit->pressure_cgs()/pmy_pack->punit->time_cgs()
                            /n_unit/n_unit;
  const Real heating_unit = pmy_pack->punit->pressure_cgs()/pmy_pack->punit->time_cgs()
                            /n_unit;
  const Real gamma_heating = phyd->psrc->hrate/heating_unit;
  auto &w0 = phyd->w0;
  auto &csg_ = csg;
  Real dt_cool = dtnew;
  Kokkos::parallel_reduce("subgrid_cooling_newdt",
                          Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, Real &min_dt) {
    const int m = idx/nkji;
    const int k = (idx - m*nkji)/nji + ks;
    const int j = (idx - m*nkji - (k - ks)*nji)/nx1 + js;
    const int i = (idx - m*nkji - (k - ks)*nji - (j - js)*nx1) + is;
    const Real rho = w0(m,IDN,k,j,i), eint = w0(m,IEN,k,j,i);
    const Real temp = temp_unit*gm1*eint/rho;
    const Real coarse = rho*(rho*ISMCoolFn(temp)/cooling_unit - gamma_heating);
    const Real rate = FLT_MIN + fabs(coarse + csg_(m,k,j,i));
    min_dt = fmin(eint/rate, min_dt);
  }, Kokkos::Min<Real>(dt_cool));
  dtnew = fmin(dtnew, dt_cool);
  return tstat;
}

} // namespace subgrid
