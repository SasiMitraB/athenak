//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file subgrid.cpp
//! \brief implementation of Subgrid class constructor, task inclusion, and the tasks
//! that add the subgrid fluxes (R, Q) and cooling correction (C) to Hydro.

#include <iostream>
#include <string>
#include <algorithm>
#include <limits>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "coordinates/coordinates.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "subgrid/subgrid.hpp"

namespace subgrid {
//----------------------------------------------------------------------------------------
// constructor, initializes data structures and parameters

Subgrid::Subgrid(MeshBlockPack *ppack, ParameterInput *pin) :
    rflx("subgrid_R",1,1,1,1,1),
    qflx("subgrid_Q",1,1,1,1,1),
    ccool("subgrid_C",1,1,1,1),
    dtnew(std::numeric_limits<float>::max()),
    pmy_pack(ppack) {
  // Subgrid terms are only implemented for single-fluid non-relativistic ideal hydro
  if (pmy_pack->phydro == nullptr || pin->DoesBlockExist("mhd") ||
      pin->DoesBlockExist("radiation") || pin->DoesBlockExist("adm") ||
      pin->DoesBlockExist("z4c")) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "<subgrid> requires <hydro> and cannot be used with <mhd>, "
              << "<radiation>, <adm> or <z4c>" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (pmy_pack->pcoord->is_special_relativistic ||
      pmy_pack->pcoord->is_general_relativistic ||
      !(pmy_pack->phydro->peos->eos_data.is_ideal)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "<subgrid> requires a non-relativistic ideal gas EOS" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (pin->GetString("time","evolution").compare("dynamic") != 0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "<subgrid> requires <time>/evolution = dynamic" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // allocate memory for closure terms, sized as Hydro arrays (including ghost zones)
  int nmb = std::max((ppack->nmb_thispack), (ppack->pmesh->nmb_maxperrank));
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
  int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
  Kokkos::realloc(rflx, nmb, 6, ncells3, ncells2, ncells1);
  Kokkos::realloc(qflx, nmb, 3, ncells3, ncells2, ncells1);
  Kokkos::realloc(ccool, nmb, ncells3, ncells2, ncells1);
}

//----------------------------------------------------------------------------------------
// destructor

Subgrid::~Subgrid() {
}

//----------------------------------------------------------------------------------------
//! \fn void Subgrid::IncludeSubgridTasks
//! \brief Inserts subgrid tasks into the "stagen" list already assembled by Hydro:
//!   flux -> [AddSubgridFluxes] -> sendf  (so SMR/AMR flux correction sees R and Q)
//!   srctrms -> [SubgridSrcTerms] -> sendu_oa
//!   newdt -> [NewTimeStep]
//! Called by MeshBlockPack::AddPhysics() after Hydro::AssembleHydroTasks().

void Subgrid::IncludeSubgridTasks(std::map<std::string, std::shared_ptr<TaskList>> tl) {
  auto &hid = pmy_pack->phydro->id;
  id.flux = tl["stagen"]->InsertTask(&Subgrid::AddSubgridFluxes, this, hid.flux,
                                     hid.sendf);
  id.srctrms = tl["stagen"]->InsertTask(&Subgrid::SubgridSrcTerms, this, hid.srctrms,
                                        hid.sendu_oa);
  id.newdt = tl["stagen"]->AddTask(&Subgrid::NewTimeStep, this, hid.newdt);

  if (id.flux == TaskID(0) || id.srctrms == TaskID(0)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Could not insert subgrid tasks into Hydro task list" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void Subgrid::ComputeSubgridTerms
//! \brief Placeholder closure: sets R, Q and C to zero in every cell (ghost zones
//! included), so that the run reproduces plain Hydro exactly. A model (e.g. an ML
//! closure evaluated from w0) replaces this function.

void Subgrid::ComputeSubgridTerms(const DvceArray5D<Real> &w0) {
  Kokkos::deep_copy(DevExeSpace(), rflx, 0.0);
  Kokkos::deep_copy(DevExeSpace(), qflx, 0.0);
  Kokkos::deep_copy(DevExeSpace(), ccool, 0.0);
  return;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus Subgrid::AddSubgridFluxes
//! \brief Computes closure terms and adds the face-averaged subgrid fluxes to the Hydro
//! fluxes: F_i(m_j) += R_ij, F_i(E) += Q_i. Runs after Hydro::Fluxes (including FOFC),
//! so FOFC neither sees nor limits these terms.

TaskStatus Subgrid::AddSubgridFluxes(Driver *pdrive, int stage) {
  auto &w0 = pmy_pack->phydro->w0;
  ComputeSubgridTerms(w0);

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  auto &r_ = rflx;
  auto &q_ = qflx;

  // fluxes in x1-direction
  auto flx1 = pmy_pack->phydro->uflx.x1f;
  par_for("sgs_flx1", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie+1,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    flx1(m,IM1,k,j,i) += 0.5*(r_(m,IRXX,k,j,i-1) + r_(m,IRXX,k,j,i));
    flx1(m,IM2,k,j,i) += 0.5*(r_(m,IRXY,k,j,i-1) + r_(m,IRXY,k,j,i));
    flx1(m,IM3,k,j,i) += 0.5*(r_(m,IRXZ,k,j,i-1) + r_(m,IRXZ,k,j,i));
    flx1(m,IEN,k,j,i) += 0.5*(q_(m,IQX,k,j,i-1) + q_(m,IQX,k,j,i));
  });
  if (pmy_pack->pmesh->one_d) {return TaskStatus::complete;}

  // fluxes in x2-direction
  auto flx2 = pmy_pack->phydro->uflx.x2f;
  par_for("sgs_flx2", DevExeSpace(), 0, nmb1, ks, ke, js, je+1, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    flx2(m,IM1,k,j,i) += 0.5*(r_(m,IRXY,k,j-1,i) + r_(m,IRXY,k,j,i));
    flx2(m,IM2,k,j,i) += 0.5*(r_(m,IRYY,k,j-1,i) + r_(m,IRYY,k,j,i));
    flx2(m,IM3,k,j,i) += 0.5*(r_(m,IRYZ,k,j-1,i) + r_(m,IRYZ,k,j,i));
    flx2(m,IEN,k,j,i) += 0.5*(q_(m,IQY,k,j-1,i) + q_(m,IQY,k,j,i));
  });
  if (pmy_pack->pmesh->two_d) {return TaskStatus::complete;}

  // fluxes in x3-direction
  auto flx3 = pmy_pack->phydro->uflx.x3f;
  par_for("sgs_flx3", DevExeSpace(), 0, nmb1, ks, ke+1, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    flx3(m,IM1,k,j,i) += 0.5*(r_(m,IRXZ,k-1,j,i) + r_(m,IRXZ,k,j,i));
    flx3(m,IM2,k,j,i) += 0.5*(r_(m,IRYZ,k-1,j,i) + r_(m,IRYZ,k,j,i));
    flx3(m,IM3,k,j,i) += 0.5*(r_(m,IRZZ,k-1,j,i) + r_(m,IRZZ,k,j,i));
    flx3(m,IEN,k,j,i) += 0.5*(q_(m,IQZ,k-1,j,i) + q_(m,IQZ,k,j,i));
  });
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus Subgrid::SubgridSrcTerms
//! \brief Applies the cooling correction, E -= beta*dt*C, using C computed in this
//! stage by AddSubgridFluxes. Runs after Hydro::HydroSrcTerms.

TaskStatus Subgrid::SubgridSrcTerms(Driver *pdrive, int stage) {
  Real beta_dt = (pdrive->beta[stage-1])*(pmy_pack->pmesh->dt);

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  auto &u0 = pmy_pack->phydro->u0;
  auto &c_ = ccool;

  par_for("sgs_cool", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    u0(m,IEN,k,j,i) -= beta_dt*c_(m,k,j,i);
  });
  return TaskStatus::complete;
}

} // namespace subgrid
