//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file subgrid.cpp
//! \brief implementation of Subgrid class constructor and destructor

#include <iostream>
#include <string>
#include <algorithm>
#include <limits>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "coordinates/coordinates.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "subgrid/subgrid.hpp"

namespace subgrid {
//----------------------------------------------------------------------------------------
// constructor, parses <subgrid>, checks the setup is supported, sets up the models and
// allocates the cache. Must be called after the Hydro constructor.

Subgrid::Subgrid(MeshBlockPack *ppack, ParameterInput *pin) :
    fsg("subgrid_f",1,1,1,1),
    ksg("subgrid_ksg",1,1,1,1),
    rsg("subgrid_R",1,1,1,1,1),
    aeval("subgrid_A_eval",1,1,1,1,1),
    aevec("subgrid_A_evec",1,1,1,1,1),
    qsg("subgrid_Q",1,1,1,1,1),
    csg("subgrid_C",1,1,1,1),
    divr("subgrid_divR",1,1,1,1,1),
    divq("subgrid_divQ",1,1,1,1),
    dtnew(std::numeric_limits<float>::max()),
    pmy_pack(ppack) {
  // (1) Check the setup is supported: single-fluid non-relativistic ideal gas Hydro on a
  // uniform grid, with enough ghost zones for the gradient stencil
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
  if (pin->DoesBlockExist("shearing_box")) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "<subgrid> cannot be used with <shearing_box>" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (pmy_pack->pmesh->multilevel) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "<subgrid> is only implemented on a uniform grid (no SMR/AMR)"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (pin->GetString("time","evolution").compare("dynamic") != 0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "<subgrid> requires <time>/evolution = dynamic" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // (2) Parse <subgrid>
  gate_kcut = pin->GetOrAddReal("subgrid","gate_kcut",0.0);

  std::string stencil_str = pin->GetOrAddString("subgrid","stencil","central2");
  int stencil_width;  // number of cells either side of a cell used by a gradient
  if (stencil_str.compare("central2") == 0) {
    stencil = SubgridStencil::central2;
    stencil_width = 1;
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "<subgrid>/stencil = '" << stencil_str << "' not implemented"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // R and Q are needed in 1 ghost layer for the divergences, so their gradient inputs
  // need (1 + stencil_width) ghost layers of u0
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  if (indcs.ng < 1 + stencil_width) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "<subgrid>/stencil = " << stencil_str << " requires at least "
              << 1 + stencil_width << " ghost zones, but <mesh>/nghost=" << indcs.ng
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // (3) Set up the models (type per term, weights read for ml models)
  SetupModel(pin, "ksg", model_ksg);
  SetupModel(pin, "r", model_r);
  SetupModel(pin, "q", model_q);
  SetupModel(pin, "c", model_c);

  // (4) Allocate the cache, sized as Hydro arrays (including ghost zones)
  int nmb = std::max((ppack->nmb_thispack), (ppack->pmesh->nmb_maxperrank));
  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
  int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
  Kokkos::realloc(fsg,   nmb,    ncells3, ncells2, ncells1);
  Kokkos::realloc(ksg,   nmb,    ncells3, ncells2, ncells1);
  Kokkos::realloc(rsg,   nmb, 6, ncells3, ncells2, ncells1);
  Kokkos::realloc(aeval, nmb, 3, ncells3, ncells2, ncells1);
  Kokkos::realloc(aevec, nmb, 9, ncells3, ncells2, ncells1);
  Kokkos::realloc(qsg,   nmb, 3, ncells3, ncells2, ncells1);
  Kokkos::realloc(csg,   nmb,    ncells3, ncells2, ncells1);
  Kokkos::realloc(divr,  nmb, 3, ncells3, ncells2, ncells1);
  Kokkos::realloc(divq,  nmb,    ncells3, ncells2, ncells1);
}

//----------------------------------------------------------------------------------------
// destructor. Cache and model views are freed when their reference counts drop to 0.

Subgrid::~Subgrid() {
}

} // namespace subgrid
