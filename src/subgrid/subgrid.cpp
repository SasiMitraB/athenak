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
#include "srcterms/srcterms.hpp"
#include "subgrid/subgrid.hpp"
#include "subgrid/subgrid_net.hpp"

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

  // (4) Check the k_sg model. k_sg has no neutral output (f = 0 is set directly by zero).
  if (model_ksg.type == SubgridModelType::neutral) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "<subgrid>/model_ksg = neutral is not defined, use zero or ml"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (model_ksg.type == SubgridModelType::ml) {
    const sn::Model &m = model_ksg.net->model;
    if (m.n_in != sn::N_FEAT || m.n_out != 1) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<subgrid>/weights_ksg must have " << sn::N_FEAT
                << " inputs and 1 output, but has " << m.n_in << " and " << m.n_out
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (!(pmy_pack->pmesh->three_d)) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<subgrid>/model_ksg = ml needs a 3D mesh (the network "
                << "is a 3D stencil)" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    // f in a ghost cell at distance g from the active cells reads u0 out to g + reach
    // (the stencil of the network plus that of its gradient inputs). f must be exact,
    // i.e. independent of the MeshBlock decomposition, in the ghost layers whose
    // primitives the reconstruction reads, and only ng layers of u0 are exchanged.
    auto *phyd = pmy_pack->phydro;
    int nrecon = (phyd->recon_method == ReconstructionMethod::dc) ? 1 :
                 (phyd->recon_method == ReconstructionMethod::plm) ? 2 : 3;
    if (phyd->use_fofc) nrecon += 1;
    int reach = m.stencil/2 + stencil_width;
    if (indcs.ng < nrecon + reach) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<subgrid>/model_ksg = ml (stencil " << m.stencil
                << ", gradient stencil " << stencil_str << ") with this reconstruction "
                << "requires at least " << nrecon + reach << " ghost zones (" << nrecon
                << " read by the reconstruction + " << reach << " for the model "
                << "inputs), but <mesh>/nghost=" << indcs.ng << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

  // (5) The cooling model: z_C from the 20 features, the gate of a gate network and, if
  // it was trained with --ksg-model (23 inputs), log10 f and log10 (1 - f) of
  // PredictKsg's f. Its second term cancels Hydro's ISM cooling, in physical units.
  if (model_c.type == SubgridModelType::ml) {
    auto fatal = [](const std::string &msg) {
      std::cout << "### FATAL ERROR in " << __FILE__ << std::endl << msg << std::endl;
      std::exit(EXIT_FAILURE);
    };
    model_gate.type = SubgridModelType::ml;
    model_gate.weights_file = pin->GetString("subgrid","weights_gate");
    ReadModelWeights(model_gate);
    const sn::Model &mc = model_c.net->model, &mg = model_gate.net->model;
    if ((mc.n_in != sn::N_COOL_IN && mc.n_in != sn::N_COOL_IN_KSG) || mc.n_out != 1) {
      fatal("<subgrid>/weights_c must have " + std::to_string(sn::N_COOL_IN) + " or " +
            std::to_string(sn::N_COOL_IN_KSG) + " inputs and 1 output, but has " +
            std::to_string(mc.n_in) + " and " + std::to_string(mc.n_out));
    }
    if (mg.n_in != sn::N_FEAT || mg.n_out < 1) {
      fatal("<subgrid>/weights_gate must have " + std::to_string(sn::N_FEAT) +
            " inputs and its logit as output 0, but has " + std::to_string(mg.n_in) +
            " inputs and " + std::to_string(mg.n_out) + " outputs");
    }
    const bool with_ksg = (mc.n_in == sn::N_COOL_IN_KSG);
    if (with_ksg && model_ksg.type != SubgridModelType::ml) {
      fatal("<subgrid>/weights_c takes the k_sg inputs log10 f, log10 (1 - f), which "
            "need <subgrid>/model_ksg = ml");
    }
    if (!(pmy_pack->pmesh->three_d)) {
      fatal("<subgrid>/model_c = ml needs a 3D mesh (the networks are 3D stencils)");
    }
    if (pmy_pack->punit == nullptr) {
      fatal("<subgrid>/model_c = ml needs <units>: C is computed from the cooling "
            "curve in cgs");
    }
    if (pmy_pack->phydro->psrc == nullptr || !(pmy_pack->phydro->psrc->ism_cooling)) {
      fatal("<subgrid>/model_c = ml needs <hydro_srcterms>/ism_cooling = true: C "
            "corrects the coarse cooling that Hydro applies");
    }
    // the cooling network reads the gate, and f, out to stencil/2 ghost layers, where
    // they are exact only if the inputs of their own networks are (reach = stencil/2 +
    // stencil_width ghost layers of u0 beyond)
    const int hc = mc.stencil/2;
    int need = hc + mg.stencil/2 + stencil_width;
    if (with_ksg) {
      need = std::max(need, hc + model_ksg.net->model.stencil/2 + stencil_width);
    }
    if (indcs.ng < need) {
      fatal("<subgrid>/model_c = ml (stencil " + std::to_string(mc.stencil) +
            ") requires at least " + std::to_string(need) + " ghost zones for exact "
            "gate and k_sg inputs, but <mesh>/nghost=" + std::to_string(indcs.ng));
    }
  }

  // (6) Allocate the cache, sized as Hydro arrays (including ghost zones)
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
