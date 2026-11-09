//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file subgrid_model.cpp
//! \brief set up of the four closure models: model type per term, and for ml models the
//! stencil network (weights and input standardisation) read from file to the device.

#include <exception>
#include <iostream>
#include <memory>
#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "globals.hpp"
#include "subgrid/subgrid.hpp"
#include "subgrid/subgrid_net.hpp"

namespace subgrid {
//----------------------------------------------------------------------------------------
//! \fn void Subgrid::SetupModel
//! \brief Reads <subgrid>/model_<term> (zero / neutral / ml) and, for ml models,
//! <subgrid>/weights_<term>, then reads the weights.

void Subgrid::SetupModel(ParameterInput *pin, const std::string &term,
                         SubgridModel &model) {
  std::string type_str = pin->GetOrAddString("subgrid","model_"+term,"zero");
  if (type_str.compare("zero") == 0) {
    model.type = SubgridModelType::zero;
  } else if (type_str.compare("neutral") == 0) {
    model.type = SubgridModelType::neutral;
  } else if (type_str.compare("ml") == 0) {
    model.type = SubgridModelType::ml;
    model.weights_file = pin->GetString("subgrid","weights_"+term);
    ReadModelWeights(model);
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "<subgrid>/model_" << term << " = '" << type_str << "' not implemented"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void Subgrid::ReadModelWeights
//! \brief Reads the stencil network of one model from model.weights_file (the weights.bin
//! format of subgrid/forward_pass/stencil_net.hpp, written by export_weights.py) to the
//! device. Every rank reads the file. The checks of the network's inputs and outputs
//! against the term it predicts are made by the caller.

void Subgrid::ReadModelWeights(SubgridModel &model) {
  auto net = std::make_shared<SubgridNet>();
  try {
    net->model = sn::load_model(model.weights_file);
  } catch (const std::exception &e) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "cannot read subgrid model '" << model.weights_file << "': " << e.what()
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  model.net = net;
  if (global_variable::my_rank == 0) {
    const sn::Model &m = net->model;
    std::cout << "subgrid model " << model.weights_file << ": stencil " << m.stencil
              << ", " << m.n_in << " inputs, " << m.n_out << " outputs, layers";
    for (int n = 0; n < m.n_nets; ++n) {
      for (int l = 0; l < m.net[n].n_layers; ++l) {
        std::cout << " " << m.net[n].fan_in[l] << "->" << m.net[n].fan_out[l];
      }
      if (n + 1 < m.n_nets) std::cout << " |";
    }
    std::cout << std::endl;
  }
  return;
}

} // namespace subgrid
