//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file subgrid_model.cpp
//! \brief set up of the four closure models: model type per term, and for ml models the
//! weights, biases and input/output scaling read from file and copied to device.

#include <iostream>
#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "subgrid/subgrid.hpp"

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
//! \brief Reads layers, weights, biases and input/output scaling of one model from
//! model.weights_file into host views, then deep copies them to device.
//! Placeholder: no file is read yet, the model has no layers.

void Subgrid::ReadModelWeights(SubgridModel &model) {
  // TODO: rank 0 reads model.weights_file and MPI_Bcast's, (or every rank reads), fills
  // host mirrors, then Kokkos::deep_copy host -> device.
  model.nlayers = 0;
  model.ninputs = 0;
  model.noutputs = 0;
  return;
}

} // namespace subgrid
