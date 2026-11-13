//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file subgrid_model.cpp
//! \brief set up of the four closure models: model type per term and, for ml models, the
//! pickle of the trained model, which subgrid_predict.py loads.

#include <iostream>
#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "subgrid/subgrid.hpp"

namespace subgrid {
//----------------------------------------------------------------------------------------
//! \fn void Subgrid::SetupModel
//! \brief Reads <subgrid>/model_<term> (zero / neutral / ml) and, for ml models,
//! <subgrid>/weights_<term>.

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
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "<subgrid>/model_" << term << " = '" << type_str << "' not implemented"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  return;
}

} // namespace subgrid
