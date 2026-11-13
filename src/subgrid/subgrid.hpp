#ifndef SUBGRID_SUBGRID_HPP_
#define SUBGRID_SUBGRID_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file subgrid.hpp
//! \brief definitions for Subgrid class, which adds the closure terms of the
//! coarse-grained (filtered) Euler equations to the Hydro module:
//!
//!   d_t m + div( m m/rho + P I ) = -div R
//!   d_t E + div( (E + P) v )     = -div Q - C  + (Hydro source terms)
//!
//! with e_int = E - 1/2 m^2/rho - rho k_sg. The closures are predicted each stage from u0
//! (including ghost zones) by the Python models of subgrid_predict.py, run by the one
//! task PredictClosures through an embedded interpreter (subgrid_python.cpp):
//!   k_sg: z0 -> f -> k_sg
//!   R:    z1..z5 -> A -> S -> R = 2 rho k_sg S
//!   Q:    w -> q, Q = v R + q
//!   C:    g (gate network), z_C (from u0, g and f) -> C
//! then SubgridDivergence: div R, div Q
//! SubgridConToPrim and SubgridNewTimeStep replace the Hydro ConToPrim and NewTimeStep
//! tasks in the stage task list. Uniform Cartesian grid and non-relativistic ideal gas
//! Hydro only.

#include <map>
#include <memory>
#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "tasklist/task_list.hpp"

// forward declarations
class Driver;
class MeshBlockPack;

// indices of the independent components of the symmetric subgrid momentum flux R_ij
enum SubgridRIndex {IRXX=0, IRYY=1, IRZZ=2, IRXY=3, IRXZ=4, IRYZ=5};
// indices of the components of vectors in the subgrid cache (Q_i, div R_i)
enum SubgridVIndex {IVECX=0, IVECY=1, IVECZ=2};

// type of model used for each closure term
//   zero:    set the physical quantity to 0 directly (f = 0, g = 0)
//   neutral: use the default network outputs (z1..z5 = 0, w = 0, g = 0)
//   ml:      evaluate the network read from the weights file
enum class SubgridModelType {zero, neutral, ml};

// stencil used for the gradients that are the model inputs. Must match the stencil
// used to make the training data.
enum class SubgridStencil {central2};

namespace subgrid {
class PythonPredictor;  // subgrid_python.hpp
} // namespace subgrid

//----------------------------------------------------------------------------------------
//! \struct SubgridModel
//! \brief one closure model. For ml models, weights_file is the pickle of the trained JAX
//! model (<project>/models/conserved/<kind>_model.py), which subgrid_predict.py loads.

struct SubgridModel {
  SubgridModelType type = SubgridModelType::zero;
  std::string weights_file;  // ml models only
};

//----------------------------------------------------------------------------------------
//! \struct SubgridTaskIDs
//  \brief container to hold TaskIDs of all subgrid tasks

struct SubgridTaskIDs {
  TaskID srctrms;
  TaskID predict;
  TaskID div;
  TaskID c2p;
  TaskID newdt;
};

namespace subgrid {

//----------------------------------------------------------------------------------------
//! \class Subgrid

class Subgrid {
 public:
  Subgrid(MeshBlockPack *ppack, ParameterInput *pin);
  ~Subgrid();

  // models for the four closure terms
  SubgridModel model_ksg;  // z0
  SubgridModel model_r;    // z1..z5
  SubgridModel model_q;    // w
  SubgridModel model_c;    // z_C
  SubgridModel model_gate; // g, the gate input of model_c (<subgrid>/weights_gate,
                           // default the gate model model_c was trained with)
  std::unique_ptr<PythonPredictor> ppredict;  // runs the models (subgrid_predict.py)

  // parameters
  Real gate_kcut;          // cut wavenumber of the cooling gate
  SubgridStencil stencil;  // gradient stencil for the model inputs

  // cache of closure terms, all sized like Hydro u0 (nmb, ..., n3, n2, n1). Valid range:
  DvceArray4D<Real> fsg;   // f,    all ng ghost layers
  DvceArray4D<Real> ksg;   // k_sg, all ng ghost layers
  DvceArray5D<Real> rsg;   // R_ij (6 comps), 1 ghost layer
  DvceArray5D<Real> aeval; // eigenvalues of A (3 comps), 1 ghost layer
  DvceArray5D<Real> aevec; // eigenvectors of A (9 comps, row-major), 1 ghost layer
  DvceArray5D<Real> qsg;   // Q_i (3 comps), 1 ghost layer
  DvceArray4D<Real> csg;   // C, active cells
  DvceArray5D<Real> divr;  // div R (3 comps), active cells
  DvceArray4D<Real> divq;  // div Q, active cells
  Real dtnew;

  // container to hold names of TaskIDs
  SubgridTaskIDs id;

  // functions...
  void AssembleSubgridTasks(std::map<std::string, std::shared_ptr<TaskList>> tl);
  // ...in "stagen_tl" list
  TaskStatus SubgridSrcTerms(Driver *d, int stage);
  TaskStatus PredictClosures(Driver *d, int stage);
  TaskStatus SubgridDivergence(Driver *d, int stage);
  TaskStatus SubgridConToPrim(Driver *d, int stage);
  TaskStatus SubgridNewTimeStep(Driver *d, int stage);

  // runs PredictClosures, SubgridDivergence and SubgridConToPrim. Used in
  // Driver::Initialize so that stage 1 starts from a consistent cache and w0.
  void EvaluateClosures(Driver *d, int stage);

 private:
  MeshBlockPack* pmy_pack;  // ptr to MeshBlockPack containing this Subgrid

  // model setup (subgrid_model.cpp)
  void SetupModel(ParameterInput *pin, const std::string &term, SubgridModel &model);
};

} // namespace subgrid
#endif // SUBGRID_SUBGRID_HPP_
