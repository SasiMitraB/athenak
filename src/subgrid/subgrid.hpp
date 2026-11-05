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
//! with e_int = E - 1/2 m^2/rho - rho k_sg. The closures are predicted each stage, in
//! this order, from u0 (including ghost zones):
//!   PredictKsg: z0 -> f -> k_sg
//!   PredictR:   z1..z5 -> A -> S -> R = 2 rho k_sg S
//!   PredictQ:   w -> q, Q = v R + q
//!   PredictC:   g, z_C -> C
//!   SubgridDivergence: div R, div Q
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

//----------------------------------------------------------------------------------------
//! \struct SubgridModel
//! \brief one fully connected network, held in device views that are captured by the
//! Predict kernels. Layer l maps a vector of length nodes(l) to one of length
//! nodes(l+1); its weights and biases start at the offsets woff(l) and boff(l).

struct SubgridModel {
  SubgridModelType type = SubgridModelType::zero;
  std::string weights_file;
  int nlayers = 0;
  int ninputs = 0;
  int noutputs = 0;
  DvceArray1D<int> nodes;         // number of nodes in each layer, (nlayers+1)
  DvceArray1D<int> woff;          // offset of each layer's weights, (nlayers)
  DvceArray1D<int> boff;          // offset of each layer's biases, (nlayers)
  DvceArray1D<Real> weights;      // all weights, row-major per layer
  DvceArray1D<Real> biases;       // all biases
  DvceArray1D<Real> in_shift;     // input scaling: x_hat = (x - in_shift)/in_scale
  DvceArray1D<Real> in_scale;
  DvceArray1D<Real> out_shift;    // output scaling: y = y_hat*out_scale + out_shift
  DvceArray1D<Real> out_scale;
};

//----------------------------------------------------------------------------------------
//! \struct SubgridTaskIDs
//  \brief container to hold TaskIDs of all subgrid tasks

struct SubgridTaskIDs {
  TaskID srctrms;
  TaskID ksg;
  TaskID rsg;
  TaskID qsg;
  TaskID csg;
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
  SubgridModel model_c;    // g, z_C

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
  TaskStatus PredictKsg(Driver *d, int stage);
  TaskStatus PredictR(Driver *d, int stage);
  TaskStatus PredictQ(Driver *d, int stage);
  TaskStatus PredictC(Driver *d, int stage);
  TaskStatus SubgridDivergence(Driver *d, int stage);
  TaskStatus SubgridConToPrim(Driver *d, int stage);
  TaskStatus SubgridNewTimeStep(Driver *d, int stage);

  // runs the Predict chain and SubgridConToPrim. Used in Driver::Initialize so that
  // stage 1 starts from a consistent cache and w0.
  void EvaluateClosures(Driver *d, int stage);

 private:
  MeshBlockPack* pmy_pack;  // ptr to MeshBlockPack containing this Subgrid

  // model setup (subgrid_model.cpp)
  void SetupModel(ParameterInput *pin, const std::string &term, SubgridModel &model);
  void ReadModelWeights(SubgridModel &model);
};

} // namespace subgrid
#endif // SUBGRID_SUBGRID_HPP_
