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
//!   d_t <m> + div( <m><m>/<rho> + R ) + grad<P> = 0
//!   d_t <E> + div( (<E>+<P>)<m>/<rho> + Q )    = -(<rho>/mu m_p)^2 Lambda(T_cg) - C
//!
//! R = subgrid momentum flux (symmetric, 6 components), Q = subgrid energy flux
//! (3 components), C = cooling correction (scalar). The subgrid EOS (SubgridIdealHydro)
//! and timestep (Subgrid::NewTimeStep) replace the Hydro ones when <subgrid> exists.
//! All other Hydro tasks are used unchanged.

#include <map>
#include <memory>
#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "tasklist/task_list.hpp"

// forward declarations
class Driver;

// indices of the independent components of the symmetric subgrid momentum flux R_ij
enum SubgridRIndex {IRXX=0, IRYY=1, IRZZ=2, IRXY=3, IRXZ=4, IRYZ=5};
// indices of the components of the subgrid energy flux Q_i
enum SubgridQIndex {IQX=0, IQY=1, IQZ=2};

//----------------------------------------------------------------------------------------
//! \struct SubgridTaskIDs
//  \brief container to hold TaskIDs of all subgrid tasks

struct SubgridTaskIDs {
  TaskID flux;
  TaskID srctrms;
  TaskID newdt;
};

namespace subgrid {

//----------------------------------------------------------------------------------------
//! \class Subgrid

class Subgrid {
 public:
  Subgrid(MeshBlockPack *ppack, ParameterInput *pin);
  ~Subgrid();

  // cell-centered closure terms, including ghost zones
  DvceArray5D<Real> rflx;   // subgrid momentum flux R, (nmb,6,n3,n2,n1)
  DvceArray5D<Real> qflx;   // subgrid energy flux Q, (nmb,3,n3,n2,n1)
  DvceArray4D<Real> ccool;  // cooling correction C, (nmb,n3,n2,n1)
  Real dtnew;

  // container to hold names of TaskIDs
  SubgridTaskIDs id;

  // functions...
  void IncludeSubgridTasks(std::map<std::string, std::shared_ptr<TaskList>> tl);
  // ...in "stagen_tl" list
  TaskStatus AddSubgridFluxes(Driver *d, int stage);
  TaskStatus SubgridSrcTerms(Driver *d, int stage);
  TaskStatus NewTimeStep(Driver *d, int stage);

  // fills rflx, qflx, ccool from the current primitives
  void ComputeSubgridTerms(const DvceArray5D<Real> &w0);

 private:
  MeshBlockPack* pmy_pack;  // ptr to MeshBlockPack containing this Subgrid
};

} // namespace subgrid
#endif // SUBGRID_SUBGRID_HPP_
