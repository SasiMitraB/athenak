//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file subgrid_tasks.cpp
//! \brief functions that control Subgrid tasks stored in tasklists in MeshBlockPack

#include <iostream>
#include <map>
#include <memory>
#include <string>

#include "athena.hpp"
#include "tasklist/task_list.hpp"
#include "mesh/mesh.hpp"
#include "hydro/hydro.hpp"
#include "subgrid/subgrid.hpp"

namespace subgrid {
//----------------------------------------------------------------------------------------
//! \fn void Subgrid::AssembleSubgridTasks
//! \brief Adds subgrid tasks to the "stagen" list already assembled by Hydro (which,
//! with <subgrid>, leaves out its ConToPrim and NewTimeStep tasks):
//!   srctrms -> [SubgridSrcTerms] -> sendu_oa
//!   bcs -> [PredictClosures -> SubgridDivergence -> SubgridConToPrim
//!           -> SubgridNewTimeStep]
//! Called by MeshBlockPack::AddPhysics() after Hydro::AssembleHydroTasks().

void Subgrid::AssembleSubgridTasks(std::map<std::string, std::shared_ptr<TaskList>> tl) {
  auto &hid = pmy_pack->phydro->id;
  id.srctrms = tl["stagen"]->InsertTask(&Subgrid::SubgridSrcTerms, this, hid.srctrms,
                                        hid.sendu_oa);
  if (id.srctrms == TaskID(0)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Could not insert subgrid tasks into Hydro task list" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  id.predict = tl["stagen"]->AddTask(&Subgrid::PredictClosures, this, hid.bcs);
  id.div   = tl["stagen"]->AddTask(&Subgrid::SubgridDivergence, this, id.predict);
  id.c2p   = tl["stagen"]->AddTask(&Subgrid::SubgridConToPrim, this, id.div);
  id.newdt = tl["stagen"]->AddTask(&Subgrid::SubgridNewTimeStep, this, id.c2p);
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void Subgrid::EvaluateClosures
//! \brief Runs PredictClosures, SubgridDivergence and SubgridConToPrim outside the task
//! list. Called in
//! Driver::Initialize after InitBoundaryValuesAndPrimitives.

void Subgrid::EvaluateClosures(Driver *pdrive, int stage) {
  // following functions return a TaskStatus, but it is ignored so cast to (void)
  (void) PredictClosures(pdrive, stage);
  (void) SubgridDivergence(pdrive, stage);
  (void) SubgridConToPrim(pdrive, stage);
  return;
}

} // namespace subgrid
