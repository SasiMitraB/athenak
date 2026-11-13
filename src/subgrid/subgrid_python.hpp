#ifndef SUBGRID_SUBGRID_PYTHON_HPP_
#define SUBGRID_SUBGRID_PYTHON_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file subgrid_python.hpp
//! \brief the bridge to the Python closure models (subgrid_predict.py). An embedded
//! Python interpreter (one per MPI rank) runs the script's Predictor, which predicts
//! k_sg, R, Q and C from host copies of u0 and fills host copies of the subgrid cache.
//! Python.h is included only by subgrid_python.cpp, so the rest of AthenaK does not see
//! it.

#include <map>
#include <string>

#include "athena.hpp"

class MeshBlockPack;

namespace subgrid {

class Subgrid;

class PythonPredictor {
 public:
  // Starts the interpreter if needed (program name = executable, so that a venv's
  // packages are found), imports script and creates its Predictor from config (the
  // <subgrid> parameters as strings). Exits with an error if Python raises.
  PythonPredictor(const std::string &script, const std::string &executable,
                  const std::map<std::string, std::string> &config);
  ~PythonPredictor();
  PythonPredictor(const PythonPredictor&) = delete;
  PythonPredictor &operator=(const PythonPredictor&) = delete;

  // from Predictor.info(): stencil of each network (0 if not ml), whether the cooling
  // network takes the k_sg channels, one line per network for the log
  int stencil_ksg = 0, stencil_r = 0, stencil_q = 0, stencil_c = 0, stencil_gate = 0;
  bool c_with_ksg = false;
  std::string description;

  // Fills fsg, ksg, rsg, aeval, aevec, qsg and csg of psg from u0 of pp
  void Predict(MeshBlockPack *pp, Subgrid *psg);

 private:
  void *predictor_ = nullptr;  // PyObject*, the script's Predictor
  // host copies of u0 and of the cache (the device views themselves on a host build)
  DvceArray5D<Real>::HostMirror u0_, rsg_, aeval_, aevec_, qsg_;
  DvceArray4D<Real>::HostMirror fsg_, ksg_, csg_;
};

} // namespace subgrid
#endif // SUBGRID_SUBGRID_PYTHON_HPP_
