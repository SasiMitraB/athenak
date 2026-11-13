//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file subgrid_python.cpp
//! \brief the prediction of the closure terms k_sg, R, Q and C by the Python models of
//! subgrid_predict.py, through an embedded interpreter (one per MPI rank).
//!
//! Each call copies u0 (with its ghost zones) to the host, hands it and the host copies
//! of the subgrid cache to Predictor.predict as memoryviews, which fills the cache in
//! place, and copies the cache back to the device. On a host build (Serial / OpenMP) the
//! host copies are the device views themselves, so nothing is copied.

// Python.h must come first (it sets feature macros of the standard headers)
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <map>
#include <string>
#include <type_traits>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "bvals/bvals.hpp"
#include "hydro/hydro.hpp"
#include "eos/eos.hpp"
#include "units/units.hpp"
#include "subgrid/subgrid.hpp"
#include "subgrid/subgrid_python.hpp"

namespace subgrid {

namespace {

// Prints the pending Python exception and exits.
[[noreturn]] void PythonFatal(const std::string &what) {
  std::cout << "### FATAL ERROR in " << __FILE__ << std::endl
            << "Python subgrid models: " << what << std::endl;
  if (PyErr_Occurred()) PyErr_Print();
  std::cout << std::flush;
  std::exit(EXIT_FAILURE);
}

PyObject *Check(PyObject *obj, const std::string &what) {
  if (obj == nullptr) PythonFatal(what);
  return obj;
}

// Starts the interpreter once per process, with the GIL released afterwards (every call
// into Python below takes it with PyGILState_Ensure).
void StartPython(const std::string &executable) {
  if (Py_IsInitialized()) return;
  PyConfig config;
  PyConfig_InitPythonConfig(&config);
  config.install_signal_handlers = 0;  // AthenaK handles the signals
  config.parse_argv = 0;
  PyStatus status = PyStatus_Ok();
  if (!executable.empty()) {
    // a venv's python here makes its site-packages importable (pyvenv.cfg)
    status = PyConfig_SetBytesString(&config, &config.program_name, executable.c_str());
  }
  if (!PyStatus_Exception(status)) status = Py_InitializeFromConfig(&config);
  PyConfig_Clear(&config);
  if (PyStatus_Exception(status)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << std::endl
              << "cannot start the Python interpreter (" << executable << "): "
              << (status.err_msg ? status.err_msg : "") << std::endl;
    std::exit(EXIT_FAILURE);
  }
  (void) PyEval_SaveThread();
}

// A writable memoryview of the n elements of type T at data (owned by AthenaK).
template <class T>
PyObject *View(T *data, size_t n, const char *name) {
  return Check(PyMemoryView_FromMemory(reinterpret_cast<char *>(data),
                                       static_cast<Py_ssize_t>(n*sizeof(T)), PyBUF_WRITE),
               std::string("memoryview of ") + name);
}

// (Re)creates h as a host mirror of d when d was (re)allocated.
template <class H, class D>
void Mirror(H &h, const D &d) {
  if (h.data() == nullptr || h.size() != d.size() ||
      (std::is_same<typename H::memory_space, typename D::memory_space>::value &&
       static_cast<const void *>(h.data()) != static_cast<const void *>(d.data()))) {
    h = Kokkos::create_mirror_view(d);
  }
}

// int attribute key of dict d, or a fatal error
int64_t DictInt(PyObject *d, const char *key) {
  PyObject *v = PyDict_GetItemString(d, key);  // borrowed
  if (v == nullptr) PythonFatal(std::string("Predictor.info() has no '") + key + "'");
  int64_t r = PyLong_AsLongLong(v);
  if (PyErr_Occurred()) PythonFatal(std::string("Predictor.info()['") + key + "'");
  return r;
}

// The face codes of subgrid_predict.py (BC_OPEN, BC_REFLECT, BC_OTHER)
int FaceCode(BoundaryFlag f) {
  if (f == BoundaryFlag::block || f == BoundaryFlag::periodic) return 0;
  if (f == BoundaryFlag::reflect) return 1;
  return 2;
}

} // namespace

//----------------------------------------------------------------------------------------
// constructor: starts Python, imports the script and creates its Predictor

PythonPredictor::PythonPredictor(const std::string &script, const std::string &executable,
                                 const std::map<std::string, std::string> &config) {
  StartPython(executable);
  PyGILState_STATE gil = PyGILState_Ensure();

  // import the script as a module, from its directory
  const size_t slash = script.find_last_of('/');
  const std::string dir = (slash == std::string::npos) ? "." : script.substr(0, slash);
  std::string name = (slash == std::string::npos) ? script : script.substr(slash + 1);
  if (name.size() > 3 && name.compare(name.size() - 3, 3, ".py") == 0) {
    name.resize(name.size() - 3);
  }
  PyObject *sys_path = Check(PySys_GetObject("path"), "sys.path");  // borrowed
  PyObject *pydir = Check(PyUnicode_FromString(dir.c_str()), "script directory");
  if (PyList_Insert(sys_path, 0, pydir) != 0) PythonFatal("sys.path.insert");
  Py_DECREF(pydir);
  PyObject *module = Check(PyImport_ImportModule(name.c_str()),
                           "cannot import " + script);

  // Predictor = module.create(config)
  PyObject *pyconfig = Check(PyDict_New(), "config dict");
  for (const auto &kv : config) {
    PyObject *v = Check(PyUnicode_FromString(kv.second.c_str()), kv.first);
    PyDict_SetItemString(pyconfig, kv.first.c_str(), v);
    Py_DECREF(v);
  }
  PyObject *pred = Check(PyObject_CallMethod(module, "create", "(O)", pyconfig),
                         "cannot set up the models of <subgrid>");
  Py_DECREF(pyconfig);
  Py_DECREF(module);
  predictor_ = pred;

  // what the Subgrid constructor checks the mesh against
  PyObject *info = Check(PyObject_CallMethod(pred, "info", nullptr), "Predictor.info()");
  if (!PyDict_Check(info)) PythonFatal("Predictor.info() must return a dict");
  stencil_ksg = static_cast<int>(DictInt(info, "stencil_ksg"));
  stencil_r = static_cast<int>(DictInt(info, "stencil_r"));
  stencil_q = static_cast<int>(DictInt(info, "stencil_q"));
  stencil_c = static_cast<int>(DictInt(info, "stencil_c"));
  stencil_gate = static_cast<int>(DictInt(info, "stencil_gate"));
  c_with_ksg = (DictInt(info, "c_with_ksg") != 0);
  PyObject *desc = PyDict_GetItemString(info, "description");  // borrowed
  if (desc != nullptr && PyUnicode_Check(desc)) {
    const char *s = PyUnicode_AsUTF8(desc);
    if (s != nullptr) description = s;
  }
  PyErr_Clear();
  Py_DECREF(info);
  PyGILState_Release(gil);
}

//----------------------------------------------------------------------------------------
// destructor: drops the Predictor. The interpreter is left running until the process
// exits (finalising it would also tear down JAX's runtime, which other ranks' or later
// Subgrid objects of this process may still use).

PythonPredictor::~PythonPredictor() {
  if (predictor_ != nullptr && Py_IsInitialized()) {
    PyGILState_STATE gil = PyGILState_Ensure();
    Py_DECREF(static_cast<PyObject *>(predictor_));
    PyGILState_Release(gil);
  }
}

//----------------------------------------------------------------------------------------
//! \fn void PythonPredictor::Predict
//! \brief Predictor.predict(buffers, meta) on host copies of u0 and of the cache of psg

void PythonPredictor::Predict(MeshBlockPack *pp, Subgrid *psg) {
  auto &indcs = pp->pmesh->mb_indcs;
  auto &u0 = pp->phydro->u0;
  const int nmb = pp->nmb_thispack;

  // (1) host copies
  Kokkos::fence();
  Mirror(u0_, u0);
  Mirror(fsg_, psg->fsg);
  Mirror(ksg_, psg->ksg);
  Mirror(rsg_, psg->rsg);
  Mirror(aeval_, psg->aeval);
  Mirror(aevec_, psg->aevec);
  Mirror(qsg_, psg->qsg);
  Mirror(csg_, psg->csg);
  Kokkos::deep_copy(u0_, u0);
  std::vector<int32_t> bcs(6*static_cast<size_t>(nmb));
  auto &mb_bcs = pp->pmb->mb_bcs.h_view;
  for (int m = 0; m < nmb; ++m) {
    for (int f = 0; f < 6; ++f) bcs[6*m + f] = FaceCode(mb_bcs(m, f));
  }

  // units of the cooling correction (as SourceTerms::ISMCooling): rho^2 Lambda /
  // cooling_unit is the code energy density per code time of n^2 Lambda
  double gm1 = pp->phydro->peos->eos_data.gamma - 1.0;
  double temp_unit = 0.0, cooling_unit = 0.0;
  if (pp->punit != nullptr) {
    temp_unit = pp->punit->temperature_cgs();
    const double n_unit = pp->punit->density_cgs()/pp->punit->mu()
                          /pp->punit->atomic_mass_unit_cgs;
    cooling_unit = pp->punit->pressure_cgs()/pp->punit->time_cgs()/n_unit/n_unit;
  }

  // (2) Predictor.predict
  PyGILState_STATE gil = PyGILState_Ensure();
  PyObject *buffers = Check(PyDict_New(), "buffers dict");
  auto put = [&](const char *name, PyObject *view) {
    PyDict_SetItemString(buffers, name, view);
    Py_DECREF(view);
  };
  put("u0", View(u0_.data(), u0_.size(), "u0"));
  put("fsg", View(fsg_.data(), fsg_.size(), "fsg"));
  put("ksg", View(ksg_.data(), ksg_.size(), "ksg"));
  put("csg", View(csg_.data(), csg_.size(), "csg"));
  put("rsg", View(rsg_.data(), rsg_.size(), "rsg"));
  put("aeval", View(aeval_.data(), aeval_.size(), "aeval"));
  put("aevec", View(aevec_.data(), aevec_.size(), "aevec"));
  put("qsg", View(qsg_.data(), qsg_.size(), "qsg"));
  put("bcs", View(bcs.data(), bcs.size(), "bcs"));
  PyObject *meta = Check(Py_BuildValue(
      "{s:i,s:i,s:i,s:i,s:i,s:i,s:i,s:d,s:d,s:d}",
      "nmb", nmb, "nvar", static_cast<int>(u0.extent_int(1)), "ng", indcs.ng,
      "nx1", indcs.nx1, "nx2", indcs.nx2, "nx3", indcs.nx3,
      "real_size", static_cast<int>(sizeof(Real)),
      "gm1", gm1, "temp_unit", temp_unit, "cooling_unit", cooling_unit), "meta dict");
  PyObject *r = PyObject_CallMethod(static_cast<PyObject *>(predictor_), "predict",
                                    "(OO)", buffers, meta);
  if (r == nullptr) PythonFatal("Predictor.predict failed");
  Py_DECREF(r);
  Py_DECREF(meta);
  Py_DECREF(buffers);  // the memoryviews go with it: Python must not keep them
  PyGILState_Release(gil);

  // (3) the cache back to the device
  Kokkos::deep_copy(psg->fsg, fsg_);
  Kokkos::deep_copy(psg->ksg, ksg_);
  Kokkos::deep_copy(psg->rsg, rsg_);
  Kokkos::deep_copy(psg->aeval, aeval_);
  Kokkos::deep_copy(psg->aevec, aevec_);
  Kokkos::deep_copy(psg->qsg, qsg_);
  Kokkos::deep_copy(psg->csg, csg_);
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus Subgrid::PredictClosures
//! \brief Predicts f and k_sg, R (with the eigen-decomposition of A), Q and C from u0,
//! all by the Python models (subgrid_predict.py): f and k_sg in all ng ghost layers, R
//! and Q in 1 ghost layer, C in the active cells.

TaskStatus Subgrid::PredictClosures(Driver *pdrive, int stage) {
  ppredict->Predict(pmy_pack, this);
  return TaskStatus::complete;
}

} // namespace subgrid
