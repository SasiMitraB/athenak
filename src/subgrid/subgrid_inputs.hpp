#ifndef SUBGRID_SUBGRID_INPUTS_HPP_
#define SUBGRID_SUBGRID_INPUTS_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file subgrid_inputs.hpp
//! \brief the standardised, padded network inputs of the subgrid models from u0, shared
//! by the Predict tasks, so that every model sees the boundaries the same way.
//!
//! Training padded the coarse grid with the run's boundaries: periodic, or an edge copy
//! at a physical boundary, applied to u0 for the gradients and again to the network
//! inputs for the stencil. Per MeshBlock and face:
//!  - block / periodic face: the ghost zones of u0 hold the neighbour's cells, so they
//!    are read as they are. A network output is exact out to ng - reach ghost layers
//!    (reach = stencil/2 + 1); beyond, reads are clamped to the array.
//!  - physical face: reads are clamped to the active cells, the training edge copy, so
//!    the active cells see exactly the training-time inputs whatever the boundary
//!    condition.
//! Every input channel of a cell, the 20 features of sn::cell_features and any channels a
//! model adds after them (the gate and k_sg channels of the cooling model), is taken at
//! the same clamped source cell, so the added channels are edge-copied as the features
//! are.

#include <cstddef>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "bvals/bvals.hpp"
#include "hydro/hydro.hpp"
#include "subgrid/subgrid_net.hpp"

namespace subgrid {

// true at a MeshBlock face on a physical boundary of the mesh, whose ghost zones are set
// by a boundary condition rather than copied from a neighbouring MeshBlock (block) or
// from the other side of the mesh (periodic)
KOKKOS_INLINE_FUNCTION bool IsPhysical(BoundaryFlag f) {
  return f != BoundaryFlag::block && f != BoundaryFlag::periodic;
}

// the cell whose f fills cell i of a MeshBlock with active cells is..ie along one axis:
// i itself, except in the ghost zones behind a physical boundary, where f is mirrored
// (reflect) or copied from the last active cell (any other condition)
KOKKOS_INLINE_FUNCTION int GhostSource(int i, int is, int ie, BoundaryFlag inner,
                                       BoundaryFlag outer) {
  if (i < is && IsPhysical(inner)) {
    return (inner == BoundaryFlag::reflect) ? Kokkos::min(2*is - 1 - i, ie) : is;
  }
  if (i > ie && IsPhysical(outer)) {
    return (outer == BoundaryFlag::reflect) ? Kokkos::max(2*ie + 1 - i, is) : ie;
  }
  return i;
}

// the output cells of a network in each MeshBlock: n3 x n2 x n1 cells from the cell
// (o3, o2, o1) of the block (indices with ghost zones, as u0). Output cell (k, j, i) of
// the network, at out(((m*n3 + k)*n2 + j)*n1 + i) (channel 0), is the block cell
// (o3 + k, o2 + j, o1 + i).
struct OutputBox {
  int o3, o2, o1;
  int n3, n2, n1;
  KOKKOS_INLINE_FUNCTION size_t Cell(int m, int k, int j, int i) const {
    return ((static_cast<size_t>(m)*n3 + (k - o3))*n2 + (j - o2))*n1 + (i - o1);
  }
};

//----------------------------------------------------------------------------------------
// The network evaluations of the subgrid models (subgrid_net.cpp). All code that launches
// a network kernel lives in that one file: with CUDA, an inline function or template
// holding a KOKKOS_LAMBDA (sn::layers, par_for over a kernel type) that is compiled in
// two files segfaults on launch, as the linker keeps one file's host-device lambda
// wrapper, whose device function pointer is null. Call these, not sn::layers or
// sn::make_work, elsewhere.

// Sizes net for the output cells box of every MeshBlock and fills its padded inputs (box
// plus stencil/2 cells each side) with the 20 standardised features of u0 (k_sg, gate).
void FillFeatureInputs(MeshBlockPack *pp, SubgridNet &net, const OutputBox &box);

// As FillFeatureInputs, plus the cooling model's channels at the same source cell:
//   20      g = 1 where the gate logit (channel 0 of gate, on gate_box) is > 0 (p > 0.5)
//   21, 22  if with_ksg: log10 f and log10 (1 - f), f the k_sg fraction of PredictKsg
// gate_box must hold every cell box reads, i.e. box plus stencil/2 of net each side.
void FillCoolingInputs(MeshBlockPack *pp, SubgridNet &net, const OutputBox &box,
                       const sn::Vec &gate, const OutputBox &gate_box,
                       const DvceArray4D<Real> &f, bool with_ksg);

// Runs net on its filled inputs: raw outputs in net.out, (n_out, nmb, box n3, n2, n1).
void EvaluateNet(SubgridNet &net);

} // namespace subgrid
#endif // SUBGRID_SUBGRID_INPUTS_HPP_
