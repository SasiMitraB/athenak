#ifndef SUBGRID_SUBGRID_EOS_HPP_
#define SUBGRID_SUBGRID_EOS_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file subgrid_eos.hpp
//! \brief EOS used by Hydro when <subgrid> exists. The coarse-grained pressure is
//!   <P>/(gamma-1) = <E> - (1/2)<m>^2/<rho> - (1/2)Tr(R)
//! so it will depend on the subgrid momentum flux R. Currently a copy of IdealHydro.

#include "athena.hpp"
#include "eos/eos.hpp"

//----------------------------------------------------------------------------------------
//! \class SubgridIdealHydro
//! \brief Derived class for ideal gas EOS in nonrelativistic subgrid hydro

class SubgridIdealHydro : public EquationOfState {
 public:
  // Following suppress warnings that MHD versions are not over-ridden
  using EquationOfState::ConsToPrim;
  using EquationOfState::PrimToCons;

  SubgridIdealHydro(MeshBlockPack *pp, ParameterInput *pin);
  void ConsToPrim(DvceArray5D<Real> &cons, DvceArray5D<Real> &prim,
                  const bool only_testfloors,
                  const int il, const int iu, const int jl, const int ju,
                  const int kl, const int ku) override;
  void PrimToCons(const DvceArray5D<Real> &prim, DvceArray5D<Real> &cons,
                  const int il, const int iu, const int jl, const int ju,
                  const int kl, const int ku) override;
};

//----------------------------------------------------------------------------------------
//! \fn void SingleC2P_SubgridHyd()
//! \brief Converts single state of conserved variables into primitive variables for
//! subgrid hydrodynamics. Conserved = (d,M1,M2,M3,E), Primitive = (d,vx,vy,vz,e)
//! where E=total energy density and e=internal energy density.
//! Currently identical to SingleC2P_IdealHyd.

KOKKOS_INLINE_FUNCTION
void SingleC2P_SubgridHyd(HydCons1D &u, const EOS_Data &eos,
                          HydPrim1D &w,
                          bool &dfloor_used, bool &efloor_used, bool &tfloor_used) {
  const Real &dfloor_ = eos.dfloor;
  Real efloor = eos.pfloor/(eos.gamma - 1.0);
  Real tfloor = eos.tfloor;
  Real sfloor = eos.sfloor;
  Real gm1 = eos.gamma - 1.0;

  // apply density floor, without changing momentum or energy
  if (u.d < dfloor_) {
    u.d = dfloor_;
    dfloor_used = true;
  }
  w.d = u.d;

  // compute velocities
  Real di = 1.0/u.d;
  w.vx = di*u.mx;
  w.vy = di*u.my;
  w.vz = di*u.mz;

  // set internal energy, apply floor, correct total energy (if needed)
  // TODO(subgrid): also subtract the subgrid kinetic energy (1/2)Tr(R)
  Real e_k = 0.5*di*(SQR(u.mx) + SQR(u.my) + SQR(u.mz));
  w.e = (u.e - e_k);
  if (w.e < efloor) {
    w.e = efloor;
    u.e = efloor + e_k;
    efloor_used = true;
  }
  // apply temperature floor
  if (gm1*w.e*di < tfloor) {
    w.e = w.d*tfloor/gm1;
    u.e = w.e + e_k;
    tfloor_used = true;
  }
  // apply entropy floor
  Real spe_over_eps = gm1/pow(w.d, gm1);
  Real spe = spe_over_eps*w.e*di;
  if (spe <= sfloor) {
    w.e = w.d*sfloor/spe_over_eps;
    efloor_used = true;
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void SingleP2C_SubgridHyd()
//! \brief Converts single state of primitive variables into conserved variables for
//! subgrid hydrodynamics. Currently identical to SingleP2C_IdealHyd.

KOKKOS_INLINE_FUNCTION
void SingleP2C_SubgridHyd(const HydPrim1D &w, HydCons1D &u) {
  u.d  = w.d;
  u.mx = w.d*w.vx;
  u.my = w.d*w.vy;
  u.mz = w.d*w.vz;
  // TODO(subgrid): also add the subgrid kinetic energy (1/2)Tr(R)
  u.e = w.e + 0.5*w.d*(SQR(w.vx) + SQR(w.vy) + SQR(w.vz));
  return;
}

#endif // SUBGRID_SUBGRID_EOS_HPP_
