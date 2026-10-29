# Subgrid module

The `subgrid` module adds the closure terms of the coarse-grained (filtered) Euler
equations to AthenaK's Hydro module. The equations are derived in chapter 1 of the theory
notes (`theory/notes/Overleaf/chapters/01-derivation.tex`, "Final Set of Equations"):

```
d_t <rho> + div <m>                                                   = 0
d_t <m>   + div( <m><m>/<rho> + R ) + grad <P>                        = 0
d_t <E>   + div( (<E> + <P>) <m>/<rho> + Q )                          = -(<rho>/mu m_p)^2 Lambda(T_cg) - C
<P>/(gamma-1) = <E> - (1/2) <m>.<m>/<rho> - (1/2) Tr(R)
```

| Term | Meaning | Components | Array |
|---|---|---|---|
| `R` | subgrid momentum flux (symmetric) | 6: `xx, yy, zz, xy, xz, yz` | `Subgrid::rflx (nmb, 6, n3, n2, n1)` |
| `Q` | subgrid energy flux | 3: `x, y, z` | `Subgrid::qflx (nmb, 3, n3, n2, n1)` |
| `C` | cooling correction | 1 | `Subgrid::ccool (nmb, n3, n2, n1)` |

**Current state:** boilerplate. `R`, `Q` and `C` are set to zero, and the EOS and
timestep are copies of the ideal-gas Hydro versions. In this state, a run with
`<subgrid>` is bitwise identical to the same run without it, on both CPU and GPU (see
[Verification](#verification)).

## Contents

- [Enabling it](#enabling-it)
- [Design](#design)
- [Files](#files)
- [Where to put the real physics](#where-to-put-the-real-physics)
- [Verification](#verification)
- [Known limitations](#known-limitations)
- [Repository and branch layout](#repository-and-branch-layout)

## Enabling it

Add a `<subgrid>` block to the input file. It may be empty:

```
<hydro>
eos = ideal
...

<subgrid>
```

This one switch turns on all three changes: the subgrid fluxes and source term, the
subgrid EOS, and the subgrid timestep. The module stops with a `FATAL ERROR` if any of
these hold:

- there is no `<hydro>` block, or there is an `<mhd>`, `<radiation>`, `<adm>` or
  `<z4c>` block;
- the coordinates are special or general relativistic, or `<hydro>/eos` is not `ideal`;
- `<time>/evolution` is not `dynamic`.

`subgrid` is registered as a valid block-name prefix in `ParameterInput::CheckBlockNames`.
Prefix matching means that later blocks such as `<subgrid_rvv>` are accepted too.

## Design

The module does not copy the Hydro class. It is an add-on object owned by the
`MeshBlockPack` (`pmbp->psubgrid`), following the pattern of `TurbulenceDriver`, and it
inserts its tasks into the task list that Hydro has already built. Everything else
(reconstruction, Riemann solvers, `RKUpdate`, boundary exchange, SMR/AMR, outputs,
problem generators) is the stock Hydro code, unchanged.

### Task graph

`Hydro::AssembleHydroTasks` builds the `stagen` list as usual. Then
`Subgrid::IncludeSubgridTasks` adds three tasks (marked `[sgs]`):

```
copyu -> flux -> [sgs] AddSubgridFluxes -> sendf -> recvf -> rkupdt
      -> srctrms -> [sgs] SubgridSrcTerms -> sendu_oa -> ... -> c2p
      -> newdt -> [sgs] NewTimeStep
```

| Task | When | What it does |
|---|---|---|
| `AddSubgridFluxes` | after `flux`, before `sendf` | Calls `ComputeSubgridTerms(w0)` to fill `R`, `Q`, `C` on all cells, ghost zones included. Then it adds face averages to the Hydro fluxes: `F_i(m_j) += 0.5*(R_ij[L] + R_ij[R])` and `F_i(E) += 0.5*(Q_i[L] + Q_i[R])`. Because this runs before `sendf`, the SMR/AMR flux correction exchanges the corrected fluxes, so conservation across refinement boundaries is kept. `RKUpdate` reads only `uflx`, so it needs no change. |
| `SubgridSrcTerms` | after `srctrms`, before `sendu_oa` | `u0(IEN) -= beta*dt*C`, using the `C` computed earlier in the same stage. |
| `NewTimeStep` | after `newdt`, last stage only | Computes `Subgrid::dtnew` (currently the Hydro CFL limit `min dx/(|v| + c_s)`). |

`InsertTask` returns `TaskID(0)` if the insertion point is missing. The module checks
for this and stops with a `FATAL ERROR` rather than silently running without its
tasks.

### Timestep

`Mesh::NewTimeStep` uses `psubgrid->dtnew` **in place of** `phydro->dtnew` when the
module is active. The two are not combined with a minimum, so a subgrid timestep
condition can be less restrictive than the Hydro CFL condition if the physics allows
it. Limits from source terms (`psrc->dtnew`), viscosity, conduction and particles still
apply as before. `Subgrid::NewTimeStep` is also called wherever AthenaK calls
`Hydro::NewTimeStep` outside the task list: in `Driver::Initialize` (first timestep)
and in `MeshRefinement::AdaptiveMeshRefinement` (after regridding).

### EOS

The Hydro constructor chooses `SubgridIdealHydro` instead of `IdealHydro` when
`<subgrid>` exists, for non-relativistic ideal gas only. Every caller of
`phydro->peos` therefore uses it automatically: `ConToPrim`, problem generators calling
`PrimToCons`, FOFC, prolongation, restarts. It reads `gamma` and the floors from
`<hydro>` exactly as `IdealHydro` does. The single-cell conversions
`SingleC2P_SubgridHyd` / `SingleP2C_SubgridHyd` are copies, kept separate from
`SingleC2P_IdealHyd` so that changing them does not affect plain Hydro. The places
where `(1/2) Tr(R)` enters (eq-pressure) are marked `TODO(subgrid)`.

The Riemann solvers and `Hydro::NewTimeStep` still use the inline sound-speed and
pressure helpers in `EOS_Data` (`IdealGasPressure`, `IdealHydroSoundSpeed`). If the
subgrid EOS changes how pressure relates to the primitives, these call sites need
attention too.

## Files

### New files (`src/subgrid/`)

| File | Contents |
|---|---|
| `subgrid.hpp` | `subgrid::Subgrid` class; `SubgridRIndex` (`IRXX, IRYY, IRZZ, IRXY, IRXZ, IRYZ`) and `SubgridQIndex` (`IQX, IQY, IQZ`) enums; `SubgridTaskIDs`. |
| `subgrid.cpp` | Constructor (checks and array allocation), `IncludeSubgridTasks`, `ComputeSubgridTerms` (zero placeholder), `AddSubgridFluxes`, `SubgridSrcTerms`. |
| `subgrid_newdt.cpp` | `Subgrid::NewTimeStep`, a copy of the non-relativistic branch of `Hydro::NewTimeStep`. |
| `subgrid_eos.hpp` | `SubgridIdealHydro` class and the inline `SingleC2P_SubgridHyd` / `SingleP2C_SubgridHyd`. |
| `subgrid_eos.cpp` | `SubgridIdealHydro` constructor, `ConsToPrim`, `PrimToCons`, copies of `eos/ideal_hyd.cpp`. |
| `subgrid_documentation.md` | This file. |

### Edits to existing AthenaK files (about 40 lines)

| File | Change |
|---|---|
| `src/CMakeLists.txt` | Adds the three `subgrid/*.cpp` sources. |
| `src/parameter_input.cpp` | Adds `"subgrid"` to the valid input-block names. |
| `src/mesh/meshblock_pack.hpp` | Forward declaration and `subgrid::Subgrid *psubgrid` member. |
| `src/mesh/meshblock_pack.cpp` | In `AddPhysics`, step "(2b) SUBGRID" right after Hydro: constructs `Subgrid` and calls `IncludeSubgridTasks(tl_map)`. Deletes it in the destructor, before `phydro`. |
| `src/hydro/hydro.cpp` | Chooses `SubgridIdealHydro` when `<subgrid>` exists (non-relativistic ideal gas). |
| `src/mesh/mesh.cpp` | `Mesh::NewTimeStep` uses `psubgrid->dtnew` instead of `phydro->dtnew` when present. |
| `src/driver/driver.cpp` | Initial timestep: calls `psubgrid->NewTimeStep`. |
| `src/mesh/mesh_refinement.cpp` | Timestep after AMR regridding: calls `psubgrid->NewTimeStep`. |

## Where to put the real physics

1. **Closure model:** replace the body of `Subgrid::ComputeSubgridTerms(w0)`. It must
   fill `rflx`, `qflx` and `ccool` on all cells that `AddSubgridFluxes` reads. The face
   averages use cells `is-1 .. ie+1` (and the same in `j` and `k`), so at least one ghost
   layer must be valid. `w0` is valid in all ghost zones, because `ConToPrim` covers
   them. The arrays are not cleared between stages, so the function must overwrite
   everything it is responsible for.
2. **Face fluxes:** if a model predicts face-centred values directly, replace the
   averaging in `AddSubgridFluxes`.
3. **EOS:** add `(1/2) Tr(R)` in `SingleC2P_SubgridHyd` / `SingleP2C_SubgridHyd`. The EOS
   can reach the closure arrays at call time through
   `pmy_pack->psubgrid->rflx` (the pack pointer exists before any conversion runs).
   Note that `ConToPrim` runs over the ghost zones and also in `Driver::Initialize`,
   before the first `AddSubgridFluxes` call, so `rflx` must hold something sensible
   there.
4. **Timestep:** edit `Subgrid::NewTimeStep`. It is the only CFL-type limit used when
   `<subgrid>` is on.
5. **Cooling:** `C` is applied in `SubgridSrcTerms`. The resolved cooling
   `-(<rho>/mu m_p)^2 Lambda(T_cg)` is not part of this module. It still comes from
   `<hydro_srcterms>` or a pgen `user_srcs_func`, which run in `srctrms`, just before
   `SubgridSrcTerms`.

## Verification

Each test problem was run twice with the same binary: as plain Hydro, and with an empty
`<subgrid>` block added. Problems used (all built-in pgens, so no `-DPROBLEM` is
needed):

| Test | Input | Overrides | Covers |
|---|---|---|---|
| `sod1d` | `inputs/hydro/sod.athinput` | `tlim=0.2`, `rsolver=hllc` | 1D |
| `implode2d` | `inputs/hydro/lw_implode.athinput` | 128², 16 meshblocks, `nghost=3`, `ppm4`, `hllc`, `tlim=0.5` | 2D, reflecting boundaries, multiple meshblocks |
| `lwave3d` | `inputs/tests/linear_wave_hydro.athinput` | 32x16x16, 4 meshblocks, `tlim=1.0` | 3D (x3 fluxes) |
| `lwave2d_amr` | `inputs/tests/linear_wave_hydro_amr.athinput` | 128x64, `tlim=1.0` | adaptive mesh refinement, flux correction, timestep after regridding |

Compared outputs:

- restart files (`rst`, full double precision `u0` and mesh state), byte for byte, from
  `<par_end>` onwards (the text before it is the input file, which differs by the
  `<subgrid>` line);
- history files (`hst`), byte for byte;
- the `cycle=… time=… dt=…` line of every cycle in the log.

`.bin` outputs are not suitable for the bitwise test: they are float32 and contain the
input file in their header.

Results (2026-10-29):

| Build | Hydro vs subgrid |
|---|---|
| CPU (`builds/subgrid_dev`, serial, double) | all 4 tests bitwise identical (restart, history, every dt) |
| GPU (`builds/subgrid_dev_gpu`, CUDA 12.8, `ADA89`, RTX 4000 Ada, double) | all 4 tests bitwise identical |

CPU vs GPU: same cycle count and identical history files for every test. Field data
agrees to float32 rounding (largest relative difference 6.6e-8, `lwave2d_amr`). The
CPU-vs-GPU difference is exactly the same with and without `<subgrid>`.

**Negative control.** To show the tasks really run, `ComputeSubgridTerms` was
temporarily changed to set `R = 1e-3` and `C = 1e-2`, and `Subgrid::NewTimeStep` to
halve `dtnew`. Every output then differed from Hydro, and the first timestep was exactly
half (2.641107e-03 → 1.320554e-03 for `sod1d`). The change was reverted and the
identity test re-run.

**Restart-header caveat.** Without mesh refinement, AthenaK never sets the coarse-cell
fields of `Mesh::mesh_indcs` (`cnx1 … cke`, 9 ints), so restart files contain
uninitialized memory at those 36 bytes. On the GPU build these bytes differed between
the hydro and subgrid runs, although every field value was identical. The comparison
masks them. This is an existing upstream behaviour and is unrelated to this module.

### Building and re-running

```bash
# CPU
cmake -S athenak -B builds/subgrid_dev -DCMAKE_BUILD_TYPE=Release
cmake --build builds/subgrid_dev -j 24

# GPU (RTX 4000 Ada, CUDA 12.8)
export PATH=/usr/local/cuda-12.8/bin:$PATH NVCC_WRAPPER_DEFAULT_ARCH=sm_89
cmake -S athenak -B builds/subgrid_dev_gpu -DCMAKE_BUILD_TYPE=Release \
      -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_ADA89=ON \
      -DCMAKE_CXX_COMPILER=$PWD/athenak/kokkos/bin/nvcc_wrapper
cmake --build builds/subgrid_dev_gpu -j 16
```

The comparison scripts used in the session (`run_compare.sh`, `cmp_outputs.py`,
`cross_compare.py`) were kept in a temporary scratch directory and are not part of the
repository.

## Known limitations

- **FOFC:** first-order flux correction runs inside `Hydro::Fluxes`, before
  `AddSubgridFluxes`, so it neither sees nor limits the subgrid fluxes.
- **STS diffusion:** the subgrid terms are not part of the super-time-stepping
  (parabolic) task lists.
- **Not tested:** MPI builds, shearing box, and passive scalars (all tests above have
  `nscalars = 0`).
- **No outputs yet** for `R`, `Q`, `C`. Adding them needs a new `var_choice` entry in
  `outputs/` (as done for `turb_force`).
- **The tensors are cell-centred and averaged to faces.** This is a choice of the
  boilerplate, not of the theory.

## Repository and branch layout

Made in the same session, before the module was added:

- The previous state of the `SubgridCGM` branch (pybind11 embedding, the
  `subgrid.cpp` / `kh_radiative_cooling.cpp` pgens, the cooling cutoff in
  `ismcooling.hpp`, the subgrid timestep limit) was moved to the branch
  **`PDFLearning`** (fork `origin`). That branch also has a backup of the embedded Python
  file, `src/pgen/source_module.py`, copied from
  `builds/subgrid_model/src/source_module.py` in the parent repo.
- `SubgridCGM` and local `main` were reset to `origin/main` = upstream
  `IAS-Astrophysics/athenak` `main` at `c5a0d7f9` ("Cyclic zoom (#713)"). This module is
  built on that commit.
