# juliaDEC

Discrete exterior calculus (DEC) on Delaunay–Voronoi meshes of the flat torus, in Julia, and the numerical experiments on discrete Hodge stars with one unknown per edge:
how accurate a symmetric star can be on centroidal Voronoi (SCVT) meshes, how to build better ones, and what they give in Poisson, Laplace eigenvalues, Darcy, advection and
shallow water.

The three stars compared throughout are

| star | symmetric | exact on | construction |
|---|---|---|---|
| `H_DEC` | yes | constants | diagonal, `abs(dual edge)/abs(edge)` |
| `H_P1` | yes | linear fields | smallest symmetric correction of `H_DEC` (one global sparse solve) |
| `H_P2` | no (1% antisymmetric) | quadratic fields | smallest row-by-row correction (local) |

## Layout

| folder | content |
|---|---|
| `src/` | the DEC library: `juliaDEC.jl` (meshes, incidence matrices, integrals), `hodge_stars.jl` (Hodge stars), and the earlier `hodges.jl`, `laplacian.jl`, `advection.jl`, … |
| `scripts/stars` | build and check the stars, smallest coefficients of symmetric stars (`min_lambda.jl`) |
| `scripts/barrier` | the skew of the primal–dual pairing, the accuracy barrier, locality |
| `scripts/elliptic` | Poisson, eigenvalues, Darcy, figures of the report |
| `scripts/transport` | Lie derivative by extrusion, advection of densities, 1-forms and scalars |
| `scripts/shallow_water` | linear and nonlinear shallow water (TRiSK in DEC variables) |
| `scripts/shared` | plotting helpers |
| `scripts/archive` | older experiments, kept for the record, not maintained |
| `notebooks/` | exploratory notebooks (outputs stripped) |
| `meshes/` | mesh generators and small SCVT and regular meshes (`mesh1`–`mesh6`) |
| `results/` | earlier CSV and text results (advection, Laplacian eigenvalues) |
| `output/` | results of the runs (not versioned) |

An index with one line per script is in [scripts/README.md](scripts/README.md).

## Getting started

Requires Julia 1.11 or later (developed with 1.12).

```julia
julia> import Pkg; Pkg.activate("."); Pkg.instantiate()
```

The packages `VoronoiMeshes`, `TensorsLite`, `TensorsLiteGeometry` and `ImmutableVectors` are not registered; `Project.toml` takes them from their repositories (github.com/favba).
Some of the earlier files in `src/` and some notebooks also use `MPASTools`, `JLD2`, `CSV` and `DataFrames`, which are not in `Project.toml`.

Run the scripts **from the repository root**:

```
julia --project=. -t 4 scripts/stars/stars_cache.jl mesh3 mesh4     # builds and caches the three stars
julia --project=. scripts/barrier/green_charge_check.jl             # a check that needs no mesh
julia --project=. -t 4 scripts/stars/min_lambda.jl 2 4 mesh2        # smallest coefficients of a symmetric star exact on quadratic fields
```

`mesh7` (65536 cells, 19 MB) and the perturbed families are not versioned; `meshes/mesh_generator*.jl` regenerate meshes of this kind.

## Scripts that depend on earlier results

Some figure scripts read result files that other scripts write to `output/ts_data` (for example `report_figures_theory.jl` reads `green_charge_mesh.jls` and `min_lambda.txt`), so run those first.

## History

The repository was reorganised in November 2026. The earlier layout (flat files, a copy of VoronoiMeshes.jl, notebooks with outputs, mesh data) is in the branch `old-layout`. `scripts/barrier/gap_skew_more.jl` also needs the mesh family `meshes_felipe`, which is only in that branch.

## License

The code on this branch is released under the [MIT License](LICENSE). Two functions come from other MIT-licensed packages: see [THIRD_PARTY.md](THIRD_PARTY.md).
The branch `old-layout` contains a copy of VoronoiMeshes.jl, which keeps the licence in its own folder.
