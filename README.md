# juliaDEC

Discrete exterior calculus (DEC) on Delaunay–Voronoi meshes of the flat torus, in Julia: the code of the experiments on discrete Hodge stars with one unknown per edge.
How accurate can a symmetric star be on centroidal Voronoi (SCVT) meshes, how to build better ones, and what they give in Poisson, Laplace eigenvalues, Darcy, advection and
linear shallow water.

The three stars compared throughout are

| star | symmetric | exact on | construction |
|---|---|---|---|
| `H_DEC` | yes | constants | diagonal, `abs(dual edge)/abs(edge)` |
| `H_P1` | yes | linear fields | smallest symmetric correction of `H_DEC` (one global sparse solve) |
| `H_P2` | no (0.7–1.8% antisymmetric) | quadratic fields | smallest row-by-row correction (local) |

## Layout

| folder | content |
|---|---|
| `src/` | the library: `juliaDEC.jl` (incidence matrices, diagonal stars, integrals on edges) and `hodge_stars.jl` (reconstructions and reductions) |
| `scripts/stars` | build and check the three stars (`three_stars.jl`, `stars_cache.jl`), smallest coefficients of symmetric stars (`min_lambda.jl`) |
| `scripts/barrier` | the skew of the primal–dual pairing, the accuracy barrier, locality |
| `scripts/elliptic` | Poisson, eigenvalues, Darcy |
| `scripts/transport` | Lie derivative by extrusion, advection of densities and 1-forms |
| `scripts/shallow_water` | linear rotating shallow water |
| `scripts/shared` | plotting helpers |
| `meshes/` | mesh generators, and the SCVT (`mesh1`–`mesh6`), regular and perturbed (`pert1`–`pert6`) meshes |
| `output/` | results of the runs (not versioned) |

**[scripts/README.md](scripts/README.md) lists which script produces which result.**

## Getting started

Requires Julia 1.11 or later (developed with 1.12).

```julia
julia> import Pkg; Pkg.activate("."); Pkg.instantiate()
```

The packages `VoronoiMeshes`, `TensorsLite`, `TensorsLiteGeometry` and `ImmutableVectors` are not registered; `Project.toml` takes them from their repositories (github.com/favba), at the commits used for the results. Newer versions changed some fields and do not run all the scripts.

Run the scripts **from the repository root**:

```
julia --project=. -t 4 scripts/stars/stars_cache.jl mesh3 mesh4     # builds and caches the three stars
julia --project=. scripts/barrier/green_charge_check.jl             # a check that needs no mesh
julia --project=. -t 4 scripts/stars/min_lambda.jl 2 4 mesh2        # smallest coefficients of a symmetric star exact on quadratic fields
```

`mesh7` (65536 cells, 19 MB) is not versioned, and the scripts that go up to mesh 7 need it: `meshes/mesh_generator_scvt.jl mesh6 mesh7` builds a mesh of this kind from `mesh6`. `scripts/barrier/gap_skew_more.jl` also needs the mesh family
`meshes_felipe`, which is in the branch `old-layout`.

## License

[MIT](LICENSE). The packages listed in `Project.toml` are installed from their own repositories and keep their own licences.
The earlier code of the project is in the branch `old-layout`; it contains a copy of VoronoiMeshes.jl, with its licence in that folder.
