# VoronoiMeshes

[![Build Status](https://github.com/favba/VoronoiMeshes.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/favba/VoronoiMeshes.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/favba/VoronoiMeshes.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/favba/VoronoiMeshes.jl)

VoronoiMeshes.jl provides tools to create, inspect and save biperiodic planar and spherical Voronoi meshes (including centroidal/Lloyd methods and utilities to work with their dual Delaunay triangulations).

Main developer: Felipe A. V. de Bragança Alves <favbalves@gmail.com>

![Bi-periodic planar mesh creation](./assets/mesh_creation_15fps_748x551.avif)

## Install Guide 

Assuming Julia already installed (for instance via juliaup).

Using Julia version 1.11 or higher:
```julia
import Pkg
Pkg.activate(temp=true)
Pkg.add(url="https://github.com/favba/VoronoiMeshes.jl.git")

# Optional packages, for grid creation; plotting; import / export to NetCDF; and import / export to VTK.
Pkg.add("DelaunayTriangulation")
Pkg.add("GLMakie")
Pkg.add("NCDatasets")
Pkg.add("ReadVTK")
Pkg.add("WriteVTK")
```

If using Julia v1.10, then the unregistered dependencies must be explicitly installed before installing this package:
```julia
import Pkg
Pkg.activate(temp=true)
Pkg.add(url="https://github.com/favba/TensorsLite.jl.git")
Pkg.add(url="https://github.com/favba/TensorsLiteGeometry.jl.git")
Pkg.add(url="https://github.com/favba/VoronoiMeshes.jl.git")

# Optional packages, for grid creation; plotting; import / export to NetCDF; and import / export to VTK.
Pkg.add("DelaunayTriangulation")
Pkg.add("GLMakie")
Pkg.add("NCDatasets")
Pkg.add("ReadVTK")
Pkg.add("WriteVTK")
```
The package and the dependencies will be ported into registered packages soon.

## Developer's Guide 

With Julia v1.11 or higher is sufficient to simply clone the repository:

```bash
git clone git@github.com:favba/VoronoiMeshes.jl.git
```

The package and the dependencies will be ported into registered packages soon.
For developers using Julia v1.10, the environment must be explicitly constructed in the following way:

Assuming a common folder for the packages and the dependencies:
```bash
git clone git@github.com:favba/VoronoiMeshes.jl.git
git clone git@github.com:favba/TensorsLite.jl.git
git clone git@github.com:favba/TensorsLiteGeometry.jl.git
```

Create project (manifests) for `VoronoiMeshes` and its unregistered dependencies in the following order:

```bash
cd TensorsLite.jl
julia --project=. -e 'import Pkg; Pkg.instantiate(); Pkg.status()'

cd ../TensorsLiteGeometry.jl
julia --project=. -e 'import Pkg; Pkg.develop(path="../TensorsLite.jl"); Pkg.instantiate(); Pkg.status()'

cd ../VoronoiMeshes.jl
julia --project=. -e 'import Pkg; Pkg.develop([Pkg.PackageSpec(path="../TensorsLite.jl"), Pkg.PackageSpec(path="../TensorsLiteGeometry.jl")]); Pkg.instantiate(); Pkg.status()'
```

Create project (manifests) for 'test'
```bash
cd test

julia --project=. -e 'import Pkg; Pkg.develop([Pkg.PackageSpec(path="../../TensorsLite.jl"), Pkg.PackageSpec(path="../../TensorsLiteGeometry.jl"), Pkg.PackageSpec(path="../../VoronoiMeshes.jl")]); Pkg.add("DelaunayTriangulation"); Pkg.add("GeometryBasics"); Pkg.add("GLMakie"); Pkg.add("LinearAlgebra"); Pkg.add("VTKBase"); Pkg.add("ReadVTK"); Pkg.add("WriteVTK"); Pkg.add("NCDatasets"); Pkg.add("SmallCollections"); Pkg.add("Test"); Pkg.instantiate(); Pkg.status()'
```

The same manifest can be used for the examples folder.

## Quick usage guide


Creating a simple centroidal biperiodic planar Voronoi mesh from N random generators:

```julia 
import Pkg
Pkg.add("DelaunayTriangulation") 
using VoronoiMeshes, DelaunayTriangulation

# Create a centroidal Voronoi mesh with 20 cells on a 1.0 × 1.2 periodic domain
mesh = VoronoiMesh(20, 1.0, 1.2)
println(mesh)

#Mesh plotting
using GLMakie
plotmesh(mesh) #Plot the Voronoi mesh
plotdualmesh!(mesh) # Plot the dual triangular mesh on top of the previous plot
```

Create a mesh from a given set of generator points (`VecArray` of `x`,`y` coordinates):

```julia
using VoronoiMeshes, DelaunayTriangulation, TensorsLite

generators = VecArray(x = rand(10), y = rand(10))
mesh = VoronoiMesh(generators, 1.0, 1.0)
println(mesh)
```

Create a regular hexagonal planar mesh:

```julia
using DelaunayTriangulation, VoronoiMeshes

# Target x period = 1.0
# Target y period = 1.2
# cell spacing = 0.05
hexmesh = create_planar_hex_mesh(1.0, 1.2, 0.05)
```
