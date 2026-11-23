# Third-party code

Two pieces of this branch come from packages of Felipe A. V. de Bragança Alves, distributed under the MIT licence reproduced below:

| code | where | source |
|---|---|---|
| `trisk_W` (the TRiSK weights) | `scripts/shallow_water/sw_linear.jl` | ported from [VoronoiOperators.jl](https://github.com/favba/VoronoiOperators.jl) |
| `create_cell_linesegments_periodic` (plotting) | `meshes/mesh_generator.jl` | closely follows [VoronoiMeshes.jl](https://github.com/favba/VoronoiMeshes.jl) |

The packages listed in `Project.toml` are not part of this repository: each one is installed from its own repository and keeps its own licence.

```
MIT License

Copyright (c) 2024 Felipe A. V. de Bragança Alves <favbalves@gmail.com> and contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
