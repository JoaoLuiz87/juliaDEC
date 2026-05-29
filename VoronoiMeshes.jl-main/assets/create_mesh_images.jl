using LinearAlgebra
using TensorsLite
using DelaunayTriangulation
using VoronoiMeshes
using LaTeXStrings
using CairoMakie
using Printf

function main()
    xp = 2.0
    yp = 1.0
    cm = 96 / 2.54

    local function den(x)
        return 64.5 - 63.5*cos((x ⋅ 𝐣)*2pi/yp)
    end

    ref_iters = (1:180, 1:160, 1:160, 1:160)
    new_diagram = VoronoiDiagram(45, xp, yp; density=den, max_iter = 0)
    fn = 0
    ref_level = 1

    l_iter = 0
    title = L"Density function: $\rho\left(\mathbf{x}\right) = 64.5 - 63.5\cos\left(2 \pi x_y\right)$\\Refinement level = %$ref_level\\Lloyd iteration = %$l_iter"
    fig = Figure(size=(19*cm, 14*cm), fontsize=2*32/3)
    ax = Axis(fig[1,1], title=title, aspect=DataAspect(), xgridvisible=false, ygridvisible=false, limits=((-0.1, 2.1), (-0.1, 1.1)))
    plotdiagram!(ax, new_diagram)
    CairoMakie.save(string("figure_",@sprintf("%04i",fn),".png"), fig, rasterize=true, px_per_unit=212/96)

    for _ = ref_iters[ref_level]
        l_iter += 1
        new_diagram = VoronoiDiagram(new_diagram.generators, xp, yp; density=den, max_iter=1)
        fn +=1

        title = L"Density function: $\rho\left(\mathbf{x}\right) = 64.5 - 63.5\cos\left(2 \pi x_y\right)$\\Refinement level = %$ref_level\\Lloyd iteration = %$l_iter"

        fig = Figure(size=(19*cm, 14*cm), fontsize=2*32/3)
        ax = Axis(fig[1,1], title=title, aspect=DataAspect(), xgridvisible=false, ygridvisible=false, limits=((-0.1, 2.1), (-0.1, 1.1)))
        plotdiagram!(ax, new_diagram)
        CairoMakie.save(string("figure_",@sprintf("%04i",fn),".png"), fig, rasterize=true, px_per_unit=212/96)
    end

    old_diagram = new_diagram

    for _ in 1:3
        ref_level += 1
        new_diagram = VoronoiDiagram(vcat(old_diagram.generators, old_diagram.vertices), xp, yp; density=den, max_iter=0)

        l_iter = 0
        title = L"Density function: $\rho\left(\mathbf{x}\right) = 64.5 - 63.5\cos\left(2 \pi x_y\right)$\\Refinement level = %$ref_level\\Lloyd iteration = %$l_iter"

        for __ in 1:10
            fn +=1
            fig = Figure(size=(19*cm, 14*cm), fontsize=2*32/3)
            ax = Axis(fig[1,1], title=title, aspect=DataAspect(), xgridvisible=false, ygridvisible=false, limits=((-0.1, 2.1), (-0.1, 1.1)))
            plotdiagram!(ax, old_diagram)
            plotdiagram!(ax, new_diagram, color=:seagreen3)
            CairoMakie.save(string("figure_",@sprintf("%04i",fn),".png"), fig, rasterize=true, px_per_unit=212/96)
        end

        fn +=1
        fig = Figure(size=(19*cm, 14*cm), fontsize=2*32/3)
        ax = Axis(fig[1,1], title=title, aspect=DataAspect(), xgridvisible=false, ygridvisible=false, limits=((-0.1, 2.1), (-0.1, 1.1)))
        plotdiagram!(ax, new_diagram)
        CairoMakie.save(string("figure_",@sprintf("%04i",fn),".png"), fig, rasterize=true, px_per_unit=212/96)

        for i = ref_iters[ref_level]
            l_iter += 1
            new_diagram = VoronoiDiagram(new_diagram.generators, xp, yp; density=den, max_iter=1)
            fn +=1

            title = L"Density function: $\rho\left(\mathbf{x}\right) = 64.5 - 63.5\cos\left(2 \pi x_y\right)$\\Refinement level = %$ref_level\\Lloyd iteration = %$l_iter"

            fig = Figure(size=(19*cm, 14*cm), fontsize=2*32/3)
            ax = Axis(fig[1,1], title=title, aspect=DataAspect(), xgridvisible=false, ygridvisible=false, limits=((-0.1, 2.1), (-0.1, 1.1)))
            plotdiagram!(ax, new_diagram)
            CairoMakie.save(string("figure_",@sprintf("%04i",fn),".png"), fig, rasterize=true, px_per_unit=212/96)
        end

        old_diagram = new_diagram
    end
end

main()
