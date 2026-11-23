# ============================================================================
#  coriolis_growth_power.jl -- largest growth rate of the linear shallow-water scheme with the consistent Coriolis operator P on finer meshes.
#  Power iteration with the Cayley map (implicit midpoint) of  B x' = K x,  B = diag(M0, I): the map amplifies a mode λ = r + iω by
#  ≈ 1 + r Δt/(1 + ω²Δt²/4), so it selects the fastest growing SLOW mode (the growing modes of P are slow, |Im λ| ≈ 0.1 f).  The eigenvalue is
#  read off by projecting the generator on span{x_n, x_{n+1}}.  Checked against the dense spectra of meshes 3 and 4.
#      julia -t 4 scripts/shallow_water/coriolis_growth_power.jl [meshes]     -> output/ts_data/coriolis_growth_power.txt
# ============================================================================
include(joinpath(@__DIR__, "sw_linear_three.jl"))
OUTP = String[]; sp(s) = (println(s); flush(stdout); push!(OUTP, s))
using Random; Random.seed!(1)
for mn in (isempty(ARGS) ? ["mesh3", "mesh4", "mesh5", "mesh6"] : ARGS)
    mesh = VoronoiMesh("meshes/$mn.nc"); S = cached_stars(mn; mesh); O0 = sw_ops(mesh, S["H_DEC"]); P = perp_p2(mesh, O0.W * O0.H)
    M0c = mass0_corrected(mesh) .* O0.L^2; MA = spdiagm(0 => O0.A); nc, ne = O0.nc, O0.ne; HD = Vector(diag(S["H_DEC"])); h = O0.L / sqrt(nc)
    for (st, Mlab, M) in (("H_DEC", "A", MA), ("H_P1", "A", MA), ("H_P2", "A", MA), ("H_P2", "M0c", M0c))
        O = sw_ops(mesh, S[st])
        K = [spzeros(nc, nc) H0J .* (O.D0t * O.H); -GJ .* O.D0 -FJ .* P]; B = [sparse(M) spzeros(nc, ne); spzeros(ne, nc) sparse(1.0I, ne, ne)]
        dt = 2.0e4; F = lu(B .- (dt / 2) .* K); R = B .+ (dt / 2) .* K; FB = lu(sparse(M))
        Lx(x) = (y = K * x; vcat(FB \ y[1:nc], y[nc+1:end]))
        x = randn(nc + ne); x[1:nc] .*= 1e-3; w = vcat(sqrt.(GJ .* O0.A), sqrt.(H0J .* HD)); nrm(x) = norm(w .* x); x ./= nrm(x)
        g = Float64[]; λold = 0.0im
        for k in 1:6000
            y = F \ (R * x); a = nrm(y); push!(g, log(a) / dt)
            if k % 500 == 0
                Z = hcat(w .* x, w .* (y ./ a)); G = hcat(w .* Lx(x), w .* Lx(y ./ a)); λ = eigvals(Z \ G); λm = λ[argmax(real.(λ))]
                done = abs(λm - λold) < 1e-4 * abs(λm); λold = λm; x = y ./ a; done && break
            else
                x = y ./ a
            end
        end
        v = x[nc+1:end]; ke = dot(v, HD .* v); ζ = O.D1 * v; kζ = sqrt(sum(abs2, ζ ./ sqrt.(O.AT)) / ke)
        sp(@sprintf("%s %-5s P, M0 = %-4s  max Re λ = %+.3e s⁻¹ (e-folding %.2f days)  |Im λ|/f = %.3f  vorticity wavelength %.2f h   [%d steps; mean growth of the last 200: %.3e]",
                    mn, st, Mlab, real(λold), 1 / real(λold) / 86400, abs(imag(λold)) / FJ, 2π / kζ / h, length(g), sum(g[end-199:end]) / 200))
    end
end
write(joinpath(TSDIR, "coriolis_growth_power" * (isempty(ARGS) ? "" : "_" * join(ARGS, "_")) * ".txt"), join(OUTP, "\n") * "\n")
