# thesis_extract.jl -- copy statements + proofs from hodge_properties_note.tex into thesis_app/<label>.tex (used by hodge_thesis.tex)
#     julia scripts/archive/thesis_extract.jl
src = split(read(joinpath(@__DIR__, "hodge_properties_note.tex"), String), "\n"); out = joinpath(@__DIR__, "thesis_app"); mkpath(out)
const BS = "\\"
labels = ["prop:topo", "thm:barrier", "prop:plane", "cor:plane", "thm:local", "prop:p2local", "prop:p1build", "prop:p2build", "cor:lift",
          "prop:sweeps", "prop:energy", "prop:bendixson", "prop:helmholtz", "prop:coercive", "prop:dmp"]
for lab in labels
    i = findfirst(l -> occursin(BS * "label{$lab}", l) && any(occursin(BS * "begin{$e}", l) for e in ("proposition", "theorem", "corollary")), src)
    i === nothing && error("label not found: $lab")
    env = first(e for e in ("proposition", "theorem", "corollary") if occursin(BS * "begin{$e}", src[i]))
    j = findnext(l -> occursin(BS * "end{$env}", l), src, i)
    k = findnext(l -> !isempty(strip(l)), src, j + 1)
    if occursin(BS * "begin{proof}", src[k]); j = findnext(l -> occursin(BS * "end{proof}", l), src, k); end
    write(joinpath(out, replace(lab, ":" => "_") * ".tex"), join(src[i:j], "\n") * "\n")
    println(lab, ": lines ", i, "-", j)
end
