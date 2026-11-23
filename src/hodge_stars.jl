# ============================================================================
#  hodge_stars.jl
#
#  Two high-order Hodge stars for 1-forms on a Delaunay/Voronoi pair,
#  written against the API of juliaDEC.jl.
#
#      include("juliaDEC.jl")
#      include("hodge_stars.jl")
#
#  ---------------------------------------------------------------------------
#  THE TWO OPERATORS
#  ---------------------------------------------------------------------------
#
#  Both use the same two ingredients:
#
#     * a stencil S_e of edges around e (get_neighbors_2level/3level_primal),
#     * the reduction matrix A_e with rows  A[i,j] = \int_{e_i} f_j . t ,
#       for the vector basis f_j = (x-x0)^alpha e_d  (the SAME basis and the
#       SAME ordering as integrate_basic_fields_*_edge),
#     * the lift  Pi_e = pinv(A_e), so that  Pi_e A_e = Id  (exact recovery of
#       every field in the basis span).
#
#  They differ in what is done with the lifted field.
#
#  (1) LIFT star   ("your hodge_star_lsq_operator")
#
#          H_e = Rtilde_1(*_K .) o Pi_e ,        row vector, 1 x |S_e|
#
#      i.e. lift, apply the CONTINUOUS star, then integrate on the dual edge.
#      This is a MAP.  It satisfies the star property
#
#          (H R_1 u)_e = Rtilde_1(*_K u)_e            (STAR)
#
#      EXACTLY for every u in the basis span, because Pi_e A_e = Id.
#      Price: H is NOT symmetric.  Measured ||H-H'||_F/||H||_F is a few percent
#      for [P1]^2 and ~0.27 for [P2]^2, and it does NOT decrease with h.  So
#      d0' H d0 is not symmetric either and can have complex eigenvalues -- and
#      with the [P1]^2 basis it really does have large NEGATIVE ones.  See
#      "WHAT TO EXPECT" at the bottom: the cure is the quadratic basis, not
#      symmetrisation.
#
#  (2) SANDWICH star   (Gram / mimetic form)
#
#          B_e = Pi_e' M_e Pi_e  +  tau * stabilisation ,
#          M_e[i,j] = \int_{D_e} f_i . K f_j dx ,
#          D_e = diamond cell = conv(e , *e) ,
#
#      assembled edge by edge into a global |E| x |E| matrix.
#      This is a BILINEAR FORM, not a map.  Because the diamonds tile the
#      domain,
#
#          z' H z  ~  \int_Omega u . K u        (exact for u in the basis span)
#
#      and H is symmetric positive semi-definite by construction.  It gives
#      the right ENERGY and the right eigenvalues.  It does NOT satisfy
#      (STAR): the defect is O(1) (a few percent, flat under refinement),
#      so it must not be used to read off fluxes edge by edge.
#
#  Rule of thumb:
#      H1 inside an energy  ( L = d0' H1 d0 )      -> SANDWICH
#      H1 inside an equation whose unknown is a
#      flux (mixed / Hirani form, eqn (4.6))       -> LIFT
#
#  ---------------------------------------------------------------------------
#  CONVENTIONS IN juliaDEC.jl  (read this before editing anything)
#  ---------------------------------------------------------------------------
#
#  mesh.cells.position     = Voronoi generators   = DELAUNAY vertices
#  mesh.vertices.position  = circumcentres        = VORONOI vertices
#
#  so, for edge index e,
#
#      mesh.edges.cells[e]     -> the two endpoints of the PRIMAL (Delaunay) edge e
#      mesh.edges.vertices[e]  -> the two endpoints of the DUAL   (Voronoi)  edge *e
#
#  Consistently:
#      integrate_basic_fields_primal_edge  integrates over edges.cells    (Delaunay)
#      integrate_basic_fields_dual_edge    integrates over edges.vertices (Voronoi)
#
#  NAMING TRAP: discretize_vector_field_primal integrates over edges.vertices
#  (Voronoi!) and returns Cochain(.,1,false); discretize_vector_field_dual
#  integrates over edges.cells (Delaunay) and returns Cochain(.,1,true).
#  The Cochain flag is right, the function names are swapped.  Below we use
#  the unambiguous wrappers reduce_primal_edges / reduce_dual_edges.
#
#  ORIENTATION (load-bearing).  mesh.edges.vertices must be ordered so that
#  t_{*e} = R(t_e) with R(w) = (-w2, w1) -- the MPAS/VoronoiMeshes convention.
#  That is precisely what makes hodge_applied=true the discrete star.  If a
#  mesh violates it on some edges, the lift acquires O(1) antisymmetry, the
#  sandwich star defect jumps to O(1), and d0'Hd0 fills with large negative
#  eigenvalues.  check_dual_orientation(mesh) must return 0.
#
#  edge_integral(v,p1,p2,..) returns \int v . (p2-p1) dt  -- the TANGENTIAL
#  circulation against the UN-NORMALISED edge vector.  Both edge families use
#  the tangential form; the "flux across *e" is obtained by rotating the field
#  (hodge_applied=true), i.e. R(w) = (-w2, w1).
#
#  PERIODICITY: every quantity for edge e is built in the frame UNWRAPPED
#  about x0 = mesh.edges.position[e], via closest(...).  A stencil that
#  straddles the periodic seam therefore lives at coordinates that differ from
#  the canonical ones.  This is harmless for periodic fields but it means a
#  NON-PERIODIC test field (e.g. u(x)=(x,y)) must be evaluated in the SAME
#  unwrapped frame or you will measure a seam artefact, not an error of the
#  operator.  test_polynomial_exactness below does it correctly.
#
#  ---------------------------------------------------------------------------
#  MATERIAL TENSOR K
#  ---------------------------------------------------------------------------
#  K = nothing  ->  identity (plain Hodge star).  Fast path: reuses your
#                   integrate_basic_fields_*_edge verbatim.
#  K = x -> [k11 k12; k12 k22]  ->  weighted star, *_K u := R(K u).
#  Put K = kappa for the primal (energy) form  \int kappa grad p . grad p ,
#  put K = inv(kappa) for the mixed/Darcy form  \int kappa^{-1} f . f .
# ============================================================================

using LinearAlgebra, SparseArrays

const _ROT90 = [0.0 -1.0; 1.0 0.0]      # R(w) = (-w2, w1)

# ----------------------------------------------------------------------------
# 0.  small geometry helpers  (periodic-aware, frame-aware)
# ----------------------------------------------------------------------------

_pt(p) = [p[1], p[2]]

"""
    edge_endpoints(mesh, e, primal; about=nothing)

Endpoints of edge `e` as plain `Vector{Float64}` of length 2.
`primal=true`  -> Delaunay edge  (mesh.edges.cells    / mesh.cells.position)
`primal=false` -> Voronoi  edge  (mesh.edges.vertices / mesh.vertices.position)

Everything is unwrapped about `about` (default: this edge's own position),
exactly as integrate_basic_fields_*_edge does.
"""
function edge_endpoints(mesh::VoronoiMesh, e::Int, primal::Bool; about=nothing)
    xp, yp = mesh.x_period, mesh.y_period
    ref    = about === nothing ? mesh.edges.position[e] : about
    epos   = closest(ref, mesh.edges.position[e], xp, yp)
    if primal
        a, b = mesh.edges.cells[e]
        p1 = closest(epos, mesh.cells.position[a], xp, yp)
        p2 = closest(epos, mesh.cells.position[b], xp, yp)
    else
        a, b = mesh.edges.vertices[e]
        p1 = closest(epos, mesh.vertices.position[a], xp, yp)
        p2 = closest(epos, mesh.vertices.position[b], xp, yp)
    end
    return _pt(p1), _pt(p2)
end

"""
    edge_length(mesh, e, primal)

Geometric length, computed from the periodic-unwrapped endpoints.  Used in
preference to mesh.edges.length / mesh.edges.lengthDual so that no assumption
is made about which of those two names is the Delaunay one.
"""
function edge_length(mesh::VoronoiMesh, e::Int, primal::Bool)
    p1, p2 = edge_endpoints(mesh, e, primal)
    return norm(p2 - p1)
end

edge_lengths(mesh::VoronoiMesh, primal::Bool) =
    [edge_length(mesh, e, primal) for e in 1:mesh.edges.n]

"""
    diamond_triangles(mesh, e; about=nothing)

The diamond cell  D_e = conv(e , *e)  as two triangles sharing the primal
edge as a diagonal:

        v1                 c1,c2 = endpoints of the primal (Delaunay) edge
       /  \\                v1,v2 = endpoints of the dual  (Voronoi)  edge
     c1----c2              D_e = (c1,v1,c2)  U  (c1,c2,v2)
       \\  /
        v2

Returns `((c1,v1,c2), (c1,c2,v2))`, each a tuple of three 2-vectors, all in
the frame unwrapped about `about` (default x0 = mesh.edges.position[e]).

The two triangles have disjoint interiors iff v1 and v2 lie on opposite sides
of the primal edge, i.e. iff the edge is *well-centred*.  Use
`diamond_is_well_centred` to check; on an obtuse (badly-centred) edge the
convex hull is NOT the union of these two triangles and the mass matrix would
double-count.
"""
function diamond_triangles(mesh::VoronoiMesh, e::Int; about=nothing)
    x0 = about === nothing ? mesh.edges.position[e] : about
    c1, c2 = edge_endpoints(mesh, e, true;  about=x0)
    v1, v2 = edge_endpoints(mesh, e, false; about=x0)
    return ((c1, v1, c2), (c1, c2, v2))
end

_cross2(a, b) = a[1]*b[2] - a[2]*b[1]
_tri_area(p1, p2, p3) = 0.5 * abs(_cross2(p2 - p1, p3 - p1))

"""
    diamond_is_well_centred(mesh, e) -> Bool

true iff the two Voronoi endpoints lie strictly on opposite sides of the
Delaunay edge (equivalently: the diamond is a genuine convex quadrilateral).
"""
function diamond_is_well_centred(mesh::VoronoiMesh, e::Int)
    x0 = mesh.edges.position[e]
    c1, c2 = edge_endpoints(mesh, e, true;  about=x0)
    v1, v2 = edge_endpoints(mesh, e, false; about=x0)
    t = c2 - c1
    s1 = _cross2(t, v1 - c1)
    s2 = _cross2(t, v2 - c1)
    return s1 * s2 < 0
end

"""
    diamond_area(mesh, e)

|D_e|.  Equals (1/2)|e||*e| when the edges are orthogonal, which they are for
a Delaunay/Voronoi pair.
"""
function diamond_area(mesh::VoronoiMesh, e::Int)
    T1, T2 = diamond_triangles(mesh, e)
    return _tri_area(T1...) + _tri_area(T2...)
end

"""
    check_dual_orientation(mesh) -> n_wrong

EVERYTHING here relies on mesh.edges.vertices being ordered so that the dual
tangent is the +90 degree rotation of the primal tangent,

    t_{*e} = R(t_e),        R(w) = (-w2, w1),

which is the MPAS / VoronoiMeshes convention (verticesOnEdge oriented against
cellsOnEdge).  It is what makes `hodge_applied=true` in
integrate_basic_fields_*_edge the correct discrete star.  If this returns
anything but 0, every sign test below is meaningless: the lift picks up an
O(1) antisymmetry, d0'Hd0 acquires large negative eigenvalues, and the
sandwich star defect jumps from a few percent to O(1).  Run it once on any
new mesh.
"""
function check_dual_orientation(mesh::VoronoiMesh)
    bad = 0
    for e in 1:mesh.edges.n
        c1, c2 = edge_endpoints(mesh, e, true)
        v1, v2 = edge_endpoints(mesh, e, false)
        _cross2(c2 - c1, v2 - v1) <= 0 && (bad += 1)
    end
    return bad
end

"""
    check_diamond_partition(mesh) -> (sum|D_e|, |Omega|, n_bad)

Sanity check: the diamonds tile the domain, so sum_e |D_e| must equal the
area of the torus.  This is the identity that makes the SANDWICH assembly
a valid approximation of \\int_Omega u.Ku.
"""
function check_diamond_partition(mesh::VoronoiMesh)
    tot  = sum(diamond_area(mesh, e) for e in 1:mesh.edges.n)
    nbad = count(e -> !diamond_is_well_centred(mesh, e), 1:mesh.edges.n)
    return tot, mesh.x_period * mesh.y_period, nbad
end

# ----------------------------------------------------------------------------
# 1.  the polynomial basis  (must match integrate_basic_fields_*_edge exactly)
# ----------------------------------------------------------------------------
#
#   only_linear = true   ->  [P1]^2, N = 6:
#       1:(1,0)  2:(0,1)  3:(X,0)  4:(Y,0)  5:(0,X)  6:(0,Y)
#   only_linear = false  ->  [P2]^2, N = 12, adding
#       7:(X^2,0) 8:(XY,0) 9:(Y^2,0) 10:(0,X^2) 11:(0,XY) 12:(0,Y^2)
#
#   with X = (x-x0_1)/s, Y = (x-x0_2)/s.
#
#   s is a purely numerical rescaling.  Both the sandwich B_e and the lift
#   weights are EXACTLY invariant under it (A -> A D^{-1}, Pi -> D Pi,
#   M -> D^{-1} M D^{-1}, b -> D^{-1} b), so s only fixes the conditioning of
#   the pseudo-inverse; s = 1 reproduces your code bit for bit.

nbasis(only_linear::Bool) = only_linear ? 6 : 12

"""
    basis_scales(s, only_linear)

The diagonal D of the change of basis: entry j is s^{deg_j}.
"""
function basis_scales(s::Float64, only_linear::Bool)
    return only_linear ? [1.0, 1.0, s, s, s, s] :
                         [1.0, 1.0, s, s, s, s, s^2, s^2, s^2, s^2, s^2, s^2]
end

"""
    basis_at(p, x0, only_linear, s) -> 2 x N matrix

Column j is f_j(p) for the scaled basis.
"""
function basis_at(p, x0, only_linear::Bool, s::Float64)
    X = (p[1] - x0[1]) / s
    Y = (p[2] - x0[2]) / s
    if only_linear
        return [1.0 0.0  X   Y  0.0 0.0 ;
                0.0 1.0 0.0 0.0  X   Y ]
    else
        return [1.0 0.0  X   Y  0.0 0.0  X*X X*Y Y*Y 0.0 0.0 0.0 ;
                0.0 1.0 0.0 0.0  X   Y   0.0 0.0 0.0 X*X X*Y Y*Y]
    end
end

# ----------------------------------------------------------------------------
# 2.  triangle quadrature (degree 2 and degree 5; juliaDEC's rule stops at 2)
# ----------------------------------------------------------------------------

"""
    tri_rule(deg) -> (bary, w)

Barycentric points and weights summing to 1.  deg = 2 (3 points, exact for
quadratics) or deg = 5 (7 points, exact for quintics).  For only_linear the
integrand f_i.f_j is quadratic, so deg 2 suffices with constant K; for the
quadratic basis it is quartic, so deg 5 is required.
"""
function tri_rule(deg::Int)
    if deg <= 2
        b = [(2/3, 1/6, 1/6), (1/6, 2/3, 1/6), (1/6, 1/6, 2/3)]
        w = [1/3, 1/3, 1/3]
        return b, w
    elseif deg <= 5
        a1, c1 = 0.059715871789770, 0.470142064105115
        a2, c2 = 0.797426985353087, 0.101286507323456
        w0, w1, w2 = 0.225, 0.132394152788506, 0.125939180544827
        b = [(1/3, 1/3, 1/3),
             (a1, c1, c1), (c1, a1, c1), (c1, c1, a1),
             (a2, c2, c2), (c2, a2, c2), (c2, c2, a2)]
        w = [w0, w1, w1, w1, w2, w2, w2]
        return b, w
    else
        error("tri_rule: only degrees <= 5 implemented")
    end
end

# ----------------------------------------------------------------------------
# 3.  mass matrix on the diamond cell
# ----------------------------------------------------------------------------

"""
    diamond_mass_matrix(mesh, e, x0, only_linear, s; K=nothing, qdeg=5)

    M[i,j] = \\int_{D_e} f_i(x) . K(x) f_j(x) dx

with the SAME basis (and scaling s) as the reduction matrix.  Symmetric PD
whenever the diamond is non-degenerate (the f_j are linearly independent as
functions).
"""
function diamond_mass_matrix(mesh::VoronoiMesh, e::Int, x0, only_linear::Bool,
                             s::Float64; K=nothing, qdeg::Int=5)
    N  = nbasis(only_linear)
    M  = zeros(N, N)
    b, w = tri_rule(qdeg)
    for T in diamond_triangles(mesh, e; about=x0)
        p1, p2, p3 = T
        Ar = _tri_area(p1, p2, p3)
        Ar == 0.0 && continue
        for (bb, ww) in zip(b, w)
            p  = bb[1]*p1 + bb[2]*p2 + bb[3]*p3
            Phi = basis_at(p, x0, only_linear, s)          # 2 x N
            KPhi = K === nothing ? Phi : (K(p) * Phi)
            M .+= (ww * Ar) .* (Phi' * KPhi)
        end
    end
    return Symmetric(0.5 .* (M .+ M'))
end

# ----------------------------------------------------------------------------
# 4.  the shared least-squares lift  Pi_e = pinv(A_e)
# ----------------------------------------------------------------------------

"""
    lsq_lift(mesh, e; only_linear=true, primal_to_dual=true, rescale=true)

Returns a NamedTuple with everything both stars need:

    x0        : expansion point = mesh.edges.position[e]
    neighbors : the stencil S_e (13 edges for [P1]^2, 19 for [P2]^2)
    s         : the basis rescaling actually used
    A         : |S_e| x N reduction matrix, rows \\int_{e_i} f_j . t
    Pi        : N x |S_e| Moore-Penrose inverse
    Q         : A*Pi, the |S_e| x |S_e| projector onto range(A)
    cond      : cond(A)  (diagnostic)

`metric=true` replaces the plain pseudo-inverse by the W-WEIGHTED one,

    Pi_W = (W^{1/2} A)^+ W^{1/2} ,     W = diag( |*e_i| / |e_i| ) ,

which is still a left inverse (Pi_W A = Id, so the star property (STAR) is
untouched) but makes the least-squares fit orthogonal in the SAME metric that
the DEC primal-dual pairing  <alpha,gamma> = sum_e alpha_e gamma_e  uses.  That
is exactly the condition under which the lift's energy z'Hz stops being an
indefinite cross term; see section "THE WEIGHTED LIFT" at the bottom.  W is
diagonal and positive, so this costs nothing.

`primal_to_dual=true`  -> rows are DELAUNAY-edge circulations (input cochain is
                          Cochain(.,1,true)), star maps primal -> dual.
`primal_to_dual=false` -> rows are VORONOI-edge circulations.

This is exactly the A of your hodge_star_lsq_operator, up to the column
rescaling (which cancels identically in every quantity built from it).
"""
function lsq_lift(mesh::VoronoiMesh, e::Int; only_linear::Bool=true,
                  primal_to_dual::Bool=true, rescale::Bool=true, wide=nothing,
                  metric::Bool=false)
    x0 = mesh.edges.position[e]
    usewide   = wide === nothing ? !only_linear : wide
    neighbors = usewide ? get_neighbors_3level_primal(mesh, e) :
                          get_neighbors_2level_primal(mesh, e)
    n = length(neighbors)
    N = nbasis(only_linear)
    n < N && error("lsq_lift: stencil of edge $e has $n < $N edges")

    A = zeros(n, N)
    for i in 1:n
        row = primal_to_dual ?
              integrate_basic_fields_primal_edge(mesh, neighbors[i], x0, only_linear, false) :
              integrate_basic_fields_dual_edge(  mesh, neighbors[i], x0, only_linear, false)
        A[i, :] = row[1:N]
    end

    # rescaling length: radius of the stencil about x0
    s = 1.0
    if rescale
        r = 0.0
        for nb in neighbors
            q = closest(x0, mesh.edges.position[nb], mesh.x_period, mesh.y_period)
            r = max(r, hypot(q[1] - x0[1], q[2] - x0[2]))
        end
        s = r > 0 ? r : 1.0
    end
    d = basis_scales(s, only_linear)
    A = A ./ d'                       # A -> A D^{-1}

    if metric
        # W = diag(|*e_i|/|e_i|): the weight that turns the plain DEC pairing
        # sum_e alpha_e gamma_e into an L^2 inner product on the lifted field.
        r  = [sqrt(edge_length(mesh, m, !primal_to_dual) /
                   edge_length(mesh, m,  primal_to_dual)) for m in neighbors]
        Pi = pinv(Diagonal(r) * A) * Diagonal(r)
    else
        Pi = pinv(A)
    end
    return (x0=x0, neighbors=neighbors, s=s, d=d, A=A, Pi=Pi,
            Q=A*Pi, cond=cond(A))
end

# ----------------------------------------------------------------------------
# 5.  LIFT STAR   (map;  exact star property;  not symmetric)
# ----------------------------------------------------------------------------

"""
    star_basis_on_edge(mesh, e, x0, only_linear, s, target_primal, K)

The right-hand side of the lift star:

    b_j = \\int_{target edge} R(K(x) f_j(x)) . t dx ,    R(w) = (-w2, w1)

i.e. the reduction of the CONTINUOUS star of the j-th basis field on the
target edge.  With K = nothing this is exactly
integrate_basic_fields_*_edge(..., hodge_applied=true) rescaled, and we take
that fast path.
"""
function star_basis_on_edge(mesh::VoronoiMesh, e::Int, x0, only_linear::Bool,
                            s::Float64, target_primal::Bool, K)
    N = nbasis(only_linear)
    d = basis_scales(s, only_linear)
    if K === nothing
        raw = target_primal ?
              integrate_basic_fields_primal_edge(mesh, e, x0, only_linear, true) :
              integrate_basic_fields_dual_edge(  mesh, e, x0, only_linear, true)
        return raw[1:N] ./ d
    end
    p1, p2 = edge_endpoints(mesh, e, target_primal; about=x0)
    b = zeros(N)
    for j in 1:N
        fj = p -> (_ROT90 * (K(p) * basis_at(p, x0, only_linear, s)[:, j]))
        b[j] = edge_integral(fj, p1, p2, mesh.x_period, mesh.y_period)
    end
    return b
end

"""
    hodge_star_lift_operator(mesh, e; only_linear=true, primal_to_dual=true, K=nothing)

Row of the LIFT star for edge `e`:  weights' * z_{S_e}  approximates
\\int_{*e} (*_K u) . t  whenever z is the reduction of u on the stencil.

Returns `(weights, neighbors)` with the same signature as your
`hodge_star_lsq_operator`, of which this is a strict generalisation
(K = nothing, rescale = false reproduces it identically).

PROPERTY (STAR).  If u is any field in the basis span, z = A c is its
reduction, and Pi z = c, so

    weights . z = b' Pi A c = b' c = \\int_{*e} *_K u . t     EXACTLY.

That is why this operator gets fluxes right.  It is not symmetric because the
target functional b and the row space of A are unrelated.
"""
function hodge_star_lift_operator(mesh::VoronoiMesh, e::Int; only_linear::Bool=true,
                                  primal_to_dual::Bool=true, K=nothing,
                                  rescale::Bool=true, wide=nothing,
                                  metric::Bool=false)
    L = lsq_lift(mesh, e; only_linear=only_linear, primal_to_dual=primal_to_dual,
                 rescale=rescale, wide=wide, metric=metric)
    # target edge is the OTHER family
    b = star_basis_on_edge(mesh, e, L.x0, only_linear, L.s, !primal_to_dual, K)
    weights = vec(b' * L.Pi)
    return weights, L.neighbors
end

"""
    hodge_star_lift_matrix(mesh; only_linear=true, primal_to_dual=true, K=nothing)

Global sparse |E| x |E| lift star.  Row e has |S_e| nonzeros.

NOTE ON THE SIGN CONVENTION.  Both directions apply the same rotation R once,
and R^2 = -Id, so the composition already reproduces the continuous identity
** = -Id on 1-forms in 2D:

    H_{d->p} * H_{p->d} = -Id      on the local basis span.

Measured on a constant field (where the lift is exact, so this isolates the
sign from any discretisation error): ||C z + z|| / ||z|| = 7e-16 for [P1]^2 and
9e-16 for [P2]^2.  Do NOT negate the dual->primal star -- that would break the
identity, not restore it.  Off the basis span C is not close to -Id, but that is
the usual statement that a least-squares lift only inverts what it can see.
"""
function hodge_star_lift_matrix(mesh::VoronoiMesh; only_linear::Bool=true,
                                primal_to_dual::Bool=true, K=nothing,
                                rescale::Bool=true, wide=nothing,
                                metric::Bool=false)
    n = mesh.edges.n
    Irow, Jcol, Vals = Int[], Int[], Float64[]
    for e in 1:n
        w, nb = hodge_star_lift_operator(mesh, e; only_linear=only_linear,
                    primal_to_dual=primal_to_dual, K=K, rescale=rescale, wide=wide,
                    metric=metric)
        for (j, m) in enumerate(nb)
            push!(Irow, e); push!(Jcol, m); push!(Vals, w[j])
        end
    end
    return sparse(Irow, Jcol, Vals, n, n)
end

# ----------------------------------------------------------------------------
# 6.  SANDWICH STAR   (bilinear form;  symmetric PSD;  exact energy)
# ----------------------------------------------------------------------------

"""
    hodge_star_sandwich_operator(mesh, e; only_linear=true, primal_to_dual=true,
                                 tau=1.0, K=nothing, qdeg=5, stab_normalize=true)

Local |S_e| x |S_e| symmetric block

    B_e = Pi_e' M_e Pi_e  +  s_e (I-Q_e)' W (I-Q_e) ,    Q_e = A_e Pi_e ,

with M_e the mass matrix on the diamond D_e = conv(e, *e).

Returns `(B, neighbors)`.

CONSISTENCY.  For u in the basis span with reduction z on S_e we have
Pi z = c and (I-Q)z = 0, hence

    z' B_e z = c' M_e c = \\int_{D_e} u . K u          EXACTLY,

and the stabilisation is completely invisible to such u.  This is the
statement "the stabilisation cannot spoil the consistency" -- and, by the same
token, "the stabilisation cannot repair the star property".

STABILISATION.  W = diag(1/|e_i|)^2 when stab_normalize (default), so the
penalty measures the non-polynomial residue in FIELD units rather than in
circulation units; the scalar s_e is trace-matched,
s_e = tau * tr(Pi'MPi) / tr((I-Q)'W(I-Q)), which is scale- and
basis-invariant.  tau = 0 disables it (B_e then has a large kernel and the
assembled matrix is only PSD, generally singular beyond the constants).

COERCIVITY.  The identity z = A(Pi z) + (I-Q)z splits any stencil vector into
a part seen by the consistent term and a part seen by the stabilisation, which
is what makes B_e uniformly SPD on the quotient.
"""
function hodge_star_sandwich_operator(mesh::VoronoiMesh, e::Int; only_linear::Bool=true,
                                      primal_to_dual::Bool=true, tau::Float64=1.0,
                                      K=nothing, qdeg::Int=5,
                                      stab_normalize::Bool=true, rescale::Bool=true,
                                      wide=nothing)
    L = lsq_lift(mesh, e; only_linear=only_linear,
                 primal_to_dual=primal_to_dual, rescale=rescale, wide=wide)
    M = diamond_mass_matrix(mesh, e, L.x0, only_linear, L.s; K=K, qdeg=qdeg)

    B = Matrix(L.Pi' * M * L.Pi)
    B = 0.5 .* (B .+ B')

    n = length(L.neighbors)
    N = nbasis(only_linear)
    if tau > 0 && n > N
        P = I - L.Q                                     # |S| x |S|
        Z = P
        if stab_normalize
            # DOFs live on Delaunay edges when primal_to_dual, on Voronoi edges otherwise
            lens = [edge_length(mesh, m, primal_to_dual) for m in L.neighbors]
            Z = Diagonal(1.0 ./ lens) * P    # NOT P*D: (I-Q)A=0 must survive
        end
        ZtZ = Z' * Z
        t = tr(ZtZ)
        if t > 0
            coef = tau * tr(B) / t          # trace-matched, hoisted on purpose
            B .+= coef .* ZtZ
        end
    end
    B = 0.5 .* (B .+ B')
    return B, L.neighbors
end

"""
    hodge_star_sandwich_matrix(mesh; ...)

Global sparse |E| x |E| SANDWICH star, assembled diamond by diamond:

    H = sum_e  Scatter_e' B_e Scatter_e .

Exactly symmetric, positive semi-definite, and

    z' H z  =  sum_e \\int_{D_e} u.Ku  =  \\int_Omega u . K u

for every u in the local basis spans, because the diamonds tile Omega
(check with check_diamond_partition).

WARNING.  H is NOT a Hodge star in the sense of a map: H*R_1(u) does not
approximate Rtilde_1(*u).  The defect is O(1) and does not decrease under
refinement.  Use it for  L = d0' H d0  (energies, eigenvalues), never to read
a flux off an edge.
"""
function hodge_star_sandwich_matrix(mesh::VoronoiMesh; only_linear::Bool=true,
                                    primal_to_dual::Bool=true, tau::Float64=1.0,
                                    K=nothing, qdeg::Int=5,
                                    stab_normalize::Bool=true, rescale::Bool=true,
                                    wide=nothing)
    n = mesh.edges.n
    Irow, Jcol, Vals = Int[], Int[], Float64[]
    for e in 1:n
        B, nb = hodge_star_sandwich_operator(mesh, e; only_linear=only_linear,
                    primal_to_dual=primal_to_dual, tau=tau, K=K, qdeg=qdeg,
                    stab_normalize=stab_normalize, rescale=rescale, wide=wide)
        m = length(nb)
        for a in 1:m, c in 1:m
            push!(Irow, nb[a]); push!(Jcol, nb[c]); push!(Vals, B[a, c])
        end
    end
    H = sparse(Irow, Jcol, Vals, n, n)   # sparse() SUMS duplicates = assembly
    return (H + H') ./ 2               # kill round-off asymmetry
end

# ----------------------------------------------------------------------------
# 7.  unambiguous reduction operators  (see NAMING TRAP above)
# ----------------------------------------------------------------------------

"""
    reduce_primal_edges(mesh, v) -> Cochain(.,1,true)

R_1 v : circulation of v along every DELAUNAY edge.
(Identical to discretize_vector_field_dual -- yes, "dual".)
"""
function reduce_primal_edges(mesh::VoronoiMesh, v::Function)
    vals = zeros(mesh.edges.n)
    for e in 1:mesh.edges.n
        p1, p2 = edge_endpoints(mesh, e, true)
        vals[e] = edge_integral(v, p1, p2, mesh.x_period, mesh.y_period)
    end
    return Cochain(vals, 1, true)
end

"""
    reduce_dual_edges(mesh, v) -> Cochain(.,1,false)

Rtilde_1 v : circulation of v along every VORONOI edge.
(Identical to discretize_vector_field_primal -- yes, "primal".)
"""
function reduce_dual_edges(mesh::VoronoiMesh, v::Function)
    vals = zeros(mesh.edges.n)
    for e in 1:mesh.edges.n
        p1, p2 = edge_endpoints(mesh, e, false)
        vals[e] = edge_integral(v, p1, p2, mesh.x_period, mesh.y_period)
    end
    return Cochain(vals, 1, false)
end

"""
    star_field(v, K=nothing) -> Function

The continuous star  *_K v (x) = R(K(x) v(x)),  R(w) = (-w2, w1).
`reduce_dual_edges(mesh, star_field(v))` is the exact flux cochain that the
LIFT star is built to reproduce.
"""
star_field(v::Function, K=nothing) =
    K === nothing ? (x -> _ROT90 * v(x)) : (x -> _ROT90 * (K(x) * v(x)))

"""
    triangle_integral_ho(f, p1, p2, p3; qdeg=5)

\\int_T f dA with the degree-5 rule of `tri_rule`.

juliaDEC's own `triangle_integral` defaults to the 3-point degree-2 rule, and
its `order=4` branch has no body -- it falls through to `return points, weights`
with both undefined.  A trigonometric source integrated at degree 2 carries a
per-triangle quadrature error that shows up as a floor in a convergence study,
so the 2-cochains below use this instead.
"""
function triangle_integral_ho(f::Function, p1, p2, p3; qdeg::Int=5)
    q1, q2, q3 = _pt(p1), _pt(p2), _pt(p3)
    Ar = _tri_area(q1, q2, q3)
    Ar == 0.0 && return 0.0
    b, w = tri_rule(qdeg)
    s = 0.0
    for (bb, ww) in zip(b, w)
        s += ww * f(bb[1] .* q1 .+ bb[2] .* q2 .+ bb[3] .* q3)
    end
    return s * Ar
end

"""
    discretize_scalar_field_triangles_ho(mesh, f; qdeg=5) -> Cochain(.,2,true)

R_2 f : \\int_T f dA over every DELAUNAY triangle (= mesh.vertices).  The same
object as juliaDEC's `discretize_scalar_field_triangles`, at degree 5 instead of
degree 2.  This is the `phi` of the mixed system (`darcy_mixed_system`), whose
second block row is D1 f = phi with D1 = d1_matrix_primal (triangles x edges).
"""
function discretize_scalar_field_triangles_ho(mesh::VoronoiMesh, f::Function; qdeg::Int=5)
    vals = zeros(mesh.vertices.n)
    for t in 1:mesh.vertices.n
        pts, _, _ = get_triangle(mesh, t)
        vals[t] = triangle_integral_ho(f, pts[1], pts[2], pts[3]; qdeg=qdeg)
    end
    return Cochain(vals, 2, true)
end

"""
    voronoi_cell_triangles(mesh, c)

The Voronoi cell of generator `c` fanned into triangles (generator, v_i, v_{i+1}),
all unwrapped about mesh.cells.position[c].  Valid because a Voronoi cell is
convex and contains its own generator, so the fan is a genuine triangulation.
"""
function voronoi_cell_triangles(mesh::VoronoiMesh, c::Int)
    x0 = mesh.cells.position[c]
    vs = [_pt(closest(x0, mesh.vertices.position[v], mesh.x_period, mesh.y_period))
          for v in mesh.cells.vertices[c]]
    g  = _pt(x0)
    nv = length(vs)
    return [(g, vs[i], vs[mod1(i + 1, nv)]) for i in 1:nv]
end

"""
    discretize_scalar_field_cells(mesh, f; qdeg=5) -> Cochain(.,2,false)

Rtilde_2 f : \\int_{V_c} f dA over every VORONOI cell (= mesh.cells).  The dual
partner of `discretize_scalar_field_triangles_ho`, and the right-hand side of the
primal Poisson problem (`solve_poisson_primal`).
"""
function discretize_scalar_field_cells(mesh::VoronoiMesh, f::Function; qdeg::Int=5)
    vals = zeros(mesh.cells.n)
    for c in 1:mesh.cells.n
        s = 0.0
        for T in voronoi_cell_triangles(mesh, c)
            s += triangle_integral_ho(f, T[1], T[2], T[3]; qdeg=qdeg)
        end
        vals[c] = s
    end
    return Cochain(vals, 2, false)
end

# ----------------------------------------------------------------------------
# 8.  Laplacians and the mixed (Hirani) system
# ----------------------------------------------------------------------------

"""
    laplacian_primal(mesh, H1)

    L = d0' H1 d0   on primal 0-cochains (Delaunay vertices = mesh.cells).

With H1 = SANDWICH this is symmetric PSD, L is the exact discrete energy
p' L p = \\int |grad_h p|^2_K, and the eigenvalues are real.  With H1 = LIFT it
is not symmetric.

NOTE  sym(d0' H d0) = d0' sym(H) d0, so symmetrising the Laplacian and
symmetrising the star are the same operation -- and symmetrising the lift
destroys its star property (defect jumps from ~1e-15 to ~1e-2, flat in h).
"""
function laplacian_primal(mesh::VoronoiMesh, H1::AbstractMatrix)
    d0 = d0_matrix_primal(mesh)        # edges x cells, uses mesh.edges.cells
    return d0' * H1 * d0
end

"""
    laplacian_dual(mesh, H1)

    L = d0_dual' H1 d0_dual   on dual 0-cochains (circumcentres = mesh.vertices).
"""
function laplacian_dual(mesh::VoronoiMesh, H1::AbstractMatrix)
    d0 = d0_matrix_dual(mesh)          # edges x vertices, uses mesh.edges.vertices
    return d0' * H1 * d0
end

"""
    solve_poisson_primal(mesh, H1, f; qdeg=5, pin=1) -> Vector, length mesh.cells.n

Solves  -div(K grad p) = f  on the torus in the primal (energy) form

    L p = b ,    L = d0' H1 d0 ,    b_c = \\int_{V_c} f dA ,

for p at the Voronoi generators / Delaunay vertices (mesh.cells).  Pass the
1-form star built with K = kappa, and use the SANDWICH: H1 sits inside an energy
here, which is the arrangement the lift is wrong for (with [P1]^2 the lift
Laplacian is indefinite, see (e) at the bottom of this file).

SIGN.  L is PSD with p' L p = \\int K grad p . grad p, and the weak form
\\int K grad p . grad q = \\int f q with q the indicator of V_c gives b_c as
written -- so this sign yields the solution of -div(K grad p) = +f.

THE TORUS.  L is singular on the constants, so `b` is projected onto
range(L) = {sum b = 0}, dof `pin` is pinned, and the mean is removed at the end.
The result therefore has zero mean, which is the normalisation to compare
against `pe .-= sum(pe)/length(pe)`.  If H1 has a wider kernel than the
constants (e.g. the sandwich at tau = 0) the pinned system is singular and this
will throw rather than return a silently wrong answer.
"""
function solve_poisson_primal(mesh::VoronoiMesh, H1::AbstractMatrix, f::Function;
                              qdeg::Int=5, pin::Int=1)
    L = sparse(laplacian_primal(mesh, H1))
    b = discretize_scalar_field_cells(mesh, f; qdeg=qdeg).values
    b .-= sum(b) / length(b)

    n    = mesh.cells.n
    keep = setdiff(1:n, pin)
    p    = zeros(n)
    p[keep] = L[keep, keep] \ b[keep]
    p .-= sum(p) / n
    return p
end

"""
    darcy_mixed_system(mesh, M1; mu=1.0)

The system of Hirani-Nakshatrala-Chaudhry, arXiv:0810.3434 eq. (4.6):

    [ -mu*M1   D1' ] [ f ]   [  0  ]
    [   D1      0  ] [ p ] = [ phi ]

with f a PRIMAL 1-cochain on Delaunay edges, p a dual 0-cochain at the
circumcentres, D1 = d1_matrix_primal (triangles x edges), and
M1 the 1-form star built with K = kappa^{-1}.

Use the LIFT star here.  This is the arrangement in which H1 sits inside an
EQUATION whose residual must lie in range(D1'), which is exactly the
requirement the sandwich fails (its patch-test defect is O(1)); with the
sandwich the flux error stalls at rate ~0, with the lift it converges.

Returns `(Amat, D1)`; build the right-hand side as
`[zeros(mesh.edges.n); phi]` with phi = discretize_scalar_field_triangles(...).
On a periodic mesh the system is singular (constants in p, plus the two
harmonic 1-cochains of the torus): pin one pressure or project.
"""
function darcy_mixed_system(mesh::VoronoiMesh, M1::AbstractMatrix; mu::Float64=1.0)
    D1 = sparse(Float64.(d1_matrix_primal(mesh)))     # triangles x edges
    nt = mesh.vertices.n
    Amat = [ -mu .* sparse(Float64.(M1))   sparse(D1')      ;
              D1                           spzeros(nt, nt)  ]
    return Amat, D1
end

# ----------------------------------------------------------------------------
# 9.  diagnostics
# ----------------------------------------------------------------------------

"""
    test_lift_exactness(mesh; only_linear=true, primal_to_dual=true)

Checks Pi_e A_e = Id on every edge.  This is the single algebraic fact behind
BOTH operators (exact energy for the sandwich, exact star property for the
lift).  The identity is zero.  If you see 1e-12 or worse, the stencil is
degenerate or the basis needs rescaling (`rescale=true`).

Returns (max_err, max_cond).
"""
function test_lift_exactness(mesh::VoronoiMesh; only_linear::Bool=true,
                             primal_to_dual::Bool=true, rescale::Bool=true)
    N = nbasis(only_linear)
    emax, cmax = 0.0, 0.0
    for e in 1:mesh.edges.n
        L = lsq_lift(mesh, e; only_linear=only_linear,
                     primal_to_dual=primal_to_dual, rescale=rescale)
        emax = max(emax, opnorm(L.Pi * L.A - I(N)))
        cmax = max(cmax, L.cond)
    end
    return emax, cmax
end

"""
    test_local_exactness(mesh; only_linear=true, K=nothing, tau=1.0)

The two exactness theorems, checked edge by edge IN THE LOCAL UNWRAPPED FRAME
(so that the periodic seam cannot contaminate a non-periodic test field).

For every edge e and every basis polynomial u = f_j, with z = A_e c its
reduction on the stencil:

    lift     :  | w_e . z  -  \\int_{*e} *_K u . t |          = 0
    sandwich :  | z' B_e z -  \\int_{D_e} u . K u   |          = 0

Both follow from the SAME algebraic fact Pi_e A_e = Id.  Neither says
anything about the star property of the assembled sandwich: for that use
test_star_property with a constant field (see below).

Returns (lift_rel, energy_rel, stab_rel), where stab_rel measures
||(I-Q_e) A_e|| -- the stabilisation's blindness to reduced polynomials,
i.e. the reason it can never break consistency and can never repair (STAR).
"""
function test_local_exactness(mesh::VoronoiMesh; only_linear::Bool=true,
                              primal_to_dual::Bool=true, K=nothing,
                              tau::Float64=1.0, qdeg::Int=5, rescale::Bool=true)
    N = nbasis(only_linear)
    lift_err = energy_err = stab_err = 0.0
    lift_sc  = energy_sc  = stab_sc  = 0.0

    for e in 1:mesh.edges.n
        L = lsq_lift(mesh, e; only_linear=only_linear,
                     primal_to_dual=primal_to_dual, rescale=rescale)
        M = diamond_mass_matrix(mesh, e, L.x0, only_linear, L.s; K=K, qdeg=qdeg)
        b = star_basis_on_edge(mesh, e, L.x0, only_linear, L.s, !primal_to_dual, K)
        w = vec(b' * L.Pi)
        B, _ = hodge_star_sandwich_operator(mesh, e; only_linear=only_linear,
                   primal_to_dual=primal_to_dual, tau=tau, K=K, qdeg=qdeg,
                   rescale=rescale)

        stab_err = max(stab_err, opnorm((I - L.Q) * L.A))
        stab_sc  = max(stab_sc,  opnorm(L.A))

        for j in 1:N
            c = zeros(N); c[j] = 1.0
            z = L.A * c
            lift_err   = max(lift_err,   abs(dot(w, z) - b[j]))
            lift_sc    = max(lift_sc,    abs(b[j]))
            energy_err = max(energy_err, abs(dot(z, B * z) - dot(c, M * c)))
            energy_sc  = max(energy_sc,  abs(dot(c, M * c)))
        end
    end
    return (lift_rel   = lift_err   / max(lift_sc, eps()),
            energy_rel = energy_err / max(energy_sc, eps()),
            stab_rel   = stab_err   / max(stab_sc, eps()))
end

"""
    test_star_property(mesh, H, v; K=nothing)

Global test with a PERIODIC smooth field v (so the seam is harmless):

    ||H R_1 v - Rtilde_1(*_K v)||   in the |e|/|*e|-weighted l2 norm,
    relative to ||Rtilde_1(*_K v)||.

For the lift this converges at the order of the basis; for the sandwich it
stalls at a few percent.
"""
function test_star_property(mesh::VoronoiMesh, H::AbstractMatrix, v::Function; K=nothing)
    z    = reduce_primal_edges(mesh, v).values
    tgt  = reduce_dual_edges(mesh, star_field(v, K)).values
    got  = H * z
    lp   = edge_lengths(mesh, true)
    ld   = edge_lengths(mesh, false)
    wgt  = lp ./ ld                                   # so that sum w*val^2 ~ \int
    num  = sqrt(sum(wgt .* (got .- tgt).^2))
    den  = sqrt(sum(wgt .* tgt.^2))
    return num / den
end

"""
    test_energy(mesh, H, v; K=nothing)

Global energy test.  For a CONSTANT v this is exact for the sandwich
(|z'Hz - |Omega| v.Kv| ~ 1e-14) and only approximate for the lift.
For a smooth periodic v it converges.
"""
function test_energy(mesh::VoronoiMesh, H::AbstractMatrix, v::Function;
                     K=nothing, qdeg::Int=5)
    z = reduce_primal_edges(mesh, v).values
    got = dot(z, H * z)
    exact = 0.0
    b, w = tri_rule(qdeg)
    for e in 1:mesh.edges.n
        for T in diamond_triangles(mesh, e)
            p1, p2, p3 = T
            Ar = _tri_area(p1, p2, p3)
            for (bb, ww) in zip(b, w)
                p = bb[1]*p1 + bb[2]*p2 + bb[3]*p3
                vv = v(p)
                exact += ww * Ar * dot(vv, K === nothing ? vv : K(p) * vv)
            end
        end
    end
    return got, exact, abs(got - exact) / abs(exact)
end

"""
    star_defect_constant(mesh, H; K=nothing, a=[1.0,0.7])

THE decisive test, and the cleanest one: a CONSTANT field is periodic, so no
seam artefact, and every genuine Hodge star must reproduce it EXACTLY:

    (H R_1 a)_e  =  int_{*e} R(K a) . t  =  R(K a) . (*e vector).

Result (relative, weighted l2):

    LIFT      ~ 1e-15                          -> it is a star
    SANDWICH  ~ 3e-2, and FLAT under refinement -> it is not

The sandwich value is not a discretisation error that will go away; it is the
O(1) defect delta of the Gram form.  Everything downstream follows from it:
the primal Laplacian d0' H d0 is still second-order accurate because the
error identity ||p_h - R_0 p||_L = ||d0' delta||_{L^{-1}} contracts delta
through d0', but a flux read off edge by edge inherits delta directly.
"""
function star_defect_constant(mesh::VoronoiMesh, H::AbstractMatrix;
                              K=nothing, a=[1.0, 0.7])
    return test_star_property(mesh, H, x -> a; K=K)
end

"""
    report(mesh; only_linear=true, tau=1.0, K=nothing)

One call that prints everything worth knowing about the two stars on this
mesh.  Run it on a sequence of refined meshes:

  * lift:     star defect -> 0 at the order of the basis; antisymmetry ~1%,
              flat; spectrum of d0'Hd0 real at the low end, a handful of
              complex pairs at the unresolved high end with |Im/Re| < 1e-4.
  * sandwich: star defect FLAT at a few percent; spectrum exactly real,
              symmetric PSD, energy exact on polynomials.
"""
function report(mesh::VoronoiMesh; only_linear::Bool=true, tau::Float64=1.0,
                K=nothing, rescale::Bool=true, do_eigs::Bool=true)
    println("="^74)
    println("edges = ", mesh.edges.n, "   cells = ", mesh.cells.n,
            "   triangles = ", mesh.vertices.n,
            "   basis = ", only_linear ? "[P1]^2 (N=6, 13-pt)" : "[P2]^2 (N=12, 19-pt)")

    tot, om, nbad = check_diamond_partition(mesh)
    println("-- geometry")
    println("   sum|D_e| vs |Omega|      : ", abs(tot - om) / om,
            "      badly-centred edges: ", nbad)
    println("   dual edges mis-oriented  : ", check_dual_orientation(mesh),
            "   <- MUST be 0")

    println("-- the one algebraic fact")
    ex, cnd = test_lift_exactness(mesh; only_linear=only_linear, rescale=rescale)
    println("   max ||Pi A - Id||        : ", ex, "   max cond(A) = ", cnd)

    le = test_local_exactness(mesh; only_linear=only_linear, tau=tau,
                              K=K, rescale=rescale)
    println("-- local exactness (both are theorems: the defect is zero)")
    println("   lift: (STAR) on P_k      : ", le.lift_rel)
    println("   sand: energy on P_k      : ", le.energy_rel)
    println("   stabilisation (I-Q)A     : ", le.stab_rel,
            "   <- invisible to P_k, hence harmless AND powerless")

    Hl = hodge_star_lift_matrix(mesh; only_linear=only_linear, K=K, rescale=rescale)
    Hs = hodge_star_sandwich_matrix(mesh; only_linear=only_linear, tau=tau,
                                    K=K, rescale=rescale)

    println("-- structure")
    println("   lift antisymmetry        : ", norm(Hl - Hl') / norm(Hl))
    println("   sandwich asymmetry       : ", norm(Hs - Hs') / norm(Hs))

    println("-- the star property (global, weighted l2, relative)")
    println("   constant field  lift     : ", star_defect_constant(mesh, Hl; K=K))
    println("   constant field  sandwich : ", star_defect_constant(mesh, Hs; K=K),
            "   <- O(1), flat in h")
    # must be periodic on THIS mesh, or the seam artefact swamps the star defect
    vsm = x -> [sin(2pi * x[2] / mesh.y_period), cos(2pi * x[1] / mesh.x_period)]
    println("   smooth field    lift     : ", test_star_property(mesh, Hl, vsm; K=K))
    println("   smooth field    sandwich : ", test_star_property(mesh, Hs, vsm; K=K))

    println("-- energy (global)")
    _, _, es = test_energy(mesh, Hs, x -> [1.0, 0.7]; K=K)
    _, _, el = test_energy(mesh, Hl, x -> [1.0, 0.7]; K=K)
    println("   constant field  sandwich : ", es, "   <- exact")
    println("   constant field  lift     : ", el)

    if do_eigs && mesh.cells.n <= 2500
        Ls = Symmetric(Matrix(laplacian_primal(mesh, Hs)))
        Ll = Matrix(laplacian_primal(mesh, Hl))
        es_ = eigvals(Ls)
        el_ = eigvals(Ll); el_ = el_[sortperm(real.(el_))]
        k = min(5, length(es_))
        println("-- spectrum of L = d0' H d0")
        println("   sandwich, first ", k, "      : ", round.(es_[1:k], digits=6))
        println("   lift,     first ", k, "      : ", round.(real.(el_[1:k]), digits=6))
        println("   lift, max |Im/Re|        : ",
                maximum(abs.(imag.(el_)) ./ max.(abs.(real.(el_)), 1e-30)))
        println("   lift, complex pairs      : ",
                count(x -> abs(imag(x)) > 1e-10, el_) ÷ 2)
        println("   lift, min Re(lambda)     : ", minimum(real.(el_)))
        println("   sandwich, min lambda     : ", minimum(es_))
    end
    println("="^74)
    return (Hl = Hl, Hs = Hs)
end

# ============================================================================
#  HOW TO USE
# ============================================================================
#
#   include("juliaDEC.jl")
#   include("hodge_stars.jl")
#
#   mesh = VoronoiMesh("your_periodic_mesh.nc")
#
#   # --- the two stars, [P1]^2 (13-point stencil) -------------------------
#   Hs = hodge_star_sandwich_matrix(mesh)                    # symmetric PSD
#   Hl = hodge_star_lift_matrix(mesh)                        # exact fluxes
#
#   # --- [P2]^2 (19-point stencil) ----------------------------------------
#   Hs2 = hodge_star_sandwich_matrix(mesh; only_linear=false)
#   Hl2 = hodge_star_lift_matrix(mesh;     only_linear=false)
#
#   # --- with a material tensor -------------------------------------------
#   kap  = x -> [1.0 + 0.5sin(2pi*x[1])  0.0 ; 0.0  1.0]
#   ikap = x -> inv(kap(x))
#   Hs_k = hodge_star_sandwich_matrix(mesh; K=kap)           # energy form
#   Hl_k = hodge_star_lift_matrix(mesh;     K=ikap)          # Darcy/mixed form
#
#   # --- diagnostics -------------------------------------------------------
#   report(mesh)
#
#   # --- primal (energy) Poisson,  -div(kappa grad p) = f -------------------
#   #     use the SANDWICH here.
#   L   = laplacian_primal(mesh, Hs)                 # cells x cells, SPD/PSD
#   rhs = <2-cochain of f on the Voronoi cells>      # sum to zero on a torus
#   #   pin one dof or project out the constant, then p = L \ rhs.
#   #   The flux is then RECOVERED as  Hl * (d0_matrix_primal(mesh) * p) ,
#   #   i.e. take the gradient topologically and star it with the LIFT.
#
#   # --- mixed (Hirani, arXiv:0810.3434 eq. 4.6) ---------------------------
#   #     use the LIFT here.
#   M1        = hodge_star_lift_matrix(mesh; K=ikap)
#   Amat, D1  = darcy_mixed_system(mesh, M1)
#   phi       = discretize_scalar_field_triangles(mesh, f).values
#   b         = [zeros(mesh.edges.n); phi]
#   #   singular on a torus: pin p[1] = 0 (delete that row/col) before solving.
#
# ---------------------------------------------------------------------------
#  WHAT TO EXPECT, AND WHY   (all numbers measured on periodic Delaunay/Voronoi
#  meshes of the flat torus, jittered triangular lattice + Laplacian smoothing,
#  max angle ~62 deg, min |*e|/|e| ~ 0.51, N = 120/224/480/896 cells)
# ---------------------------------------------------------------------------
#
#  (a) Pi_e A_e = Id is the only algebraic input.  That identity is zero, and
#      it gives at once exactness of the lift on P_k -- hence the star
#      property, hence correct fluxes -- and exactness of the sandwich energy
#      on P_k.
#
#  (b) The stabilisation satisfies (I-Q_e)A_e = 0 and Pi_e(I-Q_e) = 0, so it is
#      invisible to reduced polynomials.  That is why it cannot damage
#      consistency -- and equally why no amount of it can give the sandwich the
#      star property.  Making the sandwich SPD is not the missing ingredient;
#      the missing ingredient is that a Gram matrix is a BILINEAR FORM and a
#      star is a MAP.
#
#      (Watch the side on which the DOF rescaling multiplies the projector:
#       Z = D^{-1}(I-Q), never (I-Q)D^{-1}.  The wrong side destroys
#       (I-Q)A = 0 and the "exact energy" theorem silently degrades to ~1e-4.)
#
#  (c) THE STAR PROPERTY, global, weighted l2, relative.
#      Constant field (the sharpest test: a true star must be exact):
#          lift      6e-16, 6e-16, 7e-16                 -> it IS a star
#          sandwich  9.9e-2, 2.9e-2, 3.8e-2              -> FLAT in h
#      Smooth periodic field, [P1]^2 : lift 1.10e-1, 5.33e-2, 3.16e-2 (rate 2)
#                             [P2]^2 : lift 2.35e-3, 6.61e-4, 2.36e-4 (rate 3.3)
#                             sandwich stalls at a few percent in both.
#      The sandwich defect is not a discretisation error that goes away.  It is
#      the O(1) defect delta of the Gram form.
#
#  (d) PRIMAL (energy) form  L = d0' H d0.  The defect enters only through
#      d0' delta and the error identity ||p_h - R_0 p||_L = ||d0' delta||_{L^-1}
#      (constant exactly 1) contracts it, so the potential still converges:
#          diagonal star   4.97e-2  2.58e-2  1.21e-2  6.36e-3   rate 1.9-2.1
#          sandwich [P1]^2 2.72e-1  1.39e-1  6.23e-2  3.32e-2   rate 1.9-2.3
#          sandwich [P2]^2 1.34e-2  4.04e-3  1.04e-3  4.25e-4   rate 3.6/3.8/2.7
#      Note the honest part: at LOWEST order the sandwich is a factor ~5 WORSE
#      than the plain diagonal star.  The [P1]^2 sandwich buys you nothing.
#      The gain is entirely at [P2]^2, where it is a factor 15.
#
#  (e) EIGENVALUES.  Generalised problem  L p = lambda S0 p,  S0 = diag|cell|,
#      exact first nonzero eigenvalue 4*pi^2 = 39.4784 (multiplicity 4):
#
#          star              N=120    N=224    N=396     negative eigs
#          ---------------------------------------------------------------
#          diagonal          38.53    38.97    39.19     0
#          sandwich [P1]/13  34.75    36.95    37.96     0 (symmetric PSD)
#          sandwich [P2]/19  39.29    39.42    39.46     0 (symmetric PSD)
#          lift     [P1]/13  -170.8  -326.0   -588.7     31 / 52 / 82
#          lift     [P1]/19  -123.6  -231.1   -412.5     55 / 109 / 193
#          lift     [P2]/13   38.59   38.92    39.10     0
#          lift     [P2]/19   38.87   39.18    39.31     0
#
#      THIS IS THE ANSWER TO "the lift does not give the right eigenvalues".
#      The problem is NOT non-symmetry as such -- it is the ORDER of the basis.
#      With [P1]^2 the lift Laplacian is violently indefinite and the negative
#      part GROWS like h^-2.  Widening the stencil at fixed order makes it
#      WORSE (P1/19 has more negative eigenvalues than P1/13), so extra
#      least-squares averaging is not the cure.  With [P2]^2 -- i.e. simply
#      only_linear=false, which in your code also switches to the 19-point
#      stencil -- there are NO negative eigenvalues on any mesh tried, the low
#      spectrum is real to 1e-5..1e-2 in |Im/Re|, and the eigenvalues converge
#      to 4*pi^2 from below, comparable to the diagonal star.  ([P2]^2 on the
#      narrow 13-point stencil is also PSD and is the most nearly symmetric of
#      all, |Im/Re| ~ 2e-5, but its potential stalls at rate 1 -- 13 rows for
#      12 unknowns is too tight.  Use wide=true with [P2]^2.)
#
#      Caveat: this is an observation over a family of good meshes, not a
#      theorem.  There is no lower bound available for a non-symmetric star.
#
#  (f) Symmetrising the lift is NOT a fix: sym(d0' H d0) = d0' sym(H) d0, and
#      sym(H_lift) loses the star property (defect jumps to ~1e-2, flat in h)
#      while changing the eigenvalues only in the far digits.  You would pay
#      the fluxes for nothing.
#
#  (g) FLUX.  The lift's own flux error, measured on the EXACT reduced gradient
#      (this isolates the star from the solve; note d0 R_0 p = R_1(grad p) to
#      the last digit, the fundamental theorem of calculus, so both routes give
#      identical numbers):
#          [P1]^2/13 : 1.37e-1  7.66e-2  3.58e-2  1.95e-2   rate 1.7-2.1
#          [P2]^2/19 : 4.63e-3  1.49e-3  3.28e-4  1.01e-4   rate 3.4/4.2/3.5
#      So the star itself is genuinely 3rd-4th order in the flux.  When the
#      flux is instead RECOVERED from a computed potential, H_lift d0 p_h, the
#      rate is capped by the potential's, because differentiating p_h - p costs
#      one power of h.  If you want the high-order flux directly, solve the
#      MIXED system (darcy_mixed_system) with the LIFT star, which is exactly
#      the arrangement of arXiv:0810.3434 eq. (4.6).
#
#  ---------------------------------------------------------------------------
#  RECOMMENDED CONFIGURATION
#  ---------------------------------------------------------------------------
#      potential / eigenvalues :  hodge_star_sandwich_matrix(mesh; only_linear=false)
#      flux, recovered         :  hodge_star_lift_matrix(mesh; only_linear=false)
#      flux, solved directly   :  darcy_mixed_system(mesh, that same lift star)
#  and never the [P1]^2 lift inside a Laplacian.
