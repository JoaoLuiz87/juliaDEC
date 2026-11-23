# Scripts

Run everything from the repository root, for example `julia --project=. -t 4 scripts/stars/stars_cache.jl mesh3`. Results go to `output/ts_data` and `output/figures`.

## Which script produces what

When several scripts are listed, run them in this order: the later ones read the results of the earlier ones.

### Meshes and setting

| result | scripts |
|---|---|
| mesh table (defects, offsets, angles) | [`mesh_quality_check`](barrier/mesh_quality_check.jl) |
| offset of the SCVT meshes | [`story_offset`](barrier/story_offset.jl) |
| the four stencils; their rank; their radius | [`gap_stencil_figure`](stars/gap_stencil_figure.jl), [`p2_stencil_rank`](stars/p2_stencil_rank.jl), [`gap_stencil_radius`](stars/gap_stencil_radius.jl) |

### The barrier

| result | scripts |
|---|---|
| the barrier and the size of the coefficients | [`paper_runs`](elliptic/paper_runs.jl) (PART=E), [`p2_symmetric_check`](stars/p2_symmetric_check.jl), [`gap_barrier_figure`](barrier/gap_barrier_figure.jl) |
| skew of plane waves, exact formula | [`skew_fourier_check`](barrier/skew_fourier_check.jl) |
| skew against h | [`skew_split_check`](barrier/skew_split_check.jl), [`gap_skew_figure`](barrier/gap_skew_figure.jl) |
| skew on more centroidal families | [`lloyd_convergence_check`](barrier/lloyd_convergence_check.jl), [`gap_skew_more`](barrier/gap_skew_more.jl) |
| symmetric stars exact on quadratic fields | [`p2_symmetric_check`](stars/p2_symmetric_check.jl), [`p2_symmetric_refs`](stars/p2_symmetric_refs.jl), [`p2_symmetric_figures`](stars/p2_symmetric_figures.jl), [`paper_runs`](elliptic/paper_runs.jl) (PART=Y) |
| search for a positive symmetric star | [`p2_symmetric_sdp`](stars/p2_symmetric_sdp.jl) |
| homogeneous families | [`homogeneous_skew_check`](barrier/homogeneous_skew_check.jl), [`homogeneous_p2_check`](barrier/homogeneous_p2_check.jl) |
| smallest coefficients, certified | [`min_lambda`](stars/min_lambda.jl) |
| growth of the entries | [`report_entries`](stars/report_entries.jl), [`report_figures_row`](stars/report_figures_row.jl) |
| conditions degree by degree | [`compatibility_check`](barrier/compatibility_check.jl) |

### Locality

| result | scripts |
|---|---|
| property N in exact arithmetic | [`hypN_proof`](barrier/hypN_proof.jl) |
| H_P1 reacts to every generator, decay | [`locality_check`](barrier/locality_check.jl), [`story_realrow`](barrier/story_realrow.jl) |
| local symmetric star exact on conformal fields | [`conformal_star_check`](stars/conformal_star_check.jl) |
| balance over a patch | [`story_global`](barrier/story_global.jl), [`story_patch`](barrier/story_patch.jl) |
| short proof with Green's identity: example and mesh | [`green_charge_check`](barrier/green_charge_check.jl), [`green_charge_mesh`](barrier/green_charge_mesh.jl), [`report_figures_theory`](elliptic/report_figures_theory.jl) |

### The stars

| result | scripts |
|---|---|
| Hodge errors, bounds, asymmetry on meshes 1-7 | [`paper_runs`](elliptic/paper_runs.jl) (PART=E), [`paper_figures`](elliptic/paper_figures.jl) |
| the antisymmetric part carries the skew | [`nonsymmetric_check`](stars/nonsymmetric_check.jl) |
| H_P2 against the lift | [`lift_vs_p2_check`](stars/lift_vs_p2_check.jl) |

### Poisson, eigenvalues, Darcy

| result | scripts |
|---|---|
| errors on meshes 1-7 | [`paper_runs`](elliptic/paper_runs.jl) (PART=E), [`paper_figures`](elliptic/paper_figures.jl) |
| solutions | [`thesis_tests`](elliptic/thesis_tests.jl), [`story_figures_elliptic`](elliptic/story_figures_elliptic.jl) |
| eigenvalues | [`story_eigen`](elliptic/story_eigen.jl), [`story_figures_eigen`](elliptic/story_figures_eigen.jl) |
| complex eigenvalues of H_P2 | [`p2_spectrum_check`](stars/p2_spectrum_check.jl), [`paper_runs`](elliptic/paper_runs.jl) (PART=S), [`paper_figures`](elliptic/paper_figures.jl) |
| velocity of the Poisson problem, and the figures of the short report | [`report_runs`](elliptic/report_runs.jl), [`report_figures_elliptic`](elliptic/report_figures_elliptic.jl) |
| cost of construction and solution | [`paper_runs`](elliptic/paper_runs.jl) (PART=T) |
| Helmholtz splitting | [`star_properties_check`](stars/star_properties_check.jl) |

### Transport

| result | scripts |
|---|---|
| worked example | [`extrusion_explain`](transport/extrusion_explain.jl) |
| contractions against the definition | [`extrusion_definition_check`](transport/extrusion_definition_check.jl) |
| D L = L D in the discrete runs | [`commutation_run_check`](transport/commutation_run_check.jl) |
| density | [`extrusion_check`](transport/extrusion_check.jl), [`extrusion_cubic_check`](transport/extrusion_cubic_check.jl), [`story_transport`](transport/story_transport.jl), [`story_figures_transport`](transport/story_figures_transport.jl) |
| period T = 5; undershoots | [`extrusion_T5_check`](transport/extrusion_T5_check.jl), [`extrusion_bounds_check`](transport/extrusion_bounds_check.jl) |
| 1-forms | [`lie_testcases_check`](transport/lie_testcases_check.jl), [`extrusion_check`](transport/extrusion_check.jl), [`extrusion_cubic_check`](transport/extrusion_cubic_check.jl), [`extrusion_oneform_check`](transport/extrusion_oneform_check.jl), [`thesis_transport`](transport/thesis_transport.jl) |

### Shallow water

| result | scripts |
|---|---|
| three tests, three stars; energy | [`sw_linear_three`](shallow_water/sw_linear_three.jl) |
| ablation; solutions | [`sw_ablation`](shallow_water/sw_ablation.jl), [`sw_solutions_fig`](shallow_water/sw_solutions_fig.jl) |
| mass matrix | [`m0_barrier_check`](shallow_water/m0_barrier_check.jl), [`m0_growth_split`](shallow_water/m0_growth_split.jl), [`m0_sym_error_check`](shallow_water/m0_sym_error_check.jl) |
| Coriolis operator | [`coriolis_barrier_check`](shallow_water/coriolis_barrier_check.jl), [`coriolis_p_error_check`](shallow_water/coriolis_p_error_check.jl), [`coriolis_growth_modes`](shallow_water/coriolis_growth_modes.jl), [`coriolis_growth_power`](shallow_water/coriolis_growth_power.jl) |

## All scripts

### `stars` — Constructing and checking the Hodge stars

| script | purpose |
|---|---|
| [`commuting_check.jl`](stars/commuting_check.jl) | the commuting identity  H R_h u = R~_h *u  on the SCVT meshes. |
| [`conformal_star_check.jl`](stars/conformal_star_check.jl) | an explicit local symmetric star exact on constants, dilations and rotations |
| [`gap_stencil_figure.jl`](stars/gap_stencil_figure.jl) | fig_gap_stencils.png of paper/hodge_gap.tex (Section 2): the four stencils of one edge of SCVT mesh 3. |
| [`gap_stencil_radius.jl`](stars/gap_stencil_radius.jl) | Assumption (A3) of paper/hodge_gap.tex: every stencil, with the dual edges it involves, lies in |
| [`lift_vs_p2_check.jl`](stars/lift_vs_p2_check.jl) | the lift star of hodge_stars.jl (row e = b_eᵀ Π_e, Π_e = pinv(A_e)) against H_P2 |
| [`local_symmetric_check.jl`](stars/local_symmetric_check.jl) | how far do "symmetric" and "local" go together? |
| [`min_lambda.jl`](stars/min_lambda.jl) | the SMALLEST possible Λ = max_e (1//ẽ_e/) Σ_j /H_ej/ /e_j/ over ALL symmetric stars exact on [P_k]² on a given pattern (ring r). |
| [`nonsymmetric_check.jl`](stars/nonsymmetric_check.jl) | past the second-order ceiling with ONE unknown per edge by giving up exact symmetry |
| [`p2_spectrum_check.jl`](stars/p2_spectrum_check.jl) | do the complex eigenvalues of the H_P2 Laplacian go away under refinement? |
| [`p2_stencil_rank.jl`](stars/p2_stencil_rank.jl) | how many [P_k]^2 fields can a stencil tell apart?  (Section 2.2 of three_stars_report.tex) |
| [`p2_symmetric_check.jl`](stars/p2_symmetric_check.jl) | can H_P2 be built like H_P1, i.e. as the smallest SYMMETRIC correction of DEC exact on |
| [`p2_symmetric_figures.jl`](stars/p2_symmetric_figures.jl) | the figures of Sections 2.2 and 2.5 of three_stars_report.tex. |
| [`p2_symmetric_refs.jl`](stars/p2_symmetric_refs.jl) | reference curves for fig_ts_p2sym.png: H_DEC and H_P1 (exact on [P1]^2, 3-level stencil) |
| [`p2_symmetric_sdp.jl`](stars/p2_symmetric_sdp.jl) | is there a POSITIVE symmetric star exact on [P2]^2 on the 4-ring, on SCVT meshes 2-4? |
| [`report_entries.jl`](stars/report_entries.jl) | how the entries of a symmetric star exact on [P2]² grow: least-change star on the 4-ring, meshes 2-5 (largest off-diagonal entry relative to the |
| [`report_figures_row.jl`](stars/report_figures_row.jl) | fig_rep_row.png from ts_data/report_entries.jls (see report_entries.jl): one row of three stars on mesh 5 |
| [`star_properties_check.jl`](stars/star_properties_check.jl) | which properties of the Hodge star survive from H_DEC to H_P1 and H_P2, and what the |
| [`stars_cache.jl`](stars/stars_cache.jl) | build the three stars of three_stars.jl once per mesh and keep them on disk |
| [`three_stars.jl`](stars/three_stars.jl) | the three Hodge stars of three_stars_report.tex and the problems they are tested on. |

### `barrier` — The skew, the barrier, locality

| script | purpose |
|---|---|
| [`compatibility_check.jl`](barrier/compatibility_check.jl) | which degrees k admit a symmetric star exact on [P_k]^2 (Section 3.5 of |
| [`gap_barrier_figure.jl`](barrier/gap_barrier_figure.jl) | fig_gap_barrier.png for Section 3.2 of paper/hodge_gap.tex (Theorem barrier, Corollary size). |
| [`gap_skew_figure.jl`](barrier/gap_skew_figure.jl) | Figure fig_gap_skew.png of paper/hodge_gap.tex (Section 3.6). |
| [`gap_skew_more.jl`](barrier/gap_skew_more.jl) | Hypothesis S2 on more centroidal Voronoi meshes: is /s_h(u,v)/ ≥ c h^2 with c bounded away from 0? |
| [`green_charge_check.jl`](barrier/green_charge_check.jl) | the simple proof that a symmetric star exact on linear fields cannot be local. |
| [`green_charge_mesh.jl`](barrier/green_charge_mesh.jl) | the balance behind the non-locality theorem on SCVT mesh 4 (simple form, for the Laplacian L = D0ᵀ H D0): |
| [`homogeneous_p2_check.jl`](barrier/homogeneous_p2_check.jl) | on a replicated (homogeneous) family the skew is O(h^3), so the barrier no longer forbids a |
| [`homogeneous_skew_check.jl`](barrier/homogeneous_skew_check.jl) | is the h^2 skew of SCVT meshes a property of unstructured meshes, or of their irregularity? |
| [`hypN_proof.jl`](barrier/hypN_proof.jl) | a computer-assisted proof of Hypothesis N of barrier_hodge.tex (locality theorem) |
| [`lloyd_convergence_check.jl`](barrier/lloyd_convergence_check.jl) | how many Lloyd steps does one refinement of the SCVT family need? |
| [`locality_check.jl`](barrier/locality_check.jl) | can a symmetric star exact on [P1]^2 be built locally?  (barrier_paper.tex, Section "Locality") |
| [`mesh_quality_check.jl`](barrier/mesh_quality_check.jl) | quality of the SCVT family meshes/meshN.nc (and of any other mesh files given) |
| [`skew_fourier_check.jl`](barrier/skew_fourier_check.jl) | an exact formula for the skew of plane waves, and what Hypothesis S_2 means for the mesh |
| [`skew_split_check.jl`](barrier/skew_split_check.jl) | where does the skew of the primal-dual pairing live?  (barrier_paper.tex) |
| [`story_global.jl`](barrier/story_global.jl) | why symmetry makes a consistent star global: figure and numbers for hodge_story.tex |
| [`story_offset.jl`](barrier/story_offset.jl) | the offset of the SCVT meshes (hodge_story.tex, Chapter 2)  ->  fig_st_offset.png |
| [`story_patch.jl`](barrier/story_patch.jl) | the patch of the balance law on mesh 5 (hodge_story.tex, "Every symmetric star exact on linear fields is global") |
| [`story_realrow.jl`](barrier/story_realrow.jl) | why symmetry makes H_P1 global, on the REAL stencil (hodge_story.tex, Section "in detail") |

### `elliptic` — Poisson, eigenvalues, Darcy

| script | purpose |
|---|---|
| [`paper_figures.jl`](elliptic/paper_figures.jl) | new figures of paper/hodge_gap.tex |
| [`paper_runs.jl`](elliptic/paper_runs.jl) | the numbers of paper/hodge_barrier_paper.tex that were not computed before |
| [`report_figures_elliptic.jl`](elliptic/report_figures_elliptic.jl) | figures of hodge_report.tex, numerical part: |
| [`report_figures_theory.jl`](elliptic/report_figures_theory.jl) | figures of hodge_report.tex, theory part: |
| [`report_runs.jl`](elliptic/report_runs.jl) | runs for hodge_report.tex: the velocity (flux) of the Poisson problem, F = H D0 p_h against R̃⋆∇p, next to the nodal errors, meshes 1-7, |
| [`story_eigen.jl`](elliptic/story_eigen.jl) | eigenvalues of the Laplacian  D0ᵀ H D0 p = λ M0 p  on the unit torus, for H_DEC, H_P1, H_P2 |
| [`story_figures_eigen.jl`](elliptic/story_figures_eigen.jl) | Laplace eigenvalues for H_DEC, H_P1, H_P2 (from ts_data/story_eigen.jls)  ->  fig_st_eigen.png |
| [`story_figures_elliptic.jl`](elliptic/story_figures_elliptic.jl) | Poisson and Darcy errors of H_DEC, H_P1, H_P2 on ONE colour scale per row (from ts_data/thesis_tests.jls) |
| [`thesis_tests.jl`](elliptic/thesis_tests.jl) | more Poisson and Darcy solutions for hodge_thesis.tex |

### `transport` — Lie derivative and advection

| script | purpose |
|---|---|
| [`commutation_run_check.jl`](transport/commutation_run_check.jl) | D L = L D through the fully discrete runs (not only on matrices), with the right-hand sides of the paper: |
| [`extrusion_T5_check.jl`](transport/extrusion_T5_check.jl) | The deformational test with the period T = 5 of the planar test quoted by Nair & Lauritzen (2010), eqs. (7)-(8): u = (1/T,0) + cos(πt/T) s(x−t/T,y), |
| [`extrusion_bounds_check.jl`](transport/extrusion_bounds_check.jl) | Undershoots of the (linear, unlimited) cubic extrusion scheme: minimum and maximum of the cell means over the run, and at T. |
| [`extrusion_check.jl`](transport/extrusion_check.jl) | the extrusion-based (Mullen / Eldred) Lie derivatives of extrusion_lie.jl |
| [`extrusion_cubic_check.jl`](transport/extrusion_cubic_check.jl) | the density extrusion EXT with a MEAN-PRESERVING upwind reconstruction |
| [`extrusion_definition_check.jl`](transport/extrusion_definition_check.jl) | do our discrete contractions agree with the definition used in the literature? |
| [`extrusion_explain.jl`](transport/extrusion_explain.jl) | figures and a worked example that explain, step by step, the extrusion discretisation of |
| [`extrusion_lie.jl`](transport/extrusion_lie.jl) | the discrete contraction as the dual of a discrete extrusion (Bossavit 2002; Desbrun-Hirani-Leok- |
| [`extrusion_oneform.jl`](transport/extrusion_oneform.jl) | the Lie derivative of a primal 1-form by Cartan's formula with extrusion-based contractions. |
| [`extrusion_oneform_check.jl`](transport/extrusion_oneform_check.jl) | the 1-form Lie derivative L_1 b = I_2(u, D1 b) + D0 I_1(u, b) of extrusion_oneform.jl |
| [`lie_derivative.jl`](transport/lie_derivative.jl) | discrete Lie derivative of densities (twisted 2-forms) and of scalars (0-forms) on the |
| [`lie_testcases_check.jl`](transport/lie_testcases_check.jl) | convergence of the discrete Lie derivative (lie_derivative.jl) on five advection tests |
| [`story_figures_transport.jl`](transport/story_figures_transport.jl) | density transport with H_DEC, H_P1, H_P2 (from ts_data/story_transport.jls)  ->  fig_st_transport.png |
| [`story_transport.jl`](transport/story_transport.jl) | density transport with the cubic extrusion for H_DEC, H_P1 and H_P2 |
| [`thesis_transport.jl`](transport/thesis_transport.jl) | transport solutions and convergence for hodge_thesis.tex |

### `shallow_water` — Linear rotating shallow water

| script | purpose |
|---|---|
| [`coriolis_barrier_check.jl`](shallow_water/coriolis_barrier_check.jl) | is there a Coriolis operator P that is energy-neutral (H_s P antisymmetric) and exact on smooth fields? |
| [`coriolis_growth_modes.jl`](shallow_water/coriolis_growth_modes.jl) | the growing modes of the linear shallow-water scheme with the consistent Coriolis operator P. |
| [`coriolis_growth_power.jl`](shallow_water/coriolis_growth_power.jl) | largest growth rate of the linear shallow-water scheme with the consistent Coriolis operator P on finer meshes. |
| [`coriolis_p_error_check.jl`](shallow_water/coriolis_p_error_check.jl) | errors of W and P on the field u3 of coriolis_barrier_check.jl, in the normalisation of the proposition: max_e /(C)_e - ∫_e Ju·t/ / (/e/ ‖u‖∞), |
| [`m0_barrier_check.jl`](shallow_water/m0_barrier_check.jl) | the analogue of the skew for the 0-form star M0 (point values -> cell integrals), and the spectrum of the |
| [`m0_growth_split.jl`](shallow_water/m0_growth_split.jl) | which operator makes the consistent linear shallow-water scheme grow: P, M0ᶜ, or the star? |
| [`m0_sym_error_check.jl`](shallow_water/m0_sym_error_check.jl) | errors of the mass matrices A, M0c and M0s = (M0c + M0cᵀ)/2 in the normalisation of the proposition: max_v /(M0 R⁰p)_v − ∫_ṽ p/ / (/ṽ/ ‖p‖∞), and the |
| [`sw_ablation.jl`](shallow_water/sw_ablation.jl) | which operator limits the linear shallow-water model?  Four schemes for the linear rotating shallow-water tests |
| [`sw_jet.jl`](shallow_water/sw_jet.jl) | nonlinear rotating shallow water (TRiSK form in DEC variables) and the barotropically unstable jet |
| [`sw_linear_check.jl`](shallow_water/sw_linear_check.jl) | the non-rotating linear shallow-water test (open point "TRiSK" of commuting_symmetric_note.tex): |
| [`sw_linear_rot.jl`](shallow_water/sw_linear_rot.jl) | the linear tests of Peixoto, Alves, Fumagalli & Antonietti (PolyDG paper, §5.2) with H_DEC and H_P2 |
| [`sw_linear_three.jl`](shallow_water/sw_linear_three.jl) | the linear rotating shallow-water tests of Peixoto, Alves, Fumagalli & Antonietti (2026, §5.2) |
| [`sw_solutions_fig.jl`](shallow_water/sw_solutions_fig.jl) | fig_sw_solutions.png: the plane wave with rotation (Peixoto et al. 2026) at 1.5 days on SCVT mesh 5, depth |

### `shared` — Shared plotting helpers

| script | purpose |
|---|---|
| [`thesis_plot_helpers.jl`](shared/thesis_plot_helpers.jl) | shared plotting helpers for the thesis figures (include after the numerical libraries) |

