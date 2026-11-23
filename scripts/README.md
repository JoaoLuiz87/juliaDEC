# Index of the scripts

Run everything from the repository root, for example `julia -t 4 scripts/stars/stars_cache.jl mesh3`. Output goes to `output/ts_data` and `output/figures`.

## `scripts/stars` — Constructing and checking the Hodge stars

| script | purpose |
|---|---|
| `commuting_check.jl` | the commuting identity  H R_h u = R~_h *u  on the SCVT meshes. |
| `commuting_note_figures.jl` | fig_commuting.png for commuting_symmetric_note.tex |
| `conformal_star_check.jl` | an explicit local symmetric star exact on constants, dilations and rotations |
| `gap_stencil_figure.jl` | fig_gap_stencils.png of paper/hodge_gap.tex (Section 2): the four stencils of one edge of SCVT mesh 3. |
| `gap_stencil_radius.jl` | Assumption (A3) of paper/hodge_gap.tex: every stencil, with the dual edges it involves, lies in |
| `lift_vs_p2_check.jl` | the lift star of hodge_stars.jl (row e = b_eᵀ Π_e, Π_e = pinv(A_e)) against H_P2 |
| `local_symmetric_check.jl` | how far do "symmetric" and "local" go together? |
| `local_symmetric_figures.jl` | figures of hodge_properties_note.tex, Sections "How far local and symmetric go" and |
| `min_lambda.jl` | the SMALLEST possible Λ = max_e (1//ẽ_e/) Σ_j /H_ej/ /e_j/ over ALL symmetric stars exact on [P_k]² on a given pattern (ring r). |
| `nonsymmetric_check.jl` | past the second-order ceiling with ONE unknown per edge by giving up exact symmetry |
| `p2_min_asym_check.jl` | how symmetric can a star exact on [P2]^2 be made?  (and a local P1-exact one) |
| `p2_spectrum_check.jl` | do the complex eigenvalues of the H_P2 Laplacian go away under refinement? |
| `p2_stencil_rank.jl` | how many [P_k]^2 fields can a stencil tell apart?  (Section 2.2 of three_stars_report.tex) |
| `p2_swept_spectrum.jl` | Laplacian spectrum of the swept exact stars of p2_min_asym_check.jl on the finer meshes |
| `p2_symmetric_check.jl` | can H_P2 be built like H_P1, i.e. as the smallest SYMMETRIC correction of DEC exact on |
| `p2_symmetric_figures.jl` | the figures of Sections 2.2 and 2.5 of three_stars_report.tex. |
| `p2_symmetric_refs.jl` | reference curves for fig_ts_p2sym.png: H_DEC and H_P1 (exact on [P1]^2, 3-level stencil) |
| `p2_symmetric_sdp.jl` | is there a POSITIVE symmetric star exact on [P2]^2 on the 4-ring, on SCVT meshes 2-4? |
| `positivity_check.jl` | a cheap certificate that the symmetric [P1]^2 star is SPD (Section 4 of |
| `report_entries.jl` | how the entries of a symmetric star exact on [P2]² grow: least-change star on the 4-ring, meshes 2-5 (largest off-diagonal entry relative to the |
| `report_figures_row.jl` | fig_rep_row.png from ts_data/report_entries.jls (see report_entries.jl): one row of three stars on mesh 5 |
| `star_properties_check.jl` | which properties of the Hodge star survive from H_DEC to H_P1 and H_P2, and what the |
| `stars_cache.jl` | build the three stars of three_stars.jl once per mesh and keep them on disk |
| `three_stars.jl` | the three Hodge stars of three_stars_report.tex and the problems they are tested on. |
| `three_stars_example.jl` | the numbers of the worked example (mesh 3, edge 17) in Section 2 of three_stars_report.tex. |
| `three_stars_figures.jl` | the figures of three_stars_report.tex, from ts_data/results.jls. |
| `three_stars_run.jl` | all numbers of three_stars_report.tex. |
| `three_stars_tests.jl` | checks that the code behind three_stars_report.tex has no gaps. |

## `scripts/barrier` — The skew, the barrier, locality

| script | purpose |
|---|---|
| `barrier_figures.jl` | figure of barrier_paper.tex from ts_data/skew_split.jls (after skew_split_check.jl). |
| `circumradius_figures.jl` | the offset as a dual gradient of half the squared |
| `compatibility_check.jl` | which degrees k admit a symmetric star exact on [P_k]^2 (Section 3.5 of |
| `gap_barrier_figure.jl` | fig_gap_barrier.png for Section 3.2 of paper/hodge_gap.tex (Theorem barrier, Corollary size). |
| `gap_skew_figure.jl` | Figure fig_gap_skew.png of paper/hodge_gap.tex (Section 3.6). |
| `gap_skew_more.jl` | Hypothesis S2 on more centroidal Voronoi meshes: is /s_h(u,v)/ ≥ c h^2 with c bounded away from 0? |
| `green_charge_check.jl` | the simple proof that a symmetric star exact on linear fields cannot be local. |
| `green_charge_mesh.jl` | the balance behind the non-locality theorem on SCVT mesh 4 (simple form, for the Laplacian L = D0ᵀ H D0): |
| `homogeneous_figures.jl` | figure of homogeneous_note.tex (after homogeneous_skew_check.jl and homogeneous_p2_check.jl) |
| `homogeneous_p2_check.jl` | on a replicated (homogeneous) family the skew is O(h^3), so the barrier no longer forbids a |
| `homogeneous_skew_check.jl` | is the h^2 skew of SCVT meshes a property of unstructured meshes, or of their irregularity? |
| `hypN_proof.jl` | a computer-assisted proof of Hypothesis N of barrier_hodge.tex (locality theorem) |
| `k_norm_check.jl` | does the skew part K of the lift decay under refinement? |
| `lift_offset_figures.jl` | figures and numbers for lift_offset_symmetry.tex |
| `lloyd_convergence_check.jl` | how many Lloyd steps does one refinement of the SCVT family need? |
| `locality_check.jl` | can a symmetric star exact on [P1]^2 be built locally?  (barrier_paper.tex, Section "Locality") |
| `locality_subspace_check.jl` | which linear fields force a symmetric exact star to be global?  (barrier_paper.tex) |
| `mesh_quality_check.jl` | quality of the SCVT family meshes/meshN.nc (and of any other mesh files given) |
| `offset_scaling_check.jl` | does δ_e//ẽ/ go to zero under refinement? |
| `sh_rate_check.jl` | at what rate does s_h go to zero? It depends on the mesh. |
| `sh_rate_mesh.jl` | periodic test meshes (lattice / mapped / random / pattern) and segment quadrature, |
| `skew_fourier_check.jl` | an exact formula for the skew of plane waves, and what Hypothesis S_2 means for the mesh |
| `skew_split_check.jl` | where does the skew of the primal-dual pairing live?  (barrier_paper.tex) |
| `sphere_skew_check.jl` | the skew of the primal--dual pairing on spherical Delaunay--Voronoi meshes |
| `story_global.jl` | why symmetry makes a consistent star global: figure and numbers for hodge_story.tex |
| `story_offset.jl` | the offset of the SCVT meshes (hodge_story.tex, Chapter 2)  ->  fig_st_offset.png |
| `story_patch.jl` | the patch of the balance law on mesh 5 (hodge_story.tex, "Every symmetric star exact on linear fields is global") |
| `story_realrow.jl` | why symmetry makes H_P1 global, on the REAL stencil (hodge_story.tex, Section "in detail") |
| `window_skew_check.jl` | is the offset skew over a window a boundary term? |

## `scripts/elliptic` — Poisson, eigenvalues, Darcy, paper figures

| script | purpose |
|---|---|
| `paper_figures.jl` | new figures of paper/hodge_gap.tex |
| `paper_runs.jl` | the numbers of paper/hodge_barrier_paper.tex that were not computed before |
| `report_figures_elliptic.jl` | figures of hodge_report.tex, numerical part: |
| `report_figures_theory.jl` | figures of hodge_report.tex, theory part: |
| `report_runs.jl` | runs for hodge_report.tex: the velocity (flux) of the Poisson problem, F = H D0 p_h against R̃⋆∇p, next to the nodal errors, meshes 1-7, |
| `story_eigen.jl` | eigenvalues of the Laplacian  D0ᵀ H D0 p = λ M0 p  on the unit torus, for H_DEC, H_P1, H_P2 |
| `story_figures_eigen.jl` | Laplace eigenvalues for H_DEC, H_P1, H_P2 (from ts_data/story_eigen.jls)  ->  fig_st_eigen.png |
| `story_figures_elliptic.jl` | Poisson and Darcy errors of H_DEC, H_P1, H_P2 on ONE colour scale per row (from ts_data/thesis_tests.jls) |
| `thesis_both_stars.jl` | solution figures with H_DEC and H_P2 side by side, for hodge_thesis.tex |
| `thesis_figures.jl` | solution plots and transport convergence for hodge_thesis.tex |
| `thesis_tests.jl` | more Poisson and Darcy solutions for hodge_thesis.tex |

## `scripts/transport` — Lie derivative and advection

| script | purpose |
|---|---|
| `commutation_run_check.jl` | D L = L D through the fully discrete runs (not only on matrices), with the right-hand sides of the paper: |
| `extrusion_T5_check.jl` | The deformational test with the period T = 5 of the planar test quoted by Nair & Lauritzen (2010), eqs. (7)-(8): u = (1/T,0) + cos(πt/T) s(x−t/T,y), |
| `extrusion_bounds_check.jl` | Undershoots of the (linear, unlimited) cubic extrusion scheme: minimum and maximum of the cell means over the run, and at T. |
| `extrusion_check.jl` | the extrusion-based (Mullen / Eldred) Lie derivatives of extrusion_lie.jl |
| `extrusion_cubic_check.jl` | the density extrusion EXT with a MEAN-PRESERVING upwind reconstruction |
| `extrusion_definition_check.jl` | do our discrete contractions agree with the definition used in the literature? |
| `extrusion_explain.jl` | figures and a worked example that explain, step by step, the extrusion discretisation of |
| `extrusion_lie.jl` | the discrete contraction as the dual of a discrete extrusion (Bossavit 2002; Desbrun-Hirani-Leok- |
| `extrusion_oneform.jl` | the Lie derivative of a primal 1-form by Cartan's formula with extrusion-based contractions. |
| `extrusion_oneform_check.jl` | the 1-form Lie derivative L_1 b = I_2(u, D1 b) + D0 I_1(u, b) of extrusion_oneform.jl |
| `lie_derivative.jl` | discrete Lie derivative of densities (twisted 2-forms) and of scalars (0-forms) on the |
| `lie_spectra_check.jl` | spectra of the semi-discrete Lie derivative for steady flows (stability) |
| `lie_symmetry_check.jl` | does the asymmetry of H_P2 hurt the discrete Lie derivative? |
| `lie_tables.jl` | LaTeX tables of lie_derivative_note.tex from the result files |
| `lie_testcases_check.jl` | convergence of the discrete Lie derivative (lie_derivative.jl) on five advection tests |
| `lie_testcases_figures.jl` | figures of lie_testcases_check.jl |
| `lie_transport_animation.jl` | a density transported over an SCVT mesh by the discrete Lie derivative |
| `lie_transport_check.jl` | accuracy and stability of the discrete Lie derivatives of lie_derivative.jl |
| `lie_transport_figures.jl` | figures of lie_transport_check.jl |
| `lie_upw_trunc.jl` | truncation error of the upwind interior product UPW2 (same setting as part A of |
| `story_figures_transport.jl` | density transport with H_DEC, H_P1, H_P2 (from ts_data/story_transport.jls)  ->  fig_st_transport.png |
| `story_transport.jl` | density transport with the cubic extrusion for H_DEC, H_P1 and H_P2 |
| `story_videos.jl` | videos of the transport tests (MP4, CairoMakie + FFMPEG) |
| `thesis_transport.jl` | transport solutions and convergence for hodge_thesis.tex |
| `thesis_uniform_density.jl` | a uniform density in the deformational flow: the only error is that of the swept area (Hv), so it |

## `scripts/shallow_water` — Linear and nonlinear shallow water

| script | purpose |
|---|---|
| `coriolis_barrier_check.jl` | is there a Coriolis operator P that is energy-neutral (H_s P antisymmetric) and exact on smooth fields? |
| `coriolis_growth_modes.jl` | the growing modes of the linear shallow-water scheme with the consistent Coriolis operator P. |
| `coriolis_growth_power.jl` | largest growth rate of the linear shallow-water scheme with the consistent Coriolis operator P on finer meshes. |
| `coriolis_p_error_check.jl` | errors of W and P on the field u3 of coriolis_barrier_check.jl, in the normalisation of the proposition: max_e /(C)_e - ∫_e Ju·t/ / (/e/ ‖u‖∞), |
| `m0_barrier_check.jl` | the analogue of the skew for the 0-form star M0 (point values -> cell integrals), and the spectrum of the |
| `m0_growth_split.jl` | which operator makes the consistent linear shallow-water scheme grow: P, M0ᶜ, or the star? |
| `m0_sym_error_check.jl` | errors of the mass matrices A, M0c and M0s = (M0c + M0cᵀ)/2 in the normalisation of the proposition: max_v /(M0 R⁰p)_v − ∫_ṽ p/ / (/ṽ/ ‖p‖∞), and the |
| `sw_ablation.jl` | which operator limits the linear shallow-water model?  Four schemes for the linear rotating shallow-water tests |
| `sw_jet.jl` | nonlinear rotating shallow water (TRiSK form in DEC variables) and the barotropically unstable jet |
| `sw_jet_apvm.jl` | the unstable jet with APVM 0.5 (the same for both stars), meshes 5-7, H_DEC and H_P2 |
| `sw_jet_figures.jl` | figures of the unstable jet (H_DEC against H_P2) and of the steady zonal balance |
| `sw_jet_more.jl` | the steady zonal balance on meshes 3-7 and the jet on mesh 7, both stars  (julia -t 4 scripts/shallow_water/sw_jet_more.jl) |
| `sw_linear.jl` | the linear rotating shallow-water tests of Peixoto, Alves, Fumagalli & Antonietti (2026, §5.2) with the |
| `sw_linear_check.jl` | the non-rotating linear shallow-water test (open point "TRiSK" of commuting_symmetric_note.tex): |
| `sw_linear_rot.jl` | the linear tests of Peixoto, Alves, Fumagalli & Antonietti (PolyDG paper, §5.2) with H_DEC and H_P2 |
| `sw_linear_rot_figures.jl` | figures for the linear rotating tests (Peixoto et al. §5.2), H_DEC against H_P2 |
| `sw_linear_three.jl` | the linear rotating shallow-water tests of Peixoto, Alves, Fumagalli & Antonietti (2026, §5.2) |
| `sw_m0_sweep_check.jl` | the sweeps of local_symmetric_check.jl applied to the corrected 0-form star M0 = A + C0 |
| `sw_peixoto_all.jl` | the nonlinear tests of Peixoto, Alves, Fumagalli & Antonietti (2026, §5.1-5.4) with the three stars |
| `sw_solutions_fig.jl` | fig_sw_solutions.png: the plane wave with rotation (Peixoto et al. 2026) at 1.5 days on SCVT mesh 5, depth |
| `sw_steady_residuals.jl` | truncation errors of the three nonlinear terms of the TRiSK/DEC shallow-water model on the exact zonal |
| `sw_swept_check.jl` | the swept stars of local_symmetric_check.jl in the linear shallow-water equations (f = 0) |
| `sw_vortex_apvm.jl` | the stationary vortex on mesh 7 with APVM 0.5 (as in the jet runs): does it control the growth seen with H_P2? |
| `sw_vortex_diag.jl` | why H_P2 degrades on the vortex on mesh 7: error against time, and where the largest error sits |
| `thesis_sw.jl` | linear shallow-water solution plot for hodge_thesis.tex |

## `scripts/shared` — Shared plotting helpers

| script | purpose |
|---|---|
| `thesis_plot_helpers.jl` | shared plotting helpers for the thesis figures (include after the numerical libraries) |

## `scripts/archive` — Older experiments (not maintained)

These scripts are kept for the record. **25 of them do not run**: they need `galerkin_star.jl` or `SymStar.jl`, which are no longer part of the project; the others depend on data that is not in the repository.

| script | purpose |
|---|---|
| `ddr_mixed_poisson.jl` | DDR (k = 0, 1) for the mixed Poisson problem in 2D |
| `distortion_refinement_test.jl` | numerical test of the circumradius-jump |
| `es_recipe_check.jl` | every number of report/energy_star_recipe.tex: the hand example, the checks of each step on the |
| `gh_export_scvt.jl` | export the juliaDEC SCVT meshes (Delaunay vertices and triangles) to plain text for gh_lab.jl |
| `gh_figures.jl` | figures of galerkin_hodge.tex |
| `gh_kernel_check.jl` | is the moment system solvable?  (Theorem "solvability") |
| `gh_lab.jl` | Galerkin Hodge lab: the Hodge star extracted from a bilinear form |
| `gh_main.jl` | symmetric Hodge stars from a bilinear form: least-squares versus moment-constrained |
| `gh_reg_sweep.jl` | exact versus approximate moment conditions: does the correction Z stay bounded? |
| `gh_ring2.jl` | refinement with 2-ring triangle stencils: strong [P1]² versus [P1]² rows + [P2]² patch test |
| `gh_ring2_fine.jl` | finer levels for the relaxed star ([P1]² rows + [P2]² patch test, 2-ring stencils, ε = 1e-8) |
| `gh_weak.jl` | the relaxed target: strong [P1]² flux consistency + the [P2]² patch test (vertex closure) |
| `gh_weak_ring.jl` | the relaxed target ([P1]² rows + [P2]² patch test): stencil width and regularisation |
| `gs_1d_toy.jl` | the reconstruction-energy Hodge star in ONE dimension, where every matrix fits on a page |
| `gs_ceiling.jl` | how accurate can ANY symmetric star be on the juliaDEC SCVT meshes? |
| `gs_check.jl` | verification and convergence study of the Galerkin star (galerkin_star.jl) on the juliaDEC |
| `gs_figures.jl` | figures and worked-example numbers of galerkin_star_paper.tex (uses galerkin_star.jl) |
| `gs_hypotheses.jl` | direct checks of the hypotheses and identities used in galerkin_star_paper.tex |
| `gs_mass_test.jl` | why B_T = R_Tᵀ M_T R_T and not R_Tᵀ R_T: the Galerkin star with the mass matrix M_T replaced |
| `gs_p2_conv.jl` | convergence of DEC, the degree-1 star and the degree-2 star on the SCVT meshes 2-5 |
| `gs_p2_diagnose.jl` | why the degree-2 star's coefficients grow under refinement |
| `gs_p2_explore.jl` | the degree-2 Galerkin star: [P2]^2 reconstructions, linear moments exact, quadratic moments |
| `gs_p2_figures.jl` | fig_p2_conv.png from gs_data/p2_conv_table.txt: DEC, the degree-1 and the degree-2 stars |
| `gs_p2_worked.jl` | worked example for the degree-1 and degree-2 stars: the 12 moments (constant, linear, |
| `gs_square_grid.jl` | on the square grid (no offsets) a symmetric star that is NOT a Galerkin form is exact on |
| `gs_tutorial2d_figures.jl` | figures and numbers of hodge_method_2d.tex (the 2D method, step by step) |
| `hh_consistency.jl` | isotropic SCVT meshes: symmetric stars exact on linear fields, and their O(h²) consistency |
| `hh_figure.jl` | fig_hh_conv.png from hh_data/consistency.txt: DEC versus the symmetric, linear-exact Galerkin stars |
| `lift_sandwich_figures.jl` | figures for the subsections |
| `obstruction_figures.jl` | figures for hodge_symmetry_consistency.tex |
| `peixoto_figures.jl` | figures for the section "Relation to Peixoto (2016)" |
| `pitassi_dec_checks.jl` | checks for Part I of pitassi_dec_ddr.tex |
| `pitassi_dec_ddr_figures.jl` | new figures of pitassi_dec_ddr.tex |
| `poisson_sandwich_check.jl` | Strang identity for a Galerkin (Hiptmair) Hodge |
| `route2_check.jl` | the Pitassi form H = Rᵀ M R + γ² W Wᵀ at degree k (Proposition "route2") |
| `sandwich_dual_vectors.jl` | the dual vectors induced by the edge-based sandwich |
| `sandwich_hodge_check.jl` | the edge-based sandwich read as a (Galerkin/Hiptmair) Hodge |
| `ss_abstol.jl` | a variant: impose only the skew directions that are not already negligible |
| `ss_aniso.jl` | constant anisotropic permeability: DEC_K, the exact K-star (guide), the uniform-share variant |
| `ss_appendixB_test.jl` | run the guide's Appendix B transcription (verbatim copy in ss_data/) on the porting mesh, |
| `ss_figures.jl` | figures and worked-example numbers of symmetric_star_review.tex |
| `ss_figures_conv.jl` | fig_ss_conv.png from ss_data/{abstol,objective,aniso}.txt (included by ss_figures.jl; runs alone too) |
| `ss_growth.jl` | growth of the shares under refinement, and Tikhonov regularisation of the share solve |
| `ss_kernel.jl` | the share matrix G: kernel, compatibility, overlap condition, near-kernel under refinement |
| `ss_mesh.jl` | the perturbed lattice of the guide's Section 6.5 (porting check), shared by the ss_*.jl scripts |
| `ss_objective.jl` | which admissible shares?  minimum norm (the guide) versus closest to uniform |
| `ss_objective_lib.jl` | Poisson errors in the norms of the guide's Theorem 4.2 (shared by ss_objective.jl, ss_abstol.jl) |
| `ss_porting_check.jl` | validate SymStar.jl against Table 5 of symmetric_star_guide.pdf, and test |
| `ss_voronoimesh_example.jl` | using the symmetric star inside juliaDEC (worked example of the note) |
| `ss_wellcentred.jl` | are the juliaDEC SCVT meshes well-centred (Assumption 2.1 of the guide)? |
| `thesis_extract.jl` | copy statements + proofs from hodge_properties_note.tex into thesis_app/<label>.tex (used by hodge_thesis.tex) |

