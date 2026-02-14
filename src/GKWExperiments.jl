"""
    GKWExperiments

High-level interface for experimenting with the GKW transfer operator in
ArbNumerics.  The package re-exports helper routines for computing Hurwitz and
Dirichlet zeta values as well as discretisations of the transfer operator that
can be used in numerical experiments.

The certification infrastructure is provided by BallArithmetic.jl, which this
package re-exports for convenience.
"""
module GKWExperiments

using ArbNumerics
using BallArithmetic

# Re-export BallArithmetic certification infrastructure
using BallArithmetic.CertifScripts
export CertifScripts
export CertificationCircle, points_on, run_certification
export compute_schur_and_error, bound_res_original
export configure_certification!, set_schur_matrix!, dowork, adaptive_arcs!
export choose_snapshot_to_load, save_snapshot!, poly_from_roots, polyconv

# Re-export core BallArithmetic types and functions
export Ball, BallMatrix, BallVector
export svd_bound_L2_opnorm, svdbox

"""
    mid(x)

Return the midpoint of an ArbNumerics ball `x`.

This is a thin wrapper around [`ArbNumerics.midpoint`](https://jeffreysarnoff.github.io/ArbNumerics.jl/stable/intervalfunctions/) that is convenient
to re-export alongside the rest of the package API.
"""
mid(x) = ArbNumerics.midpoint(x)
export ArbComplex, mid

# Zeta functions (GKW-specific wrappers around libarb)
include("ArbZeta.jl")
using .ArbZeta: dirichlet_zeta, hurwitz_zeta
export dirichlet_zeta, hurwitz_zeta

# GKW operator norm bounds and H² whitening
include("Constants.jl")
using .Constants
export compute_C2, compute_Δ, is_certified
export h2_whiten, power_opnorms, lr_power_bounds_from_Ak
export poly_bridge_constant_powers_from_coeffs, poly_perturbation_bound_powers_from_coeffs
export _arb_to_float64_upper, _arb_to_bigfloat_upper

# Transfer operator matrix construction
include("GKWDiscretization.jl")
using .GKWDiscretization
export zeta_shift_table_on_circle, values_Ls_fk_from_table!
export coeffs_from_boundary, build_Ls_matrix_arb, gkw_matrix_direct, gkw_matrix_direct_fast

# Polynomial utilities for eigenvalue certification
include("Polynomials.jl")
using .Polynomials
export polyconv, polyval, polyval_derivative, poly_scale, polypow
export deflation_polynomial, coeffs_about_c_from_about_0, coeffs_about_0_from_about_c

# Eigenspace certification for GKW operator (uses BallArithmetic VBD)
include("EigenspaceCertification.jl")
using .EigenspaceCertification
export GKWEigenCertificationResult, certify_gkw_eigenspaces
export arb_to_ball_matrix
export float64_ball_to_bigfloat_ball, bigfloat_ball_to_float64_ball

# Finite-to-infinite dimensional lift (resolvent bridge and spectral stability)
include("InfiniteDimensionalLift.jl")
using .InfiniteDimensionalLift
export InfiniteDimCertificationResult
export resolvent_bridge_condition, certified_resolvent_bound
export eigenvalue_inclusion_radius, projector_approximation_error
export newton_kantorovich_error
export certify_eigenvalue_lift, verify_spectral_gap
export DeflationCertificationResult, certify_eigenvalue_deflation, backmap_inclusion_radius
export certify_eigenvalue_deflation_bigfloat
export OrdschurDirectResult, certify_eigenvalue_ordschur_direct, certify_eigenvalue_schur_direct
export deflation_truncation_error
export TwoStageCertificationResult, reverse_transfer_resolvent_bound
export projector_approximation_error_rigorous

# Newton-Kantorovich eigenpair certification (defect-based, single-space)
include("NewtonKantorovichCertification.jl")
using .NewtonKantorovichCertification
export NKCertificationResult, certify_eigenpair_nk
export assemble_eigenpair_jacobian, compute_nk_radius

# Re-export BallArithmetic parametric/ogita certification
using BallArithmetic.CertifScripts: run_certification_parametric, run_certification_ogita
using BallArithmetic: config_v1, config_v2, config_v2p5, config_v3
export run_certification_parametric, run_certification_ogita
export config_v1, config_v2, config_v2p5, config_v3

# Re-export BallArithmetic VBD types used in results
export RigorousBlockSchurResult, MiyajimaVBDResult
export rigorous_block_schur, miyajima_vbd
export collatz_upper_bound_L2_opnorm

end
