"""
    GKWExperiments

High-level interface for experimenting with the GKW transfer operator in
ArbNumerics.  The package re-exports helper routines for computing Hurwitz and
Dirichlet zeta values as well as discretisations of the transfer operator that
can be used in numerical experiments.
"""
module GKWExperiments

using ArbNumerics

"""
    mid(x)

Return the midpoint of an ArbNumerics ball `x`.

This is a thin wrapper around [`ArbNumerics.midpoint`](https://jeffreysarnoff.github.io/ArbNumerics.jl/stable/intervalfunctions/) that is convenient
to re-export alongside the rest of the package API.
"""
mid(x) = ArbNumerics.midpoint(x)
export ArbComplex, mid

include("ArbZeta.jl")
using .ArbZeta: dirichlet_zeta, hurwitz_zeta  # bring symbols into scope
export dirichlet_zeta, hurwitz_zeta

include("GKWDiscretization.jl")
using .GKWDiscretization
export zeta_shift_table_on_circle, values_Ls_fk_from_table!, coeffs_from_boundary, build_Ls_matrix_arb, gkw_matrix_direct

end
