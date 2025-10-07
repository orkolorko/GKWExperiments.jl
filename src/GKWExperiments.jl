module GKWExperiments

using ArbNumerics

mid(x) = ArbNumerics.midpoint(x)
export ArbComplex, mid

include("ArbZeta.jl")
using .ArbZeta: dirichlet_zeta, hurwitz_zeta  # bring symbols into scope
export dirichlet_zeta, hurwitz_zeta

include("GKWDiscretization.jl")
using .GKWDiscretization
export zeta_shift_table_on_circle, values_Ls_fk_from_table!, coeffs_from_boundary, build_Ls_matrix_arb, gkw_matrix_direct

end
