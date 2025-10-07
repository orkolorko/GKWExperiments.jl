module ArbZeta

using ArbNumerics
export hurwitz_zeta, dirichlet_zeta

"""
    hurwitz_zeta(s::ArbComplex{P}, a::ArbComplex{P}; prec::Int = P) -> ArbComplex{P}

Compute the Hurwitz zeta function ζ(s, a) using libarb’s
`acb_hurwitz_zeta` interface.
The precision defaults to that of `s` if not provided.
"""
function hurwitz_zeta(s::ArbComplex{P}, a::ArbComplex{P}; prec::Int = P) where {P}
    res = ArbComplex{P}()
    ccall(ArbNumerics.@libarb(acb_hurwitz_zeta),
          Cvoid,
          (Ref{ArbComplex{P}}, Ref{ArbComplex{P}}, Ref{ArbComplex{P}}, Cint),
          res, s, a, prec)
    return res
end


"""
    dirichlet_zeta(s::ArbComplex{P}; prec::Int = P) -> ArbComplex{P}

Compute the (ordinary) Dirichlet zeta function ζ(s) using libarb’s
`acb_dirichlet_zeta` interface.
The precision defaults to that of `s` if not provided.
"""
function dirichlet_zeta(s::ArbComplex{P}; prec::Int = P) where {P}
    res = ArbComplex{P}()
    ccall(ArbNumerics.@libarb(acb_dirichlet_zeta),
          Cvoid,
          (Ref{ArbComplex{P}}, Ref{ArbComplex{P}}, Cint),
          res, s, prec)
    return res
end

end # module
