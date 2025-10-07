module GKWDiscretization

using ArbNumerics
# We assume ArbZeta (with hurwitz_zeta) is in the parent package:
# include("ArbZeta.jl") is done elsewhere; here we import its symbol.
import ..ArbZeta: hurwitz_zeta

export zeta_shift_table_on_circle, values_Ls_fk_from_table!,
       coeffs_from_boundary, build_Ls_matrix_arb, gkw_matrix_direct

# ---------- Hurwitz zeta table on |w-1| = 1 ----------
"""
    zeta_shift_table_on_circle(s::ArbComplex{P}; K::Int, N::Int=1024, prec::Int=P)
        -> Z::Matrix{ArbComplex{P}}, w::Vector{ArbComplex{P}}

Precompute `Z[j+1,m] = ζ(2s + j, w_m + 1)` for j = 0..K and grid points
w_m = 1 + e^{i θ_m} with θ_m = 2π m / N on the unit circle centered at 1.

This supports the identity
(L_s (w-1)^k)(w) = ∑_{j=0}^k (-1)^{k-j} binom(k,j) ζ(2s+j, w+1).
"""
function zeta_shift_table_on_circle(s::ArbComplex{P}; K::Int, N::Int=1024, prec::Int=P) where {P}
    @assert K ≥ 0
    @assert N ≥ 2
    Z  = Matrix{ArbComplex{P}}(undef, K+1, N)
    ws = Vector{ArbComplex{P}}(undef, N)

    for m in 0:N-1
        θ = 2*ArbReal(π) * m / N
        w = ArbComplex{P}(1 + cos(θ), sin(θ))  # w_m = 1 + e^{iθ}
        ws[m+1] = w
        a = w + ArbComplex{P}(1)               # Hurwitz parameter a = w + 1
        @inbounds for j in 0:K
            Z[j+1, m+1] = hurwitz_zeta(2*s + ArbComplex{P}(j), a; prec=prec)
        end
    end
    return Z, ws
end

# ---------- Evaluate L_s on the grid for f_k(w) = (w-1)^k using the table ----------
"""
    values_Ls_fk_from_table!(vals, Z, k)

In-place fill of `vals[m] = (L_s (w-1)^k)(w_m)` using the precomputed table Z
with the binomial identity:
(L_s (w-1)^k)(w) = ∑_{j=0}^k (-1)^{k-j} binom(k,j) ζ(2s+j, w+1).

- `vals` must have length N matching size(Z,2).
- `Z` is the table from `zeta_shift_table_on_circle`.
- `k ≥ 0` is the monomial degree.
"""
function values_Ls_fk_from_table!(vals::Vector{ArbComplex{P}},
                                  Z::Matrix{ArbComplex{P}},
                                  k::Int) where {P}
    N = size(Z, 2)
    @assert length(vals) == N
    fill!(vals, ArbComplex{P}(0))
    @inbounds for j in 0:k
        cj = ((-1)^(k-j)) * binomial(big(k), j)  # integer coefficient
        for m in 1:N
            vals[m] += cj * Z[j+1, m]
        end
    end
    return vals
end

# ---------- Taylor coefficients from boundary samples via Arb DFT ----------
"""
    coeffs_from_boundary(vals) -> c

Return the (unnormalized) DFT of boundary samples `vals` divided by N.
On the unit circle |w-1|=1, this yields the Taylor coefficients at w=1:
f(1+e^{iθ}) = ∑_{k≥0} a_k e^{ikθ}  ⇒  a_k ≈ (DFT(vals)[k+1]) / N.
"""
function coeffs_from_boundary(vals::Vector{ArbComplex{P}}) where {P}
    ArbNumerics.dft(vals) ./ length(vals)
end

# ---------- Build (K+1)×(K+1) matrix of L_s in the basis (w-1)^k ----------
"""
    build_Ls_matrix_arb(s::ArbComplex{P}; K::Int=64, N::Int=1024, prec::Int=P)
        -> M::Matrix{ArbComplex{P}}

Return the matrix `M` of size (K+1)×(K+1) where the k-th column is the vector
of Taylor coefficients `b_0,…,b_K` of `(L_s (w-1)^k)(w) = ∑ b_j (w-1)^j`,
computed by:
1) precomputing a table Z[j+1,m] = ζ(2s + j, w_m + 1) on the unit circle,
2) evaluating `(L_s (w-1)^k)(w_m)` via the binomial/Hurwitz identity,
3) extracting coefficients by DFT.

Use N > K for reliable coefficient extraction.
"""
function build_Ls_matrix_arb(s::ArbComplex{P}; K::Int=64, N::Int=1024, prec::Int=P) where {P}
    @assert N > K "Use N > K so you can read at least K+1 Taylor coefficients."
    Z, _ = zeta_shift_table_on_circle(s; K=K, N=N, prec=prec)
    M = Matrix{ArbComplex{P}}(undef, K+1, K+1)
    vals = Vector{ArbComplex{P}}(undef, N)

    for k in 0:K
        values_Ls_fk_from_table!(vals, Z, k)
        c = coeffs_from_boundary(vals)             # a_j for j≥0
        @inbounds M[:, k+1] = c[1:K+1]             # take j = 0..K
    end
    return M
end

# Safe rising factorial (Pochhammer): (z)_ℓ, with (z)_0 = 1
@inline function pochhammer_arb(z::ArbComplex{P}, ℓ::Int) where {P}
    acc = ArbComplex{P}(1)
    @inbounds for m in 0:ℓ-1
        acc *= (z + ArbComplex{P}(m))
    end
    return acc
end

# Build M_{ℓ,k} = (-1)^{k+ℓ} ∑_{j=0}^k (-1)^j C(k,j) * (2s+j)_ℓ / ℓ! * ζ(2s+j+ℓ, 2)
"""
    gkw_matrix_direct(s::ArbComplex{P}; K::Int=20, prec::Int=P) -> Matrix{ArbComplex{P}}

Assemble the `(K+1)×(K+1)` Galerkin matrix of the GKW transfer operator using
its direct series expansion involving Hurwitz zeta values at the point `a = 2`.

The entry with zero-based indices `(ℓ, k)` is computed as

```
(-1)^{k+ℓ} ∑_{j=0}^k (-1)^j binomial(k, j) (2s + j)_ℓ / ℓ! * ζ(2s + j + ℓ, 2),
```

where `(2s + j)_ℓ` denotes the rising factorial.  The keyword argument `prec`
controls the Arb precision used for the intermediate Hurwitz zeta evaluations.
"""
function gkw_matrix_direct(s::ArbComplex{P}; K::Int=20, prec::Int=P) where {P}
    @assert K ≥ 0
    M = Matrix{ArbComplex{P}}(undef, K+1, K+1)

    # Cache ζ(2s + t, 2) for all t = 0..2K
    ζtab = Vector{ArbComplex{P}}(undef, 2K + 1)
    two = ArbComplex{P}(2)
    for t in 0:2K
        ζtab[t+1] = hurwitz_zeta(2*s + ArbComplex{P}(t), two; prec=prec)
    end

    # Precompute factorials in Arb to avoid repeated conversions
    fact = [ArbReal(factorial(big(ℓ))) for ℓ in 0:K]

    for k in 0:K
        # binomials for this k, in Arb to avoid Int overflow
        binoms_k = [ArbReal(binomial(big(k), j)) for j in 0:k]
        for ℓ in 0:K
            acc = ArbComplex{P}(0)
            # common sign factor (-1)^(k+ℓ)
            sgn_kl = isodd(k + ℓ) ? ArbReal(-1) : ArbReal(1)
            invℓ!  = inv(fact[ℓ+1])  # 1/ℓ!
            @inbounds for j in 0:k
                sgn_j = isodd(j) ? ArbReal(-1) : ArbReal(1)
                cbin  = binoms_k[j+1]
                poch  = pochhammer_arb(2*s + ArbComplex{P}(j), ℓ)  # (2s+j)_ℓ
                ζval  = ζtab[j + ℓ + 1]                            # ζ(2s+j+ℓ, 2)
                term  = ArbComplex{P}(sgn_kl * sgn_j) * cbin * (poch * invℓ!) * ζval
                acc  += term
            end
            M[ℓ+1, k+1] = acc
        end
    end
    return M
end



end # module
