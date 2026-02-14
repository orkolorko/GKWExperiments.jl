"""
    Constants

GKW-specific mathematical constants and bounds from the Hardy space framework.
Implements the operator norm bounds C₂ and truncation error Δ from the reference,
as well as H²(D_r) whitening transforms for polynomial perturbation analysis.
"""
module Constants

using ArbNumerics
using LinearAlgebra
using BallArithmetic: Ball, BallMatrix, svd_bound_L2_opnorm

import ..ArbZeta: hurwitz_zeta

export compute_C2, compute_Δ, is_certified
export h2_whiten, power_opnorms, lr_power_bounds_from_Ak
export poly_bridge_constant_powers_from_coeffs, poly_perturbation_bound_powers_from_coeffs
export _arb_to_float64_upper, _arb_to_bigfloat_upper

# ============================================================================
# GKW Operator Norm Bounds (Theorem 3.1 in reference)
# ============================================================================

"""
    compute_C2(N::Integer)

Compute the operator norm bound C₂ for the GKW transfer operator
``L: H²(D₁) → H²(D_{3/2})`` using the splitting formula from Theorem 3.1:

```math
C_2(N) = \\sum_{n=1}^{N-1} \\frac{\\sqrt{2n+1}}{(n-1/2)^2}
       + \\sqrt{2 + \\frac{2}{N-1/2}} \\cdot \\zeta(3/2, N-1/2)
```

Larger `N` gives tighter bounds (see Table 1 in reference).
"""
function compute_C2(N::Integer)
    N ≥ 1 || throw(ArgumentError("N must be ≥ 1"))
    acc = ArbComplex(0)
    half = ArbComplex(1) / 2
    for n in 1:(N - 1)
        nn = ArbComplex(n)
        acc += sqrt(2 * nn + 1) / (nn - half)^2
    end
    a = ArbComplex(N) - half                    # N - 1/2
    tail = hurwitz_zeta(ArbComplex(3) / 2, a)   # ζ(3/2, N-1/2)
    factor = sqrt(2 + 2 / a)
    return acc + factor * tail
end

"""
    compute_Δ(K; N=1000)

Compute the truncation error bound Δ for the rank-K Galerkin approximation
of the GKW operator (Corollary 4.1 in reference):

```math
\\|L_1 - (L_1)_K\\|_{H^2(D_1)} \\leq C_2 \\cdot (2/3)^{K+1} =: \\Delta(K)
```
"""
function compute_Δ(K; N = 1000)
    return compute_C2(N) * (ArbComplex(2) / 3)^(K + 1)
end

"""
    is_certified(cert, K; polynomial=nothing)

Check if an eigenvalue certification result satisfies the small-gain condition
``α = \\|resolvent\\| \\cdot Δ(K) < 1`` for the GKW operator.

Returns `(is_certified::Bool, certified_resolvent_bound)`.
"""
function is_certified(cert, K; polynomial = nothing)
    α = abs(cert.resolvent_original * compute_Δ(K))
    return α < 1, cert.resolvent_original / (1 - α)
end

# ============================================================================
# H²(D_r) Whitening Transforms (Section 6-7 in reference)
# ============================================================================

"""
    h2_whiten(A::AbstractMatrix, r::Real)

Apply the H²(D_r) whitening transform to matrix `A`.

The H²(D_r) operator norm is ``\\|M\\|_{(r)} = \\|C_r M C_r^{-1}\\|_2``
where ``C_r = \\text{diag}(r^n)_{n=0}^{N-1}``.

This function returns the whitened matrix ``C_r A C_r^{-1}``, so that
Euclidean norms of the result equal H²(D_r) norms of the original.
"""
function h2_whiten(A::AbstractMatrix, r::Real)
    N = size(A, 1)
    @assert size(A, 2) == N "Matrix must be square"
    T = promote_type(eltype(A), typeof(r))
    pow = [T(r)^k for k in 0:(N - 1)]
    C = Diagonal(pow)
    Cinv = Diagonal(inv.(pow))
    return C * A * Cinv
end

# ============================================================================
# Power Norm Bounds for Polynomial Perturbation Analysis (Section 9)
# ============================================================================

"""
    power_opnorms(B::BallMatrix, L::Integer)

Compute rigorous upper bounds on ``\\|B^ℓ\\|_2`` for ``ℓ = 0, 1, \\ldots, L``.

Returns a vector `norms` where `norms[ℓ+1]` bounds ``\\|B^ℓ\\|_2``.
The matrix `B` should already be whitened if H²(D_r) norms are needed.
"""
function power_opnorms(B::BallMatrix, L::Integer)
    N = size(B, 1)
    @assert size(B, 2) == N "Matrix must be square"
    T = eltype(B.c)
    RT = real(T)
    norms = zeros(Ball{RT, RT}, L + 1)
    norms[1] = Ball(one(RT))              # ‖B⁰‖ = 1
    L == 0 && return norms
    P = BallMatrix(Matrix{T}(I, N, N))    # B⁰
    for ℓ in 1:L
        P = P * B
        norms[ℓ + 1] = svd_bound_L2_opnorm(P)
    end
    return norms
end

"""
    lr_power_bounds_from_Ak(alpha::AbstractVector, ε::Real)

Compute binomial bounds ``β_m ≤ \\|L_r^m\\|`` for ``m = 0, 1, \\ldots, L``
using the telescoping identity:

```math
\\beta_m = \\sum_{t=0}^m \\binom{m}{t} \\varepsilon^t \\alpha_{m-t}
```

where `alpha[ℓ+1] = ‖A_k^ℓ‖_{(r)}` and `ε = ‖L_r - A_k‖_{(r)}`.
"""
function lr_power_bounds_from_Ak(alpha::AbstractVector, ε::Real)
    T = promote_type(eltype(alpha), typeof(ε))
    L = length(alpha) - 1
    beta = zeros(T, L + 1)
    beta[1] = one(T)                              # m=0
    for m in 1:L
        s = zero(T)
        εm = one(T)
        # accumulate Σ_{t=0}^m binom(m,t) ε^t α_{m-t}
        for t in 0:m
            s += binomial(m, t) * εm * T(alpha[m - t + 1])
            εm *= T(ε)
        end
        beta[m + 1] = s
    end
    return beta
end

"""
    poly_bridge_constant_powers_from_coeffs(c, Ak; r::Real, εr::Real)

Compute the bridge constant ``\\mathcal{C}_r^{pow}`` for polynomial perturbation
bounds from Section 9 of the reference:

```math
\\mathcal{C}_r^{pow} = \\sum_{j \\geq 1} |a_j| \\sum_{\\ell=0}^{j-1} \\alpha_\\ell \\beta_{j-1-\\ell}
```

where `c = [a₀, a₁, ..., aₐ]` are polynomial coefficients,
`Ak` is the discretization matrix, `r` is the Hardy space radius,
and `εr = ‖L_r - A_k‖_{(r)}` is the discretization error.

Returns `(Cr, α, β, s)` where:
- `Cr` is the bridge constant
- `α[ℓ+1] = ‖A_k^ℓ‖_{(r)}`
- `β[m+1]` bounds `‖L_r^m‖_{(r)}`
- `s[k+1] = (α ⋆ β)_k` is the discrete convolution
"""
function poly_bridge_constant_powers_from_coeffs(c::AbstractVector, Ak::AbstractMatrix;
        r::Real, εr::Real)
    B = BallMatrix(h2_whiten(Ak, r))
    d = length(c) - 1                   # degree
    L = max(0, d - 1)
    α = power_opnorms(B, L)             # α[ℓ+1] = ‖A_k^ℓ‖_{(r)}
    β = lr_power_bounds_from_Ak(α, εr)
    # discrete convolution s[k+1] = Σ_{ℓ=0}^k α[ℓ+1] β[k-ℓ+1], for k=0..L
    s = zeros(promote_type(eltype(α), eltype(β)), L + 1)
    for k in 0:L
        t = zero(eltype(s))
        for ℓ in 0:k
            t += α[ℓ + 1] * β[k - ℓ + 1]
        end
        s[k + 1] = t
    end
    # sum over j≥1 (skip constant term c[1])
    Cr = zero(eltype(s))
    for j in 1:d
        aj = abs(c[j + 1])           # coeff of x^j
        aj == 0 && continue
        Cr += aj * s[j]              # uses s_{j-1} (index j)
    end
    return (Cr, α, β, s)
end

"""
    poly_perturbation_bound_powers_from_coeffs(c, Ak; r::Real, εr::Real)

Compute the polynomial perturbation bound ``\\varepsilon_r \\cdot \\mathcal{C}_r^{pow}``.

Returns `(bound, Cr, α, β, s)` where `bound = |εr| * Cr`.
"""
function poly_perturbation_bound_powers_from_coeffs(c, Ak; r::Real, εr::Real)
    Cr, α, β, s = poly_bridge_constant_powers_from_coeffs(c, Ak; r = r, εr = εr)
    return (abs(εr) * Cr, Cr, α, β, s)
end

# ============================================================================
# Rigorous Arb → Float64 / BigFloat upper bounds
# ============================================================================

"""
    _arb_to_float64_upper(x)

Convert an ArbReal ball to a rigorous Float64 upper bound.

Returns `midpoint(x) + radius(x) + conversion_error`, converted to Float64
with upward rounding. The conversion error accounts for the precision lost
when truncating the Arb midpoint to Float64.
"""
function _arb_to_float64_upper(x)
    x_real = real(x)
    mid_arb = ArbNumerics.midpoint(x_real)
    rad_arb = ArbNumerics.radius(x_real)

    mid_f64 = Float64(mid_arb)
    rad_f64 = Float64(rad_arb)

    # Conversion error: |mid_arb - mid_f64|
    mid_big = parse(BigFloat, string(mid_arb))
    conv_err = Float64(abs(mid_big - BigFloat(mid_f64)))

    # Rigorous upper bound
    setrounding(Float64, RoundUp) do
        mid_f64 + rad_f64 + conv_err
    end
end

"""
    _arb_to_bigfloat_upper(x)

Convert an ArbReal ball to a rigorous BigFloat upper bound.

Returns `midpoint(x) + radius(x)` converted to BigFloat with upward rounding.
At matching precision (BigFloat ≥ Arb), the midpoint conversion is exact and
no conversion error needs to be tracked.
"""
function _arb_to_bigfloat_upper(x)
    x_real = real(x)
    mid_bf = BigFloat(ArbNumerics.midpoint(x_real))
    rad_bf = BigFloat(ArbNumerics.radius(x_real))
    setrounding(BigFloat, RoundUp) do
        mid_bf + rad_bf
    end
end

end # module
