"""
    Polynomials

Polynomial manipulation utilities for eigenvalue certification.
Provides functions for polynomial evaluation, convolution, scaling,
powers, deflation, and change of basis.
"""
module Polynomials

using BallArithmetic.CertifScripts: polyconv

export polyconv, polyval, polyval_derivative, poly_scale, polypow
export deflation_polynomial, coeffs_about_c_from_about_0, coeffs_about_0_from_about_c

"""
    polyval(coeffs, x)

Evaluate the polynomial with coefficients `coeffs` at point `x`.
Uses Horner's method for numerical stability.
`coeffs = [a₀, a₁, ..., aₙ]` represents p(x) = a₀ + a₁x + ... + aₙxⁿ.
"""
function polyval(coeffs::AbstractVector, x)
    isempty(coeffs) && return zero(promote_type(eltype(coeffs), typeof(x)))
    T = promote_type(eltype(coeffs), typeof(x))
    result = T(coeffs[end])
    @inbounds for i in (length(coeffs) - 1):-1:1
        result = result * x + T(coeffs[i])
    end
    return result
end

"""
    polyval_derivative(coeffs, x)

Evaluate the polynomial `p(x)` and its derivative `p'(x)` simultaneously
using a single Horner pass.

`coeffs = [a₀, a₁, ..., aₙ]` represents p(x) = a₀ + a₁x + ... + aₙxⁿ.

Returns `(p_val, dp_val)`.
"""
function polyval_derivative(coeffs::AbstractVector, x)
    n = length(coeffs)
    T = promote_type(eltype(coeffs), typeof(x))
    if n == 0
        z = zero(T)
        return z, z
    end
    if n == 1
        return T(coeffs[1]), zero(T)
    end
    p = T(coeffs[end])
    dp = zero(T)
    @inbounds for i in (n - 1):-1:1
        dp = dp * x + p
        p = p * x + T(coeffs[i])
    end
    return p, dp
end

"""
    poly_scale(coeffs, α)

Scale a polynomial by a constant: return coefficients of α * p(x).
"""
function poly_scale(coeffs::AbstractVector, α)
    T = promote_type(eltype(coeffs), typeof(α))
    return T(α) .* T.(coeffs)
end

"""
    polypow(coeffs, n)

Compute the n-th power of a polynomial via repeated convolution.
Returns coefficients of p(x)ⁿ.
"""
function polypow(coeffs::AbstractVector, n::Integer)
    n < 0 && throw(ArgumentError("power must be non-negative"))
    T = eltype(coeffs)
    n == 0 && return [one(T)]
    n == 1 && return copy(coeffs)
    result = copy(coeffs)
    for _ in 2:n
        result = polyconv(result, coeffs)
    end
    return result
end

"""
    deflation_polynomial(zeros, λ_tgt; q::Int=1)

Return coefficients `c` of p(x) = (α * P(x))^q where
P(x) = ∏(1 - x/ζᵢ) zeros out the given `zeros`,
and α is chosen so that p(λ_tgt) = 1.

This is used to deflate already-certified eigenvalues when
certifying subsequent eigenvalues.
"""
function deflation_polynomial(zeros::AbstractVector, λ_tgt; q::Int = 1)
    isempty(zeros) && return [one(typeof(λ_tgt))]
    # build P(x) = ∏ (1 - x/ζ)
    T = promote_type(map(typeof, zeros)..., typeof(λ_tgt))
    P = T[one(T)]
    for ζ in zeros
        P = polyconv(P, T[one(T), -one(T) / T(ζ)])  # 1 - x/ζ
    end
    # scale so that α P(λ_tgt) = 1  → α = 1 / P(λ_tgt)
    α = inv(polyval(P, T(λ_tgt)))
    p1 = poly_scale(P, α)
    return polypow(p1, q)
end

"""
    coeffs_about_c_from_about_0(a, c)

Given `a = [a₀, a₁, ..., aₐ]` with p(z) = Σ aₖ zᵏ,
return `b = [b₀, b₁, ..., bₐ]` such that p(z) = Σ bⱼ (z-c)ʲ.

Formula: bⱼ = Σₖ₌ⱼᵈ aₖ * binom(k, j) * c^(k-j).

This is useful for recentering a polynomial around a different
expansion point, e.g., for local analysis near an eigenvalue.
"""
function coeffs_about_c_from_about_0(a::AbstractVector, c)
    d = length(a) - 1
    T = promote_type(eltype(a), typeof(c))
    cc = T(c)
    b = zeros(T, d + 1)
    @inbounds for k in 0:d
        ak = T(a[k + 1])
        ak == 0 && continue
        # z^k = Σ_{j=0}^k binom(k,j) c^(k-j) (z-c)^j
        for j in 0:k
            b[j + 1] += ak * binomial(k, j) * cc^(k - j)
        end
    end
    return b
end

"""
    coeffs_about_0_from_about_c(b, c)

Given `b = [b₀, b₁, ..., bₐ]` with p(z) = Σ bⱼ (z-c)ʲ,
return `a = [a₀, a₁, ..., aₐ]` such that p(z) = Σ aₖ zᵏ.

Formula: aₖ = Σⱼ₌ₖᵈ bⱼ * binom(j, k) * (-c)^(j-k).

This is the inverse of `coeffs_about_c_from_about_0`.
"""
function coeffs_about_0_from_about_c(b::AbstractVector, c)
    d = length(b) - 1
    T = promote_type(eltype(b), typeof(c))
    mc = -T(c)
    a = zeros(T, d + 1)
    @inbounds for j in 0:d
        bj = T(b[j + 1])
        bj == 0 && continue
        # (z-c)^j = Σ_{k=0}^j binom(j,k) z^k (-c)^(j-k)
        for k in 0:j
            a[k + 1] += bj * binomial(j, k) * mc^(j - k)
        end
    end
    return a
end

end # module
